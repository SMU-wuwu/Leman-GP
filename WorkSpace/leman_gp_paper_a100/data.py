"""Train-only lexicon, four candidates per character, cached window features."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import os
import pickle
import re

import jieba
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tqdm import tqdm


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with open(temp, 'w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def entity_types(path):
    result = set()
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        if line.strip():
            label = line.split()[0].replace('_', '-')
            if label.startswith(('B-', 'I-')):
                result.add(label[2:])
    if not result:
        raise ValueError(f'No BIO entity types in {path}')
    return sorted(result)


def read_samples(path, types):
    """Return (text, [(type_id, start, exclusive_end), ...]); no truncation."""
    mapping = {name: i for i, name in enumerate(types)}
    samples, chars, labels = [], [], []

    def finish():
        if not chars:
            return
        spans, active, start = [], None, 0
        for i, label in enumerate(labels + ['O']):
            tag = label[0] if label != 'O' else 'O'
            kind = label[2:] if tag in ('B', 'I') else None
            if active is not None and (tag != 'I' or kind != active):
                spans.append((mapping[active], start, i))
                active = None
            if kind is not None:
                if kind not in mapping:
                    raise ValueError(f'Unknown entity type {kind!r} in {path}')
                if active is None:
                    active, start = kind, i
        samples.append((''.join(chars), spans))
        chars.clear()
        labels.clear()

    for line_no, line in enumerate(Path(path).read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            finish()
            continue
        fields = re.split(r'\s+', line.strip())
        if len(fields) != 2:
            raise ValueError(f'{path}:{line_no}: expected a character and BIO label')
        token, label = fields[0], fields[1].replace('_', '-')
        if label != 'O' and not label.startswith(('B-', 'I-')):
            raise ValueError(f'{path}:{line_no}: invalid BIO label {label!r}')
        for k, char in enumerate(token):
            chars.append(char)
            labels.append('I-' + label[2:] if k and label.startswith('B-') else label)
    finish()
    return samples


def build_lexicon(cfg, types):
    train_path = Path(cfg.data_dir) / 'train2.txt'
    samples = read_samples(train_path, types)
    entities, segmented = set(), Counter()
    for text, spans in tqdm(samples, desc='Build train-only lexicon', mininterval=2):
        entities.update(text[start:end] for _, start, end in spans)
        segmented.update(w.strip() for w in jieba.cut(text) if w.strip())
    words = entities | {w for w, count in segmented.items() if count >= cfg.min_segment_frequency}
    words = sorted(w for w in words if 0 < len(w) <= cfg.max_word_length and w not in ('<pad>', '<unk>'))
    return {
        'schema': cfg.schema, 'source': 'train2.txt only', 'train_sha256': sha256(train_path),
        'min_segment_frequency': cfg.min_segment_frequency,
        'max_word_length': cfg.max_word_length,
        'policy': 'all training gold mentions + jieba segments with count >= min_segment_frequency',
        'words': ['<pad>', '<unk>'] + words,
    }


class Lexicon:
    def __init__(self, record):
        self.words = record['words']
        self.trie = {}
        for word_id, word in enumerate(self.words[2:], 2):
            node = self.trie
            for char in word:
                node = node.setdefault(char, {})
            node[None] = word_id

    def match(self, text, max_words):
        matches = [[] for _ in text]
        for start in range(len(text)):
            node = self.trie
            for end in range(start, len(text)):
                node = node.get(text[end])
                if node is None:
                    break
                if None in node:
                    word_id, length = node[None], end - start + 1
                    for char in range(start, end + 1):
                        boundary = 3 if length == 1 else 0 if char == start else 2 if char == end else 1
                        matches[char].append((-length, start, word_id, boundary))
        ids = np.zeros((len(text), max_words), dtype=np.int32)
        boundaries = np.full((len(text), max_words), 4, dtype=np.int8)
        for i, candidates in enumerate(matches):
            # Label-independent selection: longest first, then earliest start and vocabulary ID.
            for slot, (_, _, word_id, boundary) in enumerate(sorted(candidates)[:max_words]):
                ids[i, slot], boundaries[i, slot] = word_id, boundary
        return ids, boundaries


def load_tokenizer(directory):
    try:
        tokenizer = AutoTokenizer.from_pretrained(directory, use_fast=True, local_files_only=True)
    except (ValueError, ImportError, OSError):
        # A complete tokenizer.json can be loaded without slow-tokenizer conversion.
        file = Path(directory) / 'tokenizer.json'
        if not file.exists():
            raise
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(file))
    if not tokenizer.is_fast:
        raise ValueError('A fast tokenizer with offset_mapping is required.')
    return tokenizer


class FeatureDataset(Dataset):
    def __init__(self, payload):
        self.items = payload['items']
        self.samples = payload['samples']
        self.types = payload['types']
        self.stats = payload['stats']
        self.pad_id = payload['pad_id']

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def prepare_split(cfg, split, lexicon_record, types):
    """Called by rank 0, then all ranks load the immutable CPU cache."""
    path = Path(cfg.data_dir) / f'{split}2.txt'
    tokenizer_files = sorted(p for p in Path(cfg.encoder_dir).iterdir()
                             if p.name.startswith(('tokenizer', 'special_tokens', 'added_tokens')))
    identity = {
        'schema': cfg.schema, 'split_sha256': sha256(path), 'types': types,
        'lexicon_sha256': hashlib.sha256(json.dumps(lexicon_record, sort_keys=True).encode()).hexdigest(),
        'tokenizer': {p.name: sha256(p) for p in tokenizer_files if p.is_file()},
        'max_chars': cfg.max_chars, 'overlap': cfg.overlap_chars,
        'token_limit': cfg.max_encoder_tokens, 'max_words': cfg.max_words,
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    target = Path(cfg.output_dir) / 'cache' / f'{split}_{digest[:16]}.pkl'
    if target.exists():
        return target
    samples = read_samples(path, types)
    tokenizer, lexicon = load_tokenizer(cfg.encoder_dir), Lexicon(lexicon_record)
    if cfg.max_chars < 1 or not 0 <= cfg.overlap_chars < cfg.max_chars:
        raise ValueError('Window overlap must be smaller than max_chars.')

    def encode_one(pair):
        sid, (text, spans) = pair
        items, start, covered = [], 0, set()
        while start < len(text):
            end = min(start + cfg.max_chars, len(text))
            while True:
                piece = text[start:end]
                encoded = tokenizer(piece, add_special_tokens=False, return_offsets_mapping=True,
                                    truncation=False, return_attention_mask=False)
                if len(encoded['input_ids']) <= cfg.max_encoder_tokens:
                    break
                if end - start == 1:
                    raise ValueError(f'One character exceeds encoder token limit in {split}, sentence {sid}')
                end = start + max(1, (end - start) // 2)
            if not encoded['input_ids']:
                raise ValueError(f'Tokenizer returned no tokens in {split}, sentence {sid}, offset {start}')
            word_ids, boundary_ids = lexicon.match(piece, cfg.max_words)
            local_spans = [(t, a - start, b - start) for t, a, b in spans if start <= a < b <= end]
            covered.update((t, a, b) for t, a, b in spans if start <= a < b <= end)
            items.append({
                'sid': sid, 'offset': start, 'length': len(piece),
                'input_ids': encoded['input_ids'], 'offsets': encoded['offset_mapping'],
                'word_ids': word_ids, 'boundary_ids': boundary_ids, 'spans': local_spans,
            })
            if end == len(text):
                break
            start = max(start + 1, end - cfg.overlap_chars)
        return items, len(set(map(tuple, spans)) - covered)

    items, uncovered = [], 0
    with ThreadPoolExecutor(max_workers=cfg.preprocessing_workers) as pool:
        results = pool.map(encode_one, enumerate(samples))
        for windows, missing in tqdm(results, total=len(samples), desc=f'Cache {split}', mininterval=2):
            items.extend(windows)
            uncovered += missing
    stats = {
        'sentences': len(samples), 'windows': len(items),
        'gold_entities': sum(len(spans) for _, spans in samples),
        'gold_not_fully_contained_in_any_window': uncovered,
        'max_encoder_tokens_observed': max(len(x['input_ids']) for x in items),
        'max_chars_observed': max(x['length'] for x in items),
        'evaluation': 'all original character gold spans; no gold dropped from metric denominator',
    }
    payload = {'identity': identity, 'items': items, 'samples': samples, 'stats': stats,
               'types': types, 'pad_id': tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0}
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix('.tmp')
    with open(temp, 'wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temp, target)
    write_json(target.with_suffix('.json'), {'identity': identity, 'stats': stats})
    return target


def load_split(path):
    with open(path, 'rb') as f:
        return FeatureDataset(pickle.load(f))


class Collator:
    def __init__(self, types, pad_id):
        self.num_types, self.pad_id = len(types), pad_id

    def __call__(self, items):
        b = len(items)
        chars, tokens = max(x['length'] for x in items), max(len(x['input_ids']) for x in items)
        words = items[0]['word_ids'].shape[-1]
        batch = {
            'input_ids': torch.full((b, tokens), self.pad_id, dtype=torch.long),
            'token_mask': torch.zeros(b, tokens, dtype=torch.bool),
            'char_pool': torch.zeros(b, chars, tokens),
            'char_mask': torch.zeros(b, chars, dtype=torch.bool),
            'word_ids': torch.zeros(b, chars, words, dtype=torch.long),
            'boundary_ids': torch.full((b, chars, words), 4, dtype=torch.long),
            'lengths': torch.tensor([x['length'] for x in items]),
            'labels': torch.zeros(b, self.num_types, chars, chars, dtype=torch.bool),
        }
        for i, x in enumerate(items):
            n, c = len(x['input_ids']), x['length']
            batch['input_ids'][i, :n] = torch.tensor(x['input_ids'])
            batch['token_mask'][i, :n] = True
            batch['char_mask'][i, :c] = True
            batch['word_ids'][i, :c] = torch.from_numpy(x['word_ids'].astype(np.int64))
            batch['boundary_ids'][i, :c] = torch.from_numpy(x['boundary_ids'].astype(np.int64))
            for token, (start, end) in enumerate(x['offsets']):
                batch['char_pool'][i, max(0, start):min(c, end), token] = 1
            for typ, start, end in x['spans']:
                batch['labels'][i, typ, start, end - 1] = True
        batch['char_pool'] /= batch['char_pool'].sum(-1, keepdim=True).clamp_min(1)
        batch['metadata'] = [(x['sid'], x['offset'], x['length']) for x in items]
        return batch
