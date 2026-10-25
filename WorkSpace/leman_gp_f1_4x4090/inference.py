"""Load either the upgraded model or the original completed model."""
from pathlib import Path
from types import SimpleNamespace
import gc
import json
import math

import torch

from config import Config
from data import load_split, prepare_split, sha256
from model import LemanGP
import paper_core
from runtime import broadcast_object, rank, source_hashes


class ParentModel(torch.nn.Module):
    """Keep the original forward and ignore the added character-ID field."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, char_ids=None, **batch):
        return self.model(**batch)


def candidate_path(requested, candidate):
    return (Path(requested.output_dir) / 'best.pt' if candidate == 'upgraded'
            else Path(requested.warm_start))


def load_candidate(requested, candidate, split, device):
    path = candidate_path(requested, candidate)
    checkpoint_sha = broadcast_object(sha256(path) if rank() == 0 else None)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    expected_schema = requested.schema if candidate == 'upgraded' else 'leman-paper-char-gp-v1'
    if checkpoint['schema'] != expected_schema:
        raise ValueError(f'Wrong checkpoint schema: {path}')
    if candidate == 'upgraded':
        if checkpoint['source_hashes'] != source_hashes(Path(__file__).parent):
            raise ValueError('Use the source_snapshot from this upgraded training run.')
        cfg = Config(**checkpoint['config'])
    else:
        if checkpoint_sha != requested.warm_start_sha256:
            raise ValueError('Parent checkpoint SHA256 differs from the completed seed42 run.')
        if checkpoint['source_hashes']['model.py'] != sha256(Path(__file__).with_name('paper_core.py')):
            raise ValueError('Parent checkpoint does not match paper_core.py.')
        cfg = SimpleNamespace(**checkpoint['config'])
    cfg.root, cfg.data_dir, cfg.encoder_dir = requested.root, requested.data_dir, requested.encoder_dir
    cfg.output_dir = str(Path(requested.output_dir) / ('parent_reference' if candidate == 'parent' else '.'))
    cfg.workers, cfg.preprocessing_workers = requested.workers, requested.preprocessing_workers
    cfg.eval_batch_size, cfg.threshold = requested.eval_batch_size, 0.0
    lexicon_path = path.parent / 'lexicon.json'
    if sha256(lexicon_path) != checkpoint['lexicon_sha256']:
        raise ValueError(f'Keep the training vocabulary alongside {path}')
    for name, expected in checkpoint['input_identity']['encoder_files'].items():
        if sha256(Path(cfg.encoder_dir) / name) != expected:
            raise ValueError(f'Tokenizer/encoder config changed: {name}')
    if split == 'dev' and sha256(Path(cfg.data_dir) / 'dev2.txt') != checkpoint['input_identity']['dev2.txt']:
        raise ValueError('Dev data differs from training model selection.')
    for name, expected in checkpoint['input_identity'].get('encoder_weights', {}).items():
        if sha256(Path(cfg.encoder_dir) / name) != expected:
            raise ValueError(f'Encoder weights changed: {name}. Restore the same ERNIE files.')
    lexicon = json.loads(lexicon_path.read_text(encoding='utf-8'))
    types = checkpoint['types']
    cache = broadcast_object(str(prepare_split(cfg, split, lexicon, types)) if rank() == 0 else None)
    dataset = load_split(cache)
    if candidate == 'upgraded':
        model = LemanGP(cfg, len(types), checkpoint['vocab_size'],
                        char_vocab_size=checkpoint['char_vocab_size'],
                        encoder_config=checkpoint['encoder_config'], pretrained=False)
        model.load_state_dict(checkpoint['model'], strict=True)
    else:
        original = paper_core.LemanGP(cfg, len(types), checkpoint['vocab_size'],
                                      encoder_config=checkpoint['encoder_config'], pretrained=False)
        original.load_state_dict(checkpoint['model'], strict=True)
        model = ParentModel(original)
    info = {
        'candidate': candidate, 'checkpoint': str(path),
        'checkpoint_sha256': broadcast_object(checkpoint_sha if rank() == 0 else None),
        'checkpoint_epoch': checkpoint['best_epoch'],
        'weight_source': checkpoint.get('weight_source', 'live'),
        'lexicon_sha256': checkpoint['lexicon_sha256'],
        'model_logit_divisor': math.sqrt(cfg.gp_head_size) if candidate == 'upgraded' else 1.0,
        'recorded_checkpoint_dev': checkpoint['dev'],
        'windows': dataset.stats,
    }
    del checkpoint
    gc.collect()
    return model.to(device), dataset, cfg, info
