"""Load original F1 v2 or fine-tuned v4 weights on the same model graph."""
from pathlib import Path
import gc
import json

import torch

from config import Config
from data import load_split, prepare_split, sha256
from model import LemanGP
from runtime import broadcast_object, rank, source_hashes


def candidate_path(requested, candidate):
    return (Path(requested.output_dir) / 'best.pt' if candidate == 'upgraded'
            else Path(requested.warm_start))


def load_candidate(requested, candidate, split, device):
    path = candidate_path(requested, candidate)
    checkpoint_sha = broadcast_object(sha256(path) if rank() == 0 else None)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint['schema'] != 'leman-f1-v2':
        raise ValueError(f'Expected the F1 v2 parameter layout: {path}')
    expected_source = (source_hashes(Path(__file__).parent) if candidate == 'upgraded' else
                       json.loads(Path(__file__).with_name('PARENT_SOURCE_HASHES.json').read_text()))
    if checkpoint['source_hashes'] != expected_source:
        raise ValueError(f'Checkpoint sources differ from the expected {candidate} model.')
    if candidate == 'parent' and checkpoint_sha != requested.warm_start_sha256:
        raise ValueError('Parent checkpoint differs from the completed F1 v2 best.pt.')
    cfg = Config(**checkpoint['config'])
    if candidate == 'parent':
        cfg.gp_positive_weight, cfg.gp_type_weights = 1.0, []
    cfg.root, cfg.data_dir, cfg.encoder_dir = requested.root, requested.data_dir, requested.encoder_dir
    cfg.output_dir = str(Path(requested.output_dir) / ('parent_reference' if candidate == 'parent' else '.'))
    cfg.workers, cfg.preprocessing_workers = requested.workers, requested.preprocessing_workers
    cfg.eval_batch_size, cfg.threshold = requested.eval_batch_size, 0.0
    lexicon_path = path.parent / 'lexicon.json'
    if sha256(lexicon_path) != checkpoint['lexicon_sha256']:
        raise ValueError(f'Keep the original lexicon.json alongside {path}')
    for family in ('encoder_files', 'encoder_weights'):
        for name, expected in checkpoint['input_identity'].get(family, {}).items():
            if sha256(Path(cfg.encoder_dir) / name) != expected:
                raise ValueError(f'Encoder/tokenizer file changed: {name}')
    if split == 'dev' and sha256(Path(cfg.data_dir) / 'dev2.txt') != checkpoint['input_identity']['dev2.txt']:
        raise ValueError('Dev data differs from checkpoint selection.')
    lexicon = json.loads(lexicon_path.read_text(encoding='utf-8'))
    types = checkpoint['types']
    cache = broadcast_object(str(prepare_split(cfg, split, lexicon, types)) if rank() == 0 else None)
    dataset = load_split(cache)
    model = LemanGP(cfg, len(types), checkpoint['vocab_size'],
                    char_vocab_size=checkpoint['char_vocab_size'],
                    encoder_config=checkpoint['encoder_config'], pretrained=False)
    model.load_state_dict(checkpoint['model'], strict=True)
    info = {
        'candidate': candidate, 'checkpoint': str(path), 'checkpoint_sha256': checkpoint_sha,
        'checkpoint_epoch': checkpoint['best_epoch'], 'weight_source': checkpoint.get('weight_source', 'live'),
        'lexicon_sha256': checkpoint['lexicon_sha256'], 'model_logit_divisor': cfg.gp_head_size ** 0.5,
        'recorded_checkpoint_dev': checkpoint['dev'], 'windows': dataset.stats,
    }
    del checkpoint
    gc.collect()
    return model.to(device), dataset, cfg, info
