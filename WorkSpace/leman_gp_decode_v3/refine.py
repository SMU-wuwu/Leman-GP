"""Four-GPU dev-only decoding exploration on an existing F1 v2 checkpoint."""
import argparse
import gc
import gzip
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'frozen_source'))

import torch
import torch.distributed as dist

from config import Config
from data import sha256, write_json
from evaluate import release_loader, score_predictions, write_predictions
from inference import load_candidate
from runtime import barrier, broadcast_object, environment, rank, setup, source_hashes
from features import collect_features
from search import POLICY, compare, predict, scored_candidates


def output_directory(cfg):
    return Path(os.environ.get('LEMAN_REFINE_OUTPUT',
                               str(Path(cfg.root) / 'output' / f'leman_decode_v3_seed{cfg.seed}')))


def save_pickle(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with open(temporary, 'wb') as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def load_features(requested, split, output, device):
    identity = None
    if rank() == 0:
        identity = {
            'checkpoint_sha256': sha256(Path(requested.output_dir) / 'best.pt'),
            'lexicon_sha256': sha256(Path(requested.output_dir) / 'lexicon.json'),
            'split': split, 'data_sha256': sha256(Path(requested.data_dir) / f'{split}2.txt'),
            'frozen_source': source_hashes(HERE / 'frozen_source'),
            'feature_script_sha256': sha256(HERE / 'features.py'),
            'floor': POLICY['floor'],
        }
    identity = broadcast_object(identity)
    cache = output / 'cache' / f'{split}_features.pkl'
    payload, reusable = None, False
    if rank() == 0 and cache.exists():
        with open(cache, 'rb') as stream:
            payload = pickle.load(stream)
        reusable = payload['identity'] == identity
    reusable = broadcast_object(reusable)
    if reusable:
        if rank() == 0:
            print(f'Reuse completed {split} score cache; no GPU forward required.', flush=True)
        return payload if rank() == 0 else None
    model, dataset, cfg, info = load_candidate(requested, 'upgraded', split, device)
    records = collect_features(model, dataset, cfg, device, output / 'cache' / f'{split}_shards', POLICY['floor'])
    release_loader(dataset)
    if rank() == 0:
        payload = {'identity': identity, 'info': info, 'samples': dataset.samples, 'types': dataset.types,
                   'records': records, 'score_scale': cfg.gp_head_size ** 0.5,
                   'environment': environment(device)}
        save_pickle(cache, payload)
    del model, dataset
    gc.collect()
    torch.cuda.empty_cache()
    return payload


def snapshot(output):
    for source in [*HERE.glob('*.py'), *HERE.glob('*.sh')]:
        target = output / 'source_snapshot' / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for source in (HERE / 'frozen_source').glob('*.py'):
        target = output / 'source_snapshot' / 'frozen_source' / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def dev(requested, output, device):
    payload = load_features(requested, 'dev', output, device)
    if rank() != 0:
        return
    snapshot(output)
    started = time.perf_counter()
    samples, types = payload['samples'], payload['types']
    result, predictions = compare(samples, payload['records'], types, requested, payload['score_scale'])
    result.update(checkpoint=payload['info'], feature_identity=payload['identity'],
                  environment=payload['environment'], search_seconds=time.perf_counter() - started,
                  score_cache_sha256=sha256(output / 'cache' / 'dev_features.pkl'))
    reference = Path(requested.output_dir) / 'dev_calibration' / 'upgraded' / 'calibration.json'
    if reference.exists():
        original = json.loads(reference.read_text(encoding='utf-8'))
        if original['checkpoint_sha256'] == payload['info']['checkpoint_sha256']:
            result['baseline_replay_delta_pp'] = {
                name: {'micro': 100 * (result['reports']['baseline'][name]['micro']['f1']
                                      - original[name]['micro']['f1']),
                       'macro': 100 * (result['reports']['baseline'][name]['macro_f1'] - original[name]['macro_f1'])}
                for name in ('full_dev', 'crossfit')}
    for family, modes in predictions.items():
        folder = output / 'dev' / family
        folder.mkdir(parents=True, exist_ok=True)
        write_json(folder / 'metrics.json', result['reports'][family])
        for name, pred in modes.items():
            write_predictions(folder / f'{name}_predictions.jsonl', samples, pred, types)
    # Keep scored candidates in the small return, so subsequent decoding analysis
    # does not need another server forward or any checkpoint transfer.
    with gzip.open(output / 'dev_scored_candidates.jsonl.gz', 'wt', encoding='utf-8') as stream:
        stream.write(json.dumps({'identity': payload['identity'], 'types': types,
                                 'score_scale': payload['score_scale'], 'policy': POLICY}) + '\n')
        for sid, rows in sorted(payload['records'].items()):
            stream.write(json.dumps({'sid': sid, 'rows': rows}, ensure_ascii=False) + '\n')
    chosen = result['chosen']
    selection = {
        'family': chosen, 'decoder': result['reports'][chosen]['decoder'],
        'checkpoint_sha256': payload['info']['checkpoint_sha256'],
        'lexicon_sha256': payload['info']['lexicon_sha256'],
        'dev_sha256': payload['identity']['data_sha256'], 'score_scale': payload['score_scale'],
        'policy': POLICY, 'method_sources': {p.name: sha256(p) for p in HERE.glob('*.py')},
        'criterion': 'largest cross-fit Micro among families with no full-dev Micro loss; '
                     'retain baseline unless cross-fit Micro strictly improves',
        'test_accessed': False,
    }
    write_json(output / 'dev_search.json', result)
    write_json(output / 'selection.json', selection)
    lines = ['# Dev decoding exploration', '', '| Decoder | Full Micro | Cross-fit Micro | Cross-fit Macro13 |',
             '|---|---:|---:|---:|']
    for name, report in result['reports'].items():
        lines.append(f'| {name} | {100 * report["full_dev"]["micro"]["f1"]:.4f} | '
                     f'{100 * report["crossfit"]["micro"]["f1"]:.4f} | '
                     f'{100 * report["crossfit"]["macro_f1"]:.4f} |')
    lines.extend(['', f'Chosen decoder: {chosen}.', '', result['scope'], '',
                  'No weights were updated. No test data was read.'])
    (output / 'SUMMARY.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n' + '\n'.join(lines), flush=True)


def test(requested, output, device):
    selection = json.loads((output / 'selection.json').read_text(encoding='utf-8'))
    if selection['dev_sha256'] != sha256(Path(requested.data_dir) / 'dev2.txt'):
        raise ValueError('Use the same dev data as decoder fitting.')
    if selection['method_sources'] != {p.name: sha256(p) for p in HERE.glob('*.py')}:
        raise ValueError('Use the decoding tools that produced selection.json.')
    payload = load_features(requested, 'test', output, device)
    if rank() != 0:
        return
    if payload['info']['checkpoint_sha256'] != selection['checkpoint_sha256']:
        raise ValueError('Checkpoint changed after decoder selection.')
    fitted = selection['decoder']
    candidates = scored_candidates(payload['records'], fitted['boundary_weight'], selection['score_scale'])
    predictions = predict(candidates, fitted, payload['types'], list(range(len(payload['samples']))))
    metrics = score_predictions(payload['samples'], predictions, payload['types'])
    metrics.update(checkpoint=payload['info'], decoder=selection,
                   data_sha256=payload['identity']['data_sha256'], environment=payload['environment'],
                   overlap_policy=('retain all' if fitted['strategy'] == 'all' else 'optimal non-overlapping intervals'),
                   selection_sha256=sha256(output / 'selection.json'),
                   scope='Frozen dev decoder; test already evaluated in earlier experimentation.')
    folder = output / 'test'
    folder.mkdir(exist_ok=True)
    write_json(folder / 'test_results.json', metrics)
    write_predictions(folder / 'test_predictions.jsonl', payload['samples'], predictions, payload['types'])
    print(f'Test Micro={metrics["micro"]["f1"]:.6f}, Macro13={metrics["macro_f1"]:.6f}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['dev', 'test'])
    args = parser.parse_args()
    device = setup()
    try:
        requested = Config.from_environment()
        output = output_directory(requested)
        output.mkdir(parents=True, exist_ok=True)
        (dev if args.stage == 'dev' else test)(requested, output, device)
        barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
