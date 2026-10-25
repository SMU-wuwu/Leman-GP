"""One distributed GPU evaluation with decoder settings already frozen on dev."""
import argparse
import csv
import json
from pathlib import Path

import torch
import torch.distributed as dist

from config import Config
from data import sha256, write_json
from decoding import decode
from evaluate import collect_scores, release_loader, score_predictions, write_predictions
from inference import load_candidate
from runtime import barrier, environment, rank, setup, world_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['test', 'dev'], default='test')
    args = parser.parse_args()
    device = setup()
    try:
        requested = Config.from_environment()
        run = Path(requested.output_dir)
        selection = json.loads((run / 'selection.json').read_text(encoding='utf-8'))
        candidate = selection['candidate']
        calibration = json.loads((run / 'dev_calibration' / candidate / 'calibration.json').read_text(encoding='utf-8'))
        if calibration['dev_sha256'] != sha256(Path(requested.data_dir) / 'dev2.txt'):
            raise ValueError('Dev changed after decoder fitting. Keep the frozen dev data.')
        model, dataset, cfg, info = load_candidate(requested, candidate, args.split, device)
        if info['checkpoint_sha256'] != calibration['checkpoint_sha256']:
            raise ValueError('Checkpoint changed after decoder fitting. Run calibrate.sh again.')
        if rank() == 0:
            print(f'Evaluate {candidate}, epoch {info["checkpoint_epoch"]}, {world_size()} CUDA GPUs, '
                  f'{torch.cuda.get_device_name(device)}, BF16, {cfg.eval_batch_size}/GPU', flush=True)
        candidates = collect_scores(model, dataset, cfg, device, calibration['candidate_floor'],
                                    run / 'cache' / 'score_shards' / f'eval_{args.split}_{candidate}',
                                    score_scale=calibration['decoder_score_scale'])
        release_loader(dataset)
        if rank() == 0:
            fitted = calibration['decoder']
            predictions = decode(candidates, [fitted['thresholds'][t] for t in dataset.types], fitted['strategy'])
            result = score_predictions(dataset.samples, predictions, dataset.types)
            raw = score_predictions(dataset.samples, decode(candidates, [0.0] * len(dataset.types), 'all'), dataset.types)
            result.update({**info, 'raw_threshold_zero': raw, 'decoder': fitted,
                           'data_sha256': sha256(Path(requested.data_dir) / f'{args.split}2.txt'),
                           'environment': environment(device), 'metric_selection_data': 'dev only',
                           'overlap_policy': ('retain all' if fitted['strategy'] == 'all'
                                              else 'optimal non-overlapping intervals')})
            folder = run / f'{args.split}_evaluation' / candidate
            folder.mkdir(parents=True, exist_ok=True)
            write_json(folder / f'{args.split}_results.json', result)
            write_predictions(folder / f'{args.split}_predictions.jsonl', dataset.samples, predictions, dataset.types)
            with open(folder / 'entity_metrics.csv', 'w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['entity_type', 'precision', 'recall', 'f1',
                                                            'support', 'true_positive', 'predicted'])
                writer.writeheader()
                for name, row in result['per_type'].items():
                    writer.writerow({'entity_type': name, **row})
            print(f'{args.split}: Micro F1={result["micro"]["f1"]:.6f}, Macro13={result["macro_f1"]:.6f}; '
                  f'raw Micro={raw["micro"]["f1"]:.6f}\nArtifacts: {folder}', flush=True)
        barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
