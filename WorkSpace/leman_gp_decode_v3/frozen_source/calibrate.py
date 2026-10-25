"""Freeze dev decoder settings; compare upgraded and original weights on dev."""
from pathlib import Path
import gc
import json
import time

import torch
import torch.distributed as dist

from config import Config
from data import sha256, write_json
from decoding import crossfit, decode, fit
from evaluate import collect_scores, release_loader, score_predictions, write_predictions
from inference import load_candidate
from runtime import barrier, broadcast_object, rank, setup


def calibrated_metrics(samples, predictions, types, fitted):
    metrics = score_predictions(samples, predictions, types)
    metrics['decoder'] = fitted['strategy']
    metrics['overlap_policy'] = ('retain all' if fitted['strategy'] == 'all'
                                 else 'maximum total positive score margin; non-overlapping intervals')
    return metrics


def main():
    device = setup()
    try:
        requested = Config.from_environment()
        output = Path(requested.output_dir) / 'dev_calibration'
        reports = {}
        for candidate in ('upgraded', 'parent'):
            started = time.perf_counter()
            model, dataset, cfg, info = load_candidate(requested, candidate, 'dev', device)
            score_scale = cfg.gp_head_size ** 0.5 if candidate == 'upgraded' else 1.0
            candidates = collect_scores(model, dataset, cfg, device, requested.calibration_floor,
                                        Path(requested.output_dir) / 'cache' / 'score_shards' / candidate,
                                        score_scale=score_scale)
            release_loader(dataset)
            del model
            gc.collect()
            torch.cuda.empty_cache()
            report = None
            if rank() == 0:
                folder = output / candidate
                folder.mkdir(parents=True, exist_ok=True)
                print(f'Fitting {candidate} on dev only: {sum(map(len, candidates.values())):,} scored spans', flush=True)
                raw = decode(candidates, [0.0] * len(dataset.types), 'all')
                raw_metrics = score_predictions(dataset.samples, raw, dataset.types)
                fitted = fit(dataset.samples, candidates, dataset.types, requested)
                full = decode(candidates, [fitted['thresholds'][t] for t in dataset.types], fitted['strategy'])
                full_metrics = calibrated_metrics(dataset.samples, full, dataset.types, fitted)
                cross_metrics, cross_predictions, folds = crossfit(dataset.samples, candidates, dataset.types, requested)
                cross_metrics['overlap_policy'] = 'decoder strategy fitted on the opposite dev half'
                report = {
                    **info, 'dev_sha256': sha256(Path(requested.data_dir) / 'dev2.txt'),
                    'raw': raw_metrics, 'full_dev': full_metrics, 'crossfit': cross_metrics,
                    'decoder_score_scale': score_scale,
                    'decoder': fitted, 'candidate_floor': requested.calibration_floor,
                    'global_grid': [-4.0 + i * requested.calibration_step
                                    for i in range(round(8.0 / requested.calibration_step) + 1)],
                    'type_search': 'one pass; within 1.0 of global threshold; support >=200 in fitting partition',
                    'seconds_including_forward': time.perf_counter() - started,
                    'selection_scope': 'dev only; no test access',
                }
                write_json(folder / 'calibration.json', report)
                write_json(folder / 'crossfit_folds.json', folds)
                write_predictions(folder / 'raw_predictions.jsonl', dataset.samples, raw, dataset.types)
                write_predictions(folder / 'calibrated_predictions.jsonl', dataset.samples, full, dataset.types)
                write_predictions(folder / 'crossfit_predictions.jsonl', dataset.samples, cross_predictions, dataset.types)
                print(f'{candidate}: raw Micro={raw_metrics["micro"]["f1"]:.6f}; '
                      f'cal Micro={full_metrics["micro"]["f1"]:.6f}; '
                      f'crossfit Micro={cross_metrics["micro"]["f1"]:.6f}; '
                      f'cal Macro={full_metrics["macro_f1"]:.6f}; strategy={fitted["strategy"]}', flush=True)
                del raw, full, cross_predictions
            reports[candidate] = broadcast_object(report)
            del candidates, dataset
            gc.collect()
        if rank() == 0:
            chosen = max(reports, key=lambda k: (reports[k]['crossfit']['micro']['f1'],
                                                reports[k]['full_dev']['micro']['f1'],
                                                reports[k]['full_dev']['macro_f1'], k == 'parent'))
            selection = {
                'candidate': chosen, 'selected': reports[chosen],
                'criterion': 'crossfit exact Micro F1; tie: full-dev Micro, Macro13, then parent',
                'comparison': {k: {name: value[name] for name in ('raw', 'full_dev', 'crossfit')}
                               for k, value in reports.items()},
                'test_accessed': False,
            }
            write_json(Path(requested.output_dir) / 'selection.json', selection)
            print(f'Frozen inference choice: {chosen}. Run bash test.sh when ready.', flush=True)
        barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
