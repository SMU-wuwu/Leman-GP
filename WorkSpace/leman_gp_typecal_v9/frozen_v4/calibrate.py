"""Compare original and fine-tuned weights with the unchanged decode-v3 search."""
from pathlib import Path
import gc
import gzip
import json

import torch
import torch.distributed as dist

from config import Config
from data import sha256, write_json
from evaluate import release_loader, score_predictions, write_predictions
from features import collect_features
from inference import load_candidate
from runtime import barrier, broadcast_object, rank, setup
from search import POLICY, compare, predict, scored_candidates


def choose_candidate(reports):
    parent, upgraded = reports['parent'], reports['upgraded']
    return ('upgraded' if upgraded['crossfit']['micro']['f1'] > parent['crossfit']['micro']['f1']
            and upgraded['full_dev']['micro']['f1'] >= parent['full_dev']['micro']['f1'] else 'parent')


def main():
    device = setup()
    try:
        requested = Config.from_environment()
        run = Path(requested.output_dir)
        reports = {}
        for candidate in ('parent', 'upgraded'):
            model, dataset, cfg, info = load_candidate(requested, candidate, 'dev', device)
            records = collect_features(model, dataset, cfg, device,
                                       run / 'cache' / 'score_shards' / candidate, POLICY['floor'])
            release_loader(dataset)
            del model
            gc.collect()
            torch.cuda.empty_cache()
            report = None
            if rank() == 0:
                folder = run / 'dev_calibration' / candidate
                folder.mkdir(parents=True, exist_ok=True)
                scale = cfg.gp_head_size ** 0.5
                result, predictions = compare(dataset.samples, records, dataset.types, requested, scale)
                result.update(**info, dev_sha256=sha256(Path(cfg.data_dir) / 'dev2.txt'))
                raw_candidates = scored_candidates(records, 0, scale)
                raw = predict(raw_candidates, {'thresholds': dict.fromkeys(dataset.types, 0.0),
                                               'strategy': 'all'}, dataset.types, range(len(dataset.samples)))
                raw_metrics = score_predictions(dataset.samples, raw, dataset.types)
                write_predictions(folder / 'raw_predictions.jsonl', dataset.samples, raw, dataset.types)
                for family, modes in predictions.items():
                    (folder / family).mkdir(exist_ok=True)
                    for mode, prediction in modes.items():
                        write_predictions(folder / family / f'{mode}_predictions.jsonl',
                                          dataset.samples, prediction, dataset.types)
                with gzip.open(folder / 'dev_scored_candidates.jsonl.gz', 'wt', encoding='utf-8') as stream:
                    stream.write(json.dumps({'checkpoint': info, 'types': dataset.types, 'score_scale': scale,
                                             'policy': POLICY, 'dev_sha256': result['dev_sha256']}) + '\n')
                    for sid, rows in sorted(records.items()):
                        stream.write(json.dumps({'sid': sid, 'rows': rows}) + '\n')
                write_json(folder / 'decoder_search.json', result)
                chosen_family = result['chosen']
                report = {**info, **result['reports'][chosen_family], 'family': chosen_family,
                          'raw': raw_metrics, 'dev_sha256': result['dev_sha256'],
                          'decoder_score_scale': scale, 'candidate_floor': POLICY['floor'],
                          'selection_scope': result['scope']}
                write_json(folder / 'calibration.json', report)
                write_json(folder / 'crossfit_folds.json', result['folds'])
                del predictions, raw, raw_candidates
            reports[candidate] = broadcast_object(report)
            del records, dataset
            gc.collect()
        if rank() == 0:
            chosen = choose_candidate(reports)
            selection = {'candidate': chosen, 'selected': reports[chosen],
                         'criterion': 'upgrade only if chosen-family crossfit Micro strictly improves '
                                      'and chosen-family full-dev Micro does not decrease; otherwise parent',
                         'comparison': reports, 'decoder_policy': POLICY, 'test_accessed': False}
            write_json(run / 'selection.json', selection)
            lines = ['# Fine-tune v4 dev comparison', '',
                     '| Candidate | Decoder | Raw Micro | Full Micro | Crossfit Micro | Crossfit Macro13 |',
                     '|---|---|---:|---:|---:|---:|']
            for candidate, report in reports.items():
                lines.append(f'| {candidate} | {report["family"]} | {100*report["raw"]["micro"]["f1"]:.4f} | '
                             f'{100*report["full_dev"]["micro"]["f1"]:.4f} | '
                             f'{100*report["crossfit"]["micro"]["f1"]:.4f} | '
                             f'{100*report["crossfit"]["macro_f1"]:.4f} |')
            lines.extend(['', f'Chosen candidate: {chosen}.', '',
                          'Both candidates use the unchanged decode-v3 search and folds. '
                          'These are development comparisons; test was not read.'])
            (run / 'SUMMARY.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
            print('\n'.join(lines), flush=True)
        barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
