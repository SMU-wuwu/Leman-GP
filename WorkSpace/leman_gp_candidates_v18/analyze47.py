from pathlib import Path
import json
import sys
import time

import numpy as np

ROOT = Path('/root/data/WorkSpace')
sys.path.insert(0, str(ROOT / 'leman_gp_typecal_v9'))
sys.path.insert(0, str(ROOT / 'leman_gp_typecal_v9' / 'frozen_v4'))

from common import named_predictions, read_cache, read_json, sha256, subset_score
from calibration_core import apply_logistic, candidates_from_scores, fit_logistic
from config import Config
from data import read_samples, write_json
from evaluate import score_predictions
from search import folds_for, predict, select_fit


def build_features(records, samples, types, gazetteer):
    keys, values = [], []
    for sid, (text, _) in enumerate(samples):
        models = []
        for source in records:
            spans = {}
            for typ, a, b, score, penalty in source[sid]:
                key = (typ, a, b)
                if key not in spans or (score, penalty) > spans[key]:
                    spans[key] = (score, penalty)
            models.append(spans)
        union = set().union(*(set(model) for model in models))
        for typ, a, b in sorted(union):
            counts = gazetteer['surfaces'].get(text[a:b], {})
            same = counts.get(types[typ], 0)
            extra = (np.log1p(b - a), np.log1p(same),
                     np.log1p(sum(counts.values()) - same))
            row = []
            for model in models:
                score, penalty = model.get((typ, a, b), (-32.0, -16.0))
                row.extend((score / 8.0, penalty, np.exp(penalty), *extra))
            keys.append((sid, typ, a, b))
            values.append(row)
    return keys, np.asarray(values, dtype=np.float64)


def coverage(keys, samples):
    candidates = set(keys)
    gold = {(sid, *span) for sid, (_, spans) in enumerate(samples) for span in spans}
    return {'gold': len(gold), 'available_gold': len(gold & candidates),
            'gold_coverage': len(gold & candidates) / len(gold),
            'unique_candidates': len(candidates)}


def metric_f1(report):
    return 100.0 * report['micro']['f1']


def main():
    started = time.perf_counter()
    cfg = Config.from_environment()
    root = Path(cfg.root)
    output = root / 'output/leman_candidates_v18_seed47'
    output.mkdir(parents=True, exist_ok=True)
    package = root / 'leman_gp_typecal_v9'
    spec = read_json(package / 'POLICY.json')
    gazetteer = read_json(package / 'inputs/v6_train_gazetteer.json')
    paths = {
        'old': package / 'inputs/v6_upgraded_features.jsonl.gz',
        'old_expanded': root / 'output/leman_inference_v7_seed42/upgraded_features.jsonl.gz',
        'fgm43': root / 'output/leman_fgm_v10_analysis_seed43/fgm_features.jsonl.gz',
        'fgm44': root / 'output/leman_fgm_v14_analysis_seed44/fgm_features.jsonl.gz',
        'fgm47': output / 'fgm_features.jsonl.gz',
    }
    caches = {name: read_cache(path) for name, path in paths.items()}
    types = caches['fgm47'][0]['types']
    samples = read_samples(root / 'data/dev2.txt', types)
    ids = list(range(len(samples)))
    folds = folds_for(samples)
    partitions = {'full': ids,
                  'held0': [sid for sid in ids if folds[sid] != 0],
                  'held1': [sid for sid in ids if folds[sid] != 1]}
    routes = {
        'fgm47': ['fgm47'],
        'old47': ['old_expanded', 'fgm47'],
        'fgm434447': ['fgm43', 'fgm44', 'fgm47'],
        'all47': ['old_expanded', 'fgm43', 'fgm44', 'fgm47'],
    }
    baseline_dir = root / 'output/leman_rdrop_v17_analysis_seed45'
    if not (baseline_dir / 'crossfit_predictions.jsonl').exists():
        baseline_dir = root / 'output/leman_fgm_v14_analysis_seed44'
    baseline = {
        'full_dev': named_predictions(baseline_dir / 'full_dev_predictions.jsonl', types),
        'crossfit': named_predictions(baseline_dir / 'crossfit_predictions.jsonl', types),
    }
    baseline_reports = {mode: score_predictions(samples, pred, types)
                        for mode, pred in baseline.items()}
    print(f'Baseline {baseline_dir.name}: full={metric_f1(baseline_reports["full_dev"]):.6f}% '
          f'crossfit={metric_f1(baseline_reports["crossfit"]):.6f}%', flush=True)
    results = {'baseline': {'folder': str(baseline_dir), 'reports': baseline_reports},
               'routes': {}, 'test_accessed': False}
    feature_cache = {}
    for route, names in routes.items():
        signature = tuple(names)
        if signature not in feature_cache:
            feature_cache[signature] = build_features(
                [caches[name][1] for name in names], samples, types, gazetteer)
        keys, values = feature_cache[signature]
        print(f'{route}: candidates={len(keys)} features={values.shape[1]} '
              f'coverage={coverage(keys, samples)["gold_coverage"]:.6f}', flush=True)
        fits, preds, reports = {}, {'full_dev': {}, 'crossfit': {}}, {}
        route_spec = {**spec, 'interaction_features':
                      [i for m in range(len(names)) for i in (6*m, 6*m+1)]}
        all_candidates = {}
        for part, fit_ids in partitions.items():
            logistic, scores = fit_logistic(keys, values, samples, types, fit_ids, route_spec)
            assert logistic['gradient_max'] < 1e-5
            candidates = candidates_from_scores(keys, scores, len(samples))
            fitted = select_fit(samples, {0.0: candidates}, types, cfg, fit_ids, 'expanded')
            fitted.update({'family': route, 'component_models': names,
                           'candidate_floor': -32.0,
                           'logistic': logistic})
            fits[part] = fitted
            calibrated = apply_logistic(keys, values, logistic)
            calibrated_candidates = candidates_from_scores(keys, calibrated, len(samples))
            held_ids = ids if part == 'full' else [sid for sid in ids if folds[sid] == int(part[-1])]
            mode = 'full_dev' if part == 'full' else 'crossfit'
            preds[mode].update(predict(calibrated_candidates, fitted, types, held_ids))
        reports = {mode: score_predictions(samples, pred, types)
                   for mode, pred in preds.items()}
        selected = {part: fits[part] for part in partitions}
        combined = {'full_dev': preds['full_dev'], 'crossfit': {}}
        for h in (0, 1):
            combined['crossfit'].update({sid: preds['crossfit'][sid]
                                         for sid in range(len(samples))
                                         if folds[sid] == h})
        # The route-specific predictions are already generated with the held-out fits.
        nested = {mode: score_predictions(samples, pred, types)
                  for mode, pred in combined.items()}
        fold_reports = []
        for h in (0, 1):
            held = [sid for sid in ids if folds[sid] == h]
            cur = subset_score(samples, combined['crossfit'], types, held)
            base = subset_score(samples, baseline['crossfit'], types, held)
            fold_reports.append({'held_out_fold': h,
                                 'route': cur, 'baseline': base,
                                 'delta_pp': metric_f1(cur) - metric_f1(base)})
        results['routes'][route] = {
            'components': names,
            'coverage': coverage(keys, samples),
            'reports': reports,
            'nested': nested,
            'fold_reports': fold_reports,
            'fit_decoder': {part: fits[part] for part in fits},
        }
        print(f'{route}: full={metric_f1(nested["full_dev"]):.6f}% '
              f'crossfit={metric_f1(nested["crossfit"]):.6f}% '
              f'fold_deltas={[round(x["delta_pp"], 6) for x in fold_reports]}', flush=True)
    # Candidate route is retained only if both held-out folds improve over baseline.
    accepted = []
    for route, result in results['routes'].items():
        if (metric_f1(result['nested']['full_dev']) >= metric_f1(baseline_reports['full_dev']) and
                metric_f1(result['nested']['crossfit']) > metric_f1(baseline_reports['crossfit']) and
                min(x['delta_pp'] for x in result['fold_reports']) >= 0):
            accepted.append(route)
    results['accepted_routes'] = accepted
    results['seconds'] = time.perf_counter() - started
    write_json(output / 'analysis_v18.json', results)
    print('Accepted routes:', accepted, flush=True)
    print('Saved:', output / 'analysis_v18.json', flush=True)


if __name__ == '__main__':
    main()
