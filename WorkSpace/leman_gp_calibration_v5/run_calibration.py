"""Fit/replay v5 using returned features on CPU; every completed fit is reusable."""
from pathlib import Path
import json
import os
import platform
import shutil
import time

import numpy as np

from common import (PACKAGE, named_predictions, paths, read_cache, read_json,
                    sha256, subset_score, verify_package)
from calibration_core import apply_logistic, candidates_from_scores, feature_rows, fit_logistic
from config import Config
from data import read_samples, write_json
from evaluate import score_predictions, write_predictions
from search import folds_for, predict, select_fit


def main():
    started = time.perf_counter()
    manifest = verify_package()
    root, output = paths()
    output.mkdir(parents=True, exist_ok=True)
    dev_path = Path(os.environ.get('LEMAN_GP_DATA', root / 'data')) / 'dev2.txt'
    if sha256(dev_path) != manifest['dev_sha256']:
        raise ValueError('Use the dev2.txt that produced the returned v4 scores.')
    spec = read_json(PACKAGE / 'POLICY.json')
    h0, parent = read_cache(PACKAGE / 'inputs/parent_features.jsonl.gz')
    h1, upgraded = read_cache(PACKAGE / 'inputs/upgraded_features.jsonl.gz')
    assert h0['types'] == h1['types'] and h0['score_scale'] == h1['score_scale'] == 8.0
    assert h0['dev_sha256'] == h1['dev_sha256'] == manifest['dev_sha256']
    for name, header in [('parent', h0), ('upgraded', h1)]:
        assert header['checkpoint']['checkpoint_sha256'] == manifest['checkpoint_sha256'][name]
    types = h1['types']
    samples = read_samples(dev_path, types)
    ids, folds = list(range(len(samples))), folds_for(samples)
    assert set(parent) == set(upgraded) == set(ids)
    partitions = {'full': ids, **{f'held{h}': [i for i in ids if folds[i] != h] for h in (0, 1)}}
    identity = {'package_manifest_sha256': sha256(PACKAGE / 'PACKAGE_MANIFEST.json'),
                'dev_sha256': manifest['dev_sha256'], 'test_accessed': False}
    fit_dir = output / 'fits'
    fit_dir.mkdir(exist_ok=True)
    fits, predictions, reports = {}, {}, {}
    for family in spec['families']:
        print(f'\nCandidate: {family}', flush=True)
        keys, values = feature_rows(parent, upgraded, family)
        fits[family], predictions[family] = {}, {'full_dev': {}, 'crossfit': {}}
        for name, fit_ids in partitions.items():
            cache_path = fit_dir / f'{family}_{name}.json'
            if cache_path.exists():
                saved = read_json(cache_path)
                if saved['identity'] != identity:
                    raise ValueError(f'Use a new output directory for changed inputs: {cache_path}')
                fitted_record = saved['fit']
                print(f'Reuse completed fit: {family}/{name}', flush=True)
                scores = apply_logistic(keys, values, fitted_record['logistic'])
                candidates = candidates_from_scores(keys, scores, len(samples))
            else:
                logistic, scores = fit_logistic(keys, values, samples, types, fit_ids, spec)
                replay = apply_logistic(keys, values, logistic)
                np.testing.assert_allclose(replay, scores, atol=1e-12, rtol=1e-12)
                if not np.isfinite(scores).all() or logistic['gradient_max'] > 1e-5:
                    raise ArithmeticError('The logistic solver did not converge to finite scores.')
                candidates = candidates_from_scores(keys, scores, len(samples))
                decoder = select_fit(samples, {0.0: candidates}, types, Config(), fit_ids, 'expanded')
                fitted_record = {'logistic': logistic, 'decoder': decoder, 'family': family}
                write_json(cache_path, {'identity': identity, 'fit': fitted_record})
            fits[family][name] = fitted_record
            held_ids = ids if name == 'full' else [i for i in ids if folds[i] == int(name[-1])]
            predictions[family]['full_dev' if name == 'full' else 'crossfit'].update(
                predict(candidates, fitted_record['decoder'], types, held_ids))
        reports[family] = {mode: score_predictions(samples, pred, types)
                           for mode, pred in predictions[family].items()}
        print(f'{family}: full Micro={100*reports[family]["full_dev"]["micro"]["f1"]:.6f}%, '
              f'crossfit Micro={100*reports[family]["crossfit"]["micro"]["f1"]:.6f}%', flush=True)
        folder = output / family
        folder.mkdir(exist_ok=True)
        for mode, pred in predictions[family].items():
            write_predictions(folder / f'{mode}_predictions.jsonl', samples, pred, types)

    selected = {name: max((fits[f][name] for f in spec['families']),
                         key=lambda f: (f['decoder']['fit_micro_f1'], f['decoder']['fit_macro_f1'],
                                        f['family'] == 'single_v4')) for name in partitions}
    combined = {'full_dev': predictions[selected['full']['family']]['full_dev'], 'crossfit': {}}
    for held in (0, 1):
        family = selected[f'held{held}']['family']
        combined['crossfit'].update({sid: pred for sid, pred in predictions[family]['crossfit'].items()
                                    if folds[sid] == held})
    nested = {mode: score_predictions(samples, pred, types) for mode, pred in combined.items()}
    base = read_json(PACKAGE / 'inputs/v4_baseline.json')
    baseline = named_predictions(PACKAGE / 'inputs/v4_crossfit_predictions.jsonl', types)
    assert score_predictions(samples, baseline, types)['micro'] == base['crossfit']['micro']
    fold_reports = []
    for held in (0, 1):
        held_ids = [i for i in ids if folds[i] == held]
        current = subset_score(samples, combined['crossfit'], types, held_ids)
        old = subset_score(samples, baseline, types, held_ids)
        fold_reports.append({'held_out_fold': held, 'v4': old, 'v5': current,
                             'micro_delta_pp': 100*(current['micro']['f1'] - old['micro']['f1'])})
    accepted = (nested['full_dev']['micro']['f1'] >= base['full_dev']['micro']['f1'] and
                nested['crossfit']['micro']['f1'] > base['crossfit']['micro']['f1'] and
                min(r['micro_delta_pp'] for r in fold_reports) >= 0)
    report = {'identity': identity, 'policy': spec, 'types': types, 'reports': reports,
              'selected_fits': selected, 'nested': nested, 'baseline': base,
              'fold_reports': fold_reports, 'accepted': accepted, 'test_accessed': False}
    write_json(output / 'comparison.json', report)
    write_json(output / 'crossfit_folds.json', folds)
    selection = {'method': 'calibrated_v4' if accepted else 'v4',
                 'fit': selected['full'] if accepted else None,
                 'baseline_decoder': base['decoder'], 'types': types,
                 'dev_sha256': manifest['dev_sha256'], 'checkpoint_sha256': manifest['checkpoint_sha256'],
                 'identity': identity, 'test_accessed': False,
                 'scope': 'Exploratory dev-only calibration. Checkpoint and route use dev; '
                          'crossfit only holds out the calibration/decoder fitting labels.'}
    write_json(output / 'selection.json', selection)
    for mode, pred in combined.items():
        write_predictions(output / f'{mode}_predictions.jsonl', samples, pred, types)
    if accepted:
        chosen_full = combined['full_dev']
    else:
        from search import scored_candidates
        chosen_full = predict(scored_candidates(upgraded, base['decoder']['boundary_weight'], 8),
                              base['decoder'], types, ids)
    write_predictions(output / 'selected_full_dev_predictions.jsonl', samples, chosen_full, types)
    write_json(output / 'replay.json', {'seconds': time.perf_counter()-started,
                                      'cache_inputs': {p.name: sha256(p) for p in (PACKAGE/'inputs').iterdir()},
                                      'python': platform.python_version(), 'numpy': np.__version__,
                                      'gpu_forward_run': False, 'test_accessed': False})
    snapshot = output / 'source_snapshot'
    for path in PACKAGE.rglob('*'):
        if not path.is_file() or 'inputs' in path.relative_to(PACKAGE).parts or '__pycache__' in path.parts:
            continue
        if path.suffix not in ('.py', '.sh', '.json', '.md', '.txt'):
            continue
        target = snapshot / path.relative_to(PACKAGE)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    lines = ['# v5 calibration comparison', '',
             '| Method | Full-dev Micro % | Crossfit Micro % | Crossfit Macro13 % |',
             '|---|---:|---:|---:|',
             f'| v4 | {100*base["full_dev"]["micro"]["f1"]:.6f} | '
             f'{100*base["crossfit"]["micro"]["f1"]:.6f} | {100*base["crossfit"]["macro_f1"]:.6f} |']
    for family, r in reports.items():
        lines.append(f'| {family} | {100*r["full_dev"]["micro"]["f1"]:.6f} | '
                     f'{100*r["crossfit"]["micro"]["f1"]:.6f} | {100*r["crossfit"]["macro_f1"]:.6f} |')
    lines.extend(['', f'Accepted: {accepted}. Deployment family: {selected["full"]["family"] if accepted else "v4"}.',
                  '', 'Checkpoint weights remain unchanged. No training or test evaluation was run.',
                  'Crossfit is a development calibration comparison, not an independent test estimate.'])
    (output / 'SUMMARY.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
