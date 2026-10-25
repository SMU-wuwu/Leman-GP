"""Select among a frozen baseline and two recall-oriented inference routes."""
from pathlib import Path
import shutil
import time

import numpy as np

from common import (PACKAGE, named_predictions, paths, read_cache, read_json,
                    sha256, subset_score, verify_package)
from calibration_core import apply_logistic, candidates_from_scores, fit_logistic
from config import Config
from data import read_samples, write_json
from evaluate import score_predictions, write_predictions
from expanded_features import features
from gazetteer import make_features
from search import folds_for, predict, select_fit


def coverage(records, samples):
    gold = available = rows = 0
    for sid, (_, spans) in enumerate(samples):
        keys = {tuple(row[:3]) for row in records[sid]}
        gold += len(spans)
        available += len(keys & set(map(tuple, spans)))
        rows += len(keys)
    return {'gold': gold, 'available_gold': available,
            'gold_coverage': available/gold, 'unique_candidates': rows}


def main():
    started = time.perf_counter()
    manifest = verify_package()
    root, output = paths()
    spec = read_json(PACKAGE / 'POLICY.json')
    for split in ('train', 'dev'):
        assert sha256(root/'data'/f'{split}2.txt') == spec[f'{split}_sha256']
    headers, records = {}, {}
    for name in ('upgraded', 'parent'):
        headers[name], records[name] = read_cache(output/f'{name}_features.jsonl.gz')
        assert headers[name]['dev_sha256'] == spec['dev_sha256']
        assert headers[name]['checkpoint']['checkpoint_sha256'] == manifest['checkpoint_sha256'][name]
        assert headers[name]['policy_sha256'] == sha256(PACKAGE/'POLICY.json')
    types = headers['upgraded']['types']
    assert headers['parent']['types'] == types
    samples = read_samples(root/'data/dev2.txt', types)
    ids = list(range(len(samples)))
    _, old_records = read_cache(PACKAGE/'inputs/v6_upgraded_features.jsonl.gz')
    gazetteer = read_json(PACKAGE/'inputs/v6_train_gazetteer.json')
    folds = folds_for(samples)
    partitions = {'full': ids, **{f'held{h}': [sid for sid in ids if folds[sid] != h] for h in (0, 1)}}
    identity = {'manifest_sha256': sha256(PACKAGE/'PACKAGE_MANIFEST.json'),
                'policy_sha256': sha256(PACKAGE/'POLICY.json'),
                'cache_sha256': {name: sha256(output/f'{name}_features.jsonl.gz') for name in records},
                'test_accessed': False}
    (output/'fits').mkdir(exist_ok=True)
    fits, predictions, reports = {}, {}, {}
    for family in spec['routes']:
        if family == 'v6':
            keys, values = make_features(old_records, [text for text, _ in samples],
                                        types, 'train_gazetteer', gazetteer)
        else:
            keys, values = features(records['parent'], records['upgraded'], samples,
                                    types, family, gazetteer, spec['raw_candidate_floor'])
        print(f'{family}: {len(keys)} unique candidates, {values.shape[1]} features', flush=True)
        fits[family], predictions[family] = {}, {'full_dev': {}, 'crossfit': {}}
        for name, fit_ids in partitions.items():
            cache = output/'fits'/f'{family}_{name}.json'
            if cache.exists():
                saved = read_json(cache)
                assert saved['identity'] == identity
                fitted = saved['fit']
                scores = apply_logistic(keys, values, fitted['logistic'])
                candidates = candidates_from_scores(keys, scores, len(samples))
            elif family == 'v6':
                fitted = {**read_json(PACKAGE/'inputs'/f'v6_fit_{name}.json')['fit'], 'family': family}
                scores = apply_logistic(keys, values, fitted['logistic'])
                candidates = candidates_from_scores(keys, scores, len(samples))
            else:
                print(f'Fit {family}/{name}', flush=True)
                logistic, scores = fit_logistic(keys, values, samples, types, fit_ids, spec)
                np.testing.assert_allclose(scores, apply_logistic(keys, values, logistic), rtol=1e-12, atol=1e-12)
                assert np.isfinite(scores).all() and logistic['gradient_max'] < 1e-5
                candidates = candidates_from_scores(keys, scores, len(samples))
                decoder = select_fit(samples, {0.0: candidates}, types, Config(), fit_ids, 'expanded')
                fitted = {'family': family, 'logistic': logistic, 'decoder': decoder}
            write_json(cache, {'identity': identity, 'fit': fitted})
            fits[family][name] = fitted
            held_ids = ids if name == 'full' else [sid for sid in ids if folds[sid] == int(name[-1])]
            mode = 'full_dev' if name == 'full' else 'crossfit'
            predictions[family][mode].update(predict(candidates, fitted['decoder'], types, held_ids))
        reports[family] = {mode: score_predictions(samples, pred, types)
                           for mode, pred in predictions[family].items()}
        folder = output/family
        folder.mkdir(exist_ok=True)
        for mode, pred in predictions[family].items():
            write_predictions(folder/f'{mode}_predictions.jsonl', samples, pred, types)
        print(f'{family}: full={100*reports[family]["full_dev"]["micro"]["f1"]:.6f}%, '
              f'crossfit={100*reports[family]["crossfit"]["micro"]["f1"]:.6f}%', flush=True)
    for mode, filename in [('full_dev', 'v6_selected_full_dev_predictions.jsonl'),
                           ('crossfit', 'v6_crossfit_predictions.jsonl')]:
        assert predictions['v6'][mode] == named_predictions(PACKAGE/'inputs'/filename, types)
    order = {route: i for i, route in enumerate(spec['routes'])}
    selected = {name: max((fits[family][name] for family in spec['routes']), key=lambda fitted: (
        fitted['decoder']['fit_micro_f1'], fitted['decoder']['fit_macro_f1'], -order[fitted['family']]))
                for name in partitions}
    combined = {'full_dev': predictions[selected['full']['family']]['full_dev'], 'crossfit': {}}
    for held in (0, 1):
        family = selected[f'held{held}']['family']
        combined['crossfit'].update({sid: pred for sid, pred in predictions[family]['crossfit'].items()
                                    if folds[sid] == held})
    nested = {mode: score_predictions(samples, pred, types) for mode, pred in combined.items()}
    fold_reports = []
    for held in (0, 1):
        held_ids = [sid for sid in ids if folds[sid] == held]
        current = subset_score(samples, combined['crossfit'], types, held_ids)
        baseline = subset_score(samples, predictions['v6']['crossfit'], types, held_ids)
        fold_reports.append({'held_out_fold': held, 'v6': baseline, 'v7': current,
                             'micro_delta_pp': 100*(current['micro']['f1']-baseline['micro']['f1'])})
    accepted = (nested['full_dev']['micro']['f1'] >= reports['v6']['full_dev']['micro']['f1'] and
                nested['crossfit']['micro']['f1'] > reports['v6']['crossfit']['micro']['f1'] and
                min(row['micro_delta_pp'] for row in fold_reports) >= 0)
    deployment = selected['full'] if accepted else fits['v6']['full']
    union = {sid: records['parent'][sid]+records['upgraded'][sid] for sid in ids}
    coverage_report = {'v6': coverage(old_records, samples),
                       'expanded_v4': coverage(records['upgraded'], samples),
                       'expanded_fusion': coverage(union, samples)}
    report = {'identity': identity, 'types': types, 'policy': spec, 'reports': reports,
              'selected_fits': selected, 'nested': nested, 'baseline': reports['v6'],
              'fold_reports': fold_reports, 'accepted': accepted, 'coverage': coverage_report,
              'test_accessed': False, 'seconds': time.perf_counter()-started}
    selection = {'method': 'expanded_calibrated_spans', 'fit': deployment, 'types': types,
                 'checkpoint_sha256': manifest['checkpoint_sha256'], 'identity': identity,
                 'gazetteer_sha256': sha256(PACKAGE/'inputs/v6_train_gazetteer.json'),
                 'dev_sha256': spec['dev_sha256'], 'test_accessed': False}
    write_json(output/'comparison.json', report)
    write_json(output/'selection.json', selection)
    write_json(output/'crossfit_folds.json', folds)
    for mode, pred in combined.items():
        write_predictions(output/f'{mode}_predictions.jsonl', samples, pred, types)
    chosen = combined['full_dev'] if accepted else predictions['v6']['full_dev']
    write_predictions(output/'selected_full_dev_predictions.jsonl', samples, chosen, types)
    for path in PACKAGE.rglob('*'):
        if path.is_file() and not any(x in path.relative_to(PACKAGE).parts for x in ('inputs', '__pycache__')):
            target = output/'source_snapshot'/path.relative_to(PACKAGE)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    print(f'Accepted={accepted}; deployment={deployment["family"]}; '
          f'nested crossfit={100*nested["crossfit"]["micro"]["f1"]:.6f}%', flush=True)


if __name__ == '__main__':
    main()
