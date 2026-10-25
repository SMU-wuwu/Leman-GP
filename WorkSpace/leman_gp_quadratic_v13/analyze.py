"""Fixed quadratic calibration on audited expanded candidates; no training or test."""
import gzip
import json
from pathlib import Path
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'leman_gp_fgm_v10_analysis'))
from analyze import build_features as shared_build_features
sys.path.insert(0, str(HERE.parent/'leman_gp_candidates_v11'))
from importlib.util import spec_from_file_location, module_from_spec
_spec = spec_from_file_location('expanded_shared', HERE.parent/'leman_gp_candidates_v11/analyze.py')
_expanded = module_from_spec(_spec)
_spec.loader.exec_module(_expanded)
expanded_features = _expanded.expanded_features

import numpy as np
from common import (PACKAGE, named_predictions, read_cache, read_json, sha256,
                    subset_score, verify_package)
from calibration_core import apply_logistic, candidates_from_scores, fit_logistic
from config import Config
from data import read_samples, write_json
from evaluate import score_predictions, write_predictions
from gazetteer import make_features
from search import folds_for, predict, select_fit


ROUTES = ['v10', 'expanded_quadratic']


def quadratic_features(records, samples, types, gazetteer):
    keys, values = expanded_features(records, samples, types, gazetteer)
    extras = []
    for index in (0,6):
        gp, boundary = values[:,index], values[:,index+1]
        extras.extend((gp*gp, gp*boundary, boundary*boundary))
    return keys, np.column_stack((values,*extras))


def coverage(keys, samples):
    candidates = {(sid, t, a, b) for sid, t, a, b in keys}
    gold = {(sid, *span) for sid, (_, spans) in enumerate(samples) for span in spans}
    return {'gold': len(gold), 'available_gold': len(gold & candidates),
            'gold_coverage': len(gold & candidates)/len(gold), 'unique_candidates': len(candidates)}


def main():
    started = time.perf_counter()
    manifest = verify_package()
    own_manifest = read_json(HERE/'PACKAGE_MANIFEST.json')
    for name, digest in own_manifest['files'].items():
        assert sha256(HERE/name) == digest
    for name, digest in own_manifest['shared_dependencies'].items():
        assert sha256(HERE.parent/name/'PACKAGE_MANIFEST.json') == digest
    assert sha256(PACKAGE/'PACKAGE_MANIFEST.json') == '9feb7005d4de355bc4419536be8416dc4c2c88721fe0527e5975cc7769aefd97'
    cfg = Config.from_environment()
    root = Path(cfg.root)
    output = root/'output/leman_quadratic_v13_seed43'
    output.mkdir(parents=True,exist_ok=True)
    previous = root/'output/leman_fgm_v10_analysis_seed43'
    plan_path = HERE/'POLICY.json'
    plan = read_json(plan_path)
    assert sha256(previous/'selection.json') == plan['baseline_selection_sha256']
    prior = read_json(previous/'comparison.json')
    v11_path = root/'output/leman_candidates_v11_seed43/comparison.json'
    assert sha256(v11_path) == plan['v11_comparison_sha256']
    v11 = read_json(v11_path)
    expanded_path = root/'output/leman_candidates_v11_seed43/fgm_features.jsonl.gz'
    expanded_v4_path = root/'output/leman_inference_v7_seed42/upgraded_features.jsonl.gz'
    assert sha256(expanded_path) == plan['expanded_fgm_cache_sha256']
    assert sha256(expanded_v4_path) == plan['expanded_v4_cache_sha256']
    header, expanded_fgm = read_cache(expanded_path)
    old_header, expanded_old = read_cache(expanded_v4_path)
    _, fgm = read_cache(previous/'fgm_features.jsonl.gz')
    _, old = read_cache(PACKAGE/'inputs/v6_upgraded_features.jsonl.gz')
    assert header['floor'] == old_header['floor'] == -32
    assert not header['test_accessed'] and not old_header['test_accessed']
    types = header['types']
    spec = read_json(PACKAGE/'POLICY.json')
    samples = read_samples(root/'data/dev2.txt', types)
    for split in ('train', 'dev'):
        assert sha256(root/'data'/f'{split}2.txt') == spec[f'{split}_sha256']
    gazetteer = read_json(PACKAGE/'inputs/v6_train_gazetteer.json')
    ids = list(range(len(samples)))
    folds = folds_for(samples)
    partitions = {'full': ids, **{f'held{h}': [sid for sid in ids if folds[sid] != h] for h in (0, 1)}}
    identity = {'manifest_sha256': sha256(HERE/'PACKAGE_MANIFEST.json'),
                'plan_sha256': sha256(plan_path), 'v9_manifest_sha256': sha256(PACKAGE/'PACKAGE_MANIFEST.json'),
                'cache_sha256': {'expanded_fgm': sha256(expanded_path), 'expanded_old':sha256(expanded_v4_path), 'fgm': sha256(previous/'fgm_features.jsonl.gz'),
                                 'old': sha256(PACKAGE/'inputs/v6_upgraded_features.jsonl.gz')},
                'test_accessed': False}
    fits, predictions, reports, coverages = {}, {}, {}, {}
    (output/'fits').mkdir(exist_ok=True)
    for family in ROUTES:
        models = [old,fgm] if family == 'v10' else [expanded_old,expanded_fgm]
        builder = shared_build_features if family == 'v10' else quadratic_features
        keys, values = builder(models, samples, types, gazetteer)
        coverages[family] = coverage(keys, samples)
        route_spec = {**spec, 'interaction_features': [0,1,6,7]}
        fits[family], predictions[family] = {}, {'full_dev': {}, 'crossfit': {}}
        print(f'{family}: {len(keys)} candidates; {values.shape[1]} features', flush=True)
        for part, fit_ids in partitions.items():
            target = output/'fits'/f'{family}_{part}.json'
            if target.exists():
                saved = read_json(target)
                assert saved['identity'] == identity
                fitted = saved['fit']
            elif family == 'v10':
                fitted = {**prior['selected_fits'][part], 'baseline_family': prior['selected_fits'][part]['family'],
                          'family': 'v10'}
            else:
                print(f'Fit {family}/{part}', flush=True)
                logistic, scores = fit_logistic(keys, values, samples, types, fit_ids, route_spec)
                np.testing.assert_allclose(scores, apply_logistic(keys, values, logistic), rtol=1e-12, atol=1e-12)
                assert logistic['gradient_max'] < 1e-5
                candidates = candidates_from_scores(keys, scores, len(samples))
                decoder = select_fit(samples, {0.0: candidates}, types, cfg, fit_ids, 'expanded')
                fitted = {'family': family, 'logistic': logistic, 'decoder': decoder}
            scores = apply_logistic(keys, values, fitted['logistic'])
            candidates = candidates_from_scores(keys, scores, len(samples))
            write_json(target, {'identity': identity, 'fit': fitted})
            fits[family][part] = fitted
            held_ids = ids if part == 'full' else [sid for sid in ids if folds[sid] == int(part[-1])]
            mode = 'full_dev' if part == 'full' else 'crossfit'
            predictions[family][mode].update(predict(candidates, fitted['decoder'], types, held_ids))
        reports[family] = {mode: score_predictions(samples, pred, types)
                           for mode, pred in predictions[family].items()}
        folder = output/family
        folder.mkdir(exist_ok=True)
        for mode, pred in predictions[family].items():
            write_predictions(folder/f'{mode}_predictions.jsonl', samples, pred, types)
        print(f'{family}: full={100*reports[family]["full_dev"]["micro"]["f1"]:.6f}%, '
              f'crossfit={100*reports[family]["crossfit"]["micro"]["f1"]:.6f}%', flush=True)
    for mode, filename in [('full_dev', 'selected_full_dev_predictions.jsonl'),
                           ('crossfit', 'crossfit_predictions.jsonl')]:
        assert predictions['v10'][mode] == named_predictions(previous/filename, types)
    assert reports['v10']['crossfit']['micro']['f1'] == prior['nested']['crossfit']['micro']['f1']
    selected = {part: max((fits[family][part] for family in ROUTES), key=lambda fitted: (
        fitted['decoder']['fit_micro_f1'], fitted['decoder']['fit_macro_f1'], -ROUTES.index(fitted['family'])))
                for part in partitions}
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
        baseline = subset_score(samples, predictions['v10']['crossfit'], types, held_ids)
        fold_reports.append({'held_out_fold': held, 'v10': baseline, 'v13': current,
                             'v11_reference': v11['fold_reports'][held]['v11'],
                             'v11_delta_pp': 100*(current['micro']['f1']-v11['fold_reports'][held]['v11']['micro']['f1']),
                             'micro_delta_pp': 100*(current['micro']['f1']-baseline['micro']['f1'])})
    accepted = (nested['full_dev']['micro']['f1'] >= reports['v10']['full_dev']['micro']['f1'] and
                nested['crossfit']['micro']['f1'] > reports['v10']['crossfit']['micro']['f1'] and
                min(row['micro_delta_pp'] for row in fold_reports) >= 0 and
                nested['crossfit']['micro']['f1'] > v11['nested']['crossfit']['micro']['f1'] and
                min(row['v11_delta_pp'] for row in fold_reports) >= 0)
    deployment = selected['full'] if accepted else fits['v10']['full']
    write_json(output/'comparison.json', {'identity': identity, 'types': types, 'plan': plan,
        'policy': spec, 'routes': ROUTES, 'reports': reports, 'selected_fits': selected,
        'nested': nested, 'baseline': reports['v10'], 'fold_reports': fold_reports, 'accepted': accepted,
        'coverage': coverages, 'v11_reference': v11['nested'], 'test_accessed': False, 'seconds': time.perf_counter()-started})
    write_json(output/'selection.json', {'method': 'FGM_type_slope_calibrated_spans', 'fit': deployment,
        'types': types, 'identity': identity, 'checkpoint': header['checkpoint'],
        'candidate_floor': -32.0 if deployment['family'] == 'expanded_quadratic' else -4.0,
        'fgm_best_checkpoint_sha256': 'efdeaaa83bc483b376837ee5bcb35df7bcb5a9b1b10d2f120eb32868f1d80506',
        'parent_checkpoint_sha256': manifest['checkpoint_sha256']['upgraded'],
        'gazetteer_sha256': sha256(PACKAGE/'inputs/v6_train_gazetteer.json'), 'test_accessed': False})
    write_json(output/'crossfit_folds.json', folds)
    for mode, pred in combined.items():
        write_predictions(output/f'{mode}_predictions.jsonl', samples, pred, types)
    chosen = combined['full_dev'] if accepted else predictions['v10']['full_dev']
    write_predictions(output/'selected_full_dev_predictions.jsonl', samples, chosen, types)
    for name in ('analyze.py', 'PACKAGE_MANIFEST.json', 'POLICY.json'):
        (output/'source_snapshot').mkdir(exist_ok=True)
        shutil.copy2(HERE/name, output/'source_snapshot'/name)
    print(f'Accepted={accepted}; deployment={deployment["family"]}; '
          f'nested crossfit={100*nested["crossfit"]["micro"]["f1"]:.6f}%', flush=True)


if __name__ == '__main__':
    main()
