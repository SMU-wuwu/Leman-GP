"""Compare v5 with train-only gazetteer calibration, without neural retraining."""
import os
from pathlib import Path
import platform
import shutil
import time

import numpy as np

from common import (PACKAGE, named_predictions, paths, read_cache, read_json,
                    sha256, subset_score, verify_package)
from calibration_core import apply_logistic, candidates_from_scores, fit_logistic
from config import Config
from data import read_samples, write_json
from evaluate import score_predictions, write_predictions
from gazetteer import build_gazetteer, make_features
from search import folds_for, predict, select_fit


def main():
    started = time.perf_counter()
    manifest = verify_package()
    root, output = paths()
    output.mkdir(parents=True, exist_ok=True)
    data = Path(os.environ.get('LEMAN_GP_DATA', root / 'data'))
    spec = read_json(PACKAGE / 'POLICY.json')
    for split in ('train', 'dev'):
        if sha256(data / f'{split}2.txt') != spec[f'{split}_sha256']:
            raise ValueError(f'Use the frozen {split}2.txt.')
    header, records = read_cache(PACKAGE / 'inputs/upgraded_features.jsonl.gz')
    types = header['types']
    assert header['dev_sha256'] == spec['dev_sha256']
    assert header['checkpoint']['checkpoint_sha256'] == manifest['checkpoint_sha256']['upgraded']
    samples = read_samples(data / 'dev2.txt', types)
    texts = [text for text, _ in samples]
    train = read_samples(data / 'train2.txt', types)
    gazetteer = build_gazetteer(train, types, spec['train_sha256'])
    write_json(output / 'train_gazetteer.json', gazetteer)
    ids, folds = list(range(len(samples))), folds_for(samples)
    assert set(records) == set(ids)
    partitions = {'full': ids, **{f'held{h}': [i for i in ids if folds[i] != h] for h in (0, 1)}}
    identity = {'package_manifest_sha256': sha256(PACKAGE / 'PACKAGE_MANIFEST.json'),
                'dev_sha256': spec['dev_sha256'], 'train_sha256': spec['train_sha256'],
                'gazetteer_sha256': sha256(output / 'train_gazetteer.json'), 'test_accessed': False}
    fit_dir = output / 'fits'
    fit_dir.mkdir(exist_ok=True)
    fits, predictions, reports = {}, {}, {}
    for family in spec['families']:
        print(f'Candidate: {family}', flush=True)
        keys, values = make_features(records, texts, types, family, gazetteer)
        fits[family], predictions[family] = {}, {'full_dev': {}, 'crossfit': {}}
        for name, fit_ids in partitions.items():
            cache = fit_dir / f'{family}_{name}.json'
            if cache.exists():
                saved = read_json(cache)
                if saved['identity'] != identity:
                    raise ValueError(f'Changed inputs require a new output directory: {cache}')
                fitted = saved['fit']
                scores = apply_logistic(keys, values, fitted['logistic'])
                candidates = candidates_from_scores(keys, scores, len(samples))
                print(f'Reuse completed fit: {family}/{name}', flush=True)
            elif family == 'v5':
                original = read_json(PACKAGE / 'inputs' / f'v5_fit_{name}.json')['fit']
                fitted = {**original, 'family': 'v5'}
                scores = apply_logistic(keys, values, fitted['logistic'])
                candidates = candidates_from_scores(keys, scores, len(samples))
                write_json(cache, {'identity': identity, 'fit': fitted})
            else:
                logistic, scores = fit_logistic(keys, values, samples, types, fit_ids, spec)
                np.testing.assert_allclose(scores, apply_logistic(keys, values, logistic), atol=1e-12, rtol=1e-12)
                if not np.isfinite(scores).all() or logistic['gradient_max'] > 1e-5:
                    raise ArithmeticError('Logistic fit did not converge.')
                candidates = candidates_from_scores(keys, scores, len(samples))
                decoder = select_fit(samples, {0.0: candidates}, types, Config(), fit_ids, 'expanded')
                fitted = {'logistic': logistic, 'decoder': decoder, 'family': family}
                write_json(cache, {'identity': identity, 'fit': fitted})
            fits[family][name] = fitted
            held_ids = ids if name == 'full' else [i for i in ids if folds[i] == int(name[-1])]
            mode = 'full_dev' if name == 'full' else 'crossfit'
            predictions[family][mode].update(predict(candidates, fitted['decoder'], types, held_ids))
        reports[family] = {mode: score_predictions(samples, pred, types)
                           for mode, pred in predictions[family].items()}
        folder = output / family
        folder.mkdir(exist_ok=True)
        for mode, pred in predictions[family].items():
            write_predictions(folder / f'{mode}_predictions.jsonl', samples, pred, types)
        print(f'{family}: full Micro={100*reports[family]["full_dev"]["micro"]["f1"]:.6f}%, '
              f'crossfit Micro={100*reports[family]["crossfit"]["micro"]["f1"]:.6f}%', flush=True)

    baseline_report = read_json(PACKAGE / 'inputs/v5_comparison.json')
    for mode in ('full_dev', 'crossfit'):
        expected = baseline_report['nested'][mode]
        assert reports['v5'][mode]['micro'] == expected['micro']
        filename = 'v5_crossfit_predictions.jsonl' if mode == 'crossfit' else 'v5_selected_full_dev_predictions.jsonl'
        assert predictions['v5'][mode] == named_predictions(PACKAGE / 'inputs' / filename, types)
    selected = {name: max((fits[f][name] for f in spec['families']), key=lambda fit: (
        fit['decoder']['fit_micro_f1'], fit['decoder']['fit_macro_f1'], fit['family'] == 'v5'))
        for name in partitions}
    combined = {'full_dev': predictions[selected['full']['family']]['full_dev'], 'crossfit': {}}
    for held in (0, 1):
        family = selected[f'held{held}']['family']
        combined['crossfit'].update({sid: pred for sid, pred in predictions[family]['crossfit'].items()
                                    if folds[sid] == held})
    nested = {mode: score_predictions(samples, pred, types) for mode, pred in combined.items()}
    fold_reports = []
    for held in (0, 1):
        held_ids = [i for i in ids if folds[i] == held]
        current = subset_score(samples, combined['crossfit'], types, held_ids)
        baseline = subset_score(samples, predictions['v5']['crossfit'], types, held_ids)
        fold_reports.append({'held_out_fold': held, 'v5': baseline, 'v6': current,
                             'micro_delta_pp': 100*(current['micro']['f1'] - baseline['micro']['f1'])})
    accepted = (nested['full_dev']['micro']['f1'] >= reports['v5']['full_dev']['micro']['f1'] and
                nested['crossfit']['micro']['f1'] > reports['v5']['crossfit']['micro']['f1'] and
                min(row['micro_delta_pp'] for row in fold_reports) >= 0)
    deployment = selected['full'] if accepted else fits['v5']['full']
    report = {'identity': identity, 'policy': spec, 'types': types, 'reports': reports,
              'selected_fits': selected, 'nested': nested, 'baseline': reports['v5'],
              'fold_reports': fold_reports, 'accepted': accepted, 'test_accessed': False}
    selection = {'method': 'calibrated_v4', 'fit': deployment, 'types': types,
                 'dev_sha256': spec['dev_sha256'], 'train_sha256': spec['train_sha256'],
                 'checkpoint_sha256': manifest['checkpoint_sha256'], 'identity': identity,
                 'gazetteer_sha256': identity['gazetteer_sha256'], 'test_accessed': False,
                 'scope': 'Exploratory dev-only calibration; same previously selected v4 checkpoint.'}
    write_json(output / 'comparison.json', report)
    write_json(output / 'selection.json', selection)
    write_json(output / 'crossfit_folds.json', folds)
    for mode, pred in combined.items():
        write_predictions(output / f'{mode}_predictions.jsonl', samples, pred, types)
    chosen = combined['full_dev'] if accepted else predictions['v5']['full_dev']
    write_predictions(output / 'selected_full_dev_predictions.jsonl', samples, chosen, types)
    write_json(output / 'replay.json', {'seconds': time.perf_counter()-started,
        'python': platform.python_version(), 'numpy': np.__version__,
        'gpu_forward_run': False, 'test_accessed': False,
        'gazetteer_surfaces': len(gazetteer['surfaces']), 'train_mentions': gazetteer['mention_count']})
    for path in PACKAGE.rglob('*'):
        if not path.is_file() or any(x in path.relative_to(PACKAGE).parts for x in ('inputs', '__pycache__')):
            continue
        if path.suffix not in ('.py', '.sh', '.json', '.md', '.txt'):
            continue
        target = output / 'source_snapshot' / path.relative_to(PACKAGE)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    lines = ['# v6 train-only gazetteer comparison', '',
             '| Method | Full-dev Micro % | Crossfit Micro % | Crossfit Macro13 % |',
             '|---|---:|---:|---:|']
    for family, row in reports.items():
        lines.append(f'| {family} | {100*row["full_dev"]["micro"]["f1"]:.6f} | '
                     f'{100*row["crossfit"]["micro"]["f1"]:.6f} | {100*row["crossfit"]["macro_f1"]:.6f} |')
    lines.extend(['', f'Accepted: {accepted}. Deployment family: {deployment["family"]}.',
                  'No neural training or test evaluation was run.'])
    (output / 'SUMMARY.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
