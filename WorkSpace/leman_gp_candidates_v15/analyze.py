"""Compare expanded seed44 candidates with the audited v14 pipeline; no training."""
from pathlib import Path
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'leman_gp_fgm_v10_analysis'))
from analyze import build_features, coverage
sys.path.append(str(HERE.parent))
from leman_gp_candidates_v11.analyze import expanded_features
from common import PACKAGE, named_predictions, read_cache, read_json, sha256, subset_score, verify_package
from calibration_core import apply_logistic, candidates_from_scores, fit_logistic
from config import Config
from data import read_samples, write_json
from evaluate import score_predictions, write_predictions
from search import folds_for, predict, select_fit
import numpy as np

ROUTES = ['v14', 'expanded_fgm44', 'expanded_fusion44']


def main():
    started = time.perf_counter()
    verify_package()
    own = read_json(HERE/'PACKAGE_MANIFEST.json')
    for name, digest in own['files'].items():
        assert sha256(HERE/name) == digest
    for name, digest in own['shared_dependencies'].items():
        assert sha256(HERE.parent/name/'PACKAGE_MANIFEST.json') == digest
    root = Path(Config.from_environment().root)
    output = root/'output/leman_candidates_v15_seed44'
    previous = root/'output/leman_fgm_v14_analysis_seed44'
    v10_folder = root/'output/leman_fgm_v10_analysis_seed43'
    v11_folder = root/'output/leman_candidates_v11_seed43'
    plan_path = HERE/'POLICY.json'
    plan = read_json(plan_path)
    assert sha256(previous/'selection.json') == plan['baseline_selection_sha256']
    assert sha256(previous/'comparison.json') == plan['baseline_comparison_sha256']
    references = {'v14': read_json(previous/'comparison.json')}
    paths = {'old': PACKAGE/'inputs/v6_upgraded_features.jsonl.gz',
             'fgm43': v10_folder/'fgm_features.jsonl.gz',
             'fgm44': previous/'fgm_features.jsonl.gz',
             'old_expanded': root/'output/leman_inference_v7_seed42/upgraded_features.jsonl.gz',
             'fgm43_expanded': v11_folder/'fgm_features.jsonl.gz',
             'fgm44_expanded': output/'fgm_features.jsonl.gz'}
    caches = {name: read_cache(path) for name, path in paths.items()}
    header = caches['fgm44_expanded'][0]
    assert header['floor'] == -32 and header['policy_sha256'] == sha256(plan_path)
    assert not header['test_accessed']
    assert header['checkpoint']['checkpoint_sha256'] == plan['checkpoint_sha256']['fgm44']
    assert header['checkpoint']['checkpoint_epoch'] == 1 and header['checkpoint']['weight_source'] == 'live'
    # Expanded candidate extraction must preserve every original candidate and score.
    assert {sid:[r for r in rows if r[3] > -4] for sid,rows in caches['fgm44_expanded'][1].items()} == caches['fgm44'][1]
    spec = read_json(PACKAGE/'POLICY.json')
    types = header['types']
    samples = read_samples(root/'data/dev2.txt', types)
    for split in ('train', 'dev'):
        assert sha256(root/'data'/f'{split}2.txt') == spec[f'{split}_sha256']
    gazetteer = read_json(PACKAGE/'inputs/v6_train_gazetteer.json')
    ids, folds = list(range(len(samples))), folds_for(samples)
    partitions = {'full': ids, **{f'held{h}':[sid for sid in ids if folds[sid] != h] for h in (0,1)}}
    identity = {'manifest_sha256':sha256(HERE/'PACKAGE_MANIFEST.json'),
                'plan_sha256':sha256(plan_path),
                'cache_sha256':{name:sha256(path) for name,path in paths.items()},
                'test_accessed':False}
    fits, predictions, reports, coverages = {}, {}, {}, {}
    feature_cache = {}
    (output/'fits').mkdir(exist_ok=True)
    for family in ROUTES:
        fits[family], predictions[family], coverages[family] = {}, {'full_dev':{},'crossfit':{}}, {}
        for part, fit_ids in partitions.items():
            inherited = references[family]['selected_fits'][part] if family in references else None
            if family == 'v14':
                floor, names = inherited['candidate_floor'], inherited['component_models']
            else:
                floor = -32.0
                names = ['fgm44_expanded'] if family == 'expanded_fgm44' else ['old_expanded','fgm43_expanded','fgm44_expanded']
            signature = (tuple(names),floor)
            if signature not in feature_cache:
                builder = build_features if floor == -4 else expanded_features
                feature_cache[signature] = builder([caches[n][1] for n in names],samples,types,gazetteer)
            keys, values = feature_cache[signature]
            coverages[family][part] = coverage(keys,samples)
            target = output/'fits'/f'{family}_{part}.json'
            metadata = {'family':family,'component_models':names,'candidate_floor':floor}
            if target.exists():
                saved = read_json(target)
                assert saved['identity'] == identity
                fitted = saved['fit']
            elif inherited is not None:
                fitted = {**metadata,'logistic':inherited['logistic'],'decoder':inherited['decoder'],
                          'inherited_fit':inherited}
            else:
                print(f'Fit {family}/{part}: {len(keys)} candidates, {values.shape[1]} features',flush=True)
                route_spec = {**spec,'interaction_features':[i for m in range(len(names)) for i in (6*m,6*m+1)]}
                logistic,scores = fit_logistic(keys,values,samples,types,fit_ids,route_spec)
                np.testing.assert_allclose(scores,apply_logistic(keys,values,logistic),rtol=1e-12,atol=1e-12)
                assert logistic['gradient_max'] < 1e-5
                decoder = select_fit(samples,{0.:candidates_from_scores(keys,scores,len(samples))},
                                     types,Config.from_environment(),fit_ids,'expanded')
                fitted = {**metadata,'logistic':logistic,'decoder':decoder}
            assert all(fitted[k] == v for k,v in metadata.items())
            fits[family][part] = fitted
            write_json(target,{'identity':identity,'fit':fitted})
            held_ids = ids if part == 'full' else [sid for sid in ids if folds[sid] == int(part[-1])]
            mode = 'full_dev' if part == 'full' else 'crossfit'
            scores = apply_logistic(keys,values,fitted['logistic'])
            predictions[family][mode].update(predict(candidates_from_scores(keys,scores,len(samples)),
                                                     fitted['decoder'],types,held_ids))
        reports[family] = {m:score_predictions(samples,p,types) for m,p in predictions[family].items()}
        (output/family).mkdir(exist_ok=True)
        for mode,pred in predictions[family].items():
            write_predictions(output/family/f'{mode}_predictions.jsonl',samples,pred,types)
        print(f'{family}: full={100*reports[family]["full_dev"]["micro"]["f1"]:.6f}%, '
              f'crossfit={100*reports[family]["crossfit"]["micro"]["f1"]:.6f}%',flush=True)
    for family,folder in [('v14',previous)]:
        for mode,filename in [('full_dev','full_dev_predictions.jsonl'),('crossfit','crossfit_predictions.jsonl')]:
            assert predictions[family][mode] == named_predictions(folder/filename,types)
        assert reports[family] == references[family]['nested']
    selected = {part:max((fits[f][part] for f in ROUTES),key=lambda fit:(
        fit['decoder']['fit_micro_f1'],fit['decoder']['fit_macro_f1'],-ROUTES.index(fit['family']))) for part in partitions}
    combined = {'full_dev':predictions[selected['full']['family']]['full_dev'],'crossfit':{}}
    for h in (0,1):
        combined['crossfit'].update({sid:pred for sid,pred in predictions[selected[f'held{h}']['family']]['crossfit'].items() if folds[sid] == h})
    nested = {m:score_predictions(samples,p,types) for m,p in combined.items()}
    fold_reports = []
    for h in (0,1):
        held = [sid for sid in ids if folds[sid] == h]
        current = subset_score(samples,combined['crossfit'],types,held)
        row = {'held_out_fold':h,'v15':current}
        for ref in references:
            baseline = subset_score(samples,predictions[ref]['crossfit'],types,held)
            row[ref] = baseline
            row[f'{ref}_delta_pp'] = 100*(current['micro']['f1']-baseline['micro']['f1'])
        fold_reports.append(row)
    accepted = (nested['full_dev']['micro']['f1'] >= reports['v14']['full_dev']['micro']['f1'] and
                all(nested['crossfit']['micro']['f1'] > reports[ref]['crossfit']['micro']['f1'] and
                    min(row[f'{ref}_delta_pp'] for row in fold_reports) >= 0 for ref in references))
    deployment = selected['full'] if accepted else fits['v14']['full']
    write_json(output/'comparison.json',{'identity':identity,'types':types,'plan':plan,'policy':spec,'routes':ROUTES,
        'reports':reports,'selected_fits':selected,'nested':nested,'baseline':reports['v14'],
        'fold_reports':fold_reports,'accepted':accepted,'coverage':coverages,'test_accessed':False,
        'seconds':time.perf_counter()-started})
    write_json(output/'selection.json',{'method':'expanded_FGM_seed44_type_slope_calibrated_spans','fit':deployment,
        'types':types,'identity':identity,
        'checkpoint_sha256':{name:caches[name][0]['checkpoint']['checkpoint_sha256'] for name in ('old','fgm43','fgm44')},
        'fgm44_used':any(n in deployment['component_models'] for n in ('fgm44','fgm44_expanded')),
        'gazetteer_sha256':sha256(PACKAGE/'inputs/v6_train_gazetteer.json'),'test_accessed':False})
    write_json(output/'crossfit_folds.json',folds)
    for mode,pred in combined.items():
        write_predictions(output/f'{mode}_predictions.jsonl',samples,pred,types)
    write_predictions(output/'selected_full_dev_predictions.jsonl',samples,
                      combined['full_dev'] if accepted else predictions['v14']['full_dev'],types)
    (output/'source_snapshot').mkdir(exist_ok=True)
    for name in [*own['files'],'PACKAGE_MANIFEST.json']:
        shutil.copy2(HERE/name,output/'source_snapshot'/name)
    shutil.copy2(plan_path,output/'source_snapshot/POLICY.json')
    print(f'Accepted={accepted}; deployment={deployment["family"]}; '
          f'nested crossfit={100*nested["crossfit"]["micro"]["f1"]:.6f}%',flush=True)


if __name__ == '__main__':
    main()
