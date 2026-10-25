"""Run the analysis routes frozen before seed44 training, using dev only."""
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

ROUTES = ['v10', 'v11', 'fgm44', 'fusion44']


def main():
    started = time.perf_counter()
    verify_package()
    own = read_json(HERE/'PACKAGE_MANIFEST.json')
    for name, digest in own['files'].items():
        assert sha256(HERE/name) == digest
    for name, digest in own['shared_dependencies'].items():
        assert sha256(HERE.parent/name/'PACKAGE_MANIFEST.json') == digest
    root = Path(Config.from_environment().root)
    output = root/'output/leman_fgm_v14_analysis_seed44'
    previous = root/'output/leman_fgm_v10_analysis_seed43'
    v11_folder = root/'output/leman_candidates_v11_seed43'
    plan_path = HERE.parent/'leman_gp_fgm_seed44_v14/EXPERIMENT_PLAN.json'
    plan = read_json(plan_path)
    assert sha256(previous/'selection.json') == plan['baseline_selection_sha256']
    assert sha256(previous/'comparison.json') == plan['baseline_v10_comparison_sha256']
    assert sha256(v11_folder/'comparison.json') == plan['baseline_v11_comparison_sha256']
    references = {'v10': read_json(previous/'comparison.json'),
                  'v11': read_json(v11_folder/'comparison.json')}
    paths = {'old': PACKAGE/'inputs/v6_upgraded_features.jsonl.gz',
             'fgm43': previous/'fgm_features.jsonl.gz',
             'fgm44': output/'fgm_features.jsonl.gz',
             'old_expanded': root/'output/leman_inference_v7_seed42/upgraded_features.jsonl.gz',
             'fgm43_expanded': v11_folder/'fgm_features.jsonl.gz'}
    caches = {name: read_cache(path) for name, path in paths.items()}
    header = caches['fgm44'][0]
    checkpoint_audit = read_json(output/'checkpoint_audit.json')
    assert checkpoint_audit['status'] == 'PASS'
    assert header['checkpoint']['checkpoint_sha256'] == checkpoint_audit['checkpoint_sha256']
    assert header['floor'] == -4 and header['plan_sha256'] == sha256(plan_path)
    assert not header['test_accessed'] and header['checkpoint']['checkpoint_epoch'] == 1
    assert header['checkpoint']['weight_source'] == 'live'
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
            floor = -4.0
            if family == 'v10' or family == 'v11' and inherited['family'] == 'v10':
                names = ['old','fgm43']
            elif family == 'v11':
                floor = -32.0
                names = ['fgm43_expanded'] if inherited['family'] == 'expanded_fgm' else ['old_expanded','fgm43_expanded']
            else:
                names = ['fgm44'] if family == 'fgm44' else ['old','fgm43','fgm44']
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
    for family,folder in [('v10',previous),('v11',v11_folder)]:
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
        row = {'held_out_fold':h,'v14':current}
        for ref in references:
            baseline = subset_score(samples,predictions[ref]['crossfit'],types,held)
            row[ref] = baseline
            row[f'{ref}_delta_pp'] = 100*(current['micro']['f1']-baseline['micro']['f1'])
        fold_reports.append(row)
    accepted = (nested['full_dev']['micro']['f1'] >= reports['v10']['full_dev']['micro']['f1'] and
                all(nested['crossfit']['micro']['f1'] > reports[ref]['crossfit']['micro']['f1'] and
                    min(row[f'{ref}_delta_pp'] for row in fold_reports) >= 0 for ref in references))
    deployment = selected['full'] if accepted else fits['v10']['full']
    write_json(output/'comparison.json',{'identity':identity,'types':types,'plan':plan,'policy':spec,'routes':ROUTES,
        'reports':reports,'selected_fits':selected,'nested':nested,'baseline':reports['v10'],
        'fold_reports':fold_reports,'accepted':accepted,'coverage':coverages,'test_accessed':False,
        'seconds':time.perf_counter()-started})
    write_json(output/'selection.json',{'method':'FGM_seed44_type_slope_calibrated_spans','fit':deployment,
        'types':types,'identity':identity,
        'checkpoint_sha256':{name:caches[name][0]['checkpoint']['checkpoint_sha256'] for name in ('old','fgm43','fgm44')},
        'fgm44_used':accepted and deployment['family'] in ('fgm44','fusion44'),
        'gazetteer_sha256':sha256(PACKAGE/'inputs/v6_train_gazetteer.json'),'test_accessed':False})
    write_json(output/'crossfit_folds.json',folds)
    for mode,pred in combined.items():
        write_predictions(output/f'{mode}_predictions.jsonl',samples,pred,types)
    write_predictions(output/'selected_full_dev_predictions.jsonl',samples,
                      combined['full_dev'] if accepted else predictions['v10']['full_dev'],types)
    (output/'source_snapshot').mkdir(exist_ok=True)
    for name in [*own['files'],'PACKAGE_MANIFEST.json']:
        shutil.copy2(HERE/name,output/'source_snapshot'/name)
    shutil.copy2(plan_path,output/'source_snapshot/EXPERIMENT_PLAN.json')
    print(f'Accepted={accepted}; deployment={deployment["family"]}; '
          f'nested crossfit={100*nested["crossfit"]["micro"]["f1"]:.6f}%',flush=True)


if __name__ == '__main__':
    main()
