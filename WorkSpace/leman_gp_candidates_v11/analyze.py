"""Expanded candidates with unchanged v10 calibrator and frozen baseline."""
import json
from pathlib import Path
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'leman_gp_fgm_v10_analysis'))
from analyze import build_features as original_features
from common import PACKAGE, named_predictions, read_cache, read_json, sha256, subset_score, verify_package
from calibration_core import apply_logistic, candidates_from_scores, fit_logistic
from config import Config
from data import read_samples, write_json
from evaluate import score_predictions, write_predictions
from search import folds_for, predict, select_fit


def expanded_features(records, samples, types, gazetteer):
    keys, values = original_features(records, samples, types, gazetteer)
    # Replace missing scores only; boundary and surface features remain unchanged.
    present = [{(sid, t, a, b) for sid, rows in source.items() for t, a, b, _, _ in rows}
               for source in records]
    for i, key in enumerate(keys):
        for model, available in enumerate(present):
            if key not in available:
                values[i, 6*model] = -32.0/8.0
    return keys, values


def main():
    started = time.perf_counter()
    verify_package()
    policy = read_json(HERE/'POLICY.json')
    manifest = read_json(HERE/'PACKAGE_MANIFEST.json')
    for name, digest in manifest['files'].items():
        assert sha256(HERE/name) == digest
    root = Path(Config.from_environment().root)
    output = root/'output/leman_candidates_v11_seed43'
    previous = root/'output/leman_fgm_v10_analysis_seed43'
    assert sha256(previous/'selection.json') == policy['baseline_selection_sha256']
    assert sha256(previous/'comparison.json') == policy['baseline_comparison_sha256']
    prior = read_json(previous/'comparison.json')
    header, fgm = read_cache(output/'fgm_features.jsonl.gz')
    old_header, old = read_cache(root/'output/leman_inference_v7_seed42/upgraded_features.jsonl.gz')
    _, base_fgm = read_cache(previous/'fgm_features.jsonl.gz')
    _, base_old = read_cache(PACKAGE/'inputs/v6_upgraded_features.jsonl.gz')
    assert {sid: [r for r in rows if r[3] > -4] for sid, rows in fgm.items()} == base_fgm
    # The v4 expanded cache is an earlier independent GPU inference. Its
    # original candidate keys agree, while floating scores need not be bitwise identical.
    assert set(old) == set(base_old)
    for sid in old:
        assert {tuple(r[:3]) for r in old[sid] if r[3] > -4} == {tuple(r[:3]) for r in base_old[sid]}
    for name, h in [('fgm', header), ('v4', old_header)]:
        assert h['floor'] == -32 and not h['test_accessed']
        assert h['checkpoint']['checkpoint_sha256'] == policy['checkpoint_sha256'][name]
        assert h['dev_sha256'] == policy['dev_sha256']
    types = header['types']
    samples = read_samples(root/'data/dev2.txt', types)
    assert sha256(root/'data/dev2.txt') == policy['dev_sha256']
    assert sha256(root/'data/train2.txt') == policy['train_sha256']
    gazetteer = read_json(PACKAGE/'inputs/v6_train_gazetteer.json')
    ids, folds = list(range(len(samples))), folds_for(samples)
    partitions = {'full': ids, **{f'held{h}':[sid for sid in ids if folds[sid] != h] for h in (0,1)}}
    identity = {'manifest_sha256':sha256(HERE/'PACKAGE_MANIFEST.json'),
                'policy_sha256':sha256(HERE/'POLICY.json'),
                'cache_sha256':{'fgm':sha256(output/'fgm_features.jsonl.gz'),
                                'old':sha256(root/'output/leman_inference_v7_seed42/upgraded_features.jsonl.gz')},
                'test_accessed':False}
    spec, routes = read_json(PACKAGE/'POLICY.json'), policy['routes']
    fits, predictions, reports, coverages = {}, {}, {}, {}
    (output/'fits').mkdir(exist_ok=True)
    for family in routes:
        records = [base_old,base_fgm] if family == 'v10' else [fgm] if family == 'expanded_fgm' else [old,fgm]
        builder = original_features if family == 'v10' else expanded_features
        keys, values = builder(records, samples, types, gazetteer)
        available = sum(tuple(key[1:]) in samples[key[0]][1] for key in keys)
        coverages[family] = {'gold':20782,'available_gold':available,'gold_coverage':available/20782,'unique_candidates':len(keys)}
        route_spec = {**spec,'interaction_features':[0,1] if family == 'expanded_fgm' else [0,1,6,7]}
        fits[family], predictions[family] = {}, {'full_dev':{},'crossfit':{}}
        print(f'{family}: {len(keys)} candidates', flush=True)
        for part, fit_ids in partitions.items():
            target = output/'fits'/f'{family}_{part}.json'
            if target.exists():
                saved = read_json(target)
                assert saved['identity'] == identity
                fitted = saved['fit']
            elif family == 'v10':
                assert prior['selected_fits'][part]['family'] == 'fusion'
                fitted = {**prior['selected_fits'][part],'family':'v10'}
            else:
                print(f'Fit {family}/{part}', flush=True)
                logistic, scores = fit_logistic(keys, values, samples, types, fit_ids, route_spec)
                assert logistic['gradient_max'] < 1e-5
                decoder = select_fit(samples,{0.0:candidates_from_scores(keys,scores,len(samples))},
                                     types,Config.from_environment(),fit_ids,'expanded')
                fitted = {'family':family,'logistic':logistic,'decoder':decoder}
            fits[family][part] = fitted
            write_json(target,{'identity':identity,'fit':fitted})
            scores = apply_logistic(keys, values, fitted['logistic'])
            held_ids = ids if part == 'full' else [sid for sid in ids if folds[sid] == int(part[-1])]
            mode = 'full_dev' if part == 'full' else 'crossfit'
            predictions[family][mode].update(predict(candidates_from_scores(keys,scores,len(samples)),
                                                     fitted['decoder'],types,held_ids))
        reports[family] = {m:score_predictions(samples,p,types) for m,p in predictions[family].items()}
        (output/family).mkdir(exist_ok=True)
        for mode,pred in predictions[family].items():
            write_predictions(output/family/f'{mode}_predictions.jsonl',samples,pred,types)
        print(f'{family}: full={100*reports[family]["full_dev"]["micro"]["f1"]:.6f}%, '
              f'crossfit={100*reports[family]["crossfit"]["micro"]["f1"]:.6f}%',flush=True)
    for mode,filename in [('full_dev','selected_full_dev_predictions.jsonl'),('crossfit','crossfit_predictions.jsonl')]:
        assert predictions['v10'][mode] == named_predictions(previous/filename,types)
    selected = {p:max((fits[f][p] for f in routes),key=lambda fit:(
        fit['decoder']['fit_micro_f1'],fit['decoder']['fit_macro_f1'],-routes.index(fit['family']))) for p in partitions}
    combined = {'full_dev':predictions[selected['full']['family']]['full_dev'],'crossfit':{}}
    for h in (0,1):
        combined['crossfit'].update({sid:pred for sid,pred in predictions[selected[f'held{h}']['family']]['crossfit'].items() if folds[sid] == h})
    nested = {m:score_predictions(samples,p,types) for m,p in combined.items()}
    fold_reports = []
    for h in (0,1):
        held = [sid for sid in ids if folds[sid] == h]
        base = subset_score(samples,predictions['v10']['crossfit'],types,held)
        current = subset_score(samples,combined['crossfit'],types,held)
        fold_reports.append({'held_out_fold':h,'v10':base,'v11':current,'micro_delta_pp':100*(current['micro']['f1']-base['micro']['f1'])})
    accepted = (nested['full_dev']['micro']['f1'] >= reports['v10']['full_dev']['micro']['f1'] and
                nested['crossfit']['micro']['f1'] > reports['v10']['crossfit']['micro']['f1'] and
                min(r['micro_delta_pp'] for r in fold_reports) >= 0)
    deployment = selected['full'] if accepted else fits['v10']['full']
    write_json(output/'comparison.json',{'identity':identity,'types':types,'policy':policy,'routes':routes,
        'reports':reports,'selected_fits':selected,'nested':nested,'baseline':reports['v10'],
        'fold_reports':fold_reports,'accepted':accepted,'coverage':coverages,'test_accessed':False,
        'seconds':time.perf_counter()-started})
    write_json(output/'selection.json',{'method':'expanded_FGM_type_slope_calibrated_spans','fit':deployment,
        'types':types,'identity':identity,'checkpoint_sha256':policy['checkpoint_sha256'],
        'candidate_floor':-32.0 if accepted and deployment['family'] != 'v10' else -4.0,
        'gazetteer_sha256':sha256(PACKAGE/'inputs/v6_train_gazetteer.json'),'test_accessed':False})
    write_json(output/'crossfit_folds.json',folds)
    for mode,pred in combined.items():
        write_predictions(output/f'{mode}_predictions.jsonl',samples,pred,types)
    write_predictions(output/'selected_full_dev_predictions.jsonl',samples,
                      combined['full_dev'] if accepted else predictions['v10']['full_dev'],types)
    (output/'source_snapshot').mkdir(exist_ok=True)
    for name in ('extract.py','analyze.py','POLICY.json','PACKAGE_MANIFEST.json'):
        shutil.copy2(HERE/name,output/'source_snapshot'/name)
    print(f'Accepted={accepted}; deployment={deployment["family"]}; nested={100*nested["crossfit"]["micro"]["f1"]:.6f}%',flush=True)


if __name__ == '__main__':
    main()
