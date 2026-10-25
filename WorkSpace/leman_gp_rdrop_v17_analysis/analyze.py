"""Execute only the four analysis routes frozen before v17 training; dev only."""
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'leman_gp_fgm_v10_analysis'))
from analyze import build_features, coverage
from common import PACKAGE, named_predictions, read_cache, read_json, sha256, subset_score, verify_package
from calibration_core import apply_logistic, candidates_from_scores, fit_logistic
from config import Config
from data import read_samples, write_json
from evaluate import score_predictions, write_predictions
from search import folds_for, predict, select_fit
import numpy as np

ROUTES = ['v14','rdrop','old_rdrop','fusion_rdrop']
COMPONENTS = {'rdrop':['rdrop'], 'old_rdrop':['old','rdrop'],
              'fusion_rdrop':['old','fgm43','fgm44','rdrop']}


def main():
    started = time.perf_counter()
    verify_package()
    own = read_json(HERE/'PACKAGE_MANIFEST.json')
    for name,digest in own['files'].items():
        assert sha256(HERE/name) == digest
    for name,digest in own['shared_dependencies'].items():
        assert sha256(HERE.parent/name/'PACKAGE_MANIFEST.json') == digest
    root = Path(Config.from_environment().root)
    output = root/'output/leman_rdrop_v17_analysis_seed45'
    plan = read_json(HERE/'EXPERIMENT_PLAN.json')
    prior = root/'output/leman_fgm_v14_analysis_seed44'
    assert sha256(prior/'comparison.json') == plan['baseline']['comparison_sha256']
    assert sha256(prior/'selection.json') == plan['baseline']['selection_sha256']
    v14 = read_json(prior/'comparison.json')
    training = root/'output/leman_rdrop_v17_seed45'
    checkpoint_audit = read_json(training/'checkpoint_audit.json')
    assert checkpoint_audit['status'] == 'PASS'
    paths = {'old':PACKAGE/'inputs/v6_upgraded_features.jsonl.gz',
             'fgm43':root/'output/leman_fgm_v10_analysis_seed43/fgm_features.jsonl.gz',
             'fgm44':prior/'fgm_features.jsonl.gz',
             'rdrop':training/'dev_calibration/upgraded/dev_scored_candidates.jsonl.gz'}
    caches = {name:read_cache(path) for name,path in paths.items()}
    types = caches['old'][0]['types']
    assert caches['rdrop'][0]['types'] == types
    assert caches['rdrop'][0]['checkpoint']['checkpoint_sha256'] == checkpoint_audit['checkpoint_sha256']
    assert checkpoint_audit['parent_sha256'] == plan['training']['parent_sha256']
    assert not read_json(training/'training_results.json')['test_evaluated']
    spec = read_json(PACKAGE/'POLICY.json')
    samples = read_samples(root/'data/dev2.txt',types)
    assert sha256(root/'data/dev2.txt') == spec['dev_sha256']
    assert sha256(root/'data/train2.txt') == spec['train_sha256']
    gazetteer = read_json(PACKAGE/'inputs/v6_train_gazetteer.json')
    ids,folds = list(range(len(samples))),folds_for(samples)
    partitions = {'full':ids,**{f'held{h}':[sid for sid in ids if folds[sid] != h] for h in (0,1)}}
    identity = {'plan_sha256':sha256(HERE/'EXPERIMENT_PLAN.json'),
                'manifest_sha256':sha256(HERE/'PACKAGE_MANIFEST.json'),
                'cache_sha256':{name:sha256(path) for name,path in paths.items()},'test_accessed':False}
    fits = {'v14':{part:{**fit,'family':'v14','inherited_fit':fit}
                    for part,fit in v14['selected_fits'].items()}}
    predictions = {'v14':{mode:named_predictions(prior/filename,types)
                    for mode,filename in [('full_dev','full_dev_predictions.jsonl'),('crossfit','crossfit_predictions.jsonl')]}}
    reports = {'v14':v14['nested']}
    coverages = {}
    (output/'fits').mkdir(parents=True,exist_ok=True)
    for family in ROUTES[1:]:
        names = COMPONENTS[family]
        keys,values = build_features([caches[n][1] for n in names],samples,types,gazetteer)
        coverages[family] = coverage(keys,samples)
        fits[family],predictions[family] = {},{'full_dev':{},'crossfit':{}}
        for part,fit_ids in partitions.items():
            target = output/'fits'/f'{family}_{part}.json'
            if target.exists():
                saved = read_json(target)
                assert saved['identity'] == identity
                fitted = saved['fit']
            else:
                print(f'Fit {family}/{part}: {len(keys)} candidates',flush=True)
                route_spec = {**spec,'interaction_features':[f for m in range(len(names)) for f in (6*m,6*m+1)]}
                logistic,scores = fit_logistic(keys,values,samples,types,fit_ids,route_spec)
                np.testing.assert_allclose(scores,apply_logistic(keys,values,logistic),rtol=1e-12,atol=1e-12)
                assert logistic['gradient_max'] < 1e-5
                decoder = select_fit(samples,{0.:candidates_from_scores(keys,scores,len(samples))},
                                     types,Config.from_environment(),fit_ids,'expanded')
                fitted = {'family':family,'component_models':names,'candidate_floor':-4.,
                          'logistic':logistic,'decoder':decoder}
                write_json(target,{'identity':identity,'fit':fitted})
            fits[family][part] = fitted
            held_ids = ids if part == 'full' else [sid for sid in ids if folds[sid] == int(part[-1])]
            scores = apply_logistic(keys,values,fitted['logistic'])
            mode = 'full_dev' if part == 'full' else 'crossfit'
            predictions[family][mode].update(predict(candidates_from_scores(keys,scores,len(samples)),
                                                    fitted['decoder'],types,held_ids))
        reports[family] = {mode:score_predictions(samples,p,types) for mode,p in predictions[family].items()}
        (output/family).mkdir(exist_ok=True)
        for mode,pred in predictions[family].items():
            write_predictions(output/family/f'{mode}_predictions.jsonl',samples,pred,types)
        print(f'{family}: full={100*reports[family]["full_dev"]["micro"]["f1"]:.6f}%; '
              f'crossfit={100*reports[family]["crossfit"]["micro"]["f1"]:.6f}%',flush=True)
    selected = {part:max((fits[family][part] for family in ROUTES),key=lambda fit:(
        fit['decoder']['fit_micro_f1'],fit['decoder']['fit_macro_f1'],int(fit['family'] == 'v14'),-len(fit.get('component_models',[])),-ROUTES.index(fit['family']))) for part in partitions}
    combined = {'full_dev':predictions[selected['full']['family']]['full_dev'],'crossfit':{}}
    for h in (0,1):
        combined['crossfit'].update({sid:p for sid,p in predictions[selected[f'held{h}']['family']]['crossfit'].items() if folds[sid] == h})
    nested = {mode:score_predictions(samples,p,types) for mode,p in combined.items()}
    fold_reports = []
    for h in (0,1):
        held = [sid for sid in ids if folds[sid] == h]
        current = subset_score(samples,combined['crossfit'],types,held)
        baseline = subset_score(samples,predictions['v14']['crossfit'],types,held)
        fold_reports.append({'held_out_fold':h,'v17':current,'v14':baseline,
                             'delta_pp':100*(current['micro']['f1']-baseline['micro']['f1'])})
    accepted = (nested['full_dev']['micro']['f1'] >= reports['v14']['full_dev']['micro']['f1'] and
                nested['crossfit']['micro']['f1'] > reports['v14']['crossfit']['micro']['f1'] and
                min(r['delta_pp'] for r in fold_reports) >= 0)
    write_json(output/'comparison.json',{'identity':identity,'types':types,'plan':plan,'routes':ROUTES,
        'reports':reports,'selected_fits':selected,'nested':nested,'baseline':reports['v14'],
        'fold_reports':fold_reports,'accepted':accepted,'coverage':coverages,
        'test_accessed':False,'seconds':time.perf_counter()-started,
        'checkpoint_audit':checkpoint_audit,
        'interpretation':'Selected v17 weights equal parent bitwise; duplicate-weight fusion cannot establish an R-Drop training gain'})
    write_json(output/'selection.json',{'fit':selected['full'] if accepted else fits['v14']['full'],
        'accepted':accepted,'identity':identity,'types':types,'test_accessed':False})
    write_json(output/'crossfit_folds.json',folds)
    for mode,pred in combined.items():
        write_predictions(output/f'{mode}_predictions.jsonl',samples,pred,types)
    print(f'Accepted={accepted}; full route={selected["full"]["family"]}; '
          f'workflow crossfit={100*nested["crossfit"]["micro"]["f1"]:.6f}%',flush=True)


if __name__ == '__main__':
    main()
