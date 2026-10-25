"""Explore probability / interval-posterior decoding of frozen final_v1 models."""
from datetime import datetime, timezone
from pathlib import Path
import json
import sys
import time

import joblib
import numpy as np

ROOT = Path('/root/data/WorkSpace')
BASE = ROOT/'output/leman_final_v1_20261019'
OUTPUT = ROOT/'output/leman_decode_v21'
sys.path[:0] = [str(ROOT/'leman_gp_typecal_v9'), str(ROOT/'leman_gp_typecal_v9/frozen_v4')]
from common import named_predictions, sha256, subset_score
from data import read_samples, write_json
from evaluate import score_predictions, write_predictions
from fast_intervals_v21 import GridIntervals, optimize
from interval_marginals_v21 import interval_marginals
from predict_hgb_v20 import decode


CONFIGS = [
    {'name': f'legacy_{mode}_T{temp:g}', 'compact_weight': 0.0, 'mode': mode, 'temperature': temp}
    for mode in ('direct', 'posterior') for temp in (0.5, 1.0, 2.0)
] + [
    {'name': f'blend{weight:g}_{mode}', 'compact_weight': weight, 'mode': mode, 'temperature': 1.0}
    for weight in (0.25, 0.5) for mode in ('direct', 'posterior')
]


def probabilities(keys, scores, samples, config):
    result = np.zeros(len(keys))
    # Keep the original post-calibrator logit floor; no new model candidates.
    valid = scores > -4.0
    if config['mode'] == 'direct':
        result[valid] = 8.0*np.exp(-np.logaddexp(0.0, -scores[valid]/config['temperature']))
        return result
    rows = [[] for _ in samples]
    positions = [[] for _ in samples]
    for i in np.flatnonzero(valid):
        sid,t,a,b = keys[i]
        rows[sid].append((int(t),int(a),int(b),float(scores[i]/config['temperature'])))
        positions[sid].append(i)
    for sid in range(len(samples)):
        p,_ = interval_marginals(rows[sid],len(samples[sid][0]))
        result[positions[sid]]=8.0*p
    return result


def predicted(keys, values, ids, thresholds):
    rows={sid:[] for sid in ids}
    for (sid,t,a,b),u in zip(keys,values):
        if sid in rows and u >= 0.01:
            rows[sid].append((int(t),int(a),int(b),float(u)))
    return {sid:decode(spans,thresholds,'weighted_interval') for sid,spans in rows.items()}


def fit_grid(keys,values,samples,ids,initial=None,refine=False):
    local={sid:i for i,sid in enumerate(ids)}
    mask=np.asarray([sid in local for sid in keys[:,0]])
    k=keys[mask].copy()
    k[:,0]=np.asarray([local[sid] for sid in k[:,0]])
    prepared=GridIntervals(k,values[mask],[samples[sid] for sid in ids])
    return optimize(prepared,initial=initial,refine=refine)


def main():
    started=time.perf_counter()
    OUTPUT.mkdir(exist_ok=True)
    freeze=json.loads((BASE/'FREEZE.json').read_text())
    identity=sha256(BASE/'FREEZE.json')
    data=np.load(BASE/'features.npz')
    keys,x,outer=data['keys'],data['X'],data['outer']
    import gzip
    with gzip.open(BASE/'caches/old.jsonl.gz','rt') as f:types=json.loads(next(f))['types']
    samples=read_samples(ROOT/'data/dev2.txt',types)
    ids=list(range(len(samples)))
    fixed=json.loads((BASE/'analysis_v20.json').read_text())
    baseline={m:named_predictions(BASE/f'legacy12/{m}_predictions.jsonl',types) for m in ('full_dev','crossfit')}
    source=ROOT/'output/leman_hgb_v20_seed47'
    models={case:{part:joblib.load((BASE if case=='legacy12' else source)/case/filename)
                  for part,filename in [('full','hgb_full_dev.joblib'),('held0','hgb_excluding_fold0.joblib'),('held1','hgb_excluding_fold1.joblib')]}
            for case in ('legacy12','compact12')}
    scores={case:{part:np.log(np.clip(m.predict_proba(x)[:,1],1e-7,1-1e-7))-
                 np.log1p(-np.clip(m.predict_proba(x)[:,1],1e-7,1-1e-7)) for part,m in parts.items()}
            for case,parts in models.items()}
    write_json(OUTPUT/'PLAN.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'baseline_freeze_sha256':identity,'configurations':CONFIGS,
        'selection':'Each partition fits thresholds and selects a decoder using only its fitting sentences. '
                    'Full-dev is a same-dev fitting/selection score; outer held predictions are kept separately.',
        'refinement':'Top three coarse configurations per partition, incumbent included as a fallback.',
        'neural_training':False,'test_accessed':False})
    all_fits={}
    combined={'full_dev':dict(baseline['full_dev']),'crossfit':{}}
    route_predictions={c['name']:{'full_dev':{},'crossfit':{}} for c in CONFIGS}
    chosen={}
    for part in ('full','held0','held1'):
        fit_ids=ids if part=='full' else np.flatnonzero(outer!=int(part[-1])).tolist()
        eval_ids=ids if part=='full' else np.flatnonzero(outer==int(part[-1])).tolist()
        mode='full_dev' if part=='full' else 'crossfit'
        baseline_fit=fixed['cases']['legacy12']['full_decoder'] if part=='full' else fixed['cases']['legacy12']['folds'][int(part[-1])]['decoder']
        best={'name':'final_v1','micro_f1':baseline_fit['fit_micro_f1'],'thresholds':baseline_fit['thresholds']}
        best_pred={sid:baseline[mode][sid] for sid in eval_ids}
        utility_cache={}
        fits=[]
        for config in CONFIGS:
            w=config['compact_weight']
            z=(1.0-w)*scores['legacy12'][part]+w*scores['compact12'][part]
            utility=probabilities(keys,z,samples,config)
            utility_cache[config['name']]=utility
            fitted=fit_grid(keys,utility,samples,fit_ids)
            entry={**config,**fitted,'refined':False}
            fits.append(entry)
            print(part,config['name'],'fit',100*entry['micro_f1'],flush=True)
        for entry in sorted(fits,key=lambda e:-e['micro_f1'])[:3]:
            refined=fit_grid(keys,utility_cache[entry['name']],samples,fit_ids,entry['thresholds_array'],True)
            if refined['micro_f1']>entry['micro_f1']:
                entry.update(refined,refined=True)
        for entry in fits:
            utility=utility_cache[entry['name']]
            thresholds=entry['thresholds_array']
            pred=predicted(keys,utility,eval_ids,thresholds)
            route_predictions[entry['name']][mode].update(pred)
            entry['held_report']=subset_score(samples,pred,types,eval_ids)
            if entry['micro_f1']>best['micro_f1']+1e-14:
                best={**entry,'thresholds':dict(zip(types,thresholds))}
                best_pred=pred
        combined[mode].update(best_pred)
        chosen[part]=best
        all_fits[part]=fits
        write_json(OUTPUT/'fits.json',{'fits':all_fits,'chosen':chosen})
        print('CHOSEN',part,best['name'],100*best['micro_f1'],flush=True)
    reports={m:score_predictions(samples,p,types) for m,p in combined.items()}
    for mode,pred in combined.items():
        write_predictions(OUTPUT/f'{mode}_predictions.jsonl',samples,pred,types)
    route_reports={name:{m:score_predictions(samples,p,types) for m,p in pair.items()} for name,pair in route_predictions.items()}
    result={'parent':'最终定稿版本1','parent_freeze_sha256':identity,'reports':reports,
            'baseline':{m:score_predictions(samples,p,types) for m,p in baseline.items()},
            'chosen':chosen,'routes':route_reports,'test_accessed':False,'neural_training':False,
            'seconds':time.perf_counter()-started,'script_sha256':sha256(Path(__file__))}
    write_json(OUTPUT/'summary.json',result)
    assert sha256(BASE/'FREEZE.json')==identity
    print('FINAL full',100*reports['full_dev']['micro']['f1'],'crossfit',100*reports['crossfit']['micro']['f1'],flush=True)


if __name__=='__main__':
    main()
