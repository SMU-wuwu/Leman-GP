"""Fixed ordered source subsets, refitted classifiers; CPU-only dev analysis."""
import os
os.environ.setdefault('OMP_NUM_THREADS','8')
os.environ.setdefault('OPENBLAS_NUM_THREADS','8')
from pathlib import Path
import sys
import time
import json
from datetime import datetime, timezone
ROOT=Path('/root/data/WorkSpace');sys.path.insert(0,str(ROOT))
import remote_wide_v23 as old
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier

OUT=ROOT/'output/manuscript_revision_20261019/source_scale'
FROZEN=ROOT/'output/leman_final_v2_20261019'


def consensus(x,n):
    gp=x[:,:6*n:6];boundary=x[:,2:6*n:6]
    return np.column_stack((gp.max(1),gp.mean(1),gp.std(1),np.quantile(gp,.25,axis=1),
        np.quantile(gp,.75,axis=1),(gp>0).mean(1),boundary.mean(1),boundary.std(1)))


def main():
    OUT.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    plan=json.loads((ROOT/'output/leman_wide_v23/PLAN.json').read_text());names=plan['models']
    records={n:old.read_cache(FROZEN/'caches'/f'{n}.jsonl.gz') for n in names}
    types=records[names[0]][0]['types'];samples=old.read_samples(ROOT/'data/dev2.txt',types)
    ids=list(range(len(samples)));outer=np.asarray(old.folds_for(samples));cfg=old.Config.from_environment()
    gaz=json.loads((FROZEN/'v6_train_gazetteer.json').read_text());priors=joblib.load(FROZEN/'train_boundary_priors.joblib')
    spec={'created_utc':datetime.now(timezone.utc).isoformat(),'source_order':names,
       'subsets':{str(n):names[:n] for n in (1,3,6,12)},
       'selection':'Use prefixes of the already frozen source order, no subset ranking/search',
       'parameters':plan['parameters'],'negative_weight':1.5,
       'features':'6 per source + 13 type one-hot + 6 context + 8 consensus; recalculate consensus for each subset',
       'candidate_pool':'Union of the selected source caches; missing-source feature convention unchanged',
       'equal_reference':'Arithmetic mean of the 12 GP native-scale features, including fixed missing-score values; no HGB; refit same decoder',
       'scope':'Exploratory system-size comparison; source count changes both available candidates and features. 12-source HGB reuses frozen final v2.',
       'script_sha256':old.sha256(__file__),'frozen_freeze_sha256':old.sha256(FROZEN/'FREEZE.json'),
       'neural_training':False,'test_accessed':False}
    old.write_json(OUT/'PLAN.json',spec);report={'plan':spec,'cases':{}}
    for n in (1,3,6,12):
        name=f'sources{n}_hgb' if n<12 else 'sources12_equal'
        folder=OUT/name;folder.mkdir()
        keys,values=old.build_features([records[k][1] for k in names[:n]],samples,types,gaz)
        ka=np.array(keys,dtype=np.int32);sids=ka[:,0]
        matrix=np.column_stack((values,np.eye(len(types))[ka[:,1]],old.candidate_features(ka,samples,priors),consensus(values,n)))
        gold=[set(map(tuple,g)) for _,g in samples]
        y=np.asarray([tuple(k[1:]) in gold[k[0]] for k in keys],dtype=np.uint8)
        np.savez_compressed(folder/'features.npz',keys=ka,X=matrix,y=y,outer=outer)
        predictions={'full_dev':{},'crossfit':{}};fits={};scores={k:np.full(len(keys),np.nan) for k in ('full','crossfit')}
        for part in ('full','held0','held1'):
            fit_ids=ids if part=='full' else np.flatnonzero(outer!=int(part[-1])).tolist()
            eval_ids=ids if part=='full' else np.flatnonzero(outer==int(part[-1])).tolist()
            mask=np.ones(len(keys),dtype=bool) if part=='full' else outer[sids]!=int(part[-1])
            if n<12:
                model=HistGradientBoostingClassifier(**plan['parameters']).fit(matrix[mask],y[mask],sample_weight=np.where(y[mask],1.,1.5))
                z=old.logits(model,matrix)
                joblib.dump(model,folder/('hgb_full_dev.joblib' if part=='full' else f'hgb_excluding_fold{part[-1]}.joblib'))
            else:z=values[:,:72:6].mean(1)
            cand=old.candidates_from_scores(keys,z,len(samples));decoder=old.refine(samples,cand,types,cfg,fit_ids)
            pred=old.predict(cand,decoder,types,eval_ids);mode='full_dev' if part=='full' else 'crossfit';predictions[mode].update(pred)
            old.write_json(folder/f'decoder_{part}.json',decoder)
            fits[part]={'report':old.subset_score(samples,pred,types,eval_ids),'decoder':decoder}
            if part=='full':scores['full'][:]=z
            else:
                held=outer[sids]==int(part[-1]);scores['crossfit'][held]=z[held]
            print(name,part,fits[part]['report']['micro']['f1'],flush=True)
        for mode,pred in predictions.items():old.write_predictions(folder/f'{mode}_predictions.jsonl',samples,pred,types)
        np.savez_compressed(folder/'scores.npz',**scores)
        report['cases'][name]={'sources':names[:n],'feature_columns':matrix.shape[1],
            'candidate_rows':len(keys),'gold_covered':int(y.sum()),'fits':fits,
            'reports':{m:old.score_predictions(samples,p,types) for m,p in predictions.items()}}
        report['seconds']=time.perf_counter()-started;old.write_json(OUT/'RESULT.json',report)
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
