"""Train-only boundary contexts for the frozen final_v1 HGB feature pool."""
from datetime import datetime, timezone
from pathlib import Path
import gzip
import json
import sys
import time

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT=Path('/root/data/WorkSpace')
BASE=ROOT/'output/leman_final_v1_20261019'
OUTPUT=ROOT/'output/leman_context_v22'
sys.path[:0]=[str(ROOT/'leman_gp_typecal_v9'),str(ROOT/'leman_gp_typecal_v9/frozen_v4')]
from common import named_predictions, sha256, subset_score
from calibration_core import candidates_from_scores
from config import Config
from context_priors_v22 import build_priors, candidate_features
from data import read_samples, write_json
from evaluate import score_predictions, write_predictions
from remote_hgb47_v20 import refine, logits
from search import predict


def consensus_features(x):
    gp=x[:,:72:6]
    boundary=x[:,2:72:6]
    return np.column_stack((gp.max(1),gp.mean(1),gp.std(1),
                            np.quantile(gp,0.25,axis=1),np.quantile(gp,0.75,axis=1),
                            (gp>0).mean(1),boundary.mean(1),boundary.std(1)))


def main():
    started=time.perf_counter()
    OUTPUT.mkdir(exist_ok=True)
    freeze=json.loads((BASE/'FREEZE.json').read_text())
    parent_sha=sha256(BASE/'FREEZE.json')
    for name,expected in freeze['data_sha256'].items():
        assert sha256(ROOT/'data'/name)==expected,name
    with gzip.open(BASE/'caches/old.jsonl.gz','rt') as f:types=json.loads(next(f))['types']
    data=np.load(BASE/'features.npz')
    keys,x,y,outer=data['keys'],data['X'],data['y'],data['outer']
    sids=keys[:,0]
    samples=read_samples(ROOT/'data/dev2.txt',types)
    ids=list(range(len(samples)))
    baseline={m:named_predictions(BASE/f'legacy12/{m}_predictions.jsonl',types) for m in ('full_dev','crossfit')}
    cfg=Config.from_environment()
    params=dict(max_iter=180,learning_rate=0.05,max_leaf_nodes=15,min_samples_leaf=35,
                l2_regularization=2.0,max_bins=127,early_stopping=False,random_state=42)
    write_json(OUTPUT/'PLAN.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'parent_freeze_sha256':parent_sha,'parameters':params,'negative_weight':1.5,
        'features':['frozen 85 parent features','six train-only empirical Bayes boundary context log odds',
                    'optional eight fixed model-disagreement summaries'],
        'boundary_prior_strength':20.0,'boundary_prior_log_odds_cap':8.0,
        'cases':['context6','context6_consensus8'],'test_accessed':False,'neural_training':False,
        'decoder':'unchanged final_v1 per-type logit threshold refinement; fit on the corresponding fitting sentences only'})
    cached=OUTPUT/'context6/features.npz'
    if cached.exists():
        stored=np.load(cached)
        assert np.array_equal(stored['keys'],keys)
        extra=stored['X'][:,85:91]
    else:
        train=read_samples(ROOT/'data/train2.txt',types)
        tables,metadata=build_priors(train,len(types))
        del train
        joblib.dump(tables,OUTPUT/'train_boundary_priors.joblib',compress=3)
        write_json(OUTPUT/'boundary_feature_metadata.json',metadata)
        extra=candidate_features(keys,samples,tables)
    consensus=consensus_features(x)
    matrices={'context6':np.column_stack((x,extra)),
              'context6_consensus8':np.column_stack((x,extra,consensus))}
    report={'parent':'最终定稿版本1','parent_freeze_sha256':parent_sha,
            'baseline':{m:score_predictions(samples,p,types) for m,p in baseline.items()},
            'cases':{},'test_accessed':False,'neural_training':False,
            'scope':'Exploratory dev-fitted HGB calibration. Full-dev fits and evaluates the calibrator '
                    'on the same data; crossfit models and thresholds exclude their held-out fold.'}
    print('Context priors built; extra features',extra.shape,flush=True)
    for name,matrix in matrices.items():
        folder=OUTPUT/name;folder.mkdir(exist_ok=True)
        np.savez_compressed(folder/'features.npz',keys=keys,X=matrix,y=y,outer=outer)
        predictions={'full_dev':{},'crossfit':{}}
        fitted={}
        full_scores=np.full(len(keys),np.nan)
        cross_scores=np.full(len(keys),np.nan)
        for part in ('full','held0','held1'):
            fit_ids=ids if part=='full' else np.flatnonzero(outer!=int(part[-1])).tolist()
            eval_ids=ids if part=='full' else np.flatnonzero(outer==int(part[-1])).tolist()
            mask=np.ones(len(keys),dtype=bool) if part=='full' else outer[sids]!=int(part[-1])
            weights=np.where(y[mask]==1,1.0,1.5)
            filename='hgb_full_dev.joblib' if part=='full' else f'hgb_excluding_fold{part[-1]}.joblib'
            model_path=folder/filename
            decoder_path=folder/f'decoder_{part}.json'
            if model_path.exists() and decoder_path.exists():
                model=joblib.load(model_path)
            else:
                model=HistGradientBoostingClassifier(**params).fit(matrix[mask],y[mask],sample_weight=weights)
            z=logits(model,matrix)
            candidates=candidates_from_scores([tuple(map(int,k)) for k in keys],z,len(samples))
            if decoder_path.exists():
                decoder=json.loads(decoder_path.read_text())
            else:
                decoder=refine(samples,candidates,types,cfg,fit_ids)
            pred=predict(candidates,decoder,types,eval_ids)
            mode='full_dev' if part=='full' else 'crossfit'
            predictions[mode].update(pred)
            result=subset_score(samples,pred,types,eval_ids)
            base=subset_score(samples,baseline[mode],types,eval_ids)
            fitted[part]={'decoder':decoder,'report':result,
                          'delta_vs_final_v1_pp':100*(result['micro']['f1']-base['micro']['f1'])}
            if not model_path.exists():joblib.dump(model,model_path)
            if not decoder_path.exists():write_json(decoder_path,decoder)
            if part=='full':
                full_scores[:]=z
            else:
                held_mask=outer[sids]==int(part[-1]);cross_scores[held_mask]=z[held_mask]
            print(name,part,'F1',100*result['micro']['f1'],'delta',fitted[part]['delta_vs_final_v1_pp'],flush=True)
        row={'parameters':params,'negative_weight':1.5,'feature_columns':matrix.shape[1],
             'fits':fitted,'reports':{m:score_predictions(samples,p,types) for m,p in predictions.items()}}
        for mode,pred in predictions.items():
            write_predictions(folder/f'{mode}_predictions.jsonl',samples,pred,types)
        np.savez_compressed(folder/'scores.npz',full=full_scores,crossfit=cross_scores)
        report['cases'][name]=row
        report['seconds']=time.perf_counter()-started
        write_json(OUTPUT/'summary.json',report)
        print('DONE',name,{m:100*r['micro']['f1'] for m,r in row['reports'].items()},flush=True)
    assert sha256(BASE/'FREEZE.json')==parent_sha


if __name__=='__main__':
    main()
