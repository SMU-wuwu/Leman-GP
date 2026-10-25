"""Wider existing seed46/47 caches, with the final_v1 feature/decoder recipe."""
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
OUTPUT=ROOT/'output/leman_wide_v23'
sys.path[:0]=[str(ROOT/'leman_gp_typecal_v9'),str(ROOT/'leman_gp_typecal_v9/frozen_v4')]
from common import named_predictions, read_cache, sha256, subset_score
from calibration_core import candidates_from_scores
from config import Config
from context_priors_v22 import candidate_features
from data import read_samples, write_json
from evaluate import score_predictions, write_predictions
from remote_hgb47_v20 import build_features, refine, logits
from remote_context_v22 import consensus_features
from search import folds_for, predict


def main():
    started=time.perf_counter();OUTPUT.mkdir(exist_ok=True)
    frozen=json.loads((BASE/'FREEZE.json').read_text())
    parent_sha=sha256(BASE/'FREEZE.json')
    names=json.loads((BASE/'analysis_v20.json').read_text())['models']
    paths={name:BASE/'caches'/f'{name}.jsonl.gz' for name in names}
    for seed in (46,47):paths[f'fgm{seed}']=ROOT/f'output/leman_candidates_v18_seed{seed}/fgm_features.jsonl.gz'
    records={name:read_cache(p) for name,p in paths.items()}
    types=records['old'][0]['types']
    samples=read_samples(ROOT/'data/dev2.txt',types)
    ids=list(range(len(samples)));outer=np.asarray(folds_for(samples))
    gaz=json.loads((ROOT/'output/leman_final_v1_assets_20261019/v6_train_gazetteer.json').read_text())
    keys,values=build_features([records[n][1] for n in names],samples,types,gaz)
    ka=np.asarray(keys,dtype=np.int32)
    x=np.column_stack((values,np.eye(len(types))[ka[:,1]]))
    gold=[set(map(tuple,spans)) for _,spans in samples]
    y=np.asarray([tuple(k[1:]) in gold[k[0]] for k in keys],dtype=np.uint8)
    prior_path=ROOT/'output/leman_context_v22/train_boundary_priors.joblib'
    tables=joblib.load(prior_path)
    extra=candidate_features(ka,samples,tables)
    del tables
    consensus=consensus_features(x)
    matrices={'wide_base85':x,'wide_context6':np.column_stack((x,extra)),
              'wide_context6_consensus8':np.column_stack((x,extra,consensus))}
    baseline={m:named_predictions(BASE/f'legacy12/{m}_predictions.jsonl',types) for m in ('full_dev','crossfit')}
    params=dict(max_iter=180,learning_rate=0.05,max_leaf_nodes=15,min_samples_leaf=35,
                l2_regularization=2.0,max_bins=127,early_stopping=False,random_state=42)
    write_json(OUTPUT/'PLAN.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'parent_freeze_sha256':parent_sha,'models':names,'parameters':params,'negative_weight':1.5,
        'cases':list(matrices),'cache_sha256':{n:sha256(p) for n,p in paths.items()},
        'candidate_rows':len(keys),'gold_covered':int(y.sum()),'gold':sum(map(len,gold)),
        'boundary_priors_sha256':sha256(prior_path),'test_accessed':False,'neural_training':False,
        'change':'Replace only the seed46/47 candidate caches by their already extracted floor=-32 versions. '
                 'Keep the other ten sources, the missing-score convention, classifier parameters and threshold search unchanged.'})
    write_json(OUTPUT/'dev_caches.json',{n:str(p) for n,p in paths.items()})
    report={'parent':'最终定稿版本1','parent_freeze_sha256':parent_sha,
            'candidate_rows':len(keys),'gold_covered':int(y.sum()),'baseline':frozen,
            'cases':{},'test_accessed':False,'neural_training':False}
    cfg=Config.from_environment();sids=ka[:,0]
    print('Expanded pool',len(keys),'gold covered',int(y.sum()),flush=True)
    for name,matrix in matrices.items():
        folder=OUTPUT/name;folder.mkdir(exist_ok=True)
        np.savez_compressed(folder/'features.npz',keys=ka,X=matrix,y=y,outer=outer)
        predictions={'full_dev':{},'crossfit':{}};fits={}
        scores={'full':np.full(len(keys),np.nan),'crossfit':np.full(len(keys),np.nan)}
        for part in ('full','held0','held1'):
            fit_ids=ids if part=='full' else np.flatnonzero(outer!=int(part[-1])).tolist()
            eval_ids=ids if part=='full' else np.flatnonzero(outer==int(part[-1])).tolist()
            mask=np.ones(len(keys),dtype=bool) if part=='full' else outer[sids]!=int(part[-1])
            model=HistGradientBoostingClassifier(**params).fit(matrix[mask],y[mask],sample_weight=np.where(y[mask]==1,1.0,1.5))
            z=logits(model,matrix);cand=candidates_from_scores(keys,z,len(samples))
            decoder=refine(samples,cand,types,cfg,fit_ids)
            pred=predict(cand,decoder,types,eval_ids)
            mode='full_dev' if part=='full' else 'crossfit';predictions[mode].update(pred)
            result=subset_score(samples,pred,types,eval_ids)
            base=subset_score(samples,baseline[mode],types,eval_ids)
            fits[part]={'decoder':decoder,'report':result,'delta_vs_final_v1_pp':100*(result['micro']['f1']-base['micro']['f1'])}
            filename='hgb_full_dev.joblib' if part=='full' else f'hgb_excluding_fold{part[-1]}.joblib'
            joblib.dump(model,folder/filename);write_json(folder/f'decoder_{part}.json',decoder)
            if part=='full':scores['full'][:]=z
            else:
                selected=outer[sids]==int(part[-1]);scores['crossfit'][selected]=z[selected]
            print(name,part,100*result['micro']['f1'],'delta',fits[part]['delta_vs_final_v1_pp'],flush=True)
        for mode,pred in predictions.items():write_predictions(folder/f'{mode}_predictions.jsonl',samples,pred,types)
        np.savez_compressed(folder/'scores.npz',**scores)
        row={'feature_columns':matrix.shape[1],'fits':fits,'reports':{m:score_predictions(samples,p,types) for m,p in predictions.items()}}
        report['cases'][name]=row;report['seconds']=time.perf_counter()-started
        write_json(OUTPUT/'summary.json',report)
        print('DONE',name,{m:100*r['micro']['f1'] for m,r in row['reports'].items()},flush=True)
    assert sha256(BASE/'FREEZE.json')==parent_sha


if __name__=='__main__':main()
