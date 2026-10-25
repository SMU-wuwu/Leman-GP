"""One fixed checkpoint/cache; raw/global/type/all-span/interval comparisons."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
from pathlib import Path
import sys
import json
from datetime import datetime, timezone
ROOT=Path('/root/data/WorkSpace');sys.path.insert(0,str(ROOT))
import remote_wide_v23 as old
from decoding import decode_sentence
import numpy as np

OUT=ROOT/'output/manuscript_revision_20261019/decoder_ablation'
FROZEN=ROOT/'output/leman_final_v2_20261019'


def fit(samples,candidates,types,ids,strategy,per_type):
    gold={sid:set(map(tuple,samples[sid][1])) for sid in ids}
    def count(sid,thresholds):
        p=decode_sentence(candidates[sid],thresholds,strategy)
        return np.array([len(p&gold[sid]),len(p),len(gold[sid])],dtype=np.int64)
    def f1(total):return 2*total[0]/(total[1]+total[2])
    best=None
    for threshold in np.arange(-4,4.0001,.25):
        th=np.full(len(types),threshold);rows={sid:count(sid,th) for sid in ids};total=sum(rows.values())
        key=(f1(total),threshold)
        if best is None or key>best[0]:best=(key,th,rows,total)
    (_,global_th),thresholds,rows,total=best;current=f1(total)
    if per_type:
        for step,radius in ((.25,2.),(.05,.4),(.01,.08)):
            for _ in range(2):
                changed=False
                for t in range(len(types)):
                    affected=[sid for sid in ids if any(r[0]==t for r in candidates[sid])]
                    fixed=total-sum((rows[sid] for sid in affected),np.zeros(3,dtype=np.int64));best=None
                    for value in np.arange(thresholds[t]-radius,thresholds[t]+radius+step*.49,step):
                        trial=thresholds.copy();trial[t]=value
                        updates={sid:count(sid,trial) for sid in affected};newtotal=fixed+sum(updates.values(),np.zeros(3,dtype=np.int64))
                        new=f1(newtotal)
                        if new>current+1e-12:best=(value,updates,newtotal);current=new
                    if best is not None:
                        thresholds[t],updates,total=best;rows.update(updates);changed=True
                if not changed:break
    return {'thresholds':dict(zip(types,thresholds.tolist())),'strategy':strategy,'global_threshold':global_th,
        'fit_micro_f1':current,'fit_sentences':len(ids),'selection_metric':'Micro F1','eligible_in_refinement':types if per_type else []}


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    header,records=old.read_cache(FROZEN/'caches/old.jsonl.gz');types=header['types']
    samples=old.read_samples(ROOT/'data/dev2.txt',types);ids=list(range(len(samples)));outer=np.asarray(old.folds_for(samples))
    candidates={sid:[(t,a,b,score/8.) for t,a,b,score,penalty in records[sid]] for sid in ids}
    spec={'created_utc':datetime.now(timezone.utc).isoformat(),'checkpoint':header['checkpoint'],
        'cache_sha256':old.sha256(FROZEN/'caches/old.jsonl.gz'),
        'scores':'Cached GP scores divided by 8 to native logits; no boundary penalty',
        'cases':['raw0_all','global_all','type_all','type_interval'],
        'global_grid':{'lower':-4,'upper':4,'step':.25},'type_refinement':[[.25,2.],[.05,.4],[.01,.08]],
        'refinement_passes_per_level':2,'types_refined':types,'folds':'frozen text hash',
        'scope':'Fixed cached candidate pool, not unrestricted full-logit threshold search; no dev-selected checkpoint uncertainty correction',
        'test_accessed':False,'neural_training':False,'script_sha256':old.sha256(__file__)}
    old.write_json(OUT/'PLAN.json',spec);report={'plan':spec,'cases':{}}
    for name in spec['cases']:
        folder=OUT/name;folder.mkdir();predictions={'full_dev':{},'crossfit':{}};fits={}
        for part in ('full','held0','held1'):
            fit_ids=ids if part=='full' else np.flatnonzero(outer!=int(part[-1])).tolist()
            eval_ids=ids if part=='full' else np.flatnonzero(outer==int(part[-1])).tolist()
            if name=='raw0_all':decoder={'thresholds':{t:0. for t in types},'strategy':'all'}
            else:decoder=fit(samples,candidates,types,fit_ids,'weighted_interval' if name=='type_interval' else 'all',name!='global_all')
            th=[decoder['thresholds'][t] for t in types]
            pred={sid:decode_sentence(candidates[sid],th,decoder['strategy']) for sid in eval_ids}
            mode='full_dev' if part=='full' else 'crossfit';predictions[mode].update(pred);fits[part]=decoder
            old.write_json(folder/f'decoder_{part}.json',decoder)
            print(name,part,old.subset_score(samples,pred,types,eval_ids)['micro']['f1'],flush=True)
        for mode,pred in predictions.items():old.write_predictions(folder/f'{mode}_predictions.jsonl',samples,pred,types)
        report['cases'][name]={'fits':fits,'reports':{m:old.score_predictions(samples,p,types) for m,p in predictions.items()}}
        old.write_json(OUT/'RESULT.json',report)
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
