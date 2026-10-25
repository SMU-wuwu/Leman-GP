"""Frozen dev inference cost. No optimizer, fitting, training or test reads.

Same single RTX4090, batch16, BF16, CPU threads4, all 5,963 dev sentences.
The timed pipeline starts with CPU window records and host-resident weights;
includes weight transfer, collation, neural candidates, CPU features/HGB/decode,
and final JSONL writing. Disk startup and text preprocessing are separate.
"""
import os
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
from pathlib import Path
from collections import defaultdict
import argparse
import gc
import gzip
import hashlib
import json
import sys
import time
import copy
import numpy as np
import torch
import joblib

ROOT = Path('/root/data/WorkSpace')
FROZEN = ROOT/'output/leman_final_v2_20261019'
OUT = ROOT/'output/manuscript_revision_20261020/efficiency'
sys.path[:0] = [str(ROOT/'leman_gp_fgm_seed46_dual4090'), str(ROOT),
                str(FROZEN/'source_snapshot')]
from config import Config
from model import LemanGP
from data import load_split, Collator, prepare_split
from evaluate import score_predictions
from context_priors_v22 import candidate_features
from predict_hgb_v22 import decode


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def read_cache(path):
    with gzip.open(path, 'rt') as f:
        header = json.loads(next(f))
        return header, {r['sid']:r['rows'] for r in map(json.loads, f)}


def write_predictions(path, samples, pred, types):
    with Path(path).open('w') as f:
        for sid, (text, _) in enumerate(samples):
            f.write(json.dumps({'sid':sid, 'text':text, 'predicted':[
                {'type':types[t], 'start':a, 'end':b, 'text':text[a:b]}
                for t,a,b in sorted(pred[sid])]}, ensure_ascii=False)+'\n')


def read_predictions(path, types):
    return {r['sid']:{(types.index(e['type']),e['start'],e['end']) for e in r['predicted']}
            for r in map(json.loads, Path(path).read_text().splitlines())}


def feature_matrix(records, samples, types, gaz, priors):
    keys, rows = [], []
    for sid, (text, _) in enumerate(samples):
        models = []
        for source in records:
            merged = {}
            for typ,a,b,score,penalty in source[sid]:
                k = (typ,a,b)
                merged[k] = max(merged.get(k,(-float('inf'),-float('inf'))), (score,penalty))
            models.append(merged)
        for typ,a,b in sorted(set().union(*models)):
            count = gaz['surfaces'].get(text[a:b], {})
            same = count.get(types[typ],0)
            extra = (np.log1p(b-a),np.log1p(same),np.log1p(sum(count.values())-same))
            row = []
            for model in models:
                score,penalty = model.get((typ,a,b),(-4.,-16.))
                row.extend((score/8.,penalty,np.exp(penalty),*extra))
            keys.append((sid,typ,a,b));rows.append(row)
    base = np.asarray(rows,dtype=np.float64)
    ka = np.asarray(keys,dtype=np.int32)
    n = len(records)
    gp,boundary = base[:,:6*n:6],base[:,2:6*n:6]
    consensus = np.column_stack((gp.max(1),gp.mean(1),gp.std(1),np.quantile(gp,.25,axis=1),
        np.quantile(gp,.75,axis=1),(gp>0).mean(1),boundary.mean(1),boundary.std(1)))
    matrix = np.column_stack((base,np.eye(len(types))[ka[:,1]],
                              candidate_features(ka,samples,priors),consensus))
    return keys,matrix


def postprocess(records, samples, types, gaz, priors, fitted, output):
    keys,matrix = feature_matrix(records,samples,types,gaz,priors)
    probabilities = np.clip(fitted['hgb'].predict_proba(matrix)[:,1],1e-7,1-1e-7)
    scores = np.log(probabilities)-np.log1p(-probabilities)
    rows = {sid:[] for sid in range(len(samples))}
    for (sid,t,a,b),z in zip(keys,scores):
        if z > -4.:rows[sid].append((t,a,b,float(z)))
    dec = fitted['decoder'];th = [dec['thresholds'][t] for t in types]
    pred = {sid:decode(rows[sid],th,dec['strategy']) for sid in rows}
    write_predictions(output,samples,pred,types)
    return pred,len(keys)


@torch.inference_mode()
def candidates(model,dataset,device,floor,batch_size=16,limit=None,raw=False):
    model.eval();records={sid:[] for sid in range(len(dataset.samples))};state={}
    def boundary_hook(module,args):
        with torch.autocast('cuda',enabled=False):
            logits=model.boundary(args[0].float()).reshape(*args[0].shape[:2],model.num_types,2)
            state['boundary']=torch.nn.functional.logsigmoid(logits)
    hook = None if raw else model.attention.register_forward_pre_hook(boundary_hook)
    collate=Collator(dataset.types,dataset.pad_id)
    length=min(len(dataset),limit) if limit else len(dataset)
    try:
        for start in range(0,length,batch_size):
            batch=collate([dataset[i] for i in range(start,min(start+batch_size,length))])
            meta=batch.pop('metadata');batch.pop('labels')
            gpu={k:v.to(device) for k,v in batch.items()}
            with torch.autocast('cuda',dtype=torch.bfloat16):scores=model(**gpu)
            boundary=None if raw else state.pop('boundary')
            for typ in range(scores.shape[1]):
                plane=scores[:,typ]*8.
                ids=torch.nonzero(plane>floor,as_tuple=False);row,a,b=ids.T
                penalty=torch.zeros_like(plane[row,a,b]) if raw else .5*(boundary[row,a,typ,0]+boundary[row,b,typ,1])
                values=torch.stack((plane[row,a,b],penalty),-1).float().cpu().tolist()
                for (row,a,b),(score,penalty) in zip(ids.cpu().tolist(),values):
                    sid,offset,n=meta[row]
                    if 0<=a<=b<n:records[sid].append((typ,offset+a,offset+b+1,score,penalty))
            del gpu,scores
    finally:
        if hook is not None:hook.remove()
    return records


def main():
    p=argparse.ArgumentParser();p.add_argument('--repeats',type=int,default=1);args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    torch.cuda.set_device(0);torch.set_num_threads(4);device=torch.device('cuda:0')
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
    torch.backends.cudnn.benchmark=False
    names=['old','fgm43','fgm44','ema43','fgm46','soup2','soup3','parent','v4ema','v2ema','rdrop','fgm47']
    lineage=json.loads((OUT.parent/'CANDIDATE_LINEAGE.json').read_text())['sources']
    start=time.perf_counter()
    run=ROOT/'output/leman_finetune_v4_seed42'
    dataset=load_split(run/'cache/dev_d7ab938f313842cd.pkl')
    types=dataset.types;samples=dataset.samples
    gaz=json.loads((FROZEN/'v6_train_gazetteer.json').read_text())
    priors=joblib.load(FROZEN/'train_boundary_priors.joblib')
    assets={}
    for n in [1,3,6,12]:
        folder=(FROZEN/'wide_context6_consensus8' if n==12 else
            ROOT/f'output/manuscript_revision_20261019/source_scale/sources{n}_hgb')
        assets[n]={'hgb':joblib.load(folder/'hgb_full_dev.joblib'),
                   'decoder':json.loads((folder/'decoder_full.json').read_text()),'folder':str(folder)}
    setup_seconds=time.perf_counter()-start
    original={n:read_cache(FROZEN/'caches'/f'{n}.jsonl.gz') for n in names}
    # Confirm this generic n-source predictor reproduces every frozen full-dev output.
    replay={}
    for n in [1,3,6,12]:
        pred,count=postprocess([original[k][1] for k in names[:n]],samples,types,gaz,priors,
                              assets[n],OUT/f'replay_sources{n}.jsonl')
        expected=read_predictions(Path(assets[n]['folder'])/'full_dev_predictions.jsonl',types)
        assert pred==expected,f'Frozen decoder replay failed: {n}'
        replay[str(n)]={'predictions_exact':True,'candidate_rows':count}
    save(OUT/'FROZEN_REPLAY.json',replay)
    print('All four frozen CPU decoders replay exactly',flush=True)
    weights={};load_records={};model=None;initial_cfg=None
    for name in names:
        t=time.perf_counter();path=Path(lineage[name]['path'])
        checkpoint=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        state=checkpoint['model']
        if name in ['ema43','v4ema','v2ema']:
            shadow=checkpoint['ema']['shadow'];state={k:shadow.get(k,v) for k,v in state.items()}
        # Materialize just inference tensors in RAM; optimizer state stays on disk.
        weights[name]={k:v.detach().clone() for k,v in state.items()}
        if model is None:
            cfg=Config()
            for k,v in checkpoint['config'].items():setattr(cfg,k,v)
            initial_cfg=copy.deepcopy(cfg)
            model=LemanGP(cfg,len(types),checkpoint['vocab_size'],char_vocab_size=checkpoint['char_vocab_size'],
                          encoder_config=checkpoint['encoder_config'],pretrained=False).to(device).eval()
        load_records[name]={'path':str(path),'startup_seconds':time.perf_counter()-t,
            'inference_tensor_bytes':sum(v.numel()*v.element_size() for v in weights[name].values()),
            'checkpoint_bytes':path.stat().st_size,'frozen_checkpoint_sha256':lineage[name]['sha256'],
            'floor':original[name][0].get('floor',original[name][0].get('policy',{}).get('floor'))}
        assert load_records[name]['floor'] is not None
        print('Loaded',name,round(load_records[name]['startup_seconds'],2),flush=True)
        del checkpoint,state;gc.collect()
    model.load_state_dict(weights['old'],strict=True)
    candidates(model,dataset,device,-4.,limit=64)
    torch.cuda.synchronize()
    save(OUT/'PLAN.json',{'names':names,'sentences':len(samples),'windows':len(dataset),'batch_size':16,
        'devices_used':1,'available_devices':torch.cuda.device_count(),'gpu':torch.cuda.get_device_name(),
        'torch':torch.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
        'tf32':True,'precision':'BF16 encoder, original FP32 custom layers','cpu_threads':4,
        'workers':0,'warmup_windows':64,'repeats':args.repeats,'setup_seconds':setup_seconds,
        'timing_boundary':__doc__,'weight_loads':load_records,'test_read':False,'training':False,
        'dev_sha256':digest(ROOT/'data/dev2.txt'),'script_sha256':digest(__file__),
        'model_source_sha256':digest(ROOT/'leman_gp_fgm_seed46_dual4090/model.py')})
    results=[]
    for case in ['single_raw',1,3,6,12]:
        active=['old'] if case=='single_raw' else names[:case]
        for rep in range(args.repeats):
            print('BEGIN',case,'repeat',rep+1,flush=True)
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t0=time.perf_counter()
            records=[];stages=[]
            for name in active:
                t=time.perf_counter();model.load_state_dict(weights[name],strict=True);torch.cuda.synchronize()
                weight_transfer=time.perf_counter()-t;t=time.perf_counter()
                rows=candidates(model,dataset,device,0. if case=='single_raw' else load_records[name]['floor'],raw=case=='single_raw')
                torch.cuda.synchronize();elapsed=time.perf_counter()-t
                records.append(rows);stages.append({'source':name,'weight_transfer_seconds':weight_transfer,
                    'candidates_seconds':elapsed,'rows':sum(map(len,rows.values()))})
                print(case,name,'candidates',round(elapsed,3),flush=True)
            t=time.perf_counter();output=OUT/f'{case}_repeat{rep+1}_predictions.jsonl'
            if case=='single_raw':
                pred={sid:{(t,a,b) for t,a,b,_,_ in rows} for sid,rows in records[0].items()}
                write_predictions(output,samples,pred,types);count=sum(map(len,pred.values()))
            else:pred,count=postprocess(records,samples,types,gaz,priors,assets[case],output)
            torch.cuda.synchronize();post=time.perf_counter()-t;total=time.perf_counter()-t0
            row={'case':case,'repeat':rep+1,'pipeline_seconds':total,'sentences_per_second':len(samples)/total,
                'postprocess_and_write_seconds':post,'stages':stages,'candidate_rows':count,
                'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
                'dev':score_predictions(samples,pred,types),'prediction_sha256':digest(output)}
            if case!='single_raw':
                expected=read_predictions(Path(assets[case]['folder'])/'full_dev_predictions.jsonl',types)
                row['comparison_to_frozen_predictions']={'changed_sentences':sum(pred[i]!=expected[i] for i in pred),
                    'added_spans':sum(len(pred[i]-expected[i]) for i in pred),
                    'removed_spans':sum(len(expected[i]-pred[i]) for i in pred)}
            results.append(row);save(OUT/'RESULT.json',results)
            print('END',case,round(total,3),row['dev']['micro']['f1'],flush=True)
            del records;gc.collect()
    # Measure uncached tokenizer/window/lexicon preparation separately, using the
    # same dev and frozen train lexicon. This also creates a reusable CPU cache.
    initial_cfg.output_dir=str(OUT/'fresh_preprocessing')
    lex=json.loads((run/'lexicon.json').read_text());t=time.perf_counter()
    cache=prepare_split(initial_cfg,'dev',lex,types);preprocess_seconds=time.perf_counter()-t
    fresh=load_split(cache)
    assert fresh.samples==samples and len(fresh)==len(dataset)
    save(OUT/'PREPROCESSING.json',{'seconds':preprocess_seconds,'cache':str(cache),
        'included':'read dev, tokenizer load, tokenization, lexicon matching, windows, write CPU cache',
        'sentences':len(samples),'windows':len(dataset),'preprocessing_workers':initial_cfg.preprocessing_workers})
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
