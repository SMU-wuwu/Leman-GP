"""Generate new-source candidates only; no optimizer, backward or fit."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='4'
os.environ['TOKENIZERS_PARALLELISM']='false'
import argparse,copy,gc,gzip,json,sys,time
from pathlib import Path
from settings import ROOT,OUT,SOURCES
from evidence_core import sha256,write_json,read_bio,score
sys.path.insert(0,str(ROOT/'leman_short_mechanism_v1'))
import torch
from config import Config
from model import LemanGP
from data import Collator,entity_types,prepare_split,load_split

def assets(name,split,preflight=False):
    record=SOURCES[name];path=Path(record['path'])
    assert sha256(path)==record['sha256'],name
    ck=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    c=Config()
    for k,v in ck['config'].items():setattr(c,k,v)
    c.root=str(ROOT);c.data_dir=str(ROOT/'data');c.encoder_dir=str(ROOT/'model')
    c.preprocessing_workers=4
    # Reuse the immutable dev cache; new test preprocessing lives in this run.
    c.output_dir=str(ROOT/'output/leman_controlled_ablation_v1/resources' if split=='dev' else OUT/'preprocessing')
    lexpath=ROOT/'output/leman_controlled_ablation_v1/resources/lexicon.json'
    assert sha256(lexpath)==ck['lexicon_sha256']
    lex=json.loads(lexpath.read_text());types=entity_types(ROOT/'data/label.txt');assert ck['types']==types
    data=load_split(prepare_split(c,split,lex,types))
    independent=read_bio(ROOT/'data'/f'{split}2.txt')
    assert len(data.samples)==len(independent)
    for (ta,ga),(tb,gb) in zip(data.samples,independent):
        assert ta==tb and {(types[t],a,b) for t,a,b in ga}==gb
    model=LemanGP(c,len(types),len(lex['words']),char_vocab_size=len(lex['characters']),
        encoder_config=ck['encoder_config'],pretrained=False)
    model.load_state_dict(ck['model'],strict=True);model.eval()
    identity=dict(source=name,checkpoint=record,split=split,split_sha256=sha256(ROOT/'data'/f'{split}2.txt'),
        types=types,lexicon_sha256=sha256(lexpath),weight_source=ck.get('weight_source'),variant=c.variant,
        floor=-32.,batch_size=16,precision='BF16 encoder; FP32 custom layers; TF32 enabled',
        model_source_sha256={p.name:sha256(p) for p in (ROOT/'leman_short_mechanism_v1').glob('*.py')},
        inference_source_sha256=sha256(__file__))
    del ck
    # Gold must not reach inference collation, neural modules or candidate filtering.
    data.samples=[(text,[]) for text,_ in data.samples]
    for item in data.items:item['spans']=[]
    return model,data,identity,independent

@torch.inference_mode()
def candidates(model,data):
    model.to('cuda:0');state={};collate=Collator(data.types,data.pad_id)
    def hook(module,args):
        with torch.autocast('cuda',enabled=False):
            logits=model.boundary(args[0].float()).reshape(*args[0].shape[:2],model.num_types,2)
            state['b']=torch.nn.functional.logsigmoid(logits)
    handle=model.attention.register_forward_pre_hook(hook)
    records={sid:[] for sid in range(len(data.samples))}
    try:
        for start in range(0,len(data),16):
            batch=collate([data[i] for i in range(start,min(start+16,len(data)))])
            meta=batch.pop('metadata');batch.pop('labels')
            batch={k:v.to('cuda:0') for k,v in batch.items()}
            with torch.autocast('cuda',dtype=torch.bfloat16):scores=model(**batch)
            bound=state.pop('b')
            for typ in range(scores.shape[1]):
                plane=scores[:,typ]*8.; ids=torch.nonzero(plane>-32.,as_tuple=False);row,a,b=ids.T
                penalty=.5*(bound[row,a,typ,0]+bound[row,b,typ,1])
                values=torch.stack((plane[row,a,b],penalty),-1).float().cpu().tolist()
                for (r,a,b),(z,p) in zip(ids.cpu().tolist(),values):
                    sid,off,n=meta[r]
                    if 0<=a<=b<n:records[sid].append((typ,off+a,off+b+1,z,p))
            if start%1024==0:print('PROGRESS',start,len(data),flush=True)
            del batch,scores,bound
    finally:handle.remove()
    return records

def main():
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['dev','test'],default='dev');p.add_argument('--preflight',action='store_true');a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    names=list(SOURCES)
    if a.split=='test':
        assert not a.preflight
        freeze=json.loads((OUT/'SELECTED_FREEZE.json').read_text())
        names=freeze['new_sources']
        for n,h in freeze['files'].items():assert sha256(OUT/n)==h,n
    if not a.preflight:
        import subprocess
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU occupied'
        torch.cuda.set_device(0);torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False);torch.backends.cudnn.benchmark=False
    for name in names:
        path=OUT/'candidates'/a.split/f'{name}.jsonl.gz';receipt=path.with_suffix('.receipt.json')
        if receipt.exists():assert sha256(path)==json.loads(receipt.read_text())['sha256'];continue
        print('LOAD',a.split,name,flush=True);t0=time.monotonic()
        model,data,identity,gold=assets(name,a.split,a.preflight)
        if a.preflight:
            write_json(OUT/f'PREFLIGHT_{name}.json',dict(identity=identity,strict_load=True,windows=len(data),sentences=len(gold)))
            del model,data;gc.collect();continue
        records=candidates(model,data)
        path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
        with gzip.open(tmp,'wt',encoding='utf-8') as f:
            f.write(json.dumps(identity)+'\n')
            for sid in sorted(records):f.write(json.dumps(dict(sid=sid,rows=records[sid]))+'\n')
        os.replace(tmp,path)
        z=dict(sha256=sha256(path),identity=identity,candidate_rows=sum(map(len,records.values())),seconds=time.monotonic()-t0)
        if a.split=='dev':
            raw={sid:{(identity['types'][t],b,e) for t,b,e,v,_ in rows if v>0.} for sid,rows in records.items()}
            z['raw_dev']=score(gold,raw,identity['types'])
            saved=json.loads((Path(SOURCES[name]['path']).parent/'training_results.json').read_text())['best_dev_micro_f1']
            z['saved_raw_dev_f1']=saved;z['raw_f1_delta']=z['raw_dev']['micro']['f1']-saved
            assert abs(z['raw_f1_delta'])<.001,'Unexpected numeric/protocol mismatch'
        write_json(receipt,z);print('COMPLETE',a.split,name,json.dumps(z.get('raw_dev',{}).get('micro',{})),flush=True)
        del model,data,records;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':main()
