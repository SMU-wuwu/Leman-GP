"""Frozen endpoint evaluation on test; resumable predictions, no fitting or optimizer."""
import argparse, dataclasses, gc, json, os, sys, time
from pathlib import Path
from collections import defaultdict
ROOT=Path('/root/data/WorkSpace')
sys.path.insert(0,str(ROOT/'leman_short_mechanism_v1'))
import numpy as np
import torch
from torch.utils.data import DataLoader
from config import Config, VARIANTS
from model import LemanGP
from data import entity_types, prepare_split, load_split, sha256, write_json
from controlled_data import Collator, label_names, extract_spans
from evaluate import score_predictions, write_predictions

ORDER=['F','no_lexicon','no_bmes','no_recurrence','no_gru_linear','no_rope','no_attention','max_words3','max_words5','gp_reset','crf','uniform_lexicon','fixed_gate','no_char','no_boundary','fgm_control','no_fgm','clean_two_pass','weighted_control','unit_loss','encoder_gp']

def load_model(variant):
 c=Config.make(ROOT,variant,42);run=Path(c.output_dir);cp=run/'final_ema.pt'
 endpoint=json.loads((run/'PRIMARY_ENDPOINT.json').read_text())
 digest=sha256(cp);assert digest==endpoint['sha256'],(variant,'checkpoint identity')
 ck=torch.load(cp,map_location='cpu',mmap=True,weights_only=False)
 for k,v in ck['config'].items():
  if hasattr(c,k):setattr(c,k,v)
 c.workers=0
 lex=json.loads((Path(c.resource_dir)/'lexicon.json').read_text())
 types=entity_types(Path(c.data_dir)/'label.txt')
 # Construct on CPU so rotary and other persistent buffers are materialized.
 m=LemanGP(c,len(types),len(lex['words']),char_vocab_size=len(lex['characters']),encoder_config=ck['encoder_config'],pretrained=False)
 m.load_state_dict(ck['model'],strict=True,assign=True)
 del ck
 m.eval();m.requires_grad_(False)
 return c,m,lex,types,digest

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'output/test_ablation_20261021');p.add_argument('--variants',nargs='+',default=ORDER);p.add_argument('--batch-size',type=int,default=4);p.add_argument('--limit-batches',type=int,default=0);a=p.parse_args()
 torch.set_num_threads(int(os.environ.get('LEMAN_TEST_THREADS','2')));torch.set_num_interop_threads(1)
 a.out.mkdir(parents=True,exist_ok=True)
 for v in a.variants:
  assert v in ORDER
  out=a.out/v;out.mkdir(exist_ok=True)
  if (out/'COMPLETE.json').exists():print('SKIP completed',v,flush=True);continue
  c,m,lex,types,digest=load_model(v)
  cache_cfg=dataclasses.replace(c,output_dir=str(a.out/'resources'))
  ds=load_split(prepare_split(cache_cfg,'test',lex,types))
  assert sum(len(g) for _,g in ds.samples)==20524
  identity={'split':'test','test_sha256':sha256(Path(c.data_dir)/'test2.txt'),'checkpoint_sha256':digest,'checkpoint':'final_ema.pt','threshold':c.threshold,'precision':'fp32','device':'cpu','optimizer_steps':0,'training':False,'variant':v,'sentences':len(ds.samples),'windows':len(ds),'gold':20524,'batch_size':a.batch_size,'order':'length-sorted windows; all windows retained','source_sha256':sha256(__file__)}
  pr=out/'PROTOCOL.json'
  if pr.exists():assert json.loads(pr.read_text())==identity,'Resume protocol changed'
  else:write_json(pr,identity)
  state=out/'PROGRESS.json';pred=defaultdict(set);done=set();emissions={}
  if state.exists():
   s=json.loads(state.read_text());done=set(s['done_window_ids']);pred.update({int(k):set(map(tuple,z)) for k,z in s.get('predictions',{}).items()})
   if v=='crf' and (out/'emissions.pt').exists():emissions=torch.load(out/'emissions.pt',weights_only=False)
  order=sorted(range(len(ds)),key=lambda i:(ds.items[i]['length'],i))
  remaining=[i for i in order if i not in done];started=time.time();steps=0
  collate=Collator(types,ds.pad_id,v)
  with torch.inference_mode():
   for start in range(0,len(remaining),a.batch_size):
    ids=remaining[start:start+a.batch_size];batch=collate([ds.items[i] for i in ids]);inputs={k:z for k,z in batch.items() if torch.is_tensor(z)}
    inputs.pop('labels',None);inputs.pop('crf_tags',None)
    scores=m(**inputs)
    if v=='crf':
     values=scores.numpy()
     for row,(sid,off,n) in enumerate(batch['metadata']):
      if sid not in emissions:emissions[sid]=[np.zeros((len(ds.samples[sid][0]),values.shape[-1]),np.float32),np.zeros(len(ds.samples[sid][0]),np.int32)]
      emissions[sid][0][off:off+n]+=values[row,:n];emissions[sid][1][off:off+n]+=1
    else:
     for row,t,b,e in torch.nonzero(scores>c.threshold,as_tuple=False).tolist():
      sid,off,n=batch['metadata'][row];assert 0<=b<=e<n
      pred[sid].add((t,off+b,off+e+1))
    done.update(ids);steps+=1
    if steps%10==0 or start+a.batch_size>=len(remaining) or (a.limit_batches and steps>=a.limit_batches):
     if v=='crf':torch.save(emissions,out/'emissions.pt')
     write_json(state,{'done_window_ids':sorted(done),'predictions':{str(i):sorted(z) for i,z in pred.items()},'seconds_this_session':time.time()-started})
     print(v,'windows',len(done),'/',len(ds),'seconds',round(time.time()-started,1),'recent_windows_per_second',round((start+len(ids))/(time.time()-started),3),flush=True)
    if a.limit_batches and steps>=a.limit_batches:break
  if len(done)!=len(ds):print('BENCHMARK ONLY',v,flush=True);return
  if v=='crf':
   names=label_names(types);mapping={t:i for i,t in enumerate(types)}
   for sid,(total,count) in emissions.items():
    assert np.all(count>0)
    logits=torch.from_numpy(total/count[:,None]).float()[None];mask=torch.ones(logits.shape[:2],dtype=torch.bool)
    tags=m.crf.decode(logits,mask)[0]
    pred[sid]={(mapping[t],b,e) for t,b,e in extract_spans([names[i] for i in tags])}
  result=score_predictions(ds.samples,pred,types);result.update(split='test',checkpoint_sha256=digest,threshold=c.threshold,device='cpu',precision='fp32',optimizer_steps=0,coverage=ds.stats)
  write_predictions(out/'test_predictions.jsonl',ds.samples,pred,types);write_json(out/'test_results.json',result)
  write_json(out/'COMPLETE.json',{'split':'test','all_windows_evaluated':True,'weights_unchanged':all(p.grad is None for p in m.parameters()),'optimizer_steps':0,'metric':result['micro'],'prediction_sha256':sha256(out/'test_predictions.jsonl'),'checkpoint_sha256':digest})
  print('COMPLETE',v,result['micro'],flush=True)
  del m,ds,emissions,pred;gc.collect()
if __name__=='__main__':main()
