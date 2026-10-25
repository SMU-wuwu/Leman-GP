"""Paired mechanistic interventions: shared frozen encoder, no optimizer."""
import argparse, hashlib, json, math, sys, types as pytypes
from pathlib import Path
import torch
from torch.nn import functional as F
from types import SimpleNamespace
from collections import defaultdict
from torch.utils.data import DataLoader, Subset
from itertools import chain
from data import Collator
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'short_ablation'))
from config import Config
from model import LemanGP
from data import load_split, entity_types, write_json, sha256
from train import prepare_resources
from evaluate import evaluation_loader, score_predictions, write_predictions
from runtime import amp_context, to_device, seed_everything
from parent_model import safe_l1_cdist, boundary_loss

CONDITIONS=['baseline','attention_identity','attention_norm_only','attention_half_context','attention_uniform','attention_dot','lexicon_norm_only','lexicon_drop_top','lexicon_drop_bottom']

def state_digest(model):
    h=hashlib.sha256()
    for n,t in model.state_dict().items():
        h.update(n.encode()); h.update(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def attention_forward(a,x,mask,mode):
    if mode=='attention_identity':return x*mask.unsqueeze(-1)
    if mode=='attention_norm_only':return a.norm(x)*mask.unsqueeze(-1)
    b,n,h=x.shape
    def split(layer):return layer(x).reshape(b,n,a.heads,a.head_dim).transpose(1,2)
    q,k,v=a.qnorm(split(a.q)),a.knorm(split(a.k)),split(a.v)
    tau=F.softplus(a.raw_temperature).view(1,a.heads,1,1)+1e-4
    if mode=='attention_uniform':scores=torch.zeros((b,a.heads,n,n),device=x.device)
    elif mode=='attention_dot':scores=(q@k.transpose(-1,-2))*tau/math.sqrt(a.head_dim)
    else:scores=-safe_l1_cdist(q,k)*tau/math.sqrt(a.head_dim)
    w=scores.masked_fill(~mask[:,None,None,:],-1e30).softmax(-1)
    ctx=a.out((a.dropout(w)@v).transpose(1,2).reshape(b,n,h))
    scale=.5 if mode=='attention_half_context' else 1.
    return a.norm(x+scale*a.dropout(ctx))*mask.unsqueeze(-1)

def lexicon_forward(a,x,words,bmes,mode,counters):
    valid=words.ne(0); had=valid.any(-1,keepdim=True)
    if mode=='lexicon_norm_only':return torch.where(had,a.norm(x),x)
    candidates=a.projection(torch.cat([a.word_embedding(words),a.boundary_embedding(bmes)],-1))
    scores=(a.bilinear(x).unsqueeze(2)*candidates).sum(-1)/math.sqrt(x.shape[-1])
    oldweights=scores.masked_fill(~valid,-1e30).softmax(-1)*valid
    if mode in ('lexicon_drop_top','lexicon_drop_bottom'):
        eligible=valid.sum(-1)>=2
        idx=(scores.masked_fill(~valid,-1e30).argmax(-1) if mode.endswith('top') else scores.masked_fill(~valid,1e30).argmin(-1))
        removed=torch.zeros_like(valid).scatter_(-1,idx.unsqueeze(-1),True)&eligible.unsqueeze(-1)
        assert torch.equal(removed.sum(-1),eligible.long())
        counters['positions']+=int(eligible.sum());counters['removed_mass']+=float((oldweights*removed).sum())
        valid=valid&~removed
    w=scores.masked_fill(~valid,-1e30).softmax(-1)*valid
    ctx=(w.unsqueeze(-1)*candidates).sum(2)
    gate=torch.sigmoid(a.gate(torch.cat([x,ctx],-1)))
    return torch.where(had,a.norm(x+a.dropout(gate*ctx)),x)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='/root/data/WorkSpace');p.add_argument('--out',type=Path,required=True);p.add_argument('--checkpoint',choices=['final','parent'],default='final');a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    assert not (a.out/'COMPLETE.json').exists(),'Use a fresh directory; completed results are immutable.'
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    device=torch.device('cuda:0');c=Config.make(a.root,'F',42);c.workers=0;c.eval_batch_size=16
    run=Path(c.output_dir);identity=json.loads((run/'run_config.json').read_text())['input_identity']
    types=entity_types(Path(c.data_dir)/'label.txt');paths=prepare_resources(c,identity,types);dev=load_split(paths[1])
    checkpoint=run/'final_ema.pt' if a.checkpoint=='final' else Path(c.warm_start)
    ck=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    lex=json.loads((Path(c.resource_dir)/'lexicon.json').read_text());seed_everything(42)
    m=LemanGP(c,len(types),len(lex['words']),char_vocab_size=len(lex['characters']),encoder_config=ck['encoder_config'],pretrained=False)
    m.load_state_dict(ck['model'],strict=True);del ck;m.to(device).eval()
    before=state_digest(m)
    write_json(a.out/'PROTOCOL.json',{'conditions':CONDITIONS,'checkpoint':str(checkpoint),'checkpoint_sha256':sha256(checkpoint),'split':'dev','selection':'all 5963 sentences, no result-based filtering','threshold':0,'optimizer_steps':0,'training':False,'lexicon_removal':'one candidate per character with >=2 valid candidates; rank from model, no gold access','batch_layout':'original two rank-strided partitions, 16 windows per batch; sequential on one GPU','encoder_cache':'same batch tensor for every condition; per-batch only','source_sha256':sha256(__file__)})
    attn_original=m.attention.forward;lex_original=m.adapter.forward;encoder_original=m.encoder.forward
    loaders=[DataLoader(Subset(dev,list(range(rank,len(dev),2))),batch_size=c.eval_batch_size,collate_fn=Collator(dev.types,dev.pad_id),num_workers=0) for rank in (0,1)]
    predictions={k:defaultdict(set) for k in CONDITIONS};counters={k:{'positions':0,'removed_mass':0.} for k in CONDITIONS};qa={}
    with torch.inference_mode():
        for batchno,batch in enumerate(chain.from_iterable(loaders)):
            inp=to_device(batch,device);inp.pop('labels',None)
            with amp_context(device):hidden=encoder_original(input_ids=inp['input_ids'],attention_mask=inp['token_mask']).last_hidden_state
            m.encoder.forward=lambda **kw:SimpleNamespace(last_hidden_state=hidden)
            if batchno==0:
                with amp_context(device):original=m(**inp); repeat=m(**inp)
                qa['original_repeat_max_abs']=float((original-repeat).abs().max())
                print('repeat QA',qa,flush=True)
                m.attention.forward=lambda x,mask:attention_forward(m.attention,x,mask,'baseline')
                m.adapter.forward=lambda x,w,b:lexicon_forward(m.adapter,x,w,b,'baseline',{'positions':0,'removed_mass':0.})
                with amp_context(device):reimplemented=m(**inp)
                qa['baseline_reimplementation_max_abs']=float((original-reimplemented).abs().max())
                qa['baseline_reimplementation_changed_predictions']=int(((original>0)!=(reimplemented>0)).sum())
                print('reimplementation QA',qa,flush=True)
                assert qa['baseline_reimplementation_changed_predictions']==0 and qa['baseline_reimplementation_max_abs']<0.01,qa
                m.attention.forward=attn_original;m.adapter.forward=lex_original
            for mode in CONDITIONS:
                m.attention.forward=(lambda x,mask,mode=mode:attention_forward(m.attention,x,mask,mode)) if mode.startswith('attention_') else attn_original
                m.adapter.forward=(lambda x,w,b,mode=mode:lexicon_forward(m.adapter,x,w,b,mode,counters[mode])) if mode.startswith('lexicon_') else lex_original
                with amp_context(device):scores=m(**inp)
                assert torch.isfinite(scores).all()
                indices=torch.nonzero(scores>0,as_tuple=False).cpu().tolist()
                for row,t,s,e in indices:
                    sid,offset,n=batch['metadata'][row]
                    assert 0<=s<=e<n
                    predictions[mode][sid].add((t,offset+s,offset+e+1))
            m.attention.forward=attn_original;m.adapter.forward=lex_original
            if batchno%25==0:print('batch',batchno,'/',sum(map(len,loaders)),flush=True)
    m.encoder.forward=encoder_original
    assert counters['lexicon_drop_top']['positions']==counters['lexicon_drop_bottom']['positions']
    after=state_digest(m);assert before==after
    records={}
    for mode in CONDITIONS:
        out=a.out/mode;out.mkdir(exist_ok=True);records[mode]=score_predictions(dev.samples,predictions[mode],types)
        write_json(out/'dev_results.json',records[mode]);write_predictions(out/'dev_predictions.jsonl',dev.samples,predictions[mode],types)
        print(mode,records[mode]['micro'],flush=True)
    expected=run/'epoch_005_ema/dev_predictions.jsonl' if a.checkpoint=='final' else Path(a.root)/'output/leman_short_mechanism_v1/fixed_interventions/F/dev_predictions.jsonl'
    def readpred(path):
        return {r['sid']:{(x['type'],x['start'],x['end']) for x in r['predicted']} for r in map(json.loads,path.read_text().splitlines())}
    old=readpred(expected);new=readpred(a.out/'baseline/dev_predictions.jsonl')
    qa['baseline_predictions_identical_to_saved']=old==new
    changes=[{'sid':i,'saved_only':sorted(old[i]-new[i]),'current_only':sorted(new[i]-old[i])} for i in old if old[i]!=new[i]]
    qa['baseline_numerical_differences']=changes
    write_json(a.out/'BASELINE_RECONCILIATION.json',qa)
    assert sum(len(r['saved_only'])+len(r['current_only']) for r in changes)<=20,qa
    write_json(a.out/'COMPLETE.json',{'conditions':len(CONDITIONS),'metrics':records,'counters':counters,'qa':qa,'weight_state_before':before,'weight_state_after':after,'weights_unchanged':before==after,'optimizer_steps':0,'test_read':False})

if __name__=='__main__':main()
