"""Read-only probes on a hash-selected panel; never used for fitting or selection."""
import csv,json,hashlib,math
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader
from controlled_data import Collator
from paper_core import safe_l1_cdist
from runtime import rank,world_size,amp_context,to_device,rng_state,restore_rng,barrier
from data import write_json

def write_csv(path,rows):
    if not rows:return
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def panel_ids(dataset,n):
    return sorted(range(len(dataset.samples)),key=lambda i:(hashlib.sha256(('leman-mechanism-panel-v1\0'+dataset.samples[i][0]).encode()).hexdigest(),i))[:n]

@torch.inference_mode()
def probe(model,dataset,cfg,device,out,tag):
    barrier()
    if rank()!=0:
        barrier();return
    state=rng_state();was_training=model.training;model.eval()
    out=Path(out)/'mechanisms'/tag;out.mkdir(parents=True,exist_ok=True)
    ids=panel_ids(dataset,cfg.probe_sentences);selected=set(ids)
    windows=[x for x in dataset.items if x['sid'] in selected]
    trace={};handles=[]
    def hook(name):
        def save(module,inputs,output):
            trace[name]=(tuple(x.detach() if torch.is_tensor(x) else x for x in inputs),output.detach())
        return save
    for n in ('encoder','char_projection','adapter','sequence_norm','attention','boundary'):
        if not hasattr(model,n):continue
        if n=='encoder':
            def enc(m,i,o):trace['encoder']=((),o.last_hidden_state.detach())
            handles.append(model.encoder.register_forward_hook(enc))
        else:handles.append(getattr(model,n).register_forward_hook(hook(n)))
    lexicon=json.loads((Path(cfg.resource_dir)/'lexicon.json').read_text())
    word_vocab=lexicon['words'];word_lookup={w:i for i,w in enumerate(word_vocab)}
    slots=[];chars=[];heads=[];margins=[];boundary=defaultdict(lambda:[[],[]]);heat_saved=False
    try:
        loader=DataLoader(windows,batch_size=min(8,cfg.eval_batch_size),shuffle=False,num_workers=0,
            collate_fn=Collator(dataset.types,dataset.pad_id,cfg.variant),generator=torch.Generator().manual_seed(17001))
        for batch in loader:
            trace.clear();inputs=to_device(batch,device)
            with amp_context(device):loss,scores=model(**inputs)
            mask=inputs['char_mask'];labels=inputs['labels'];words=inputs['word_ids'];valid=words.ne(0)
            x=torch.bmm(inputs['char_pool'].float(),trace['encoder'][1].float())
            cres=torch.zeros_like(x)
            if 'char_projection' in trace:cres=model.char_scale.detach()*trace['char_projection'][1]
            character=x+cres
            lexw=gate=lexchange=None
            if hasattr(model,'adapter'):
                a=model.adapter
                be=a.boundary_embedding(inputs['boundary_ids'])
                if cfg.variant=='no_bmes':be=torch.zeros_like(be)
                cand=a.projection(torch.cat([a.word_embedding(words),be],-1))
                if cfg.variant=='uniform_lexicon':lexw=valid.float()/valid.sum(-1,keepdim=True).clamp_min(1)
                else:
                    ls=(a.bilinear(character).unsqueeze(2)*cand).sum(-1)/math.sqrt(character.shape[-1])
                    lexw=ls.masked_fill(~valid,-1e30).softmax(-1)*valid
                ctx=(lexw.unsqueeze(-1)*cand).sum(2)
                gate=(torch.full_like(character,.5) if cfg.variant=='fixed_gate' else
                      torch.sigmoid(a.gate(torch.cat([character,ctx],-1))))
                y=torch.where(valid.any(-1,keepdim=True),a.norm(character+gate*ctx),character)
                lexchange=(y-character).norm(dim=-1)/character.norm(dim=-1).clamp_min(1e-8)
            if 'attention' in trace and cfg.variant!='attention_norm_only':
                a=model.attention;seq=trace['attention'][0][0];b,l,d=seq.shape
                def split(layer):return layer(seq).reshape(b,l,a.heads,a.head_dim).transpose(1,2)
                q,k=a.qnorm(split(a.q)),a.knorm(split(a.k))
                tau=torch.nn.functional.softplus(a.raw_temperature)+1e-4
                affinity=(torch.zeros((b,a.heads,l,l),device=q.device) if cfg.variant=='uniform_attention' else q@k.transpose(-1,-2) if cfg.variant=='dot_attention' else -safe_l1_cdist(q,k))
                att=(affinity*tau[None,:,None,None]/math.sqrt(a.head_dim)).masked_fill(~mask[:,None,None,:],-1e30).softmax(-1)
                att_change=(trace['attention'][1]-seq).norm(dim=-1)/seq.norm(dim=-1).clamp_min(1e-8)
                for i,(sid,offset,n) in enumerate(batch['metadata']):
                    same=torch.zeros(n,n,dtype=torch.bool,device=device)
                    for typ,st,en in dataset.samples[sid][1]:
                        aa,bb=max(st-offset,0),min(en-offset,n)
                        if st>=offset and en<=offset+n:same[aa:bb,aa:bb]=True
                    entityq=same.any(-1);base=same.float().mean(-1)
                    dist=(torch.arange(n,device=device)[:,None]-torch.arange(n,device=device)[None,:]).abs()
                    for h in range(a.heads):
                        w=att[i,h,:n,:n]
                        mass=(w*same).sum(-1)
                        heads.append(dict(sid=sid,offset=offset,head=h,n_chars=n,temperature=float(tau[h]),
                            entropy=float(-(w*w.clamp_min(1e-20).log()).sum(-1).mean()),
                            mean_distance=float((w*dist).sum(-1).mean()),self_mass=float(w.diag().mean()),
                            entity_queries=int(entityq.sum()),within_entity_mass=float(mass[entityq].mean()) if entityq.any() else None,
                            uniform_within_entity_mass=float(base[entityq].mean()) if entityq.any() else None))
                    if not heat_saved:
                        np.savez_compressed(out/'attention_example.npz',weights=att[i,:,:n,:n].cpu().numpy())
                        write_json(out/'attention_example.json',{'sid':sid,'offset':offset,'text':dataset.samples[sid][0][offset:offset+n],
                            'selection':'first window of fixed hash panel; not selected by correctness'})
                        heat_saved=True
            else:
                att_change=((trace['attention'][1]-trace['attention'][0][0]).norm(dim=-1)/trace['attention'][0][0].norm(dim=-1).clamp_min(1e-8)) if 'attention' in trace else torch.full(mask.shape,float('nan'),device=device)
            if 'boundary' in trace:
                logits=trace['boundary'][1].reshape(*mask.shape,model.num_types,2)
                target=torch.stack((labels.any(-1),labels.any(-2)),-1).permute(0,2,1,3)
                for t,name in enumerate(dataset.types):
                    for j,side in enumerate(('start','end')):
                        boundary[(name,side)][0].extend(logits[:,:,t,j][mask].float().cpu().tolist())
                        boundary[(name,side)][1].extend(target[:,:,t,j][mask].int().cpu().tolist())
            for i,(sid,offset,n) in enumerate(batch['metadata']):
                truth=dataset.samples[sid][1]
                for j in range(n):
                    count=int(valid[i,j].sum());covered=any(a<=offset+j<b for t,a,b in truth)
                    for slot in range(words.shape[-1]):
                        wid=int(words[i,j,slot])
                        if wid:
                            slots.append(dict(sid=sid,offset=offset,position=offset+j,slot=slot,word_id=wid,word=word_vocab[wid],bmes=int(inputs['boundary_ids'][i,j,slot]),consumed_by_lexicon=int(lexw is not None),weight=float(lexw[i,j,slot]) if lexw is not None else None))
                    chars.append(dict(sid=sid,offset=offset,position=offset+j,char=dataset.samples[sid][0][offset+j],
                        gold_entity_char=int(covered),candidate_count=count,
                        lexicon_entropy=float(-(lexw[i,j]*lexw[i,j].clamp_min(1e-20).log()).sum()) if lexw is not None else None,
                        gate_mean=float(gate[i,j].mean()) if gate is not None and count else None,
                        lexicon_relative_change=float(lexchange[i,j]) if lexchange is not None else None,
                        char_relative_residual=float(cres[i,j].norm()/x[i,j].norm().clamp_min(1e-8)),
                        attention_relative_change=float(att_change[i,j]) if torch.isfinite(att_change[i,j]) else None))
                if cfg.variant!='crf':
                    for t,a,b in truth:
                        if not (offset<=a<b<=offset+n):continue
                        aa,bb=a-offset,b-offset-1;value=float(scores[i,t,aa,bb]);neighbors=[]
                        for sa in (aa-1,aa,aa+1):
                            for sb in (bb-1,bb,bb+1):
                                if 0<=sa<=sb<n and (sa,sb)!=(aa,bb) and not bool(labels[i,t,sa,sb]):
                                    neighbors.append(float(scores[i,t,sa,sb]))
                        word_id=word_lookup.get(dataset.samples[sid][0][a:b],-1)
                        present=[]
                        for pos0 in range(aa,bb+1):
                            role=3 if aa==bb else 0 if pos0==aa else 2 if pos0==bb else 1
                            present.append(bool(((words[i,pos0]==word_id)&(inputs['boundary_ids'][i,pos0]==role)).any()))
                        margins.append(dict(gold_word_in_task_vocab=int(word_id>=2),gold_word_available_all_characters=int(all(present)),lexicon_consumed=int(hasattr(model,'adapter')),sid=sid,offset=offset,type=dataset.types[t],start=a,end=b,length=b-a,
                            gold_score=value,detected=int(value>0),best_wrong_nearby=max(neighbors) if neighbors else None,
                            boundary_margin=value-max(neighbors) if neighbors else None))
        summary=[]
        from sklearn.metrics import average_precision_score
        for (typ,side),(logits,y) in boundary.items():
            s=np.asarray(logits);y=np.asarray(y,dtype=bool);p=s>0;tp=int((p&y).sum())
            summary.append(dict(type=typ,side=side,positions=len(y),gold=int(y.sum()),predicted=int(p.sum()),tp=tp,
                precision=tp/max(1,int(p.sum())),recall=tp/max(1,int(y.sum())),
                average_precision=float(average_precision_score(y,s)) if y.any() else None))
        write_csv(out/'characters.csv',chars);write_csv(out/'lexical_slots.csv',slots);write_csv(out/'attention_heads.csv',heads)
        write_csv(out/'gold_span_margins.csv',margins);write_csv(out/'boundary_metrics.csv',summary)
        write_json(out/'PANEL.json',{'sentence_ids':ids,'sentence_count':len(ids),'window_count':len(windows),
            'text_sha256':[hashlib.sha256(dataset.samples[i][0].encode()).hexdigest() for i in ids],
            'scope':'fixed dev panel; descriptive; overlapping windows may repeat positions; not corpus counts',
            'weight_tag':tag,'variant':cfg.variant,'used_for_training_or_selection':False})
    finally:
        for h in handles:h.remove()
        model.train(was_training);restore_rng(state)
    barrier()

def gradient_record(model,step):
    groups=defaultdict(lambda:[0.,0.,0])
    for n,p in model.named_parameters():
        if p.grad is None:continue
        g=n.split('.')[0];groups[g][0]+=float(p.grad.detach().double().square().sum())
        groups[g][1]+=float(p.detach().double().square().sum());groups[g][2]+=p.numel()
    return {'step':step,'groups':{g:{'gradient_norm':math.sqrt(v[0]),'parameter_norm':math.sqrt(v[1]),'numel':v[2]} for g,v in groups.items()}}
