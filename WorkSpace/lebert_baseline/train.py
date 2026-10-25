"""Two-GPU LEBERT baseline. Test is a separate explicit command."""
import argparse
from collections import defaultdict
import json
import hashlib
import math
import os
from pathlib import Path
import random
import signal
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader,DistributedSampler
from transformers import BertTokenizer,get_linear_schedule_with_warmup
from tqdm import tqdm
from data import WindowDataset,collate,extract_spans,label_names,read_bio,read_types,score,sha
from model import LebertCRF

STOP=False


def stop(signum,frame):
    global STOP
    STOP=True


def json_save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def atomic_save(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp');torch.save(value,tmp);os.replace(tmp,path)


def rng():return {'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state(),'numpy':np.random.get_state(),'python':random.getstate()}


def restore(state):
    torch.set_rng_state(state['torch']);torch.cuda.set_rng_state(state['cuda']);np.random.set_state(state['numpy']);random.setstate(state['python'])


@torch.no_grad()
def evaluate(model,dataset,types,device,batch_size,output=None):
    model.eval();num_tags=len(dataset.labels)
    totals=[np.zeros((len(t),num_tags),dtype=np.float32) for t,_ in dataset.samples]
    weights=[np.zeros(len(t),dtype=np.int32) for t,_ in dataset.samples]
    loader=DataLoader(dataset,batch_size=batch_size,collate_fn=collate,shuffle=False,
                      generator=torch.Generator().manual_seed(1122))
    for batch in loader:
        meta=batch.pop('meta');batch.pop('labels')
        batch={k:v.to(device) for k,v in batch.items()}
        with torch.autocast('cuda',dtype=torch.bfloat16):emissions=model(**batch)
        emissions=emissions.float().cpu().numpy();lengths=batch['char_mask'].sum(1).tolist()
        for i,((sid,offset),n) in enumerate(zip(meta,lengths)):
            totals[sid][offset:offset+n]+=emissions[i,:n];weights[sid][offset:offset+n]+=1
    predictions={}
    order=sorted(range(len(totals)),key=lambda i:len(totals[i]))
    for start in range(0,len(order),batch_size):
        ids=order[start:start+batch_size];n=max(len(totals[i]) for i in ids)
        emissions=torch.zeros(len(ids),n,num_tags,device=device);mask=torch.zeros(len(ids),n,dtype=torch.bool,device=device)
        for j,sid in enumerate(ids):
            assert np.all(weights[sid]>0)
            x=totals[sid]/weights[sid][:,None];emissions[j,:len(x)]=torch.from_numpy(x).to(device);mask[j,:len(x)]=True
        for sid,tags in zip(ids,model.crf.decode(emissions,mask)):
            predictions[sid]=set(extract_spans([dataset.labels[t] for t in tags]))
    result=score(dataset.samples,predictions,types)
    result.update({'evaluation':'Original character span/type; average overlapping-window emissions then full-sentence CRF Viterbi',
                   'sentences':len(dataset.samples),'windows':len(dataset),'gold_outside_complete_windows':dataset.uncovered_gold})
    if output:
        with Path(output).open('w') as f:
            for sid,(text,gold) in enumerate(dataset.samples):
                ent=lambda e:{'type':e[0],'start':e[1],'end':e[2],'text':text[e[1]:e[2]]}
                f.write(json.dumps({'sid':sid,'text':text,'gold':[ent(e) for e in gold],
                    'predicted':[ent(e) for e in sorted(predictions[sid])]},ensure_ascii=False)+'\n')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='/root/data/WorkSpace');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epochs',type=int,default=20);p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--eval-batch-size',type=int,default=32);p.add_argument('--save-steps',type=int,default=250)
    p.add_argument('--encoder-lr',type=float,default=1e-5);p.add_argument('--head-lr',type=float,default=1e-4)
    p.add_argument('--warmup',type=float,default=.1);p.add_argument('--test',action='store_true');args=p.parse_args()
    local=int(os.environ.get('LOCAL_RANK',0));world=int(os.environ.get('WORLD_SIZE',1));rank=int(os.environ.get('RANK',0))
    torch.cuda.set_device(local);device=torch.device('cuda',local)
    if world>1:dist.init_process_group('nccl',device_id=device)
    barrier=lambda:dist.barrier() if world>1 else None
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    root=Path(args.root);out=root/'output'/f'lebert_agcner_seed{args.seed}';out.mkdir(parents=True,exist_ok=True)
    encoder=root/'resources/bert-base-chinese';resource=root/'resources/tencent_200d_first1m'
    for path in (encoder/'config.json',encoder/'model.safetensors',resource/'vectors.npy',resource/'words.json'):
        if not path.is_file():raise FileNotFoundError(f'Missing resource: {path}; run prepare_resources.py first')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    tokenizer=BertTokenizer.from_pretrained(encoder,local_files_only=True)
    # Schema comes from the declared label list, not test annotations.
    types=read_types(root/'data/label.txt')
    words=json.loads((resource/'words.json').read_text());wordids={w:i for i,w in enumerate(words) if i>0}
    vectors=torch.from_numpy(np.load(resource/'vectors.npy'))
    sources={x:sha(Path(__file__).parent/x) for x in ('train.py','model.py','data.py','prepare_resources.py')}
    identity={'train':sha(root/'data/train2.txt'),'dev':sha(root/'data/dev2.txt'),'labels':sha(root/'data/label.txt'),
              'bert_weights':sha(encoder/'model.safetensors'),'word_resource':sha(resource/'vectors.npy'),
              'word_ids':sha(resource/'words.json'),'sources':sources,
              'bert_config':sha(encoder/'config.json'),'tokenizer_vocab':sha(encoder/'vocab.txt'),
              'tokenizer_config':sha(encoder/'tokenizer_config.json') if (encoder/'tokenizer_config.json').exists() else None}
    def dataset(split):
        cache=root/'output/lebert_data_cache';cache.mkdir(exist_ok=True)
        cache_identity={k:identity[k] for k in ('word_ids','labels','tokenizer_vocab','tokenizer_config')}
        cache_identity.update(data=sha(root/'data'/f'{split}2.txt'),implementation=sources['data.py'])
        stamp=hashlib.sha256(json.dumps(cache_identity,sort_keys=True).encode()).hexdigest()[:24]
        path=cache/f'{split}_{stamp}.pt'
        if rank==0 and not path.exists():
            d=WindowDataset(read_bio(root/'data'/f'{split}2.txt'),tokenizer,wordids,types)
            atomic_save(path,d)
        barrier();return torch.load(path,weights_only=False)
    last=out/'last.pt'
    saved=torch.load(out/'best.pt' if args.test else last,map_location='cpu',weights_only=False) if args.test or last.exists() else None
    if saved:
        if saved['identity']!=identity:raise ValueError('Input resources or code changed; use the recorded package for resume/evaluation')
        if not args.test and (saved['world_size']!=world or saved['args']!=vars(args)):raise ValueError('Resume requires the saved training settings and GPU count')
    model=LebertCRF(str(encoder),vectors,len(label_names(types)),pretrained=saved is None)
    del vectors,words
    if saved:model.load_state_dict(saved['model'])
    model.to(device)
    if args.test:
        test=dataset('test')
        if rank==0:
            result=evaluate(model,test,types,device,args.eval_batch_size,out/'test_predictions.jsonl');json_save(out/'test_metrics.json',result);print(result['micro'])
        barrier()
        if world>1:dist.destroy_process_group()
        return
    train=dataset('train');dev=dataset('dev')
    del wordids
    sampler=DistributedSampler(train,num_replicas=world,rank=rank,shuffle=True,seed=args.seed)
    generator=torch.Generator()
    loader=DataLoader(train,batch_size=args.batch_size,sampler=sampler,collate_fn=collate,generator=generator)
    optimizer=torch.optim.Adam([{'params':model.bert.parameters(),'lr':args.encoder_lr},
        {'params':[v for n,v in model.named_parameters() if not n.startswith('bert.')],'lr':args.head_lr}],weight_decay=0.)
    total_steps=len(loader)*args.epochs
    scheduler=get_linear_schedule_with_warmup(optimizer,int(args.warmup*total_steps),total_steps)
    epoch0=0;batch0=0;global_step=0;best=-1.;history=[]
    if saved:
        optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        epoch0,batch0,global_step=saved['next_epoch'],saved['next_batch'],saved['global_step'];best=saved['best'];history=saved['history']
    wrapped=DistributedDataParallel(model,device_ids=[local],broadcast_buffers=False) if world>1 else model
    if saved:restore(saved['rng'][rank])
    if rank==0:
        json_save(out/'run_config.json',{'args':vars(args),'identity':identity,'world_size':world,
          'effective_batch':world*args.batch_size,'types':types,'train_sentences':len(train.samples),'train_windows':len(train),
          'dev_sentences':len(dev.samples),'dev_windows':len(dev),'test_read_by_train':False,
          'model':'LEBERT-CRF, adapter after BERT layer 1','optimizer':'Adam; weight_decay=0',
          'protocol':'AgCNER reimplementation; external fixed vector vocabulary; BIOES CRF; 20 epochs; best dev Micro F1',
          'environment':{'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'cuda':torch.version.cuda}})
    def checkpoint(next_epoch,next_batch,epoch_copy=False):
        states=[None]*world
        if world>1:dist.all_gather_object(states,rng())
        else:states=[rng()]
        if rank==0:
            state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
              'next_epoch':next_epoch,'next_batch':next_batch,'global_step':global_step,'best':best,'history':history,
              'rng':states,'identity':identity,'world_size':world,'args':vars(args),
              'epoch_loss_sum':loss_sum if next_batch else 0.,'epoch_batches':count if next_batch else 0}
            atomic_save(last,state)
            if epoch_copy:atomic_save(out/f'epoch_{next_epoch:03d}.pt',state)
        barrier()
    for epoch in range(epoch0,args.epochs):
        sampler.set_epoch(epoch);generator.manual_seed(args.seed+epoch)
        wrapped.train();started=time.perf_counter()
        loss_sum=saved.get('epoch_loss_sum',0.) if saved and epoch==epoch0 and batch0 else 0.
        count=saved.get('epoch_batches',0) if saved and epoch==epoch0 and batch0 else 0
        bar=tqdm(total=len(loader),initial=batch0 if epoch==epoch0 else 0,desc=f'Epoch {epoch+1}/{args.epochs}',disable=rank!=0,mininterval=2)
        for index,batch in enumerate(loader):
            if epoch==epoch0 and index<batch0:continue
            batch.pop('meta');batch={k:v.to(device) for k,v in batch.items()};optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):loss,_=wrapped(**batch)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();scheduler.step();global_step+=1;loss_sum+=float(loss.detach());count+=1
            flag=torch.tensor(int(STOP),device=device)
            if world>1:dist.all_reduce(flag,op=dist.ReduceOp.MAX)
            if global_step%args.save_steps==0 or flag.item():checkpoint(epoch,index+1)
            bar.update();bar.set_postfix(loss=loss_sum/count)
            if flag.item():
                bar.close()
                if world>1:dist.destroy_process_group()
                return
        bar.close();barrier()
        if rank==0:
            result=evaluate(model,dev,types,device,args.eval_batch_size,out/'latest_dev_predictions.jsonl')
            history.append({'epoch':epoch+1,'global_step':global_step,'train_loss_rank0':loss_sum/max(1,count),
                            'seconds_current_session_epoch':time.perf_counter()-started,'dev':result})
            if result['micro']['f1']>best:
                best=result['micro']['f1'];atomic_save(out/'best.pt',{'model':model.state_dict(),'identity':identity,'args':vars(args),'epoch':epoch+1,'dev':result})
                (out/'best_dev_predictions.jsonl').write_bytes((out/'latest_dev_predictions.jsonl').read_bytes());json_save(out/'best_dev_metrics.json',result)
            json_save(out/'history.json',history);print(f'epoch={epoch+1}, dev Micro F1={result["micro"]["f1"]:.6f}',flush=True)
        barrier();checkpoint(epoch+1,0,epoch_copy=(epoch+1)%5==0);batch0=0
    if world>1:dist.destroy_process_group()


if __name__=='__main__':main()
