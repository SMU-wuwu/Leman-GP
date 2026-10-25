"""Reuse completed B/C final EMA controls. No training, no optimizer updates."""
import argparse,json,gc
from pathlib import Path
import torch,torch.distributed as dist
from config import Config
from model import LemanGP
from data import sha256,load_split,entity_types,write_json
from train import prepare_resources
from recipe import training_recipe
from controlled_evaluate import evaluate
from probes import probe
from runtime import setup,rank,broadcast_object,input_identity,atomic_save,source_hashes,barrier

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',default='/root/data/WorkSpace');a=p.parse_args();device=setup()
 try:
  for v,phase in [('fgm_control','C'),('weighted_control','B')]:
   c=Config.make(a.root,v,42);out=Path(c.output_dir);out.mkdir(parents=True,exist_ok=True)
   source=Path(a.root)/'output/leman_controlled_ablation_v1/F_seed42'/phase
   receipt=json.loads((source/'COMPLETE.json').read_text());ckpath=source/'last.pt'
   ck=torch.load(ckpath,map_location='cpu',weights_only=False,mmap=True)
   assert ck['epoch']==c.epochs and ck['next_batch']==0 and receipt['completed_epochs']==c.epochs
   assert ck['config']['variant']=='F' and ck['config']['seed']==42
   for name in ('encoder_lr','head_lr','batch_size','weight_decay','warmup_fraction','gp_positive_weight','rare_type_exponent','boundary_weight','ema_decay'):
    assert getattr(c,name)==ck['config'][name],(v,name)
   identity=broadcast_object(input_identity(c) if rank()==0 else None);types=entity_types(Path(c.data_dir)/'label.txt')
   assert identity==ck['input_identity']
   paths=broadcast_object(prepare_resources(c,identity,types) if rank()==0 else None)
   train,dev=map(load_split,paths);lex=json.loads((Path(c.resource_dir)/'lexicon.json').read_text())
   c.gp_type_weights=training_recipe(train.samples,types,c,identity['train2.txt'])['ordered_weights']
   m=LemanGP(c,len(types),len(lex['words']),char_vocab_size=len(lex['characters']),encoder_config=ck['encoder_config'],pretrained=False)
   weights={k:ck['ema']['shadow'].get(k,val) for k,val in ck['model'].items()};m.load_state_dict(weights,strict=True);m.to(device)
   result=evaluate(m,dev,c,device,output_dir=out/f'epoch_{c.epochs:03d}_ema')
   probe(m,dev,c,device,out,f'epoch_{c.epochs:03d}_ema')
   if rank()==0:
    source_sha=sha256(ckpath)
    payload={'config':c.to_dict(),'model':{k:x.detach().cpu() for k,x in m.state_dict().items()},'encoder_config':m.encoder.config.to_dict(),
        'input_identity':identity,'lexicon_sha256':ck['lexicon_sha256'],'types':types,'origin':{'source':str(ckpath),'sha256':source_sha,'zero_new_updates':True}}
    atomic_save(payload,out/'final_ema.pt');h=sha256(out/'final_ema.pt')
    write_json(out/'PRIMARY_ENDPOINT.json',{'checkpoint':'final_ema.pt','sha256':h,'epoch':c.epochs,'weight_source':'ema','dev':result,'reused_completed_phase':phase})
    write_json(out/'history.json',ck['history'])
    write_json(out/'REUSED_CONTROL.json',{'source':str(ckpath),'sha256':source_sha,'epoch':c.epochs,'new_training_updates':0,
        'original_final_ema_micro':ck['history'][-1]['dev_ema']['micro'],'reevaluated_micro':result['micro'],
        'source_config':ck['config'],'matching_config_fields_checked':True})
    write_json(out/'COMPLETE.json',{'reused_control':True,'source_hashes':source_hashes(Path(__file__).parent),'primary_sha256':h,'test_read':False})
   barrier();from evaluate import release_loader
   release_loader(dev);del m,ck,weights,train,dev;gc.collect();torch.cuda.empty_cache()
 finally:dist.destroy_process_group()
if __name__=='__main__':main()
