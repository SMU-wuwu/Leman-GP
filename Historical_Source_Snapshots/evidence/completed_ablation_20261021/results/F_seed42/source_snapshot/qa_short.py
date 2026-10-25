"""Synthetic structural and two-rank update tests, no task training."""
import argparse,json,copy
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import BertConfig
from config import Config,VARIANTS
from model import LemanGP,transfer_parent
from parent_model import LemanGP as Parent
from controlled_data import Collator
from runtime import seed_everything,setup,to_device,amp_context,gradient_norm_fp64

def cfg(v):
 c=Config.make('/tmp/synthetic_only',v,42)
 for k,x in dict(word_dim=8,boundary_dim=4,char_dim=8,gru_hidden=8,gru_layers=2,attention_heads=2,gp_head_size=8,max_chars=16,max_span=16,gp_type_weights=[1.,1.],workers=0).items():setattr(c,k,x)
 return c

def enc():return BertConfig(vocab_size=32,hidden_size=32,num_hidden_layers=1,num_attention_heads=2,intermediate_size=48,max_position_embeddings=64).to_dict()
def batch(v,device):
 rows=[]
 for sid,n in enumerate([5,3]):
  rows.append(dict(sid=sid,offset=0,length=n,input_ids=list(range(5,5+n)),offsets=[(i,i+1) for i in range(n)],char_ids=list(range(2,2+n)),word_ids=np.full((n,cfg(v).max_words),2,dtype=np.int32),boundary_ids=np.full((n,cfg(v).max_words),3,dtype=np.int8),spans=[(0,0,2),(1,n-1,n)]))
 return to_device(Collator(['A','B'],0,v)(rows),device)

def main():
 p=argparse.ArgumentParser();p.add_argument('--variants',nargs='+',default=list(VARIANTS));p.add_argument('--ddp',action='store_true');p.add_argument('--output',required=True);a=p.parse_args()
 device=setup() if a.ddp else torch.device('cpu');torch.set_num_threads(2)
 seed_everything(42);parent=Parent(cfg('F'),2,10,char_vocab_size=12,encoder_config=enc(),pretrained=False)
 state=copy.deepcopy(parent.state_dict());records={}
 for v in a.variants:
  seed_everything(42);m=LemanGP(cfg(v),2,10,char_vocab_size=12,encoder_config=enc(),pretrained=False)
  tr=transfer_parent(m,state);m.to(device);x=batch(v,device)
  if v=='F':
   parent.to(device).eval();m.eval()
   with amp_context(device):a0=parent(**x);b0=m(**x)
   for x0,y0 in zip(a0,b0):torch.testing.assert_close(x0,y0,rtol=0,atol=0)
  if v=='no_boundary':
   m.eval();parent.eval();px=batch('F',device);px.pop('labels');mx=dict(x);mx.pop('labels')
   with amp_context(device):torch.testing.assert_close(parent(**px),m(**mx),rtol=0,atol=0)
  if v=='no_recurrence':
   m.eval();xx=torch.randn(2,5,32,device=device);mask=torch.ones(2,5,dtype=torch.bool,device=device)
   y=m.pointwise_gru(xx,mask);xx[:,0]+=1.;yy=m.pointwise_gru(xx,mask)
   assert torch.equal(y[:,1:],yy[:,1:]),'Pointwise path must not mix positions'
  m.train();wrapped=DDP(m,device_ids=[device.index],broadcast_buffers=False) if a.ddp else m
  seed_everything(100+(dist.get_rank() if a.ddp else 0));opt=torch.optim.AdamW((p for p in m.parameters() if p.requires_grad),lr=1e-4)
  for _ in range(2):
   opt.zero_grad(set_to_none=True)
   if v=='clean_two_pass' and a.ddp:
    from clean_control import two_clean_backward
    loss,_=two_clean_backward(wrapped,x,lambda:amp_context(device),lambda val,stage:None)
   elif v=='fgm_control' and a.ddp:
    from adversarial import fgm_backward
    loss,_=fgm_backward(wrapped,x,epsilon=.5,clean_weight=.5,autocast=lambda:amp_context(device))
   else:
    with amp_context(device):loss,_=wrapped(**x)
    loss.backward()
   assert torch.isfinite(loss) and torch.isfinite(gradient_norm_fp64(m.parameters()))
   assert all(p.grad is not None for p in m.parameters() if p.requires_grad),(v,[n for n,p in m.named_parameters() if p.requires_grad and p.grad is None])
   opt.step()
  if a.ddp:
   for p0 in m.parameters():
    ref=p0.detach().clone();dist.broadcast(ref,src=0);assert torch.equal(ref,p0),v
  records[v]={'transfer':tr,'two_updates_finite':True,'all_trainable_parameters_used':True,'rank_sync':a.ddp}
  del opt,wrapped,m
  if device.type=='cuda':torch.cuda.empty_cache()
 if not a.ddp or dist.get_rank()==0:Path(a.output).write_text(json.dumps({'variants':records,'baseline_matches_parent':True,'no_boundary_inference_identical':True,'no_recurrence_position_isolation':True,'synthetic_only':True},indent=2)+'\n')
 if a.ddp:dist.destroy_process_group()
if __name__=='__main__':main()
