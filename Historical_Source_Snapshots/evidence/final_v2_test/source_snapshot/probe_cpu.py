import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4'
import sys,time,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import frozen_pipeline as fp
import torch
from contextlib import nullcontext
torch.set_num_threads(4);torch.set_num_interop_threads(1)
p=fp.ROOT/'output/leman_finetune_v4_seed42/best.pt'
c=torch.load(p,map_location='cpu',weights_only=False,mmap=True);cfg=fp.Config()
for k,v in c['config'].items():setattr(cfg,k,v)
d=fp.load_split(fp.ROOT/'output/leman_finetune_v4_seed42/cache/dev_d7ab938f313842cd.pkl')
m=fp.LemanGP(cfg,len(d.types),c['vocab_size'],char_vocab_size=c['char_vocab_size'],encoder_config=c['encoder_config'],pretrained=False).eval();m.load_state_dict(c['model'],strict=True)
t=time.monotonic()
# This probe checks CPU execution only, on development windows, with no fitting.
r=fp.candidates(m,d,torch.device('cpu'),-4.,batch_size=4,limit=8)
print(json.dumps({'windows':8,'seconds':time.monotonic()-t,'rows':sum(map(len,r.values())),'device':'cpu','dtype':'float32','test_read':False}),flush=True)
