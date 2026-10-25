"""Synthetic checks only; never reads the research dataset or downloaded models."""
import argparse
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
import torch
from transformers import BertConfig,BertModel,BertTokenizer
from data import WindowDataset,collate,extract_spans,score
from model import LinearCRF,LebertCRF,LexiconAdapter
from prepare_resources import prepare_vectors


def unit_checks():
    torch.manual_seed(117)
    crf=LinearCRF(3)
    with torch.no_grad():
        for p in crf.parameters():p.normal_(0,.2)
    e=torch.randn(2,4,3,requires_grad=True)
    tags=torch.tensor([[0,2,1,2],[1,2,0,0]])
    mask=torch.tensor([[1,1,1,1],[1,0,0,0]],dtype=torch.bool)
    expected=[];paths=[]
    for b,n in enumerate([4,1]):
        candidates=list(itertools.product(range(3),repeat=n));scores=[]
        for path in candidates:
            v=crf.start[path[0]]+crf.end[path[-1]]
            v=v+sum(e[b,j,t] for j,t in enumerate(path))
            v=v+sum(crf.transitions[path[j-1],path[j]] for j in range(1,n))
            scores.append(v)
        scores=torch.stack(scores);gold=tuple(tags[b,:n].tolist())
        expected.append(torch.logsumexp(scores,0)-scores[candidates.index(gold)])
        paths.append(list(candidates[int(scores.argmax())]))
    loss=crf(e,tags,mask)
    torch.testing.assert_close(loss,torch.stack(expected).mean())
    assert crf.decode(e,mask)==paths
    loss.backward();assert e.grad[1,1:].abs().sum()==0
    altered=e.detach().clone();altered[1,1:]=1e4
    torch.testing.assert_close(crf(altered,tags,mask),loss)
    assert crf.decode(altered,mask)==paths
    crf.zero_grad(set_to_none=True)
    crf(e.detach()[:1,:1],tags[:1,:1],mask[:1,:1]).backward()
    assert crf.transitions.grad is not None and crf.transitions.grad.abs().sum()==0

    adapter=LexiconAdapter(8,6,0,1e-12).eval()
    chars=torch.randn(2,4,8,requires_grad=True);words=torch.randn(2,4,3,6,requires_grad=True)
    valid=torch.zeros(2,4,3,dtype=torch.bool)
    torch.testing.assert_close(adapter(chars,words,valid),adapter.norm(chars))
    valid[0,1,:2]=True
    out=adapter(chars,words,valid)
    out.square().sum().backward()
    assert words.grad[0,1,:2].abs().sum()>0 and words.grad[1].abs().sum()==0
    assert extract_spans(['B-X','E-X','O','S-X','I-X','O'])==[('X',0,2),('X',3,4),('X',4,5)]
    metrics=score([('甲乙',[('X',0,1)])],{0:{('X',0,1),('X',1,2)}},['X'])
    assert metrics['micro']['tp']==1 and metrics['micro']['fp']==1 and metrics['micro']['fn']==0

    with tempfile.TemporaryDirectory(prefix='lebert-unit-') as tmp:
        tmp=Path(tmp);(tmp/'vocab.txt').write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\n甲\n乙\n丙\n丁\n戊\n己\n')
        tok=BertTokenizer(vocab_file=str(tmp/'vocab.txt'))
        rows=[('甲乙丙丁戊己',[('X',0,2),('X',3,6)]),('乙',[('X',0,1)])]
        d=WindowDataset(rows,tok,{'甲乙':1,'丁戊己':2,'甲':3,'乙':4},['X'],max_chars=4,overlap=2)
        assert d.uncovered_gold==0 and len(d)==3
        cfg=BertConfig(vocab_size=11,hidden_size=16,num_hidden_layers=2,num_attention_heads=4,intermediate_size=32)
        model=LebertCRF(None,torch.randn(5,6),5,pretrained=False,config=cfg)
        assert model.bert.config._attn_implementation=='eager'
        batch=collate([d[i] for i in range(len(d))]);batch.pop('meta')
        loss,_=model(**batch);loss.backward()
        assert torch.isfinite(loss) and model.adapter.bilinear.grad.abs().sum()>0
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        text='2 200\n甲乙  '+'  '.join(['0.1']*200)+'\n丁戊己 '+ ' '.join(['0.2']*200)+'\n'
        (tmp/'vectors.txt').write_text(text)
        prepare_vectors(str(tmp/'vectors.txt'),tmp/'vectors',2)
        assert np.load(tmp/'vectors/vectors.npy').shape==(3,200)
        assert json.loads((tmp/'vectors/words.json').read_text())[1:]==['甲乙','丁戊己']
    return {'crf_exhaustive_nll_and_viterbi':'PASS','padding_loss_gradient_and_decode':'PASS',
            'empty_and_partial_lexicon_mask':'PASS','bioes_and_exact_spans':'PASS',
            'overlapping_windows_gold_coverage':'PASS','tiny_bert_adapter_backward':'PASS','word_vector_parser':'PASS'}


def fixture(root):
    torch.manual_seed(212)
    root=Path(root);(root/'data').mkdir(parents=True)
    encoder=root/'resources/bert-base-chinese';encoder.mkdir(parents=True)
    (encoder/'vocab.txt').write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\n甲\n乙\n丙\n丁\n戊\n己\n')
    BertTokenizer(vocab_file=str(encoder/'vocab.txt')).save_pretrained(encoder)
    cfg=BertConfig(vocab_size=11,hidden_size=16,num_hidden_layers=2,num_attention_heads=4,intermediate_size=32)
    BertModel(cfg,add_pooling_layer=False).save_pretrained(encoder)
    resource=root/'resources/tencent_200d_first1m';resource.mkdir(parents=True)
    (resource/'words.json').write_text(json.dumps(['<pad>','甲乙','丙','丁戊','己']))
    arr=np.random.default_rng(113).normal(0,.1,(5,200)).astype('float32');arr[0]=0;np.save(resource/'vectors.npy',arr)
    examples=['甲 B-X\n乙 I-X\n丙 O\n','丁 B-X\n戊 I-X\n己 O\n','丙 O\n己 B-X\n','甲 B-X\n']
    (root/'data/train2.txt').write_text('\n'.join(examples*2)+'\n')
    (root/'data/dev2.txt').write_text('\n'.join(examples)+'\n')
    (root/'data/label.txt').write_text('O\nB-X\nI-X\n')


def integration_checks(result_dir):
    result_dir=Path(result_dir);result_dir.mkdir(parents=True,exist_ok=True)
    base=Path(tempfile.mkdtemp(prefix='lebert-ddp-'));here=Path(__file__).resolve().parent
    continuous=base/'continuous';resumed=base/'resumed'
    fixture(continuous);fixture(resumed)
    stop_script=base/'pause.py'
    stop_script.write_text('import sys\nsys.path.insert(0,'+repr(str(here))+')\nimport train\ntrain.STOP=True\ntrain.main()\n')
    env=os.environ.copy();env['OMP_NUM_THREADS']='1';env['TOKENIZERS_PARALLELISM']='false'
    env['NCCL_P2P_DISABLE']='1';env['NCCL_IB_DISABLE']='1'
    def run(script,root,name,extra=()):
        entry=base/f'{name}_entry.py'
        entry.write_text('import sys,runpy,faulthandler\nsys.path.insert(0,'+repr(str(here))+')\nfaulthandler.dump_traceback_later(30)\nrunpy.run_path('+repr(str(script))+',run_name="__main__")\n')
        command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',str(entry),
                 '--root',str(root),'--epochs','2','--batch-size','2','--eval-batch-size','2','--save-steps','1',*extra]
        with (result_dir/f'{name}.log').open('w') as f:
            subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,env=env,check=True,cwd=here)
    run(here/'train.py',continuous,'continuous')
    run(stop_script,resumed,'pause')
    state=torch.load(resumed/'output/lebert_agcner_seed42/last.pt',weights_only=False,map_location='cpu')
    assert state['global_step']==1 and state['next_batch']==1
    run(here/'train.py',resumed,'resume')
    a=torch.load(continuous/'output/lebert_agcner_seed42/last.pt',weights_only=False,map_location='cpu')
    b=torch.load(resumed/'output/lebert_agcner_seed42/last.pt',weights_only=False,map_location='cpu')
    assert a['global_step']==b['global_step']==4
    maxdiff=max(float((a['model'][k]-b['model'][k]).abs().max()) for k in a['model'])
    assert maxdiff==0, maxdiff
    assert [x['dev'] for x in a['history']]==[x['dev'] for x in b['history']]
    assert [x['train_loss_rank0'] for x in a['history']]==[x['train_loss_rank0'] for x in b['history']]
    assert not (continuous/'data/test2.txt').exists() and not (resumed/'data/test2.txt').exists()
    (resumed/'data/test2.txt').write_bytes((resumed/'data/dev2.txt').read_bytes())
    run(here/'train.py',resumed,'explicit_test',('--test',))
    test=json.loads((resumed/'output/lebert_agcner_seed42/test_metrics.json').read_text())
    assert test['micro']['support']==4
    return {'synthetic_root':str(base),'ddp_bf16_two_epochs':'PASS','resume_parameter_max_difference':maxdiff,
            'resume_dev_predictions_and_metrics':'PASS','training_without_test_file':'PASS','explicit_test':'PASS',
            'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),'world_size':2}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ddp',action='store_true');p.add_argument('--output',default='qa_results');a=p.parse_args()
    result={'unit':unit_checks()}
    if a.ddp:result['integration']=integration_checks(a.output)
    Path(a.output).mkdir(parents=True,exist_ok=True)
    (Path(a.output)/'QA_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
