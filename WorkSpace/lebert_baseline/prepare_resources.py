"""Prepare a fixed external vector prefix, without reading any task split."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
import urllib.request
import numpy as np

VECTOR_URL='https://ai.tencent.com/ailab/nlp/en/data/tencent-ailab-embedding-zh-d200-v0.2.0.tar.gz'


def prepare_vectors(source,output,limit):
    if limit<1:raise ValueError('limit must be positive')
    output=Path(output)
    if output.exists():
        record=json.loads((output/'RESOURCE.json').read_text())
        if record['source']!=source or record['fixed_prefix_rows']!=min(limit,record['original_rows']):
            raise ValueError('Existing vector resource has different source/prefix settings')
        print(f'Using completed vector resource: {output}',flush=True)
        return
    output.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix=output.name+'.partial-',dir=output.parent))
    try:
        _write_vectors(source,staging,limit)
        os.replace(staging,output)
    except BaseException:
        shutil.rmtree(staging)
        raise


def _write_vectors(source,output,limit):
    remote=source.startswith(('https://','http://'))
    raw=urllib.request.urlopen(source,timeout=60) if remote else open(source,'rb')
    archive=None
    if source.endswith(('.tar.gz','.tgz')):
        archive=tarfile.open(fileobj=raw,mode='r|gz')
        member=next(m for m in archive if m.isfile() and m.name.endswith(('.txt','.vec')))
        stream=archive.extractfile(member)
    else:stream=raw
    reader=io.TextIOWrapper(stream,encoding='utf-8')
    first=next(reader);total,dim=map(int,first.split());assert dim==200
    n=min(total,limit);matrix=np.lib.format.open_memmap(output/'vectors.npy',mode='w+',dtype='float32',shape=(n+1,dim));matrix[0]=0
    words=['<pad>'];digest=hashlib.sha256(first.encode());used=0
    for line in reader:
        digest.update(line.encode());fields=line.rstrip().rsplit(None,dim)
        if len(fields)!=dim+1:raise ValueError(f'Expected word plus {dim} vector values')
        used+=1;words.append(fields[0]);matrix[used]=np.asarray(fields[1:],dtype=np.float32)
        if used==n:break
    assert used==n and np.isfinite(matrix).all()
    matrix.flush();del matrix
    (output/'words.json').write_text(json.dumps(words,ensure_ascii=False))
    (output/'RESOURCE.json').write_text(json.dumps({'source':source,'fixed_prefix_rows':n,'original_rows':total,
        'dimension':dim,'consumed_text_sha256':digest.hexdigest(),'task_data_read':False,
        'policy':'First 1,000,000 external vector rows by default, following the author demo scan budget; task-independent vocabulary, no AgCNER gold induction'},indent=2))
    raw.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='/root/data/WorkSpace');p.add_argument('--vectors',default=VECTOR_URL)
    p.add_argument('--limit',type=int,default=1000000);p.add_argument('--bert-only',action='store_true');p.add_argument('--vectors-only',action='store_true');a=p.parse_args()
    root=Path(a.root)/'resources'
    if not a.vectors_only:
        from huggingface_hub import HfApi,snapshot_download
        record=root/'BERT_RESOURCE.json'
        revision=json.loads(record.read_text())['revision'] if record.exists() else HfApi().model_info('google-bert/bert-base-chinese').sha
        root.mkdir(parents=True,exist_ok=True)
        record.write_text(json.dumps({'repo':'google-bert/bert-base-chinese','revision':revision},indent=2)+'\n')
        snapshot_download('google-bert/bert-base-chinese',local_dir=root/'bert-base-chinese',
            revision=revision,
            allow_patterns=['config.json','model.safetensors','vocab.txt','tokenizer.json','tokenizer_config.json'])
    if not a.bert_only:prepare_vectors(a.vectors,root/'tencent_200d_first1m',a.limit)
