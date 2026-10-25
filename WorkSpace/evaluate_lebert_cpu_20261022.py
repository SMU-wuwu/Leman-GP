"""Evaluate the previously selected LEBERT checkpoint on AgCNER test, without training."""
import os
os.environ['OMP_NUM_THREADS'] = '16'
os.environ['MKL_NUM_THREADS'] = '16'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import sys, json, time, hashlib
from pathlib import Path
import numpy as np
import torch
from transformers import BertTokenizer

ROOT = Path('/root/data/WorkSpace')
sys.path.insert(0, str(ROOT / 'lebert_baseline'))
from model import LebertCRF
from data import WindowDataset, collate, read_bio, read_types, label_names, extract_spans, score, sha

def main():
    started = time.time()
    torch.set_num_threads(16)
    torch.set_num_interop_threads(1)
    out = ROOT / 'output/lebert_agcner_test_cpu_20261022'
    out.mkdir(exist_ok=True)
    assert not (out / 'RESULT.json').exists(), 'Completed evaluation already exists'
    checkpoint = ROOT / 'output/lebert_agcner_seed42/best.pt'
    checkpoint_sha = sha(checkpoint)
    assert checkpoint_sha == 'b6adb971761966dfbc8a3bec14663dc79a78bb296062f378256a4bf5ebc4a8fe'
    record = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    identity = record['identity']
    for key, relative in [('train','data/train2.txt'),('dev','data/dev2.txt'),('labels','data/label.txt'),
                          ('word_ids','resources/tencent_200d_first1m/words.json'),
                          ('bert_config','resources/bert-base-chinese/config.json'),
                          ('tokenizer_vocab','resources/bert-base-chinese/vocab.txt')]:
        assert sha(ROOT/relative) == identity[key], key
    for filename, expected in identity['sources'].items():
        assert sha(ROOT/'lebert_baseline'/filename) == expected, filename
    types = read_types(ROOT/'data/label.txt')
    encoder = ROOT/'resources/bert-base-chinese'
    words = json.loads((ROOT/'resources/tencent_200d_first1m/words.json').read_text())
    tokenizer = BertTokenizer.from_pretrained(encoder, local_files_only=True)
    rows = read_bio(ROOT/'data/test2.txt')
    assert len(rows) == 5974 and sum(len(g) for _, g in rows) == 20524
    dataset = WindowDataset(rows, tokenizer, {w:i for i,w in enumerate(words) if i>0}, types)
    model = LebertCRF(encoder, torch.zeros_like(record['model']['word_embeddings.weight']), len(dataset.labels), pretrained=False)
    model.load_state_dict(record['model'], strict=True)
    model.eval()
    model.requires_grad_(False)
    del words, record
    num_tags = len(dataset.labels)
    totals = [np.zeros((len(t),num_tags),dtype=np.float32) for t,_ in rows]
    weights = [np.zeros(len(t),dtype=np.int32) for t,_ in rows]
    order = sorted(range(len(dataset)),key=lambda i:len(dataset[i]['ids']))
    print(json.dumps({'state':'inference','sentences':len(rows),'windows':len(dataset),'checkpoint_sha256':checkpoint_sha,'seconds':time.time()-started}),flush=True)
    with torch.inference_mode():
        for start in range(0,len(order),16):
            batch = collate([dataset[i] for i in order[start:start+16]])
            meta = batch.pop('meta'); batch.pop('labels')
            emissions = model(**batch).float().numpy()
            lengths = batch['char_mask'].sum(1).tolist()
            for i,((sid,offset),length) in enumerate(zip(meta,lengths)):
                totals[sid][offset:offset+length] += emissions[i,:length]
                weights[sid][offset:offset+length] += 1
            if start % 320 == 0:
                status = {'state':'inference','windows_done':min(start+16,len(order)),'windows':len(order),'seconds':time.time()-started}
                (out/'PROGRESS.json').write_text(json.dumps(status))
                print(json.dumps(status),flush=True)
        predictions = {}
        order = sorted(range(len(rows)),key=lambda i:len(rows[i][0]))
        for start in range(0,len(order),32):
            ids=order[start:start+32]; n=max(len(rows[i][0]) for i in ids)
            emissions=torch.zeros(len(ids),n,num_tags);mask=torch.zeros(len(ids),n,dtype=torch.bool)
            for j,sid in enumerate(ids):
                assert np.all(weights[sid]>0)
                x=totals[sid]/weights[sid][:,None]
                emissions[j,:len(x)] = torch.from_numpy(x);mask[j,:len(x)] = True
            for sid,tags in zip(ids,model.crf.decode(emissions,mask)):
                predictions[sid]=set(extract_spans([dataset.labels[t] for t in tags]))
    result = score(rows,predictions,types)
    with (out/'test_predictions.jsonl').open('w') as f:
        for sid,(text,gold) in enumerate(rows):
            ent=lambda e:{'type':e[0],'start':e[1],'end':e[2],'text':text[e[1]:e[2]]}
            f.write(json.dumps({'sid':sid,'text':text,'gold':[ent(e) for e in gold], 'predicted':[ent(e) for e in sorted(predictions[sid])]},ensure_ascii=False)+'\n')
    result.update({'dataset':'AgCNER','split':'test','sentences':len(rows),'windows':len(dataset),'gold_outside_complete_windows':dataset.uncovered_gold,
        'checkpoint_sha256':checkpoint_sha,'checkpoint_epoch':9,'test_annotation_sha256':sha(ROOT/'data/test2.txt'),
        'prediction_sha256':sha(out/'test_predictions.jsonl'),'precision':'FP32','device':'CPU','threads':16,'batch_size':16,
        'training':False,'optimizer_steps':0,'weights_changed':False,'seconds':time.time()-started,'torch_version':torch.__version__,
        'script_sha256':sha(Path(__file__)),'evaluation':'Exact original character span/type; mean overlapping-window emissions; full-sentence CRF Viterbi',
        'resource_and_source_identity_verified':True})
    (out/'RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({'state':'complete','micro':result['micro'],'seconds':result['seconds']}),flush=True)

if __name__ == '__main__': main()
