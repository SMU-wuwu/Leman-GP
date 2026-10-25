"""Merge disjoint CPU test-window intervention shards without changing weights."""
import argparse
import json
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--shards',type=int,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    directories=[a.input/f'shard_{i}' for i in range(a.shards)]
    complete=[json.loads((d/'COMPLETE.json').read_text()) for d in directories]
    protocols=[json.loads((d/'PROTOCOL.json').read_text()) for d in directories]
    seen=set()
    for i,r in enumerate(complete):
        assert r['shard']==i and r['shards']==a.shards
        assert r['weights_unchanged'] and r['optimizer_steps']==0
        assert r['split']=='test' and not (seen & set(r['window_ids']))
        seen.update(r['window_ids'])
    assert seen==set(range(6097))
    assert len({r['checkpoint_sha256'] for r in protocols})==1
    assert len({r['weight_state_before'] for r in complete})==1
    conditions=complete[0]['conditions']
    a.out.mkdir(parents=True,exist_ok=True)
    for condition in conditions:
        groups=[read_jsonl(d/condition/'test_predictions.jsonl') for d in directories]
        base=groups[0]
        assert len(base)==5974
        for rows in groups[1:]:
            assert [(r['sid'],r['text'],r['gold']) for r in rows]==[(r['sid'],r['text'],r['gold']) for r in base]
        merged=[]
        for idx,row in enumerate(base):
            preds={}
            for rows in groups:
                for e in rows[idx]['predicted']:
                    preds[(e['type'],e['start'],e['end'])]=e
            nr=dict(row)
            nr['predicted']=[preds[k] for k in sorted(preds)]
            merged.append(nr)
        out=a.out/condition
        out.mkdir(parents=True,exist_ok=True)
        (out/'test_predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in merged))
    protocol=dict(protocols[0])
    protocol.update(batch_layout='length-sorted test windows, disjoint CPU shards; all windows evaluated',
                    shards=a.shards,windows=len(seen))
    (a.out/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2)+'\n')
    summary={'split':'test','weights_unchanged':True,'optimizer_steps':0,'all_windows_evaluated':True,
             'windows':len(seen),'shards':a.shards,'conditions':conditions,
             'weight_state_before':complete[0]['weight_state_before'],
             'weight_state_after':complete[0]['weight_state_after']}
    (a.out/'COMPLETE.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
