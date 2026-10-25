"""Inventory raw external corpora without flattening or changing annotations."""
from collections import Counter,defaultdict
from pathlib import Path
import json
from evidence_core import sha256,write_json

ROOT=Path('/root/data/WorkSpace/data')
OUT=Path('/root/data/WorkSpace/output/manuscript_revision_20261019/external_data')


def bio_rows(path,weibo=False):
    samples=[];units=[];labels=[];diagnostic=Counter();malformed=[]
    def finish():
        if not units:return
        offsets=[0]
        for unit in units:offsets.append(offsets[-1]+len(unit))
        spans=[];active=None
        for i,tag in enumerate(labels+['O']):
            typ=tag[2:] if tag.startswith(('B-','I-')) else None
            if active and (not tag.startswith('I-') or typ!=active[0]):
                spans.append((active[0],offsets[active[1]],offsets[i]));active=None
            if typ and active is None:active=(typ,i)
        samples.append((''.join(units),spans));units.clear();labels.clear()
    for line_no,line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(),1):
        if not line.strip():finish();continue
        token,label=line.split();label=label.replace('_','-')
        if weibo:
            assert token[-1].isdigit(),(path,line_no,token)
            diagnostic['segmentation_index_rows']+=1;token=token[:-1]
            if len(token)!=1:
                diagnostic['multi_codepoint_units']+=1
                if len(malformed)<4:malformed.append({'line':line_no,'unit':token,'label':label})
        units.append(token);labels.append(label)
    finish()
    return samples,{'parsing':('Trailing Weibo segmentation digit removed for audit only; complete remaining unit preserved with reversible offsets. No training conversion written.' if weibo else 'Original BIO units preserved; character offsets obtained by concatenation. No training conversion written.'),
                   **diagnostic,'multi_codepoint_examples':malformed}


def main():
    OUT.mkdir(parents=True,exist_ok=True);allrows={};report={'datasets':{},'neural_training':False,'script_sha256':sha256(__file__)}
    for name in ('CMeEE','People','Resume','Weibo'):
        parts={};allrows[name]={}
        for path in sorted((ROOT/name).iterdir()):
            split=next(s for s in ('train','dev','test') if s in path.name.lower())
            if name=='CMeEE':
                data=json.loads(path.read_text());rows=[];bad=[]
                for sid,row in enumerate(data):
                    spans=[]
                    for e in row.get('entities',[]):
                        a,b=e['start_idx'],e['end_idx']+1
                        if row['text'][a:b]!=e['entity']:bad.append({'sid':sid,'entity':e})
                        assert 0<=a<b<=len(row['text'])
                        spans.append((e['type'],a,b))
                    rows.append((row['text'],spans))
                parsing={'annotation_surface_mismatches':len(bad),'inclusive_end_converted_to_exclusive':True,
                         'test_label_status':'unavailable_all_entities_empty' if split=='test' else 'provided'}
            else:rows,parsing=bio_rows(path,name=='Weibo')
            allrows[name][split]=rows
            support=Counter(t for _,spans in rows for t,_,_ in spans);lengths=Counter(b-a for _,spans in rows for _,a,b in spans)
            overlap=0;nested=0;dupe_entities=0
            for _,spans in rows:
                dupe_entities+=len(spans)-len(set(spans));has_overlap=False;has_nested=False
                for i,(_,a,b) in enumerate(spans):
                    for _,c,d in spans[i+1:]:
                        if max(a,c)<min(b,d):has_overlap=True
                        if (a<=c and d<=b) or (c<=a and b<=d):has_nested=True
                overlap+=has_overlap;nested+=has_nested
            parts[split]={'file':str(path),'sha256':sha256(path),'sentences':len(rows),'unique_texts':len({t for t,_ in rows}),
                'annotations':sum(support.values()),'type_support':dict(sorted(support.items())),
                'sentences_with_overlapping_gold':overlap,'sentences_with_nested_gold':nested,'duplicate_entity_entries':dupe_entities,
                'entities_longer_than_64':sum(n for l,n in lengths.items() if l>64),
                'max_entity_chars':max(lengths,default=None),'max_sentence_chars':max(map(lambda r:len(r[0]),rows)),
                'parsing':parsing}
        overlaps={}
        for a,b in (('train','dev'),('train','test'),('dev','test')):
            ta={t for t,_ in allrows[name][a]};tb={t for t,_ in allrows[name][b]};common=ta&tb
            overlaps[f'{a}_{b}']={'unique_texts':len(common),f'{b}_rows':sum(t in common for t,_ in allrows[name][b])}
        report['datasets'][name]={'splits':parts,'text_overlap':overlaps}
    report['required_protocols']={
        'CMeEE':'Preserve all nested JSON spans; no conversion to single-layer BIO. Existing test has zero provided annotations and is not a scorable local test. Propose fixed text-group inner dev from official train, evaluate on official labeled dev with its name retained, or obtain official test evaluation.',
        'People':'Three entity types; verify source/version before equating this split with literature People’s Daily/MSRA scores.',
        'Resume':'Eight entity types; fixed train/dev/test files with per-corpus train-only lexicon.',
        'Weibo':'Eight NAM/NOM subtypes; retain distinction and additionally report NAM/NOM/overall. Remove segmentation suffix, preserve source-unit offsets; corrupted multi-codepoint units need explicit converter policy.',
        'all':'New task heads and per-dataset train-only lexical resources require neural training; current AgCNER 13-type head is not directly scorable.'}
    write_json(OUT/'EXTERNAL_DATA_AUDIT.json',report)
    for name,d in report['datasets'].items():print(name,{s:(r['sentences'],r['annotations'],r['sentences_with_nested_gold']) for s,r in d['splits'].items()},flush=True)


if __name__=='__main__':main()
