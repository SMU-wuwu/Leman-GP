"""Reconstruct training resources and record available checkpoint ancestry."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import importlib.util
import json
import subprocess
import sys
import jieba
import joblib
import numpy as np
from evidence_core import read_bio, sha256, write_json

ROOT = Path('/root/data/WorkSpace')
OUT = ROOT / 'output/manuscript_revision_20261019/resource_audit'
FROZEN = ROOT / 'output/leman_final_v2_20261019'
RUNS = ['leman_paper_a100_seed42', 'leman_f1_4x4090_seed42',
        'leman_finetune_v4_seed42', 'leman_fgm_v10_seed43',
        'leman_fgm_v14_seed44', 'leman_fgm_seed46_dual4090',
        'leman_fgm_seed47_dual4090', 'leman_rdrop_v17_seed45']


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    datasets = {s: read_bio(ROOT/'data'/f'{s}2.txt') for s in ('train', 'dev', 'test')}
    types = sorted({t for _, spans in datasets['train'] for t, _, _ in spans})
    source_hashes = {n: sha256(ROOT/'data'/n) for n in ('train2.txt','dev2.txt','test2.txt','label.txt')}
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'data_sha256': source_hashes,
              'script_sha256': sha256(__file__), 'neural_training': False,
              'test_use': 'Dataset identity, duplicate and coverage audit only; no model inference or selection',
              'splits': {}, 'overlap': {}, 'runs': {}, 'resources': {}}
    for split, samples in datasets.items():
        texts = Counter(t for t, _ in samples)
        report['splits'][split] = {'sentences': len(samples), 'entities': sum(len(g) for _,g in samples),
            'unique_texts': len(texts), 'duplicate_extra_rows': sum(c-1 for c in texts.values()),
            'type_support': dict(Counter(t for _,g in samples for t,_,_ in g))}
    text_maps = {s: defaultdict(list) for s in datasets}
    for s, samples in datasets.items():
        for sid, (text, spans) in enumerate(samples): text_maps[s][text].append((sid, spans))
    for a,b in (('train','dev'),('train','test'),('dev','test')):
        shared = text_maps[a].keys() & text_maps[b].keys()
        report['overlap'][f'{a}_{b}'] = {'shared_unique_texts': len(shared),
            f'{a}_rows': sum(len(text_maps[a][t]) for t in shared),
            f'{b}_rows': sum(len(text_maps[b][t]) for t in shared),
            'texts_with_any_label_disagreement': sum(len({tuple(sorted(g)) for s in (a,b) for _,g in text_maps[s][t]})>1 for t in shared)}
    prior_texts = set(text_maps['train']) | set(text_maps['dev'])
    seen_test = set(); kept=[]; excluded_prior=[]; excluded_internal=[]
    for sid,(text,_) in enumerate(datasets['test']):
        if text in prior_texts: excluded_prior.append(sid)
        elif text in seen_test: excluded_internal.append(sid)
        else: kept.append(sid)
        seen_test.add(text)
    write_json(OUT/'test_dedup_ids.json', {'rule':'Exact raw character text, no normalization; remove train/dev matches, then retain first test occurrence',
        'kept':kept,'excluded_seen_train_dev':excluded_prior,'excluded_internal_repeat':excluded_internal})
    report['test_dedup'] = {'retained':len(kept),'excluded_seen_train_dev':len(excluded_prior),'excluded_internal_repeat':len(excluded_internal)}
    train = datasets['train']; entity_words=set(); segmented=Counter(); mentions=defaultdict(Counter)
    for text, gold in train:
        entity_words.update(text[a:b] for _,a,b in gold)
        segmented.update(w.strip() for w in jieba.cut(text) if w.strip())
        for t,a,b in gold: mentions[text[a:b]][t]+=1
    chars = ['<pad>','<unk>'] + sorted({c for text,_ in train for c in text})
    best_hashes={}
    for name in RUNS:
        run=ROOT/'output'/name
        cfg_record=json.loads((run/'run_config.json').read_text()); cfg=cfg_record['config']
        actual=json.loads((run/'lexicon.json').read_text())
        words=['<pad>','<unk>']+sorted(w for w in entity_words | {w for w,c in segmented.items() if c>=cfg['min_segment_frequency']}
                                    if 0<len(w)<=cfg['max_word_length'] and w not in ('<pad>','<unk>'))
        vocab_ok=words==actual['words']; char_ok=chars==actual.get('characters')
        ckpt=run/'best.pt';best_hashes[str(ckpt)]=sha256(ckpt)
        source_dir=run/'source_snapshot'
        source_checks={n:(sha256(source_dir/n)==v if (source_dir/n).exists() else None)
                       for n,v in cfg_record.get('source_hashes',{}).items()}
        row={'path':str(run),'run_config_sha256':sha256(run/'run_config.json'),
             'lexicon_sha256':sha256(run/'lexicon.json'),'lexicon_matches_record':sha256(run/'lexicon.json')==cfg_record['lexicon_sha256'],
             'rebuilt_words_equal':vocab_ok,'rebuilt_characters_equal':char_ok,
             'vocabulary_entries':len(words),'character_entries':len(chars),
             'max_word_length':cfg['max_word_length'],'best_pt_sha256':best_hashes[str(ckpt)],
             'parent_checkpoint':cfg.get('warm_start'), 'parent_recorded_sha256':cfg.get('warm_start_sha256'),
             'config':cfg,'environment':cfg_record.get('environment'),
             'effective_batch_size':cfg_record.get('effective_batch_size'),
             'model_selection':cfg_record.get('model_selection'),
             'test_read_by_train':cfg_record.get('test_read_by_train'),
             'source_hash_checks':source_checks,
             'data_hashes_match':all(cfg_record['input_identity'][n]==source_hashes[n] for n in ('train2.txt','dev2.txt','label.txt'))}
        write_json(OUT/'run_records'/f'{name}.json',cfg_record)
        report['runs'][name]=row
        print(name,'vocab',vocab_ok,'chars',char_ok,'source',all(v is True for v in source_checks.values()),flush=True)
    for row in report['runs'].values():
        if row['parent_checkpoint']:
            row['parent_file_matches_record']=best_hashes.get(row['parent_checkpoint'])==row['parent_recorded_sha256']
    final_vocab=json.loads((ROOT/'output/leman_fgm_seed46_dual4090/lexicon.json').read_text())
    vocab_set=set(final_vocab['words'][2:])
    for split, samples in datasets.items():
        gold=[text[a:b] for text,g in samples for _,a,b in g]
        report['splits'][split].update({'gold_mentions_in_lexicon':sum(w in vocab_set for w in gold),
            'gold_mentions_seen_train_surface':sum(w in entity_words for w in gold),
            'gold_mentions_unseen_train_surface':sum(w not in entity_words for w in gold)})
    gaz=json.loads((FROZEN/'v6_train_gazetteer.json').read_text())
    report['resources']['gazetteer']={'sha256':sha256(FROZEN/'v6_train_gazetteer.json'),
        'train_sha256_matches':gaz['train_sha256']==source_hashes['train2.txt'],
        'rebuilt_counts_equal':{w:dict(c) for w,c in mentions.items()}==gaz['surfaces'],
        'unique_train_gold_surfaces':len(mentions)}
    prior_src=FROZEN/'source_snapshot/context_priors_v22.py'
    spec=importlib.util.spec_from_file_location('priors',prior_src);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    indexed=[(text,[(types.index(t),a,b) for t,a,b in gold]) for text,gold in train]
    rebuilt,info=mod.build_priors(indexed,len(types));actual=joblib.load(FROZEN/'train_boundary_priors.joblib')
    exact=rebuilt==actual
    report['resources']['boundary_priors']={'sha256':sha256(FROZEN/'train_boundary_priors.joblib'),
        'rebuild_source_sha256':sha256(prior_src),'all_values_equal':exact,'rebuild':info}
    counts=np.array([sum(c[t] for c in mentions.values()) for t in types])
    expected=np.clip(((np.median(counts)+1)/(counts+1))**.25,1,1.5)
    report['resources']['type_weights']={'types':types,'counts':counts.tolist(),'weights':expected.tolist(),
        'all_finetune_records_equal':all(np.array_equal(expected,np.asarray(row['config']['gp_type_weights']))
            for row in report['runs'].values() if 'gp_type_weights' in row['config'])}
    deps=json.loads((FROZEN/'DEPENDENCIES.json').read_text())
    report['candidate_sources']={}
    for name, dep in deps.items():
        c=dep['header'].get('checkpoint',{})
        report['candidate_sources'][name]={'cache_sha256_matches':sha256(FROZEN/'caches'/f'{name}.jsonl.gz')==dep['cache_sha256'],
            'checkpoint_metadata':c}
    report['gpu']=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],capture_output=True,text=True).stdout.splitlines()
    write_json(OUT/'RESOURCE_AUDIT.json',report)
    print('COMPLETE',report['test_dedup'],report['resources']['gazetteer']['rebuilt_counts_equal'],exact,flush=True)


if __name__=='__main__':main()
