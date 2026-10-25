"""Check frozen candidate cache identities against all current source files."""
from pathlib import Path
import json
from evidence_core import sha256, write_json

ROOT=Path('/root/data/WorkSpace')
FROZEN=ROOT/'output/leman_final_v2_20261019'
OUT=ROOT/'output/manuscript_revision_20261019/resource_audit'


def main():
    deps=json.loads((FROZEN/'DEPENDENCIES.json').read_text())
    audit=json.loads((OUT/'RESOURCE_AUDIT.json').read_text())
    hashed={str(ROOT/'output'/n/'best.pt'):r['best_pt_sha256'] for n,r in audit['runs'].items()}
    rows={}
    for name,dep in deps.items():
        h=dep['header'];c=h.get('checkpoint',{})
        if name.startswith('soup'):
            path=ROOT/'output/leman_soup_v16'/f'{name}.pt';expected=h['checkpoint_sha256']
        elif 'origin' in c:path=Path(c['origin']);expected=c['origin_sha256']
        else:path=Path(c['checkpoint']);expected=c['checkpoint_sha256']
        if str(path) not in hashed:hashed[str(path)]=sha256(path)
        row={'path':str(path),'bytes':path.stat().st_size,'sha256':hashed[str(path)],'matches_frozen_cache':hashed[str(path)]==expected,
             'cache_sha256':sha256(FROZEN/'caches'/f'{name}.jsonl.gz'),
             'weight_source':c.get('weight_source',h.get('recipe'))}
        if 'parents' in h:
            row['soup_parents']={n:{**r,'actual_sha256':hashed[r['path']],'matches':hashed[r['path']]==r['sha256']} for n,r in h['parents'].items()}
        rows[name]=row
        print(name,row['matches_frozen_cache'],flush=True)
    write_json(OUT/'CANDIDATE_LINEAGE.json',{'sources':rows,'all_checkpoint_hashes_match':all(r['matches_frozen_cache'] for r in rows.values()),
        'distinct_checkpoint_files':len({r['path'] for r in rows.values()}),
        'independent_training_replicates':False,
        'reason':'Sources include shared-parent finetunes, EMA snapshots and averaged weights; source count is not seed count',
        'script_sha256':sha256(__file__)})
    encoder_cfg=json.loads((ROOT/'model/config.json').read_text())
    root_record=json.loads((OUT/'run_records/leman_paper_a100_seed42.json').read_text())
    expected=root_record['input_identity']['encoder_weights']
    checks={n:{'sha256':sha256(ROOT/'model'/n),'expected':v} for n,v in expected.items()}
    write_json(OUT/'ENCODER_IDENTITY.json',{'config':encoder_cfg,'config_sha256':sha256(ROOT/'model/config.json'),
        'pretrained_weight_checks':checks,'all_weight_hashes_match':all(v['sha256']==v['expected'] for v in checks.values()),
        'manuscript_action':'Describe the actual ERNIE 3.0 causal-language-model backbone; do not reuse ERNIE 2.0 three-level masking figure as its pretraining diagram'})
    # The root paper model has no character residual branch or saved character table.
    root_lex=json.loads((ROOT/'output/leman_paper_a100_seed42/lexicon.json').read_text())
    write_json(OUT/'ROOT_CHARACTER_TABLE_NOTE.json',{'root_lexicon_has_characters':'characters' in root_lex,
        'interpretation':'rebuilt_characters_equal=false in the root audit denotes an absent table, not a mismatched trained character vocabulary; all seven later character vocabularies match train reconstruction'})


if __name__=='__main__':main()
