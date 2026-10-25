"""Inspect the completed seed44 run and archive metadata, never weights."""
import gc
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT/'leman_gp_fgm_v10'))
from runtime import source_hashes
from data import sha256, write_json


def read(path):
    return json.loads(path.read_text())


def main():
    run = ROOT/'output/leman_fgm_v14_seed44'
    out = ROOT/'output/leman_fgm_v14_analysis_seed44'
    out.mkdir(exist_ok=True)
    plan_path = ROOT/'leman_gp_fgm_seed44_v14/EXPERIMENT_PLAN.json'
    plan = read(plan_path)
    assert read(run/'execution_plan.json') == {'plan_sha256': sha256(plan_path), 'plan': plan}
    results, config, history = [read(run/name) for name in
                                ('training_results.json', 'run_config.json', 'history.json')]
    assert results['seed'] == 44 and results['completed_epochs'] == 3
    assert results['global_step'] == 1146 and not results['test_evaluated']
    assert results['per_gpu_batch_size'] == 32 and results['effective_batch_size'] == 128
    assert not config['test_read_by_train'] and config['environment']['world_size'] == 4
    assert config['adversarial']['epsilon'] == 0.5
    assert config['adversarial']['clean_weight'] == 0.5
    assert config['source_hashes'] == source_hashes(ROOT/'leman_gp_fgm_v10')
    for name, digest in config['source_hashes'].items():
        assert sha256(run/'source_snapshot'/name) == digest
    cfg = config['config']
    assert (cfg['seed'], cfg['epochs'], cfg['batch_size']) == (44, 3, 32)
    assert cfg['encoder_lr'] == 3e-6 and cfg['head_lr'] == cfg['reused_head_lr'] == 1e-5
    assert cfg['ema_decay'] == 0.995
    assert sha256(Path(cfg['warm_start'])) == plan['training']['parent_checkpoint_sha256']
    for name in ('train2.txt', 'dev2.txt', 'label.txt'):
        assert sha256(ROOT/'data'/name) == config['input_identity'][name]
    assert sha256(run/'lexicon.json') == config['lexicon_sha256']
    assert [row['epoch'] for row in history] == [1, 2, 3]
    best = torch.load(run/'best.pt', map_location='cpu', weights_only=False)
    assert best['source_hashes'] == config['source_hashes']
    assert best['input_identity'] == config['input_identity']
    assert best['config'] == cfg and best['best_epoch'] == results['best_epoch']
    assert best['best_f1'] == best['dev']['micro']['f1'] == results['best_dev_micro_f1']
    assert best['best_f1'] == max([read(run/'parent_baseline.json')['micro']['f1']] +
                                [row[k]['micro']['f1'] for row in history for k in ('dev_live','dev_ema')])
    assert all(torch.isfinite(t).all() for t in best['model'].values() if t.is_floating_point())
    record = {'status':'PASS','checkpoint_sha256':sha256(run/'best.pt'),
              'best_epoch':best['best_epoch'],'weight_source':best['weight_source'],
              'best_micro':best['best_f1'],'input_identity':best['input_identity'],
              'source_files':len(best['source_hashes']),'test_accessed':False}
    best_state = best['model']
    del best
    gc.collect()
    last = torch.load(run/'last.pt', map_location='cpu', weights_only=False)
    assert (last['epoch'],last['next_batch'],last['global_step'],last['world_size']) == (3,0,1146,4)
    assert len(last['rng_by_rank']) == 4 and len(last['partial_by_rank']) == 4
    assert last['optimizer']['state'] and last['scheduler'] and last['ema']
    assert last['source_hashes'] == config['source_hashes'] and last['history'] == history
    assert last['best_f1'] == results['best_dev_micro_f1']
    assert all(torch.equal(t,last['best_checkpoint']['model'][k]) for k,t in best_state.items())
    record.update(last_epoch=3,global_step=1146,rng_ranks=4,
                  last_checkpoint_sha256=sha256(run/'last.pt'),
                  execution_plan_sha256=sha256(run/'execution_plan.json'))
    write_json(out/'checkpoint_audit.json',record)
    write_json(run/'checkpoint_audit.json',record)
    files = [p for p in run.rglob('*') if p.is_file() and 'cache' not in p.relative_to(run).parts
             and p.suffix not in ('.pt','.lock')]
    archive = ROOT/'output/return_leman_fgm_v14_training.zip'
    manifest = {'files':{str(p.relative_to(run)):sha256(p) for p in files},'test_evaluated':False}
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p,str(p.relative_to(run)))
        z.writestr('RETURN_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(record,indent=2),flush=True)
    print(f'Training metadata: {archive}',flush=True)


if __name__ == '__main__':
    main()
