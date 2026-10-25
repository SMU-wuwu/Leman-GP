"""Exercise the real trainer on tiny synthetic corpora, including exact resume."""
import argparse
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import torch
from qa import fixture
from data import write_json

HERE = Path(__file__).resolve().parent


def equal(a, b, name='root'):
    if torch.is_tensor(a):
        assert torch.equal(a, b), name
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), name
        for k in a:
            equal(a[k], b[k], name+'.'+str(k))
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b), name
        for i, (x, y) in enumerate(zip(a, b)):
            equal(x, y, name+f'[{i}]')
    else:
        assert a == b, name


def launch(root, phase, variant='F', interrupt=None):
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               TOKENIZERS_PARALLELISM='false', LEMAN_GP_WORLD_SIZE='2')
    env.pop('LEMAN_QA_STOP_AFTER', None)
    if interrupt is not None:
        env['LEMAN_QA_STOP_AFTER'] = str(interrupt)
    cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1', '--nproc_per_node=2',
           str(HERE/'train.py'), '--root', str(root), '--variant', variant, '--seed', '42',
           '--phase', phase, '--qa-config', str(root/'qa_config.json')]
    log = root/f'{variant}_{phase}_{"interrupt" if interrupt else "run"}.log'
    with log.open('ab') as f:
        rc = subprocess.call(cmd, cwd=HERE, env=env, stdout=f, stderr=subprocess.STDOUT)
    if interrupt is None and rc:
        raise RuntimeError(f'QA trainer failed: {log}')
    if interrupt is not None and not rc:
        raise AssertionError('The synthetic interruption did not stop the trainer')
    return root/'output/leman_controlled_ablation_v1'/f'{variant}_seed42'/phase


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    base = args.output.resolve()
    base.mkdir(parents=True, exist_ok=False)
    complete, resumed = base/'uninterrupted', base/'resumed'
    fixture(complete)
    shutil.copytree(complete, resumed)
    uninterrupted = launch(complete, 'A')
    interrupted = launch(resumed, 'A', interrupt=2)
    state = torch.load(interrupted/'last.pt', map_location='cpu', weights_only=False)
    assert state['global_step'] == 2 and state['epoch'] == 0 and state['next_batch'] == 2
    assert not (interrupted/'COMPLETE.json').exists()
    del state
    finished = launch(resumed, 'A')
    a = torch.load(uninterrupted/'last.pt', map_location='cpu', weights_only=False)
    b = torch.load(finished/'last.pt', map_location='cpu', weights_only=False)
    for key in ('model', 'optimizer', 'scheduler', 'ema', 'global_step', 'epoch', 'next_batch', 'best_f1', 'best_epoch'):
        equal(a[key], b[key], key)
    equal(a['best_checkpoint']['model'], b['best_checkpoint']['model'], 'best_model')
    assert (finished/'epoch_001.pt').exists() and (finished/'epoch_002.pt').exists()
    for x, y in zip(a['history'], b['history']):
        for key in ('dev_live', 'dev_ema'):
            equal(x[key]['micro'], y[key]['micro'], key)
        equal(x['train_loss'], y['train_loss'], 'epoch_loss')
    del a, b
    results = {'interruption_at_step2': True, 'resume_weights_optimizer_scheduler_EMA_best_exact': True,
               'epoch_loss_and_dev_counts_exact': True, 'periodic_snapshots_present': True}
    for phase in ('B', 'C'):
        target = launch(complete, phase)
        receipt = json.loads((target/'COMPLETE.json').read_text())
        assert receipt['completed_epochs'] == 2
        entry = json.loads((target/'entry_baseline.json').read_text())
        assert receipt['best_dev_micro_f1'] >= entry['micro']['f1']
        results[f'F_phase_{phase}'] = 'PASS: same-seed stage transfer and entry candidate'
    for phase in 'ABC':
        target = launch(complete, phase, 'crf')
        assert (target/'best_dev/dev_predictions.jsonl').exists()
        results[f'CRF_phase_{phase}'] = 'PASS: real trainer and full-sentence decoder'
    results.update(real_task_data_read=False, synthetic_only=True)
    write_json(base/'PIPELINE_QA.json', results)
    print(json.dumps(results), flush=True)


if __name__ == '__main__':
    main()
