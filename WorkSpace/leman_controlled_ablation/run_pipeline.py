"""Launch or resume one variant/seed, then collect compact reviewer evidence."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from datetime import datetime, timezone
import zipfile

HERE = Path(__file__).resolve().parent
VARIANTS = ('F', 'no_lexicon', 'no_attention', 'dot_attention', 'crf', 'max_words3', 'max_words5', 'encoder_gp')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_package():
    manifest = json.loads((HERE/'PACKAGE_MANIFEST.json').read_text())
    for name, expected in manifest['files'].items():
        if sha(HERE/name) != expected:
            raise ValueError(f'Package file changed: {name}')
    return {p.name: sha(p) for p in sorted(HERE.glob('*.py'))}


def collect(folder):
    receipt = {}
    for path in sorted(folder.rglob('*.pt')):
        receipt[str(path.relative_to(folder))] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
    (folder/'CHECKPOINTS.json').write_text(json.dumps(receipt, indent=2)+'\n')
    paths = [p for p in sorted(folder.rglob('*')) if p.is_file()
             and p.suffix in ('.json', '.jsonl', '.csv', '.log', '.py', '.sh', '.md', '.txt')]
    manifest = {'files': {str(p.relative_to(folder)): sha(p) for p in paths},
                'weights_included': False, 'test_run_by_collector': False}
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    target = folder.parent/f'return_controlled_{folder.name}_{stamp}.zip'
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in paths:
            z.write(path, str(path.relative_to(folder)))
        z.writestr('RETURN_MANIFEST.json', json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(f'Collected: {target}\nSHA256: {sha(target)}', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='/root/data/WorkSpace')
    p.add_argument('--variant', choices=VARIANTS, default='F')
    p.add_argument('--seed', type=int, choices=[42, 43, 44], default=42)
    p.add_argument('--collect-only', action='store_true')
    args = p.parse_args()
    root = Path(args.root).resolve()
    folder = root/'output/leman_controlled_ablation_v1'/f'{args.variant}_seed{args.seed}'
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/'.pipeline.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.collect_only:
            collect(folder)
            return 0
        sources = verify_package()
        stopping = False
        def stop(signum, frame):
            nonlocal stopping
            stopping = True
            (folder/'STOP').touch()
            print('\nStop requested; waiting for the current optimizer step and checkpoint save.', flush=True)
        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
        env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                   TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1', LEMAN_GP_WORLD_SIZE='2')
        env.setdefault('CUDA_VISIBLE_DEVICES', '0,1')
        for phase in 'ABC':
            target = folder/phase
            target.mkdir(exist_ok=True)
            if stopping or (folder/'STOP').exists() or (target/'STOP').exists():
                print(f'Remove STOP before resuming: {folder}', flush=True)
                return 75
            receipt = target/'COMPLETE.json'
            if receipt.exists():
                saved = json.loads(receipt.read_text())
                if saved['source_hashes'] != sources or sha(target/'best.pt') != saved['best_sha256']:
                    raise ValueError(f'Completed phase {phase} differs from its receipt')
                print(f'Using completed {phase}: best epoch {saved["best_epoch"]}', flush=True)
                continue
            command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
                       '--nproc_per_node=2', str(HERE/'train.py'), '--root', str(root),
                       '--variant', args.variant, '--seed', str(args.seed), '--phase', phase]
            print(f'Starting phase {phase}; automatically resumes last.pt when present.', flush=True)
            process = subprocess.Popen(command, cwd=HERE, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, bufsize=0, start_new_session=True)
            with (target/'train.log').open('ab', buffering=0) as log:
                while True:
                    chunk = os.read(process.stdout.fileno(), 65536)
                    if not chunk:
                        break
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
                    # Preserve terminal carriage returns; store readable log lines.
                    log.write(chunk.replace(b'\r', b'\n'))
            rc = process.wait()
            if rc or not receipt.exists():
                print(f'Pipeline stopped at {phase}. Rerun this command to resume; completed phases are retained.', flush=True)
                return 75 if stopping or (folder/'STOP').exists() else (rc or 1)
        final = json.loads((folder/'C/COMPLETE.json').read_text())
        (folder/'SELECTED_MODEL.json').write_text(json.dumps({
            'checkpoint': str(folder/'C/best.pt'), 'sha256': final['best_sha256'],
            'variant': args.variant, 'seed': args.seed, 'raw_dev_micro_f1': final['best_dev_micro_f1'],
            'stage': 'C', 'best_epoch_in_stage': final['best_epoch'],
            'test_evaluated': False}, indent=2)+'\n')
        collect(folder)
        print('A/B/C complete. Selected model and dev predictions saved. Test was not evaluated.', flush=True)
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
