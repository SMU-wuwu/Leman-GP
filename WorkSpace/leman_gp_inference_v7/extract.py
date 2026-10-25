"""Expand dev candidates using existing GPU checkpoints; do not train."""
import gc
import gzip
import json
from pathlib import Path
import time

import torch
import torch.distributed as dist

from common import PACKAGE, paths, read_json, sha256, verify_package
from config import Config
from data import write_json
from evaluate import release_loader
from features import collect_features
from inference import load_candidate
from runtime import barrier, broadcast_object, environment, rank, setup


def main():
    manifest = verify_package()
    policy = read_json(PACKAGE / 'POLICY.json')
    root, output = paths()
    requested = Config.from_environment()
    requested.output_dir = str(root / 'output/leman_finetune_v4_seed42')
    requested.warm_start = str(root / 'output/leman_f1_4x4090_seed42/best.pt')
    device = setup()
    output.mkdir(parents=True, exist_ok=True)
    try:
        for name in ('upgraded', 'parent'):
            target = output / f'{name}_features.jsonl.gz'
            completed = broadcast_object(target.exists() if rank() == 0 else None)
            if completed:
                if rank() == 0:
                    with gzip.open(target, 'rt', encoding='utf-8') as stream:
                        header = json.loads(next(stream))
                    assert header['checkpoint']['checkpoint_sha256'] == manifest['checkpoint_sha256'][name]
                    assert header['policy_sha256'] == sha256(PACKAGE / 'POLICY.json')
                    print(f'Reuse completed inference cache: {target}', flush=True)
                continue
            started = time.perf_counter()
            model, dataset, cfg, identity = load_candidate(requested, name, 'dev', device)
            assert identity['checkpoint_sha256'] == manifest['checkpoint_sha256'][name]
            records = collect_features(model, dataset, cfg, device,
                                       output / 'shards' / name,
                                       floor=policy['raw_candidate_floor'])
            if rank() == 0:
                header = {'types': dataset.types, 'checkpoint': identity,
                          'dev_sha256': sha256(Path(cfg.data_dir) / 'dev2.txt'),
                          'policy_sha256': sha256(PACKAGE / 'POLICY.json'),
                          'floor': policy['raw_candidate_floor'], 'environment': environment(device),
                          'seconds': time.perf_counter() - started, 'test_accessed': False}
                temporary = target.with_suffix(target.suffix + '.tmp')
                with gzip.open(temporary, 'wt', encoding='utf-8') as stream:
                    stream.write(json.dumps(header, ensure_ascii=False) + '\n')
                    for sid in range(len(dataset.samples)):
                        stream.write(json.dumps({'sid': sid, 'rows': sorted(records[sid])}) + '\n')
                temporary.replace(target)
                write_json(output / f'{name}_extraction.json', {
                    **header, 'cache_sha256': sha256(target),
                    'candidate_rows': sum(map(len, records.values()))})
                print(f'{name}: {sum(map(len, records.values()))} candidates, '
                      f'{header["seconds"]:.1f}s', flush=True)
            release_loader(dataset)
            del model, dataset, records
            gc.collect()
            torch.cuda.empty_cache()
            barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
