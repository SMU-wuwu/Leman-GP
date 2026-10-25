"""Inference from saved EMA shadows; optimizer state is never restored."""
import gc
import gzip
import json
from pathlib import Path
import shutil
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
    verify_package()
    policy = read_json(PACKAGE/'POLICY.json')
    root, output = paths()
    output.mkdir(parents=True, exist_ok=True)
    device = setup()
    requested = Config.from_environment()
    requested.output_dir = str(root/'output/leman_finetune_v4_seed42')
    requested.warm_start = str(root/'output/leman_f1_4x4090_seed42/best.pt')
    try:
        if rank() == 0:
            source = root/'output/leman_inference_v7_seed42/upgraded_features.jsonl.gz'
            shutil.copy2(source, output/'upgraded_features.jsonl.gz')
        model, dataset, cfg, base_identity = load_candidate(requested, 'upgraded', 'dev', device)
        for name, run, epoch in [('final_v4', 'leman_finetune_v4_seed42', 5),
                                  ('final_v2', 'leman_f1_4x4090_seed42', 12)]:
            target = output/f'{name}_features.jsonl.gz'
            if broadcast_object(target.exists() if rank() == 0 else None):
                continue
            started = time.perf_counter()
            origin = root/'output'/run/'last.pt'
            origin_sha = broadcast_object(sha256(origin) if rank() == 0 else None)
            checkpoint = torch.load(origin, map_location='cpu', mmap=True, weights_only=False)
            assert checkpoint['schema'] == 'leman-f1-v2' and checkpoint['epoch'] == epoch
            expected = read_json(PACKAGE/'frozen_v4'/'PARENT_SOURCE_HASHES.json') if name == 'final_v2' else {
                p.name: sha256(p) for p in (PACKAGE/'frozen_v4').glob('*.py')}
            assert checkpoint['source_hashes'] == expected
            assert checkpoint['types'] == dataset.types
            assert checkpoint['lexicon_sha256'] == base_identity['lexicon_sha256']
            assert checkpoint['input_identity']['dev2.txt'] == policy['dev_sha256']
            for field in ('max_chars', 'overlap_chars', 'max_span', 'max_encoder_tokens'):
                assert checkpoint['config'][field] == getattr(cfg, field)
            shadow = checkpoint['ema']['shadow']
            parameters = dict(model.named_parameters())
            assert set(shadow) == {name for name, p in parameters.items() if p.requires_grad}
            saved_state = checkpoint['model']
            assert set(saved_state) == set(model.state_dict())
            for key, value in shadow.items():
                assert value.shape == parameters[key].shape
                assert value.dtype == saved_state[key].dtype
            # Same semantics as transfer.EMA.model_state(): average trainable
            # parameters, retaining checkpoint buffers and frozen parameters.
            ema_state = {key: shadow.get(key, value) for key, value in saved_state.items()}
            model.load_state_dict(ema_state, strict=True)
            ema_updates = checkpoint['ema']['updates']
            del parameters, ema_state, saved_state, shadow, checkpoint
            gc.collect()
            records = collect_features(model, dataset, cfg, device, output/'shards'/name,
                                       floor=policy['raw_candidate_floor'])
            if rank() == 0:
                header = {'types': dataset.types, 'checkpoint': {
                    'candidate': name, 'origin': str(origin), 'origin_sha256': origin_sha,
                    'checkpoint_epoch': epoch, 'weight_source': 'ema.shadow', 'ema_updates': ema_updates},
                    'dev_sha256': policy['dev_sha256'], 'policy_sha256': sha256(PACKAGE/'POLICY.json'),
                    'floor': policy['raw_candidate_floor'], 'environment': environment(device),
                    'seconds': time.perf_counter()-started, 'test_accessed': False}
                temp = target.with_suffix('.gz.tmp')
                with gzip.open(temp, 'wt', encoding='utf-8') as stream:
                    stream.write(json.dumps(header, ensure_ascii=False)+'\n')
                    for sid in range(len(dataset.samples)):
                        stream.write(json.dumps({'sid': sid, 'rows': sorted(records[sid])})+'\n')
                temp.replace(target)
                write_json(output/f'{name}_extraction.json', {**header, 'cache_sha256': sha256(target)})
                print(f'{name} completed in {header["seconds"]:.1f}s', flush=True)
            del records
            barrier()
        release_loader(dataset)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
