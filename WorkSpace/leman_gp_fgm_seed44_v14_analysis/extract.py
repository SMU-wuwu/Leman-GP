"""Extract dev candidates from the completed FGM best.pt; no training or test."""
import gc
import gzip
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'leman_gp_fgm_v10'))

import torch
import torch.distributed as dist
from config import Config
from data import load_split, prepare_split, sha256, write_json
from evaluate import release_loader
from features import collect_features
from model import LemanGP
from runtime import barrier, broadcast_object, environment, rank, setup, source_hashes


def main():
    device = setup()
    requested = Config.from_environment()
    run = Path(requested.root)/'output/leman_fgm_v14_seed44'
    output = Path(requested.root)/'output/leman_fgm_v14_analysis_seed44'
    output.mkdir(parents=True, exist_ok=True)
    try:
        path = run/'best.pt'
        digest = broadcast_object(sha256(path) if rank() == 0 else None)
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        assert ckpt['source_hashes'] == source_hashes(HERE.parent/'leman_gp_fgm_v10')
        assert ckpt['schema'] == 'leman-f1-v2'
        cfg = Config(**ckpt['config'])
        assert cfg.seed == 44 and cfg.epochs == 3
        assert digest == json.loads((output/'checkpoint_audit.json').read_text())['checkpoint_sha256']
        cfg.root, cfg.data_dir, cfg.encoder_dir = requested.root, requested.data_dir, requested.encoder_dir
        assert sha256(run/'lexicon.json') == ckpt['lexicon_sha256']
        for name in ('train2.txt', 'dev2.txt', 'label.txt'):
            assert sha256(Path(cfg.data_dir)/name) == ckpt['input_identity'][name]
        for group in ('encoder_files', 'encoder_weights'):
            for name, expected in ckpt['input_identity'][group].items():
                assert sha256(Path(cfg.encoder_dir)/name) == expected
        lexicon = json.loads((run/'lexicon.json').read_text())
        cached = broadcast_object(str(prepare_split(cfg, 'dev', lexicon, ckpt['types'])) if rank() == 0 else None)
        dataset = load_split(cached)
        model = LemanGP(cfg, len(ckpt['types']), ckpt['vocab_size'],
                        char_vocab_size=ckpt['char_vocab_size'],
                        encoder_config=ckpt['encoder_config'], pretrained=False)
        model.load_state_dict(ckpt['model'], strict=True)
        header = {'types': ckpt['types'], 'checkpoint': {
            'checkpoint': str(path), 'checkpoint_sha256': digest,
            'checkpoint_epoch': ckpt['best_epoch'], 'weight_source': ckpt['weight_source'],
            'dev': ckpt['dev'], 'source_hashes': ckpt['source_hashes'],
            'lexicon_sha256': ckpt['lexicon_sha256']},
            'dev_sha256': ckpt['input_identity']['dev2.txt'],
            'plan_sha256': sha256(HERE.parent/'leman_gp_fgm_seed44_v14/EXPERIMENT_PLAN.json'),
            'floor': -4.0, 'environment': environment(device), 'test_accessed': False}
        del ckpt
        gc.collect()
        model.to(device)
        started = time.perf_counter()
        records = collect_features(model, dataset, cfg, device, output/'shards/fgm', floor=-4.0)
        if rank() == 0:
            target = output/'fgm_features.jsonl.gz'
            temporary = target.with_suffix('.gz.tmp')
            with gzip.open(temporary, 'wt', encoding='utf-8') as stream:
                stream.write(json.dumps(header, ensure_ascii=False)+'\n')
                for sid in range(len(dataset.samples)):
                    stream.write(json.dumps({'sid': sid, 'rows': sorted(records[sid])})+'\n')
            temporary.replace(target)
            write_json(output/'extraction.json', {**header, 'seconds': time.perf_counter()-started,
                                                  'cache_sha256': sha256(target)})
            print(f'FGM dev candidates saved: {target}', flush=True)
        release_loader(dataset)
        barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
