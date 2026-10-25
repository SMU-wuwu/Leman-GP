"""Evaluate best.pt on both GPUs once; no epochs or repeated model forwards."""
import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist

from config import Config
from data import load_split, prepare_split, sha256, write_json
from evaluate import evaluate
from model import LemanGP
from runtime import broadcast_object, environment, rank, setup, source_hashes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--split', choices=['test', 'dev'], default='test')
    args = parser.parse_args()
    device = setup()
    try:
        requested = Config.from_environment()
        run_dir = Path(requested.output_dir)
        path = Path(args.checkpoint) if args.checkpoint else run_dir / 'best.pt'
        # Inference checkpoint contains weights, not optimizer state; load on CPU first.
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        if checkpoint.get('schema') != requested.schema:
            raise ValueError('This evaluator requires this package\'s character-level GP checkpoint.')
        if checkpoint['source_hashes'] != source_hashes(Path(__file__).parent):
            raise ValueError('Use the same source files as training (available in source_snapshot).')
        cfg = Config(**checkpoint['config'])
        cfg.root, cfg.data_dir, cfg.encoder_dir = requested.root, requested.data_dir, requested.encoder_dir
        cfg.output_dir, cfg.workers = str(run_dir), requested.workers
        cfg.preprocessing_workers = requested.preprocessing_workers
        if requested.eval_batch_size:
            cfg.eval_batch_size = requested.eval_batch_size
        lexicon_path = run_dir / 'lexicon.json'
        if sha256(lexicon_path) != checkpoint['lexicon_sha256']:
            raise ValueError('lexicon.json differs from the training vocabulary.')
        lexicon = json.loads(lexicon_path.read_text(encoding='utf-8'))
        for name, expected in checkpoint['input_identity']['encoder_files'].items():
            if sha256(Path(cfg.encoder_dir) / name) != expected:
                raise ValueError(f'Encoder/tokenizer file changed: {name}. Copy original model files for inference.')
        types = checkpoint['types']
        cache = broadcast_object(str(prepare_split(cfg, args.split, lexicon, types)) if rank() == 0 else None)
        dataset = load_split(cache)
        model = LemanGP(cfg, len(types), checkpoint['vocab_size'],
                        encoder_config=checkpoint['encoder_config'], pretrained=False)
        model.load_state_dict(checkpoint['model'], strict=True)
        best_epoch = checkpoint['best_epoch']
        del checkpoint
        model.to(device)
        if rank() == 0:
            print(f'Test uses 2 CUDA GPUs: {torch.cuda.get_device_name(device)}; BF16; '
                  f'{cfg.eval_batch_size}/GPU; one forward/batch; checkpoint epoch {best_epoch}', flush=True)
        result_dir = run_dir / f'{args.split}_evaluation'
        metrics = evaluate(model, dataset, cfg, device, args.split, result_dir)
        if rank() == 0:
            metrics.update({'checkpoint_sha256': sha256(path), 'checkpoint_epoch': best_epoch,
                            'data_sha256': sha256(Path(cfg.data_dir) / f'{args.split}2.txt'),
                            'environment': environment(device)})
            write_json(result_dir / f'{args.split}_results.json', metrics)
            print(f'{args.split}: P={metrics["micro"]["precision"]:.6f}, R={metrics["micro"]["recall"]:.6f}, '
                  f'Micro F1={metrics["micro"]["f1"]:.6f}, Macro F1={metrics["macro_f1"]:.6f}', flush=True)
            print(f'Artifacts: {result_dir}', flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
