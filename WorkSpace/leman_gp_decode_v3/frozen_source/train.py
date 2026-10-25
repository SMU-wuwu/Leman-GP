"""Four-GPU training. Re-running the same command automatically resumes last.pt."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import shutil
import signal
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from config import Config
from data import build_lexicon, entity_types, load_split, prepare_split, sha256, write_json, Collator
from evaluate import evaluate
from model import LemanGP
from transfer import EMA, load_parent, optimizer_groups, transfer_weights
from runtime import (NUMERICAL_POLICY, ResumeBatchSampler, all_ranks_finite, amp_context,
                     atomic_save, barrier, broadcast_object, choose_batch_size,
                     clip_gradients_fp64, environment, gather_objects, gradient_norm_fp64,
                     input_identity, optimizer_to, rank, restore_rng, rng_state,
                     seed_everything, setup, source_hashes, sum_tensor, to_device,
                     world_size, write_numerical_failure)


STOP_REQUESTED = False


def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def stop_requested(output, device):
    flag = STOP_REQUESTED or (Path(output) / 'STOP').exists()
    value = torch.tensor(int(flag), device=device)
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return bool(value.item())


def select_resume(output, option):
    if option == 'none':
        if any(output.glob('*.pt')):
            raise FileExistsError(f'{output} contains checkpoints. Use automatic resume or a new output path.')
        return None
    if option != 'auto':
        path = Path(option).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    if (output / 'last.pt').exists():
        return output / 'last.pt'
    snapshots = list(output.glob('epoch_*.pt'))
    if snapshots:
        return max(snapshots, key=lambda p: int(p.stem.split('_')[-1]))
    if (output / 'best.pt').exists():
        raise RuntimeError('Only best.pt is present. Restore last.pt or an epoch checkpoint to resume training.')
    return None


def resumed_config(requested, checkpoint):
    saved = Config(**checkpoint['config'])
    # Runtime paths/CPU workers may change after migration. Training settings remain frozen.
    names = {'SEED': 'seed', 'EPOCHS': 'epochs', 'BATCH_SIZE': 'batch_size',
             'ENCODER_LR': 'encoder_lr', 'HEAD_LR': 'head_lr', 'REUSED_HEAD_LR': 'reused_head_lr'}
    for suffix, name in names.items():
        if 'LEMAN_GP_' + suffix in os.environ and getattr(requested, name) != getattr(saved, name):
            raise ValueError(f'Resume requires {name}={getattr(saved, name)}. Use a new output directory for a changed setting.')
    for name in ('root', 'data_dir', 'encoder_dir', 'output_dir', 'warm_start', 'workers', 'preprocessing_workers'):
        setattr(saved, name, getattr(requested, name))
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', default='auto', help='auto (default), none, or a complete checkpoint path')
    args = parser.parse_args()
    device = setup()
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    cfg = Config.from_environment()
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    try:
        source = source_hashes(Path(__file__).parent)
        resume_path = broadcast_object(str(select_resume(output, args.resume) or '') if rank() == 0 else None)
        checkpoint = torch.load(resume_path, map_location='cpu', weights_only=False) if resume_path else None
        if checkpoint is not None:
            if checkpoint.get('schema') != cfg.schema or 'optimizer' not in checkpoint:
                raise ValueError('Use a full checkpoint created by this package; legacy BCE checkpoints cannot resume this model.')
            if checkpoint['source_hashes'] != source:
                raise ValueError('Source code differs from the saved run. Restore the run source_snapshot to resume.')
            if checkpoint['world_size'] != world_size():
                raise ValueError('Resume with the same four-GPU world size.')
            cfg = resumed_config(cfg, checkpoint)
        identity = broadcast_object(input_identity(cfg) if rank() == 0 else None)
        if checkpoint is not None and identity != checkpoint['input_identity']:
            raise ValueError('Training data, label map, tokenizer or encoder files changed since the checkpoint.')
        types = entity_types(Path(cfg.data_dir) / 'label.txt')
        parent_sha = None
        if checkpoint is None:
            actual_sha = sha256(cfg.warm_start) if rank() == 0 else None
            if rank() == 0 and actual_sha != cfg.warm_start_sha256:
                raise ValueError('Warm-start checkpoint SHA256 differs from the completed seed42 run.')
            parent_sha = broadcast_object(actual_sha)
        parent, parent_lexicon = (load_parent(cfg.warm_start, identity, types)
                                  if checkpoint is None else (None, None))
        lexicon_path = output / 'lexicon.json'
        if rank() == 0:
            if checkpoint is None and not lexicon_path.exists():
                write_json(lexicon_path, build_lexicon(cfg, types))
            elif not lexicon_path.exists():
                raise FileNotFoundError('Copy lexicon.json with the run when migrating a checkpoint.')
        barrier()
        lexicon = json.loads(lexicon_path.read_text(encoding='utf-8'))
        if lexicon['train_sha256'] != identity['train2.txt'] or lexicon['schema'] != cfg.schema:
            raise ValueError('The saved lexicon does not belong to this training dataset/protocol.')
        lexicon_sha = sha256(lexicon_path)
        if checkpoint is not None and lexicon_sha != checkpoint['lexicon_sha256']:
            raise ValueError('Lexicon IDs changed since the checkpoint.')
        paths = None
        if rank() == 0:
            paths = [str(prepare_split(cfg, split, lexicon, types)) for split in ('train', 'dev')]
        paths = broadcast_object(paths)
        train_data, dev_data = map(load_split, paths)
        # Record the actual code even when startup sizing fails before the
        # normal run_config/source_snapshot are written. Old run records remain.
        if rank() == 0:
            startup_snapshot = output / 'startup_source_snapshot'
            startup_snapshot.mkdir(exist_ok=True)
            for path in Path(__file__).parent.glob('*.py'):
                shutil.copy2(path, startup_snapshot / path.name)
            write_json(output / 'startup_record.json', {
                'config': cfg.to_dict(), 'environment': environment(device),
                'source_hashes': source, 'input_identity': identity,
                'numerical_policy': NUMERICAL_POLICY,
                'resume_from': resume_path or None,
                'stage': 'before_batch_selection', 'test_read_by_train': False,
            })
        barrier()
        seed_everything(cfg.seed)
        model = LemanGP(cfg, len(types), len(lexicon['words']),
                        char_vocab_size=len(lexicon['characters']),
                        encoder_config=(checkpoint or parent)['encoder_config'], pretrained=False)
        if parent is not None:
            reused, transfer = transfer_weights(model, parent, parent_lexicon, lexicon)
            transfer.update(parent_checkpoint=cfg.warm_start,
                            parent_checkpoint_sha256=parent_sha)
            del parent, parent_lexicon
            if rank() == 0:
                write_json(output / 'weight_transfer.json', transfer)
        else:
            transfer = checkpoint['weight_transfer']
            reused = set(transfer['copied_keys'])
        model.to(device)
        if checkpoint is not None:
            model.load_state_dict(checkpoint['model'], strict=True)
        if checkpoint is None:
            cfg.batch_size, batch_report = choose_batch_size(model, train_data, cfg, device)
        else:
            batch_report = {'mode': 'restored_from_checkpoint', 'per_gpu_batch_size': cfg.batch_size}
        cfg.eval_batch_size = cfg.eval_batch_size or cfg.batch_size
        wrapped = DDP(model, device_ids=[device.index], output_device=device.index,
                      broadcast_buffers=False, gradient_as_bucket_view=True)
        optimizer = torch.optim.AdamW(optimizer_groups(model, cfg, reused), fused=True)
        ema = EMA(model, cfg.ema_decay)
        sampler = DistributedSampler(train_data, num_replicas=world_size(), rank=rank(),
                                     shuffle=True, seed=cfg.seed, drop_last=False)
        steps_per_epoch = math.ceil(len(sampler) / cfg.batch_size)
        total_steps = steps_per_epoch * cfg.epochs
        warmup_steps = int(total_steps * cfg.warmup_fraction)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        epoch, next_batch, global_step, best_f1, best_epoch = 0, 0, 0, -1.0, 0
        history, partial_loss, partial_examples = [], 0.0, 0
        best_checkpoint = None
        if checkpoint is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            optimizer_to(optimizer, device)
            scheduler.load_state_dict(checkpoint['scheduler'])
            ema.load_state_dict(checkpoint['ema'])
            epoch, next_batch, global_step = checkpoint['epoch'], checkpoint['next_batch'], checkpoint['global_step']
            best_f1, best_epoch, history = checkpoint['best_f1'], checkpoint['best_epoch'], checkpoint['history']
            if checkpoint['steps_per_epoch'] != steps_per_epoch:
                raise ValueError('Number of training batches changed; restore the original configuration.')
            part = checkpoint['partial_by_rank'][rank()]
            partial_loss, partial_examples = part['loss_sum'], part['examples']
            restore_rng(checkpoint['rng_by_rank'][rank()])
            if rank() == 0:
                best_checkpoint = checkpoint.get('best_checkpoint')
                if best_epoch and best_checkpoint is None:
                    raise ValueError('The full checkpoint is missing its best-model payload.')
                if best_checkpoint is not None:
                    atomic_save(best_checkpoint, output / 'best.pt')
            checkpoint = None
            gc.collect()
        else:
            seed_everything(cfg.seed + rank())
        if rank() == 0:
            print(f'\nF1 v2: {world_size()} GPUs, {cfg.batch_size}/GPU, global batch {cfg.batch_size * world_size()}, BF16', flush=True)
            print(f'Encoder LR {cfg.encoder_lr:g}; reused head LR {cfg.reused_head_lr:g}; new head LR {cfg.head_lr:g}; epochs {cfg.epochs}; steps/epoch {steps_per_epoch}', flush=True)
            print('Numerics: encoder BF16 / automatic SDPA; custom layers FP32; gradient norm FP64; '
                  'no score clamping or loss replacement', flush=True)
            print('L1 distance: independent batch/head chunks; each CUDA backward workspace <=512 MiB', flush=True)
            print(f'Train: {train_data.stats}; Dev: {dev_data.stats}', flush=True)
            print(f'Resume: {resume_path or "new run"}; epoch={epoch + 1}, next batch={next_batch}, global step={global_step}', flush=True)
            snapshot = output / 'source_snapshot'
            snapshot.mkdir(exist_ok=True)
            for path in Path(__file__).parent.glob('*.py'):
                shutil.copy2(path, snapshot / path.name)
            write_json(output / 'run_config.json', {
                'config': cfg.to_dict(), 'environment': environment(device),
                'source_hashes': source, 'input_identity': identity, 'lexicon_sha256': lexicon_sha,
                'batch_selection': batch_report, 'steps_per_epoch': steps_per_epoch,
                'warmup_steps': warmup_steps, 'total_steps': total_steps,
                'effective_batch_size': cfg.batch_size * world_size(),
                'train_stats': train_data.stats, 'dev_stats': dev_data.stats,
                'loss': 'GP multilabel categorical CE + 0.1 * typed start/end multilabel categorical CE',
                'weight_transfer': transfer,
                'numerical_policy': NUMERICAL_POLICY,
                'model_selection': 'dev exact character-span micro F1, threshold=0; live and EMA',
                'test_read_by_train': False, 'resume_from': resume_path or None,
            })
        barrier()

        def save_state(next_epoch, next_index, loss_sum, examples, snapshot_epoch=None):
            states = gather_objects(rng_state())
            partials = gather_objects({'loss_sum': loss_sum, 'examples': examples})
            if rank() == 0:
                payload = {
                    'schema': cfg.schema, 'config': cfg.to_dict(), 'source_hashes': source,
                    'input_identity': identity, 'lexicon_sha256': lexicon_sha, 'world_size': world_size(),
                    'encoder_config': model.encoder.config.to_dict(), 'types': types,
                    'vocab_size': len(lexicon['words']), 'model': model.state_dict(),
                    'char_vocab_size': len(lexicon['characters']), 'weight_transfer': transfer,
                    'ema': ema.state_dict(),
                    'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                    'amp': 'bf16', 'scaler': None, 'epoch': next_epoch, 'next_batch': next_index,
                    'global_step': global_step, 'steps_per_epoch': steps_per_epoch,
                    'best_f1': best_f1, 'best_epoch': best_epoch, 'history': history,
                    'best_checkpoint': best_checkpoint,
                    'rng_by_rank': states, 'partial_by_rank': partials,
                }
                atomic_save(payload, output / 'last.pt')
                if snapshot_epoch is not None:
                    target = output / f'epoch_{snapshot_epoch:03d}.pt'
                    temporary = target.with_suffix('.tmp')
                    if temporary.exists():
                        temporary.unlink()
                    try:
                        os.link(output / 'last.pt', temporary)
                    except OSError:
                        shutil.copy2(output / 'last.pt', temporary)
                    os.replace(temporary, target)
            barrier()

        # Features are deterministic; reuse workers across epochs and evaluations.
        batch_sampler = ResumeBatchSampler(sampler, cfg.batch_size, 0)
        generator = torch.Generator().manual_seed(cfg.seed + 10000 + rank())
        kwargs = {'batch_sampler': batch_sampler, 'num_workers': cfg.workers, 'pin_memory': True,
                  'collate_fn': Collator(types, train_data.pad_id), 'generator': generator}
        if cfg.workers:
            kwargs.update(persistent_workers=True, prefetch_factor=2, multiprocessing_context='spawn')
        loader = DataLoader(train_data, **kwargs)
        for current_epoch in range(epoch, cfg.epochs):
            sampler.set_epoch(current_epoch)
            start_batch = next_batch if current_epoch == epoch else 0
            batch_sampler.start_batch = start_batch
            wrapped.train()
            torch.cuda.reset_peak_memory_stats(device)
            loss_sum = partial_loss if current_epoch == epoch else 0.0
            examples = partial_examples if current_epoch == epoch else 0
            started = time.perf_counter()
            bar = tqdm(loader, total=steps_per_epoch, initial=start_batch,
                       desc=f'Epoch {current_epoch + 1}/{cfg.epochs}', disable=rank() != 0,
                       mininterval=1, dynamic_ncols=True)
            for batch_index, batch in enumerate(bar, start_batch):
                optimizer.zero_grad(set_to_none=True)
                with amp_context(device):
                    loss, _ = wrapped(**to_device(batch, device))
                if not all_ranks_finite(loss):
                    diagnostic = write_numerical_failure(
                        output, model, stage='train_loss', step=global_step, loss=loss, batch=batch)
                    barrier()
                    raise FloatingPointError(f'Non-finite loss on one or more ranks at step {global_step}. '
                                             f'Rank {rank()} diagnostic: {diagnostic}')
                loss.backward()
                grad_norm = gradient_norm_fp64(model.parameters())
                if not all_ranks_finite(grad_norm):
                    diagnostic = write_numerical_failure(
                        output, model, stage='train_gradient', step=global_step,
                        loss=loss, norm=grad_norm, batch=batch)
                    barrier()
                    raise FloatingPointError(f'Non-finite gradient on one or more ranks at step {global_step}. '
                                             f'Rank {rank()} diagnostic: {diagnostic}')
                clip_gradients_fp64(model.parameters(), cfg.grad_clip, grad_norm)
                optimizer.step()
                ema.update(model)
                scheduler.step()
                global_step += 1
                value, n = float(loss.detach()), len(batch['metadata'])
                loss_sum += value * n
                examples += n
                if rank() == 0:
                    bar.set_postfix(loss=f'{value:.4f}', lr=f'{optimizer.param_groups[0]["lr"]:.2g}', refresh=False)
                stop = stop_requested(output, device)
                if stop or global_step % cfg.save_every_steps == 0:
                    save_state(current_epoch, batch_index + 1, loss_sum, examples)
                if stop:
                    if rank() == 0:
                        print('\nSaved last.pt at a completed optimizer step. Remove STOP (if present) and rerun train.sh to continue.', flush=True)
                    return
            del bar
            optimizer.zero_grad(set_to_none=True)
            train_loss_sum, train_examples = sum_tensor([loss_sum, examples], device)
            training_seconds = time.perf_counter() - started
            peaks = gather_objects(torch.cuda.max_memory_allocated(device) / 2**30)
            metrics_live = evaluate(model, dev_data, cfg, device)
            with ema.average_parameters(model):
                metrics_ema = evaluate(model, dev_data, cfg, device)
            use_ema = metrics_ema['micro']['f1'] > metrics_live['micro']['f1']
            metrics = dict(metrics_ema if use_ema else metrics_live)
            metrics['weight_source'] = 'ema' if use_ema else 'live'
            entry = {'epoch': current_epoch + 1, 'global_step': global_step,
                     'train_loss': train_loss_sum / max(1, train_examples),
                     'training_seconds_this_session': training_seconds,
                     'peak_allocated_gib_by_rank': peaks, 'dev': metrics,
                     'dev_live': metrics_live, 'dev_ema': metrics_ema}
            history.append(entry)
            if metrics['micro']['f1'] > best_f1:
                best_f1, best_epoch = metrics['micro']['f1'], current_epoch + 1
                if rank() == 0:
                    best_checkpoint = {
                        'schema': cfg.schema, 'config': cfg.to_dict(),
                        'model': {key: tensor.detach().cpu().clone() for key, tensor in
                                  (ema.model_state(model) if use_ema else model.state_dict()).items()},
                        'encoder_config': model.encoder.config.to_dict(), 'types': types,
                        'vocab_size': len(lexicon['words']), 'lexicon_sha256': lexicon_sha,
                        'char_vocab_size': len(lexicon['characters']), 'weight_transfer': transfer,
                        'weight_source': metrics['weight_source'],
                        'source_hashes': source, 'input_identity': identity,
                        'best_epoch': best_epoch, 'best_f1': best_f1, 'dev': metrics,
                    }
                    atomic_save(best_checkpoint, output / 'best.pt')
            if rank() == 0:
                write_json(output / 'history.json', history)
                write_json(output / 'training_results.json', {
                    'best_epoch': best_epoch, 'best_dev_micro_f1': best_f1,
                    'completed_epochs': current_epoch + 1, 'global_step': global_step,
                    'per_gpu_batch_size': cfg.batch_size, 'effective_batch_size': cfg.batch_size * world_size(),
                    'seed': cfg.seed, 'test_evaluated': False,
                })
                print(f'\nEpoch {current_epoch + 1}: dev Micro={metrics["micro"]["f1"]:.6f}, '
                      f'Macro={metrics["macro_f1"]:.6f}; best={best_f1:.6f} at {best_epoch}; '
                      f'peak GiB={peaks}', flush=True)
            save_state(current_epoch + 1, 0, 0.0, 0,
                       current_epoch + 1 if (current_epoch + 1) % cfg.save_every_epochs == 0 else None)
            next_batch, partial_loss, partial_examples = 0, 0.0, 0
        if rank() == 0:
            print(f'Training finished. Best raw-dev Micro F1={best_f1:.6f}, epoch={best_epoch}. Next: dev calibration.', flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
