"""One controlled phase. The pipeline launcher runs A/B/C in separate processes."""
import argparse
from dataclasses import replace
import fcntl
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

from adversarial import fgm_backward
from config import Config, VARIANTS, PHASES
from controlled_data import Collator
from controlled_evaluate import evaluate
from data import build_lexicon, entity_types, load_split, prepare_split, sha256, write_json
from model import LemanGP, initialization_record
from recipe import training_recipe
from transfer import EMA, optimizer_groups
from runtime import (NUMERICAL_POLICY, ResumeBatchSampler, all_ranks_finite, amp_context,
                     atomic_save, barrier, broadcast_object, clip_gradients_fp64,
                     environment, gather_objects, gradient_norm_fp64, input_identity,
                     optimizer_to, rank, restore_rng, rng_state, seed_everything,
                     setup, source_hashes, sum_tensor, to_device, world_size,
                     write_numerical_failure)

STOP_REQUESTED = False


def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def stopping(output, device):
    flag = STOP_REQUESTED or (output/'STOP').exists() or (output.parent/'STOP').exists()
    value = torch.tensor(int(flag), device=device)
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return bool(value.item())


def resume_path(output):
    if (output/'last.pt').exists():
        return output/'last.pt'
    snapshots = sorted(output.glob('epoch_*.pt'))
    if snapshots:
        return snapshots[-1]
    if (output/'best.pt').exists():
        raise RuntimeError('Only best.pt remains: restore last.pt or an epoch snapshot for resume.')
    return None


def prepare_resources(cfg, identity, types):
    resource = Path(cfg.resource_dir)
    resource.mkdir(parents=True, exist_ok=True)
    with (resource/'.prepare.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        record = resource/'INPUT_IDENTITY.json'
        if record.exists() and json.loads(record.read_text()) != identity:
            raise ValueError('The controlled matrix resource directory belongs to different inputs')
        write_json(record, identity)
        lex_path = resource/'lexicon.json'
        if not lex_path.exists():
            write_json(lex_path, build_lexicon(cfg, types))
        lexicon = json.loads(lex_path.read_text())
        if lexicon['train_sha256'] != identity['train2.txt'] or lexicon['schema'] != cfg.schema:
            raise ValueError('Train-only lexicon identity mismatch')
        data_cfg = replace(cfg, output_dir=str(resource))
        paths = [str(prepare_split(data_cfg, s, lexicon, types)) for s in ('train', 'dev')]
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/root/data/WorkSpace')
    parser.add_argument('--variant', choices=VARIANTS, default='F')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--phase', choices=PHASES, default='A')
    parser.add_argument('--qa-config', type=Path, help='Synthetic fixture configuration only')
    args = parser.parse_args()
    cfg = Config.make(args.root, args.variant, args.seed, args.phase)
    if args.qa_config:
        overrides = json.loads(args.qa_config.read_text())
        if not (Path(cfg.root)/'SYNTHETIC_QA_ONLY').exists():
            raise ValueError('QA overrides require a separate synthetic fixture root')
        for k, v in overrides.items():
            setattr(cfg, k, v)
        cfg.qa = True
    device = setup()
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    loader = None
    try:
        source = source_hashes(Path(__file__).parent)
        identity = broadcast_object(input_identity(cfg) if rank() == 0 else None)
        types = entity_types(Path(cfg.data_dir)/'label.txt')
        paths = broadcast_object(prepare_resources(cfg, identity, types) if rank() == 0 else None)
        train_data, dev_data = map(load_split, paths)
        lexicon_path = Path(cfg.resource_dir)/'lexicon.json'
        lexicon = json.loads(lexicon_path.read_text())
        lexicon_sha = sha256(lexicon_path)
        recipe = training_recipe(train_data.samples, types, cfg, identity['train2.txt'])
        cfg.gp_type_weights = recipe['ordered_weights']
        if cfg.variant == 'crf':
            recipe['active_loss'] = 'BIOES CRF NLL + boundary MLCCE; GP weights not applied'
        else:
            recipe['active_loss'] = recipe['loss']
        recipe['boundary_weight'] = cfg.boundary_weight
        recipe['boundary_loss'] = 'MLCCE over start/end positions' if cfg.boundary_weight else 'disabled'
        path = broadcast_object(str(resume_path(output) or '') if rank() == 0 else None)
        checkpoint = torch.load(path, map_location='cpu', weights_only=False, mmap=True) if path else None
        parent = None
        origin = {'kind': 'public_pretrained_encoder', 'encoder_weights': identity['encoder_weights']}
        if checkpoint is not None:
            for name, expected in [('source_hashes', source), ('input_identity', identity),
                                   ('lexicon_sha256', lexicon_sha), ('world_size', world_size()),
                                   ('config', cfg.to_dict())]:
                if checkpoint[name] != expected:
                    raise ValueError(f'Resume mismatch: {name}. Use the saved package and settings.')
            origin = checkpoint['origin']
        elif cfg.phase != 'A':
            previous = Path(cfg.warm_start)
            complete = json.loads((previous.parent/'COMPLETE.json').read_text())
            parent_sha = broadcast_object(sha256(previous) if rank() == 0 else None)
            if parent_sha != complete['best_sha256']:
                raise ValueError('Previous phase best.pt differs from its completed receipt')
            parent = torch.load(previous, map_location='cpu', weights_only=False, mmap=True)
            if (parent['config']['variant'] != cfg.variant or parent['config']['seed'] != cfg.seed
                    or parent['config']['phase'] != chr(ord(cfg.phase)-1)
                    or parent['input_identity'] != identity or parent['source_hashes'] != source
                    or parent['lexicon_sha256'] != lexicon_sha):
                raise ValueError('Stage transfer must use the same variant, seed, data and code')
            origin = {'kind': 'previous_phase_best', 'path': str(previous), 'sha256': parent_sha,
                      'best_epoch': parent['best_epoch'], 'weight_source': parent['weight_source']}
        seed_everything(cfg.seed)
        saved = checkpoint if checkpoint is not None else parent
        model = LemanGP(cfg, len(types), len(lexicon['words']), char_vocab_size=len(lexicon['characters']),
                        encoder_config=None if saved is None else saved['encoder_config'], pretrained=saved is None)
        if saved is not None:
            model.load_state_dict(saved['model'], strict=True)
        if rank() == 0 and checkpoint is None:
            write_json(output/'initialization.json', {'origin': origin, 'parameters': initialization_record(model)})
        model.to(device)
        wrapped = DDP(model, device_ids=[device.index], broadcast_buffers=False, gradient_as_bucket_view=True)
        optimizer = torch.optim.AdamW(optimizer_groups(model, cfg, set()), fused=True)
        ema = EMA(model, cfg.ema_decay)
        sampler = DistributedSampler(train_data, num_replicas=world_size(), rank=rank(),
                                     shuffle=True, seed=cfg.seed, drop_last=False)
        steps_per_epoch = math.ceil(len(sampler)/cfg.batch_size)
        total_steps = steps_per_epoch*cfg.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps*cfg.warmup_fraction), total_steps)
        epoch = next_batch = global_step = best_epoch = 0
        best_f1 = -1.0
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
                raise ValueError('Training batch count changed')
            part = checkpoint['partial_by_rank'][rank()]
            partial_loss, partial_examples = part['loss_sum'], part['examples']
            if rank() == 0:
                best_checkpoint = checkpoint['best_checkpoint']
                if best_checkpoint is not None:
                    atomic_save(best_checkpoint, output/'best.pt')
            restore_rng(checkpoint['rng_by_rank'][rank()])
        else:
            seed_everything(cfg.seed + rank())
        del checkpoint, parent, saved
        gc.collect()
        if rank() == 0:
            shutil.copy2(lexicon_path, output/'lexicon.json')
            snapshot = output/'source_snapshot'
            snapshot.mkdir(exist_ok=True)
            for p in Path(__file__).parent.iterdir():
                if p.is_file() and p.suffix in ('.py', '.sh', '.json', '.md', '.txt'):
                    shutil.copy2(p, snapshot/p.name)
            write_json(output/'training_recipe.json', recipe)
            write_json(output/'run_config.json', {
                'config': cfg.to_dict(), 'source_hashes': source, 'input_identity': identity,
                'lexicon_sha256': lexicon_sha, 'environment': environment(device), 'origin': origin,
                'effective_batch_size': cfg.batch_size*world_size(), 'steps_per_epoch': steps_per_epoch,
                'total_steps': total_steps, 'warmup_steps': int(total_steps*cfg.warmup_fraction),
                'train_stats': train_data.stats, 'dev_stats': dev_data.stats,
                'selection': 'raw dev Micro, best across epochs/live/EMA; strict improvement; live wins ties; B/C include epoch0',
                'EMA': 'decay=min(0.995,(1+updates)/(10+updates)); reset at phase boundary',
                'train_sampler': 'DistributedSampler shuffle with seed; tail padded equally across ranks',
                'test_read_by_train': False, 'resume_from': path or None,
                'numerical_policy': NUMERICAL_POLICY})
            print(f'{cfg.variant} seed{cfg.seed} phase{cfg.phase}: {world_size()} GPUs, {cfg.batch_size}/GPU; '
                  f'epochs={cfg.epochs}, steps/epoch={steps_per_epoch}; resume epoch={epoch+1} batch={next_batch}', flush=True)
        barrier()

        def make_best(metrics, selected, selected_epoch):
            return dict(schema=cfg.schema, config=cfg.to_dict(), source_hashes=source, input_identity=identity,
                        lexicon_sha256=lexicon_sha, types=types, vocab_size=len(lexicon['words']),
                        char_vocab_size=len(lexicon['characters']), encoder_config=model.encoder.config.to_dict(),
                        origin=origin, weight_source=selected, best_epoch=selected_epoch, best_f1=metrics['micro']['f1'],
                        dev=metrics, model={k: v.detach().cpu().clone() for k, v in
                            (ema.model_state(model) if selected == 'ema' else model.state_dict()).items()})

        def save_state(next_epoch, next_index, loss_sum, examples, snapshot_epoch=None):
            states = gather_objects(rng_state())
            partials = gather_objects({'loss_sum': loss_sum, 'examples': examples})
            if rank() == 0:
                payload = dict(schema=cfg.schema, config=cfg.to_dict(), source_hashes=source, input_identity=identity,
                    lexicon_sha256=lexicon_sha, world_size=world_size(), encoder_config=model.encoder.config.to_dict(),
                    types=types, vocab_size=len(lexicon['words']), char_vocab_size=len(lexicon['characters']),
                    origin=origin, model=model.state_dict(), ema=ema.state_dict(), optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(), epoch=next_epoch, next_batch=next_index, global_step=global_step,
                    steps_per_epoch=steps_per_epoch, best_f1=best_f1, best_epoch=best_epoch, history=history,
                    best_checkpoint=best_checkpoint, rng_by_rank=states, partial_by_rank=partials)
                atomic_save(payload, output/'last.pt')
                if snapshot_epoch is not None:
                    target = output/f'epoch_{snapshot_epoch:03d}.pt'
                    tmp = target.with_suffix('.tmp')
                    tmp.unlink(missing_ok=True)
                    try:
                        os.link(output/'last.pt', tmp)
                    except OSError:
                        shutil.copy2(output/'last.pt', tmp)
                    os.replace(tmp, target)
            barrier()

        batch_sampler = ResumeBatchSampler(sampler, cfg.batch_size, 0)
        kwargs = dict(batch_sampler=batch_sampler, num_workers=cfg.workers, pin_memory=True,
                      collate_fn=Collator(types, train_data.pad_id, cfg.variant),
                      generator=torch.Generator().manual_seed(cfg.seed+10000+rank()))
        if cfg.workers:
            kwargs.update(persistent_workers=True, prefetch_factor=2, multiprocessing_context='spawn')
        loader = DataLoader(train_data, **kwargs)
        if not path:
            if cfg.phase != 'A':
                initial = evaluate(model, dev_data, cfg, device)
                best_f1 = initial['micro']['f1']
                if rank() == 0:
                    best_checkpoint = make_best(initial, 'phase_entry', 0)
                    atomic_save(best_checkpoint, output/'best.pt')
                    write_json(output/'entry_baseline.json', initial)
            save_state(0, 0, 0.0, 0)
        for current_epoch in range(epoch, cfg.epochs):
            sampler.set_epoch(current_epoch)
            start_batch = next_batch if current_epoch == epoch else 0
            batch_sampler.start_batch = start_batch
            wrapped.train()
            torch.cuda.reset_peak_memory_stats(device)
            loss_sum = partial_loss if current_epoch == epoch else 0.0
            examples = partial_examples if current_epoch == epoch else 0
            started = time.perf_counter()
            bar = tqdm(loader, total=steps_per_epoch, initial=start_batch, desc=f'{cfg.phase} Epoch {current_epoch+1}/{cfg.epochs}',
                       disable=rank() != 0, mininterval=1, dynamic_ncols=True)
            fgm_stats = None
            for index, batch in enumerate(bar, start_batch):
                optimizer.zero_grad(set_to_none=True)
                def check_finite(value, stage):
                    if not all_ranks_finite(value):
                        diagnostic = write_numerical_failure(output, model, stage=stage, step=global_step, loss=value, batch=batch)
                        raise FloatingPointError(f'Nonfinite {stage}: {diagnostic}')
                inputs = to_device(batch, device)
                if cfg.phase == 'C':
                    loss, fgm_stats = fgm_backward(wrapped, inputs, epsilon=cfg.fgm_epsilon,
                        clean_weight=cfg.fgm_clean_weight, autocast=lambda: amp_context(device), check_finite=check_finite)
                else:
                    with amp_context(device):
                        loss, _ = wrapped(**inputs)
                    check_finite(loss, 'loss')
                    loss.backward()
                norm = gradient_norm_fp64(model.parameters())
                check_finite(norm, 'gradient_norm')
                clip_gradients_fp64(model.parameters(), cfg.grad_clip, norm)
                optimizer.step()
                ema.update(model)
                scheduler.step()
                global_step += 1
                value, n = float(loss.detach()), len(batch['metadata'])
                loss_sum += value*n
                examples += n
                if rank() == 0:
                    bar.set_postfix(loss=f'{value:.4f}', lr=f'{optimizer.param_groups[0]["lr"]:.2g}', refresh=False)
                stop = stopping(output, device)
                if cfg.qa and global_step == int(os.environ.get('LEMAN_QA_STOP_AFTER', '-1')):
                    stop = True
                if stop or global_step % cfg.save_every_steps == 0:
                    save_state(current_epoch, index+1, loss_sum, examples)
                if stop:
                    if rank() == 0:
                        print('Interrupted after a completed optimizer step; last.pt saved. Rerun the same command.', flush=True)
                    return 75
            optimizer.zero_grad(set_to_none=True)
            train_loss, train_examples = sum_tensor([loss_sum, examples], device)
            train_seconds = time.perf_counter()-started
            peaks = gather_objects(torch.cuda.max_memory_allocated(device)/2**30)
            live = evaluate(model, dev_data, cfg, device)
            with ema.average_parameters(model):
                averaged = evaluate(model, dev_data, cfg, device)
            use_ema = averaged['micro']['f1'] > live['micro']['f1']
            metrics = averaged if use_ema else live
            selected = 'ema' if use_ema else 'live'
            history.append(dict(epoch=current_epoch+1, global_step=global_step,
                train_loss=train_loss/max(1, train_examples), training_seconds_this_session=train_seconds,
                peak_allocated_gib_by_rank=peaks, dev_live=live, dev_ema=averaged,
                selected=selected, dev=metrics, last_batch_fgm=fgm_stats))
            if metrics['micro']['f1'] > best_f1:
                best_f1, best_epoch = metrics['micro']['f1'], current_epoch+1
                if rank() == 0:
                    best_checkpoint = make_best(metrics, selected, best_epoch)
                    atomic_save(best_checkpoint, output/'best.pt')
            if rank() == 0:
                write_json(output/'history.json', history)
                print(f'\n{cfg.phase} epoch {current_epoch+1}: Micro={metrics["micro"]["f1"]:.6f}; '
                      f'best={best_f1:.6f} epoch={best_epoch}; peak GiB={peaks}', flush=True)
            save_state(current_epoch+1, 0, 0.0, 0,
                       current_epoch+1 if (current_epoch+1) % cfg.save_every_epochs == 0 else None)
            next_batch, partial_loss, partial_examples = 0, 0.0, 0
        # Export predictions of the selected weights without altering saved live/EMA/resume state.
        best = torch.load(output/'best.pt', map_location='cpu', weights_only=False, mmap=True)
        model.load_state_dict(best['model'], strict=True)
        final = evaluate(model, dev_data, cfg, device, output_dir=output/'best_dev')
        if rank() == 0:
            receipt = dict(variant=cfg.variant, seed=cfg.seed, phase=cfg.phase, completed_epochs=cfg.epochs,
                global_step=global_step, best_epoch=best_epoch, best_dev_micro_f1=best_f1,
                exported_dev_micro_f1=final['micro']['f1'], best_sha256=sha256(output/'best.pt'),
                source_hashes=source, input_identity=identity, test_read=False, qa=cfg.qa)
            write_json(output/'COMPLETE.json', receipt)
            write_json(output/'training_results.json', receipt)
            print(f'Phase {cfg.phase} complete; best raw-dev Micro={best_f1:.6f}, epoch={best_epoch}', flush=True)
        barrier()
        return 0
    finally:
        if loader is not None and loader._iterator is not None:
            loader._iterator._shutdown_workers()
        if 'dev_data' in locals():
            from evaluate import release_loader
            release_loader(dev_data)
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    raise SystemExit(main())
