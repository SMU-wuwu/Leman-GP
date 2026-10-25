"""Distributed runtime, memory-based batch selection and recoverable checkpoints."""
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
import gc
import math
import os
import random
import subprocess
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler
import transformers

from data import sha256, write_json


NUMERICAL_POLICY = {
    'revision': 'f1-v2-based-on-a100-step0-fix2',
    'encoder_autocast': 'bf16',
    'encoder_sdpa_backend': 'automatic_pytorch_sdpa',
    'custom_layers': 'fp32_autocast_disabled_before_projections',
    'gradient_norm': 'fp64_before_clipping',
    'fp16_bf16_reduced_precision_reduction': False,
    'nonfinite_handling': 'record_per_rank_diagnostics_and_stop_before_update',
    'score_clamping': False,
    'replace_nonfinite_loss': False,
    'l1_cdist': 'exact_l1_chunked_over_independent_batch_head_matrices',
    'cdist_backward_max_elements': 1 << 27,
}


def distributed():
    return dist.is_initialized()


def rank():
    return dist.get_rank() if distributed() else 0


def world_size():
    return dist.get_world_size() if distributed() else 1


def barrier():
    if distributed():
        dist.barrier()


def setup():
    count = int(os.environ.get('WORLD_SIZE', 1))
    if count != 2 or not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError('Run the supplied shell launcher with two visible CUDA GPUs.')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl', timeout=timedelta(hours=2))
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '2')))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
    torch.backends.cudnn.benchmark = False
    return torch.device('cuda', local_rank)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'].cpu())


def gather_objects(value):
    if not distributed():
        return [value]
    result = [None] * world_size()
    dist.all_gather_object(result, value)
    return result


def broadcast_object(value):
    values = [value]
    if distributed():
        dist.broadcast_object_list(values, src=0)
    return values[0]


def sum_tensor(values, device):
    values = torch.as_tensor(values, dtype=torch.float64, device=device)
    if distributed():
        dist.all_reduce(values)
    return values.cpu().tolist()


def amp_context(device):
    return torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()


def all_ranks_finite(value):
    """All ranks take the same failure branch, including a forward-only failure."""
    flag = torch.isfinite(value.detach()).all().to(dtype=torch.int32)
    if distributed():
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def gradient_norm_fp64(parameters):
    """Do not overflow the squared sum of otherwise finite FP32 gradients."""
    parameters = list(parameters)
    norms = [torch.linalg.vector_norm(p.grad.detach(), dtype=torch.float64)
             for p in parameters if p.grad is not None]
    if not norms:
        device = parameters[0].device if parameters else 'cpu'
        return torch.zeros((), dtype=torch.float64, device=device)
    return torch.linalg.vector_norm(torch.stack(norms))


@torch.no_grad()
def clip_gradients_fp64(parameters, max_norm, total_norm):
    """Call only after checking the original norm; never zero invalid gradients."""
    if not torch.isfinite(total_norm).item():
        raise FloatingPointError('Cannot clip non-finite gradients; preserve them for diagnostics.')
    coefficient = (max_norm / (total_norm + 1e-6)).clamp(max=1.0)
    groups = {}
    for parameter in parameters:
        if parameter.grad is not None:
            grad = parameter.grad
            groups.setdefault((grad.device, grad.dtype), []).append(grad)
    for (device, dtype), gradients in groups.items():
        torch._foreach_mul_(gradients, coefficient.to(device=device, dtype=dtype))


def _json_number(value):
    value = float(value)
    return value if math.isfinite(value) else str(value)


def write_numerical_failure(output, model, *, stage, step, loss=None, norm=None, batch=None):
    """Record the original gradients, without serializing data/model tensors."""
    bad, finite, missing = [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            missing.append(name)
            continue
        grad = parameter.grad.detach()
        finite_mask = torch.isfinite(grad)
        row = {'name': name, 'shape': list(grad.shape), 'dtype': str(grad.dtype)}
        if not finite_mask.all().item():
            row.update(nan=int(torch.isnan(grad).sum()),
                       positive_inf=int(torch.isposinf(grad).sum()),
                       negative_inf=int(torch.isneginf(grad).sum()))
            bad.append(row)
        else:
            row['norm_fp64'] = _json_number(torch.linalg.vector_norm(grad, dtype=torch.float64))
            row['max_abs'] = _json_number(grad.abs().max()) if grad.numel() else 0.0
            finite.append(row)
    finite.sort(key=lambda row: row['norm_fp64'], reverse=True)
    metadata = {}
    if batch is not None:
        metadata['shapes'] = {k: list(v.shape) for k, v in batch.items() if torch.is_tensor(v)}
        metadata['examples'] = batch.get('metadata', [])
    device = next(model.parameters()).device
    report = {
        'stage': stage, 'global_step': step, 'rank': rank(), 'world_size': world_size(),
        'loss': None if loss is None else _json_number(loss.detach()),
        'gradient_norm_fp64': None if norm is None else _json_number(norm.detach()),
        'nonfinite_gradients': bad, 'largest_finite_gradients': finite[:24],
        'parameters_without_gradients': missing, 'batch': metadata,
        'numerical_policy': NUMERICAL_POLICY, 'torch': torch.__version__,
        'source_hashes': source_hashes(Path(__file__).parent),
        'transformers': transformers.__version__, 'cuda': torch.version.cuda,
        'device': str(device),
        'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'optimizer_step_executed_for_failed_batch': False,
    }
    path = Path(output) / 'diagnostics' / f'numerical_failure_{stage}_step{step}_rank{rank()}_{time.time_ns()}.json'
    write_json(path, report)
    return path


def to_device(batch, device):
    return {k: (v.to(device, non_blocking=True) if k != 'lengths' else v)
            for k, v in batch.items() if k != 'metadata'}


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with open(temp, 'wb') as f:
        torch.save(value, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def optimizer_to(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


class ResumeBatchSampler(Sampler):
    """Skip completed batches before DataLoader work, without consuming model RNG."""
    def __init__(self, sampler, batch_size, start_batch=0):
        self.sampler, self.batch_size, self.start_batch = sampler, batch_size, start_batch

    def __iter__(self):
        indices = list(self.sampler)
        for start in range(self.start_batch * self.batch_size, len(indices), self.batch_size):
            yield indices[start:start + self.batch_size]

    def __len__(self):
        return max(0, (len(self.sampler) + self.batch_size - 1) // self.batch_size - self.start_batch)


def environment(device):
    try:
        drivers = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=driver_version,name,memory.total', '--format=csv,noheader'],
            text=True, timeout=10).strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        drivers = []
    return {'python': sys.version, 'torch': torch.__version__, 'transformers': transformers.__version__,
            'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
            'gpu': torch.cuda.get_device_name(device),
            'memory_gib': torch.cuda.get_device_properties(device).total_memory / 2**30,
            'driver_gpu_listing': drivers, 'world_size': world_size(), 'amp': 'bf16',
            'tf32': True, 'numerical_policy': NUMERICAL_POLICY}


def source_hashes(directory):
    return {p.name: sha256(p) for p in sorted(Path(directory).glob('*.py'))}


def input_identity(cfg):
    base = Path(cfg.data_dir)
    enc = Path(cfg.encoder_dir)
    names = ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'tokenizer.model',
             'special_tokens_map.json', 'added_tokens.json']
    result = {name: sha256(base / name) for name in ('train2.txt', 'dev2.txt', 'label.txt')}
    result['encoder_files'] = {name: sha256(enc / name) for name in names if (enc / name).exists()}
    # Hash local encoder weights for resume integrity; never read a test split here.
    weight_files = sorted(set(enc.glob('*.safetensors')) | set(enc.glob('pytorch_model*.bin')))
    result['encoder_weights'] = {p.name: sha256(p) for p in weight_files}
    if not weight_files:
        raise FileNotFoundError(f'No encoder weights found in {enc}')
    return result


def choose_batch_size(model, dataset, cfg, device):
    """Probe forward/backward memory and finiteness; restore RNG, keep weights.

    Reserve Adam's two moments, a DDP gradient bucket and the EMA shadow.
    Both GPUs must pass. Keep additional headroom for CUDA/NCCL workspaces.
    """
    if cfg.batch_size:
        return cfg.batch_size, {'mode': 'explicit', 'per_gpu_batch_size': cfg.batch_size}
    state, was_training = rng_state(), model.training
    trials, chosen = [], 0
    params = [p for p in model.parameters() if p.requires_grad]
    state_reserve = []
    model.train()
    max_tokens = dataset.stats['max_encoder_tokens_observed']
    max_chars = dataset.stats['max_chars_observed']
    candidates = sorted(set(x for x in [4, 8, 12, 16, 24, 32, 48, 64, cfg.max_auto_batch]
                            if x <= cfg.max_auto_batch))
    try:
        # Gradients are produced by backward; these four copies cover persistent
        # states that are allocated only after probing in the actual training run.
        for p in params:
            state_reserve.extend(torch.empty_like(p) for _ in range(4))
        for batch_size in candidates:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            free, total = torch.cuda.mem_get_info(device)
            external_used = total - free - torch.cuda.memory_reserved(device)
            budget = max(0, total * cfg.memory_fraction - external_used)
            success, error, batch, outputs, diagnostic = True, '', None, None, None
            try:
                ids = dataset.items[0]['input_ids'][0]
                batch = {
                    'input_ids': torch.full((batch_size, max_tokens), ids, device=device, dtype=torch.long),
                    'token_mask': torch.ones(batch_size, max_tokens, device=device, dtype=torch.bool),
                    'char_pool': torch.zeros(batch_size, max_chars, max_tokens, device=device),
                    'char_mask': torch.ones(batch_size, max_chars, device=device, dtype=torch.bool),
                    'char_ids': torch.ones(batch_size, max_chars, device=device, dtype=torch.long),
                    'word_ids': torch.ones(batch_size, max_chars, cfg.max_words, device=device, dtype=torch.long),
                    'boundary_ids': torch.zeros(batch_size, max_chars, cfg.max_words, device=device, dtype=torch.long),
                    'lengths': torch.full((batch_size,), max_chars, dtype=torch.long),
                    'labels': torch.zeros(batch_size, len(dataset.types), max_chars, max_chars,
                                          device=device, dtype=torch.bool),
                }
                char_index = torch.arange(max_chars, device=device)
                batch['char_pool'][:, char_index, char_index.clamp_max(max_tokens - 1)] = 1
                batch['labels'][:, 0, 0, 0] = True
                with amp_context(device):
                    outputs = model(**batch)
                if not torch.isfinite(outputs[0]).item():
                    success, error = False, 'nonfinite_loss'
                    diagnostic = str(write_numerical_failure(
                        cfg.output_dir, model, stage=f'batch_probe_{batch_size}', step=0,
                        loss=outputs[0], batch=batch))
                else:
                    outputs[0].backward()
                    norm = gradient_norm_fp64(params)
                    if not torch.isfinite(norm).item():
                        success, error = False, 'nonfinite_gradient'
                        diagnostic = str(write_numerical_failure(
                            cfg.output_dir, model, stage=f'batch_probe_{batch_size}', step=0,
                            loss=outputs[0], norm=norm, batch=batch))
                torch.cuda.synchronize(device)
                if success and torch.cuda.max_memory_allocated(device) > budget:
                    success, error = False, 'memory_budget'
            except torch.cuda.OutOfMemoryError:
                success = False
                # An OOM while writing a failure report must not disguise a
                # known numerical failure as an acceptable capacity limit.
                if not error.startswith('nonfinite_'):
                    error = 'cuda_oom'
            peak = torch.cuda.max_memory_allocated(device) / 2**30
            model.zero_grad(set_to_none=True)
            batch, outputs = None, None
            gc.collect()
            torch.cuda.empty_cache()
            local = {'rank': rank(), 'batch': batch_size, 'ok': success, 'peak_gib': peak,
                     'budget_gib': budget / 2**30, 'reason': error, 'diagnostic': diagnostic}
            rows = gather_objects(local)
            trials.append(rows)
            if rank() == 0:
                print(f'Batch sizing: {batch_size}/GPU; ' + '; '.join(
                    f"GPU{r['rank']} peak {r['peak_gib']:.1f} GiB, {r['reason'] or 'OK'}" for r in rows), flush=True)
            failures = [r for r in rows if r['reason'].startswith('nonfinite_')]
            if failures:
                raise FloatingPointError(f'Numerical failure during batch selection: {failures}')
            if not all(r['ok'] for r in rows):
                break
            chosen = batch_size
        if not chosen:
            raise RuntimeError('No batch fits the memory budget. Check GPU occupancy or set LEMAN_GP_BATCH_SIZE.')
        return chosen, {'mode': 'automatic', 'per_gpu_batch_size': chosen,
                        'memory_fraction': cfg.memory_fraction, 'trials': trials,
                        'probe_tokens': max_tokens, 'probe_chars': max_chars,
                        'world_size': world_size(),
                        'reserved_parameter_copies': 4,
                        'reserved_states': ['adam_first_moment', 'adam_second_moment',
                                            'ddp_gradient_bucket', 'ema_shadow'],
                        'reserved_state_gib': sum(p.numel() * p.element_size() for p in params) * 4 / 2**30}
    finally:
        state_reserve.clear()
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        model.train(was_training)
        restore_rng(state)
