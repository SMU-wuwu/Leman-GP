"""Two-GPU inference entry; historical four-GPU runtime stays unchanged."""
from datetime import timedelta
import os

import torch
import torch.distributed as dist


def setup():
    if int(os.environ.get('WORLD_SIZE', '1')) != 2 or torch.cuda.device_count() != 2:
        raise RuntimeError('Launch with exactly two visible CUDA GPUs and two processes.')
    local = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    dist.init_process_group('nccl', timeout=timedelta(hours=2))
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '2')))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
    torch.backends.cudnn.benchmark = False
    return torch.device('cuda', local)
