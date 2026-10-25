#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
TASK_PYTHON=${TASK_PYTHON:-/root/miniconda3/bin/python}
"$TASK_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 extract.py
"$TASK_PYTHON" analyze.py
