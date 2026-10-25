#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export LEMAN_GP_ROOT="${LEMAN_GP_ROOT:-/root/data/WorkSpace}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/root/miniconda3/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 predict.py "$@"
