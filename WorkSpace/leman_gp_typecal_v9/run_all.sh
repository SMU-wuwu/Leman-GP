#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PATH="/root/miniconda3/bin:$PATH"
export LEMAN_GP_ROOT="${LEMAN_GP_ROOT:-/root/data/WorkSpace}"
export LEMAN_CALIBRATION_OUTPUT="${LEMAN_CALIBRATION_OUTPUT:-$LEMAN_GP_ROOT/output/leman_typecal_v9_seed42}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
mkdir -p "$LEMAN_CALIBRATION_OUTPUT"
python calibrate.py 2>&1 | tee "$LEMAN_CALIBRATION_OUTPUT/calibrate.log"
