#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec "${LEMAN_GP_PYTHON:-/root/miniconda3/bin/python}" -u run_pipeline.py "$@"
