#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python - <<'PY'
import sys
if not (3, 10) <= sys.version_info[:2] <= (3, 12):
    raise SystemExit('Use Python 3.10, 3.11 or 3.12 for this package.')
PY
if ! python - <<'PY'
import torch
version = tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2])
assert version >= (2, 7) and torch.version.cuda is not None
PY
then
    python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126
fi
python -m pip install -r requirements.txt
