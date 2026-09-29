#!/usr/bin/env bash
# Wait for a RUNNING Kaggle kernel, pull its output, analyse it, render the figures.
# Never pushes -- run_kernel.py --attach only polls. Detached use:
#   setsid scripts/after_kernel.sh kernels/k03_grid_acts analyze_k03.py >> log 2>&1 < /dev/null &
set -euo pipefail
cd "$(dirname "$0")/.."
KDIR=$1; ANALYSIS=$2
echo "=== $(date -Is) attach $KDIR ==="
.venv/bin/python -u kernels/run_kernel.py "$KDIR" --attach
echo "=== $(date -Is) $ANALYSIS ==="
PYTHONPATH=. .venv/bin/python -u "$ANALYSIS"
echo "=== $(date -Is) render_all ==="
PYTHONPATH=. .venv/bin/python -u scripts/render_all.py
echo "=== $(date -Is) done ==="
