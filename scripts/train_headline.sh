#!/usr/bin/env bash
# Train the paper's headline engine run from scratch: n=113, seed 0, 40,000 steps, float64,
# on the from-scratch autograd engine. One process, ~5 CPU-hours.
#
#   scripts/train_headline.sh                 # writes results/from_scratch/WE_engine_n113_s0.npz
#
# It only calls run_n7.py, the script that produced the deposited run, with that run's
# arguments. Training is deterministic and full-batch, so when the deposit is present the
# result is compared against it and must match to the bit (W_E, W_U, logits on the full grid).
# Interrupted? Re-run the same command: run_n7.py resumes from its last checkpoint.
# Every other sweep and seed: `scripts/sweep_table.py --commands`.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-results/from_scratch}

N7_OUT="$OUT" PYTHONPATH=. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -u run_n7.py 113 0 40000

REF=results/n7_engine/WE_engine_n113_s0.npz
if [ ! -f "$REF" ]; then
  echo "No deposit at $REF (scripts/fetch_artifacts.sh); nothing to compare against."
  exit 0
fi
.venv/bin/python - "$OUT/WE_engine_n113_s0.npz" "$REF" <<'PY'
import sys, numpy as np
new, ref = (np.load(p) for p in sys.argv[1:])
bad = [k for k in ("step", "W_E", "W_U", "logits_all") if not np.array_equal(new[k], ref[k])]
if bad:
    sys.exit(f"MISMATCH against the deposit: {', '.join(bad)}")
print(f"OK: bit-identical to the deposited run ({sys.argv[2]}): step, W_E, W_U, logits_all")
PY
