#!/usr/bin/env bash
# N7 -- 12 from-scratch-engine runs (4 moduli x 3 seeds), PAR at a time. ZERO Kaggle quota.
# Pre-registered in experiments/PREREGISTER_n7_engine.md.
#
#   scripts/run_n7_all.sh                     # the real 40k run
#   N7_STEPS=400 N7_OUT=/tmp/rehearsal scripts/run_n7_all.sh    # full-pipeline rehearsal
#
# Safe to re-invoke: run_n7.py exits immediately on a run that is already complete, and
# resumes one that was interrupted. Neither case restarts finished work.
set -euo pipefail
cd "$(dirname "$0")/.."

STEPS=${N7_STEPS:-40000}
PAR=${N7_PAR:-4}
OUT=${N7_OUT:-results/n7_engine}
# N9 (the float32 precision A/B) is the same 12-run machinery on a different grid, so it
# reuses this script rather than forking it. LOG_PREFIX must move with the grid: writing a
# float32 run into logs/n7_n113_s0.log would append a second experiment to N7's record.
GRID=${N7_GRID:-$(for s in 0 1 2; do for n in 113 121 125 119; do echo "$n $s"; done; done)}
LOG_PREFIX=${N7_LOG_PREFIX:-n7}
mkdir -p "$OUT" logs

# Measured: more BLAS threads is SLOWER here (458 ms at 12 vs 426 solo) -- the matmuls
# are too small to amortise the sync -- so parallel PROCESSES are the win, not threads.
#
# MEMORY CEASED TO BE THE BINDING CONSTRAINT on 2026-09-12. Every footprint figure in
# this file's history -- "2.2 GB per run", "2.7 GB at n=125", the 13.2 GB 6-way thrash,
# the 17 GB that OOM-killed a 3-way run -- was the no_grad() eval leak, not the cost of
# training. Each `with no_grad(): m(xte)` retained its whole forward pass (~176 MB at
# n=113, once per 100 steps), because operators assigned a self-capturing `_backward`
# closure onto tensors that _child() had already decided needed no graph. Fixed at the
# one point all of them route through (src/autograd/engine.py, the _backward property);
# guarded by test_nograd_leak.py.
#
# MEASURED AFTER THE FIX: ~300 MB per run, FLAT -- Tensor and cell counts are constant
# across 1,400 steps where they previously grew by 252 and 693. That is ~20x smaller, so
# the constant below is 800 MB/run: still a 2.7x margin over measurement, per the lesson
# that a guard is only as good as its constant. CPU, not memory, now sets PAR; 4 leaves
# a third of the box's 12 threads for the desktop.
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=. N7_OUT="$OUT" STEPS

# An OOM at hour 12 loses the night. Post-fix measurement is ~300 MB/run FLAT; the guard
# uses 800 MB/run, a 2.7x margin. Checking RSS alone is what hid the original problem:
# under memory pressure RSS plateaus while the footprint keeps growing into swap, so
# measure RSS + VmSwap, and over 30+ minutes -- a 3-minute window showed a convincing
# false plateau ~50 minutes before the OOM that ended the 2026-09-11 run.
FREE_MB=$(free -m | awk '/^Mem:/{print $7}')
NEED_MB=$(( PAR * 800 + 1500 ))
if [ "$FREE_MB" -lt "$NEED_MB" ]; then
  echo "REFUSING TO START: ${FREE_MB} MB available, need ~${NEED_MB} MB for ${PAR} runs" >&2
  echo "(800 MB/run, a 2.7x margin over the measured 300 MB, plus 1500 MB headroom)." >&2
  echo "Free memory or lower N7_PAR." >&2
  exit 1
fi
NRUNS=$(echo "$GRID" | grep -c .)
echo "$(date -Is)  ${LOG_PREFIX}: ${NRUNS} runs, ${PAR}-way, ${STEPS} steps, dtype=${ENGINE_DTYPE:-float64}, out=${OUT}, ${FREE_MB} MB free"

run_one() {
  local n=$1 s=$2
  .venv/bin/python -u run_n7.py "$n" "$s" "$STEPS" >> "logs/${LOG_PREFIX}_n${n}_s${s}.log" 2>&1 \
    || echo "$(date -Is)  FAILED: n=$n s=$s -- see logs/${LOG_PREFIX}_n${n}_s${s}.log" >&2
}
export -f run_one
export LOG_PREFIX

# The default GRID is seed-major: all four moduli at seed 0 before any seed 1. A cut at
# any hour then leaves four moduli at low seed count rather than one modulus fully done --
# modulus coverage is what E2 and E4 need, and seeds 3-4 are an independent later wave.
echo "$GRID" | xargs -P "$PAR" -n2 bash -c 'run_one "$0" "$1"'

echo "$(date -Is)  all runs returned. Analysis:"
echo "  PYTHONPATH=. .venv/bin/python analyze_n7.py $OUT"
echo "  PYTHONPATH=. .venv/bin/python analyze_n4.py $OUT"
echo "  PYTHONPATH=. .venv/bin/python scripts/render_all.py"
