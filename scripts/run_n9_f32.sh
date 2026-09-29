#!/usr/bin/env bash
# N9 -- the precision A/B (O17 / C30). The SAME engine, the SAME script, the SAME seeds
# and the SAME data split as N7's n=113 runs, with ONE variable changed: float32.
# Pre-registered in experiments/PREREGISTER_n9_precision.md, committed before launch.
#
#   scripts/run_n9_f32.sh                                  # the real 40k run, ~3-4 h
#   N7_STEPS=400 N7_OUT=/tmp/n9_rehearsal scripts/run_n9_f32.sh   # pipeline rehearsal
#
# It is a wrapper, not a fork: run_n7_all.sh takes the grid, the log prefix and the output
# directory as parameters, so the only thing that differs between the two arms of the A/B
# is ENGINE_DTYPE. Forking the launcher is how the two arms would quietly drift apart.
set -euo pipefail
cd "$(dirname "$0")/.."

export ENGINE_DTYPE=float32
export N7_OUT=${N7_OUT:-results/n9_f32}
export N7_LOG_PREFIX=n9_f32
export N7_PAR=${N7_PAR:-3}                      # 3 seeds, 2 BLAS threads each
export N7_GRID=${N7_GRID:-$'113 0\n113 1\n113 2'}

exec bash scripts/run_n7_all.sh
