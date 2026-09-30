#!/usr/bin/env bash
# Reproduce every claim that does not need GPU quota, from a bare clone.
#
# Reproducibility decree (2026-09-11): every code path and experiment must be
# reproducible. This is the single entry point. It runs only things that regenerate
# from artifacts already in the repo or from cheap local compute -- the GPU runs
# themselves are relaunched with kernels/run_kernel.py and are listed at the end.
set -euo pipefail
cd "$(dirname "$0")/.."
P=".venv/bin/python"
export PYTHONPATH=.

hdr() { printf '\n\033[1m== %s ==\033[0m\n' "$1"; }

hdr "0. environment"
$P -V
$P -c "import numpy,torch;print('numpy',numpy.__version__,'| torch',torch.__version__)"
git log --oneline -1
[ -z "$(git status --porcelain)" ] && echo "tree clean" || echo "WARNING: tree is dirty"

hdr "1. self-checks (every module that carries a claim)"
$P src/tasks/algebra.py          # C1, C2
$P src/analysis/runs.py          # C27 retraction -- window endpoint, three states, column name
$P scripts/modulus_table.py --selfcheck   # C1, C2 -- the paper's Table 1, regenerated
$P scripts/sweep_table.py --selfcheck     # every experiment: exact command, stamped argv, no training-path drift since its SHA
$P run_c12_collapse.py            # C12, C13 -- Softmax Collapse, ~2 min
$P test_paper_numbers.py         # the paper's numbers, re-derived from the artifacts
$P scripts/reconstruct_provenance.py  # the code behind the unstamped k02/k03 artifacts
$P test_analysis.py              # transforms + sparsity
$P src/analysis/jblocks.py       # the size-controlled block statistic (C20 re-test)
$P test_autograd.py              # C10 -- 23 finite-difference op checks
$P test_model.py                 # C10 -- vs PyTorch to ~1e-15
$P test_ablation.py              # the ablation harness, vs a synthetic Clock
$P test_memory.py                # C9b
$P test_mechinterp.py            # the 2D DFT bin convention
$P test_run_kernel.py            # kernel ids, placeholders, wrong-account guard
$P test_generator_equivariance.py  # the readout must not depend on which g we picked
$P test_nograd_leak.py           # the reference-cycle leak that cost three misdiagnoses
$P test_provenance.py            # every saved artifact carries a real SHA
$P test_crt_law.py               # C8 -- the CRT-dual law, the best-supported claim here
$P test_grok.py                  # the grok detector itself
$P analyze_gate2.py --selfcheck  # G2 -- planted local clock, white noise, the fibre map
$P analyze_excursions.py --selfcheck  # C27 retraction -- censoring, recovery, gate
$P test_crt_null.py --selfcheck       # C28-NULL -- bin convention, null B, power
$P run_o18_transplant.py --selfcheck  # O18-T -- kernel patches, layout map is a permutation
$P analyze_omega.py --selfcheck       # omega(n) -- deciding cases, rotation trap
$P analyze_k09.py --selfcheck        # k09 -- non-cyclic refusal, disjoint P1 bands
$P analyze_o5.py --selfcheck          # O5 -- the 1-run-arm / size-threshold identifiability test
$P run_o18_cpu.py --selfcheck    # O18 -- the k03 patch list still matches its source
$P run_o21_probe.py --selfcheck   # O21 -- planted promotion detected, clean op not flagged
$P run_o21_numerics.py --selfcheck  # O21 -- the accuracy ranking is itself checked
$P run_intervention.py --selfcheck  # I1 -- planted bin convention, closure both ways, keep_dc
$P test_reproduce.py             # THIS FILE covers every claim-carrying script

hdr "1b. O21 -- the engine's dtype invariance (~2 min, no training)"
# Criterion 3 first: a probe reporting "0 float64 sites" is indistinguishable from a probe
# that cannot read dtypes at all, so the float64 run is the positive control and must come
# before the float32 one is believed.
$P run_o21_probe.py float64      # criterion 3: must read float64 at >=90% of sites
$P run_o21_probe.py float32      # criteria 1-2: candidate (c), NEP-50 promotion
$P run_o21_numerics.py           # criteria A1-A3: candidates (a) and (b)

hdr "2. cross-checks against the authors' released code"
# Needs reference/ -- run scripts/restore_external.sh first. Skips cleanly if absent.
$P refcheck.py                   # C5, C8 on their n=165 checkpoint; O1 convention; C13

hdr "2b. the paper's mathematics in Lean 4 (lean/README.md)"
# Needs elan; mathlib comes from its prebuilt cache (cd lean && lake exe cache get).
if command -v lake >/dev/null || [ -x "$HOME/.elan/bin/lake" ]; then
  scripts/lean_gate.sh --selfcheck   # a planted sorry and native_decide must be rejected
  scripts/lean_gate.sh               # every row of the appendix table
else
  echo "  elan not installed -- Lean gate skipped"
fi

hdr "3. analyses that regenerate from saved artifacts"
$P analyze_scout.py              # two-basis sparsity across moduli
$P analyze_k02.py                # the 5-seed grid, pre-registered criteria
$P analyze_n4.py                 # N4 -- restricted/excluded ablation (C6 causal)
$P jclass_spectra.py             # C7
[ -d results/k03_grid_acts ] && $P analyze_k03.py || echo "  k03 not pulled yet -- skipped"
[ -d results/k04_extended  ] && $P analyze_k04.py || echo "  k04 not pulled yet -- skipped"
[ -d results/k05_lowdata   ] && $P analyze_k05.py || echo "  k05 not pulled yet -- skipped"
[ -d results/k06_horizon   ] && $P analyze_k06.py || echo "  k06 not pulled yet -- skipped"
[ -d results/k07_arma_seeds ] && $P analyze_k07.py || echo "  k07 not pulled yet -- skipped"
[ -d results/k08_init      ] && $P analyze_k08.py || echo "  k08 not pulled yet -- skipped"
[ -d results/k09_primes_zdd ] && $P analyze_k09.py || echo "  k09 not pulled yet -- skipped"
[ -d results/k08_init      ] && $P analyze_n4.py results/k08_init || echo "  k08 -- skipped"  # C32
[ -d results/n7_engine     ] && $P analyze_n7.py  || echo "  N7 not run yet -- skipped"        # C29
[ -d results/n7_engine     ] && $P analyze_n4.py results/n7_engine || echo "  N7 -- skipped"   # E3
[ -d results/n9_f32        ] && $P analyze_n9.py  || echo "  N9 not run yet -- skipped"        # C30 retraction
[ -d results/k03_grid_acts ] && $P analyze_o4.py  || echo "  k03 -- skipped"                   # C31
[ -d results/k03_grid_acts ] && $P analyze_gate2.py || echo "  k03 -- skipped"                 # G2
[ -d results/k03_grid_acts ] && $P analyze_r2_sameness.py || echo "  k03 -- skipped"            # R2
[ -d results/k04_extended  ] && $P analyze_r2_sameness.py results/k04_extended --controls \
                             || echo "  R2 controls -- skipped"
[ -d results/k06_horizon   ] && $P analyze_excursions.py results/k06_horizon || echo "  k06 -- skipped"  # C27 RETRACTED
[ -d results/n7_engine     ] && $P test_crt_null.py results/n7_engine || echo "  N7 -- skipped"  # C28-NULL
[ -d results/o20_n63       ] && CRT_NULL_N=63 $P test_crt_null.py results/o20_n63 || echo "  O20 -- skipped"  # the failed-ramp arm
[ -d results/k03_grid_acts ] && $P analyze_omega.py || echo "  k0* -- skipped"   # O7 -> omega(n)
[ -d results/k03_grid_acts ] && $P analyze_omega.py --confirm || echo "  k0* -- skipped"   # C37, W1-W6
[ -d results/k05_lowdata   ] && $P analyze_excursions.py results/k05_lowdata || echo "  k05 -- skipped"  # O9
[ -d results/k01_scout     ] && $P analyze_o5.py || echo "  k01 -- skipped"      # O5 NOT ANSWERABLE

hdr "4. figures (regenerated from artifacts, never hand-made)"
$P scripts/render_all.py
echo "  -> figures/INDEX.md"

hdr "done"
cat <<'TXT'
GPU runs are not re-executed here. To relaunch one:
    .venv/bin/python kernels/run_kernel.py kernels/<dir>            # push + poll + pull
    .venv/bin/python kernels/run_kernel.py kernels/<dir> --attach   # attach, no re-push
Each pushed kernel has the exact git SHA baked into it by run_kernel.py::git_sha, and
stamps it into every .npz it writes. Recover it with src/provenance.py::read.
TXT
