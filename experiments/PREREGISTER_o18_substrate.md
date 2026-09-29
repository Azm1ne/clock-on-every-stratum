# Pre-registration: O18 — is the engine/torch sparsity gap the EXECUTION SUBSTRATE?

**Commit this file BEFORE launching.** **Date:** 2026-09-14 · **Script:** `run_o18_cpu.py` ·
**Arm:** thesis (diagnostic bisect) · **Compute:** ~40 min local CPU, 3-way. **Zero Kaggle
quota.**

## Hypothesis

The ~0.19 Gini_mult gap between the from-scratch engine and the k03 PyTorch kernel is
produced by the **execution substrate** — Kaggle **T4 GPU** versus local **CPU** — and not
by anything in either implementation's code.

## Why this is the test that is left

C30 said the gap was **precision**; N9 falsified it (`f = −0.07`, the sign is wrong).
Already eliminated and not to be re-eliminated: protocol · hyperparameters · tail window ·
optimizer (both reduce to `p(1−lr ·wd) − lr ·m̂/(√v̂+ε)`) · gradients (2.68e-15) · minibatch
noise (**both full-batch**) · init distribution (norms agree to ~1 %) · the train/test split
(**byte-identical, the same 3,830 pairs, verified by set comparison**) · precision.
**Left standing: the specific init draw, and the substrate.** This run varies the substrate
alone: the k03 kernel's **own source**, unmodified except for device and job list, on CPU.

`run_o18_cpu.py` applies a short, explicit patch list to `kernels/k03_grid_acts/run.py` and
**asserts each patch matches exactly once**, rather than copying the training loop — a copy
can drift from the code that produced the numbers being compared against, and then the A/B
is not an A/B. Its `_selfcheck` asserts the loss line, the optimiser step, the split RNG and
the hyperparameter line all survive the patch.

## Prediction

**No directional prediction is made, and that is deliberate.** Both branches are decisive
and the criterion is the *fraction of the gap*, not its sign:

- CPU reads **≈ 0.77** (engine-like) → the difference is **GPU-specific**, and every torch
  number in this project — all of k02–k08 — inherits a substrate artefact;
- CPU reads **≈ 0.55** (Kaggle-like) → the gap is **in the code**, and the two
  implementations can be bisected line by line, starting from the init draw.

Pre-registering a "we expect X" here would be theatre. What is pre-registered is the
statistic, its thresholds, and the fact that **neither branch may be reported as
confirmation of a story chosen afterwards** — C30's retraction is exactly that failure.

## Success criteria

Reference values, all already on disk and fixed before this run:
engine float64 **0.7536** · engine float32 **0.7671** · **k03 T4 torch float32 0.5665**
(5-seed tail means, `analyze_n9.py`).

| # | criterion | statistic | threshold |
|---|---|---|---|
| **Q0** | the CPU runs grok | test acc > 0.99 | **≥ 2/3 seeds**, or Q1–Q3 are not scored |
| **Q1** | **PRIMARY.** Fraction of the Gini_mult gap explained by the substrate | `f = (Gini_cpu − Gini_T4) / (Gini_engine_f64 − Gini_T4)`, tail mean over the final 10k steps, amplitude protocol | **SUBSTRATE HELD** `f ≥ 0.70` · **NOT HELD** `f ≤ 0.30` · `0.30 < f < 0.70` is **PARTIAL** and must be reported as such, not rounded to either |
| **Q2** | the embedding-norm gap moves with it | `f` on `‖W_E‖` (T4 15.152 vs engine 11.610) | same bands. A substrate explanation must move **both** or it is not the mechanism |
| **Q3** | neuron tuning moves with it | tuned fraction at 0.85 (T4 92–100 %, engine 70–93 %) | descriptive; **this is the fact no single convergence axis explains** (Entry 29) and it is the one any answer must fit |
| **Q4** | grokking time | steps to acc > 0.99 | descriptive, **3 seeds, variance reported**. Seed variance on grok time is a factor of 3.8 in this project; no ordering is interpretable at 3 seeds |

Thresholds are N9's verbatim, so the two diagnostics are comparable and neither was tuned.

## What would falsify this

`f ≤ 0.30` on Q1. Then the substrate is not the cause either, only the init **draw** is
left, and the next step is a direct numerical bisect of the two implementations from
identical weights.

## Known confounds

- **Seed.** 3 seeds (0, 1, 2), matching N9 and k03's first three. Q1 is a comparison of
  means; per-seed values are reported.
- **BLAS threading is not a scientific variable but it is a numerical one.** Recorded in
  the log; the run is full-batch float32 either way.
- **Only one modulus (113).** This is a bisect, not a claim about moduli. It inherits
  nothing about 121/125/119 and must not be written up as if it did.
- **Tail window.** Fixed at the final 10k steps, the window C26 and N7's E2 already use.

## Analysis plan

`PYTHONPATH=. .venv/bin/python analyze_n9.py results/o18_cpu` — the **same scorer N9 used**,
pointed at the new directory, so the comparison cannot drift. Any other analysis is
exploratory and labelled so.

## Outcome (filled in AFTER the run — never edit anything above)

**Date scored:** 2026-09-14, 22:07 · 3 runs × 40,000 steps, sequential, local CPU,
**108 min wall, zero Kaggle quota** · artifacts `results/o18_cpu/`, SHA `846dbb1` ·
log `logs/o18_cpu.log` ("CPU torch 2.14.0+cpu threads=6") · score `logs/o18_score.log`.

### Result: the hypothesis is FALSIFIED. The substrate explains essentially none of the gap.

| regime | tail Gini_mult | mean | `‖W_E‖` | tuned neurons | grok steps |
|---|---|---|---|---|---|
| engine float64 (N7) | 0.7623 · 0.7307 · 0.7679 | **0.7536** | 11.610 | 0.887 | 4,500 · 6,500 · 9,700 |
| torch **T4 GPU** (k03) | 0.5573 … 0.5825 | **0.5665** | 15.152 | 0.969 | 6,800 … 5,400 |
| **torch CPU (this run)** | **0.5759 · 0.5672 · 0.5942** | **0.5791** | **14.296** | **0.938** | **5,800 · 7,200 · 5,600** |

**Torch on a CPU behaves like torch on a T4, not like the engine.**

- **Q0 HELD** — 3/3 grokked, test acc 1.0000 at every seed.
- **Q1 NOT HELD**, `f = 0.068` (threshold: HELD ≥ 0.70, NOT HELD ≤ 0.30). The substrate
  accounts for **6.8 %** of the Gini_mult gap.
- **Q2 NOT HELD**, `f = 0.242`. About **a quarter** of the embedding-norm gap does move
  with the substrate — small, not zero, and reported as measured.
- **Q3 (descriptive)** — tuned neurons: CPU 0.938 vs T4 0.969 vs engine 0.887, so
  `f = 0.384`, i.e. **PARTIAL** on the band. **Do not read this as a third of the tuning
  gap being hardware**: the CPU seeds are 0.998 / 0.814 / 1.000 and one seed decides the
  mean. Three seeds cannot separate 0.94 from 0.97.
- **Q4 (descriptive)** — grok steps 5,800 / 7,200 / 5,600 against T4's
  6,800 / 7,600 / 5,600 / 6,600 / 5,400 and the engine's 4,500 / 6,500 / 9,700. All three
  bands overlap; **nothing is claimed**, per the pre-registration's own warning that seed
  variance on grok time is a factor of 3.8 in this project.

### What this closes, and what is left

**O18's candidate list is now down to one.** Eliminated, with the run that did it:
protocol · hyperparameters · tail window · optimizer · gradients (2.68e-15) · minibatch
noise (both full-batch) · init **distribution** · the train/test split (byte-identical) ·
**precision** (N9) · **the execution substrate** (this run). What remains is the specific
random **init draw** — numpy's stream versus torch's — or a structural difference not yet
found.

**And it is now cheap to attack.** Both implementations run on the same box, so the next
step is not another sweep: initialise the engine *from torch's own sampled weights* (or
the reverse) and compare step by step. If the gap survives byte-identical starting
weights it is in the update path, and the update path is 400 lines.

### Deviations from this pre-registration, and why

1. **The analysis plan named a command that did not work.** §"Analysis plan" says
   "`analyze_n9.py results/o18_cpu` — the same scorer N9 used". It found **zero runs**:
   `analyze_n7.load` globs `WE_engine_*`, and `run_o18_cpu.py` writes `WE_B_thesis_*`.
   It did not error — it printed *"P0 NOT ASSESSABLE (no runs)"*, indistinguishable from a
   sweep that failed to grok, on a run that had grokked at step 5,800. **I pre-registered a
   command without running it**, which is the failure LAB_PROTOCOL.md already records for
   `analyze_n7.py`. Fixed (`f203dec`) so `load` accepts either naming.
2. **That same scorer labelled this arm "engine float32".** Its labels were hard-coded for
   N9. Fixed (`b01f6b6`) to read `config.engine` from each artifact and to include the
   directory, and to print **both** sign conventions named, since N9's `f` and O18's `f`
   are complements and nothing on screen said so.
3. **`device` was not stamped into these three artifacts.** The substrate is this
   experiment's one variable and it was not in the npz. `run_o18_cpu.py` now stamps
   `device` / `torch_version` / `threads`; these three runs predate the line and their
   device is recorded here and in `logs/o18_cpu.log`.
4. **3 seeds were run, not the single seed STATE.md budgeted** — the 400-step smoke
   measured 46 ms/step, so three fit in the time one was expected to take.
5. **No criterion or threshold in §"Success criteria" was altered after the data existed.**
