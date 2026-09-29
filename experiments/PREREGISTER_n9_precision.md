# Pre-registration: N9 — the precision A/B (O17 / C30)

**Commit this file BEFORE launching the run.** Git's timestamp is the evidence that the
criteria predated the result.

**Date:** 2026-09-14 · **Script:** `run_n9_f32.sh`
driving `run_n7.py` unchanged · **Arm:** **thesis**, engine, `ENGINE_DTYPE=float32` ·
**Runs:** n=113 × seeds 0,1,2 × 40,000 steps · **Compute:** local CPU, ~3–4 h wall at
3-way, **zero Kaggle quota**

## Why

N7's E5 predicted the engine and the PyTorch kernels would agree and **they did not**, at
all three moduli, in the same direction, by 4–8σ. The chase eliminated the boring causes
(same `tail_gini` function on both sides; betas, init scales, lr, wd, steps, train_frac and
architecture all match; the gap survives a window-independent read of the final weights in
7/7 grokked pairs). What remains is that `src/autograd/engine.py` trained every Tensor in
**float64** while the Kaggle/PyTorch kernels train in **float32**.

That is a hypothesis, not a finding. This run tests it by changing **that one variable and
nothing else**: the same script, the same seeds, the same data split, the same 40,000
steps, on the same engine, with `ENGINE_DTYPE=float32`.

It matters beyond housekeeping. Every pre-N7 number in this project is float32 and every
N7 number is float64; if precision moves the headline sparsity statistic by 0.19 Gini then
the paper must report which regime each number lives in, and precision becomes an
unrecognised knob on the lazy→rich transition that nobody in the corpus studies.

## Hypothesis

**The engine–PyTorch sparsity gap is caused by training precision.** float32's larger
gradient noise under `wd=1.0` leaves the model in a less converged, less
feature-learned solution; float64 lets weight decay drive a sparser, smaller-norm
multiplicative circuit.

## Prediction

The float32 engine reads in **PyTorch's** band, not the float64 engine's. Reference
numbers, all measured on disk **before this run** (`tail_gini`, mean Gini_mult over the
final 10k steps; `|W_E|` on the n rows of the final embedding):

| regime | source | tail Gini_mult per seed | mean | `|W_E|` |
|---|---|---|---|---|
| engine **float64** | `results/n7_engine`, s0–2 | 0.7623 ± 0.0005 · 0.7307 ± 0.0738 · 0.7679 ± 0.0023 | **0.7536** | 11.40 · 11.40 · 12.03 |
| torch **float32** | `results/k03_grid_acts` B_thesis, s0–4 | 0.5573 · 0.5370 · 0.5743 · 0.5812 · 0.5825 | **0.5665** | 15.69 · 15.76 · 15.60 · 13.56 · 15.15 |

The bands are separated by **0.187 Gini** against within-run sd ≈ 0.02, so a 3-seed read
is not a marginal test.

Define **fraction explained** for a statistic `S`:

```
f(S) = (mean_float64_engine(S) - mean_float32_engine(S))
     / (mean_float64_engine(S) - mean_torch_float32(S))
```

`f = 1` means precision accounts for the whole engine–torch gap; `f = 0` means precision
accounts for none of it. `f` is computed over the **grokked** float32 seeds only (see
Controls).

## Success criteria — exact, and implemented as code before the run

| # | criterion | threshold | implemented in |
|---|---|---|---|
| P0 | the float32 engine still groks | test acc > 0.99 in **≥2 of 3** seeds | `analyze_n9.py::p0` |
| P1 | **primary.** precision explains the Gini_mult gap | `f(Gini_mult) ≥ 0.70` → HELD · `≤ 0.30` → NOT HELD · between → **PARTIAL** | `analyze_n9.py::p1` |
| P2 | precision explains the embedding-norm gap | same three-way rule on `f(|W_E|)` | `analyze_n9.py::p2` |
| P3 | the sparsity is still **multiplicative**, not an artifact | per grokked seed, Gini_mult > Gini_add and > Gini_rand | `analyze_n9.py::p3` |

P1 is the claim. P2 is a second, independent statistic on the same mechanism: if precision
is the cause, both should move together, and a P1-without-P2 result means the story is
about sparsity alone and not about weight decay reaching a smaller-norm solution.

## What would falsify this

- **P1 ≤ 0.30** — the float32 engine stays at ~0.75. Precision is **eliminated**, C30 is
  withdrawn, and the engine–torch gap must be hunted elsewhere (optimizer implementation,
  `eps`, the data split, the MLP-at-read-out-position optimisation, torch's init). That is
  a useful negative: it is the result that keeps a precision artifact out of the paper.
- **P0 fails** (≤1 of 3 seeds groks) — the comparison is void, not negative. An ungrokked
  run is the untrained control (`analyze_n7.grokked`), never a data point. Report as
  "not assessable" and re-run at more seeds.
- **P1 HELD but P3 fails** — float32 is sparse in no particular basis, i.e. the statistic
  is reading something other than a clock. Would retract, not support, the mechanism.

## Required controls

| claim shape | control |
|---|---|
| "sparse in basis X" | Gini_add and Gini_rand alongside Gini_mult, every seed (P3) — LAB_PROTOCOL.md forbids reporting Gini_mult alone |
| "X differs from Y" | 3 seeds here against 3 float64 and 5 torch seeds already on disk; within-run sd reported |
| "precision is the cause" | the **float64 arm is the control** and it is already run, at the same seeds, the same split and the same SHA-lineage |
| "the flag actually works" | `test_autograd.py::test_dtype_flag` asserts weights, grads, Adam state and loss are **all** float32 — a float32-weights/float64-grads arm is not this experiment |

## Known confounds

- **Speed is not free of the variable.** float32 runs 270 ms/step vs float64's 422
  (measured, 40-step window, single process). Nothing in the analysis reads wall time, but
  the two arms therefore finish at different wall clocks; compare by **step**, never by
  time.
- **The engine's float32 is still not torch.** Adam implementation, `eps`, and the
  read-out-position MLP optimisation differ. So `f < 1` is expected even if precision is
  the whole *dtype* story; `f` measures how much of the gap dtype accounts for, and the
  residual is named, not explained.
- **Seed variance.** n=113 float64 s1 carries within-run sd 0.0738, an order above its
  siblings. Three seeds bound this loosely; if the three float32 seeds straddle the 0.30/0.70
  boundary the honest verdict is PARTIAL, and the fix is seeds 3–4, not a re-read.
- **`max`'s tie-count fix** (this commit) changes the float64 arm's gradients by nothing
  (`test_model.py` still 2.68e-15) but is a code change between the two arms' SHAs. It is
  the *reason* the float32 arm is honest; recorded here so the SHA difference is not
  mistaken for a silent divergence.

## No prediction is made about grokking time — deliberately

O1b asks why this project groks at ~4,500 steps where the literature reports 9k–14k, and
"float64 groks faster" is the tempting story. **The numbers on disk do not support it at
n=113**: engine float64 groks at 4,500 / 6,500 / 9,700 and torch float32 at 6,800 / 7,600 /
5,600 / 6,600 / 5,400 — overlapping bands, torch's mean *lower*. Grokking time is recorded
as **exploratory** and no criterion depends on it. Pre-registering this now is what stops
whichever direction it lands in from becoming a post-hoc explanation.

## Analysis plan — decided now

`PYTHONPATH=. .venv/bin/python analyze_n9.py`, reading `results/n9_f32/` against
`results/n7_engine/` (float64) and `results/k03_grid_acts/` (torch float32), scoring P0–P3
in that order. Statistics are the **existing** ones, unmodified: `analyze_n7.tail_gini`,
`analyze_n7.mult_amplitude` / `add_amplitude` / `rand_amplitude`,
`src.analysis.sparsity.gini` / `participation_ratio` / `key_freqs_5x_median`. Gini on
**amplitude**, PR on **energy**.

Also recorded per seed, **exploratory, no criterion attached**: grok step, PR, `n_key`,
final train/test loss and acc, and the float32 arm's wall-clock rate.

Anything not listed above is exploratory and must be labelled so in the write-up.

## Outcome (filled in AFTER the run — never edit anything above)

**Date scored:** 2026-09-14, 18:49 · 3 runs × 40,000 steps, 3-way local CPU, 3 h 13 min
wall, zero Kaggle quota · all three artifacts `git_dirty=False` at SHA `a4b3d14`, `dtype`
stamped `float32` · full output `logs/n9_finish.log`

### Result: the hypothesis is FALSIFIED. Precision explains none of the gap.

| regime | tail Gini_mult per seed | mean | `|W_E|` mean |
|---|---|---|---|
| engine **float64** | 0.7623 · 0.7307 · 0.7679 | 0.7536 | 11.610 |
| engine **float32** | **0.7607 · 0.7785 · 0.7622** | **0.7671** | **11.642** |
| torch **float32** | 0.5573 · 0.5370 · 0.5743 · 0.5812 · 0.5825 | 0.5665 | 15.152 |

Changing the engine to float32 moved Gini_mult by **+0.014** — *away from* torch, not
toward it — and `|W_E|` by **+0.03** against a 3.5 gap.

- **Criteria met:**
  - **P0 HELD** — 3/3 grokked, test acc 1.0000 at all three seeds.
  - **P1 NOT HELD**, `f = −0.07` (threshold: HELD ≥ 0.70, NOT HELD ≤ 0.30). Not merely
    below threshold — the sign is wrong.
  - **P2 NOT HELD**, `f = 0.01`. The embedding-norm gap is untouched.
  - **P3 HELD** — the float32 engine is still multiplicative: Gini_mult 0.76–0.78 against
    Gini_add 0.005–0.016 and Gini_rand 0.11–0.13, `n_key` 5–8, PR 3.9–4.4.

- **Exploratory (no criterion, as pre-registered): grokking time is precision-invariant.**
  float32 groks at **4,500 / 6,400 / 9,600**; float64 at **4,500 / 6,500 / 9,700**. The
  refusal to predict this was the right call — "float64 groks faster, which is why we grok
  at 4,500 where the literature reports 9k–14k" was the tempting story, and it is false.
  **O1b is not explained by precision** and returns to open.

- **Deviations from this pre-registration: none.** The run executed as specified: same
  script, same seeds, same 40,000 steps, one variable changed. No criterion was altered
  after the data existed.

### What this eliminates, and what is left

C30 claimed precision changes the measured sparsity of a grokked circuit. **It does not.**
C30 is **RETRACTED**. The engine-vs-torch gap it was invented to explain is still there and
is now harder, not easier, to explain — the elimination list for a difference of ~10σ is:

| candidate | status |
|---|---|
| analysis protocol | eliminated — the same `tail_gini` / `mult_amplitude` on both sides |
| hyperparameters | eliminated — lr, wd, betas, steps, train_frac read from the k03 kernel source |
| tail window | eliminated — window-independent check on final weights, 7/7 |
| optimizer | eliminated — both reduce to `p(1−lr ·wd) − lr ·m̂/(√v̂+ε)`; verified by reading both |
| **training precision** | **ELIMINATED BY THIS RUN** |
| gradient correctness | eliminated — `test_model.py`, 2.68e-15 against PyTorch |
| minibatch noise | eliminated — **both are full-batch**; k03 calls `m(xtr)` on the whole training set every step, as `run_n7.py` does |
| initialisation *distribution* | eliminated — both `N(0, 1/√d_model)`; measured per-parameter norms agree to ~1% |
| **the train/test split** | **eliminated — byte-identical.** `modular_data(113,"mul",0.30,0)` and k03's `dataset()` select the *same 3,830 pairs*, verified by set comparison, not by reading the comment that claimed it |

**Still standing:** the specific random init *draw* (numpy's stream vs torch's, same
distribution), and execution substrate (k03 ran on a Kaggle **T4 GPU**; the engine runs on
CPU). Nothing else survives.

**A second, orthogonal observation that argues the two differ in STRUCTURE, not magnitude.**
Neuron-level tuning (C31's statistic) runs the *opposite way* to Gini:

| regime | tail Gini_mult | tuned neurons at n=113 |
|---|---|---|
| engine float64 | 0.754 | 82.4 · 90.2 · 93.4 % |
| engine float32 | 0.767 | 82.2 · 70.7 · 90.0 % |
| torch float32 | 0.567 | 94.9 · 91.8 · 100.0 · 99.8 · 98.0 % |

The engine has the **sparser embedding** and the **less pure neurons**; torch has the
reverse. A single "more/less converged" axis does not produce that. It is the shape of a
*different solution*, which is Clock-and-Pizza territory (2306.17844: small changes induce
qualitatively different algorithms) and should be tested with their discriminator rather
than with another sparsity scalar.

**Next test, and it is cheap:** run the k03 recipe with **torch on CPU**, locally, one seed.
If it reads ~0.77 the difference is GPU-specific; if it reads ~0.55 the difference is in the
code and can be bisected against the engine line by line. Either answer is decisive, and it
costs no quota.
