# Pre-registration: O21 — why is the engine dtype-invariant when torch is not?

**Commit this file BEFORE launching the run.**

**Date:** 2026-09-15 · **Script:** `run_o21_probe.py`
**Arm:** methods (instrumentation, not training) · **Cost:** ~1 min local CPU, zero quota

## Background — what is already settled, so this does not re-test it

O18 is **closed** (C34): training precision changes measured circuit sparsity **in
PyTorch** — `f = +1.114` on Gini_mult, float32 0.5791 → float64 0.7736 at n = 113. It is
an **interaction**, not C30's retracted main effect.

The residue is O21. The engine reads tail `Gini_mult` **0.7671 at float32** (N9) and
**0.7536 at float64**, i.e. it sits at the float64 answer *at either dtype*, while torch
does not. N9 verified the float32 engine arm is genuinely float32 by ancestry (the NEP-50
fix `6991152` is an ancestor of N9's `a4b3d14`). **Nothing explains the invariance.**

## Hypothesis

The engine is dtype-invariant because **part of its computation is not actually performed
at `ENGINE_DTYPE`**. `Tensor.__init__` (`src/autograd/engine.py:63`) casts every tensor's
`.data` to `DTYPE` **on construction** — so an intermediate that numpy promoted to float64
is computed in float64 and then *rounded down*, which is silently more accurate than a
true float32 computation and is invisible to any check that inspects a finished `Tensor`.

Three candidates, from `STATE.md` O21: **(a)** a different accumulation order, **(b)** a
different softmax formulation, **(c)** residual NEP-50 promotion beyond the five tensors
`test_autograd.py::test_dtype_flag` covers. This probe tests **(c)** directly and is
diagnostic for (a)/(b) only by elimination.

## Prediction

Under `ENGINE_DTYPE=float32`, at least one operator site produces a **pre-cast** intermediate
of dtype float64 in the forward, the backward, or both. `test_dtype_flag` cannot see these
because it inspects **parameters, parameter gradients, Adam state and the loss** — all of
which are downcast by the time they are checked.

## Success criteria — exact, implemented as code before the run

| # | criterion | threshold | implemented in |
|---|---|---|---|
| **1 (PRIMARY)** | distinct forward sites whose **pre-cast** `_child` data is float64 under `ENGINE_DTYPE=float32` | **≥ 1 ⇒ promotion CONFIRMED**; 0 ⇒ candidate (c) is DEAD | `run_o21_probe.py::probe` |
| **2** | distinct backward sites whose `_accum` gradient is float64 under float32 | ≥ 1 ⇒ promotion CONFIRMED | same |
| **3 (POSITIVE CONTROL)** | the same probe at `ENGINE_DTYPE=float64` must report **float64 at essentially every site** | ≥ 90 % of sites float64, **else the instrument is broken and criteria 1–2 are void** | same |
| **4 (NEGATIVE CONTROL)** | a deliberately planted promotion (`Tensor(x) * np.int64 array`) must be **detected** at float32 | detected, else the instrument is blind | `run_o21_probe.py::_selfcheck` |

**Criterion 3 is the one that matters most and it is why this is not just a print
statement.** A probe that reports "0 float64 sites" is indistinguishable from a probe that
cannot read dtypes at all. This project has been burned four times by a check whose failure
mode looks like its success mode (`git rev-parse | uniq | wc -l`; the `pgrep` liveness
check; the C7b/C19/C20 controls; the CRT predicted-set off-by-one, which was caught **by a
positive control, never by reading**). The float64 run is the positive control.

## What would falsify this

**Criteria 1 and 2 both zero, with criterion 3 passing.** That kills candidate (c)
outright: the engine really does compute in float32 end to end, and the invariance is then
(a) accumulation order or (b) the softmax formulation — a genuinely different question,
answered by comparing the engine's `softmax`/`log_softmax` against torch's, not by this
probe.

## Known confounds

- **NEP 50 makes Python scalars weak**, so `array_f32 * 1e-3` stays float32. Promotion
  needs an **array** operand of a wider dtype — an int64 index array, a float64 constant
  built by `np.zeros(...)` or `.astype(float)`. The probe must therefore report the
  *operand* dtypes, not only the result, or a confirmed promotion cannot be attributed.
- **The mask and the init draw are built in float64 and cast down by construction**
  (`transformer.py:18,28`). That is by design, identical at both dtypes, and must not be
  counted as a promotion — it is a constructor cast, not an arithmetic one. The probe
  counts only sites reached from `_child`, i.e. actual operations.

## Analysis plan — decided now

One forward + backward + optimizer step of the real `Transformer` at n = 113, batch as in
`run_n7.py`, seed 0, under each of `ENGINE_DTYPE=float32` and `float64`. Record for every
`_child` call the op name, the pre-cast result dtype and both operand dtypes; for every
`_accum` call the gradient dtype. Report the distinct offending sites with their operator
and a source location. **No training, no sweep, one step.**

## Outcome

*(to be filled after the run — nothing above this line is edited)*

---

# AMENDMENT — the pre-specified follow-up, written BEFORE it is run

**2026-09-15, immediately after the probe returned 0/0 with criterion 3 passing.**

The body above pre-specifies this: *"the invariance is then (a) accumulation order or
(b) the softmax formulation — a genuinely different question, answered by comparing the
engine's softmax/log_softmax against torch's."* This amendment gives that comparison exact
criteria before it runs. **Nothing above this line is edited.**

## Hypothesis

The engine's float32 arithmetic is **more accurate than torch's float32 arithmetic** on the
same inputs — not wider, but better-rounded — so it lands nearer the float64 answer without
ever leaving float32. Two places this could come from: the **matmul accumulation** (numpy's
BLAS `sgemm` vs torch's, which may block, vectorise or FMA differently) and the
**log-softmax formulation**.

## Method — no training, pure numerics

For each of matmul and log_softmax: take one input, evaluate it three ways — numpy at
float32, torch at float32, and a **float64 reference** treated as ground truth — and measure
each float32 result's distance from that reference. Real logits from a forward pass, not
only random arrays, since the question is about this model's operating regime.

## Success criteria — exact

| # | criterion | threshold | meaning |
|---|---|---|---|
| **A1** | relative error to the float64 reference, numpy vs torch, on **matmul** | numpy's error **≤ 0.5×** torch's ⇒ candidate (a) **SUPPORTED** | the engine's matmul is better-rounded |
| **A2** | same, on **log_softmax** | numpy's error **≤ 0.5×** torch's ⇒ candidate (b) **SUPPORTED** | the engine's softmax is better-rounded |
| **A3** | if both ratios lie in **[0.5, 2.0]** | the two implementations are **numerically equivalent at float32**, and BOTH remaining candidates are dead | O21 survives as an open question with no candidate left |

The 0.5× threshold is not arbitrary: O18's measured effect is a **factor-1.1 shift in an
interaction**, and a numerical mechanism that explains it must show a *systematic* accuracy
difference, not a coin-flip. Two implementations that differ only by rounding noise land
near 1.0 with random sign.

## What would falsify it

**A3 firing.** If numpy and torch agree to within a factor of two in accuracy at float32,
then neither accumulation order nor softmax formulation explains the engine's dtype
invariance, and O21 becomes a question with **no surviving candidate** — which is a
reportable state, not a failure, and belongs in Limitations.

## Known confound, stated now

A float32 result compared against a float64 reference reads ~1e-7 and **looks exactly like
a layout bug** (`LAB_PROTOCOL.md`). The reference is built by promoting the *same* float32 inputs
with `.double()`; the float32 → float64 → float32 round trip is exact, so the inputs are
unchanged and any difference is arithmetic, not representation.

## Outcome

*(to be filled after the run)*

---

# OUTCOME — 2026-09-15

**Run:** `run_o21_probe.py {float64,float32}` and `run_o21_numerics.py`, n = 113, seed 0,
one step. SHA at run: `b7108d4`, tree clean. Total cost **under two minutes**, zero quota.

## Criteria 3 and 4 — the controls, reported first because 1 and 2 are void without them

| # | criterion | result | verdict |
|---|---|---|---|
| **4** | planted `float32 × int64` promotion must be detected | detected; a clean float32 op flagged **0** sites; the constructor confirmed to cast the planted result back down to float32 | **PASS** |
| **3** | at `ENGINE_DTYPE=float64` the probe must read float64 at ≥ 90 % of sites | **100.00 %** — 115/115 calls (46 forward, 69 backward), 0 exceptions | **PASS** |

Criterion 4 also *demonstrates the hiding mechanism*: a genuine float64 promotion is
rounded back to float32 by `Tensor.__init__`, so it is invisible to any check that inspects
a finished tensor. That is why `test_dtype_flag` passes while O21 stayed open.

## Criteria 1 and 2 — the question

| # | criterion | result | verdict |
|---|---|---|---|
| **1 (PRIMARY)** | forward sites whose **pre-cast** data is float64 under float32 | **0** across 12 distinct sites, 46 calls | **NOT FOUND** |
| **2** | backward sites whose `_accum` gradient is float64 under float32 | **0** across 16 distinct sites, 69 calls | **NOT FOUND** |

**Fraction of calls at the requested dtype: 1.0000 at both dtypes.** Loss identical to six
decimals (4.747254) in both regimes, as expected from a shared init draw.

### ⇒ Candidate (c), residual NEP-50 promotion, is DEAD.

The engine computes in float32 end to end. The `.astype(g.dtype)` fix in `max()`'s backward
(`6991152`) was the only such site, and nothing else survives.

## Amendment criteria A1–A3 — the pre-specified follow-up

Relative error against a float64 reference built from the **same** float32 inputs:

| case | shape | numpy f32 | torch f32 | ratio | verdict |
|---|---|---|---|---|---|
| matmul, synthetic | (3830,128) @ (128,512) | 5.037e-07 | 5.037e-07 | **1.000** | A3 |
| matmul, real MLP | (3830,512) @ (512,128) | 4.547e-07 | 3.880e-07 | **1.172** | A3 |
| log_softmax, real logits | (3830,113) | 1.206e-07 | 1.178e-07 | **1.024** | A3 |
| log_softmax, logits × 20 | (3830,113) | 7.656e-08 | 7.365e-08 | **1.039** | A3 |

Every ratio lies inside [0.5, 2.0]. **A3 FIRES.**

### ⇒ Candidates (a) accumulation order and (b) softmax formulation are BOTH DEAD.

numpy and torch are numerically indistinguishable at float32 on this model's own shapes and
its own activations, including a deliberately hard high-logit case.

## What this means, stated exactly

**O21's three enumerated candidates are eliminated. O21 remains OPEN with no surviving
candidate mechanism.** That is a reportable state, not a failure — and it is the honest one.

It does **not** re-open O18, which is closed by C34 on the torch side: float32 → float64
moves torch's Gini_mult 0.5791 → 0.7736, `f = +1.114`. What is unexplained is narrower and
should be written narrowly: **the engine reaches the float64 answer at either dtype
(0.7671 / 0.7536, Δ 0.0135) while torch moves 14× further with the same variable
(0.5791 / 0.7736, Δ 0.1945).**

**Do not write this as "the engine is more accurate."** It is not: the numerics are
identical. Whatever separates the two implementations is not in the dtype of any
intermediate, not in the matmul accumulation, and not in the softmax — and the eliminated
list now also carries protocol, hyperparameters, tail window, optimiser, gradients
(2.68e-15), minibatch noise, init distribution, the byte-identical split, the substrate
(`f = 0.068`), the update path (one AdamW step, 8.65e-15) and the init draw (`f = +1.156`).

**Cumulative: seven hypotheses eliminated across O18 and O21; one answered (precision, on
torch); the engine's dtype-invariance unexplained.** Paper treatment: a Limitations line and
a methods appendix, never a claim.
