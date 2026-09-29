# Pre-registration: I1 — the internal character intervention

**Date:** 2026-09-22 · **Script:**
`run_intervention.py` (+ `src/analysis/intervention.py`) · **Arm:** thesis

## Why this run exists

Every causal statement in this project so far routes through `ablation_report`, which
FFTs the **saved logit tensor**, masks, inverts and measures cross-entropy. The panel's
Methodology reviewer and the EIC both accepted that as scoped, and §5 says so plainly:
it shows the *output* carries character structure the loss depends on. It does not show
the **network internally computes with those characters**, because the transform is a
post-hoc operation on the output, not an intervention on the mechanism.

This run does the intervention that the word "causal" actually requires: modify a weight
matrix the network reads, **let the network run**, and measure the loss at the output.
Softmax attention and the ReLU MLP sit downstream of the edit, so nothing about the
result is algebraically forced the way a logit-space mask partly is.

## Hypothesis

The key multiplicative characters of the embedding are the representation the network
**computes with**, not merely the basis in which its output is sparse: deleting them from
`W_E` and running the unmodified forward pass destroys the model, and no equal-sized set
of non-key characters does comparable damage.

## Prediction

At n=113, on a grokked engine checkpoint whose `W_E` and `logits_all` reproduce the
archived `results/n7_engine/WE_engine_n113_s*.npz` exactly:

- **excluded** (the key characters deleted from `W_E`, conjugates closed) raises
  cross-entropy on the unit grid by **at least 100×** over baseline.
- **restricted** (only the key characters and DC kept in `W_E`) leaves a **working
  model**: unit-grid accuracy stays at or above 0.90.
- the **excluded** loss is beyond **all** draws of a null that deletes a random
  equal-sized, conjugate-closed, non-key character set.

I explicitly do **not** predict that restricted *improves* on baseline. In logit space it
does at 6 of 7 runs (n=113 units-only: 9.0× better), but that test zeroes everything
outside the key set **in the output**, which partly builds the result in. Here the edit is
upstream of two nonlinearities and no such guarantee exists. Predicting the logit-space
magnitudes would be importing a number from a different experiment.

## Success criteria — exact, and implemented as code before the run

Every threshold below is **inherited from a criterion this project already fixed for
another purpose**, not chosen for this run. That is deliberate: it is the one defence
against the C6/C7 ordering error that does not depend on my restraint.

| # | criterion | threshold | inherited from | implemented in |
|---|---|---|---|---|
| I1 | necessity: `excluded / baseline` loss ratio | **≥ 100×** | Gate 2's **G2**, fixed for the stratum sweep | `run_intervention.py::verdict` |
| I2 | sufficiency: restricted unit-grid accuracy | **≥ 0.90** | **`FAIL_ACC`**, the C27 excursion floor (`932ef89`, 09-15) | `run_intervention.py::verdict` |
| I3 | specificity: permutation p of `excluded` against the random-set null | **p < 0.01** | **C22/C23** and Gate 2's **G1** | `run_intervention.py::verdict` |

`B = 10,000` draws, matching R1. The floor is `1/10001 = 9.999e-05`; a result at the floor
is reported as a **bound**, never as `p = 0.0000` (R1's lesson, and the reason B is not 200).

**All three must hold, on at least 2 of the 3 seeds, for I1 to be recorded as met.**
A 1-of-3 result is reported as such and claims nothing.

## What would falsify this

- `excluded / baseline < 100×` — the characters are not necessary to the computation, and
  §5's scoping stands exactly as shipped.
- `p >= 0.01` — a random character set does comparable damage, so the *key* set is not
  special and the specificity claim dies even if the magnitude survives.
- Restricted accuracy `< 0.90` **together with** I1 passing would mean the characters are
  necessary but not sufficient — a real and reportable outcome, not a failure of the run.
  It is n=119's logit-space signature (necessary, `restricted` 2.0× worse) appearing
  internally, and it would be written up as that.

## Required controls

| claim shape | control | status |
|---|---|---|
| "component C is responsible" | ablate C **and** a random equal-sized set | I3's null, B = 10,000, conjugate-closed, non-key, same `|K|` |
| "this measures the same thing as the published test" | run the **logit-space** ablation on the same checkpoint and reproduce the archived engine numbers | positive control, asserted before any internal number is read |
| "the checkpoint is the published model" | `W_E` and `logits_all` equal the archive | asserted; the run aborts if not |
| instrument correctness | plant a pure character of known index, assert it lands where expected; assert an unmasked round-trip is the identity | `intervention.py::_selfcheck` |

## Known confounds

- **Set size.** Three claims have died here (C7b, C20, C19). The null draws exactly `|K|`
  characters, conjugate-closed the same way, so the compared sets are the same size.
- **DC.** Restricted keeps DC, excluded never touches it — the same convention
  `fourier_ablate(keep_dc=True)` uses, so the two tests are comparable.
- **Non-unit rows.** The character basis is defined on units only. Row 0 (a non-unit at
  n=113) and the `=` token row are **passed through untouched**, and the loss is reported
  on the unit sub-grid. Stated, not silently done.
- **Bin convention.** Three bin bugs in this project have been lost to reasoning. The
  convention is settled in `_selfcheck` by planting a signal of known frequency and using
  `argmax`, which no threshold can rescue or break.
- **Seed.** Three seeds, reported separately; no pooling into a mean.

## Analysis plan — decided now

1. Retrain n=113 seeds 0/1/2 with `run_n7.py` at HEAD, with `snapshot()` retaining all
   nine parameter matrices. The training path is **numerically identical** to the archived
   runs' SHA `f9d15b8`: the only diffs are a stamped `dtype` key, a `DTYPE` default equal
   to the old hard-coded `float64`, and a NEP-50 `.astype` that is inert at float64;
   `algebra.py`'s additions are not imported by any training file. Checked, recorded here.
2. Assert each retrained `W_E` and `logits_all` equals its archive.
3. Positive control: `ablation_report` on the retrained checkpoint reproduces the
   published engine logit-space numbers.
4. Key set from `unit_logit_grid` — **the same function the published test uses**. Not
   re-derived; two answers to "which characters are the key set" would mean one is wrong.
5. Intervention: baseline / restricted / excluded on `W_E`, full forward each time.
6. Null: B = 10,000 random equal-sized non-key sets, same forward.
7. Report median and range across seeds, with n. **No `±` below n = 5** (C36's lesson).

Anything not in this list is exploratory and will be labelled so. In particular, an
intervention on the **MLP activations** is *not* pre-registered here; `W_E` is the
cleanest causal statement and if it settles the question the MLP arm is unnecessary.

## Outcome (filled in AFTER the run — never edit anything above)

**Run 2026-09-22. Retrain 3 × 4.99–5.04 h, measurement 3 × ~113 min, all local CPU, zero
Kaggle quota.** Artifacts `results/i1_internal/I1_engine_n113_s{0,1,2}.npz`.

- **Result: I1 HELD on 3 of 3 seeds** (the criterion required 2 of 3).

| seed | baseline | restricted (acc) | excluded (acc) | excluded/baseline | `p_perm` |
|---|---|---|---|---|---|
| 0 | 1.2102e-05 | 2.0775e-06 (1.0000) | 4.7136e+01 (0.0089) | 3.895e+06 | 0.00010 (0 of 10,000) |
| 1 | 1.2743e-07 | 3.2178e-07 (1.0000) | 3.5425e+01 (0.0089) | 2.780e+08 | 0.00010 (0 of 10,000) |
| 2 | 1.5480e-07 | 2.2235e-07 (1.0000) | 2.3883e+01 (0.0091) | 1.543e+08 | 0.00050 (4 of 10,000) |

`excluded/baseline` median **1.543e+08**, range 3.895e+06–2.780e+08, n = 3. Reported as
median-and-range with n, never `±`, per C36. Excluded accuracy 0.0089–0.0091 **is chance**
(1/112 = 0.00893): deleting the key characters from `W_E` does not degrade the model, it
removes it.

- **Criteria met: I1 ✅ · I2 ✅ · I3 ✅, all three on all three seeds.**
  - **I1** (≥ 100×, inherited from Gate 2's G2): passes by 4–6 orders of margin.
  - **I2** (restricted accuracy ≥ 0.90, inherited from `FAIL_ACC`): 1.0000 on every seed.
  - **I3** (`p < 0.01`, inherited from C22/C23): 0.00010 / 0.00010 / 0.00050. Seeds 0 and 1
    sit at the design floor `1/10001`; **reported as the bound, not as `p = 0`** (R1).

**The positive control passed at a level the pre-registration did not anticipate.** Step 2
asked that each retrained checkpoint equal its archive. All three are **bit-identical** —
`max|ΔW_E| = max|ΔW_U| = max|Δlogits| = 0.000e+00` — and the grok steps reproduce
4,500 / 6,500 / 9,700 exactly. The `3.8e-06` logit gap the runner prints is float32 *storage*
of the archived tensor against a float64 recomputation, not a model difference.

### ⚠️ The null has a heavy upper tail that reaches the effect, and the reason is the design

On seed 2 the control **maximum** (2.4464e+01) **exceeds** the excluded loss (2.3883e+01);
on seed 0 it reaches 99.2 % of it. Had only `excluded / median(control)` been reported this
would have been invisible. It is not a failure of the result — it is the inherited pool.
`random_character_sets` draws from `all_freq_labels` **including the key characters**,
exactly as `analyze_n4` does, so a draw can partially reproduce the intervention.

**EXPLORATORY, not pre-registered** (`run_intervention.py --summary`): null damage is a clean
monotone dose-response in how many key characters a draw happens to remove.

| overlap | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| **0** | 1.26e-05 (n=2651) | 1.41e-07 (n=4892) | 1.64e-07 (n=4925) |
| 1 | 4.95e+00 | 4.58e+00 | 3.40e+00 |
| 2 | 1.03e+01 | 1.07e+01 | 7.44e+00 |
| 3 | 1.65e+01 | 1.79e+01 | 1.09e+01 |
| 4 | 2.39e+01 | 2.10e+01 | 1.49e+01 |
| 5 | 3.00e+01 | — | — |

A draw containing **no** key character does **nothing**: mean 1.04× / 1.11× / 1.06× baseline.
**12,468 zero-overlap draws across three seeds and not one exceeds 1.05e-04**, against an
excluded loss of 24–47. All four seed-2 draws that beat `excluded` carry overlap 3, and every
top-8 draw on every seed carries 3–5. Asserted in `--summary`: no zero-overlap draw reaches
`excluded`, on any seed.

**This makes the inherited pool conservative, and the conservatism is now quantified** — it
inflates `p` and it is the whole of the tail. The pre-registration chose that pool for
comparability with C22/C23; that choice stands.

### Deviations from this pre-registration, and why

1. **The analysis plan's step 3 ("positive control: `ablation_report` reproduces the
   published engine logit-space numbers") was run but is not a separate check**, because
   step 2 subsumed it: a bit-identical `logits_all` makes the logit-space ablation identical
   by construction. The logit-space numbers are computed and stored (`logit_*`) for the
   side-by-side, not as a control.
2. **The key-overlap decomposition above is exploratory** and is labelled so everywhere it
   appears. It was written after seeing the tail.
3. **`restricted` is worse than `baseline` on seeds 1 and 2** (2.5× and 1.4×) while better on
   seed 0 (5.8×). The pre-registration predicted this was possible and declined to predict
   improvement; it is the n=119 logit-space signature — necessary, not perfectly sufficient —
   and accuracy is 1.0000 in all three cases with both losses eight orders below excluded. A
   detector shortfall, not a failed circuit.
4. No deviation on I1, I2, I3, on B, on the null construction, or on the key-set source.
