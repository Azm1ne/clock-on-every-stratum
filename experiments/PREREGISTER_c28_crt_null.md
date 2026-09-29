# Pre-registration: C28-NULL — is the early CRT enrichment CRT-SPECIFIC?

**Committed BEFORE the structure-matched null was ever computed.** The uniform-null numbers
quoted in §Motivation already existed (they are C28 and the free pre-check below); **no
number produced by the null this file specifies exists yet.**

**Date:** 2026-09-15 · **Script:** `test_crt_null.py`
**Arm:** thesis · **Compute:** zero (re-analysis of `results/*` already on disk)

## Motivation — and why the question O14 named is NOT the question worth testing

C28 says "CRT structure is imprinted within 150 training steps". O14 proposed resolving it
with a pre-registered sweep of enrichment vs step. **That sweep is already answered, for
free**, off `results/n7_engine`'s `we_traj` (every 1,000 steps) — run 2026-09-15 before this
file, and reported here so it cannot later be passed off as a result of this experiment:

| step | s0 | s1 | s2 | test_acc | uniform-null verdict |
|---|---|---|---|---|---|
| 0 (own init) | 1.017 (p .12) | 1.006 (p .36) | 0.964 (p .99) | 0.03 | **rejects 0/3** |
| 1,000 | 1.450 | 1.225 | 1.224 | 0.054–0.075 | p = 0.000, **3/3** |
| 10,000 | 1.704 | 1.665 | 1.501 | 0.13–0.15 (ungrokked) | 3/3 |
| 20,000 | 3.471 | 3.392 | 2.420 | grokked | 3/3 |

So C28's observation is stronger than stated — not a threshold event at step 150 but a
**monotone ramp** already significant 5,000+ steps before grokking — **and it survives the
control C28 lacked**: step 0 is the model's *own* random init, same seed, same split, and it
rejects 0/3. That is settled. It is not what this pre-registration tests.

**The open risk is the NULL.** `test_crt_law.permutation_test` compares the predicted set
against *uniformly random same-size subsets*. The predicted set at n = 119 is
`{7,14,17,21,28,34,35,42,49,51,56}` — a **union of two subgroups** of Z/119, 11 of 59 folded
bins. A uniform-random null is weak against **any** alternative that concentrates energy on
arithmetic progressions through 0. This is the size/structure confound that has already
killed C7b, C20 and C19, and LAB_PROTOCOL.md's prescribed defence is exactly this file's PRIMARY:
*run the statistic on a control that has the structure but not the effect.*

**The stakes are larger than C28.** C8 is **VERIFIED on 9/9 fresh moduli** using this same
test. Either the CRT set beats a structure-matched null — and C8 becomes much stronger — or
it does not, and C8's instrument is implicated. This is the G2 lesson run forward rather
than backward: score a load-bearing statistic against a harder control *before* reusing it.

## Hypothesis

The early enrichment of embedding energy on the CRT-dual frequency set is specific to the
**true factorisation of n**, not a generic preference for subgroup-shaped frequency sets.

## Prediction

The predicted set beats a null drawn from *other* same-size unions of multiples-of-d sets,
at every step from the first post-init snapshot onward, with the margin growing to grokking.

## A falsifier I considered and am DISCARDING, with the reason

Generator equivariance (C25) cannot test this. In the **additive** basis the CRT set is the
union of two subgroups of Z/n, and I verified before writing this file that it is fixed by
**all 96 units mod 119** (`t ·P = P` for every t coprime to n). It does not move under
relabelling, so an "enrichment follows the relabelled set" criterion could never fail.
Recorded because proposing it was my error and a later reader would otherwise re-propose it.

## Success criteria — exact, implemented in `test_crt_null.py` before the run

Three nulls, all at the same set size as the predicted set, N = 2,000 draws each:

- **Null A (existing, for comparison only):** uniform random same-size subsets.
- **Null B — PRIMARY:** same-size sets built as a union of "multiples of d" progressions,
  d drawn from 2..⌊n/2⌋ **excluding every divisor of n** (so no draw can be the true CRT
  set or part of it). Preserves the subgroup/AP shape exactly; changes only *which* one.
- **Null C:** uniform random subsets **matched on the mean frequency** of the predicted set
  (within ±2), isolating a low-|k| bias specifically. (Measured: predicted mean |k| = 32.2
  against 30.0 for uniform 1..59, so a low-frequency bias is *a priori* unlikely — C is a
  cheap guard, not the main event.)

| # | criterion | threshold | implemented in |
|---|---|---|---|
| 1 | **PRIMARY.** CRT set beats Null B at the first post-init snapshot, n=119 | p_B < 0.01 in ≥2/3 grokked seeds at step 1,000 | `test_crt_null.py::early` |
| 2 | It still beats Null B at the grokked checkpoint | p_B < 0.01 in ≥2/3 seeds at 40k | `::late` |
| 3 | Monotone ramp under Null B | Spearman ρ(step, enrichment) > 0.8 over 0–20k, ≥2/3 seeds | `::ramp` |
| 4 | **Null-calibration.** Null B does not reject at init | step 0 rejects **0/3** at p_B < 0.01 | `::early` |
| 5 | **Discrimination (the G2 lesson).** Score the SAME statistic on FAILED runs and print the pass rate on both arms | reported, not thresholded — see below | `::discriminate` |
| 6 | Vacuous moduli skipped **loudly** | every prime and prime power printed as VACUOUS, never counted as support | `::main` |

**Criterion 5 is deliberately not a pass/fail.** G2's falsifier fired because a statistic
that was load-bearing elsewhere passed on 100 % of FAILED measurements. The honest output
here is the **pair** of pass rates. If FAILED runs pass Null B at a rate comparable to
grokked runs, the statistic does not discriminate and **that is the finding**, whatever
happens to criteria 1–3. Failed runs available at testable (≥2 CRT component) moduli:
k04 n=54 (2 seeds), n=63 (2), n=100 (1) — n=49 and n=125 are prime powers and are VACUOUS.

## What would falsify this

- CRT set fails to beat Null B (p_B ≥ 0.01) while beating Null A. That is the live
  possibility and it would mean **the enrichment is about subgroup shape, not about the
  factorisation of n** — retracting C28's interpretation and forcing a re-examination of C8.
- Null B rejects at step 0 (criterion 4 fails) ⇒ the null is miscalibrated and nothing
  downstream of it is readable.
- The ramp is flat or non-monotone under Null B ⇒ "imprinted early" is not a description of
  a process, just of a fixed offset.

## Required controls

| claim shape | control |
|---|---|
| "energy is enriched on set S" | Nulls A, B and C, all size-matched; B is shape-matched |
| "the effect is early / from training" | the model's **own init** at step 0, same seed, same split |
| "this distinguishes a grokked model" | FAILED runs at testable moduli, pass rate printed beside the grokked arm |
| "it is about the factorisation of n" | prime and prime-power moduli declared VACUOUS and excluded |

## Known confounds

1. **Subgroup shape** — the whole point; Null B.
2. **Low-frequency bias** — Null C.
3. **Set size** — all three nulls are size-matched.
4. **Modulus coverage.** Only n = 119 in `results/n7_engine` has both ≥2 CRT components and
   1,000-step early resolution, so the PRIMARY rests on **one modulus, 3 seeds**. Secondary
   moduli (120, 165, 54, 63, 100) come in at 4,000-step resolution from k03/k04/k06 and are
   reported separately, never pooled with the PRIMARY.
5. **Engine vs torch.** n7_engine is the from-scratch engine; k0* is torch. O18 is unresolved,
   so the two arms are **never pooled** and each number says which trained it.
6. **Train/test split leakage into the statistic.** The enrichment is computed on `W_E`
   alone, which sees no labels — but the split is seed-dependent, which is why step 0 of the
   *same seed* is the calibration control rather than a fresh random matrix.

## Analysis plan — decided now

`PYTHONPATH=. .venv/bin/python test_crt_null.py` (self-checked; added to `reproduce.sh`).
Sweep `we_traj` at n=119 for `results/n7_engine` seeds 0–2, all snapshots 0..40,000; report
enrichment and p under A, B, C at each. Then the grokked-vs-FAILED discrimination table
across k03/k04/k06. **Any analysis not listed here is exploratory and will be labelled so.**

## Outcome (filled in AFTER the run — never edit anything above)

**Run 2026-09-15**, `PYTHONPATH=. .venv/bin/python test_crt_null.py results/n7_engine`,
output `logs/c28_null.log`, 42 s, zero quota. Criteria implemented before the run; the
script's `--selfcheck` asserts the bin convention by planting a cosine of known frequency.

- **Result: the enrichment IS CRT-specific — and the statistic does NOT discriminate a
  grokked model from a failed one.** Both halves matter and they point opposite ways.

  | # | criterion | verdict | numbers |
  |---|---|---|---|
  | 1 | **PRIMARY** p_B < 0.01 at step 1,000 | **HELD 3/3** | p_B = 0.000 in every seed, enrichment 1.450 / 1.225 / 1.224 at test acc 0.054–0.075 |
  | 2 | p_B < 0.01 at the final snapshot | **HELD 3/3** | enrichment 3.448 / 3.195 / 3.152 |
  | 3 | ramp ρ > 0.8 over 0–20k | **HELD 3/3** | ρ = **1.000 / 0.991 / 0.999** |
  | 4 | null calibration — step 0 must NOT reject | **HELD 0/3** | p_B = 0.047 / 0.196 / 0.998 |
  | 5 | discrimination — pass rate on both arms | **FIRED — see below** | grokked 43/43, FAILED **2/2** |

  **Criterion 1 is the finding.** The CRT-dual set beats a null made of *other* same-size
  unions of arithmetic progressions, at every step from the first post-init snapshot. The
  enrichment is about the **factorisation of n**, not about subgroup shape. Null C (mean-|k|
  matched) agrees throughout, so it is not a low-frequency artefact either. **C8's instrument
  survives its hardest available test**, which was the larger stake.

  **Criterion 5 fired exactly as G2's did.** Every FAILED run at a well-posed modulus also
  rejects at p_B = 0.000. A model at **test acc 0.2236** shows significant CRT enrichment.
  So the permutation p is **not** evidence that a model has learned anything — for the third
  time in this project (C22/C23's control-mean, G2's G1, now this), a p-value that is
  load-bearing elsewhere passes on the failure arm.

  **What DOES separate is the magnitude, and the clean comparison is within one modulus:**

  | n = 63, identical predicted set (7 of 31 bins) | enrichment |
  |---|---|
  | s0, grokked (acc 0.9969) | **4.687** |
  | s1, FAILED (acc 0.2236) | 1.626 |
  | s2, FAILED (acc 0.2749) | 1.574 |

  Across all moduli the two arms do not overlap: **grokked 3.27–17.44** (44 runs, min 3.273)
  against **FAILED 1.52–2.67** (4 runs, max 2.671). Report the **enrichment**, never the p.

  **The caveat this forces onto C28, and it is a real one.** The early-training enrichment is
  **1.22–1.45 at step 1,000** — *below* the band a memorising model reaches at convergence
  (1.52–1.63 at n=63). So "CRT structure is imprinted early" survives as a statement about
  **significance**, and does **not** survive as evidence that what appears early is the
  grokking circuit forming. The honest claim is: a weak but unambiguously CRT-specific
  signal is present from the first thousand steps and **grows monotonically by ~2.5×
  through grokking** (ρ ≈ 1.000); its early magnitude alone does not distinguish it from
  structure a non-generalising model also acquires.

- **Criteria met:** 1 HELD · 2 HELD · 3 HELD · 4 HELD · 5 fired (reported, not thresholded,
  as specified). 6 HELD — 113/121/125 printed VACUOUS and excluded.

- **Deviations from this pre-registration, and why:**
  1. **Run classification uses the MEDIAN OF THE FINAL 10 SAMPLES, not the final row.**
     Not anticipated when this file was written, and found because the first run classified
     k06's `n119_s0` as **FAILED** on its final row (0.6798) — the censored transient that
     C27 was retracted for that same day (FINDINGS §3.5b). Window median 0.9987 ⇒ grokked.
     The script now prints a `[censoring]` line whenever the two disagree by > 0.05. This
     changes the FAILED arm from 5 runs to 4 and is a **defect fix, not a criteria change**;
     leaving it would have put a working model on the control side.
  2. **`p_B` is reported as `n/a` where null B is not constructible**, rather than counted.
     At n = 54 the predicted set is **14 of 27 bins (52 %)**, so a size-matched union of
     1–3 non-divisor progressions rarely exists. Not foreseen. The pre-registered draw
     procedure was **not** changed to force a fit; the runs are reported as not-testable and
     a `near-VACUOUS set` flag now prints wherever the predicted set exceeds 40 % of bins
     (n = 54 at 52 %, n = 98 at 51 %). **Consequence: the FAILED arm at a well-posed modulus
     is only 2 runs, both n = 63.** That is thin and criterion 5's conclusion is stated with
     that limit attached.
  3. Secondary moduli were read from k03/k04/k06 in the criterion-5 table as planned, and
     are **not** pooled with the PRIMARY n=119 engine sweep.
