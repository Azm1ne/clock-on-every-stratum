# Pre-registration: I1b — the internal character intervention at a composite modulus

**Date:** 2026-09-22 · **Script:**
`run_intervention.py` (+ `src/analysis/intervention.py`), **unchanged** · **Arm:** thesis

## Why this run exists

I1 (`PREREGISTER_i1_internal_intervention.md`, 2026-09-22) showed that deleting the key
multiplicative characters from `W_E` and running the **unmodified forward pass** returns the
network to chance at **n = 113**, on 3 of 3 seeds. Its Outcome scoped the result honestly:
§5's logit-space scoping still governs the other 23 moduli, and §7's new subsection is
n = 113 only.

n = 113 is **prime**. Every nonzero residue is a unit, so the "non-unit rows are passed
through untouched" clause of I1 covered exactly one row (0) plus the `=` token. The entire
point of this project is the **non-square-free, non-regular** case, and at a prime it does
not arise. An internal causal claim that holds only where the ring is a field is the weakest
possible version of this paper's thesis.

n = 121 = 11² is the smallest modulus in the primary set that carries the structure:

| | n = 113 | n = 121 |
|---|---|---|
| units | 112 (all nonzero) | **110** |
| non-units passed through untouched | 1 (row 0) | **11** (0 and ten nilpotents) |
| J-classes | 2 | **3** |
| non-regular J-class | none | **d = 11, size 10, all nilpotent** |
| chance on the unit grid | 1/112 = 0.00893 | **1/110 = 0.00909** |

So at n = 121 the intervention deletes the unit-stratum characters while leaving the network
**eleven untouched embedding rows** and a whole non-regular stratum to work with. That is a
materially harder test than n = 113, and it is the one the title claims.

## Hypothesis

The key multiplicative characters of the embedding are the representation the network
computes with **on the unit stratum of a non-square-free modulus**, not merely the basis in
which its output is sparse. Deleting them from `W_E` and running the unmodified forward pass
destroys the model on the unit sub-grid, and no equal-sized set of non-key characters does
comparable damage — **even though the non-unit rows of `W_E` are left intact.**

## Prediction

At n = 121, on grokked engine checkpoints whose `W_E` and `logits_all` reproduce the archived
`results/_archive_i1/WE_engine_n121_s{0,1,2}.npz` (md5s committed before this run):

- **excluded** (key characters deleted from `W_E`, conjugates closed) raises cross-entropy on
  the **unit sub-grid** (110 × 110 of 121 × 121 cells) by **at least 100×** over baseline.
- **restricted** (only the key characters and DC kept in `W_E`) leaves a working model:
  unit-grid accuracy **≥ 0.90**.
- the **excluded** loss is beyond the draws of a null that deletes a random equal-sized,
  conjugate-closed, non-key character set, at **p < 0.01**.

I explicitly do **not** predict:

- that the magnitudes match n = 113's (3.9e+06–2.8e+08). Importing a number from a different
  modulus would be the same error as importing one from a different experiment.
- that `restricted` improves on `baseline`. I1 saw it worse on 2 of 3 seeds and said in
  advance that it might be.
- anything about the **non-unit** cells. The character basis is not defined there, the loss
  is not reported there, and this run makes no claim about them. See "Known confounds".

## Success criteria — exact, inherited, and already implemented

Every threshold is **identical to I1's**, which inherited each one from a criterion fixed
earlier for another purpose. Nothing here is chosen for this run, and nothing is re-tuned
after n = 113: reusing a threshold across arms is what makes the two arms comparable, and
moving one now — in either direction — would be the C6/C7 ordering error with an extra step.

| # | criterion | threshold | inherited from | implemented in |
|---|---|---|---|---|
| I1 | necessity: `excluded / baseline` loss ratio | **≥ 100×** | Gate 2's **G2** | `run_intervention.py::verdict` |
| I2 | sufficiency: restricted unit-grid accuracy | **≥ 0.90** | **`FAIL_ACC`**, the C27 excursion floor (`932ef89`, 09-15) | `run_intervention.py::verdict` |
| I3 | specificity: permutation p of `excluded` against the random-set null | **p < 0.01** | **C22/C23** and Gate 2's **G1** | `run_intervention.py::verdict` |

`B = 10,000` draws, matching R1 and I1. The floor is `1/10001 = 9.999e-05`; a result at the
floor is reported as a **bound**, never as `p = 0.0000`.

**All three must hold, on at least 2 of the 3 seeds, for I1b to be recorded as met.**
A 1-of-3 result is reported as such and claims nothing.

## What would falsify this

- `excluded / baseline < 100×` at a modulus where it was 3.9e+06–2.8e+08 at n = 113. This is
  the outcome that would matter most: it would say the internal character claim **does not
  survive the ring structure this paper is about**, and §7's n = 113 subsection would have to
  stay n = 113 forever. It is a publishable negative result and would be written as one.
- `p >= 0.01` — a random character set does comparable damage, so the key set is not special
  on the unit stratum of a composite modulus.
- Restricted accuracy `< 0.90` **with** I1 passing — necessary but not sufficient. A real and
  reportable outcome, not a failed run; it is n = 119's logit-space signature appearing
  internally, and the eleven passed-through non-unit rows would be the first thing to suspect.

## Required controls

| claim shape | control | status |
|---|---|---|
| "component C is responsible" | ablate C **and** a random equal-sized set | I3's null, B = 10,000, conjugate-closed, non-key, same `\|K\|` |
| "the checkpoint is the published model" | retrained `W_E` and `logits_all` equal the archive | asserted; **the run aborts if not** |
| "the instrument works at this modulus" | `intervention.py::_selfcheck`, which already asserts non-unit pass-through and the keep/remove partition on unit rows | run before the arm |
| "the eleven non-unit rows really are passed through" | `_selfcheck` asserts non-unit pass-through **structurally**, both directions, on every call | already asserted — deliberately **not** re-measured per-run, because a second answer to a question the instrument already guarantees is how two answers to "which characters are the key set" nearly happened (C22/C23) |

## Known confounds

- **Set size.** Three claims have died here (C7b, C20, C19). The null draws exactly `|K|`
  characters, conjugate-closed the same way.
- **The unit sub-grid is not the whole task.** Baseline, restricted and excluded losses are
  all reported on the 110 × 110 unit cells. `internal_baseline_full` records the full-grid
  loss for context. **No criterion reads the full grid**, and no claim is made about the
  non-unit cells — the character basis does not reach them. Stated, not silently done.
- **Non-unit rows are passed through.** This is the design, and at n = 121 it is eleven rows
  rather than one. It makes the test **conservative**: the network keeps machinery the
  intervention does not touch. If `excluded` still destroys the model, the passed-through
  rows cannot be carrying the unit computation.
- **Chance is 1/110, not 1/112.** A unit times a unit is a unit, so the excluded model's
  accuracy floor moves with the modulus. Writing 1/112 here would misread a correct result.
- **Bin convention.** Settled in `_selfcheck` by planting a signal of known frequency and
  using `argmax`. Four bin bugs in this project have been lost to reasoning.
- **Prime-power vacuity does not apply.** n = 121 is vacuous for the **CRT-dual law** (C8:
  one CRT component, so the predicted set is every frequency). That is a statement about C8
  and **not** about this run, which tests character necessity inside the network and never
  invokes the CRT prediction. Recorded here so the exclusion is not mistakenly imported.
- **Seed.** Three seeds, reported separately; no pooling into a mean. Median + range + n, and
  **no `±` below n = 5** (C36).

## Analysis plan — decided now

1. Retrain n = 121 seeds 0/1/2 with `run_n7.py` at HEAD into `results/i1_internal/`, with
   `snapshot()` retaining all nine parameter matrices. The archived runs carry two SHAs
   (`f9d15b8` for seed 0, `cafbb36e` for seeds 1–2); `git diff --name-only f9d15b8 cafbb36e`
   over `src/`, `run_n7.py` and `scripts/run_n7_all.sh` is **empty**, so the sweep carried one
   training-code state. That state differs from HEAD in exactly the four files I1 checked and
   found inert (`run_n7.py`, `scripts/run_n7_all.sh`, `src/autograd/engine.py`,
   `src/tasks/algebra.py`) — a stamped `dtype` key, a `DTYPE` default equal to the old
   hard-coded `float64`, a NEP-50 `.astype` inert at float64, and `algebra.py` additions that
   no training file imports. I1's retrain confirmed this **empirically**: bit-identical at
   n = 113. Checked again here, recorded, and **asserted by the run itself** at step 2.
2. Assert each retrained `W_E` and `logits_all` equals its archive. The run aborts otherwise.
3. Key set from `unit_logit_grid` — the same function the published test uses. Not re-derived.
4. Intervention: baseline / restricted / excluded on `W_E`, full forward each time.
5. Null: B = 10,000 random equal-sized non-key sets, same forward.
6. Report median and range across seeds, with n. No `±` below n = 5.

**`run_intervention.py` and `src/analysis/intervention.py` are not modified for this arm.**
That is a criterion, not a convenience: the n = 113 and n = 121 numbers are comparable only
if one instrument produced both. Any change either file needs in order to run at n = 121 is a
**deviation**, to be recorded in the Outcome with the diff.

Anything not in this list is exploratory and will be labelled so, **including** any
key-overlap decomposition of the null (I1's was written after seeing its tail; if the tail
recurs here the decomposition is again exploratory, not promoted by having been done once).

## Outcome (filled in AFTER the run — never edit anything above)

**Run 2026-09-23. Retrain 3 × 5.92–5.97 h (launched 2026-09-22 23:39, finished 05:34–05:37),
measurement 3 × 128.0–128.2 min, both 3-way parallel under `systemd-run --user` with
`MemoryMax=6G`. All local CPU, zero Kaggle quota.** Artifacts
`results/i1_internal/I1_engine_n121_s{0,1,2}.npz`, stamped `266b77d`, **`git_dirty=False`**.

- **Result: I1b HELD on 3 of 3 seeds** (the criterion required 2 of 3).

| seed | baseline | restricted (acc) | excluded (acc) | excluded/baseline | `p_perm` |
|---|---|---|---|---|---|
| 0 | 1.7482e-06 | 3.4061e-07 (1.0000) | 5.0365e+01 (0.0000) | 2.881e+07 | 0.00010 (0 of 10,000) |
| 1 | 7.7458e-03 | 1.0560e-01 (0.9596) | 5.7783e+01 (0.0120) | 7.460e+03 | 0.00010 (0 of 10,000) |
| 2 | 1.6364e-03 | 2.2674e-03 (1.0000) | 1.8467e+01 (0.0000) | 1.128e+04 | 0.00030 (2 of 10,000) |

`excluded/baseline` median **1.128e+04**, range 7.460e+03–2.881e+07, n = 3. Median-and-range
with n, never `±`, per C36. Grok steps 10,700 / 10,300 / 5,000. Key sets `|K|` = 6 / 4 / 4:
s0 {2, 22, 40, 44, 48, 55}, s1 {3, 11, 18, 42}, s2 {22, 44, 48, 55}.

- **Criteria met: I1 ✅ · I2 ✅ · I3 ✅, all three on all three seeds.**
  - **I1** (≥ 100×, Gate 2's G2): passes by 2–5 orders.
  - **I2** (restricted accuracy ≥ 0.90, `FAIL_ACC`): 1.0000 / **0.9596** / 1.0000. Seed 1
    clears the bar by 0.06 — the smallest margin anywhere in this arm.
  - **I3** (`p < 0.01`, C22/C23): 0.00010 / 0.00010 / 0.00030. Seeds 0 and 1 sit at the design
    floor `1/10001`; **reported as the bound, never as `p = 0`** (R1).

**The positive control passed bit-identically**, as at n = 113: `max|ΔW_E| = 0.000e+00` on all
three seeds. The `3.8e-06` logit gap the runner prints is float32 *storage* of the archived
tensor against a float64 recomputation, not a model difference. These are the published
models, so this describes the runs C33's engine arm already measures.

**The prediction that mattered held at a non-square-free modulus.** §"What would falsify this"
named `excluded/baseline < 100×` as the outcome that would confine §7's internal claim to
n = 113 forever. It reads 7.46e+03 at worst. The eleven passed-through non-unit rows do not
rescue the model.

### ⚠️ Excluded accuracy is BELOW chance on two seeds — and no criterion tests it

Chance on the 110 × 110 unit sub-grid is 1/110 = 0.00909. The excluded model reads
**0.0000 / 0.0120 / 0.0000**. At n = 113 the same intervention read 0.0089 / 0.0089 / 0.0091 —
chance almost exactly, which is what "it removes the model" predicts. Here two of three seeds
land *below* chance, i.e. the excluded model is worse than guessing on the unit grid.

**This is descriptive and is not part of the verdict.** None of I1/I2/I3 reads accuracy against
chance, and promoting it now would be the C6/C7 ordering error. It is recorded because it is a
real difference between the two arms and because an anti-chance readout is the kind of thing
that has a mechanism behind it — most plausibly that the surviving non-key characters plus the
intact non-unit rows put mass on *specific* wrong residues rather than spreading it. **That is
a hypothesis, not a finding, and it needs its own pre-registration.**

### ⚠️ The null's upper tail reaches the effect again, on seed 2

| seed | excluded | control max | max / excluded |
|---|---|---|---|
| 0 | 50.3648 | 45.2470 | 89.8 % |
| 1 | 57.7832 | 49.6228 | 85.9 % |
| 2 | **18.4667** | **18.6887** | **101.2 %** |

Seed 2's control maximum **exceeds** its excluded loss, exactly as seed 2 did at n = 113.
`excluded / median(control)` reads 3.12e+06 / 8.84e+03 / 1.09e+04 and **hides it completely**.
Reported here because the ratio to a centre cannot show it.

The cause is the inherited pool, as at n = 113: `random_character_sets` draws from
`all_freq_labels` **including the key characters**, so a draw can partially perform the
intervention. **Both draws that beat `excluded` on seed 2 carry overlap 3 of `|K|` = 4**, and
every top-5 draw on every seed carries overlap 2–4. The pool makes `p` conservative; the
pre-registration chose it for comparability with C22/C23 and that choice stands.

**EXPLORATORY, not pre-registered** (`run_intervention.py --summary`), and exploratory *again*
rather than promoted by having been done once, exactly as §"Analysis plan" said in advance:

| overlap | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| **0** | 1.35e-06 (n=4825) | 5.92e-03 (n=7304) | 1.70e-03 (n=7297) |
| 1 | 6.36e+00 | 1.13e+01 | 5.47e+00 |
| 2 | 1.37e+01 | 2.70e+01 | 9.28e+00 |
| 3 | 2.28e+01 | 2.96e+01 | 1.63e+01 |
| 4 | 2.82e+01 | — | — |

**19,426 zero-overlap draws across three seeds and not one exceeds 1.71e-02**, against an
excluded loss of 18–58. A draw containing no key character does nothing: 0.77× / 0.76× / 1.04×
baseline. ⚠️ Two of those three are **below** baseline, where n = 113 read 1.04–1.11× — removing
a random non-key set very slightly *helps*. Small, descriptive, and not tested here.

### Deviations from this pre-registration, and why

1. **`run_intervention.py` and `src/analysis/intervention.py` are UNMODIFIED**, which was a
   criterion. Neither appears in the diff of `266b77d`; both were last touched by `a65a1e6`,
   the n = 113 arm. One instrument produced both arms, so the two are comparable.
2. **⚠️ `--summary` POOLS THE TWO ARMS, and its headline figure must not be quoted.** It globs
   `I1_*.npz` under `I1_OUT`, and I1b writes into the same `results/i1_internal/` as I1, so it
   printed `median 1.635e+07, range 7.460e+03–2.780e+08, **n = 6**` across n = 113 *and*
   n = 121 — precisely the cross-modulus import §"Prediction" forbids. The n = 121 figures in
   this Outcome were computed on the three n = 121 artifacts alone (`I1_OUT` pointed at a
   directory holding only those), and re-derived independently from the `.npz`. **The script
   is deliberately NOT patched here**: changing it after the measurement would break the
   criterion in item 1 for no gain. It needs an `n` filter, as a follow-up, before `--summary`
   is used for anything.
3. **`restricted` is worse than `baseline` on seeds 1 and 2** (13.6× and 1.39×) while better on
   seed 0 (0.19×). §"Prediction" said in advance this was not predicted; it is the same
   necessary-not-sufficient signature I1 recorded, and I2 is scored on accuracy, which holds.
4. **Magnitudes do not match n = 113's** (7.46e+03–2.88e+07 against 3.90e+06–2.78e+08, roughly
   two orders lower at the floor). §"Prediction" explicitly declined to predict that they
   would. No claim is made from the comparison.
