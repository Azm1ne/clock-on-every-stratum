# Pre-registration: R1 — the permutation null at B = 10,000, and multiplicity

**Commit this file BEFORE launching the run.** Git's timestamp is the evidence that the
criteria predated the result.

**Date:** 2026-09-21
**Scripts:** `analyze_n4.py`, `analyze_gate2.py` · **Arm:** thesis · **Cost:** local CPU, **zero Kaggle quota**

## Why this exists

The peer-review panel's C-5 (Methodology and EIC, `paper/review/EDITORIAL_DECISION.md`):
a 200-draw permutation null has a floor of `1/201 ≈ 0.005` and cannot express the
`p = 0.0000` the paper printed in eight places, and the ~300-test family carries no
multiplicity statement. `LAB_PROTOCOL.md` records the same defect independently: *"report the
bound, or raise B."* The last pass fixed the **display** (0.0000 → "0 of 200 draws,
p < 0.005"). This raises **B**.

## Hypothesis

This is a **precision** change, not a hypothesis about the world. Raising B cannot make a
key set more or less damaging; it can only resolve *how far* below the null each test sits
and expose any test whose 200-draw `p = 0` was luck.

## Prediction

1. Every test whose 200-draw `p` was `0.0` and whose excluded loss exceeds the control
   **maximum** by orders of magnitude stays at the floor: `b = 0`, `p̂ = 1/10001 ≈ 1.0e-4`.
2. **The marginal tests are where the number can move, and we predict movement.** At 200
   draws `n = 98`'s seed read `0.015` (3 draws) and `n = 49` read `0.010` (2 draws). At
   10,000 draws these become estimates with ~50× the resolution, and either may cross
   `0.01` in either direction. **§7.2's "27 of 29" is therefore at risk and we commit to
   reporting whatever it becomes**, including 26 or 28.
3. No test that was `p ≥ 0.05` becomes significant.

## The estimator, fixed now

`p̂ = (1 + b) / (B + 1)` with `b` the number of draws at least as damaging as the key set —
the Phipson–Smyth convention. It is never zero, which is the point: an unbiased Monte-Carlo
p-value cannot claim more resolution than its draw count. The paper reports `p̂`, and where
`b = 0` it also states the count in words ("0 of 10,000 draws").

## Success criteria — exact, and implemented as code before the run

| # | criterion | threshold | implemented in |
|---|---|---|---|
| R1-1 | every previously-reported `p = 0.0` test has `b = 0` at B = 10,000 | ≥ 95% of them | `analyze_n4.py`, `analyze_gate2.py` |
| R1-2 | the estimator is `(1+b)/(B+1)` everywhere, never `b/B` | exact | both scripts + `test_paper_numbers.py` |
| R1-3 | no number in the paper quotes a p below the achievable floor | `p̂ ≥ 1/(B+1)` | `test_paper_numbers.py` |
| R1-4 | the re-run reproduces the B = 200 result where both are defined | sign and order of magnitude of `excluded`, `restricted`, `baseline` unchanged to 1e-9 | diff against the committed artifacts |

**R1-4 is the real check.** `baseline`, `restricted` and `excluded` do not depend on B at
all. If any of them moves, the change broke something and the p-values are worthless.

## Multiplicity — the family, counted now, before any p is read

Counted from the committed artifacts, **320 permutation tests**:

| artifact | tests |
|---|---|
| `results/gate2/k03_grid_acts.json` (measurable strata) | 157 |
| `results/gate2/n7_engine.json` | 26 |
| `results/k03_grid_acts_n4_ablation.npz` | 31 |
| `results/k04_extended_n4_ablation.npz` | 29 |
| `results/k08_init_n4_ablation.npz` | 24 |
| `results/k05_lowdata_n4_ablation.npz` | 15 |
| `results/k09_primes_zdd_n4_ablation.npz` | 13 |
| `results/n7_engine_n4_ablation.npz` | 11 |
| `results/k02_grid_n4_ablation.npz` | 7 |
| `results/k07_arma_seeds_n4_ablation.npz` | 5 |
| `results/n7_engine_preleakfix_n4_ablation.npz` (retired arm, counted anyway) | 2 |
| **total** | **320** |

At `B = 10,000` the floor is `p̂ = 1.0e-4`, and we commit **now** to reporting all three of
these rather than the flattering one:

- **Benjamini–Hochberg at α = 0.01:** every test at the floor is adjusted to `1.0e-4`. Clears.
- **Bonferroni at α = 0.05:** per-test level `1.56e-4 > 1.0e-4`. Clears.
- **Bonferroni at α = 0.01:** per-test level `3.13e-5 < 1.0e-4`. **Does not clear, and
  cannot at this B** — it would need `B ≥ 32,000`. We state this rather than omitting the
  comparison that fails.

**Why B = 10,000 and not 32,000.** Measured cost after hoisting the constant forward FFT
out of the null loop (`fourier_spectrum`, bit-identical, asserted in `test_ablation.py`):
≈ 56 ms/draw on the heaviest stratum run and 35–60 ms/draw on the single-modulus path, so
B = 10,000 is ≈ 10 h of sequential local CPU and B = 32,000 is ≈ 32 h. 10,000 is also the
number the reviewer asked for. **The cost, not the science, is the reason** — recorded here
so the choice is not later read as a statistical judgement.

## What would falsify this

- Any of `baseline` / `restricted` / `excluded` moving (R1-4) — that is a code defect, not
  a finding, and the run is void.
- A test previously at `p = 0.0` reading `b > 0` at 10,000 draws is **not** a falsification:
  it is the resolution the old B could not deliver, and it gets reported.

## Required controls

Unchanged from the measurements being re-run: the null is random conjugate-closed character
sets of **exactly** the key set's bin count, drawn from all non-DC bins. Raising B changes
only how many are drawn. The RNG seed stays 0 so the first 200 draws of the new null are the
old null, which makes R1-4 checkable draw-for-draw.

## Known confounds

**The one that matters: re-running is an opportunity to silently change something else.**
Guard: the scripts are re-run with `N_CONTROL` as the *only* difference, and R1-4 diffs
every B-independent quantity against the committed artifacts. The gate-2 artifact has
already been shown to reproduce bit-for-bit at a different SHA seven days later
(`results/gate2/k03_grid_acts.json`, 2026-09-21), so any diff beyond provenance is real.

## Analysis plan — decided now

1. `N_CONTROL = 10000` in `analyze_n4.py` and `analyze_gate2.py`.
2. Re-run all four families, each writing to its existing artifact path.
3. `test_paper_numbers.py` gains: the estimator form, the floor, the family count, and the
   three multiplicity verdicts above.
4. Every `p` in the paper is re-derived from the new artifacts; the abstract's `p < 0.005`
   becomes the new floor statement.

Anything else found while doing this is **exploratory** and labelled so.

## Outcome (filled in AFTER the run — never edit anything above)

**STATUS 2026-09-21, second update: SIX OF NINE LANDED.** Nine sweeps launched under
`systemd-run --user` (`MemoryMax=1400M`, `OMP_NUM_THREADS=2`). Landed, all
`Result=success`: **k05, k07, k02, n7, k09, k04**. Still running: `gate2-k03` (157),
`n4-k03` (31), `n4-k08` (24). The collection procedure is in `STATE.md` §7.

**R1-4, the real check — PASSES on every family that has landed and been scored.**
`test_paper_numbers.py` recomputes the B-independent quantities from the new artifacts:
k09's five median/range rows (including n=128's `2.77`–`476`) and k04's `29` analysable runs
over `11` moduli are **identical**. Nothing moved that must not move.

**R1-1 / R1-2 hold:** every landed artifact reads `n_draws = 10000`, and the minimum `p̂` is
exactly `1/10001 = 9.999e-05`. The estimator can no longer print `0.0`.

**Prediction 2 was right, and the number moved.** §7.2's count is now **28 of 29**, not 27.
`n = 49`'s single analysable run crossed the threshold **downward** — `p = 0.010` (two draws
of 200, landing exactly on it) → `p̂ = 0.0050` (49 of 10,000). `n = 98` did not move: `0.015`
from 3 of 200, `0.0151` from 150 of 10,000. `n = 63` reads `0.0002` where the 200-draw null
could only say "0 of 200". **What the re-run actually cost is a sentence, not a count** —
§7.2 read `n = 49`'s old `p = 0.010` as the statistic "correctly" reporting the absence of a
circuit in a run that did not grok. It was not doing that, and §7.2 now says so, tied to the
same limit Gate 2 measures in G1.

- Result: **R1-1, R1-2, R1-4 met on six of nine families; R1-3 pending the paper pass.**
- Criteria met: *three of four, on the families scored so far.*
---

## FINAL, 2026-09-22 — all nine families plus the two control arms

**Everything launched has landed `Result=success`.** The three that were still running on
09-21 (`gate2-k03`, `n4-k03`, `n4-k08`) finished, and the two arms this file recorded as
overlooked (deviations 3 and 4) were run afterwards, one at a time, each committed before
the next started: `analyze_gate2.py results/n7_engine` (`e8dbd97`, 47 min),
`... results/k04_extended --controls` (`cbe13f3`, 2 min 10 s),
`... results/n7_engine --controls` (`bbc9b85`).

### Verdict, criterion by criterion

| # | threshold | measured | verdict |
|---|---|---|---|
| R1-1 | ≥ 95% of previously-`p = 0.0` tests have `b = 0` | **171 of 200 = 85.5%** | **NOT MET** |
| R1-2 | estimator is `(1+b)/(B+1)`, never `b/B` | `n_draws = 10000` everywhere, min `p̂` = `1/10001` | MET |
| R1-3 | no paper number quotes a `p` below `1/(B+1)` | asserted in `test_paper_numbers.py` | MET |
| R1-4 | `baseline`/`restricted`/`excluded` unmoved to 1e-9 | **606 values across 4 tracked artifacts, 0 moved** | MET |

Per artifact for R1-1 (`git show <commit>~1:<path>` against the working tree, counting
tests whose old `p_perm` was exactly `0.0` and whose new `n_ge` is `0`):

    k03_grid_acts            157 -> 133   84.7%
    n7_engine                 24 ->  19   79.2%
    k04_extended_controls     16 ->  16  100.0%
    n7_engine_controls         3 ->   3  100.0%
    TOTAL                    200 -> 171   85.5%

### ⚠️ R1-1 IS NOT MET, AND THE EARLIER ENTRY ABOVE CLAIMED IT WAS

The 09-21 update reads *"R1-1 / R1-2 hold: every landed artifact reads `n_draws = 10000`,
and the minimum `p̂` is exactly `1/10001`."* **That sentence is evidence for R1-2 and for
nothing else.** R1-1 is a statement about how many previously-zero tests *stay* at the
floor, and it was never computed until now. Left standing, it would have shipped a
pre-registered criterion as met on the strength of a different criterion's evidence. The
earlier text is kept above rather than edited, because the mistake is part of the record.

**This file contradicts itself on what 85.5% means, and both halves are quoted here rather
than the convenient one.** The criteria table sets R1-1 at ≥ 95%. The falsification section
says, in advance: *"A test previously at `p = 0.0` reading `b > 0` at 10,000 draws is **not**
a falsification: it is the resolution the old B could not deliver, and it gets reported."*
Prediction 1 is narrower still — it promises the floor only for tests *"whose excluded loss
exceeds the control maximum by orders of magnitude"*, a qualifier the criteria table drops.

**Our reading: the 95% was a mis-calibrated guess about magnitude, not a hypothesis about
the world, and the document's own reasoning anticipated the direction.** 29 of 200 tests
resolved off the floor; every one of them sits at `p̂ ≤ 0.0023`, far below anything 200 draws
could express, and none crossed a decision threshold except `n = 49`, which crossed
**downward** (prediction 2, called in advance). So nothing is retracted and no verdict
flips. **But R1-1 as written is not met, and we report it as not met** rather than reading
the falsification clause as retroactive permission to move the bar. A criterion that can be
reinterpreted after seeing the number is not a criterion — that is the C6/C7 error this
project made once.

**What it cost, plainly:** three absolute phrasings. Gate 2's "0 of 200 in all 157" became
"0 of 10,000 in 133 of 157"; k03's "not one of 200, any run" became "26 of 31"; and §7.2's
"the statistic correctly says so" was retired. k08 and k09 survived intact; k04 moved in our
favour, 27 → 28 of 29.

### Deviations, final state

- **Deviation 1** (R1-4 not runnable as written for seven families, the `.npz` being
  gitignored) — stands as recorded. For the **four tracked** artifacts R1-4 ran as specified,
  and `scripts/r1_diff.py` now implements it so the next re-run is one command.
- **Deviation 2** (the engine arm run separately) — done as planned, committed before the
  control arms started.
- **Deviation 3** (the two control arms overlooked, still at B = 200 with `p_perm = 0.0`) —
  **RESOLVED.** Both re-run and committed. No tracked artifact in the project now prints
  `p_perm = 0.0`.
- **Deviation 4** (control-arm membership wrong, independently of B) — **RESOLVED, and it
  cost more than the count.** `k04_extended/WE_B_thesis_n49_s2` joins the arm, "19 failed
  measurements" becomes **20**, and because its one measurable stratum passes G0, G1, G2, G4
  *and* G5, **four of the six F9 rows move**: G0 `0/19 → 1/20`, G1 `19/19 → 20/20`,
  G2 `0/19 → 1/20`, G5 `0/13 → 1/14`. *"G0, G2 and G5 separate completely"* is retired.
  ⚠️ That run clears `FAIL_ACC = 0.90` by **0.009** on a *rising* window. **Researcher
  decision 2026-09-22: the threshold stays at 0.90**, defended by ordering rather than by
  margin — 0.90 is the C27 excursion floor (`932ef89`, 09-15), centralised `37dba23` (09-21),
  and this arm was not scored until `cbe13f3` (09-22).
- **Deviation 5** (FFT hoisted out of the permutation loop) — stands; `test_ablation.py`
  asserts bit-identity.
- **Deviation 6, NEW.** R1-4 on `results/gate2/n7_engine.json` is **not** a pure B diff: its
  committed version is from 2026-09-14, so the comparison spans a schema addition
  (`+n_draws`, `+n_ge`, from `e82ca11`) and a defect fix. Three `split_logged` values moved,
  `1.000000 → 0.999596 / 0.996570 / 0.997756`, because the old artifact read `hist[-1][3]` —
  **TRAIN accuracy on the engine, TEST accuracy in the kernels**. `check_split` was therefore
  comparing recomputed test against logged train and passing only because both sat inside its
  0.01 tolerance: a silent false pass, fixed by `43df124`. `split_recomputed` is unmoved in
  all three, and none of R1-4's named quantities moved. **Recorded because "R1-4 failed" and
  "R1-4 failed *for a reason that is not B*" are different findings.**

- Result: **R1-2, R1-3, R1-4 MET. R1-1 NOT MET (85.5% against ≥ 95%), reported as such.**
- Criteria met: **three of four.** The run is **not** void — R1-4, the check this file calls
  "the real check", passed on every value it names.

- **Deviations from this pre-registration, recorded as they arose:**
  1. **R1-4 cannot be run as written for seven of the nine families.** This file says to diff
     the B-independent quantities "against the committed artifacts". The
     `results/*_n4_ablation.npz` artifacts are **gitignored**, so no committed B = 200 version
     exists and the re-run overwrote them in place. R1-4 therefore rests on
     `test_paper_numbers.py`, which recomputes 13 of those quantities from the `.npz` and
     asserts the published values — a weaker check for coverage but a stronger one per number,
     since it compares against the paper rather than against a previous run of the same code.
     **The two `results/gate2/*.json` artifacts are tracked**, so for them R1-4 is a literal
     `git diff` and is the check as specified. *A copy should have been taken before launch;
     it was not, and this is the cost.*
  2. **`analyze_gate2.py results/n7_engine` (26 of the 320 tests) was not launched with the
     others.** Its output path is tracked, and two concurrent writers of tracked artifacts
     make the second stamp `git_dirty=True` against the first's uncommitted write. It runs
     after the k03 one lands and is committed.
  3. **⚠️ THE TWO GATE-2 CONTROL ARTIFACTS WERE OVERLOOKED AND ARE STILL AT B = 200.**
     `results/gate2/k04_extended_controls.json` (16 measurable) and
     `results/gate2/n7_engine_controls.json` (3) are **not in the family table above** and no
     sweep was launched for them, so after R1 the project still holds two tracked artifacts
     whose `p_perm` reaches **`0.0`** — the exact display this pre-registration exists to
     retire. They are not a small corner: they are the **19 failed stratum-measurements** the
     paper cites in §8 and draws in F9, and §7.2 now cross-references them. `N_CONTROL` is
     already `10000` in `analyze_gate2.py`, so the fix is two commands and no code change:
     `analyze_gate2.py results/k04_extended --controls` and the same for `results/n7_engine`.
     **They must run AFTER `r1-gate2-k03` lands and is committed** — both write tracked
     artifacts, and deviation 2 is exactly the concurrency trap that applies.
  4. **AND THE CONTROL ARM'S MEMBERSHIP IS WRONG, INDEPENDENTLY OF B.** `--controls` selects
     runs by `run_state`, which reads `hist[-1]` — the last row — the defect `LAB_PROTOCOL.md` has
     forbidden since the C27 retraction and which `src/analysis/runs.py` now fixes in one
     place. Scored both ways over 162 runs, 4 change state; `LAB_PROTOCOL.md` records that **none
     of the four is in k03**, which is true and covers the *primary* arm — but the **control**
     arm is built from **k04**, and `k04_extended/WE_B_thesis_n49_s2` is one of the four:
     last row `0.9631` (near-grok, excluded) → window median `0.8908` (**FAILED**, included).
     So the corrected rule **adds a run to the control arm**, and "19 failed measurements"
     becomes **20**. Every count in F9 moves with it.
     **This is the right direction and it biases against our own finding**: it is one more
     failed measurement for G1 to pass on. It is also the same run §7.2 now reports as a
     non-grokked model whose key set survives the ablation test — the two are one phenomenon.
     Recorded here rather than acted on, because `analyze_gate2.py` is mid-sweep.
  5. **The forward FFT was hoisted out of the permutation loop** (`fourier_spectrum`) to make
     B = 10,000 affordable — 1.9×. Not a deviation from the statistics: `test_ablation.py`
     asserts the ablated grid is **bit-identical**, not merely close, so R1-4 covers it.
