# Pre-registration: R2 — is it the SAME clock on every stratum, or one clock each?

**Commit this file BEFORE launching the run.** Git's timestamp is the evidence that the
criteria predated the result.

**Date:** 2026-09-21
**Script:** `analyze_r2_sameness.py` (new) · **Arm:** thesis · **Cost:** local CPU, **zero Kaggle quota**

## Why this exists

The panel's load-bearing finding (Devil's Advocate, CRITICAL, sustained by the editor as
*"the one revision that cannot be done by rewriting"*):

> Every statistic is computed **inside** one stratum. "The same clock on every stratum" is
> inferred from "a clock here, and a clock there" — which is also what a model routing each
> stratum to a private sub-circuit would produce.

The paper already owns the right instrument on **one** stratum: `K_h = K_W` at 14 of 14
grokked runs (`06-results-clock.tex`, `\eqref{eq:keyagree}`) — two independently recovered
key sets, compared element for element. This lifts that comparison **between** strata.

## Hypothesis

**H.** A grokked model runs one mechanism over the whole table, indexed per stratum by that
stratum's own local group. Two strata that share a local group therefore share a key set.

**H0 (the alternative the panel named).** Each stratum is served by a private sub-circuit.
Two strata may then both be sparse and both be diagonal while selecting **different**
characters of the same group — which is what every statistic measured so far would
report as success.

## Prediction

Strata sharing a local group `G_m` have **equal** key sets, element for element, in the
large majority of pairs. Under H0 the sets agree only at the combinatorial chance rate.

## The measurement, specified before it runs

**Per stratum.** For each measurable stratum `(d, e)` of a run (`analyze_gate2.strata`,
`|G_m| ≥ 10` and `≥ 100` held-out cells), collapse the logit block onto its local group
(`analyze_gate2.collapse`, the existing code), take the `2r`-dimensional character transform,
and read the **diagonal** energies `E[κ, κ]` for `κ ≠ 0` — the same diagonal Gate 2 ablates.

**The detector is the project's existing one, not a new one.** `key_freqs_5x_median`
(Nguyen `2606.17399`'s: above 5× the median) applied to **amplitude** `√E`, DC dropped,
conjugates folded to one canonical representative per pair — the Discrete-Log-Clock protocol
`LAB_PROTOCOL.md` mandates, and the same detector `analyze_o4.py` uses for `K_W`.

**The bin convention is settled by planting, never by reasoning.** Three bin-convention bugs
in this project so far and reasoning has lost every one. `analyze_r2_sameness.py::_selfcheck`
plants a pure diagonal character of known `κ` into a synthetic block, at a cyclic **and** a
non-cyclic local group, and asserts the detector returns exactly `{κ}`. **If that assertion
fails, no number below means anything.**

**Pairs.** Within one run, all unordered pairs of measurable strata whose local groups are
identical (same `n/m` **and** same `orders` tuple).

**⚠️ TRANSPOSE PAIRS ARE EXCLUDED FROM THE PRIMARY STATISTIC AND REPORTED SEPARATELY.**
`(d, e)` and `(e, d)` are transposes of one commutative task, so their diagonal spectra
agree for a reason that has nothing to do with a shared circuit. Counting them would be a
size-confound in a new costume, which is the defect that killed C7b, C19 and C20. Structural
counts, computed from the algebra alone before any model was opened:

| n | measurable strata | same-`G_m` pairs | of which transposes | **primary pairs** |
|---|---|---|---|---|
| 113 | 1 | 0 | 0 | **0** (a field: one stratum, nothing to compare) |
| 119 | 4 | 3 | 1 | **2** |
| 120 | 6 | 4 | 2 | **2** |
| 121 | 3 | 1 | 1 | **0** |
| 125 | 3 | 1 | 1 | **0** |
| 165 | 15 | 34 | 4 | **30** |

So the test lives at **n = 165, 120 and 119**, and `n = 165`'s `G_11` — eight strata sharing
one local group — carries it. At 5 seeds per modulus that is **170 primary pairs** in k03.
**n = 121 and n = 125 contribute nothing**, and we say so rather than reporting their
transpose pairs as support.

## Success criteria — exact, and implemented as code before the run

| # | criterion | threshold | implemented in |
|---|---|---|---|
| **R2-1 PRIMARY** | pooled exact key-set equality over primary pairs, grokked runs | **≥ 0.80** | `analyze_r2_sameness.py` |
| **R2-2 PRIMARY** | exact combinatorial null for R2-1: P(≥ observed agreements \| independent uniform key sets of the observed sizes) | **< 0.01** | same |
| R2-3 | cell-shuffle control: same statistic with each block's cells permuted | **< 0.30** | same |
| R2-4 | failed-run control, where measurable strata exist at all | **< 0.50**, or reported NOT SCORABLE | same |
| R2-5 | transpose pairs, reported separately | no threshold — **descriptive only** | same |
| R2-6 | agreement is not an artefact of tiny key sets | median `\|K\|` reported per group; pairs with `\|K\| = 0` excluded and **counted** | same |

**R2-2 is why R2-1's threshold is not 1.0.** At `G_11` the folded non-DC bin count is 5, so
two independent 2-element sets coincide by chance with probability `1/C(5,2) = 0.10`.
A raw agreement rate is not evidence at small `|G_m|`; the null is.

**R2-3 and R2-4 exist because of `gate2_discriminate.py`'s lesson** — a statistic that is
load-bearing elsewhere is not automatically load-bearing here, and the pass rate on a
control that should fail is printed beside the pass rate on the data, every time.

## What would falsify this

R2-1 below 0.80, or R2-2 at `p ≥ 0.01`, falsifies H at the strata we can measure. **That
outcome is publishable and we commit to publishing it**: it would mean the paper's title
claim holds per stratum but not across strata, §8.4's scoping becomes the result rather
than a limitation, and the panel's Devil's Advocate was right. The project has retracted
five claims; this would be the sixth.

R2-3 or R2-4 **passing** (a control agreeing as well as the data) voids R2-1 regardless of
its value — it would mean the statistic measures block geometry, not a circuit.

## Required controls

| claim shape | control |
|---|---|
| "these two strata select the same characters" | the exact combinatorial null over equal-sized random sets (R2-2) |
| "…because one mechanism serves both" | cell-shuffle, which keeps every size and destroys the structure (R2-3) |
| "…in a model that learned the task" | a run that **failed** to generalise (R2-4), three-state rule, never a near-grok |

## Known confounds

1. **Transposes** — excluded above, and the reason is written down.
2. **Small local groups** — handled by R2-2 (the null), not by a threshold on the rate.
3. **Detector boundary noise.** `key_freqs_5x_median` is a hard threshold; a frequency at
   4.9× median in one stratum and 5.1× in another produces a spurious disagreement. This
   biases **against** H, which is the direction we want; we additionally report the Jaccard
   overlap as a descriptive companion so a near-miss is visible rather than invisible.
4. **Nested rather than equal local groups.** Strata whose groups are nested (`q' | q`) are
   **not** in the primary statistic: comparing them needs a canonical map between two
   different dual groups, and any such map is a modelling choice made after seeing the data.
   Reported as exploratory (R2-7) if computed at all.

## Analysis plan — decided now

1. `analyze_r2_sameness.py results/k03_grid_acts`, reusing `analyze_gate2`'s algebra
   (`strata`, `jclass_dlog`, `fibre_map`, `block`, `collapse`) rather than a second copy —
   a second copy of the block assembly is exactly the defect `ablation.unit_logit_grid` was
   extracted to prevent.
2. Self-check first, planted frequency, both group shapes. It refuses to run on real data
   if the planting fails.
3. Report R2-1..R2-6 with counts, never rates alone.
4. Both outcomes are pre-written into the paper: HELD strengthens §8.4 from a limitation to
   a measurement; NOT HELD replaces §8.4's "we did not measure this" with "we measured it
   and it does not hold across strata", and the title claim is scoped in §1.

Anything else found while doing this is **exploratory** and labelled so.

## Outcome (filled in AFTER the run — never edit anything above)

**Run 2026-09-21, `analyze_r2_sameness.py results/k03_grid_acts`, 25 grokked runs.**

- **Result: H HELD on the strata we can measure, and the panel's alternative is rejected.**

| # | criterion | threshold | observed | verdict |
|---|---|---|---|---|
| R2-1 | exact key-set equality, primary pairs | ≥ 0.80 | **158 / 160 = 0.9875** | **HELD** |
| R2-2 | Monte-Carlo null for R2-1 | < 0.01 | **1.0e-4** (the floor at B = 10,000) | **HELD** |
| R2-3 | cell-shuffle control | < 0.30 | **0 / 160 = 0.0000** | **OK** |
| R2-4 | failed-run control | < 0.50 | **2 / 2 = 1.00** | **FAILS — see below** |
| R2-5 | transpose pairs (descriptive) | — | 54 / 54 | reported, never support |
| R2-6 | empty key sets excluded | — | **0 of 160 excluded**; median \|K\| = 2 | clean |

Mean Jaccard on primary pairs **0.9982**. The two disagreements are both in `n = 165`
seed 1, which reads 26 / 28; the other four seeds are 28 / 28.

**⚠️ R2-4 FAILED AND THE REASON MATTERS MORE THAN THE NUMBER.** The failed-run control was
supposed to show the statistic does *not* pass on a model that never generalised. It read
2 / 2. But the control has only **two scorable pairs in the whole project**, and both come
from **one run at test accuracy 0.7502** (`k04` `n = 54` seed 0) — a model that has plainly
learned much of the task, which the three-state rule buckets with the 0.146 memorisers.
Every genuinely failed run either has **no primary pairs at all** (`n = 49`, `n = 63`,
`n = 125`: the algebra gives them none) or produces **no key set at all** (`n = 54` seed 1,
accuracy 0.2733: `K = {}` on all three strata). The control's own null reads **p = 1.0**.

**So R2-4 is underpowered, not passing, and the honest statement is that R2-1 has no
failed-run control.** This is `LAB_PROTOCOL.md`'s G2 lesson recurring — a statistic that is
load-bearing elsewhere is not automatically load-bearing here — and it goes in the paper
as a stated limitation, not as a footnote. What does discriminate, **post-hoc and labelled
as such**: a key set *exists* in 160 of 160 grokked primary strata and in 2 of 4 failed
ones, and the engine's failed `n = 125` seed 1 does not even agree with its own transpose
(0 / 1) where every grokked run agrees 54 / 54.

- **Criteria met:** R2-1, R2-2, R2-3, R2-6. R2-4 not met. R2-5 descriptive.

- **Deviations from this pre-registration, and why:**
  1. **The structural table above is wrong in one cell and the code is right.** It says
     `n = 165` gives 30 primary + 4 transpose; the algebra gives **28 + 6**. The by-hand
     count caught `G_11`'s four transposes and missed the one in `G_33` and the one in
     `G_55`. The total, 34 same-group pairs, was right, and the qualitative claim (n = 165
     carries the test) is unchanged. Corrected **here** rather than above, and asserted in
     `analyze_r2_sameness.py::_selfcheck` so a hand count cannot reintroduce it. The
     primary total is therefore **160**, not the 170 predicted.
  2. **Two self-check failures changed the implementation before any real data was read**,
     both recorded in the script: (a) a *perfectly* pure planted spectrum has median 0, so
     "above 5× the median" selects float noise — the planting now carries a broadband floor
     and the degenerate case is asserted separately; (b) **two empty key sets compare
     equal**, so an undetected clock would have scored as a perfect match. R2-6 already
     excluded empty pairs, and `equal` is now false for an empty set by construction.
  3. R2-7 (nested local groups) was **not computed**, as the pre-registration allowed.
