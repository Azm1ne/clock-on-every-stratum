# Pre-registration: R3 — does the network-free indicator select exactly `P(n)`?

**Commit this file BEFORE launching the run.** Git's timestamp is the evidence that the
criteria predated the result.

**Date:** 2026-09-21
**Script:** `test_crt_law.py::indicator_setequality` (new) · **Arm:** thesis · **Cost:** seconds, **zero Kaggle quota**

## Why this exists

The panel's C-17 (Domain, recommended): `09-results-crt.tex` asserts the predicted set

```
P(n) = { multiples of n/q, one family per MAXIMAL PRIME POWER q || n, up to n/2 }
```

and supports it on trained models, while the paragraph that claims the mechanism *needs no
network* reports a **Gini** of the `J`-class indicators (`0.829 / 0.829 / 0.683 / 0.018`)
— a sparsity number, which is not the claim. The claim is a **set**. The reviewer also
notes that `1_{J_d}` is an inclusion–exclusion difference whose additive transform is a
Ramanujan sum with **support strictly larger than `P(n)`**, and that the naive
magnitude argument would favour the *prime* set `{n/p}` that the model data reject.

So the network-free object can be asked the same question the network is asked, and it is
the cheapest discriminating test in the paper.

## Hypothesis

The `P(n)` law is a property of the **stratification**, not of the network: the `J`-class
indicators select the same key frequencies a trained model does.

## Prediction, committed with both halves

1. **Support is strictly larger than `P(n)`.** We predict the reviewer is right and the
   raw non-zero support of the indicator transform exceeds `P(n)` at every composite `n`
   tested. **A test of support equality would fail, and would deserve to.**
2. **The KEY set equals `P(n)`.** With the project's existing detector —
   `key_freqs_5x_median` on additive amplitude, the same detector used for `K_W` — the key
   frequencies of the indicator family equal `P(n)` exactly, with no misses and no extras,
   at `n = 165` and `n = 120`.
3. **The prime variant is rejected by the same test.** Replacing each maximal prime power
   `q` by its prime `p` gives a different set at `n = 120` (`{24, 40, 48, 60}` against
   `{15, 24, 30, 40, 45, 48, 60}`), and the indicator selects the **`q`** set.
4. **`n = 113` is vacuous and must be excluded, not counted.** One `J`-class besides `{0}`,
   `n/q = 1`, so `P(113)` is every bin. A prime power is never support for this law
   (`LAB_PROTOCOL.md`).

## Success criteria — exact, and implemented as code before the run

| # | criterion | threshold | implemented in |
|---|---|---|---|
| R3-1 | key set of the indicator family `=` `P(n)` at `n = 165` and `n = 120` | exact set equality | `test_crt_law.py::indicator_setequality` |
| R3-2 | the prime variant `{multiples of n/p}` is **not** selected where it differs | set inequality at `n = 120` | same |
| R3-3 | raw support `⊋ P(n)` (the honest half) | strict superset at every composite tested | same |
| R3-4 | vacuous moduli excluded by rule, not by hand | `ω(n) = 1 ⇒ skipped and named` | same |
| R3-5 | positive control on a published number before any new number is believed | enrichment at `n = 165` reads `17.44` | already in `test_crt_law.py` |

**R3-5 is not optional.** The fourth bin-convention bug in this project was caught only by
a positive control: `predicted()` returns **frequencies** and `freq_energy()` returns bins
at position `frequency − 1`, and passing them unshifted read *depletion* at `p ≈ 1.0` on
the best-supported claim in the project. Any new spectral statistic runs on a modulus whose
answer is already published **first**.

## What would falsify this

R3-1 failing at either modulus. That outcome is reportable: it would mean the `P(n)` law is
a property of trained networks and **not** of the stratification alone, which would weaken
§9's "needs no network" sentence to a claim about the model — and that sentence would be
rewritten, not deleted.

## Required controls

| claim shape | control |
|---|---|
| "the indicator selects `P(n)`" | the prime-set variant, which differs at `n = 120` (R3-2) |
| "…and this is structural" | a field, `n = 113`, where the prediction is vacuous by construction (R3-4) |
| "…measured with a sound instrument" | the published `n = 165` enrichment, re-derived (R3-5) |

## Known confounds

1. **Bin convention.** Three families of index in this project (`energy` keeps DC,
   `mult_amplitude` drops it, `freq_energy` is frequency − 1). The script settles it by
   planting a pure frequency and asserting where it lands, never by reasoning.
2. **Which indicators.** The family is every `J`-class of `n` except `J_n = {0}`, each
   mean-centred, exactly as `analyze_scout.py` builds it for the Gini this replaces —
   so the object under test is the one the paper already reports on, not a new one.
3. **Detector sensitivity.** `5×`-median is a hard threshold. The companion number
   reported beside every verdict is the ratio of the smallest selected to the largest
   rejected amplitude, so a near-miss is visible rather than invisible.

## Analysis plan — decided now

Implemented as a function in `test_crt_law.py` — the file that already owns `predicted()`
and the bin-shift convention — and **not** as a second copy elsewhere. `09-results-crt.tex`
replaces the Gini sentence with the set-equality result and keeps the Gini as a
parenthetical. `test_paper_numbers.py` gains every number the new sentence asserts.

## Outcome (filled in AFTER the run — never edit anything above)

**Run 2026-09-21, `test_crt_law.py::indicator_setequality`, ten composite moduli.**

- **Result: R3-1 is NOT HELD, and the failure is more informative than the prediction was.**

| # | criterion | observed | verdict |
|---|---|---|---|
| R3-1 | key set of the indicators `=` `P(n)` at 165 and 120 | `{33, 55, 66}` vs `P(165)`'s 8; `{20, 40, 60}` vs `P(120)`'s 7 | **NOT HELD** |
| R3-2 | the prime variant is not selected | at `n = 120` the indicator picks `{20, 40, 60}`, which is neither `P(n)` nor `{24, 40, 48, 60}` | **HELD** |
| R3-3 | raw support is a strict superset of `P(n)` | 82 of 82 frequencies at `n = 165` against `|P| = 8` | **HELD** — the reviewer was right |
| R3-4 | vacuous moduli excluded by rule | `n = 113` and `n = 121` skipped, `ω(n) = 1` | **HELD** |
| R3-5 | positive control before any new number is believed | **17.44 at n = 165 and 3.27 at n = 119, exactly the published values** | **HELD** |

**What the data say.** The indicators are strongly *enriched* on `P(n)` — permutation
enrichment 4.87 at `n = 165`, 3.32 at 120, 7.15 at 119, all at `p < 1/20{,}000` — but they
do not *select* it. Their key set equals `P(n)` at only 2 of 10 moduli (99, 143). At
`n = 120` the indicator's top frequency `20 = 120/6` is the dual of `6`, a divisor that is
**not** a maximal prime power, so it lies outside `P(n)` and outside the prime variant.

**And that is the result worth having, because the contrast is with the network.** Measured
with one instrument, one bin convention, in one run:

| n | `P(n)` | network `K` (`W_E`) | indicator `K` |
|---|---|---|---|
| 165 | 8 frequencies | **all 8, no extras** | 3 of them |
| 120 | 7 frequencies | **all 7, no extras** | `{20, 40, 60}`, one of which is not in `P(n)` |

The stratification supplies a **larger candidate family** — the duals of *every* divisor,
42 frequencies at `n = 165` — and the indicator's key set is a subset of that family at all
ten moduli. **The trained network selects the maximal-prime-power subfamily `P(n)` exactly.**
So `P(n)` is not a property of the stratification alone; picking `P(n)` out of the divisor
duals is something the network does.

- **Criteria met:** R3-2, R3-3, R3-4, R3-5. R3-1 not met, as reported above.

- **Deviations from this pre-registration, and why:**
  1. **§9's sentence is rewritten, not deleted, exactly as this file committed to.** "We
     confirm the mechanism on raw `J`-class indicators with no network involved" overstates
     what the indicators do; it becomes the enrichment result plus the network contrast.
  2. **The enrichment statistic on indicators is exploratory** — this file pre-registered a
     set-equality test, and the enrichment was computed after it failed. It is labelled as
     such wherever it appears. What is *not* post-hoc is the contrast table above: both
     columns use the detector and bin convention fixed before the run.
  3. **A documentation ambiguity surfaced and is resolved, not a defect:** `LAB_PROTOCOL.md`
     carries two enrichment values for `n = 165`, 14.08 and 17.44. Both are correct and they
     are different artifacts — the k01 scout checkpoint reads 14.08 (and 3.30 at `n = 119`),
     the k03 runs read 17.44 (and 3.27). `test_crt_law.main()` reads the scout, and now says
     so in its own output.
