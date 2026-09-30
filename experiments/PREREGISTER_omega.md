# Pre-registration: ω(n) — does the band separation survive the confound that killed C7b, C19 and C20?

**Commit this file BEFORE computing any statistic named below.** **Date:** 2026-09-15 · **Script:** `analyze_omega.py`
(extended) · **Arm:** thesis · **Compute:** **zero** — re-analysis of `W_E` already on disk.

## ⚠️ Read this first: what this pre-registration can and cannot do

**The band separation was observed before this file was written.** `analyze_omega.py` ran on
2026-09-15 while testing O7 and printed ω = 1 → Gini_add **0.017–0.184** (6 moduli),
ω = 2 → **0.394–0.481** (9), ω = 3 → **0.578–0.589** (3), non-overlapping, with
ρ(ω, Gini_add) = **+0.843**. Those numbers are on the record and in `FINDINGS.md` §5.1c,
labelled exploratory.

**This is therefore a CONFIRMATORY test of an exploratory observation, on the same data. It
cannot convert ω into a pre-registered discovery and must never be written up as one.** What
it can do — and the only thing claimed for it — is **kill the specific confounds that have
already destroyed three claims in this project**: C7b (𝒥-class size), C19 (φ(n)), C20 (block
cell count). Every one died the same way: the grouping variable was structurally coupled to
a size, and the statistic read the size. ω(n) is collinear with zero-divisor density
(ρ = **+0.674**, already measured) and potentially with n itself.

**The statistics below have never been computed.** That is what makes committing this file
before running them meaningful, and it is the honest limit of the exercise.

## Hypothesis

Additive-basis sparsity in the learned embedding is carried by **ω(n)**, the count of
distinct prime factors — not by zero-divisor density (C9), not by cyclicity (O7's dead
premise), and not by the size of n.

## Why ω is the mechanistically expected variable, not a fished one

This is the part that distinguishes ω from a correlation found by sweeping variables.
**C8's CRT-dual law already implies it.** C8 (VERIFIED, 12 moduli, 3–5 seeds — the
best-supported claim in the project) says the key additive frequencies are the multiples of
n/q over the maximal prime powers q ‖ n: one family per maximal prime power. Then

- at ω = 1 the predicted set is **every frequency** — vacuous, nothing to concentrate on,
  so a prime power has no additive structure to be sparse in. This is *why* its clock must
  live in the multiplicative basis (C6);
- each further distinct prime adds a family, hence more concentration, hence higher
  Gini_add.

The prediction of monotone increase in ω therefore **precedes** the observation, in the
sense that it is a consequence of an already-verified claim. It does not rescue the ordering
of discovery, but it does mean the hypothesis is not post hoc in substance.

## Prediction

ω survives every control below; zdd's apparent effect is substantially mediated by ω.
C9 is **not** retracted either way — ρ(zdd, Gini_add) = +0.862 and zdd still predicts
*within* ω bands (+0.886 at ω = 1, +0.533 at ω = 2). The claim is about which variable is
primary, not about zdd being spurious.

## Success criteria — thresholds fixed now

Population: the **18 moduli** with grokked runs, per-modulus mean Gini_add over grokked runs
only, classified on the **median of the final 10 samples** (never the last row — C27).

| # | criterion | statistic | threshold |
|---|---|---|---|
| **W1** | **PRIMARY. ω separates at matched zero-divisor density.** The confound test that C7b, C19 and C20 all failed. | Over all modulus pairs (i, j) with \|zdd_i − zdd_j\| ≤ 0.05 **and** ω_i ≠ ω_j: the fraction where the higher-ω modulus has the higher Gini_add. Sign test. | **HELD** ≥ 0.80 **and** sign-test p < 0.05. **NOT HELD** ≤ 0.60. Between = **PARTIAL**, reported as such. Null "zdd explains it" predicts 0.50. If fewer than 6 such pairs exist, W1 is **NOT ASSESSABLE** and must be reported that way, not skipped |
| **W2** | ω survives partial correlation on zdd | Spearman ρ(ω, Gini_add \| zdd), via rank residuals of both on zdd | **HELD** ≥ +0.50. If zdd fully mediated ω this goes to ≈ 0. Raw is +0.843 |
| **W3** | zdd does **not** survive the reverse | ρ(zdd, Gini_add \| ω) | descriptive, and reported **whatever it shows**. A high value here means both variables carry signal and the paper must say so rather than crowning ω |
| **W4** | **SIZE CONTROL.** The bands are absent in a residue-axis random-orthogonal control | band ranges of `gini(rand_amplitude(W, n))` by ω | **HELD** iff the control bands **overlap** (no separation). ⚠️ The per-modulus control values (0.117–0.155) are already visible in the exploratory output, so this criterion **confirms something already seen** and is labelled so. It is included because it is the correct size control, not because it is a blind test |
| **W5** | **SIZE CONTROL, and this one IS blind.** n itself does not carry the effect | ρ(n, Gini_add) and ρ(n, ω) | **HELD** iff \|ρ(n, Gini_add)\| ≤ 0.40. Above that, n is a live confound and ω cannot be claimed without a further matched control |
| **W6** | the separation is not chance | permutation: shuffle ω labels across the 18 moduli 10,000×; statistic = **minimum gap between adjacent band ranges** (observed: 0.394 − 0.184 = 0.210) | **HELD** p < 0.01. This is the project's standing threshold (C8, C22, C23, C32) |

**Thresholds are copied from this project's existing practice** (p < 0.01 from C8/C22/C23;
the HELD/NOT-HELD/PARTIAL band structure from N9 and O18) so that none of them was tuned
for this question.

## What would falsify this

- **W1 ≤ 0.60** — ω does not separate at matched zdd, so the band structure is zdd wearing
  ω's clothes and ω must be dropped. This is the C19 outcome and it is a live possibility:
  **ρ(zdd, Gini_add) = +0.862 is already HIGHER than ρ(ω, Gini_add) = +0.843.** By raw
  correlation alone zdd is the better predictor, and the entire case for ω rests on the
  discreteness of the bands and on the matched pairs.
- **W5 > 0.40** — n carries it, and this is C19 again with a different size variable.
- **W4 showing bands in the control** — the effect is an artifact of spectrum length.

## Known confounds, named before scoring

- **ω is collinear with zdd** (ρ = +0.674, measured). W1/W2/W3 exist for exactly this.
- **ω may be collinear with n.** W5.
- **Unequal band sizes: 6 / 9 / 3 moduli.** The ω = 3 band has three members and its
  apparent tightness (0.578–0.589) is not a precision estimate. No claim rests on the
  width of that band.
- **Unequal run counts per modulus** (2 to 35). Per-modulus means, not per-run pooling, so
  n = 113's 35 runs cannot dominate. Stated because pooling would have let it.
- **Every modulus is measured in the torch/scout arm.** ω is *not* claimed for the engine
  arm, and O18 is open. Level shifts between implementations are irrelevant to a
  *between-modulus* ordering measured within one arm, but say so rather than assume it.
- **One prime.** ω = 1 contains five prime powers and a single prime (113), which is also
  the band's extreme low value (0.017). **If 113 alone drives the ω = 1 band being low, the
  result is about prime powers vs composites, not about ω.** Report the ω = 1 band with 113
  **excluded** alongside the full band; the remaining five read 0.125–0.184, still clear of
  ω = 2's 0.394 — this is stated now so it cannot be presented later as a robustness check
  that happened to work.

## Analysis plan — decided now

`PYTHONPATH=. .venv/bin/python analyze_omega.py --confirm`, added to `analyze_omega.py`
alongside its existing `_selfcheck`, printing W1–W6 with their verdicts. Registered in
`scripts/reproduce.sh`. Anything else is exploratory and labelled so.

## Outcome (filled in AFTER scoring — never edit anything above)

**Date scored:** 2026-09-15 · `analyze_omega.py --confirm` · log `logs/omega_confirm.log` ·
18 moduli, 115 grokked runs, per-modulus means · **zero compute.**

### Result: the confounds are dead. The crown is not awarded.

| # | statistic | observed | verdict |
|---|---|---|---|
| **W1** | ω at matched zdd (\|Δzdd\| ≤ 0.05) | **5 discordant pairs** — threshold was ≥ 6 | **NOT ASSESSABLE** |
| **W2** | ρ(ω, Gini_add \| zdd) | **+0.700** (raw +0.843) | **HELD** (≥ +0.50) |
| **W3** | ρ(zdd, Gini_add \| ω) | **+0.739** (raw +0.862) | descriptive — **and it is the higher of the two** |
| **W4** | random-orthogonal control bands | ω=1 0.117–0.148 · ω=2 0.125–0.155 · ω=3 0.128–0.147 — **fully overlapping** | **HELD** |
| **W5** | ρ(n, Gini_add) | **+0.015** (ρ(n, ω) = +0.282) | **HELD**, and this one was blind |
| **W6** | permutation, 10,000 shuffles of ω | observed min adjacent band gap **+0.097**, p = **1.0 × 10⁻⁴** (the floor at 10,000 draws) | **HELD** |

**Robustness (pre-registered, and it holds):** the ω = 1 band without the project's single
prime reads **0.125–0.184** across the five remaining prime powers, still clear of ω = 2's
0.394. The band is not an artifact of n = 113's extreme 0.017.

### What this establishes, and what it does not

**Establishes — the size confound that killed C7b, C19 and C20 is definitively absent here.**
W5 is the strongest number on the table and it was blind: ρ(n, Gini_add) = **+0.015**. The
magnitude of n carries essentially nothing. W4 adds that the residue-axis random-orthogonal
control shows **no band structure whatsoever** — three fully overlapping ranges inside
0.117–0.155 — so the separation is not a property of spectrum length. W6 puts the band
separation at the permutation floor.

**Does NOT establish — that ω displaces zero-divisor density.** The pre-registration's own
W3 clause fired: *"A high value here means both variables carry signal and the paper must
say so rather than crowning ω."* ρ(zdd, Gini_add | ω) = **+0.739** is *higher* than
ρ(ω, Gini_add | zdd) = **+0.700**. Each survives partialling on the other, at comparable
strength. **The honest claim is that ω(n) and zero-divisor density are two partially
independent real predictors of additive sparsity, and this data cannot rank them.** The
exploratory write-up's framing — that zdd is "partly a proxy for ω" — is not supported in
that direction; the mediation runs both ways at almost equal strength. `FINDINGS.md` §5.1c
must be amended accordingly.

**The PRIMARY criterion could not be scored, and the threshold stays where it was.** W1
found 5 zdd-matched discordant pairs against a pre-registered minimum of 6. That minimum was
fixed before the pair count was known, and moving it now to reach a verdict is precisely the
manoeuvre the pre-registration exists to prevent. **W1 is NOT ASSESSABLE and the paper must
say so.**

### Exploratory follow-up — reported because it is informative, labelled because it is not scored

Sweeping the matching tolerance (**exploratory, not pre-registered, and it does not change
the W1 verdict**):

| \|Δzdd\| ≤ | discordant pairs | favouring ω |
|---|---|---|
| 0.03 | 3 | **3/3** |
| **0.05** (pre-registered) | **5** | **5/5** |
| 0.07 | 11 | **11/11** |
| 0.10 | 18 | **18/18** |

**Every zdd-matched discordant pair favours ω, at every tolerance, without a single
exception.** The cleanest is n = 119 (ω = 2, zdd 0.193) → Gini_add **0.440** against
n = 125 (ω = 1, zdd 0.200) → **0.184**: near-identical zero-divisor density, 2.4× the
additive sparsity. This is suggestive and it is *not* a result — the pairs share moduli and
are not independent, so no p-value is defensible, and the scored test says NOT ASSESSABLE.
**Closing W1 properly needs more moduli, not a wider tolerance.** Two or three additions
chosen to fill the zdd axis at contrasting ω would do it, and they are cheap.

### Deviations from this pre-registration, and why

1. **None to the criteria or thresholds.** Nothing in §"Success criteria" was altered after
   the data existed. W1's threshold was not moved when it failed to be reachable.
2. **The sign test was specified without a sidedness.** It was implemented **two-sided and
   exact** (the conservative reading) in `binom_two_sided`. Moot — W1 was never scored.
3. **The pairs' non-independence was not named in §"Known confounds"** and should have been:
   18 moduli generate overlapping pairs, so a sign test over them overstates significance.
   `analyze_omega.py --confirm` now prints this warning next to the statistic, and the
   fraction rather than the p-value is treated as the number of record.
4. **W6's illustrative parenthetical was arithmetically wrong, and it is corrected here
   rather than above the line.** §"Success criteria" W6 reads *"minimum gap between adjacent
   band ranges (observed: 0.394 − 0.184 = 0.210)"*. The **statistic** is defined correctly —
   the minimum over adjacent pairs — but the number quoted is the ω1→ω2 gap, and the
   *minimum* is the ω2→ω3 gap, **0.578 − 0.481 = 0.097**. The threshold (p < 0.01) and the
   verdict (HELD, p = 1.0 × 10⁻⁴) are untouched; only the illustration was wrong. Left
   standing above as written, per the never-edit rule.
5. **A self-check in the first implementation was itself wrong** and the assertion caught
   it: it asserted ρ(x, y \| z) = 1 for x and y both *fully determined* by z, where the
   partial is **undefined** (zero-variance residual), not 1. Replaced with a
   random-construction check that tests both directions — spurious shared dependence
   vanishing, and a genuine link surviving.

---

## Amendment, 2026-09-15 (same day): W1 is now HELD, on 23 moduli

**Nothing above this line is edited.** The scoring above was correct for the **18** moduli
that existed when it was run. k09 added five (127, 131, 128, 123, 91) chosen to fill the zdd
axis at contrasting ω — `experiments/PREREGISTER_k09_primes_zdd.md`, itself committed before
those runs existed — and W1 was rescored on **23 moduli, 127 grokked runs**.

| # | 18 moduli (above) | **23 moduli (now)** | verdict |
|---|---|---|---|
| **W1** | 5 pairs → **NOT ASSESSABLE** | **10 pairs, 10/10 favour ω**, fraction **1.000**, exact two-sided sign test **p = 1.95 × 10⁻³** | **HELD** |
| W2 | +0.700 | **+0.627** | HELD (≥ +0.50) |
| W3 | +0.739 | **+0.735** | still the higher of the two |
| W4 | overlapping | overlapping (0.117–0.153 / 0.125–0.155 / 0.128–0.147) | HELD |
| W5 | +0.015 | **−0.185** | HELD (\|ρ\| ≤ 0.40) |
| W6 | gap +0.097, p = 1.0e−4 | gap **+0.082**, p = **1.0e−4** | HELD |

**The threshold was never moved — moduli were added instead.** That was the explicit
conclusion of the scoring above ("closing W1 needs more moduli, not a wider tolerance"), and
it is what was done.

**The ten pairs, every one favouring ω:** 49↔143 · 75↔128 · 75↔165 · 81↔123 · 91↔125 ·
98↔105 · 105↔128 · 119↔125 · 125↔143 · 128↔165.

⚠️ **The caveats above survive unchanged and matter more now, not less.** The pairs **share
moduli and are not independent**, so the sign-test p is optimistic and descriptive; **the
fraction is the statistic**. And **W3 still exceeds W2** (+0.735 vs +0.627): each variable
still survives partialling on the other, so **ω is the better predictor on the matched-pair
test but does not displace zero-divisor density.** The conclusion recorded above —
*"two partially independent real predictors"* — is unchanged.

**This remains a CONFIRMATORY test of an EXPLORATORY observation.** The bands were seen
before this file was written. What k09 adds is that the PRIMARY confound test is now
scorable and passes on data that did not exist when the criterion was set — which is
stronger than the original scoring, and still not a pre-registered discovery.

### The designed falsifier fired and ω survived it

k09's P1 put ω against zdd at **n = 128 = 2⁷**, a prime power (ω = 1) with **half its
residues zero divisors** (zdd = 0.500), where the two predicted **disjoint** intervals:
ω 0.017–0.184, zdd 0.434–0.589. **Observed: 0.261 — between them**, so P1 is **AMBIGUOUS**
and neither predictor was accurate.

But the band structure **survives**: n=128 extends ω = 1 to 0.015–0.261 and the ω = 1 / ω = 2
bands still do not overlap (0.261 vs 0.394). **§5.1c-bis is not retracted.** It is the
honest middle outcome — ω wins the matched-pair test 10/10 and loses the point prediction.
