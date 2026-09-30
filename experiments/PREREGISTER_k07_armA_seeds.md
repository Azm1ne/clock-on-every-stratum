# Pre-registration: k07 — the replication arm at 5 seeds

**Commit this file BEFORE launching.** Git's timestamp is the evidence the criteria
predated the result.

**Date:** 2026-09-11 · **Kernel:** `kernels/k07_arma_seeds`
**Arm:** **replication** (units only) · **Runs:** 5 seeds · **Account:** `account-b`

## Why

**C4 is the project's only complete replication of 2606.17399, and it is ONE SEED.** So is
**C21**, the first circuit-level (not embedding-only) readout of the multiplication model.
The replication arm is the *only* arm whose numbers may be compared to their published
ones (protocol invariant 1), so a single seed is the weakest possible footing for the
comparison the whole thesis leans on.

This is not a new experiment. It is k02's Arm A with seeds 1–4 added and the activations
saved that k02 did not save.

## Hypothesis

C4 and C21 are properties of the trained circuit, not of seed 0.

## Predictions

### S1 — the arm groks reliably

**Prediction:** all **5/5** seeds reach test acc > 0.99. k02's Arm A seed 0 grokked at step
6,400 and k02 Arm B's n=113 grokked 5/5.

**Falsified if** fewer than 4 of 5 grok.

### S2 — C4's replication numbers hold at 5 seeds

Published (2606.17399): Gini_mult 0.579, PR_mult 4.1, Gini_add 0.071, PR_add 52.7,
**4** key frequencies. Ours at seed 0: 0.543 / 4.31 / 0.060 / 53.31 / 4.

**Prediction, on the 5-seed means:** |Gini_mult − 0.579| < **0.10** (the same threshold
C6 and P4 use); **PR_mult within 1.0 of 4.1** (seed 0 is 0.21 away); key-frequency count
is exactly **4 in ≥4 of 5 seeds**.

**Falsified if** any of the three misses. Frequency *values* are seed-dependent and are not
predicted — and, per C25, are only defined up to multiplication by a unit mod φ.

### S3 — C21 holds at 5 seeds

> **AMENDED 2026-09-11, BEFORE THE RUN LANDED — and why.** As first written, S3 predicted
> the top-8 components would be "**100% `a+b`** (zero `a−b` among the top 8)", transcribed
> from C21's wording in `FINDINGS.md`. **C21's wording is wrong**, and the data that shows
> it is k02 seed 0 — already in the repo, and the very measurement C21 was derived from:
>
> ```
> (-16,-16) 1.841e+06 a+b    (-56,-56) 1.002e+06 a+b
> ( 16, 16) 1.841e+06 a+b    ( 55, 55) 7.783e+05 a+b
> (  3,  3) 1.463e+06 a+b    (-55,-55) 7.783e+05 a+b
> ( -3, -3) 1.463e+06 a+b    (-16, 16) 1.522e+05 a-b   <-- not a+b
> ```
>
> **7 of 8 components are `a+b`, carrying 98.37% of top-8 energy; 1.63% is `a−b`.** So the
> criterion as originally written was **already falsified by the data it was copied from** —
> it was a mis-transcription, not a prediction. Amending it now, before k07's data exists,
> is the honest fix; carrying it unchanged would have manufactured a failure, and fixing it
> after seeing k07 would have been the Gate 1 C6/C7 error. (C18's "100% `a+b`" on Gate 1 is
> **correct** — measured 100.00%. The overstatement is specific to C21.)

**Prediction (amended):** in **≥4 of 5** seeds, `a+b` terms carry **≥95%** of the top-8
logit-DFT energy in **discrete-log** coordinates, and the key set recovered by **ablation
rank** equals the key set by **embedding norm**.

Threshold from the seed-0 measurement above (98.37%), with room for seed variation.
**Falsified if** it holds in ≤3 of 5.

### S4 — C22 holds on this arm at 5 seeds

**Prediction:** permutation p < 0.01 in **5/5** seeds (200 random equal-sized character
sets, none as damaging as the key set). Seed 0 on this arm: p = 0.0000.

**Falsified if** any seed has p ≥ 0.01.

## Success criteria — implemented as code before the run

| # | criterion | threshold | implemented in |
|---|---|---|---|
| 1 | S1 grok | ≥4/5 seeds acc > 0.99 | `analyze_k07.py` |
| 2 | S2 C4 | \|ΔGini_mult\| < 0.10, \|ΔPR_mult\| < 1.0, 4 key freqs in ≥4/5 | `analyze_k07.py` |
| 3 | S3 C21 | 100% `a+b` and set equality in ≥4/5 | `analyze_k07.py` |
| 4 | S4 C22 | permutation p < 0.01 in 5/5 | `analyze_n4.py` |
| 5 | provenance | real git SHA **and** the account that ran it | `run_kernel.py` |

## Required controls

| claim shape | control |
|---|---|
| "sparse in basis X" | additive + multiplicative + **random orthogonal** |
| "these characters are responsible" | 200 random equal-sized sets, permutation p |
| "matches the published number" | the **replication** arm only; never pooled with the thesis arm |
| "the circuit is `a+b`" | read in **discrete-log** coordinates (`dlog=True`), never raw |

## Known confounds

- **Protocol invariant 1.** This arm is units-only and may **not** be pooled with k02 Arm B,
  k04, k05 or k06. Any statement mixing them is invalid.
- **Convention.** Gini on **amplitude**; participation ratio on **energy**. This is the pair
  that was open question O1 for three sessions.
- **Key-frequency values are not comparable across seeds or generators** (C25) — only the
  count, and the set up to multiplication by a unit mod φ.
- **Our grok step (6,400) is already outside their reported 9k–14k** (open question O1b).
  This run measures the spread of that difference; it does not explain it. **No mechanism
  for O1b may be claimed from this run** — that is k08.

## Analysis plan — decided now

`analyze_k07.py`, committed before the run. Criterion 4 goes through `analyze_n4.py`
unchanged. Anything else is **exploratory and labelled**.

## Outcome (filled in AFTER the run — never edit anything above)

**Run landed 2026-09-11 13:16.** All 5 runs on `account-b`, SHA `75bcbe7`, every
artifact carrying both the SHA and the account. Output: `results/k07_arma_seeds/chain.log`,
`results/k07_arma_seeds_n4.txt`.

- **Result: 3 of 4 held.**

| # | prediction | verdict | numbers |
|---|---|---|---|
| S1 | the arm groks reliably | **HELD** | **5/5**, acc 1.0000 every seed. Grok steps **6400, 5800, 8000, 14000, 4600** |
| S2 | C4's published comparison | **NOT HELD** | \|ΔGini_mult\| **0.032** ✓ (<0.10), \|ΔPR_mult\| **0.57** ✓ (<1.0), but **exactly 4 key frequencies in only 3/5** (seeds 1 and 4 gave 6 and 5) against a predicted ≥4/5 |
| S3 | C21 at 5 seeds | **HELD** | `a+b` share of top-8 energy: **98.37%, 100.00%, 100.00%, 100.00%, 100.00%** — **5/5** above 95% |
| S4 | C22 on this arm | **HELD** | permutation p = **0.0000 in 5/5**; restricted improves over baseline by 9.0×, 15.0×, 5.4×, 35.4×, 3.4× |

5-seed means: Gini_mult **0.547 ± 0.010**, PR_mult **4.67 ± 0.44**, Gini_add
**0.017 ± 0.002**, PR_add **55.80 ± 0.05**, key count **4.6 ± 0.8**.
Published: 0.579 / 4.1 / 0.071 / 52.7 / 4.

- **Criteria met: 3 of 4 (S1, S3, S4). S2 fails on its key-frequency-count clause only.**

- **What this changes.**

  **C21 survives the move from 1 seed to 5, and the "100%" was right about the population
  and wrong about the seed it was quoted from.** Four of five seeds are exactly 100.00%;
  the outlier at 98.37% is **seed 0**, the one C21 was written from. The amended ≥95%
  criterion holds 5/5. C22 on this arm is unambiguous — p = 0.0000 in every seed.

  **C4's spectral numbers hold; its key-frequency COUNT does not.** Gini and the
  participation ratio are close to published and tight across seeds (sd 0.010 and 0.44),
  but the 5×-median detector returns 4, 6, 4, 4, 5 — so "4 key frequencies, matching their
  4" is a **seed-0 property, not a property of the arm**. S2 is recorded NOT HELD rather
  than rescued by relaxing the clause. The honest restatement of C4 is: the sparsity
  statistics replicate at 5 seeds; the key-frequency count is 4.6 ± 0.8 and agrees with
  their 4 in 3 of 5 seeds.

  **An incidental observation that bears on O1b, and it is not pre-registered here:**
  grok steps span **4,600–14,000** (mean 7,760). **Seed 3 grokked at 14,000 — inside
  2606.17399's reported 9k–14k band.** So the "we grok faster than published" gap is at
  least partly seed variance rather than a systematic difference, and a single-seed
  comparison (6,400 vs 9k–14k) was never able to show otherwise. This **weakens the premise
  of k08**, which was launched before these seeds existed. k08 still measures whether
  initialisation scale moves grokking time — a real question — but it should no longer be
  described as explaining a gap that may not exist. **Exploratory, labelled, and not
  counted as a criterion.**

- **Deviations from this pre-registration, and why:**
  1. **S3 was amended before the run landed**, and the amendment is boxed in the S3 section
     above with the measurement that forced it. The original criterion had been transcribed
     from C21's wording, which was itself wrong; it was already falsified by k02 seed 0,
     data that predates k07. The amended threshold (≥95% of top-8 energy) was set from that
     same pre-existing measurement (98.37%).
  2. **`analyze_k07.py` initially implemented the pre-amendment criterion.** For one commit
     the committed criterion and the committed code disagreed. Fixed before the data landed
     and verified against the 98.37% checkpoint.
  3. No other deviation. S1, S2 and S4 were computed exactly as specified.
