# Pre-registration: k02_grid — 5-seed thesis grid + O1 replication test

**Written 2026-09-11, while the run is still in flight and BEFORE any of its output has
been analysed.** Committed for timestamp evidence per plan §7.

**Kernel:** `kernels/k02_grid/run.py` · **Arms:** A = replication, B = thesis

## Hypotheses and predictions

### Arm A — O1: is our IPR gap explained by epochs and dataset?

**H:** Our IPR_mult of 12.3 against 2606.17399's published 4.1 (at matching Gini 0.550 vs
0.579 and matching key-frequency count 4 vs 4) is caused by under-training and by including
zero divisors, not by a different circuit.

**Prediction:** at 40k epochs on units only, **IPR_mult falls from 12.3 to below 6**,
approaching their 4.1. Gini_mult stays in [0.50, 0.65]. Key-frequency count stays 4.

**Falsified if:** IPR_mult stays above 8 with the other two metrics unchanged — which would
mean the extra spectral background is a real difference in the learned circuit, not a
training artifact, and H5.3's "at-grok is not converged" does not explain it.

### Arm B — do the single-seed findings survive 5 seeds?

| # | claim under test | prediction | falsified if |
|---|---|---|---|
| B1 | C3 all moduli grok | all 6 moduli reach test acc > 0.99 in ≥4 of 5 seeds | any modulus fails in ≥2 seeds |
| B2 | C6 clock survives prime powers | Gini_mult(121), Gini_mult(125) within 0.1 of Gini_mult(113), mean over 5 seeds | either differs by > 0.15 |
| B3 | O2 n=119 is slowest | mean grok step at 119 exceeds every other modulus by > 1 SD | 119 falls inside the spread of the others |
| B4 | C8 CRT-dual law | permutation enrichment p < 0.01 for 165, 120, 119 in ≥4 of 5 seeds | enrichment fails in ≥2 seeds at any modulus |
| B5 | C9 Gini_add ~ zero-divisor density | Spearman ρ > 0.8 across the 6 moduli, seed-averaged | ρ < 0.6 |
| B6 | C7b (already RETRACTED) regular > non-regular | **no prediction — this was retracted.** Reported as exploratory only | — |

## Criteria are implemented in code, committed before analysis

`analyze_scout.py` (basis sparsity), `jclass_spectra.py` (per-class, size-matched),
`test_crt_law.py` (permutation test, 20k perms). **These files are already committed and
must not be edited to accommodate the results.** If one has a genuine bug, fix it, say so
explicitly in `LAB_NOTEBOOK.md`, and re-run every modulus — not just the failing one.

## Required controls

- Basis sparsity: additive, multiplicative, **and random-orthogonal**, always reported together.
- `Gini_mult` never reported alone (a purely additive signal scores 0.53–0.63 there).
- 𝒥-class comparisons: **size-matched within a single model only.** The cross-class
  aggregate is confounded by class size — this is what invalidated C7b.
- Grokking-time comparisons: report variance across the 5 seeds, not just means, and report
  both raw steps and steps normalised by φ(n).

## Known confounds

- **φ(n) differs across moduli** (32 to 112), so effective task size differs — B3 and any
  grok-time claim must report the φ(n)-normalised figure alongside the raw one.
- **Unit fraction differs enormously** (99.1% at n=113, 26.7% at n=120): Arms A and B are
  different datasets and **must never be pooled**.
- **n=120 is a weak basis cell** (φ=32, factors (2,2,2,4), most characters real). Use it for
  𝒥-class structure, not as a sparsity data point.
- Prime powers make the CRT-dual prediction **vacuous** (n/q = 1). Never counted as support.

## Analysis plan — fixed now

1. `analyze_scout.py` over all 30 Arm-B runs → seed-mean and SD per modulus.
2. `test_crt_law.py` per seed → fraction of seeds with p < 0.01.
3. `jclass_spectra.py` → size-matched pairs only.
4. Arm A: IPR_mult, Gini_mult, key-freq count at n=113 units-only.

**Anything beyond these four is exploratory and will be labelled as such.**

## Outcome (fill in AFTER the run — do not edit anything above)

- Result:
- Criteria met:
- Deviations from this pre-registration, and why:
