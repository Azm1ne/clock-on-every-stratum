# Pre-registration: k04 — the extended modulus set (N6)

**Commit this file BEFORE launching.** Git's timestamp is the evidence the criteria
predated the result.

**Date:** 2026-09-11 · **Kernel:** `kernels/k04_extended`
**Arm:** thesis (all n² pairs) · **Runs:** 12 moduli × 3 seeds = 36

## Why

C19 (φ(n)-normalised grokking time separates cyclic from non-cyclic unit groups), C9
(Gini_add tracks zero-divisor density) and C8 (the CRT-dual law) all rest on **6 moduli**.
Six points cannot carry a claim about an algebraic property, and C19's non-cyclic side is
inflated by a single extreme point (φ(120) = 32). This run adds 12 moduli, chosen before
seeing any of their results.

## The modulus set, and why each is in it

| n | factorization | φ(n) | cyclic | sq-free | non-reg 𝒥 | zero-div density | CRT-testable |
|---|---|---|---|---|---|---|---|
| 49 | 7² | 42 | ✓ | ✗ | 1 | 0.143 | vacuous |
| 54 | 2 ·3³ | 18 | ✓ | ✗ | 4 | 0.667 | ✓ |
| 63 | 3² ·7 | 36 | ✗ | ✗ | 2 | 0.429 | ✓ |
| 75 | 3 ·5² | 40 | ✗ | ✗ | 2 | 0.467 | ✓ |
| 81 | 3⁴ | 54 | ✓ | ✗ | 3 | 0.333 | vacuous |
| 98 | 2 ·7² | 42 | ✓ | ✗ | 2 | 0.571 | ✓ |
| 99 | 3² ·11 | 60 | ✗ | ✗ | 2 | 0.394 | ✓ |
| 100 | 2² ·5² | 40 | ✗ | ✗ | 5 | 0.600 | ✓ |
| 105 | 3 ·5 ·7 | 48 | ✗ | ✓ | 0 | 0.543 | ✓ |
| 143 | 11 ·13 | 120 | ✗ | ✓ | 0 | 0.161 | ✓ |
| 147 | 3 ·7² | 84 | ✗ | ✗ | 2 | 0.429 | ✓ |
| 169 | 13² | 156 | ✓ | ✗ | 1 | 0.077 | vacuous |

**5 cyclic / 7 non-cyclic** (C19 needs both). **10 of 12 non-square-free** — the thesis
territory all three cluster papers stop short of. **9 CRT-testable**, 3 prime powers that
are **vacuous by construction and never counted as support** (C8's standing rule).
Zero-divisor density spans **0.077–0.667**, filling the middle of k02's 0.009–0.733.
Adds **three new prime powers** (49, 81, 169) to the two we have (121, 125).

Protocol identical to k02 Arm B — 40k epochs, train_frac 0.30, AdamW lr 1e-3 wd 1.0,
d_model 128 / 4 heads / d_head 32 / d_mlp 512 — so k04 and k02 Arm B **pool**. Nothing here
may be pooled with the units-only replication arm.

## Predictions

### P1 — C19 survives 18 moduli

k02: cyclic 57.1–90.9, non-cyclic 144.0–552.5, complete separation on 6 points.
**Prediction:** pooling k02 Arm B + k04 (18 moduli), median steps/φ(n) for cyclic is below
non-cyclic, and a **Mann–Whitney U test gives p < 0.05**.

**Falsified if** p ≥ 0.05. Complete separation is *not* predicted — with 18 points overlap
is expected; the claim is a real distributional difference, not a clean gap. If it fails,
**C19 is retracted** and the 6-point separation was small-sample luck.

### P2 — C9 survives 18 moduli

k02: Spearman ρ(zero-divisor density, Gini_add) = 0.943 on 6 moduli.
**Prediction:** pooled over 18 moduli, **ρ > 0.6**, and its permutation p < 0.01.

**Falsified if** ρ ≤ 0.6. Threshold set below 0.943 deliberately: 6-point Spearman is a
high-variance statistic and the honest expectation is regression toward the mean.

### P3 — C8 (CRT-dual law) holds on 9 fresh moduli

**Prediction:** threshold-free permutation enrichment of the predicted frequency set
(multiples of n/q for maximal prime powers q ‖ n), 4,000 permutations, **p < 0.01 in ≥2 of
3 seeds** at each modulus, holding at **≥7 of the 9** CRT-testable moduli.

**Falsified if** it holds at ≤5 of 9. The three prime powers are excluded by construction.

### P4 — C6 (the clock survives prime powers) extends to 49, 81, 169

k02: |Gini_mult(121) − Gini_mult(113)| = 0.008, |125 − 113| = 0.005, seed sd 0.026–0.048.
**Prediction:** each of 49, 81, 169 has Gini_mult within **0.10** of the k02 cyclic baseline
(mean of 113/121/125 = 0.559), in ≥2 of 3 seeds. Same threshold k02 used.

**Falsified if** any of the three sits more than 0.10 away. That would mean the clock does
**not** survive at higher prime-power depth (81 = 3⁴ is the deepest we have tested).

### P5 — grokking rate, and the dataset-size confound stated up front

train_frac 0.30 is held fixed for poolability, but C14 established that small moduli fall
below critical dataset size (n=17 needs 0.8). The smallest here is 49 (2,401 pairs, 720
training examples).

**Pre-registered handling:** report the grok rate per modulus. **Any modulus grokking in
fewer than 2 of 3 seeds is excluded from P1's timing analysis and reported as a
dataset-size null — never as an algebraic effect.** This exclusion rule is fixed now,
before any result, precisely so it cannot be chosen later.

## Success criteria — implemented as code before the run

| # | criterion | threshold | implemented in |
|---|---|---|---|
| 1 | P1 C19 | Mann–Whitney p < 0.05 over 18 moduli | `analyze_k04.py` |
| 2 | P2 C9 | Spearman ρ > 0.6, perm p < 0.01 | `analyze_k04.py` |
| 3 | P3 C8 | ≥2/3 seeds at ≥7 of 9 moduli | `analyze_k04.py` |
| 4 | P4 C6 | \|ΔGini_mult\| < 0.10 at 49, 81, 169 | `analyze_k04.py` |
| 5 | P5 exclusion | grok in <2/3 seeds ⇒ dropped from timing | `analyze_k04.py` |
| 6 | provenance | every `.npz` carries a real git SHA | `run_kernel.py::git_sha` |

## Required controls

| claim shape | control |
|---|---|
| "sparse in basis X" | additive + multiplicative + **random orthogonal**, dimension-matched |
| "X groks faster than Y" | 3 seeds, variance reported, φ(n)-normalised alongside raw |
| "frequency set is predicted" | threshold-free **permutation** test, not a 5×median detector |
| "prime powers behave like fields" | prime powers are **vacuous** for C8 — never counted |

## Known confounds

- **Dataset size** — handled by the P5 exclusion rule, fixed above.
- **φ(n) as an extreme point.** k02's non-cyclic side was inflated by φ(120)=32; this set's
  smallest φ is 18 (n=54), so the confound is *not* removed. Report P1 both with and
  without the two smallest-φ moduli; the with-all version is the pre-registered one.
- **Seed count is 3, not 5.** Grokking-time seed variance spans a factor of 3.8 (Entry 14).
  3 seeds is the plan's floor, chosen for budget. Timing conclusions carry that caveat.
- Cyclicity is correlated with being a prime power in this set (all 5 cyclic moduli are
  p^k or 2p^k — that is what cyclic *means* for Z/nZ). C19 cannot separate "cyclic" from
  "prime-power-like" and does not claim to.

## Analysis plan — decided now

`analyze_k04.py`, pooling `results/k02_grid` Arm B with `results/k04_extended`. Statistics:
Mann–Whitney U (P1), Spearman + permutation (P2), the existing threshold-free CRT
permutation test (P3), Gini_mult deltas (P4). Anything else is **exploratory and labelled**.

## Outcome (filled in AFTER the run — never edit anything above)

**Run landed 2026-09-11 09:41.** All 36 runs completed (12 moduli × 3 seeds), every `.npz`
stamped with the real git SHA `9135fe9`. Analysed by `analyze_k04.py`, written and
committed (`4b4c19f`) **before** the run started. Output: `results/k04_extended/analysis.txt`.

- **Result:**

| # | prediction | verdict | numbers |
|---|---|---|---|
| P1 | C19 survives 18 moduli | **NOT HELD → C19 RETRACTED** | cyclic median 88.0 vs non-cyclic 325.0 steps/φ(n) — the predicted direction — but Mann-Whitney exact **p = 0.181** (6 vs 9 moduli after P5). Sensitivity run, dropping the two smallest φ: p = 0.282 |
| P2 | C9 survives 18 moduli | **HELD** | Spearman ρ(zero-divisor density, Gini_add) = **0.794**, permutation p < 5e-5. Predicted > 0.6; regression from the 6-point 0.943 was anticipated |
| P3 | C8 holds on 9 fresh moduli | **HELD — 9 of 9** | enrichment 1.5×–14.4×, **3/3 seeds p < 0.01 at every one** of 54, 63, 75, 98, 99, 100, 105, 143, 147. Predicted ≥7 of 9 |
| P4 | C6 extends to 49, 81, 169 | **NOT HELD** | 81 = 3⁴: |Δ| 0.046/0.024/0.005 → **3/3** ✓. 169 = 13²: 0.023/0.013/0.108 → **2/3** ✓. **49 = 7²: 0.373/0.348/0.115 → 0/3** ✗ |
| P5 | dataset-size exclusion | **APPLIED** | grok rate < 2/3: **49 (0/3), 54 (1/3), 63 (1/3)** — all excluded from P1 and reported as dataset-size nulls |
| 6 | provenance | **HELD for k04** | all 36 k04 artifacts carry SHA `9135fe9`; the 30 pooled k02 artifacts carry `unknown` and predate the stamping line |

- **Criteria met:** 2 of 4 scientific predictions (P2, P3); P1 and P4 falsified as written.

- **What this changes.**

  **C19 is retracted, and the exploratory follow-up says why.** The 6-point "complete
  separation" was reading **φ(n)**, not cyclicity. Over the 15 grokking moduli,
  ρ(−φ(n), steps/φ(n)) = **+0.875** (permutation p < 5e-5) while raw steps *decrease* with
  φ (ρ = −0.668) — so dividing by φ over-corrects and manufactures the ordering. k02's
  cyclic moduli happened to be the three largest φ (112, 110, 100) against non-cyclic
  96, 32, 80; over 18 moduli the median φ is 105 cyclic vs 60 non-cyclic. The extended set
  breaks it directly: **98 = 2 ·7² is cyclic and slow (360.3)**, **143 = 11 ·13 is non-cyclic
  and fast (90.6)**. This is the third instance of the same size confound, after C7b (class
  size) and C20 (block cell count).

  **P3 is the strongest result in the project so far.** The CRT-dual law holds at 9 of 9
  fresh moduli, in 3 of 3 seeds each, at moduli chosen before any of them was run — 10 of
  the 12 non-square-free, which is the territory all three cluster papers stop short of.

  **P4's single failure is at a modulus that never learned the task.** n=49 grokked in
  **0 of 3** seeds and is a P5 dataset-size null (720 training examples). Its embedding has
  no clock to measure, and its Gini_mult seed spread (0.186 / 0.211 / 0.444) is what an
  ungrokked model looks like. **P4 is recorded NOT HELD as written**, because the
  pre-registration scoped the P5 exclusion to P1's timing analysis and nothing else, and
  re-scoping it now — after seeing which modulus failed — is exactly the Gate 1 C6/C7 error.
  The observation that C6 extends at both prime-power moduli that *did* grok (81 = 3⁴, the
  deepest nilpotency tested, and 169 = 13²) is **post-hoc and labelled so**. The clean test
  is a re-run of 49 above critical dataset size.

- **Deviations from this pre-registration, and why:**
  1. **20,000 permutations instead of the stated 4,000** for P3, because `test_crt_law.py`
     already defaulted to 20,000 and reusing it unchanged was simpler than lowering it.
     Strictly conservative — it lowers the resolution floor from 2.5e-4 to 5e-5 and cannot
     make a null look significant.
  2. **P1's exact Mann-Whitney has a p-floor that the 6-modulus version could never clear.**
     At 3 cyclic vs 3 non-cyclic the smallest attainable two-sided p is 0.10, so C19's
     original "complete separation" was not capable of reaching p < 0.05 on any data. Noted
     when `analyze_k04.py` was dry-run on k02 alone, before k04 landed — not after.
  3. No other deviation. P1, P2, P3, P4 and P5 were computed exactly as specified.
