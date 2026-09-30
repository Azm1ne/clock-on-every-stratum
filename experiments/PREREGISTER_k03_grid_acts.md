# Pre-registration: k03 — the activation grid (N8)

**Commit this file BEFORE launching the run.** Git's timestamp is the evidence that the
criteria predated the result.

**Date:** 2026-09-11 · **Kernel:** `kernels/k03_grid_acts`
**Arm:** thesis (Arm B) + replication (Arm A)

## Why this run exists

Three things, in priority order.

1. **C20 must be re-tested with a size-controlled statistic.** Before this run was designed,
   the seed-0 k02 activations were re-read and the C20 statistic was found **confounded by
   block cell count** — Spearman ρ(cells, variance) = **+0.85 to +1.00** at all six moduli.
   A flat-variance control (i.i.d. noise, identical variance in every block by construction)
   reproduces ρ = **+0.918** at n=125 against **+0.929** on the real data
   (`src/analysis/jblocks.py::_selfcheck`). This is the same class-size confound that
   retracted C7b. C20 as written in STATE.md is **not supported** and is downgraded pending
   this run.
2. **The whole neuron-level story rests on seed 0**, because k02 saved `mlp_acts` only for
   seed 0 of each modulus.
3. **A convergence audit.** Every spectral number we hold is read from a single step-40k
   snapshot. `W_E` trajectory snapshots make "40k is converged" a measurement instead of an
   assumption, and bear directly on O1.

## Hypotheses and predictions

### H1 — 𝒥-class depth grades activation variance, over and above block size

**Prediction.** In a per-modulus weighted least-squares fit
`var_block ~ β0 + β1 ·log2(cells) + β2 ·depth(a) + β3 ·depth(b)` on the size-controlled
statistic, the depth coefficients are **negative** and the fit including them beats the
size-only fit. Pre-registered threshold: **β2 + β3 < 0 in ≥4 of 5 seeds at both n=121 and
n=125**, and the mean over seeds is more than 2 seed-sd below 0.

**Falsified if** β2 + β3 is not consistently negative, or its seed-mean is within 2 sd of 0.
Then variance is a block-size effect and carries no depth law, and C20 is **retracted**.

### H2 — the "constant-zero blocks carry no variance" reading is wrong

Seed 0 already contradicts it: at n=121, `J_11 × J_11` composes to constant 0 over 100
cells and carries variance **2.380**, four times the equally-sized, also-constant-zero
`J_1 × J_121` at **0.577**, and comparable to non-zero blocks. The `0.000` entries reported
in LAB_NOTEBOOK Entry 13 are `np.var` of a **single cell** (J_n = {0}, one element).

**Prediction.** `var(J_11×J_11) > 2 × var(J_1×J_121)` at n=121 in **≥4 of 5 seeds**.

**Falsified if** it holds in ≤2 seeds. Either way the single-cell zeros are an artifact and
that sentence is struck from Entry 13's finding by a correcting entry.

### H3 — O6: the depth grading is not additive

Seed 0: at n=125 `(J_1,J_25)` = 2.559 exceeds `(J_5,J_5)` = 2.057 at equal total depth 2 —
and both blocks have **exactly 400 cells**, so this pair is already size-matched and is the
most trustworthy part of the original finding.

**Prediction.** `var(J_1,J_25) > var(J_5,J_5)` at n=125 in **≥4 of 5 seeds**.

### H4 — convergence audit: is step 40,000 converged?

**Prediction.** On the size-controlled `W_E` spectral metrics, the change over the final
quarter of training is small: `|m(40k) − m(30k)| / m(40k) < 0.05` for Gini_mult and for the
participation ratio, in **≥4 of 5 seeds at every modulus**.

**Falsified if** the metrics are still moving by >5% over the last 10k steps. Then every
spectral number in the ledger is a transition-time snapshot rather than a converged value,
and C4/C6/C9 need re-reading against a trajectory. This is the hazard named by 2607.06639
(**Tier 4, unverified 2026 preprint — used only as a hypothesis source, cited nowhere**).

### H5 — O1: is our participation ratio still falling at 40k?

Arm A, n=113, units only. **Prediction.** PR_mult is flat over the last 10k steps
(<5% change). If flat, undertraining is eliminated as an explanation for the 12.5-vs-4.1 gap
and the remaining candidates are `W_E` normalisation or a real circuit difference. If still
falling, undertraining returns and the run should be extended.

Note the arithmetic that already disfavours undertraining: 2606.17399 uses **identical**
hyperparameters (30% train, AdamW lr 1e-3, wd 1.0, 40k epochs, d_model 128 / 4 heads /
d_head 32 / d_mlp 512, no LayerNorm) and groks at **9,000–14,000** where we grok at
**6,400** — so we have *more* post-grok training than they do (≈5.2 T_grok vs ≈2.5), not
less.

## Success criteria — implemented as code before the run

| # | criterion | threshold | implemented in |
|---|---|---|---|
| 1 | every block measured at equal cell count; sub-threshold blocks dropped, never zeroed | exact | `src/analysis/jblocks.py::block_variance` |
| 2 | flat-variance input scores flat | spread < 0.25 at 16 cells | `jblocks.py::_selfcheck` (PASSES) |
| 3 | H1 depth coefficient | β2+β3 < 0 in ≥4/5 seeds, n=121 and n=125 | `analyze_k03.py` |
| 4 | H2 constant-zero counterexample | ratio > 2 in ≥4/5 seeds | `analyze_k03.py` |
| 5 | H3 non-additivity | holds in ≥4/5 seeds | `analyze_k03.py` |
| 6 | H4 convergence | <5% drift over final 10k, ≥4/5 seeds, all moduli | `analyze_k03.py` |
| 7 | H5 Arm A PR trajectory | <5% drift over final 10k | `analyze_k03.py` |

## Required controls

| claim shape | control |
|---|---|
| "variance is graded by depth" | **block cell count**, held fixed by subsampling AND entered as a regressor |
| "constant-zero blocks are quiet" | size-matched constant-zero blocks that are *not* quiet (n=121 J_11×J_11) |
| "sparse in basis X" | additive + multiplicative + random-orthogonal, as always |
| "X groks faster than Y" | 5 seeds, variance reported |

## Known confounds

- **Block cell count** — the one that broke C20. |J_d| = φ(n/d), so depth and size are
  structurally coupled; only n=120 and n=165 have exactly-equal-size blocks of differing
  depth. Everywhere else the size-controlled regression is the only valid instrument.
- **Distinct-product count.** A block spanning few distinct products cannot express much
  variance in a product-dependent quantity. Not separable from size at these moduli;
  recorded as a limitation, not controlled.
- Seed. φ(n). Unit fraction. Prime powers remain vacuous for the CRT-dual law.

## Analysis plan — decided now

`analyze_k03.py`, using `src/analysis/jblocks.py` for every block statistic and the existing
`analyze_scout.py` path for spectra. Statistics: the weighted LS fit of H1; the two
size-matched ratios of H2 and H3; the drift fractions of H4/H5. Anything else is
**exploratory and will be labelled so**.

## Outcome (filled in AFTER the run — never edit anything above)

**Run landed 2026-09-11 06:20.** 31 checkpoints, 5 seeds × 6 moduli + Arm A, activations
and `logits_all` saved for **every** seed. Analysed by `analyze_k03.py`; output in
`results/k03_grid_acts/chain.log`.

- **Result:**

| # | hypothesis | verdict | numbers |
|---|---|---|---|
| H1 | depth grades block variance over and above size | **NOT HELD → C20 RETRACTED** | β₂+β₃ = −70.25 ± 40.48 (n=121), −14.16 ± 9.43 (n=125). Negative in **5/5** seeds at both, but the seed-mean is **1.74 sd** and **1.50 sd** from zero — inside the pre-registered 2 sd band |
| H2 | constant-zero blocks are NOT quiet | **HELD** | `J_11×J_11` vs `J_1×J_121`, both composing to constant 0 and equally sized: ratio 4.13, 3.16, 12.25, 5.82, 5.13 — **5/5** seeds above 2× |
| H3 | the grading is not additive in depth (O6) | **NOT HELD** | `(J_1,J_25)` > `(J_5,J_5)` in only **2/5** seeds (2.559/2.057, 0.219/0.218, then 3.425/5.699, 2.109/2.145, 1.311/2.144). O6 was single-seed noise |
| H4 | step 40k is converged | **PARTIAL** | Gini drift < 5% everywhere; **PR drift**: 113 ✓ (1.0%, 2.3%), 121 ✓ (2.8%) — but **125 5.6%, 119 5.9%, 120 7.7%, 165 4.8%**, converged in only 2–3 of 5 seeds |
| H5 | Arm A PR is flat at 40k | **HELD** | Arm A n=113: Gini drift 0.4%, PR drift 1.0%, PR_mult 3.99 |

- **Criteria met:** 2 of 5 outright (H2, H5); H1 and H3 falsified as pre-registered; H4
  passes at the two cyclic prime-power moduli and fails at the four composite ones.

- **What this changes.** **C20 is retracted.** The depth coefficient has the predicted sign
  in every seed, so the effect is not absent — but it is not separable from zero at 5 seeds,
  and Entry 16 already showed a flat-variance noise control reproducing ρ = +0.918. Sign
  without separation is not a claim. **O6 is closed as noise.** H2 is the positive result:
  "constant-zero blocks are quiet" is definitively wrong — two blocks that both compose to
  constant 0, at equal cell count, differ in variance by 3–12×.

  **H4 is the most consequential.** At 125, 119, 120 and 165 the participation ratio is
  still moving 4.8–7.7% over the final 10k steps. **Every PR number reported at those four
  moduli is a measurement at a non-stationary point**, and cross-modulus PR comparisons
  there inherit that. Gini is stable (<3% everywhere) and is unaffected.

- **Deviations from this pre-registration, and why:**
  1. **The criteria table (row 3) and the H1 text disagreed.** The table said "β₂+β₃ < 0 in
     ≥4/5 seeds"; the H1 text said falsified if "not consistently negative, **or** its
     seed-mean is within 2 sd of 0". `analyze_k03.py` applied the **stricter** H1 text.
     By the table alone H1 would have read HELD (5/5 negative). Recorded rather than
     silently taken: the stricter reading is the one written as the hypothesis, and
     choosing the looser one after seeing a 5/5 sign count is precisely the Gate 1 error.
  2. The degrees-of-freedom note printed a hardcoded "n=121" for both moduli; fixed to
     report the real block count per modulus (121: 6 blocks / 2 df; 125: 13 / 9 df). No
     number changed.
