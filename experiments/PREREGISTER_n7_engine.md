# Pre-registration: N7 — T2 on the from-scratch engine

**Commit this file BEFORE launching.** Git's timestamp is the evidence the criteria
predated the result.

**Date:** 2026-09-11 · **Script:** `run_n7.py`,
driven by `scripts/run_n7_all.sh` · **Arm:** **thesis** (all n² pairs) · **Runs:** 4 moduli
× 3 seeds = 12 · **Engine:** from-scratch autograd · **Compute:** local CPU, **zero Kaggle
quota**

## Why

Constraint 2 says the autograd engine and the architecture are the researcher's own, and
that **anything trained on PyTorch autograd is scout data, not paper data**. Every
experimental result in `STATE.md` is scout data except the n=17 engine tests and Gate 1.
That includes C6 — the clock surviving at prime powers — which is the thesis's headline.

Gate 1 passed, so the engine is load-bearing and permitted (plan §5). This run moves the
central T2 cell onto it.

This is not a new experiment. It is k03's Arm B restricted to four moduli and three seeds,
re-run on the from-scratch engine, saving the same artifacts.

## Hypothesis

The mechanistic findings of T2 are properties of the task and the architecture, not of the
autograd implementation that trained them.

## Predictions

### E1 — the engine groks at all four moduli

**Prediction:** test acc > 0.99 in **≥2 of 3 seeds** at each of 113, 121, 125, 119.

Reference: k02 Arm B grokked 5/5 at 113, 121, 119, 120, 165 and **4/5 at 125** (seed 1
final acc 0.9779). 125 is therefore the one modulus with a known failure at this train
fraction, and the ≥2/3 threshold is set to accommodate it without excusing a broader
failure.

**Falsified if** any modulus groks in ≤1 of 3 seeds.

### E2 — C6 transfers to paper data (the headline)

**Statistic (amended 2026-09-11, before any run — see "Estimator noise" below):**
Gini_mult for a run is the **mean over the final 10,000 steps**, from `we_traj`
checkpoints saved every 1,000 steps (10 points), with the within-run sd reported
alongside. It is **not** a single-checkpoint read.

**Prediction:** |Gini_mult(121) − Gini_mult(113)| < **0.10** and
|Gini_mult(125) − Gini_mult(113)| < **0.10**, per seed, in **≥2 of 3 seeds** each.

Threshold 0.10 is the one C6, k04's P4 and k05's Q2 already use. Scout values (5 seeds):
113 = 0.564 ± 0.026, 121 = 0.556 ± 0.028, 125 = 0.558 ± 0.048 — observed deltas 0.008 and
0.005, far inside it.

**Convention, non-negotiable:** Gini on **amplitude** — `test_crt_law.freq_energy`,
i.e. `sqrt(sin² + cos²)`, DC dropped, sin/cos combined — over the unit rows in
discrete-log order. This is the Discrete-Log-Clock protocol and the convention behind
every Gini number this project reports. The participation ratio goes on **energy**.

**Reported alongside, always** (never Gini_mult alone — a purely additive signal scores
0.53–0.63 in the multiplicative basis): Gini_add, Gini in a **random orthogonal** basis,
the participation ratio on **energy**, and the key-frequency count.

**Falsified if** either prime power exceeds 0.10 from 113 in ≥2 of 3 seeds.

#### Estimator noise — why E2 averages, measured before the run

Read at a single checkpoint, Gini_mult does not sit still. On k03's saved trajectories the
final-12k window gives:

| run | trajectory over the final 12k | sd | range |
|---|---|---|---|
| 113 s=1 | 0.567 → 0.561 → 0.522 → 0.528 | 0.020 | 0.045 |
| 121 s=2 | 0.564 → 0.556 → 0.492 → 0.504 | 0.032 | 0.073 |
| 125 s=2 | 0.498 → 0.577 → 0.621 → 0.578 | 0.044 | 0.122 |

The full trajectories show this is **oscillation about a flat mean, not a trend** — Gini
plateaus by ~16–20k and then jitters. A single-checkpoint E2 would therefore compare two
values each carrying sd ≈ 0.02–0.044, so their difference carries sd ≈ 0.045: the 0.10
threshold would be ~2.2 sd, and only ~1.6 sd at 125. Averaging 10 tail checkpoints cuts
the estimator sd to ≈ 0.01, putting the threshold at ≳4 sd. **This is why `we_traj` is
saved every 1,000 steps rather than every 2,500.**

This does not disturb C6: its deltas (0.008, 0.005) are far inside 0.10 under either
estimator. It does mean part of C6's reported across-seed spread (±0.026 to ±0.048) is
estimator noise rather than seed variance.

### E3 — the causal result transfers (C22 and C23)

**Prediction:** the restricted/excluded ablation gives permutation **p < 0.01** in **≥2 of
3 seeds** at every one of the four moduli.

Statistic: the fraction of **200** random equal-sized character sets at least as damaging
as the key set. Descriptive ratios quote the **median** control, never the mean — the
control is bimodal and a mean over 20 draws ranged 21×–440× across control RNG seeds alone.

113/121/125 run in **discrete-log** coordinates. **119 = 7×17 is non-cyclic**
((Z/119)* ≅ Z_6 × Z_16), has no discrete log, and runs through the **product-character**
path on exponent tuples — a different code path, which is why it is in this design.

**Falsified if** any modulus gives p ≥ 0.01 in ≥2 of 3 seeds.

### E4 — the CRT-dual law transfers, at 119 only

**Prediction:** key additive frequencies are enriched in multiples of n/q over maximal
prime powers q ‖ n, permutation **p < 0.01** in **≥2 of 3 seeds** at n = 119 (q ∈ {7, 17},
so n/q ∈ {17, 7}).

**This criterion is VACUOUS at 113, 121 and 125** — one CRT component means n/q = 1 and the
predicted set is every frequency. They are not counted as support, and 113 is prime besides.
119 is the only modulus in this design that tests C8 at all.

**Falsified if** p ≥ 0.01 in ≥2 of 3 seeds at 119.

### E5 — the engine and PyTorch agree (costs no new runs)

**Prediction:** at 113, 121 and 125, |mean Gini_mult(engine, seeds 0–2) − mean
Gini_mult(k03 torch, seeds 0–2)| is **within 1 across-seed sd of the k03 value** at that
modulus (0.026, 0.028, 0.048 respectively).

This is a paired comparison against runs that already exist on disk, at the **same seeds**.

**Falsified if** any modulus exceeds its own 1-sd band. Note C10 verifies the engine's
gradients against PyTorch to 2.3e-15, so a systematic disagreement here would be a finding
about the *science* (initialisation RNG, data split RNG, accumulated float32 divergence
over 40k steps), not evidence of an engine bug — and must be investigated as such rather
than written off.

## Criteria are conditioned on grokking (amended 2026-09-11, before any real run)

A 150-step rehearsal of the full 12-run sweep was used as an **untrained control** — the
defence LAB_PROTOCOL.md prescribes after three claims died to confounds. It passed **E2 3/3 and
E4 3/3 on models at test accuracy 0.03**:

- **E2** compared Gini_mult ≈ 0.025 against ≈ 0.025, so |delta| ≈ 0.001 cleared the 0.10
  threshold trivially. A threshold on a *difference* is meaningless when both terms are at
  the noise floor.
- **E4** found CRT enrichment 1.14–1.22 at p = 0.000 after 150 steps. The test itself is
  **not** biased — on pure random embeddings it gives enrichment 1.000 ± 0.016 and rejects
  0/10 at p < 0.01 (measured at n = 119, 120, 165). What this shows is that CRT structure
  is imprinted **very early**, long before the model can predict anything. Interesting, and
  logged as exploratory, but it means E4 can pass on a model that never grokked.

**Therefore E2, E4 and E5 count only seeds whose run reached test acc > 0.99.** A seed that
did not grok is the untrained control, not a data point: it is excluded and **named** in the
output. Where fewer than 2 seeds remain, the verdict is **NOT ASSESSABLE**, never "HELD" and
never "NOT HELD" — those are different claims and the report must not flatten them.

E1 is deliberately left unconditioned: it *is* the grokking criterion.

## What would falsify this

E2 failing is the outcome that matters: it would mean C6 does **not** transfer to paper
data and must be reported as scout-only for the thesis. E5 failing would put every
PyTorch-trained number in the project in question. Both are recorded as results either way.

## Required controls

| claim shape | control |
|---|---|
| "sparse in the multiplicative basis" | Gini in the **additive** basis and in a **random orthogonal** basis, both reported per run |
| "these frequencies are responsible" | ablate the key set **and 200 random equal-sized character sets**; report permutation p, and median-control ratios only as description |
| "the engine reproduces torch" | paired at the **same seeds**, against k03 runs already on disk |
| "moduli differ" | identical hyperparameters at every modulus (plan §3.3); **no per-condition tuning** |
| "this Gini differs from that one" | the estimator's own within-run sd, from the tail trajectory, reported with every Gini |

## Known confounds

- **Seed.** Three seeds is the plan's minimum and seed variance has killed three claims
  (O2, C19, C7b). Every number is reported per seed as well as aggregated. No raw
  grokking-time ordering is interpreted.
- **Dataset size.** train_frac 0.30 is below critical dataset size at small n (C14; k05
  showed 49 groks 0/3 at 0.30 and 3/3 at 0.50). A grok failure at 125 is a **dataset-size
  null**, not an algebraic effect, and is reported as such. **125 is not tuned alone** —
  per-condition tuning destroys the cross-modulus comparison (plan §3.3).
- **Estimator noise.** Quantified above and handled by the tail mean. Note this also
  corrects a narrow defect found while designing this run: `analyze_k03.spectral()` feeds
  **energy** into `gini`, the convention LAB_PROTOCOL.md forbids, and it is the sole source of
  C24's *"Gini drifts <3% everywhere"* clause. In the amplitude convention every reported
  number uses, that drift is 0.3–16%. C24's load-bearing **PR** claim is unaffected (PR
  belongs on energy, which is what it used). `analyze_k04`, `k05`, `k07` and `refcheck`
  all use the correct convention.
- **Horizon.** C24 established that 40k steps is not a converged point for the
  participation ratio at 125 and 119 (PR drifts 5.6% and 5.9% over the final 10k). PR is
  recorded but **no PR claim is made at 125 or 119**. Gini drifts <3% everywhere, so E2 is
  unaffected.
- **φ(n) / size.** E2 compares Gini, not a time or a count, so the size confound that
  killed C7b, C19 and C20 does not apply. No statistic here is normalised by φ(n).
- **Generator choice.** Exponent coordinates need a primitive root and `primitive_root(n)`
  returns the smallest. C25 proved Gini, PR and the ablation losses are exactly invariant
  to it, so this is controlled by an existing test rather than re-checked here.

## Analysis plan — decided now

| criterion | script | statistic |
|---|---|---|
| E1 | `analyze_n7.py` | final test acc per run |
| E2 | `analyze_n7.py` (via `test_crt_law.freq_energy` + `src/analysis/sparsity.gini`) | **mean over the final 10k steps** of Gini on **amplitude**, DC dropped, sin/cos combined. PR on **energy** |
| E3 | `analyze_n4.py results/n7_engine` | permutation p over 200 draws |
| E4 | `analyze_n7.py` | CRT enrichment permutation p at 119 |
| E5 | `analyze_n7.py` | paired against `results/k03_grid_acts` |

Figures regenerate from the saved artifacts via `scripts/render_all.py`. **Any analysis not
listed here is exploratory and must be labelled as such in the write-up.**

## Run configuration — fixed, identical across all four moduli (plan §3.3, §3.4)

```
task:            a*b mod n, all n^2 pairs (thesis arm, NOT units-only)
moduli:          113 (prime), 121 (11^2), 125 (5^3), 119 (7*17, non-cyclic)
seeds:           0, 1, 2
d_model 128 · n_heads 4 · d_head 32 · d_mlp 512 · ReLU · no LayerNorm · learned pos emb
optimizer:       AdamW, lr 1e-3, wd 1.0, betas (0.9, 0.98), full batch
train_frac:      0.30
steps:           40,000
checkpoint:      resume state every 2,500 steps; W_E trajectory every 1,000 steps
                 (we_traj, for E2's tail mean); full snapshot every 5,000 + at grok + at end
saved per run:   W_E, W_U, hist, hist_cols, y_all, mlp_acts, attn, logits_all,
                 we_traj, we_traj_steps
                 -- activations for EVERY seed, not just seed 0
```

Seeds 3 and 4 may be added later as an independent second wave without re-running
anything; the 3-seed result stands on its own and any extension will be reported as such.

## Outcome (filled in AFTER the run — never edit anything above)

*Filled 2026-09-14, session 7. The sweep completed 2026-09-13 00:32; the analysis was
delayed two days by a deadlock in `n7_finish.sh` (see below) and re-run on 2026-09-14.
Every number below comes from `logs/n7_finish.log` at the `2026-09-14` timestamps.*

- **Result:** 12/12 runs, `a*b mod n`, 40k steps, train_frac 0.30, seeds 0-2, from-scratch
  engine (float64), SHA `f9d15b8` (seed 0) / `cafbb36` (seeds 1-2) — diff verified to touch
  no training code. 23 h 04 min wall at 4-way, **zero Kaggle quota**. `analyze_n7`,
  `analyze_n4` and `render_all` all **exit 0**. **11 of 12 runs grokked**; `n125_s1` did not
  (final test acc 0.6082), reproducing k02's 4/5-at-n=125-with-seed-1-failing.

  Grok steps — 113: 4,500 / 6,500 / 9,700 · 121: 10,700 / 10,300 / 5,000 ·
  125: 12,200 / none / 8,700 · 119: 16,300 / 15,600 / 21,300.

- **Criteria met:**
  - **E1 HELD at all four moduli** (criterion: groks in >=2/3 seeds) — 113 3/3, 121 3/3,
    125 2/3, 119 3/3.
  - **E2 HELD** (criterion: |Gini_mult(prime power) - Gini_mult(113)| < 0.10, tail mean over
    the final 10k steps) — 121: 0.013 / 0.196 / 0.067 -> 2/3; 125: 0.078 / 0.001 -> 2/2
    (seed 1 excluded, did not grok).
  - **E3 HELD, 10 of 11** grokked runs at permutation p <= 0.005 over 200 draws.
    **`n119_s2` MISSES at p = 0.115** — recorded as a miss, not rounded away.
  - **E4 HELD** — only n=119 is testable (113/121/125 are prime powers, VACUOUS by
    construction). s1 enrichment 3.20, s2 3.15, both p = 0.0000 -> 2/2.
  - **E5 NOT HELD at all three moduli.** engine 0.754+-0.016 / 0.662+-0.092 / 0.805+-0.036
    vs k03 torch 0.556+-0.015 / 0.543+-0.020 / 0.543+-0.052; |delta| 0.197 / 0.119 / 0.262
    against 1-sd bands 0.026 / 0.028 / 0.048.

- **Deviations from this pre-registration, and why:** **None to the criteria.** No criterion
  was edited, reworded or re-thresholded after seeing a result — E5's failure is reported as
  a failure and `n119_s2`'s p = 0.115 as a miss. Two procedural deviations, neither touching
  the scoring:
  1. **The analysis ran 2 days late.** `n7-sweep` was launched with `RemainAfterExit=yes`,
     which keeps a systemd unit `active` after its script exits, and `n7_finish.sh` waited
     for `is-active` to stop returning `active` — a condition that could never become false.
     It polled 21 h 42 min until a logout stopped both units. Training data was unaffected;
     the chain was simply re-run. Fixed in `scripts/n7_finish.sh` the same day.
  2. **The sweep spans two git SHAs** (HEAD moved mid-run). Verified benign by
     `git diff --name-only f9d15b8 cafbb36`, which touches no file under `src/`, `run_n7.py`
     or `scripts/run_n7_all.sh`.

- **Interpretation of E5 — flagged as NOT YET A CLAIM.** The pre-registration said a
  systematic disagreement would be "a finding about the science, not an engine bug". That
  was not taken at face value. Ruled out: protocol (same `tail_gini`/`mult_amplitude` on
  both sides), hyperparameters (the k03 kernel source has identical lr/wd/betas/init
  scales), tail window (a window-independent check on final weights reproduces it in **7 of
  7** grokked matched pairs, engine sparser by +0.17..+0.28), and the optimizer (the
  engine's AdamW is algebraically identical to `torch.optim.AdamW`). The one difference left
  is **`src/autograd/engine.py:54`: the engine trains float64, k03 trains float32**. This is
  a **correlation with a named mechanism, not a tested mechanism** (C30 / O17). The decisive
  A/B — engine at float32, n=113 s0, ~20k steps — is **not yet run** and must itself be
  pre-registered.
