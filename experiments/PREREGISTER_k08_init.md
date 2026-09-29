# Pre-registration: k08 — does initialisation scale set the grokking time? (O1b)

**Commit this file BEFORE launching.** Git's timestamp is the evidence the criteria
predated the result.

**Date:** 2026-09-11 · **Kernel:** `kernels/k08_init`
**Arm:** **replication** (units only, n=113) · **Runs:** 3 conditions × 8 seeds = 24
**Account:** `account-b`

## Why, and a correction to the premise

**O1b:** we grok at **6,400** steps (Arm A) and **6,900** (Gate 1) where 2606.17399 and
Nanda report **9k–14k**, under hyperparameters we verified byte-identical. Entry 16
proposed a mechanism from the knowledge graph — Kunin et al. **2406.06158**, *"Upstream
Init Decreases Time to Grok"* — and `STATE.md` recorded it as: *"our init scales the
downstream projections down (`W_O` by 1/√(n_heads ·d_head), `W_out` by 1/√d_mlp) while
upstream weights use 1/√d_model, i.e. an upstream-heavy initialisation."*

**That premise is wrong, and this pre-registration corrects it before testing anything.**
Measured against fan-in at our configuration (d_model 128, n_heads 4, d_head 32, d_mlp 512):

| matrix | fan_in | 1/√fan_in | our scale | ratio |
|---|---|---|---|---|
| `W_Q/K/V` | 128 | 0.08839 | 0.08839 | **1.00** |
| `W_O` | n_heads ·d_head = **128** | 0.08839 | 0.08839 | **1.00** |
| `W_in` | 128 | 0.08839 | 0.08839 | **1.00** |
| `W_out` | 512 | 0.04419 | 0.04419 | **1.00** |
| `W_U` | 128 | 0.08839 | 0.08839 | **1.00** |

`n_heads ·d_head = d_model`, so `W_O` is **not** downscaled relative to upstream at all, and
`W_out`'s smaller scale is exactly its own fan-in. **Our initialisation is plain fan-in
scaling on every matrix.** There is no pre-existing upstream-heavy asymmetry to blame for
O1b, so the asymmetry has to be **imposed as a condition** rather than detected.

## Hypothesis

Grokking time on this task moves monotonically with the **relative scale of downstream
weights to upstream weights** at initialisation, in the direction Kunin et al. report:
upstream-heavy groks sooner.

## The design

Three conditions, identical in every other respect (n=113, units only, 40k epochs, AdamW
lr 1e-3 wd 1.0, d_model 128 / 4 heads / d_head 32 / d_mlp 512, train_frac 0.30):

| condition | what changes | downstream/upstream scale ratio |
|---|---|---|
| `fanin` | nothing — our current init, the baseline | 1× |
| `upstream_heavy` | `W_O`, `W_out`, `W_U` multiplied by **0.25** | 0.25× |
| `downstream_heavy` | `W_O`, `W_out`, `W_U` multiplied by **4.0** | 4× |

**8 seeds per condition**, not 3 or 5. Seed variance on grokking time spans a factor of
**3.8** (Entry 14), and C19 died to a timing statistic read off too few points; 8 per arm is
the minimum at which the exact Mann-Whitney below can reach p < 0.05 at all.

The modulus is **fixed**, so φ(n) is constant and the confound that retracted C19 cannot
operate here. This is the one place in the project where a grokking-time comparison is
clean.

## Predictions

### I1 — the ordering

**Prediction:** median grok step is ordered
`upstream_heavy < fanin < downstream_heavy`, and the extreme pair
(`upstream_heavy` vs `downstream_heavy`) differs at **exact Mann-Whitney p < 0.05**
(8 vs 8 → 12,870 label assignments, so p < 0.05 is attainable).

**Falsified if** the ordering breaks, or p ≥ 0.05 on the extreme pair. Then initialisation
scale is **not** what makes us faster than the published runs, and O1b stays open with its
most plausible mechanism eliminated — which is a useful negative.

### I2 — magnitude, against the actual discrepancy

The gap to explain is 6,400 → 9k–14k, i.e. **1.4×–2.2×**.

**Prediction:** median grok step of `downstream_heavy` is at least **1.4×** that of
`fanin`. **Falsified if** below 1.4× — the effect would then exist but be too small to
account for O1b, and that must be said plainly rather than reported as support.

### I3 — the circuit is unchanged

**Prediction:** in all three conditions, among seeds that grok, Gini_mult stays within
**0.10** of the `fanin` condition's mean, and the C22 permutation p < 0.01 in ≥6 of 8 seeds.

**Falsified if** either misses. This is the control that separates "init changes *when* the
circuit forms" from "init changes *which* circuit forms" — and only the first is a
grokking-time result.

### I4 — everything still groks

**Prediction:** ≥6 of 8 seeds grok in each condition. **Falsified if** any condition groks
in <6, in which case its timing statistics are reported as censored and **excluded from
I1**, exactly as k04's P5 rule did.

## Success criteria — implemented as code before the run

| # | criterion | threshold | implemented in |
|---|---|---|---|
| 1 | I1 ordering | median ordering + exact MW p < 0.05 on the extreme pair | `analyze_k08.py` |
| 2 | I2 magnitude | downstream_heavy median ≥ 1.4 × fanin median | `analyze_k08.py` |
| 3 | I3 same circuit | \|ΔGini_mult\| < 0.10; C22 p < 0.01 in ≥6/8 | `analyze_k08.py` |
| 4 | I4 grok rate | ≥6/8 per condition, else censored and dropped from I1 | `analyze_k08.py` |
| 5 | provenance | real git SHA **and** the account that ran it | `run_kernel.py` |

## Required controls

| claim shape | control |
|---|---|
| "X groks faster than Y" | 8 seeds per arm, full distribution reported, exact permutation test |
| "the circuit is the same" | Gini_mult **and** the causal ablation, not sparsity alone (I3) |
| "this explains O1b" | the magnitude check I2, against the real 1.4×–2.2× gap |
| "sparse in basis X" | additive + multiplicative + random orthogonal |

## Known confounds

- **Only the scale changes, not the parameterisation.** Multiplying three matrices by a
  constant also changes the effective learning rate on them under AdamW — AdamW is
  scale-invariant in the gradient direction but **weight decay is not**, and wd = 1.0 here
  is large. A difference could therefore be a weight-decay effect rather than an
  initialisation effect. **Stated now**, not discovered later; distinguishing them needs a
  wd sweep, which this run does not do.
- **8 seeds is still small** for a distribution whose spread is a factor of 3.8. The exact
  test is used precisely because the normal approximation is not credible here.
- **This is the replication arm** and does not pool with any thesis-arm run.
- A null result on I1 does **not** clear initialisation entirely — only this particular
  upstream/downstream contrast at this configuration.

## Analysis plan — decided now

`analyze_k08.py`, committed before the run: grok-step distributions per condition, exact
Mann-Whitney on the extreme pair (the same enumeration `analyze_k04.py` uses), median
ratios, Gini_mult per condition, and the C22 permutation p through `analyze_n4.py`.
Anything else is **exploratory and labelled**.

## Outcome (filled in AFTER the run — never edit anything above)

**Run 2026-09-11, 24 runs, SHA `75bcbe7`, account `account-b`. All 24 artifacts
stamped. Analysed by `analyze_k08.py`, committed before the data existed.**

- **Result:** initialisation scale controls grokking time **strongly, monotonically, and
  in the direction opposite to the prediction.**

  | condition | downstream scale | grok steps (8 seeds) | median | mean |
  |---|---|---|---|---|
  | `upstream_heavy` | 0.25× | 13800, 13800, 15800, 16600, 21000, 21800, 30600, 32600 | **18,800** | 20,750 ± 6,864 |
  | `fanin` (ours) | 1.0× | 4600, 5800, 6200, 6400, 6400, 8000, 11600, 14000 | **6,400** | 7,875 ± 3,033 |
  | `downstream_heavy` | 4.0× | 3400, 3400, 4000, 4400, 4600, 4600, 4800, 5600 | **4,500** | 4,350 ± 691 |

  Scaling the downstream projections **up** makes grokking **faster** — 4.2× between the
  extremes across a 16× scale range, exact Mann-Whitney U = 64.0, **p = 0.0002** over all
  12,870 label assignments. Kunin et al. `2406.06158` predicts upstream-heavy
  initialisation *decreases* time to grok; here it is **2.9× slower** than plain fan-in.

- **Criteria met:**

  | # | criterion | verdict | why |
  |---|---|---|---|
  | I1 | ordering `upstream < fanin < downstream`, extreme pair p < 0.05 | **NOT HELD** | the ordering is exactly **reversed**. p = 0.0002 is comfortably significant, but significance in the wrong direction falsifies the prediction as written |
  | I2 | `downstream_heavy / fanin` median ratio ≥ 1.4× | **NOT HELD** | observed **0.70×**. The effect is real but cannot account for a 1.4×–2.2× gap, and points the wrong way besides |
  | I3 | Gini_mult within 0.10 of `fanin` in every condition **and** C22 permutation p < 0.01 in ≥6/8 seeds | **HELD, both halves** | *Sparsity:* 0.566 ± 0.042 / 0.559 ± 0.021 / 0.553 ± 0.023; \|Δ\| = 0.007 / 0.000 / 0.006. *Causal* (scored 2026-09-11, `analyze_n4.py results/k08_init`): permutation p = **0.0000 in 24 of 24 runs** — **8/8 in every condition**, against a threshold of 6/8. Excluded/median-control separation 4.8e+02× to 2.6e+07×; the control mean is unstable at every run, as C22's restatement requires |
  | I4 | ≥6 of 8 seeds grok per condition | **HELD** | **8/8 in all three** |
  | 5 | provenance on every artifact | **HELD** | `git_sha 75bcbe7`, account `account-b`, ×24 |

  **What this settles.** I1 and I2 both fail, so **initialisation scale is not what makes
  us faster than the published runs.** O1b stays open with its most plausible mechanism
  **eliminated** — which is precisely the useful negative this pre-registration named in
  advance as the I1 failure mode.

  **What I3 adds, and it is the interesting half.** The circuit is the *same* circuit in
  all three conditions — Gini_mult moves by at most 0.007 while the grok time moves by
  4.2×. Initialisation scale changes **when** the clock forms, not **which** circuit
  forms. That is what makes this a timing result rather than a different-model result,
  and it is the control that would have invalidated the whole reading had it failed.
  With the causal half now scored, I3 rests on both legs the pre-registration demanded:
  the key characters are not merely sparse in the same basis, they are **causally
  load-bearing in all 24 runs** — removing them costs 8 to 26 orders of magnitude, and
  not one of 200 random equal-sized character sets is as damaging, in any run.

  **Unplanned observation, flagged as exploratory — the O8/O11 shape appears at n=113.**
  The *excluded* loss is catastrophic everywhere, but the *restricted* loss (keep only the
  key characters) is **worse than baseline in 3 of 8 `upstream_heavy` seeds** (s0, s4, s7)
  and better in 8/8 of both `fanin` and `downstream_heavy`. That is the same "necessary
  but not sufficient" pattern O11 records at n=119 — and it is the first time it has been
  seen at n=113, where C21/C22 otherwise hold cleanly. It was **not pre-registered**, it
  is a single condition at 8 seeds, and it is recorded here only so the next person does
  not rediscover it as a surprise. Whether it is the 5×-median detector failing under an
  upstream-heavy init, or a second mechanism, is exactly the question O8 already asks.

- **Deviations from this pre-registration, and why:** **none.** All four criteria were
  scored exactly as written, by code committed before the data existed. The premise
  correction (our init is plain fan-in on every matrix, because `n_heads ·d_head = d_model`,
  so the asymmetry is *imposed* here rather than inherited) was made **in this file before
  launch**, not after.

  **Confound, stated in advance and NOT resolved by this run:** weight decay is 1.0 and is
  **not scale-invariant**, so a 16× change in downstream weight scale changes the effective
  decay pressure on those weights too. This result cannot separate "initialisation scale"
  from "decay pressure at that scale". Separating them needs a weight-decay sweep that k08
  does not do. **Do not write this up as a pure initialisation effect.**

  **Read alongside Entry 22:** k07 showed our grok steps span 4,600–14,000 across 5 seeds,
  with seed 3 landing *inside* the published 9k–14k band. So the gap O1b exists to explain
  may not exist. Both results point the same way — stop attributing the difference to any
  mechanism until `2606.17399`'s **per-seed** numbers are in hand. That is a question for
  the corpus, not for more GPU.
