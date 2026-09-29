# Pre-registration: G2 — Gate 2, the stratified-clock verdict

**Commit this file BEFORE computing a single per-stratum model number.** Git's timestamp is
the evidence that the criteria predated the result. This pre-registration exists *because*
Gate 1's C6/C7 were edited after failing (LAB_NOTEBOOK Entry 12), and because the verdict
proposed below is a **branch the gate does not contain** — declaring a fourth branch after
looking at the data would be that same ordering error, worse.

**Date:** 2026-09-14
**Script:** `analyze_gate2.py` (written after this file, scored against it) ·
**Arm:** thesis · **Compute:** zero training. Re-analysis of `logits_all` / `mlp_acts`
already on disk (k03, N7, k04). No Kaggle quota.

---

## 0. Why the gate as written cannot be answered

`grokking_research_plan.md` §5 asks: *"Do the circuits differ across algebraic
conditions?"* and offers three branches:

| branch | verdict | status against our evidence |
|---|---|---|
| (a) **Yes, circuits differ** | this is the thesis | **ticked** by C23 (non-cyclic moduli have no discrete log at all and need product characters) and C8/C9 (the additive signature tracks zero-divisor density across 18 moduli, as the algebra predicts) |
| (b) **No, circuits are invariant** | weaker but real | **ticked** by C6 (Gini_mult at 121/125 differs from the field by 0.008/0.005, inside seed spread) and C31 (neurons single-frequency on the same key set at 113, 121 and 125) |
| (c) **Composite moduli do not grok** | pivot to *why* | **dead.** C3: 18 moduli grok; C29: all four grok on the from-scratch engine |

**Both (a) and (b) are ticked because the question conflates two levels.** Every
multiplicative statistic this project has ever computed — C6, C21, C22, C25, C31 — is
computed on the **unit sub-grid** via `unit_index` / `_reorder_dlog`. That is 98.2 % of the
multiplication table at n = 113 and **82.6 % at 121, 64.0 % at 125, 65.1 % at 119 and
23.5 % at 165**. The remainder has never been measured, and the model solves it.

## 1. Hypothesis — THE FOURTH BRANCH, declared before the measurement

> **The local mechanism is invariant; the stratification is not.** The multiplication table
> mod n partitions into algebraic strata indexed by (d, e) = (gcd(a,n), gcd(b,n)). On every
> stratum large enough to carry one, the network runs **the same character-based clock**,
> but on the stratum's own **local group** G_m = (Z/(n/m)Z)\*, where m = gcd(de, n). What
> the algebra of n changes is **how many strata there are and how they nest** — one for a
> field, a nested **p-adic chain** for a prime power, a flat **CRT divisor lattice** for a
> square-free composite.

### 1.1 The algebra this rests on (theorem, not measurement)

For x ∈ J_d write x = d ·u with u a unit mod n/d (the 𝒥-class is a torsor under the units —
this is C7's "action coordinate", stated exactly). For y = e ·v likewise. Then with
m = gcd(de, n) and de = m ·w, gcd(w, n/m) = 1:

```
x ·y mod n  =  m · ( w · (u v)  mod  n/m )
```

So **the output on stratum (d,e) is a fixed unit relabelling of the product u ·v in the
smaller ring Z/(n/m)Z**, and the reduction maps (Z/(n/d)Z)\* → G_m and (Z/(n/e)Z)\* → G_m
are surjective. Three consequences, all used below:

1. The target is **exactly constant on the fibres** of those two reductions.
2. Re-indexing the stratum by discrete log in G_m turns `u ·v` into **addition of exponent
   tuples**, so a local clock puts the logit energy on the **diagonal (κ, κ)** — Nanda's
   `f(a+b)` signature, in the local group.
3. The stratum lattice is theory-derived: at n = 121 the chain is G_1 = Z₁₁₀ ⊳ G_11 = Z₁₀ ⊳
   trivial (and J_11 ·J_11 ≡ 0 exactly — a **trivial** stratum, not a small one); at
   n = 125 it is Z₁₀₀ ⊳ Z₂₀ ⊳ Z₄ ⊳ trivial; at n = 165 it is a **lattice with 8 joins**,
   not a chain. `algebra.py::j_multiplication_table` already carries this.

### 1.2 What this claims against the literature

- **Chughtai, Chan & Nanda 2302.03025** — universality is two-level: the circuit *family* is
  always GCR, the precise circuit is arbitrary. Our fourth branch is that shape, made
  algebraic: the *family* (character clock) is invariant, the *index set* (which group) is
  set by the stratum.
- **Chen et al. 2607.07066** — GCR "localizes to disjoint algebraic strata that partition
  the input space". Their Table 1 is 113, 143, 154, 165 and marks **every one
  `Square-free: True`**. A p-adic filtration with **nilpotent** non-units (121, 125) is the
  complement of their design. That is the novelty claim, and it is the only one made here.
- **Zhong et al. 2306.17844 (Clock/Pizza)** — small hyperparameter/init changes induce
  *qualitatively different algorithms* on a fixed task. This is the standing objection, and
  it is why §6 scopes the declaration to one implementation and one precision regime.

---

## 2. Scope, fixed now

**Primary arm:** `results/k03_grid_acts`, arm `B_thesis`, **PyTorch float32**, n ∈
{113, 121, 125, 119, 120, 165} × seeds 0–4. This is the regime C6, C21, C22, C23 and C31
live in, and the only regime in which a 5-seed statement is possible.

**Secondary (replication) arm:** `results/n7_engine`, the from-scratch engine, float64,
n ∈ {113, 121, 125, 119} × seeds 0–2. **Reported separately and NEVER pooled with the
primary arm** — O18 is open and the two implementations may be running different algorithms
(C30's retraction, LAB_NOTEBOOK Entry 29).

**Negative control arm:** runs that **FAILED** to generalise, with activations on disk —
engine `n125_s1` (test acc 0.6082) and k04 `n49_s0` / `n49_s1` / `n54_s1` (0.15–0.27), the
same arm `analyze_o4.py::control_arm` already uses. Per LAB_PROTOCOL.md's three-state rule,
**near-groks are not controls**: k03 `n125_s1` (0.9779) is excluded from both arms.

### 2.1 Which strata are IN SCOPE — rule fixed before looking

A stratum (d, e) is **measurable** iff all of:

| # | rule | why this number |
|---|---|---|
| S-a | `d < n` and `e < n` | J_n = {0}; the whole block is identically 0. Not a mechanism |
| S-b | `|G_m| ≥ 10` | the 8-bin single-frequency family is 8 of `|G_m|²` bins maximised over `|G_m|/2` candidates. Below `|G_m| = 10` the chance level is so high the statistic cannot discriminate — the `jblocks` lesson: **drop a block that cannot supply the count, never report it as a number** |
| S-c | held-out cells in the block ≥ 100 | fewer than 100 unseen cells cannot separate a clock from memorisation |
| S-d | `G_m` cyclic, for the **neuron** endpoint only | `freq_fraction` is a 2-D cyclic statistic. Non-cyclic G_m is **skipped loudly** on that endpoint and still scored on every other |

Everything failing S-a..S-c is reported in the output table as **NOT MEASURABLE** with its
size, never as a zero. The held-out split is reproducible offline and exactly:
`np.random.default_rng(seed).permutation(n*n)`, `cut = int(0.30 * n*n)` — identical in
`kernels/k03_grid_acts/run.py::dataset` and `run_n7.py`.

### 2.2 The measurement pipeline, fixed now

For each run and each measurable stratum (d, e):

1. Slice the block from `logits_all` (n², n) and `mlp_acts` (n², 512).
2. Index rows by `dlog(u)` in (Z/(n/d)Z)\*, columns by `dlog(v)` in (Z/(n/e)Z)\* —
   `transforms.unit_index`, which returns exponent **tuples** and so handles non-cyclic
   components identically (the r = 1 cyclic case is a special case, not a separate path).
3. **Fibre-collapse** both axes onto G_m by averaging over the fibres of the reduction maps.
   The result is a square `|G_m| × |G_m|` grid whose targets are well-defined (§1.1.1).
4. Score G0–G5 on that grid with the existing modules — `ablation.ablation_report`,
   `neurons.freq_fraction`, `neurons.shuffle_control`. **No new statistic is invented that
   the project does not already use elsewhere.**

---

## 3. Success criteria — exact, implemented as code before the run

`analyze_gate2.py` implements these and prints PASS/FAIL per criterion. A stratum is
**"carries a local clock"** iff G1 **and** G2 hold on it.

| # | criterion | threshold | justification of the threshold | implemented in |
|---|---|---|---|---|
| **G0** | **Not memorisation.** Held-out accuracy on the block | **≥ 0.95** | the block must be *generalised*, or the mechanism question is moot. The obvious alternative hypothesis for a 1,100-cell block is rote memorisation, and this is the only thing that separates them | `analyze_gate2.held_out_acc` |
| **G1** | **Causal (PRIMARY).** Permutation p for the diagonal frequency set against **200** random equal-cardinality sets of non-DC bins | **p < 0.01 in ≥ 4/5 seeds**, at every measurable stratum | this is the project's established statistic verbatim (C22, C23, C32; LAB_PROTOCOL.md mandates 200 draws and a permutation p, never a ratio-to-control-mean). 4/5 matches how C6/C22/C31 report | `analyze_gate2.perm_p` |
| **G2** | **Necessity.** Excluded loss (diagonal removed) vs baseline | **≥ 100×** baseline | C22 measured 2.6e7× and 1.85e+01 vs 1.76e-06 at the unit stratum. 100× is deliberately two orders *below* precedent so the criterion is not tuned to what we expect | `analyze_gate2.ablate` |
| **G3** | **The collapse is real, not an averaging artefact.** Fibre-collapse residual (share of block logit variance destroyed by fibre-averaging) vs a **permuted-fibre control** with identical fibre sizes | **residual ≤ 0.5 × control residual**, ≥ 4/5 seeds | a size-matched control is the defence LAB_PROTOCOL.md prescribes after C7b/C19/C20; 0.5× is a factor-of-two separation, the weakest claim worth making. **Strata with fibre size 1 on both axes are exempt and marked N/A** — there is nothing to collapse | `analyze_gate2.fibre_residual` |
| **G4** | **Spectral (descriptive).** Diagonal share of non-DC logit energy, D, as a ratio to chance `R = D ·(|G_m|+1)` | **R ≥ 5.0**, ≥ 4/5 seeds | chance is exactly `(|G_m|−1)/(|G_m|²−1) = 1/(|G_m|+1)`, so R is **size-normalised by construction** — the single defence against the confound that killed C7b, C19 and C20. R ≥ 5 is far below C21's unit-stratum reading (98.4 % of top-8 energy on the diagonal) | `analyze_gate2.diag_share` |
| **G5** | **Neuron (secondary, cyclic G_m only).** Tuned fraction at threshold 0.85 on the collapsed `mlp_acts`, minus the same run's per-stratum **shuffle control** | **real − shuffled ≥ 0.20**, ≥ 4/5 seeds | an *additive* margin, not a ratio: at `|G_m| = 10` the shuffle chance level is high, so C31's RATIO_MIN = 10 is arithmetically unreachable and would be a rigged criterion. 0.20 absolute is the same shape as C31's controls without the size trap | `analyze_gate2.neuron_tuning` |
| **G6** | **Stratification differs across n, and the difference is the algebra.** The set of local groups {G_m} with its nesting, per modulus | 121 and 125 give **nested chains**; 165 and 120 give **non-chain lattices**; 113 gives a **single** stratum | pure algebra from `algebra.j_multiplication_table`; the *model-side* content is that G1+G2 hold on **every** member of each set, which the per-stratum table reports | `analyze_gate2.lattice` |

### 3.1 The verdict rule — fixed now, so it cannot be chosen afterwards

| observation | Gate 2 verdict |
|---|---|
| G1+G2 hold at **every** measurable non-unit stratum, at ≥ 2 moduli, **and** G6 shows the stratum sets differ | **FOURTH BRANCH DECLARED:** local mechanism invariant, stratification not |
| G1+G2 hold at the unit stratum but **fail at ≥ 1 measurable non-unit stratum while G0 holds there** | **BRANCH (a):** circuits genuinely differ *within* a modulus — a stronger and more interesting result than the fourth branch, and it must be reported as such, not smoothed into it |
| G0 fails at the non-unit strata (block not generalised) | **NO VERDICT.** The strata are memorised; the mechanism question is unanswerable from this data and the gate stays open |
| G1+G2 hold everywhere **and** G6 shows no structural difference | **BRANCH (b):** invariance, full stop |

---

## 4. What would falsify this

Concretely, any one of:

- **A measurable non-unit stratum with G0 ≥ 0.95 but permutation p ≥ 0.01** (G1 fails). The
  model generalises on that block by something that is *not* a local character clock. The
  fourth branch is wrong and branch (a) is right.
- **G3 fails** — the block's variance is *not* concentrated on the fibres of the reduction.
  Then the model is not computing in Z/(n/m)Z at all, the collapse is an artefact, and every
  number built on it (G1, G2, G4, G5) is measuring the averaging, not the model.
- **G4 at chance (R ≈ 1) while G1 passes.** Then the ablation is finding *something* causal
  that is not the diagonal — i.e. not `f(u ·v)` — and "the same clock" is the wrong
  description even if "a mechanism" is right.
- **The negative-control arm passing G1.** A failed run must not show a local clock. If
  `n125_s1` (acc 0.6082) reads the same as a grokked run, the statistic is measuring the
  grid, not the model, and nothing here is a result.

## 5. Required controls

| claim shape | control, as implemented |
|---|---|
| "the diagonal is responsible" | ablate the diagonal **and 200 random equal-cardinality non-DC sets** (G1) |
| "sparse/concentrated in the local character basis" | ratio to the **exact** chance level `1/(|G_m|+1)` (G4), plus the per-stratum shuffle control (G5) |
| "the model collapses to the local ring" | **permuted-fibre** control with identical fibre sizes (G3) |
| "this is the mechanism, not memorisation" | **held-out** accuracy per block (G0) |
| "this is a property of grokked models" | the **failed-run** arm — engine `n125_s1`, k04 `n49`/`n54` — scored identically (§4) |
| "it is not one implementation's artefact" | the engine arm, reported separately and never pooled (O18) |

## 6. Known confounds, and what is done about each

1. **SIZE. The confound that has already killed three claims (C7b, C19, C20).** Strata have
   wildly different cell counts (12,100 down to 100) and wildly different `|G_m|`
   (110 down to 2). **Every criterion here is either size-normalised in closed form (G4) or
   compared to a control built on the same grid (G1, G3, G5).** No criterion compares a raw
   number across strata of different sizes, and none is reported for a stratum below the
   S-b/S-c cut.
2. **Memorisation.** Small blocks are cheap to memorise. G0 is the discriminator and it
   gates everything else.
3. **Averaging.** Fibre-collapse could manufacture smoothness. G3 is the discriminator.
4. **Circularity in key-set choice.** Deliberately avoided: the ablated set is the
   **entire diagonal**, fixed by the algebra of §1.1.2. **No key-frequency detector is used
   anywhere in G1–G4**, so the permutation test cannot be rigged by choosing the set that
   ablation says is most damaging (the circularity O8/O11's ablation-rank probe would
   introduce here).
5. **Precision / implementation regime (O18).** The primary declaration is scoped to the
   **k03 torch float32** arm. The engine arm is a replication, not a pool.
6. **Seed.** Every criterion is `≥ 4/5 seeds` on the primary arm. No single-seed statement
   is made.
7. **The train/test split.** Reproduced offline from `(n, seed)`, and `analyze_gate2.py`
   carries a self-check asserting the recomputed split reproduces the run's own final test
   accuracy to within 0.01 before any stratum number is reported.

## 7. Analysis plan — decided now

`PYTHONPATH=. .venv/bin/python analyze_gate2.py [results_dir]` prints, per run:
the in-scope stratum table (with NOT MEASURABLE rows and their reason), G0–G5 per stratum,
the G6 lattice, and a PASS/FAIL line per criterion. `--selfcheck` runs the planted-signal
tests. **Any analysis not listed in §3 is exploratory and must be labelled as such.**

Self-checks required before any number is believed (written first, TDD):

- a **planted local clock** — a synthetic block that is exactly `cos(2πk(dlog u + dlog v)/|G_m|)`
  — must read R ≈ |G_m|+1... i.e. D ≈ 1, permutation p = 0, G3 residual ≈ 0;
- a **planted non-clock** — white noise on the same block — must read R ≈ 1 and p ≈ 1;
- the fibre map must be verified **by planting a signal that depends only on `u mod n/m`**
  and confirming the residual is ~0, and one that depends on `u` itself and confirming it is
  not. **Bin and fibre conventions are settled by planting, never by reasoning** — three
  bin-convention bugs in this project, and reasoning has lost every time.

## 8. Outcome (filled in AFTER the run — never edit anything above)

**Date scored:** 2026-09-14, 20:38 · zero training, zero Kaggle quota · artifacts
`results/gate2/{k03_grid_acts,n7_engine,*_controls}.json`, all provenance-stamped,
`git_dirty=False` · logs `logs/gate2_*.log` · full table
`scripts/gate2_table.py`, discrimination `scripts/gate2_discriminate.py`.

### Result: the FOURTH BRANCH IS DECLARED — and the falsifier in §4 fired

**32 measurable strata over 6 moduli, 157 stratum-run measurements** on the primary arm
(26 non-unit + 6 unit as positive controls). Twelve distinct local groups, from Z₁₀ to
Z₂×Z₄×Z₁₀. The measured fraction of the multiplication table rises from **23.5 % → 82.3 %**
at n=165 and **64.0 % → 89.6 %** at n=125.

- **Criteria met** (primary arm, k03 torch float32, 5 seeds — 4 at n=125):
  **G0, G1, G2, G3, G4, G5 and G6 all HELD** at every measurable stratum. G0 1.0000 ±
  0.0000; G1 p = 0.0000 in all 157; G2 3.98e+02 … 1.0e+08; G3 residual 0.000–0.132 against
  permuted-fibre controls 0.396–0.929 (worst ratio 0.161 vs the 0.5 criterion); G4 D =
  0.837–0.997, i.e. 84–100 % of its exact ceiling; G5 0.193–1.000 against a shuffle control
  of **0.0000 everywhere**; G6 121/125 nested chains, 119/120/165 non-chain lattices.
- **Replication arm** (from-scratch engine, float64, 2–3 seeds): the **same verdict** at
  119, 121, 125. One miss — the n=119 **unit** stratum fails G1 in 1 of 2 seeds, the same
  place N7's E3 missed (`n119_s2`, p = 0.115) and where O8/O11 live.
- **Control arm** (FAILED runs only): **NO VERDICT**, which is the correct control
  behaviour — the failed runs do not generalise, so the mechanism question is unanswerable
  there.

### THE §4 FALSIFIER FIRED. G1 — the criterion this file named PRIMARY — does not separate.

| criterion | grokked | FAILED | separates? |
|---|---|---|---|
| G0 held-out acc | **100 %** | **0 %** | YES |
| **G1 perm p — PRIMARY** | 99 % | **100 %** | **NO** |
| G2 excluded/baseline | **100 %** | **0 %** | YES |
| G3 fibre residual | 100 % | 33 % | partly |
| G4 R ≥ 5 | 100 % | 63 % | partly |
| G5 neuron tuning | 98 % | **0 %** | YES |

**30 % of every block is training data**, and a memorising model's logits are still a
function of u ·v *there* — so the diagonal still carries the energy. G1 tests "the logits
are a function of the product", which memorisation satisfies. The verdict survives because
"carries a local clock" was defined **in advance** as G1 ∧ G2 and G2 separates perfectly,
as do G0 and G5 — but **the weight rests on G0 ∧ G2 ∧ G5**, and this file was wrong to
name G1 primary. It did so because G1 is the project's established statistic (C22/C23/C32),
which is exactly the kind of inherited authority that needed checking rather than citing.

### Deviations from this pre-registration, and why

1. **Two criteria-affecting defects in the FIRST implementation, both fixed and re-run
   before any number here was recorded.** (a) §2's near-grok exclusion was written but not
   implemented — k03 `n125_s1` (0.9779) was being scored in the primary arm. (b) The
   control arm read `hist[-1][3]` as test accuracy, which is **train** accuracy on the
   engine (`run_n7.py` writes a 5-column `hist`), and it conflated FAILED runs with
   near-groks. Both are the binary-gate/fixed-index failures LAB_PROTOCOL.md already documents.
2. **`scripts/gate2_discriminate.py` is new and NOT pre-registered.** It exists only to
   score the §4 falsifier quantitatively. It makes the result weaker, not stronger.
3. **`lut_r2` is new and NOT pre-registered** (labelled EXPLORATORY in code and write-up).
   It quantifies how far the logit tensor is from a scaled one-hot: **0.012–0.074**.
4. **`freq_fraction_nd`, the product-character neuron statistic, is new and NOT
   pre-registered** (labelled EXPLORATORY). §2.1 S-d skips a non-cyclic G_m on the neuron
   endpoint, which removed neuron evidence from exactly the lattice moduli. It is reported
   in starred columns and **never scored as G5**.
5. **No criterion or threshold in §3 was altered after the data existed.** The verdict rule
   in §3.1 was applied as written.
