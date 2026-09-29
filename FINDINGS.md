# Findings

Every result, organised by topic rather than by date. `LAB_NOTEBOOK.md` is the chronological
record; `STATE.md` is the operational resume point; **this file is the scientific one**.

**Verification vocabulary.** `VERIFIED` = reproduced and self-checked.
`1 seed` = single run, not yet claimable. `SUGGESTIVE` = a pattern, not a result.
`RETRACTED` = we disproved our own earlier claim. Every number below carries its status.

**Two arms, never pooled.** *Replication arm*: units only, 40k epochs, matches 2606.17399 —
only this arm is comparable to their numbers. *Thesis arm*: all n² pairs, zero divisors
included, since they are the object of study. Unit fraction varies from 99.1% (n=113) to
26.7% (n=120), so pooling them would confound every cross-modulus comparison.

---

## 1. Algebraic design (VERIFIED, self-checking)

The modulus set, regenerated computationally — cyclicity by explicit primitive-root search,
never by table lookup (`src/tasks/algebra.py::_selfcheck`).

| n | factorisation | field | φ(n) | cyclic units | ω(n) | square-free | 𝒥-classes | **non-regular** | nilpotents |
|---|---|---|---|---|---|---|---|---|---|
| 113 | 113 | ✅ | 112 | ✅ | 1 | ✅ | 2 | **0** | 1 |
| 121 | 11² | ❌ | 110 | ✅ | 1 | ❌ | 3 | **1** | 11 |
| 125 | 5³ | ❌ | 100 | ✅ | 1 | ❌ | 4 | **2** | 25 |
| 119 | 7·17 | ❌ | 96 | ❌ | 2 | ✅ | 4 | **0** | 1 |
| 120 | 2³·3·5 | ❌ | 32 | ❌ | 3 | ❌ | 16 | **8** | 4 |
| 165 | 3·5·11 | ❌ | 80 | ❌ | 3 | ✅ | 8 | **0** | 1 |

Reproduces 2607.07066's Theorem D.17 (square-free ⟹ every 𝒥-class regular) on their own
moduli 165, 143, 154 — a free correctness check on our implementation.

### 1.1 The structural difference, in one line (VERIFIED)

Green's 𝒥-classes of the monoid `(Z/nZ, ·)` are `J_d = {x : gcd(x,n) = d}`, composing as
`J_d · J_e → J_gcd(de,n)` (`algebra.py::j_multiplication_table`):

```
square-free (165, 119):  J_d · J_d = J_d        closed — a group; their Thm 3.4 applies
n = 121 (depth 2):       J_11 · J_11 = {0}      null semigroup — one step to zero
n = 125 (depth 3):       J_5 · J_5 → J_25 → {0} graded — two steps
```

**Square-free: the diagonal is closed. Prime power: it descends toward zero.** That is
exactly why the local-character account breaks, and it makes 121 vs 125 a clean graded
variable (nilpotency depth 2 vs 3) at nearly identical model and dataset size.

---

## 2. Grokking occurs at every modulus (**5 seeds**, PyTorch autograd)

k02 Arm B, 40k epochs, all n² pairs. Pre-registered criterion B1 (grok in ≥4 of 5 seeds):
**HELD at every modulus.**

| n | grok steps, 5 seeds | mean | sd | grokked |
|---|---|---|---|---|
| 113 | 6800, 7600, 5600, 6600, 5400 | 6,400 | 810 | 5/5 |
| 121 | 7200, 7200, 13400, 10000, 12200 | 10,000 | 2,533 | 5/5 |
| 125 | 9400, **None**, 3200, 13000, 8400 | 8,500 | 3,506 | **4/5** |
| 119 | 21400, 24800, 15800, 6600, 23800 | 18,480 | 6,710 | 5/5 |
| 120 | 10000, 13200, 23200, 16200, 25800 | 17,680 | 5,961 | 5/5 |
| 165 | 6400, 15200, 12000, 9000, 15000 | 11,520 | 3,419 | 5/5 |

**n=125 seed 1 never reached 99%** (final acc 0.9779). Recorded, not discarded.

**Seed variance is enormous** — n=119 spans 6,600 to 24,800, a factor of 3.8. **Raw
grokking-time orderings are not interpretable.** See §2.2.

### 2.1 RETRACTED: "n=119 is uniquely slow" (was open question O2)

The single-seed scout showed 119 at 15,900 against a next-slowest 10,700, and it was flagged
as a puzzle. At 5 seeds: **119 = 18,480 ± 6,710, 120 = 17,680 ± 5,961 — heavily
overlapping.** Pre-registered criterion B3 **NOT HELD. The effect was noise.** O2 closed.

### 2.2 ❌ RETRACTED: "φ(n)-normalised grokking time separates cyclic from non-cyclic" (C19)

> **RETRACTED 2026-09-11 (k04, pre-registered P1).** Over **18 moduli** the separation is
> gone: cyclic median 88.0 against non-cyclic 325.0 steps/φ(n) — the predicted direction —
> at exact Mann-Whitney **p = 0.181**. Pre-registered threshold was p < 0.05, and the
> pre-registration said in advance that failure means retraction.
>
> **And the confound is identified, not just suspected.** Over the 15 grokking moduli,
> **ρ(−φ(n), steps/φ(n)) = +0.870** (permutation p < 5e-5), while **raw steps decrease with
> φ** (ρ = −0.665; midranks, P9a 2026-09-29 — the ordinal ranks read +0.875 / −0.668). Dividing by φ(n) over-corrects and manufactures the ordering. The six
> k02 moduli made it look clean only because the three cyclic ones were the three largest
> φ (112, 110, 100) against non-cyclic 96, 32, 80 — over 18 moduli the median φ is 105
> cyclic versus 60 non-cyclic.
>
> **Two counterexamples from the extended set settle it:** 98 = 2·7² is **cyclic and slow**
> (360.3); 143 = 11·13 is **non-cyclic and fast** (90.6).
>
> This is the **third** claim lost to a size confound, after C7b (𝒥-class size) and C20
> (block cell count). The table below is kept for the record and must not be cited.

Normalising by φ(n) (the effective task size, as §H2.5 requires):

| n | φ(n) | unit group | steps / φ(n) |
|---|---|---|---|
| 113 | 112 | **cyclic** | **57.1** |
| 125 | 100 | **cyclic** | **85.0** |
| 121 | 110 | **cyclic** | **90.9** |
| 165 | 80 | non-cyclic | 144.0 |
| 119 | 96 | non-cyclic | 192.5 |
| 120 | 32 | non-cyclic | 552.5 |

**[HISTORICAL READING, RETRACTED — the paragraph below is what was believed at 6 moduli.
Do not quote it as a finding.]** "Complete separation, no overlap: cyclic 57–91,
non-cyclic 144–553. The split is on **cyclicity of the unit group** — not regularity, not
ω(n), not square-freeness. The raw ordering is noise; the normalised one is clean."

**Why it is wrong:** the last clause is exactly backwards. Normalising by φ(n) is what
manufactured the separation (ρ(−φ, steps/φ) = +0.870), and this sentence leaked out of its
retraction once already — it was paraphrased into `STATE.md`'s O2 closure as live support
for a retracted claim, and had to be corrected there on 2026-09-12.

**Status at the time: STRONG PATTERN, not a claim** — 6 moduli, 3 per group, φ(120)=32 an
extreme point. The extended set was run, and retracted it. The full 18-modulus table is in
`results/k04_extended/analysis.txt`.

**What replaces it, as an observation and not a claim:** grokking time per unit of φ(n) is
strongly predicted by φ(n) itself. No algebraic property of the unit group has been shown to
predict grokking time once that is controlled for.

### 2.3 The original single-seed scout (superseded, kept for the record)

`a·b mod n`, 25k epochs, config matched to 2607.07066.

| n | non-regular 𝒥 | grok step | final test acc |
|---|---|---|---|
| 113 | 0 | **6,400** | 1.0000 |
| 165 | 0 | 6,900 | 0.9999 |
| 121 | 1 | 7,700 | 1.0000 |
| 125 | 2 | 9,500 | 1.0000 |
| 120 | 8 | 10,700 | 1.0000 |
| 119 | 0 | **15,900** | 0.9991 |

**Gate 2's worst case — "composite moduli do not grok" — is eliminated.** Cost: 37 min GPU.

Within the cyclic family, grok time is monotone in non-regular class count
(113→6.4k < 121→7.7k < 125→9.5k). **n=119 breaks any global ordering** — slowest by 50%
despite zero non-regular classes. It is the only case that is square-free ∧ non-cyclic ∧
zero-divisor-poor. One seed; **not interpreted**. (Open question O2.)

---

## 3. The discrete-log clock survives at prime powers (**5 seeds** — the central result)

k02 pre-registered criterion B2. Protocol: amplitude, DC dropped, sin/cos combined.

| n | Gini_mult (mean ± sd) | Gini_add (mean ± sd) | key_mult (seed 0) |
|---|---|---|---|
| 113 (field) | **0.564 ± 0.026** | 0.018 ± 0.004 | [13, 27, 33, 43] |
| **121 = 11²** | **0.556 ± 0.028** | 0.125 ± 0.010 | [11, 29, 33, 49] |
| **125 = 5³** | **0.558 ± 0.048** | 0.177 ± 0.030 | [5, 25, 36, 40, 50] |
| 119 | non-cyclic | 0.421 ± 0.095 | — |
| 120 | non-cyclic | 0.587 ± 0.029 | — |
| 165 | non-cyclic | 0.579 ± 0.023 | — |

**|Gini_mult(121) − Gini_mult(113)| = 0.008 · |125 − 113| = 0.005.** Predicted < 0.10; the
observed differences are an order of magnitude below the threshold and far inside the seed
spread. **HELD.**

> Prime-power local rings — with nilpotents and non-regular 𝒥-classes, the case 2607.07066
> names as its open problem — carry **the same multiplicative-basis sparsity as the field**.
> Cyclicity of the unit group is what the clock needs: not primality, not regularity.

### 3.1 The original single-seed measurement (superseded, kept for the record)

Protocol: amplitude (`√(‖s_k‖²+‖c_k‖²)`), DC dropped, sin/cos combined — the
2606.17399 protocol. Squared energy inflates Gini ≈0.55 → ≈0.90 and is **not** comparable.

| n | Gini_mult | Gini_add | Gini_random | key freqs |
|---|---|---|---|---|
| 113 (field) | 0.550 | 0.011 | 0.096 | **4** |
| **121 (11²)** | 0.521 | 0.134 | 0.083 | **4** |
| **125 (5³)** | 0.587 | 0.184 | 0.096 | **4** |
| 119 | 0.536 | 0.265 | 0.081 | 7 |
| 165 | 0.629 | 0.595 | 0.097 | 5 |
| 120 | 0.588 | 0.574 | 0.110 | 5 |

> ⚠️ *(2026-09-29, P6 fix.)* Until today this table printed **energy** Ginis under the
> amplitude header above (0.902 / 0.879 / 0.905 / 0.853 / 0.885 / 0.813 mult; 0.040 / 0.364 /
> 0.437 / 0.566 / 0.877 / 0.869 add), because `analyze_scout.py` called `gini(energy(...))`.
> The values are now `analyze_scout.py`'s protocol output (`analyze_n7.mult_amplitude` /
> `add_amplitude`). The random column is Gini on √(folded energy) of a full-row random basis.
> The conclusion below does not move: 121 and 125 sit level with the field.

**121 and 125 are local rings with nilpotents and non-regular 𝒥-classes, and they are as
sparse in the multiplicative basis as the field, with the same 4 key frequencies.**
Cyclicity of the unit group is what the clock mechanism needs — not primality, not
regularity.

**n=120 is a weak cell for basis analysis** and should not be used as a sparsity data point:
φ(120)=32 with factors (2,2,2,4), most characters are real, and the mult/add distinction
partly collapses (a planted *multiplicative* signal still reads Gini_add 0.76).

---

## 3.2 ❌ RETRACTED: activation variance is graded by 𝒥-class depth (C20)

> **RETRACTED 2026-09-11 by the pre-registered size-controlled re-test (k03, H1).** With
> block size controlled by equal-cell subsampling, the depth coefficient β₂+β₃ is negative
> in **5/5 seeds** at both 121 and 125 — the predicted sign every time — but its seed-mean
> sits **1.74 sd** and **1.50 sd** from zero, inside the pre-registered 2 sd band.
> β₂+β₃ = −70.25 ± 40.48 (n=121, 6 blocks / 2 df) and −14.16 ± 9.43 (n=125, 13 / 9 df).
> **A consistent sign without separation from zero is not a claim.**
>
> **H2, the counterexample, HELD in 5/5 seeds and is the positive result:** `J_11×J_11` and
> `J_1×J_121` at n=121 **both** compose to constant zero and have **equal cell counts**,
> yet differ in variance by **3.2×–12.3×**. "Constant-zero blocks are quiet" is false.
>
> **H3 (open question O6, non-additivity of the grading) held in 2/5 seeds — O6 is closed
> as single-seed noise.**
>
> The original objection, which the re-test confirms: The statistic below is **confounded
> by block cell count**: `|J_d| = φ(n/d)`, so 𝒥-class depth and block size are structurally
> coupled. ρ(cells, variance) = **+0.856..+0.949** at all six moduli, and a **flat-variance
> noise control** reproduces ρ = +0.942 at n=125 against +0.928 on real data *(midranks, P20,
> 2026-09-30; these read +0.85..+1.00, +0.918 and +0.929 under the ordinal ranks retired by the
> 2026-09-23 audit — the control now scores ABOVE the real data)* — i.e. the
> grading is what a model with *no* depth structure would also produce. The `0.000` entries
> are `np.var` of a **single cell**, zero by definition rather than by learning. The claim
> "constant-zero blocks are quiet" is **false** as stated: n=121 `J_11×J_11` is constant 0
> over 100 cells at variance 2.380, four times the equally-sized constant-zero
> `J_1×J_121` at 0.577.
>
> This is the same size confound that retracted C7b (§6.1) and, later, C19 (§2.2).
> **The tables below are kept for the record and must not be cited.**


Per-𝒥-class-block activation variance across inputs, averaged over all 512 neurons.
Rows and columns are the 𝒥-classes of the operands; the entry is how much the block's
activations *vary*, which separates real computation from saturation at a constant.

**n = 125 = 5³** (depth of `J_{5^k}` is k):

| | J_1 | J_5 | J_25 | J_125 |
|---|---|---|---|---|
| **J_1** | **6.306** | 3.764 | 2.559 | 0.418 |
| **J_5** | 3.766 | 2.057 | 1.549 | 0.186 |
| **J_25** | 2.563 | 1.549 | 1.283 | 0.057 |
| **J_125** | 0.420 | 0.186 | 0.057 | **0.000** |

**n = 121 = 11²**

| | J_1 | J_11 | J_121 |
|---|---|---|---|
| **J_1** | **4.847** | 3.488 | 0.577 |
| **J_11** | 3.487 | 2.380 | 0.224 |
| **J_121** | 0.576 | 0.224 | **0.000** |

**n = 113** (the degenerate field case): J_1×J_1 = 11.576, J_1×J_113 = 1.495, 0.000.

> **Variance decays monotonically down the nilpotency filtration.** Blocks whose product is
> a constant zero carry ~zero variance — the network has learned those inputs need no
> computation, and allocates representational capacity in proportion to the information a
> block actually carries.

Commutativity is a free correctness check and holds: 3.488 vs 3.487, 3.764 vs 3.766.

**Anomaly (O6):** at n=125, `(J_1,J_25)` = 2.559 > `(J_5,J_5)` = 2.057 despite equal total
depth. **The grading is not additive in depth** — which stratum an operand comes from
matters, not only where the product lands.

**Status: RETRACTED.** The size-controlled re-test it needed has now been run (k03, 5 seeds,
`src/analysis/jblocks.py`) and is reported in the box at the head of this section.

---

## 3.3 The clock is CAUSAL, not just descriptive — at every modulus (C22, C23)

Protocol invariant 5: *"every 'X is responsible' claim needs an ablation, not just a
sparsity number."* §3 measured sparsity. This measures responsibility.

**Method.** Logits over all unit pairs, relabelled into **exponent coordinates** — where
multiplication becomes addition — then a Fourier ablation in that basis (Nanda §4.4):
*restricted* keeps only the key characters, *excluded* removes exactly them. The control
that makes it evidence is a **random equal-sized character set** (20 draws): without it,
"removing these hurts" only says that removing energy hurts.

`analyze_n4.py`. Artifacts: `results/n4_ablation.npz` (k02 seed 0),
`results/k03_grid_acts_n4_ablation.npz` (**5 seeds**),
`results/k04_extended_n4_ablation.npz` (12 fresh moduli × 3 seeds) — all stamped.

> **⚠️ The separation numbers this section first reported have been RESTATED (2026-09-11).**
> `excluded / mean(random control)` over 20 draws is not a stable statistic: the control
> distribution is bimodal — ~73% of random character removals leave the loss untouched and
> the rest are catastrophic — so its **mean** is decided by which few catastrophic draws
> land. At n=121 seed 0 that ratio ranges over **21×–440× across control RNG seeds alone**,
> on identical data. It is replaced by a **permutation p**: the fraction of random
> equal-sized character sets at least as damaging as the key set, over **10,000** draws
> (**200** when this was written; raised by R1, 2026-09-21) — the
> same threshold-free logic C8 uses. The **conclusion is unchanged and strengthened**; only
> the numbers that expressed it have moved. Found by `test_generator_equivariance.py`.

**At 5 seeds (k03, which saved `logits_all` for every seed):**

| n | (Z/nZ)* | restricted / baseline | permutation p | p < 0.01 in |
|---|---|---|---|---|
| 113 | Z_112 | 55.5× ± 104.1 | **0.0000** | **5/5** |
| **121 = 11²** | Z_110 | 2.07× ± 1.26 | **0.0000** | **5/5** |
| **125 = 5³** | Z_100 | 2.10× ± 0.57 | **0.0000** | **5/5** |
| **119 = 7·17** | Z_6×Z_16 | **0.97× ± 0.74** | **0.0000** | **5/5** |
| **120** | Z_2³×Z_4 | 1.34× ± 0.21 | **0.0000** | **5/5** |
| **165** | Z_2×Z_4×Z_10 | 1.18× ± 0.60 | **0.0000** | **5/5** |

Not one of 200 random equal-sized character sets, at any of the six moduli in any of the
five seeds, is as damaging as the key set. The old statistic's `4/5` at n=120 was an
artefact of its own noise.

> **⚠️ AMENDED 2026-09-21 (R1), and the sentence above does not survive B = 10,000 intact.**
> At 50× the draws, **26 of the 31 runs** are still beyond *every* draw; the other five are
> beaten by **1 or 2 draws**, which a 200-draw null could not have resolved. Every run still
> reads **p̂ ≤ 0.0003** and **31/31 pass p < 0.01**, so the claim holds and the absolute
> phrasing does not. The pre-registration says in advance that `b > 0` at 10,000 draws is
> resolution, not falsification. F4's panel run, n=121 seed 0, is still at the floor:
> 0 of 10,000, excluded 12.8785. Asserted in `test_paper_numbers.py`.

> ⚠️ **CORRECTION 2026-09-16 — the column above is MISLABELLED, and it is a heavy-tailed
> MEAN.** Re-derived from the re-stamped artifact (`ccb590b`, `git_dirty=False`) by
> `test_paper_numbers.py`. Two defects, neither of which touches the claim:
>
> 1. **The column headed `restricted / baseline` holds `baseline / restricted`** — the
>    "×~better" direction. The values reproduce exactly (55.47 / 2.07 / 2.10 / 0.97 / 1.34 /
>    1.18), so only the *label* is wrong — but a drafter copying the header prints the ratio
>    upside down, and one nearly did.
> 2. **It is a mean over a heavy-tailed distribution — the same defect that retired the
>    control-mean statistic three paragraphs above.** At n=113 one seed of five reads
>    **264×** where the other four read 1.56–8.0, so "55.5× ± 104.1" is a summary whose sd is
>    twice its value and which is decided by one run. At n=165 one seed reads 7.09e-04.
>
> **The honest table — MEDIAN, with the range, and the statistic that actually carries C22:**
>
> | n | median baseline/restricted | range over 5 seeds | seeds where restricted improves | permutation p < 0.01 |
> |---|---|---|---|---|
> | 113 | **2.34×** | 1.56 – 264 | 5/5 | **5/5** |
> | 121 = 11² | **1.85×** | 0.349 – 4.22 | 4/5 | **5/5** |
> | 125 = 5³ | **1.98×** | 1.46 – 2.87 | 5/5 | **5/5** |
> | 119 = 7·17 | **0.67×** | 0.0146 – 1.96 | **2/5** | **5/5** |
> | 120 | **1.25×** | 1.18 – 1.75 | 5/5 | **5/5** |
> | 165 | **1.42×** | 7.09e-04 – 1.63 | 4/5 | **5/5** |
>
> Restricted improves on baseline in **25 of 30** runs. **C22/C23 are untouched** — the
> permutation p is 5/5 at every modulus and is not a ratio at all. n=119's 2/5 is the
> "necessary but not sufficient" pattern §3.3 already records, now with its seed count.
> **The paper reports the median and the range, never the mean.**

**Across k04's twelve fresh moduli** (3 seeds each): **p < 0.01 in 28 of 29** analysable
runs, including **81 = 3⁴ in 3/3** (the old statistic said 2/3), 105 and 147 **3/3** (both
2/3 before), and 169, 143, 99, 100, 75 all 3/3. The sole exception is **n=98**, 2/3, with one
seed at **p̂ = 0.0151** (150 of 10,000 draws).

> **⚠️ AMENDED 2026-09-21 (R1). This read "27 of 29" until the null was deepened to
> B = 10,000, and the number the amendment costs is not the count — it is a sentence.**
> `PREREGISTER_r1_permutation_B.md` named the two marginal tests before the re-run and
> committed to reporting whatever they became, *including 26 or 28*. It became **28**:
> **n = 49 = 7²** crossed the threshold downward, from `p = 0.010` (two draws of 200,
> landing exactly on it) to **p̂ = 0.0050** (49 of 10,000). n=98 did not move: 0.015 from
> 3 of 200, 0.0151 from 150 of 10,000. n=63 reads **0.0002** where the 200-draw null could
> only say "0 of 200".
>
> **The retired sentence is "there is no circuit there to find and the statistic correctly
> says so."** That run did not grok — window-median held-out accuracy **0.891**, baseline
> loss 1.3e-01, four orders worse than any grokked run here — and the permutation test
> passes on it anyway. This is the **same limit Gate 2 measures directly**: G1, the same
> statistic and the pre-registered PRIMARY criterion, passes on **20 of 20 FAILED**
> stratum-measurements (§8, F9) — and *that very run* is one of the 20, added to the arm
> on 2026-09-22 by the corrected endpoint rule (**was 19 of 19**). The permutation p is evidence that a circuit is
> load-bearing *where one exists*; it is **not** a detector of whether the network
> generalised. Paper §7.2 now says so explicitly; asserted in `test_paper_numbers.py`.

**The seed-0 table that C22 was first established on:**

| run | (Z/nZ)* | baseline | restricted | excluded | permutation p |
|---|---|---|---|---|---|
| 113 units-only | Z_112 | 4.77e-06 | **5.31e-07** (9.0× better) | 2.40e+01 | **0.0000** |
| 113 all pairs | Z_112 | 6.16e-06 | **7.71e-07** (8.0× better) | 2.43e+01 | **0.0000** |
| **121 = 11²** | Z_110 | 4.96e-07 | **2.10e-07** (2.4× better) | 1.29e+01 | **0.0000** |
| **125 = 5³** | Z_100 | 5.04e-06 | **1.76e-06** (2.9× better) | 1.85e+01 | **0.0000** |
| **119 = 7·17** | Z_6×Z_16 | 4.90e-06 | 9.95e-06 (2.0× worse) | 1.16e+01 | **0.0000** |
| **120** | Z_2³×Z_4 | 8.55e-03 | **7.22e-03** (1.2× better) | 7.71e+00 | **0.0000** |
| **165 = 3·5·11** | Z_2×Z_4×Z_10 | 5.48e-04 | **3.47e-04** (1.6× better) | 1.09e+01 | **0.0000** |

Keeping only the key characters **improves** the loss at six of seven runs; removing them
destroys it by 3–7 orders; and **no** random equal-sized character set out of 200 comes
close at any modulus (at 10,000 draws, 26 of 31 runs — see the amendment in §3.3). 113 improving 8.0× reproduces Gate 1's 7.7× on addition, so the test
is calibrated. The baseline, restricted and excluded columns are **exactly invariant** under
a change of generator (§3.6); only the key-frequency *labels* move, by k → t·k mod φ.

> **C22 — at prime powers (121, 125) the clock frequencies are causally responsible**, not
> merely a sparse description. This is what promotes §3 from a sparsity coincidence to a
> mechanism.

> **C23 (N4b) — the same holds where there is no discrete log at all.** 119, 120 and 165
> have **non-cyclic** unit groups, so no single generator and no discrete log exist. They
> do have an exponent coordinate: (Z/nZ)* = Z_o1 × … × Z_or, every unit is uniquely
> ∏ gᵢ^eᵢ, and multiplication is still **addition of exponent tuples**. A character is then
> a multi-index and the logit grid is a 2r-dimensional array. Ablating there separates the key
> set from the **median** control by **10.0 to 4.6e7×** over k03's 15 runs at 119/120/165 —
> **the product-character circuit is causal too**. *(P15, 2026-09-29: the 10.8–59.6× first
> quoted here was the retired excluded/mean(20-draw control) ratio.)*

**The key set recovered by ablation rank equals the key set by embedding norm at every one
of the seven runs** — two independent routes, same characters. At the non-cyclic moduli the
recovered set is close to one character per cyclic factor: 120 picks exactly the four
generators (1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1); 165 picks (1,0,0), (0,1,0) and two
in the Z_10 factor.

**Two honest caveats.**

1. **n=119 is the one run where restricted is worse than baseline** (2.0×) — the six key
   characters are *necessary* (excluded is catastrophic) but not *sufficient*. The model at
   119 uses structure the 5×-median detector does not recover. Both losses are ~1e-5, six
   orders below excluded, so this is a shortfall in the detector, not a failed circuit —
   but it is not the clean "restricted improves" signature the other six show. **New open
   question.**
2. ~~**1 seed.**~~ **Closed.** k03 saved `logits_all` for all five seeds and k04 for all
   three; the tables above are the multi-seed result. C22 and C23 now rest on 5 seeds at
   the six original moduli and 3 seeds at twelve more.

**A constraint earned from the planted-signal test, not from the data:** a key set that does
not **generate** the character group cannot be an exact clock — two units that every key
character maps to the same value cannot be separated no matter how the characters are
weighted. ([(0,2), (1,1)] over Z_4 × Z_6 tops out at 50% accuracy.) `analyze_n4.py` reports
it as a diagnostic; every real key set so far generates its full group.

### 3.3b The intervention acts INSIDE the network, not on the logit tensor (C39 — I1, 2026-09-22)

Everything in §3.3 FFTs the **saved logit grid**, masks it and inverts. That shows the
*output* carries character structure the loss depends on; it cannot separate "the network
computes with these characters" from "the network's output happens to be expressible in
them", because the mask is applied after the forward pass is over. The paper scoped its
language to match, and the reviewer panel accepted that scoping.

**I1 removes the scoping at n = 113.** It edits `W_E` in the multiplicative character basis
and runs the **unmodified forward pass**. Softmax attention and the ReLU MLP sit downstream
of the edit, so no part of the result is algebraically forced. Pre-registered in
`experiments/PREREGISTER_i1_internal_intervention.md` (`07ddcd0`, committed before the
retrain launched); all three thresholds are **inherited** — Gate 2's G2 (100×), `FAIL_ACC`
(0.90), C22/C23 (p < 0.01) — so none was chosen after seeing a number.

| seed | baseline | restricted (acc) | excluded (acc) | excluded/baseline | `p_perm` |
|---|---|---|---|---|---|
| 0 | 1.2102e-05 | 2.0775e-06 (**1.0000**) | 4.7136e+01 (**0.0089**) | 3.895e+06 | **1/10001 = 9.9990e-05** (0 of 10,000) |
| 1 | 1.2743e-07 | 3.2178e-07 (**1.0000**) | 3.5425e+01 (**0.0089**) | 2.780e+08 | **1/10001 = 9.9990e-05** (0 of 10,000) |
| 2 | 1.5480e-07 | 2.2235e-07 (**1.0000**) | 2.3883e+01 (**0.0091**) | 1.543e+08 | **5/10001 = 4.9995e-04** (4 of 10,000) |

`excluded/baseline` **median 1.543e+08, range 3.895e+06–2.780e+08, n = 3** — median, range
and n, never `±` at this sample size (C36). **Excluded accuracy 0.0089–0.0091 is chance**
(1/112 = 0.00893): deleting the key characters from the embedding does not degrade the
model, it removes it. I1 ✅ I2 ✅ I3 ✅ on **3 of 3** seeds, against a criterion of 2 of 3. The `p` is
Phipson–Smyth `(1+b)/(B+1)`, so seeds 0 and 1 sit **at the design floor** `1/10001`
and that is reported as the floor rather than rounded to `0.0001` — a four-decimal
display cannot distinguish "at the floor" from "just above it".

> **C39 — the key multiplicative characters are the representation the network COMPUTES
> with, not merely the basis its output is sparse in.** At n = 113, deleting them from
> `W_E` and running the network forward returns it to chance; keeping only them leaves
> accuracy at 1.0000.

**The checkpoints are the published models, bit-for-bit.** Every completed run in this
project had been unrecoverable — `run_n7.py` deletes the resume checkpoint on a clean finish
and `snapshot()` saved only `W_E`/`W_U`, so **0 of 232 artifacts carried the attention or MLP
weights** and no forward pass could be re-run. `snapshot()` now saves all nine matrices
(`cb3f941`), and the three n = 113 seeds were retrained at HEAD. They reproduce the archive
**exactly** — `max|ΔW_E| = max|ΔW_U| = max|Δlogits| = 0.000e+00`, grok steps 4,500 / 6,500 /
9,700 — so this is a claim about the models §3.3 and §3.7 already describe. The training path
was checked to be numerically inert between the archive's SHA `f9d15b8` and HEAD before the
retrain, not assumed.

**The projections were verified against the spectrum, not only against the loss.** Restricted
preserves key-bin amplitude exactly (197.016 → 197.016 at seed 0, `analyze_n7.mult_amplitude`) and zeroes the rest to
4.6e-15; excluded does the inverse at 2.3e-16. The key set holds **81.6 %** of embedding
amplitude in 8 of 56 bins, and the 5×-median detector recovers the same set on all three
seeds — C21's two-route agreement, on the internal arm. *(Re-derived on the protocol helper,
P19, 2026-09-30: this read 263.552, 1.6e-13, 2.3e-15 and **80.8 %** from a computation path no
longer in the repo. `test_paper_numbers.py` now pins every value here.)*

#### ⚠️ The null's upper tail REACHES the effect, and only the permutation p shows it

On seed 2 the control **maximum** (2.4464e+01) **exceeds** the excluded loss (2.3883e+01); on
seed 0 it reaches 99.2 % of it. `excluded / median(control)` reads 8.2e+07× and would have
hidden this completely — the same heavy-tail trap that retired C22/C23's control-mean ratio,
appearing in a new arm.

The cause is the design, and it is **conservative**: `random_character_sets` draws from
`all_freq_labels` **including the key characters**, exactly as `analyze_n4` does, so a draw
can partially reproduce the intervention. Keeping that pool was a pre-registered choice for
comparability with C22/C23, and it inflates `p`.

**EXPLORATORY (not in the pre-registered analysis plan; written after seeing the tail)** —
`run_intervention.py --summary`. Null damage is monotone **in the mean** in how many key
characters a draw happens to remove. ⚠️ **The within-level spread is wide** — F10's right
panel shows overlap-1 draws spanning four orders — so this is a statement about means and
not about individual draws; a single key character can be worth almost everything or
almost nothing. *Drawing the figure is what showed that; the word "clean" was in this
paragraph until the panel was rendered and looked at.*

| key characters removed | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| **0** | **1.26e-05** (n=2651) | **1.41e-07** (n=4892) | **1.64e-07** (n=4925) |
| 1 | 4.95e+00 | 4.58e+00 | 3.40e+00 |
| 2 | 1.03e+01 | 1.07e+01 | 7.44e+00 |
| 3 | 1.65e+01 | 1.79e+01 | 1.09e+01 |
| 4 | 2.39e+01 | 2.10e+01 | 1.49e+01 |
| 5 | 3.00e+01 | — | — |

A draw containing **no** key character does **nothing**: 1.04× / 1.11× / 1.06× baseline.
**12,468 zero-overlap draws across three seeds and not one exceeds 1.05e-04**, against an
excluded loss of 24–47. All four seed-2 draws that beat `excluded` carry overlap 3, and every
top-8 draw on every seed carries 3–5. Asserted in `--summary`: no zero-overlap draw reaches
`excluded`, on any seed. **The tail is not noise — it is draws that partially perform the
intervention.**

#### Two honest caveats

1. **`restricted` is worse than `baseline` on seeds 1 and 2** (2.5× and 1.4×) and better on
   seed 0 (5.8×). This is n = 119's logit-space signature (§3.3, caveat 1) appearing
   internally: the key characters are *necessary* and not perfectly *sufficient*. Accuracy is
   1.0000 in all three cases and both losses sit eight orders below excluded, so it is a
   shortfall in the 5×-median detector, not a failed circuit. The pre-registration declined
   to predict improvement in advance, precisely because the logit-space test partly builds
   that result in and this one cannot.
2. **One modulus, three seeds.** C39 is n = 113 only. §3.3's logit-space result remains the
   multi-modulus evidence, and the §5 scoping stands everywhere except here. Extending I1 to
   a composite modulus costs one retrain per seed (~5 h local CPU, zero quota) and is the
   obvious next step. **→ Done: §3.3c extends it to n = 121 (C40).**

### 3.3c …and it survives the ring structure: n = 121 = 11², non-square-free (C40 — I1b, 2026-09-23)

n = 113 is prime, so I1's clause "non-unit rows pass through untouched" covered one row. The
case this project is about — **non-square-free, with a non-regular stratum** — does not arise
at a prime. **I1b re-runs the identical instrument at n = 121 = 11²**: 110 units, **11**
non-unit rows passed through untouched (0 and ten nilpotents), three 𝒥-classes, and a
**non-regular class at d = 11**. The network keeps eleven intact embedding rows and a whole
non-regular stratum to work with after the edit, so this is a harder test than n = 113.
Pre-registered in `experiments/PREREGISTER_i1b_composite_intervention.md` (`712979d`,
committed before the retrain launched). **All three thresholds inherited verbatim from I1**,
and `run_intervention.py` / `src/analysis/intervention.py` **unmodified** — a criterion, so
one instrument produced both arms. Chosen over n = 125 because 125 seed 1 is the FAILED
negative control; 121 gives 3/3 grokked.

| seed | baseline | restricted (acc) | excluded (acc) | excluded/baseline | `p_perm` |
|---|---|---|---|---|---|
| 0 | 1.7482e-06 | 3.4061e-07 (**1.0000**) | 5.0365e+01 (**0.0000**) | 2.881e+07 | **1/10001** (0 of 10,000) |
| 1 | 7.7458e-03 | 1.0560e-01 (**0.9596**) | 5.7783e+01 (**0.0120**) | 7.460e+03 | **1/10001** (0 of 10,000) |
| 2 | 1.6364e-03 | 2.2674e-03 (**1.0000**) | 1.8467e+01 (**0.0000**) | 1.128e+04 | **3/10001** (2 of 10,000) |

`excluded/baseline` **median 1.128e+04, range 7.460e+03–2.881e+07, n = 3** (C36: never `±`).
I1 ✅ I2 ✅ I3 ✅ on **3 of 3** seeds against a pre-registered 2 of 3. Seed 1's restricted
accuracy 0.9596 clears I2's 0.90 by 0.06 — the smallest margin in either arm. Key sets
`|K|` = 6 / 4 / 4: s0 {2, 22, 40, 44, 48, 55}, s1 {3, 11, 18, 42}, s2 {22, 44, 48, 55}.

> **C40 — the internal character claim holds at a non-square-free modulus, with the
> non-unit rows left intact.** At n = 121, deleting the key characters from `W_E` and
> running the unmodified forward pass takes unit-grid accuracy to 0.0000 / 0.0120 / 0.0000;
> keeping only them leaves 1.0000 / 0.9596 / 1.0000. The falsifier the pre-registration
> named — `excluded/baseline < 100×`, which would have confined C39 to n = 113 — reads
> 7.46e+03× at worst.

**The checkpoints are the published models.** Retrained at HEAD, they reproduce
`results/_archive_i1/WE_engine_n121_s{0,1,2}.npz` with `max|ΔW_E| = 0.000e+00`, grok steps
10,700 / 10,300 / 5,000. The ~3.8e-06 logit gap the runner prints is float32 *storage* of the
archived tensor against a float64 recomputation. Artifacts stamped `266b77d`, `git_dirty=False`.

#### ⚠️ Excluded accuracy is BELOW chance on 2 of 3 seeds — descriptive only

Chance on the 110 × 110 unit grid is 1/110 = 0.00909 (**not** 1/121 — the grid is units only).
Excluded reads 0.0000 / 0.0120 / 0.0000 where n = 113 read chance almost exactly
(0.0089–0.0091). **No criterion reads accuracy against chance**, so this is not part of the
verdict; promoting it now would be the C6/C7 ordering error. One plausible reading — the
surviving characters plus the intact non-unit rows put mass on *specific* wrong residues — is
a hypothesis needing its own pre-registration.

#### ⚠️ The null's upper tail reaches the effect again, on seed 2

| seed | excluded | control max | max / excluded | excluded / median(control) |
|---|---|---|---|---|
| 0 | 50.3648 | 45.2470 | 89.8 % | 3.12e+06 |
| 1 | 57.7832 | 49.6228 | 85.9 % | 8.84e+03 |
| 2 | **18.4667** | **18.6887** | **101.2 %** | 1.09e+04 |

Same seed, same cause as at n = 113: the inherited pool includes the key characters, and
**both draws that beat `excluded` on seed 2 carry overlap 3 of |K| = 4**. The ratio to the
control median hides it; the permutation `p` (3/10001) does not.

**EXPLORATORY, not pre-registered** — mean null damage by key-character overlap:

| overlap | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| **0** | 1.35e-06 (n=4825) | 5.92e-03 (n=7304) | 1.70e-03 (n=7297) |
| 1 | 6.36e+00 | 1.13e+01 | 5.47e+00 |
| 2 | 1.37e+01 | 2.70e+01 | 9.28e+00 |
| 3 | 2.28e+01 | 2.96e+01 | 1.63e+01 |
| 4 | 2.82e+01 | — | — |

**19,426 zero-overlap draws and not one exceeds 1.71e-02**, against excluded 18–58. A
zero-overlap draw reads 0.77× / 0.76× / 1.04× baseline — two of three slightly *below*
baseline, where n = 113 read 1.04–1.11×. Small, descriptive, untested.

#### Caveats

1. **`restricted` is worse than `baseline` on seeds 1 and 2** (13.6× and 1.39×; 0.19× on
   seed 0). The necessary-not-sufficient signature of §3.3b, larger here. I2 scores accuracy,
   which holds.
2. **Magnitudes are about two orders below n = 113's at the floor** (7.46e+03 vs 3.90e+06).
   The pre-registration declined to predict that they would match; no claim is made from the
   comparison.
3. **Two moduli, three seeds each, engine float64.** §3.3's logit-space result remains the
   evidence at the other **21** of the paper's 23 moduli, and the §5 scoping stands there.
4. **`run_intervention.py --summary` pooled n = 113 and n = 121** (✅ fixed 2026-09-25: `--summary <n>` is now required) — it globbed one directory and
   printed `median 1.635e+07 … n = 6`. That is a cross-modulus import and **must never be
   quoted**. Every n = 121 figure here is from the three n = 121 artifacts alone, and each is
   asserted in `test_paper_numbers.py`.

### 3.4 The clock at three FRESH prime powers — and the one that never learned (k04 P4)

Pre-registered P4: Gini_mult at 49 = 7², 81 = 3⁴ and 169 = 13² within 0.10 of the k02
cyclic baseline 0.559, in ≥2 of 3 seeds.

| n | Gini_mult per seed | \|Δ\| vs 0.559 | within 0.10 | grokked |
|---|---|---|---|---|
| **81 = 3⁴** | 0.605, 0.583, 0.554 | 0.046, 0.024, 0.005 | **3/3** ✓ | 3/3 |
| **169 = 13²** | 0.582, 0.572, 0.667 | 0.023, 0.013, 0.108 | **2/3** ✓ | 3/3 |
| 49 = 7² | 0.186, 0.211, 0.444 | 0.373, 0.348, 0.115 | **0/3** ✗ | **0/3** |

**P4 is NOT HELD as written.** It is recorded that way deliberately.

**n=49 never grokked in any seed.** 2,401 pairs, 720 training examples at train_frac 0.30 —
well under critical dataset size (C14, §8.2). Its embedding has no clock to measure, and a
Gini_mult spread of 0.186 / 0.211 / 0.444 is what an ungrokked model looks like. The
pre-registered P5 exclusion rule was scoped to *P1's timing analysis* and nothing else, so
extending it to P4 now — after seeing which modulus failed — would be exactly the Gate 1
C6/C7 error (§7.2).

**Post-hoc, and labelled so:** at both prime powers that actually learned the task, C6
extends — including **81 = 3⁴, the deepest nilpotency filtration tested**. The clean test is
a re-run of 49 above critical dataset size.

---

### 3.5 ⚠️ Measurement audit: 40k steps is NOT converged at four moduli (k03 H4)

Criterion: < 5% drift in the final 10k steps, in ≥4 of 5 seeds.

| run | Gini drift | PR drift | seeds converged | verdict |
|---|---|---|---|---|
| Arm A n=113 | 0.4% | 1.0% | 1/1 | **CONVERGED** |
| n=113 | 0.9% | 2.3% | 5/5 | **CONVERGED** |
| n=121 | 1.1% | 2.8% | 4/5 | **CONVERGED** |
| **n=125** | 1.2% | **5.6%** | 2/5 | **STILL MOVING** |
| **n=119** | 1.0% | **5.9%** | 3/5 | **STILL MOVING** |
| **n=120** | 2.8% | **7.7%** | 3/5 | **STILL MOVING** |
| **n=165** | 1.0% | **4.8%** | 3/5 | **STILL MOVING** |

> **Every participation-ratio number reported at 125, 119, 120 or 165 is measured at a
> non-stationary point**, and cross-modulus PR comparisons involving them inherit that.
> ~~**Gini is stable** (< 3% drift everywhere) and is unaffected~~ — **this clause is
> CORRECTED below (2026-09-11); it was measured in the wrong convention.** H5 also held:
> Arm A's PR is flat at 40k (drift 1.0%, PR_mult 3.99), which is what closed O1's last
> loose end.

#### ⚠️ Correction, 2026-09-11 — the Gini-drift column was measured on ENERGY

Found while designing N7. `analyze_k03.spectral()`, the sole source of the Gini column
above, feeds **energy** into `gini`. LAB_PROTOCOL.md's protocol puts Gini on **amplitude**;
energy reads **0.902** where amplitude reads **0.539** on the very same run *(P13, 2026-09-29: the
checkpoint that gives 0.902 is k07 s0, and the amplitude protocol reads **0.543** on it)*. The PR column
is unaffected — **PR belongs on energy**, which is what it used, so C24's load-bearing
claim stands untouched. `analyze_k04`, `analyze_k05`, `analyze_k07` and `refcheck` all use
the correct convention, so **C6's numbers are sound**.

Re-measured on the same k03 trajectories in the amplitude convention, over the final 12k
steps:

| run | Gini(amplitude) trajectory | drift | Gini(energy) drift, as published above |
|---|---|---|---|
| 113 s=1 | 0.567 → 0.561 → 0.522 → 0.528 | **6.9%** | 1.2% |
| 121 s=2 | 0.564 → 0.556 → 0.492 → 0.504 | **10.7%** | 2.6% |
| 125 s=2 | 0.498 → 0.577 → 0.621 → 0.578 | **16.0%** | 3.6% |

Across 113/121/125 × 3 seeds the amplitude drift spans **0.3%–16.0%**, not "< 3%
everywhere". `gini(energy)` only *looked* stable because it sits at ≈0.90, where squaring
compresses relative change near the top of the range.

**This is oscillation about a flat mean, not a trend.** Full trajectories show Gini_mult
plateauing by ~16–20k and then jittering; n=125 s=2 wanders 0.498–0.621 with no direction.
So the correct reading is **not** "Gini has not converged" but **"Gini read at a single
checkpoint is a noisy estimator, sd ≈ 0.02–0.044."**

**Consequences, stated plainly:**

1. **C6 stands.** Its deltas (0.008 and 0.005) are an order of magnitude below the 0.10
   threshold under either estimator.
2. **But part of C6's reported across-seed spread (±0.026 to ±0.048) is estimator noise,
   not seed variance.** The two were never separated, and the published ± conflates them.
3. **Any future single-checkpoint Gini comparison is a ~2σ test at a 0.10 threshold**, and
   only ~1.6σ at n=125. N7 therefore defines Gini_mult as the **mean over the final 10k
   steps** and saves `we_traj` every 1,000 steps to supply it — see
   `experiments/PREREGISTER_n7_engine.md`, "Estimator noise".

This is the fourth time a statistic in this project has been read in the wrong convention
or off a confounded estimator. The general defence is the one that caught it again:
**measure the statistic's own noise floor before comparing two values of it.**

---

### 3.5b ❌ RETRACTED: "a network can de-grok" (C27). It is right-censoring.

**Dated correction, 2026-09-15.** `analyze_excursions.py results/k06_horizon`.

C27 read the **final logged sample** of k06's n=119 seed 0 — test acc 0.6798, down from
0.9989 two hundred steps earlier — as a converged state, and called it de-grokking. It is
a transient that the run ended in the middle of.

| | |
|---|---|
| post-grok excursions below acc 0.90, 11 grokked k06 runs | **19** |
| length of every one of them | **exactly 1 logging interval (200 steps)** |
| excursions that **recovered** | **18 of 19** |
| the 19th | starts at step **120,000** — the last sample. No next observation exists |
| the same run's earlier spike | step 115,400, acc **0.4869** — *worse*, and recovered in 200 steps |
| accuracy from step 32,000 to 119,800 | flat **0.9985–0.9992** (88,000 steps) |

**It is not double descent either.** Epoch-wise double descent requires a visible ascent;
there is none. Eighty-eight thousand steps of flat 0.999 accuracy end in a single-interval
discontinuity. There is no second descent because there was no first ascent.

**The circuit was never lost.** (Z/119)\* = Z6×Z16 is non-cyclic, so the multiplicative
readout is the product-character transform (`analyze_n7.mult_amplitude`), not `freq_energy`:

| step | \|W_E\| | Gini_mult | PR_mult |
|---|---|---|---|
| 116,000 | 12.824 | 0.5758 | 6.710 |
| 120,000 | 12.552 | **0.6297** *(sparser)* | 5.634 |

What broke is the **readout**, not the representation: final max logit **3.71** against
14–95 in every sibling run, and max softmax p 0.939 against 0.9996–1.0.

**A second, independent defect in the same claim.** The quoted "PR collapsing 19.17 → 12.56"
and "\|ΔGini\| = 0.193" are the **additive** column. `analyze_k06.spectra` returns
`Gini_mult = None` whenever the unit group is non-cyclic, and 119 = 7·17. Reproduced
exactly: PR_add 19.174@40k → 12.557@120k, Gini_add 0.2573 → 0.4499 (Δ = 0.1926). On a
multiplication run the additive column is the **control** basis (LAB_PROTOCOL.md), so C27 quoted a
control-basis statistic, read off a transient checkpoint, as evidence about a circuit.

**What survives as an observation — not a claim.** Post-grok transient instability is
ubiquitous under wd = 1.0, and its rate rises with horizon: **0.33 % of post-grok samples
at 120k steps against 0.09 % at 40k.** A mechanism is *conjectured only*: it is consistent
with the Softmax Collapse confirmed in §8.1 (with β₂ = 0.98 Adam's `v` has a ~34-step half
life, so one re-awakened gradient after a long dead zone produces a maximal-size step), but
200-step logging cannot see inside a sub-200-step event. Nothing here is measured.

**Consequences.**
1. **k06's R3 gets stronger**, not weaker: \|ΔGini\| is 0.008–0.048 in **12 of 12** runs,
   not 11 of 12. Amendment appended to `experiments/PREREGISTER_k06_horizon.md`.
2. ⚠️ **`results/k06_horizon/WE_B_thesis_n119_s0.npz` stores a transient final `W_E` and
   `logits_all`.** Do not quote a converged number from it. All nine results directories
   were swept: **it is the only censored artifact in the project.**
3. **O13 closes** ("why did n=119 s0 de-grok?" — it did not). **O19 opens**: the engine
   shows **0** excursions in 420,100 post-grok steps at *twice* the logging resolution,
   where torch shows 33 in 4,389,800 (expected ~3.2, P(0) ≈ 0.04). Suggestive, confounded
   by float64-vs-float32, not pre-registered — and cheap to settle inside O18.

**The transferable lesson.** A binary read of a single final sample cannot distinguish a
transient from a state — the same family as C26 (Gini at one checkpoint) and as the
grokked/near-grok/FAILED gate. **Define a run's endpoint as a window, never as its last
row**, and treat an excursion that touches the final sample as *censored*, not as an
outcome. `analyze_excursions.py` reports censoring explicitly for exactly this reason.

### 3.6 The readout does not depend on which primitive root we picked

Exponent coordinates need a generator, and `primitive_root(n)` silently returns the
**smallest** one. Every multiplicative-basis result in this document — C4, C6, C21, C22,
C23 — has used it without anyone checking whether the readout moves when that arbitrary
choice moves.

`test_generator_equivariance.py`. Replacing g by g^t with gcd(t, φ) = 1 relabels the same
group, and the consequences are exact, not approximate:

| quantity | under g → g^t | measured |
|---|---|---|
| exponent map | dlog_{g^t}(x) = t⁻¹·dlog_g(x) mod φ | exact, unit by unit, at 113/121/125 |
| Gini_mult, participation ratio | **invariant** (the spectrum is permuted) | to **1e-10** |
| baseline / restricted / excluded loss | **invariant** (the diagonal maps to the diagonal) | to **1e-10** |
| key frequencies | **k → t·k mod φ** — they move | exact set equality, every t tested |
| permutation p | invariant | identical under 5 generators |
| excluded / **median** control | invariant | 1.2% spread over 5 generators |

Non-cyclic unit groups have no single generator; the same holds per cyclic component, and
119, 120 and 165 pass under one-component and all-component twists.

> **C21 is not an artefact of one labelling.** "Key set by ablation rank equals key set by
> embedding norm" is a statement about two routes agreeing inside a coordinate system, and
> the coordinate system can be changed without moving either. Equally, **a key-frequency
> value is not a property of the model** — only the set, up to multiplication by a unit mod
> φ, is. Reporting "key freqs [11, 29, 33, 49] at n=121" is reporting a representative of
> an orbit, and it should be said that way.

This check is also what caught the control-mean defect boxed at the head of §3.3: the
generator variation in the old separation statistic (21×–72×) turned out to be *smaller*
than its variation across control RNG seeds alone (21×–440×).

---

### 3.7 The clock is in the NEURONS, not just the embedding (C31 — 5 seeds)

Pre-registered in `experiments/PREREGISTER_o4_neurons.md`, scored by `analyze_o4.py` on
activations k03 had already saved. **No training, no quota.** This closes open question O4
and removes the standing objection that C6 — the central result above — was measured only
on `W_E`.

Statistic: 2606.17399's and Nanda's. For each of the 512 MLP neurons, take the activation
grid h(a,b) over the units, reordered by discrete log; the share of its DC-free 2D spectral
energy that any **single** frequency's 8-bin family explains; a neuron is *tuned* if that
share exceeds **0.85**.

| n | tuned, per seed (dlog) | raw coords | cell-shuffled | distinct frequencies |
|---|---|---|---|---|
| 113 (field) | 94.9 · 91.8 · 100.0 · 99.8 · 98.0 % | **0.0 %** | **0.00 %** | 4–5 |
| **121 = 11²** | **74.0 · 79.9 · 89.3 · 73.2 · 78.9 %** | **0.0 %** | **0.00 %** | 4–5 |
| **125 = 5³** | **76.0 · 72.9 · 71.7 · 85.7 %** | **0.0 %** | **0.00 %** | 3–5 |

n=113 reads **94.9 %** against 2606.17399's published **96.9 %** — the method reproduces the
number it is borrowed from before it is pointed at the open question.

**The neurons sit on exactly the frequencies the embedding named.** In all 14 grokked runs
the tuned neurons' frequency set is the embedding's key set, element for element — n113 s0
[13, 27, 33, 43], n121 s0 [11, 29, 33, 49], n125 s0 [5, 25, 36, 40]: the same lists printed
in §3's table above, arrived at from activations instead of weights.

> The embedding spectrum and the MLP agree on which frequencies carry the circuit. C6 is
> not an artefact of looking only at `W_E`.

**The two controls are the point.** The same neurons in raw integer coordinates are **0.0 %**
tuned — the tuning is a property of the *multiplicative* coordinate, exactly as C6 says of
the embedding. And a per-neuron cell shuffle, which keeps the grid size and the activation
distribution but destroys 2D structure, reads **0.00 %** at every grid side (112, 110, 100).
That second control exists because the chance level of this statistic is eight bins out of
s² maximised over s/2 candidate frequencies, and therefore *moves with the order of the unit
group* — a bare percentage compared across 113, 121 and 125 would have been the fourth size
confound in this project.

**The matched-failure control (exploratory).** Runs that failed to generalise, same code and
hyperparameters:

| run | final test acc | tuned (dlog) | mean max-fraction |
|---|---|---|---|
| engine `n125_s1` | 0.6082 | **6.8 %** | 0.295 |
| k04 `n49` s0 / s1 | 0.146 / 0.195 | **0.0 %** | 0.24 / 0.29 |
| k04 `n54` s1 | 0.273 | **0.0 %** | 0.222 |

against 71.7–100 % for the grokked runs. Every sparsity result in this project had been
corroborated and never contrasted; this is the contrast, and it comes from artifacts that
were already on disk. `n63` is skipped loudly — (Z/63Z)* = Z₆×Z₆ has no discrete log.

**Stated honestly:** the prime powers do **not** match the prime. 72–89 % against 92–100 %
is a real gap, in the same direction at both 121 and 125, and it is reported as measured and
not explained. k03's `n125_s1` (0.9779, a near-grok, neither grokked nor failed) reads
74.8 % and is excluded from both arms rather than counted as a control — a model at 97.8 %
test accuracy has learned the circuit.


### 3.8 The clock runs on EVERY algebraic stratum, each on its own local group (C33 — Gate 2)

**This is the Gate 2 result.** Criteria pre-registered in
`experiments/PREREGISTER_gate2_strata.md`, committed `bca6236` **before a single
per-stratum number existed**; scored by `analyze_gate2.py`; artifact
`results/gate2/k03_grid_acts.json`, SHA `067a326`, `git_dirty=False`. Zero training,
zero Kaggle quota — re-analysis of `logits_all` and `mlp_acts` already on disk.

#### ⚠️ Prior work: the stratified framing is NOT ours, and the citation is load-bearing

**`2607.07066` (Chen et al., *Multiplication Beyond Groups: Stratified Fourier Mechanisms in
Transformer Circuits*, ICML 2026 Mechanistic Interpretability Workshop, arXiv 8 July 2026)
published strata, the "monoid extension", local characters and class-sensitive attention
routing — two months before Gate 2 was declared here (2026-09-14).** It is listed in
`STATE.md` §3 as one of the three papers naming this territory, but it was **not cited in
this section**, which is where the result is actually recorded. That was a gap in the
scientific record, not merely in the eventual write-up, and it is corrected here.

**What is theirs:** the stratified framing itself, on **four moduli — 113, 143, 154, 165, all
square-free** — with evidence they describe as *"only correlational"*. The square-free /
nilpotent distinction is already set out below under **"Novelty, stated exactly"**; this
block adds only what that one does not carry — their limitations **in their own words**, and
the mapping from each to the evidence here.

**What is ours, and it is exactly the two limitations they state for themselves:**

| their stated limitation (verbatim) | what this section supplies |
|---|---|
| *"our evidence is **only correlational**, meaning causation cannot be directly inferred. In future work, we plan to apply causal intervention techniques, such as **ablations** and activation patching"* | **G2**: excluded/baseline **3.98 × 10²–2.71 × 10⁸** over 157 stratum-measurements, plus C22/C23/C32's 200-draw permutation ablations at p < 0.01 |
| *"non-square-free moduli introduce **non-regular 𝒥-classes** that contain nilpotent elements, which **breaks the reduction to local group characters**. **Future work must investigate** how networks handle these algebraic 'one-way' collapses"* | **the reduction does not break.** 14 of our 23 moduli are non-square-free and **six** are prime powers (`MASTER_PLAN.md` §0.1, computed from `algebra.py`; the figures "13 of 18" and "five" stood here until 2026-09-15 and were **pre-k09**); `x = d·u` makes the 𝒥-class a torsor under the units, so the clock runs on G_m = (Z/(n/m)Z)\*. Nilpotents **collapse the stratum onto a smaller local group** rather than destroying the character structure |

Independent derivation is not a defence and is not claimed. **Any write-up of this section
must cite them as the origin of the stratified framing and position this as its causal,
non-square-free completion.**

#### The gap this closes

Every multiplicative statistic in §3 before this one — C6, C21, C22, C25, C31 — is computed
on the **unit** sub-grid. The table below is what that covers, and what the stratum analysis
now covers:

| n | unit stratum alone | measured here | local groups |
|---|---|---|---|
| 113 | 98.2 % | 98.2 % | Z₁₁₂ (there is no non-unit stratum) |
| 121 = 11² | 82.6 % | **97.7 %** | Z₁₁₀ ⊳ Z₁₀ |
| 125 = 5³ | 64.0 % | **89.6 %** | Z₁₀₀ ⊳ Z₂₀ |
| 119 = 7·17 | 65.1 % | **88.6 %** | Z₆×Z₁₆, Z₁₆ |
| 120 | 7.1 % | **23.1 %** | Z₂³×Z₄, Z₂²×Z₄ |
| 165 = 3·5·11 | 23.5 % | **82.3 %** | Z₂×Z₄×Z₁₀, Z₄×Z₁₀, Z₂×Z₁₀, Z₁₀ |

#### The algebra it rests on (a theorem, not a hunch)

For x ∈ J_d write x = d·u with u a unit mod n/d — the 𝒥-class is a torsor under the units,
which is C7's "action coordinate" stated exactly. Likewise y = e·v. With m = gcd(de, n) and
de = m·w, gcd(w, n/m) = 1:

```
x · y  mod n   =   m · ( w · (u·v)  mod  n/m )
```

Three consequences, and the whole measurement is built on them:

1. a stratum is a fixed **unit relabelling of multiplication in the smaller ring**
   Z/(n/m)Z, on the local group G_m = (Z/(n/m)Z)\*;
2. the target is **exactly constant on the fibres** of (Z/(n/d)Z)\* ↠ G_m — asserted at
   every stratum of all six moduli by `analyze_gate2._selfcheck`, not assumed;
3. multiplying units is **adding exponent tuples**, so a local clock must put its logit
   energy on the **diagonal (κ, κ)** in local-group coordinates.

`analyze_gate2.py` re-indexes each block by discrete log on both axes, fibre-collapses onto
G_m, and scores G0–G5 on the collapsed grid. **No key-frequency detector is used anywhere**:
the ablated set is the *whole* diagonal, fixed by the algebra. Choosing the set by ablation
rank and then asking a permutation test whether that set is unusually damaging is circular,
and this avoids it by construction.

#### Result — 32 measurable strata, 6 moduli, 157 stratum-run measurements

Full table: `PYTHONPATH=. .venv/bin/python scripts/gate2_table.py`. Summary of every
criterion, over **26 non-unit strata** (plus the 6 unit strata as positive controls):

| # | criterion | threshold | observed | verdict |
|---|---|---|---|---|
| **G0** | held-out accuracy per block | ≥ 0.95 | **1.0000 ± 0.0000** at every measurable stratum, every seed | **PASS** |
| **G1** | permutation p, diagonal vs **10,000** random equal-cardinality bin sets | p < 0.01 in ≥ 4/5 seeds | **p̂ ≤ 0.0023 in all 157**; beyond *every* draw in **133/157** ⚠️ *amended 2026-09-21 (R1) from "p = 0.0000 in all 157", measured at 200 draws; 23,710 B-independent values in this artifact are unchanged* | **PASS** |
| **G2** | excluded loss / baseline | ≥ 100× | **3.98 × 10²** (min) to **2.71 × 10⁸** ⚠️ *corrected 2026-09-16 from "1.0 × 10⁸"; re-derived from `results/gate2/k03_grid_acts.json` by `test_paper_numbers.py`. The measured max is LARGER than the figure published here, which is the flattering direction, so it is recorded as a correction rather than adopted silently* | **PASS** |
| **G3** | fibre residual vs permuted-fibre control | ≤ 0.5 × control | real **0.000–0.132**, control **0.396–0.929**; worst ratio **0.161** | **PASS** |
| **G4** | diagonal energy share D, vs exact chance 1/(\|G_m\|+1) | R ≥ 5.0 | D = **0.837–0.997**, i.e. **84–100 % of the ceiling** at every stratum | **PASS** |
| **G5** | tuned neurons − per-stratum shuffle control | ≥ 0.20 | **0.193–1.000** against a shuffle control of **0.0000** everywhere | **PASS** (see caveat) |
| **G6** | stratum sets differ by the algebra of n | chain vs lattice | 121 and 125 **nested chains**; 119, 120, 165 **non-chain lattices**; 113 a **single** stratum | **PASS** |

Twelve distinct local groups appear — Z₁₀, Z₁₆, Z₂₀, Z₁₀₀, Z₁₁₀, Z₁₁₂, Z₂×Z₁₀, Z₄×Z₁₀,
Z₂²×Z₄, Z₂×Z₄×Z₁₀, Z₂³×Z₄, Z₆×Z₁₆ — and every one carries the same circuit.

#### GATE 2: DECLARED, on a fourth branch the gate does not contain

The verdict rule was fixed in §3.1 of the pre-registration, before the data existed:

> **The local mechanism is invariant; the stratification is not.** On every stratum large
> enough to carry one, the network runs **the same character-based clock** — but on that
> stratum's own local group G_m. What the algebra of n changes is **how many strata there
> are and how they nest**: one for a field, a nested **p-adic chain** for a prime power, a
> flat **CRT divisor lattice** for a square-free composite.

Both of the gate's first two branches were ticked by our evidence because the question
conflates two levels. Branch (c) — "composite moduli don't grok" — has been dead since C3.

**Scope.** Declared for the **k03 PyTorch float32 arm, 5 seeds** (4 at n=125 — the
`n125_s1` near-grok at 0.9779 is excluded by the pre-registration, per LAB_PROTOCOL.md's
grokked / near-grok / FAILED rule). The engine arm is a separate replication and is
**never pooled** — and the reason is now *known* rather than suspected. ✅ **O18 CLOSED
2026-09-15 (C34): the engine/torch difference is TRAINING PRECISION, acting on PyTorch.**
At n=113 the same torch code reads Gini_mult **0.5791 at float32** and **0.7728 at float64**
(`f = +1.110`; this line read 0.7736 and +1.114 until 2026-09-28, see the O18 correction below), while the engine is dtype-invariant. **The "may be running different
algorithms" worry is retired** — float64 moves torch's neuron tuning 0.938 → 0.770 *and* its
Gini 0.579 → 0.774, i.e. both axes onto the engine's joint position, which is what a single
convergence axis looks like. **Still never pool the arms**: they sit in different precision
regimes, so their absolute numbers are not comparable. C33 is a **within-arm** result and is
untouched.

#### The pre-registered falsifier FIRED, and it names the weakest criterion

§4 of the pre-registration named its own falsifier: *"The negative-control arm passing G1.
A failed run must not show a local clock. If `n125_s1` (acc 0.6082) reads the same as a
grokked run, the statistic is measuring the grid, not the model."* **It fired.**
`scripts/gate2_discriminate.py`, over 183 grokked and **20** FAILED stratum-measurements:

| criterion | grokked pass | **FAILED** pass | separates? |
|---|---|---|---|
| **G0** held-out acc ≥ 0.95 | 183/183 **100 %** | 1/20 **5 %** | **YES** |
| **G1** perm p < 0.01 — *pre-registered PRIMARY* | 182/183 99 % | **20/20 100 %** | **NO** |
| **G2** excluded/baseline ≥ 100× | 183/183 **100 %** | 1/20 **5 %** | **YES** |
| **G3** fibre resid ≤ 0.5 × control | 122/122 100 % | 2/6 33 % | partly |
| **G4** R ≥ 5 | 183/183 100 % | 13/20 65 % | partly |
| **G5** tuned − shuffled ≥ 0.20 | 109/111 98 % | 1/14 **7 %** | **YES** |

> **⚠️ AMENDED 2026-09-22. This table read `0/19 · 19/19 · 0/19 · 2/6 · 12/19 · 0/13`
> until the corrected endpoint rule admitted one run to the control arm.** Classifying by
> the median of the final ten samples rather than by the last row moves
> `k04_extended/WE_B_thesis_n49_s2` from near-grok (last row **0.9631**) to FAILED (window
> median **0.8908**). **Four of the six rows move, and every one of them moves because of
> that single run**: its one measurable stratum — the unit stratum, the other three being
> below the |G_m| ≥ 10 floor — passes G0, G1, G2, G4 **and** G5.
>
> **So "G0, G2 and G5 separate completely" is retired; they separate at 1/20, 1/20 and
> 1/14.** The run is a genuine borderline: its window median clears the FAILED gate of
> 0.90 by **0.009**, and its final ten samples rise monotonically 0.842 → 0.963, so it was
> still learning when the budget ended rather than plateaued. Verified by planting the
> threshold: **FAIL_ACC 0.89 reclassifies it near-grok and restores 0/19; 0.91 leaves it
> FAILED.** It is the same run §7.2 above already describes (baseline loss 0.1256, window
> median 0.891), so the two sections agree — this is that run entering the Gate 2 control
> arm formally. ⚠️ **The FAILED-arm thresholds are now load-bearing on a 0.009 margin and
> are a researcher decision.**
>
> **DECIDED 2026-09-22 by the researcher: `FAIL_ACC` stays at 0.90, so the numbers above
> stand.** The 0.009 margin is worth knowing, but the threshold is **inherited, not tuned to
> this result**: 0.90 is the excursion floor adopted 2026-09-15 when C27 was retracted
> (`932ef89`), centralised into `src/analysis/runs.py` on 09-21 (`37dba23`), and this arm was
> not scored until 09-22 (`cbe13f3`) — **a week after the gate was fixed.** Moving it now,
> having seen which side the run falls on, would be the C6/C7 ordering error the project
> committed once and does not repeat.

**Why G1 fails to separate, plainly: 30 % of every block is training data.** A model that
has merely *memorised* its training cells still has logits that are a function of u·v
*there*, so the diagonal still carries the energy and removing it still hurts more than
removing a random bin set. G1 tests "the logits are a function of the product", and
memorisation satisfies that. It is a **necessary condition, not evidence** — and it was
named PRIMARY on the strength of being the project's established statistic (C22, C23, C32),
which is precisely the kind of inherited authority that should have been checked.

**The verdict stands**, because "carries a local clock" was defined in advance as
**G1 ∧ G2**, and G2 separates perfectly, as do G0 and G5. But **its weight rests on
G0 ∧ G2 ∧ G5**, and quoting "all six criteria passed" would misrepresent it.

The corrected control arm returns **NO VERDICT** — the failed runs do not generalise, so
the mechanism question is unanswerable there, which is the right behaviour for a control.
(Fixing it also exposed a binary-gate defect of exactly the kind LAB_PROTOCOL.md warns about: the
first control arm conflated FAILED runs with **near-groks** — k04 n=75 at 0.9855, n=99 at
0.9794, n=100 at 0.9880, engine n119_s0 at 0.9869 — and those "declared the fourth branch",
which is what a model that has learned the circuit is supposed to do. Three states now:
grokked > 0.99, FAILED < 0.90, near-grok in neither arm.)

#### Replication on the from-scratch engine (float64, 3 seeds) — independent of O18

`results/gate2/n7_engine.json`. The **same verdict**, at 3 moduli (119, 121, 125), 2–3
seeds each: every measurable non-unit stratum carries a local clock. This matters for the
standing objection that O18 puts Gate 2 in doubt — the two implementations disagree about
**sparsity magnitude** (Gini_mult 0.75 vs 0.57) and still agree about **the stratified
clock**. One miss, reported: the n=119 **unit** stratum fails G1 in 1 of 2 seeds, which is
the same place N7's E3 missed (`n119_s2`, p = 0.115) and where O8/O11 already live.

#### What is NOT claimed

- **G1, G2 and G4 are partly entailed by correctness, and must not be quoted as the
  evidence.** Any function on a stratum that is *correct* is a function of
  dlog(u) + dlog(v), so a perfectly confident **lookup table** already reads D = 1.000,
  p = 0.0000 and fibre residual 0 — `analyze_gate2._selfcheck` plants exactly that and
  measures it. What these criteria do establish is not "the model is correct" but
  "**the logit tensor depends on (u, v) essentially only through u·v mod n/m**", i.e. the
  computation has collapsed to the local ring. `lut_r2` = **0.012–0.074** says the logits
  are very far from a scaled one-hot, so that collapse is a real statement about the tensor
  — but a smooth function of u·v would also read D ≈ 1, so these criteria cannot by
  themselves say the mechanism is a **character** circuit.
- **G5 is the endpoint that says "clock".** Single-frequency MLP neurons in local-group
  coordinates are entailed by nothing, and the shuffle control reads **0.0000 at every
  stratum**. Note `freq_fraction` is cyclic-only, so G5 as pre-registered covers 121, 125,
  119's sub-strata and 165's Z₁₀ strata. The non-cyclic strata are covered by
  `freq_fraction_nd` (product characters, **0.182–0.877** against shuffle 0.0000) which is
  **EXPLORATORY — added after the pre-registration** and never scored as G5.
- **The model does NOT solve every stratum.** LAB_NOTEBOOK Entry 30 recorded
  "every stratum at acc 1.0000"; that was the **full grid, 30 % of which is training data**.
  On held-out cells, **15 stratum-runs fall to 0.379–0.833** (13 strata, seed means 0.500–0.833;
  P10, 2026-09-29). Every one of them is below the
  measurability cut, so none affects the verdict — but the claim as written in Entry 30 is
  corrected here.
- **That degradation is not evidence about the algebra.** Every failing stratum has
  |G_m| ≤ 8 **and** ≤ 64 cells, and those are structurally coupled, so it cannot separate
  "the local group is too small to carry a clock" from "too few training examples". It is
  consistent with critical dataset size (C14). It is reported because it is *why* the
  measurability cut exists.
- **113 contributes nothing to the stratum claim.** A field has one stratum. It is the
  positive control, and that is all.

#### Novelty, stated exactly

Chen et al. **2607.07066** show GCR "localizes to disjoint algebraic strata that partition
the input space". **Their Table 1 is 113, 143, 154, 165 and marks every one
`Square-free: True`.** A p-adic filtration with **nilpotent** non-units — n = 121, where
J₁₁·J₁₁ ≡ 0 exactly, and n = 125, where it does not — is the complement of their design and
the case they name as open. Chughtai, Chan & Nanda **2302.03025** argue universality is
two-level: the circuit *family* is invariant, the precise circuit arbitrary. The fourth
branch is that shape made algebraic — the family is the character clock, the index set is
set by the stratum.

---

### 3.9 It is the SAME clock, not one clock per stratum (C38 — R2, 2026-09-21)

**This answers the peer-review panel's one finding that could not be answered by rewriting.**
Pre-registered in `experiments/PREREGISTER_r2_cross_stratum.md`, committed `373a085`
**before any cross-stratum number existed**; scored by `analyze_r2_sameness.py`; artifact
`results/gate2/r2_sameness_k03_grid_acts.json`, `git_dirty=False`. Zero training, zero
Kaggle quota — re-analysis of `logits_all` already on disk.

**The objection.** Every Gate 2 statistic is computed *inside* one stratum, so "the same
clock on every stratum" was inferred from "a clock here, and a clock there" — which is also
what a model routing each stratum to a **private sub-circuit** would produce. The instrument
existed at `K_h = K_W`, 14/14, on the unit stratum (C31) and had never been lifted between
strata.

**The measurement.** For each measurable stratum, collapse the logit block onto its local
group, read the diagonal amplitude spectrum, and take the key set with the project's
existing detector (`key_freqs_5x_median`, the one that produces `K_W`). Compare strata whose
local groups are **identical**.

| | statistic | observed | pre-registered |
|---|---|---|---|
| **R2-1** | key sets **exactly** equal, primary pairs | **158/160 = 0.9875** | ≥ 0.80 |
| **R2-2** | Monte-Carlo null, key sets redrawn at observed sizes | **1.0e-4** (the floor at B=10,000) | < 0.01 |
| **R2-3** | cell-shuffle control | **0/160** | < 0.30 |
| R2-4 | failed-run control | **2/2 = 1.00** | < 0.50 — **FAILED** |
| R2-5 | transpose pairs | 54/54 | descriptive only |
| R2-6 | empty key sets excluded | **0 of 160** excluded; median \|K\| = 2 | — |

Mean Jaccard **0.9982**. Both disagreements are in **n=165 seed 1** (26/28); the other four
seeds read 28/28.

**⚠️ TRANSPOSES ARE NOT SUPPORT AND ARE NOT COUNTED.** `(d,e)` and `(e,d)` are transposes of
one **commutative** task, so their diagonal spectra agree for a reason that has nothing to do
with a shared circuit. Excluding them leaves **28 primary pairs at n=165, 2 at 120, 2 at 119
and NONE at 121 or 125** — whose only same-group pair is a transpose. n=165's `G_11`, eight
strata on one local group, carries the result. **My own pre-registration's table said 30+4 at
n=165 and the algebra says 28+6**; corrected in that file's Outcome section and asserted in
the script's self-check.

**⚠️ THE FAILED-RUN CONTROL DID NOT WORK, AND THE CLAIM SHIPS SAYING SO.** There are exactly
**two** scorable failed-run pairs in the whole project and both come from **one run at test
accuracy 0.7502** — a model that has learned most of the task, which the three-state rule
buckets with the 0.146 memorisers. Its own null reads **p = 1.0**: two pairs cannot
discriminate. Every genuinely failed run either has no comparable pair by algebra (49, 63,
125) or yields **no key set at all** (54 seed 1, accuracy 0.2733). **Underpowered, not
passing** — and the same shape as G1, which passes on 100% of failed stratum measurements.
What does separate, **post-hoc and labelled**: a key set *exists* in 160/160 grokked primary
strata against 2/4 failed ones, and the engine's failed n=125 seed 1 does not agree even with
its own transpose (0/1) where every grokked run agrees 54/54.

**Two self-check failures changed the instrument before it saw real data** (both recorded in
`analyze_r2_sameness.py`): a *perfectly* pure planted spectrum has median 0, so "above 5×
the median" selects float noise; and **two empty key sets compare equal**, so an undetected
clock would have scored as a perfect match.

---

## 4. Replication of 2606.17399 (VERIFIED against published numbers)

`a·b mod 113`, their exact protocol, units only:

**COMPLETE replication (corrected, Entry 16).** Arm A, 40k epochs, grok step 6,400,
acc 1.0000:

| metric | published | ours |
|---|---|---|
| Gini additive | 0.071 | **0.060** |
| participation ratio, additive | 52.7 | **53.31** |
| Gini multiplicative | 0.579 | **0.543** |
| key frequencies | 4 — {2,8,47,56} | 4 — {13,27,33,43} |
| **participation ratio, multiplicative** | **4.1** | **4.31** |

Frequency *values* are seed-dependent; the count is the invariant. Primitive root g=3 in both.

**Convention, and it is load-bearing:** Gini is on **amplitude** `√(‖s_k‖²+‖c_k‖²)`; the
participation ratio is on the normalised **energy** `p_k = |A_k|²/Σ|A_k|²`, its standard
definition. Feeding amplitude into the participation ratio gives **12.5** where the correct
value is 4.31 — that single line was open question O1 for three sessions (§4.0).

### 4.0 ✅ CLOSED: O1, the "IPR gap", was our own convention bug

For three sessions the replication had one stubborn disagreement — IPR_mult 12.5 against a
published 4.1, with every other metric matching — and two plausible scientific explanations
on the table (their `W_E` centring, or a real circuit difference). Neither was the cause.

`analyze_k02.py` fed **amplitude** into `participation_ratio`. The published 4.1 is the
**energy** convention, which is the standard definition. On energy the same checkpoint
reads **4.31**. `refcheck.py` and `analyze_scout.py` had always used energy and had always
matched — PR 4.85 on **their own** n=165 checkpoint against 4.76 on our independent scout
(§4.1), which is the cross-check that located the bug. *(Corrected 2026-09-29: this sentence
called 4.76 "their published" value; it is our scout's, per the §4.1 table and Entry 5.)*

**The pre-registered Arm A prediction therefore HELD** (IPR_mult < 6 at 40k epochs); Entry
14's FAILED verdict is superseded. The "do not cite our IPR" instruction is **lifted**.

**Why it survived three sessions:** the additive column, which we used as the sanity check,
**cannot discriminate between the two conventions** — a flat spectrum gives 55.96 on one and
55.86 on the other. The control we trusted was structurally blind to the error.

**Three "IPR"s exist in this literature.** 2606.17399 uses energy-normalised participation
ratio; `2406.03495` (Doshi/Gromov) uses a **reciprocal** convention `(‖u‖₂ᵣ/‖u‖₂)^2ʳ`,
bounded in [0,1] and *increasing* with sparsity. Always state which. (Protocol invariant 2.)

### 4.1 Pipeline validated against third-party weights (VERIFIED)

Our analysis code run on **2607.07066's own published checkpoint**
(`P165_d128_h4_mlp512_s1.pt`, seed 1, final loss 1.96e-05):

| quantity | their checkpoint | our independent scout |
|---|---|---|
| Gini multiplicative | **0.637** | **0.629** |
| participation ratio | **4.85** | **4.76** |

⚠️ *(2026-09-29, P6, fixed at the source.)* Until today this pair read **0.884 / 0.885**,
Gini on the **energy** spectrum (`refcheck.py` and `analyze_scout.py` called
`gini(energy(...))`). Both scripts now use the protocol helpers
(`analyze_n7.mult_amplitude`: amplitude, DC dropped, one bin per conjugate pair) and read
**0.637 / 0.629**. WP3.2's interim "amplitude" pair **0.626 / 0.622** was `gini(√energy())`,
which keeps DC and scales each self-conjugate bin by 1 where the others are scaled by 2 (n = 165
has several order-2 characters); it is not the protocol either. The agreement stands (one
pipeline on both sides).

Two independently trained models, two codebases, the same numbers. Their 𝒥-class claim also
replicates: embedding separation ratio **1.03 vs 0.63** for a shuffled-label control — the
absolute separation is modest and the control is what makes it meaningful.

---

## 5. The additive basis carries the CRT stratification (**9 fresh moduli, 3/3 seeds each**)

### 5.1 Key additive frequencies are the CRT-component duals (C8)

> The key additive frequencies of a grokked `a·b mod n` transformer are the multiples of
> `n/q`, one family per **maximal prime power** `q ‖ n` — the additive duals of the CRT
> components.

> **✅ k04 pre-registered P3: HELD at 9 of 9 CRT-testable moduli, 3 of 3 seeds each.**
> Threshold-free permutation enrichment (20,000 permutations), every one p < 0.01:
>
> | n | 54 | 63 | 75 | 98 | 99 | 100 | 105 | 143 | 147 |
> |---|---|---|---|---|---|---|---|---|---|
> | enrichment (3 seeds) | 2.7/1.5/6.1× | 4.7/1.6/1.6× | 3.7/7.6/3.3× | 7.6/6.8/8.3× | 5.9/4.2/7.3× | 3.7/6.9/5.5× | 11.8/14.4/11.0× | 10.6/6.3/3.4× | 8.6/5.1/4.1× |
> | seeds p<0.01 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
>
> Predicted ≥7 of 9. The moduli were fixed in a committed pre-registration **before any of
> them was run**, and **10 of the 12 are non-square-free** — the territory all three cluster
> papers stop short of. Prime powers (49, 81, 169) stay excluded as **vacuous by
> construction**: with one CRT component `n/q = 1`, so the predicted set is every frequency.
>
> This is the best-supported claim in the project: 12 moduli, 3–5 seeds each, a threshold-free
> test, and an exact-set-equality check against a **third-party published checkpoint** (§4.1).

Mechanism: a function supported on multiples of `d` has additive Fourier support on
multiples of `n/d`; 𝒥-classes are unions of arithmetic progressions generated by the CRT
components. The raw 𝒥-class indicators, with **no network involved**, are sparse in the
additive basis: `Gini_add(indicators)` = 0.406 (n=121), 0.450 (n=165), 0.400 (n=120),
**0.000 (n=113)** — a field has no stratification to encode.

> ⚠️ *(2026-09-29, P6b, fixed at the source.)* Until today these four read
> **0.829 / 0.829 / 0.683 / 0.018**, Gini on the **energy** spectrum with the DC bin included
> (`analyze_scout.py`: `gini(energy(additive_dft(ind)))`). The script now uses
> `analyze_n7.add_amplitude` (amplitude, DC dropped) and reads **0.406 / 0.450 / 0.400 / 0.000**.
> WP3.6's interim **0.416 / 0.457 / 0.395 / 0.018** was `gini(√energy())`, DC kept, not the
> protocol. n = 113 is exactly 0 because the two field indicators ({0} and the units) are a
> centred delta, whose DFT is flat. The contrast with the field is unchanged. Asserted in
> `test_paper_numbers`.

> **⚠️ AMENDED 2026-09-21 (R3). "Confirmed on raw indicators" was too strong, and a Gini was
> never the claim.** `experiments/PREREGISTER_r3_indicator_setequality.md` asked the
> indicators the question the section actually makes — a **set** — and the answer is that
> they are **enriched** on `P(n)` (4.87× at n=165, 3.32× at 120, 7.15× at 119, every one
> beyond all 20,000 permutations) but **do not select it**: their key set equals `P(n)` at
> only **2 of 10** composite moduli. At n=165 they hold 3 of the 8 predicted frequencies; at
> n=120 they pick **{20, 40, 60}**, and **20 = 120/6 is the dual of a divisor that is not a
> maximal prime power**, so it is outside `P(n)` *and* outside the prime variant. The reason
> is structural and the reviewer named it: `1_{J_d}` transforms to the Ramanujan sum
> `c_q(k) = μ(q/g)φ(q)/φ(q/g)` (`q = n/d`, `g = gcd(k,q)`), so the stratification's energy
> **concentrates** on the frequencies sharing a factor with n — the multiples of the duals of
> **every** divisor, **42 frequencies at n=165 carrying 30.5× the mean energy of the other 40,
> against |P(165)| = 8**. ⚠️ *Corrected 2026-09-28: this said "support", which is false — the
> indicators' energy is nonzero at all 82 frequencies at n=165 (Ramanujan sums do not vanish
> at square-free q). Found while writing the Lean statements; the conclusion below is
> unaffected, since it rests on what the detector selects, not on support.*
>
> **The contrast is the finding, and it runs in the paper's favour.** Same detector, same bin
> convention, one run: the **network's** key set is all 8 of `P(165)` and all 7 of `P(120)`,
> no misses and no extras. **Selecting the maximal-prime-power subfamily out of the divisor
> duals is something the trained network does, not something the algebra hands it.** C8 is
> unchanged — it was always a claim about trained models. What is retired is the sentence
> that read the indicators as confirming it.
>
> The enrichment numbers here are **exploratory** (computed after the pre-registered
> set-equality test failed); the network-vs-indicator contrast is not. R3-5's positive
> control ran first and reproduced the published 17.44 (n=165) and 3.27 (n=119) exactly.

**Exact set equality**, no misses and no extras:

| n | predicted = observed | source |
|---|---|---|
| 165 | {15,30,33,45,55,60,66,75} | **their published checkpoint** |
| 165 | {15,30,33,45,55,60,66,75} | our scout, independent training |
| 120 | {15,24,30,40,45,48,60} | our scout |

n=120 is the sharper test: generators are `120/8`, `120/3`, `120/5` — **maximal prime
powers, not primes.** Using `n/p` predicts {24,40,48,60} and misses {15,30,45}.

**5-seed confirmation** (k02 criterion B4, permutation test, p<0.01 required in ≥4 of 5):

| n | enrichment per seed | seeds with p<0.01 |
|---|---|---|
| 165 | 17.4×, 13.3×, 11.2×, 11.2×, 11.8× | **5/5** |
| 120 | 13.8×, 11.4×, 9.8×, 7.6×, 13.3× | **5/5** |
| 119 | 3.3×, 11.1×, 5.2×, 7.9×, 6.4× | **5/5** |

**HELD at every applicable modulus.** 113/121/125 are prime powers — vacuous by
construction, never counted as support.

**Original single-seed measurement** (20,000-permutation test, kept for the record):

| n | enrichment | null | p |
|---|---|---|---|
| 165 (ours) | **14.08×** | 1.05 | < 5e-5 |
| 165 (**their checkpoint**) | **13.79×** | 1.04 | < 5e-5 |
| 120 | **11.76×** | 1.06 | < 5e-5 |
| **119** | **3.30×** | 1.01 | **< 5e-5** |

**n=119 was enriched all along** — invisible to the threshold, not absent (resolved O3).
Prime powers are **VACUOUS by construction**: one CRT component means `n/q = 1`, the
predicted set is every frequency, and the test says nothing. Never counted as support.

### 5.1b The CRT law survives a STRUCTURE-MATCHED null — and its p-value does not discriminate (C28-NULL)

Pre-registered in `experiments/PREREGISTER_c28_crt_null.md` (committed `832635c` **before
any structure-matched-null number existed**). Script `test_crt_null.py`, log
`logs/c28_null.log`, zero compute.

**Why this was needed.** `permutation_test` scores the CRT-dual set against *uniformly
random* same-size subsets. But the predicted set is a **union of two subgroups** of Z/n —
11 of 59 folded bins at n = 119 — so a uniform null is weak against *any* alternative that
concentrates energy on arithmetic progressions through 0. That is the size/structure
confound that killed C7b, C20 and C19. C8 is VERIFIED on 9/9 moduli using this test, so the
instrument was carrying more weight than it had been asked to bear.

**Three size-matched nulls, 2,000 draws each:**

| null | construction | what it controls |
|---|---|---|
| **A** | uniform random subsets | set size only — the existing test |
| **B** *(primary)* | unions of "multiples of d" progressions, d over **non-divisors of n** | the subgroup/AP shape itself |
| **C** | uniform subsets matched on mean \|k\| within ±2 | a low-frequency bias |

**Result 1 — the signal is CRT-specific.** n = 119, engine, 3 seeds, every `we_traj` snapshot:

| step | enrichment (s0/s1/s2) | p_B | test acc |
|---|---|---|---|
| 0 — the model's **own init**, same seed and split | 1.017 / 1.006 / 0.964 | 0.047 / 0.196 / 0.998 — **rejects 0/3** | 0.03 |
| 1,000 | 1.450 / 1.225 / 1.224 | **0.000 ×3** | 0.054–0.075 |
| 40,000 | 3.448 / 3.195 / 3.152 | 0.000 ×3 | grokked |

Ramp ρ(step, enrichment) over 0–20k = **1.000 / 0.991 / 0.999**. Null C agrees throughout.
So the enrichment tracks the **factorisation of n** — not subgroup shape, not low \|k\|.
**C8's instrument survives its hardest available test.**

**Result 2 — the p-value does not discriminate, for the third time.** Every FAILED run at a
well-posed modulus also rejects at p_B = 0.000, including one at **test acc 0.2236**. This is
the same failure as C22/C23's control-mean and as G2's G1 criterion. **What separates is the
magnitude**, and the clean comparison is within a single modulus, where the predicted set is
identical by construction:

| n = 63, 7 of 31 bins | enrichment |
|---|---|
| s0 grokked (0.9969) | **4.687** |
| s1 FAILED (0.2236) | 1.626 |
| s2 FAILED (0.2749) | 1.574 |

Across all moduli the arms do not overlap: grokked **3.27–17.44** (44 runs) vs FAILED
**1.52–2.67** (4 runs). **Report the enrichment, never the p.**

**Result 3 — a caveat this forces onto C28.** The step-1,000 enrichment (**1.22–1.45**) is
*below* the band a memorising model reaches at convergence (1.52–1.63 at n=63). "CRT
structure is imprinted early" therefore survives as a statement about **significance** and
**not** as evidence that what appears early is the grokking circuit forming. That is O20.

**Limits, stated.** The PRIMARY rests on **one modulus (n = 119) and 3 seeds** — the only
one in the project with ≥2 CRT components *and* 1,000-step early resolution. The FAILED arm
is **2 runs at a well-posed modulus**: at n = 54 and n = 98 the predicted set is 52 % and
51 % of all bins, near-vacuous, and null B is not constructible there (reported `n/a`, never
counted as a pass). Engine and torch arms are never pooled.

### 5.1b-bis O20 — the early CRT ramp is NOT the grokking circuit (2026-09-15)

**Pre-registered** `experiments/PREREGISTER_o20_failed_ramp.md`, committed `6042c56`
**before the runs existed**. Three engine seeds at n = 63 = 3²·7, 20,000 steps,
`train_frac` 0.30, 1.11–1.25 h local CPU, **zero quota**. Scored by
`CRT_NULL_N=63 test_crt_null.py results/o20_n63`.

**The question C28-NULL left open.** The CRT enrichment ramp is real, monotone
(ρ ≈ 1.000), CRT-specific and survives a structure-matched null. But its magnitude does not
separate — a grokked run reads 1.22–1.45 at step 1,000 while a **failed** model reaches
1.57–1.63 at convergence. So: is the ramp the generalising circuit forming, or structure any
model fitting 30 % of the table acquires?

**All three seeds FAILED** — `grok_step=None`, test-accuracy window medians **0.1801 /
0.1655 / 0.3183**, train accuracy 1.000 from step 2,000. A clean memorisation arm.

| seed | final acc | enrich @1k | enrich @20k | ramp ρ (0–20k) |
|---|---|---|---|---|
| 0 | 0.1803 | 1.397 | **1.653** | **1.000** |
| 1 | 0.1655 | 1.339 | **1.459** | **0.988** |
| 2 | 0.3177 | 1.422 | **1.655** | **0.995** |

`p_A`, `p_B`, `p_C` are **0.0000 from step 1,000** in every failed seed. Pre-registered
criterion 1 (ρ ≥ 0.80) fires 3/3; criterion 2 (final > 1.45) fires 3/3.

**Independent corroboration from data already on disk.** `test_crt_null.py`'s discrimination
arm scores k04's **torch** runs at the same modulus: failed seeds read **1.626** and
**1.574** (acc 0.2236, 0.2749), both p < 0.01, against the grokked seed's 4.687. The FAILED
arm passes null A **4/4**. Two implementations, two data sources, one answer.

#### What this changes, and what it does not

**It extends C28, it does not contradict it.** C28 already records that the permutation p
does not separate grokked from failed and that **only the magnitude does**. What is new is
that the **ramp shape does not separate either**. Monotonicity was the last property that
might have made the early signal diagnostic of circuit formation, and it is not.

**Untouched:** the CRT-dual law itself (C8, 9/9 fresh moduli), its structure-specificity
(C28's three nulls), and its **causal** role in grokked models (C22/C23/C36, p < 0.01).
Enrichment *magnitude* still separates cleanly — grokked runs reach 3.3–17.4 where these
failures top out at 1.66.

⚠️ **The early ramp must never be presented as a progress measure or as evidence of circuit
formation.** Wherever it appears in the paper, the failed-run ramp appears beside it. This
is the Gate 2 G1 lesson for the third time: **a statistic that fires on failures is not
evidence, whatever it does on successes.**

⚠️ **Near-miss worth keeping.** The engine's `hist_cols` is
`[step, train_loss, test_loss, train_acc, test_acc]`, so **index 3 is `train_acc` = 1.000**
in all three failed runs. Positional indexing would have classified every FAILED run as
grokked and inverted the experiment. Read by name.

### 5.1c Additive sparsity is carried by ω(n), not by cyclicity (O7 resolved by replacement)

**EXPLORATORY, NOT PRE-REGISTERED — not a numbered claim.** `analyze_omega.py`,
`logs/omega.log`, 18 moduli / 115 grokked runs, zero compute. Runs classified on a window
median, never the last row (§3.5b).

O7 asked whether **cyclicity** modulates C9. It does not. The variable is **ω(n), the count
of distinct prime factors**, and it separates Gini_add into bands with **no overlap**:

| ω(n) | moduli | runs | Gini_add range | mean |
|---|---|---|---|---|
| **1** | 49, 81, 113, 121, 125, 169 | 57 | **0.017 – 0.184** | 0.130 |
| **2** | 54, 63, 75, 98, 99, 100, 119, 143, 147 | 39 | **0.394 – 0.481** | 0.450 |
| **3** | 105, 120, 165 | 19 | **0.578 – 0.589** | 0.585 |

Residue-axis random-orthogonal control: **0.117 – 0.155** throughout (P9c 2026-09-29: 0.157 reproduced on no population).

**Why it is ω and not cyclicity.** The two are collinear here (ρ = −0.463), which is why
cyclicity looked causal (ρ(cyclic, Gini_add) = −0.641). **The moduli where they disagree
decide it** — and both go to ω:

| n | factorisation | cyclic? | ω | Gini_add |
|---|---|---|---|---|
| 54 | 2·3³ | **yes** | 2 | **0.456** |
| 98 | 2·7² | **yes** | 2 | **0.453** |
| *whole ω=1 band* | — | yes | 1 | *tops out at 0.184* |

**C9 is not retracted.** ρ(zero-divisor density, Gini_add) = **+0.858** still holds, and zdd
continues to predict *within* ω bands (+0.886 at ω=1, +0.502 at ω=2). ρ(ω, zdd) = **+0.766**,
so the two are substantially collinear. *(Midranks, 18 moduli, 2026-09-29 (P9b). The ordinal ranks read +0.862, +0.533 and +0.674,
and `experiments/PREREGISTER_omega.md` quotes those as committed.)*

> ⚠️ **AMENDED 2026-09-15.** This paragraph previously ended *"so zero-divisor density is
> partly a proxy for ω"*. **The confirmatory test below does not support that direction.**
> ρ(zdd, Gini_add | ω) = **+0.739** is *higher* than ρ(ω, Gini_add | zdd) = **+0.700** —
> each survives partialling on the other, at comparable strength, so the mediation runs
> **both ways** and neither variable is the other's proxy. See §5.1c-bis.

**This is the variable C8 already implies, which is why it is not a fished correlation.**
The CRT-dual predicted set is built as one family per maximal prime power (§5.1). At ω = 1
that set is *every* frequency — vacuous, with nothing to concentrate on — so a prime power
has no additive structure to be sparse in. Each further prime adds a family, hence more
concentration, hence higher Gini_add. **It also explains why a prime power's clock must live
in the multiplicative basis (C6): additively, there is nothing there.**

**What it needed to become a claim:** a pre-registration with a size control — ω is
collinear with both zdd and n — and a permutation test. **Done 2026-09-15; see below.**

#### 5.1c-bis The confirmatory test — the confounds are dead, the crown is not awarded

`experiments/PREREGISTER_omega.md`, committed `60ee31b` **before any statistic below was
computed**; `analyze_omega.py --confirm`; log `logs/omega_confirm.log`; zero compute.

⚠️ **This is a CONFIRMATORY test of an EXPLORATORY observation, on the same data.** The
bands were seen first. It cannot make ω a pre-registered discovery and must never be written
up as one. What it does is kill the confounds that destroyed C7b (𝒥-class size), C19 (φ(n))
and C20 (block cell count) — every one of which was a grouping variable coupled to a size,
with the statistic reading the size.

> **UPDATED 2026-09-15 by k09.** W1 is now **HELD** on **23 moduli** — the table below is
> the original 18-moduli scoring, kept because it is what the criterion saw when it was
> scored. See §5.1e.

| # | statistic | observed (18 moduli) | verdict |
|---|---|---|---|
| **W1** | **PRIMARY** — ω at matched zdd (\|Δzdd\| ≤ 0.05) | **5 discordant pairs**, threshold ≥ 6 → **10 pairs, 10/10, p = 1.95e−03 at 23 moduli** | **NOT ASSESSABLE** → **HELD** |
| **W2** | ρ(ω, Gini_add \| zdd) | **+0.700** | **HELD** (≥ +0.50) |
| **W3** | ρ(zdd, Gini_add \| ω) | **+0.739** | descriptive — **the higher of the two** |
| **W4** | random-orthogonal control bands by ω | 0.117–0.148 · 0.125–0.155 · 0.128–0.147, **fully overlapping** | **HELD** |
| **W5** | ρ(n, Gini_add) — **blind** | **+0.015** | **HELD** |
| **W6** | permutation, 10,000 shuffles of ω | min adjacent gap **+0.097**, p = **1.0 × 10⁻⁴** | **HELD** |

**The size confound is absent.** W5 was blind and reads ρ(n, Gini_add) = **−0.185** at the
23 moduli (the table above is the 18-modulus scoring, where it read +0.015) — the
magnitude of n carries essentially nothing. W4 adds that the residue-axis control has **no
band structure at all**, three overlapping ranges inside 0.117–0.155, so the separation is
not an artifact of spectrum length. W6 puts it at the permutation floor. **Pre-registered
robustness holds:** dropping the project's single prime leaves the ω = 1 band at
**0.125–0.184** across five prime powers, still clear of ω = 2's 0.394.

**ω does not displace zero-divisor density.** W3's own pre-registered clause fired. The
defensible statement is: **ω(n) and zdd are two partially independent real predictors of
additive sparsity, and this data cannot rank them.**

**W1 is not assessable and its threshold was not moved.** Five zdd-matched discordant pairs
against a pre-registered minimum of six, fixed before the pair count was known. *Exploratory
and not scored:* the direction is unanimous at every tolerance — **3/3, 5/5, 11/11, 18/18**
at \|Δzdd\| ≤ 0.03/0.05/0.07/0.10, no exceptions — the cleanest being n = 119 (ω = 2,
zdd 0.193) at **0.440** against n = 125 (ω = 1, zdd 0.200) at **0.184**. The pairs share
moduli and are not independent, so no p-value is defensible. **Closing W1 needs more moduli,
not a wider tolerance.**

### 5.1e k09 — W1 closes, and ω survives a designed falsifier at n = 2⁷

`experiments/PREREGISTER_k09_primes_zdd.md`, committed `ed9a1f2` **before the runs existed**;
Kaggle **Tesla T4**, 15 runs, 93 min, SHA `ed9a1f2`; `logs/k09_score.log`.
Five moduli chosen to do two jobs: give the project **more than one prime**, and fill the
zero-divisor-density axis at contrasting ω so W1 becomes scorable.

**W1 is HELD: 10 discordant zdd-matched pairs, 10/10 favouring ω**, fraction 1.000, exact
two-sided sign test **p = 1.95 × 10⁻³**. The threshold was never moved — moduli were added,
which is what the original scoring said was required. ⚠️ The pairs share moduli and are
**not independent**; the fraction is the statistic, the p is descriptive.

**The designed falsifier fired, and ω survived it — but not cleanly.** n = 128 = 2⁷ is a
prime power (ω = 1) with **half its residues zero divisors** (zdd = 0.500), the one place the
two predictors disagreed. Pre-registered as disjoint intervals: **ω 0.017–0.184** against
**zdd 0.434–0.589**.

| | predicted | observed |
|---|---|---|
| ω(n) = 1 | 0.017 – 0.184 | |
| zdd = 0.500 | 0.434 – 0.589 | |
| **n = 128 = 2⁷** | | **0.261 — between both** |

**P1 is AMBIGUOUS and is reported as such.** Neither predictor was accurate. What it settles:
n=128 lands on **ω's side** — it extends the ω = 1 band to 0.015–0.261 and the ω = 1 / ω = 2
bands **still do not overlap** (0.261 vs 0.394), so **§5.1c-bis is not retracted** — and in
W1 it appears in 3 of the 10 pairs and favours ω in all three. **ω is the better predictor of
the two and neither predicts this modulus.** That is the concrete form of W3's finding.

⚠️ **ω does not displace zdd, and zdd does not displace ω**: ρ(ω, G | zdd) = **+0.804**,
ρ(zdd, G | ω) = **+0.578**, both on midranks (2026-09-23 audit). ⚠️ *Until that audit these
read +0.627 / +0.735 — the same pair in the opposite order. `argsort(argsort())` is an
ORDINAL rank; ω has three distinct values over 23 moduli, so inside a band it ranked by n.
The ordering was a tie artifact. Neither direction may be quoted as a ranking.*

#### n = 113 is not special (G1 closes)

The project had trained exactly one prime. It now has three.

| n | Gini_mult | Δ vs n=113's 0.5665 | neurons tuned (dlog) | raw | shuffled | key-set overlap |
|---|---|---|---|---|---|---|
| **127** | 0.5562 | **0.0103** | 0.908 · 0.887 · 0.998 | 0.000 | 0.000 | 4/4 · 5/5 · 4/4 |
| **131** | 0.5915 | **0.0250** | 0.963 · 0.975 · 0.975 | 0.000 | 0.000 | 4/5 · 4/4 · 4/4 |

Both inside the 0.10 threshold C6 already uses, 3/3 groks each. **88.7–99.8 % of MLP neurons
are single-frequency in discrete-log coordinates against 0.0 % raw and 0.0 % shuffled**, and
the tuned neurons sit on **exactly** the embedding's key frequencies in 5 of 6 runs —
reproducing C31's 14/14 pattern at moduli it had never seen. **Every "prime vs composite"
sentence now rests on three primes, not one.**

#### The clock is CAUSAL at 2⁷ — and C6's prime-power set reaches p = 2

`analyze_n4.py`, **10,000**-draw permutation in exponent-tuple (product-character)
coordinates. **0 of 10,000 draws as damaging as the key set in every analysable run —
13/13 across all five moduli, i.e. p̂ = 1/10,001 = 1.0 × 10⁻⁴, the floor** ⚠️ *amended
2026-09-21 (R1) from 200 draws, where the same 13/13 read `p < 0.005`; this family is one
of the few where the absolute statement survived the extra resolution intact.*

**⚠️ CORRECTED 2026-09-16, after the reviewer panel (Entry 52). TWO defects in the table
below as it originally stood, both of them already-documented families:**
1. **The column was labelled `restricted/baseline` and held `baseline/restricted`** — the
   upside-down-ratio defect `LAB_PROTOCOL.md` records for FINDINGS §3.3, here for a third time.
2. **Every value was a MEAN over 2–3 draws of a distribution this project has shown to be
   heavy-tailed**, with the ± being the *population* sd of two points at n=128. The means
   describe no run in the sample: **n=131's "262×" has a median of 2.88×**, and n=127's
   "19.62×" a median of 5.04×. Report the median and the range.

| n | local group | **baseline/restricted**, median | range | runs | p < 0.01 in |
|---|---|---|---|---|---|
| **128 = 2⁷** | Z₂×Z₃₂ | **239.52×** | 2.77 – 476.26 | 2 | **2/2** |
| 131 | Z₁₃₀ | **2.88×** | 2.44 – 781.26 | 3 | 3/3 |
| 127 | Z₁₂₆ | **5.04×** | 2.18 – 51.63 | 3 | 3/3 |
| 91 = 7·13 | Z₆×Z₁₂ | 117.26× | 1.45 – 233.07 | 2 | 2/2 ⚠️ |
| 123 = 3·41 | Z₂×Z₄₀ | **1.76×** | 0.00 – 2.19 | 3 | 3/3 |

**What survives unchanged is the claim**: the permutation p is the primary statistic and
every one of the 13 runs is beyond every one of its 10,000 draws. What does not survive is any
sentence quoting a mean of these ratios. Asserted in `test_paper_numbers.py`.

**n = 128 is the finding.** The first power of 2 here, ω = 1, zdd = 0.500, unit group
Z₂×Z₃₂ with **no discrete log** — and the product-character circuit is causally responsible
at **239×**. C6's prime-power set was 7², 11², 5³, 3⁴, 13²; it now reaches **p = 2 and
depth 7**, the deepest nilpotency tested.

⚠️ **n = 123 reproduces the O8/O11 "necessary but not sufficient" pattern** at a modulus
neither had seen: p < 0.01 in 3/3, yet **baseline/restricted median only 1.76×, range
0.003–2.19×, n = 3** (P16, 2026-09-29: the 1.32× ± 0.95 first written here was a mean of three
under an upside-down label). Consistent
with O8/O11's 2026-09-14 re-scoping — a per-**run** property, not a per-modulus one.

⚠️ **`analyze_n4.py` and `analyze_k09.py` disagree on n = 91's seed count** (2 vs 1): n4
admitted the 0.976 **near-grok**. Under the three-state rule a near-grok is neither data nor
control, so n=91's row is not counted as independent support. Flagged, not pooled.

#### C8 at two fresh CRT-testable moduli

n = 123 = 3·41: enrichment **7.59 / 3.28 / 7.43**, permutation **p = 0.0000 in 3/3**.
n = 91 = 7·13: enrichment **8.57, p = 0.0000**, but only one seed grokked, so it is
**NOT ASSESSABLE** at the ≥2/3 rule and is not counted as support. 127/131/128 are
**VACUOUS** by construction (one maximal prime power) and never counted.

⚠️ **A pre-registered argument of mine was falsified by its own run.** The modulus set was
justified by calibrating against k04's grok rates — the transition sits at ~1,700–2,000
training examples, and n=91 has 2,484, above n=81's 3/3. **n = 91 grokked 1/3** (one
near-grok 0.976, one failure 0.358), and n = 128 grokked 2/3 and slowly (23,400 / 22,000
against 3,200–11,000 at the primes). **Training-set size alone does not predict grokability.**

**Figure:** `figures/omega_bands.pdf` — the bands, the zdd confound, the flat
random-orthogonal size control, and the n=128 prediction drawn **beside** its outcome.

### 5.1d n = 49, 54, 63 are dataset-size nulls, and the clock survives at 7² (O9 resolved)

The answer was already on disk in k05, which sweeps `train_frac` ∈ {0.30, 0.50, 0.80}:

| n | f30 | f50 | f80 |
|---|---|---|---|
| 49 = 7² | 0/3 | 2/3 + 1 near-grok | **3/3** |
| 54 = 2·3³ | 1/3 | **3/3** | **3/3** |
| 63 = 3²·7 | 1/3 | **3/3** | **3/3** |

Monotone; no modulus resists. At f30 they hold 720 / 874 / 1,190 training examples, which is
the C14 critical-dataset-size regime — **not an algebraic obstruction.**

**C6's P4 hostage is discharged: the clock survives at a prime power.** Across the 5 grokked
n = 49 runs — Gini_mult **0.516–0.645**, Gini_add 0.107–0.174, random-orthogonal 0.098–0.195,
2–5 key frequencies, PR_mult 2.1–5.7. n = 63 is skipped loudly: (Z/63)\* = Z6×Z6, no
discrete log.

⚠️ **n = 54 is the residue worth carrying.** Gini_mult 0.395–0.499 sits *level with* Gini_add
0.440–0.477, and `n_key` = 0 in 4 of 7 runs — the multiplicative basis is **not** preferred
there. Consistent with §5.1c (ω = 2 puts real structure in the additive basis) and with C33
(the unit stratum is only 18 of 54 residues, so a unit-only statistic sees a third of the
table). Both readings are post-hoc and neither is tested.

### 5.2 Novelty — stated honestly

2606.23044 already argues *"for square-free composite moduli, the CRT predicts which prime
channels are task-relevant."* **The CRT connection is not ours.** What is:

| | 2606.23044 | this work |
|---|---|---|
| embedding | **imposed** by design (PFE) | **learned** — emerges under gradient descent |
| architecture | per-prime MLP encoder + classifier | 1-layer transformer |
| task | addition | **multiplication** |
| moduli | square-free only | includes **n=120, non-square-free** |

The defensible claim: *a transformer with a standard learned embedding, trained on modular
multiplication, spontaneously develops the frequency structure PFE builds in by hand — and
does so at non-square-free moduli where the plain CRT channel argument does not directly
apply.* **The n=120 result is the load-bearing part.**

### 5.3 Additive sparsity tracks zero-divisor density (**18 moduli** — C9)

**k04 pre-registered P2, pooling k02 Arm B with the 12 fresh moduli: Spearman ρ = 0.786
over 18 moduli, permutation p < 5e-5.** Predicted > 0.6, with regression from the 6-point
value anticipated in advance. **HELD.**

Extended-set values (3 seeds each): 49 → 0.099, 54 → 0.314, 63 → 0.238, 75 → 0.421,
81 → 0.184, 98 → 0.453, 99 → 0.444, 100 → 0.449, 105 → 0.589, 143 → 0.394, 147 → 0.471,
169 → 0.134.

The original 6-modulus measurement (k02 criterion B5, 5-seed means, **ρ = 0.943**,
Pearson r = 0.898) is below.

| n | zero-divisor density | Gini_add (5-seed mean) |
|---|---|---|
| 113 | 0.009 | 0.018 |
| 121 | 0.091 | 0.125 |
| 119 | 0.193 | **0.421** |
| 125 | 0.200 | **0.177** |
| 165 | 0.515 | 0.579 |
| 120 | 0.733 | 0.587 |

**Exploratory (not pre-registered):** at near-identical density, non-cyclic n=119 (0.421)
more than doubles cyclic n=125 (0.177). Cyclicity may modulate this relationship (O7).

> **Two bases, two roles: the additive basis says *which stratum*; the multiplicative basis
> does the arithmetic *within* it.**

---

## 6. An answer to 2607.07066's open problem (1 seed)

Their mechanism needs each 𝒥-class to be a group (Thm 3.4: regular `J_d ≅ (Z/(n/d)Z)^×`,
with local identity and local inverse `c♯`, decoding `Logit(c) ∝ χ_{ρ_d}(a b c♯)`).
Non-regular classes have no idempotent, hence no `c♯` — their stated gap.

Index `J_d` not by a group law it lacks, but by the **unit-group action**: `J_d = {d·u : u a
unit mod n/d}` is a bijection (|J_d| = φ(n/d)) and is defined whether or not `J_d` is
regular. Then take the product-character transform in that coordinate.

| n | class | regular | order | Gini | random control | ratio |
|---|---|---|---|---|---|---|
| 121 | **J_11** | **NO** | 10 | 0.345 | 0.124 | **2.79×** |
| 125 | **J_5** | **NO** | 20 | 0.526 | 0.156 | **3.37×** |
| 120 | **J_2** | **NO** | (2,2,4) | 0.476 | 0.162 | **2.93×** |
| 113 | J_1 | yes | 112 | 0.553 | 0.148 | 3.74× |
| 119 | J_1 | yes | (6,16) | 0.528 | 0.128 | 4.13× |
| 165 | J_3 | yes | (4,10) | 0.596 | 0.123 | 4.84× |

**Every class beats its random-basis control (all ratios > 1.9), regular or not.**

> The network does not need local invertibility. A non-regular 𝒥-class is not a group, but
> it *is* a set acted on transitively by the units. The network represents the **action
> coordinate** and builds characters of the *acting* group rather than of the class itself.
> Local inverses are sufficient for their mechanism, not necessary.

### 6.1 RETRACTED: "regular classes carry more structure than non-regular"

The aggregate (regular 3.71× vs non-regular 2.72×) is **confounded by class size** — `J_1`
is always regular and always largest, and Gini rises with size here. Size-matched within
one model, at n=120, the only place such pairs exist:

```
size 8:   regular J_5 2.82×, J_8 1.95×  |  non-regular J_4 2.43×, J_6 2.07×
size 16:  regular J_3 3.21×             |  non-regular J_2 2.93×
```

**They overlap. NOT SUPPORTED at 1 seed.** Classes smaller than 8 elements are unresolvable
and are excluded rather than interpreted.

---

## 7. GATE 1 — the from-scratch engine reproduces Nanda et al. (VERIFIED, PASS)

`a+b mod 113`, 30% train, 25k steps on our own autograd. **Grokked at step 6,900**. 3.14 h
local CPU, **zero Kaggle quota**.

> **Comparator corrected 2026-09-15 (Entry 38, O1b closed).** This line used to read
> "(Nanda's own run: 9k–14k)", which was then carried into O1b as a gap to explain. **9.4k–14k
> is Nanda's *Cleanup phase* of their mainline run** — the width of one run's test-loss drop —
> not a grokking time. Their stated grokking time **at our weight decay** is Appendix D.1:
> "**5-10k epochs** ... with weight decay λ = 1.0". **6,900 is inside that band, and Gate 1
> agrees with them on grokking time as well as on mechanism.** They also state that grokking
> time differs by random seed across their five seeds.

| criterion | ours | Nanda 2301.05217 |
|---|---|---|
| test accuracy | **1.0000** | — |
| key frequencies in `W_E` | **5** — {1,5,23,33,45} | **5** |
| embedding sparsity | Gini 0.761, PR 7.33 | — |
| ablate key freqs | 6.777 vs baseline 1.31e-07 | same direction |
| **ablate all non-key freqs** | **1.79e-08 — improves 7.3×** | improves 70% |
| causal key freqs | **4** — {1,5,33,45} | 5 |
| **neurons single-frequency tuned** | **85.0%** of 512 | **84.6%** |

**85.0% against a published 84.6%**, from an independent implementation with its own
autograd — the strongest single piece of evidence the engine is sound.

### 7.1 The vestigial frequency (VERIFIED, 1 seed)

k=23 has **embedding norm 23.1** (noise floor ≈ 1) but a **logit ablation delta below
4e-10** — present in `W_E`, does nothing downstream. Nanda reports the same: *"Of the six
non-zero frequencies, five key frequencies appear in later parts of the network."*
Per-frequency causal deltas: k=33 **+0.414**, k=45 **+0.072**, k=5 **+0.046**, k=1
**+0.026**; everything else < 7e-09.

**Embedding-level frequencies ≠ circuit-level frequencies. Report the causal set.**

### 7.2 ⚠️ Methodological caveat on Gate 1

**C6 and C7 initially failed and were edited after the failure.** Recorded because §7 of the
plan warns against exactly this. Mitigating: the **pre-registered H1.1 criteria are C1–C5
and those passed unmodified**; C6/C7 were additions beyond H1.1; both fixes implemented what
Nanda's paper actually states (C7 had scored a single 2D bin where he scores a degree-2
polynomial spanning eight bins — hence the initial 9.6%). The corrected C7 landing at 85.0%
against a published 84.6% is independent evidence the fix is right. **But the ordering was
wrong. For T2, criteria are frozen before the run.**

### 7.3 Full mechanistic readout of the Gate 1 circuit (VERIFIED, 1 seed)

Once the model's internals were actually rendered (`scripts/look.py`), three independent
measurements agree on the same circuit.

**Logit 2D DFT — the top 8 components are all `a+b`:**

| component | norm | independent ablation delta |
|---|---|---|
| (±33,±33) | **100%** | **+0.414** |
| (±45,±45) | 31.9% | +0.072 |
| (±5,±5) | 28.0% | +0.046 |
| (±1,±1) | 24.0% | +0.026 |

**The rank order by logit energy is identical to the rank order by causal ablation.**
`a−b` appears first at 3.5%; among the top 16 components the energy share is
**a+b 100%, a−b 0%**. **k=23 does not appear at all** — a third confirmation, alongside its
near-zero ablation delta, that it is vestigial in `W_E` and absent from the circuit.

**Neuron energy decomposition** (which function of (a,b) each neuron holds):

| term | energy | neurons dominated |
|---|---|---|
| `a` alone | 35.2% | 229 |
| `b` alone | 35.2% | 206 |
| **`a+b`** | **19.9%** | **77** |
| `a−b` | 8.3% | **0** |

Perfect a/b symmetry, as a commutative task requires. **Composition ratio
`a+b / (a+b + a−b)` = 70.6%.** ~70% of neuron energy sits in the linear input terms
(`cos(wa)`, `cos(wb)`) that a ReLU carries before its quadratic cross-terms produce `a±b`.

**Method note.** The bin convention was wrong in the first implementation — `(k,k)` is
`f(a+b)` and `(k,−k)` is `f(a−b)`, the opposite of what I reasoned. Reported backwards it
inverts the conclusion. Now verified empirically against planted signals
(`test_mechinterp.py`), never by reasoning.

---

## 8. Engine and training dynamics (VERIFIED)

### 8.1 Softmax Collapse — H1.4 confirmed on our own engine

n=17, mul, 40% train:

```
step      loss        |grad|      max logit   p_correct
   0    2.871e+00    4.150e-01       0.5      0.037533
 400    4.706e-04    1.764e-03      47.0      0.998725
 800    5.721e-06    3.208e-05      46.1      0.999984
1200    1.299e-07    9.214e-07      57.6      1.000000
```

**Gradient norm falls 5.5 orders of magnitude while `p_correct` reaches exactly 1.000000.**
Prieto et al.'s no-gradient regime, measured directly. After that only weight decay moves
the weights. **StableMax mitigates**: |grad| 2.3e-6 vs softmax's 2.3e-7 at step 4000 —
~10× more gradient at the same logit scale.

Scout signal worth following up (O5): at step 300, max logit was **156 (n=121) vs 54
(n=113)** — composite moduli may hit collapse first.

#### ✅ 2026-09-15 — C12 now has a script and a stamped artifact, and C13's NUMBER does not reproduce

`run_c12_collapse.py` → `results/c12_collapse/softmax_collapse_n17.npz`, SHA `e0ead8e`,
`git_dirty=False`, in `reproduce.sh`. Same task (n=17, mul, `train_frac` 0.40), **seed 0
fixed** — the table above came from an ad-hoc run whose seed and config were never recorded,
so this reproduces the **phenomenon**, not the row values.

**C12 REPRODUCES.** |grad| falls **6.2 orders** (3.409e-01 → 2.353e-07) while `p_correct`
reaches exactly **1.000000** and the max logit sits at ~50. Published rows and this run agree
to a factor of ~2-4 on every quantity, which is seed variation, and the step-4000 softmax
gradient matches the published 2.3e-7 **exactly** at 2.353e-07.

⚠️ **C13's published magnitude does NOT reproduce, and one of its clauses is false.**

| | published (§8.1 above) | this run, seed 0 |
|---|---|---|
| stablemax \|grad\| @4000 | 2.3e-6 | **7.510e-07** |
| softmax \|grad\| @4000 | 2.3e-7 | 2.353e-07 ✅ |
| ratio | **~10×** | **3.2×** |
| "at the same logit scale" | asserted | **FALSE — 50.1 vs 19,154.3, a factor of 383** |

The **direction** holds — StableMax retains more gradient, and its `p_correct` stops at
0.999998 rather than saturating at exactly 1.0, which is the mechanism. The **factor of 10**
and the **equal-logit-scale claim** do not. StableMax's logits grow without the softmax's
saturation ceiling, so the two arms are not at a comparable logit scale at any step past ~100
and the ratio is not a like-for-like comparison.

**C13 should be written as a mechanism, never as a 10× number.** Neither arm generalizes
(test acc 0.178 / 0.126) — that is C14 at n=17/40%, not a StableMax failure, and it is the
configuration `LAB_PROTOCOL.md` documents as guaranteed to fail.

### 8.1b TWO PRECISION REGIMES — which number came from which

> ✅ **RESOLVED 2026-09-15 (C34, `PREREGISTER_o18_f64.md`). The mechanism is PRECISION, acting
> on PyTorch.** The 2×2 at n=113, tail Gini_mult: **engine f32 0.7671 · engine f64 0.7536 ·
> torch f32 0.5791 · torch f64 ~~0.7736~~ **0.7728**.** `f = ~~+1.114~~ **+1.110**` on Gini,
> `+0.968` on `‖W_E‖`, both HELD.
>
> ⚠️ **CORRECTED 2026-09-16.** All four cells were recomputed from their own artifacts by
> `test_paper_numbers.py` (`tail_gini`, mean over the final 10k steps, 3 seeds each).
> **Three of four reproduce to 4 dp; the torch-float64 cell does not** — the artifacts read
> **0.772770** (seeds 0.7618 / 0.7856 / 0.7710) against the **0.7736** written here, so the
> pre-registered F1 is **+1.110**, not +1.114. **The verdict is unchanged** — F1's threshold is
> 0.70 and both values clear it by a wide margin — but the paper and `figures/F7_precision.png`
> now recompute every cell rather than quote this table. ⚠️ Reminder, since it bit again:
> `analyze_n9.py` on `results/o18_f64` prints **"NOT HELD (f = −0.10)"**, because its
> baseline/target are the *engine's* two dtype cells. **Compute the committed formula.**
> Dtype is **inert in the engine** and **decisive in torch** — an **interaction**, so
> **C30 (a main effect) stays retracted.** Float64 also moves torch's neuron tuning
> 0.938 → 0.770, which dissolves the "one convergence axis cannot do that" anomaly recorded
> below. **All of k02–k09 is float32.** Section title amended from "(C30, mechanism under
> test)"; the text below is the state as it stood while it was under test.

**Every number in this file was produced in one of two training-precision regimes, and they
do not agree.** This section exists so that no figure in the paper is quoted without its
regime attached; it records the *fact*, which is settled, and not the *cause*, which is not.

| regime | where | what produced it |
|---|---|---|
| **float32** | every Kaggle kernel — k00–k08, so C3, C4, C5, C6, C8, C9, C21–C24, C26, C27, **C31** | PyTorch autograd, `results/k*` |
| **float64** | the from-scratch engine — Gate 1, C10, C29, and all of `results/n7_engine` | `src/autograd/engine.py`, which built every Tensor as float64 |

The disagreement is not subtle. At **byte-identical** hyperparameters (lr, wd 1.0, betas
(0.9, 0.98), init scales, seeds, steps, train_frac, architecture, and the same data split
RNG), N7's E5 predicted the two would agree and they did not, in the same direction, at
every modulus:

| n | engine float64, tail Gini_mult | torch float32 | Δ | torch 1-sd band |
|---|---|---|---|---|
| 113 | 0.754 ± 0.016 | 0.556 ± 0.015 | **+0.197** | 0.026 |
| 121 | 0.662 ± 0.092 | 0.543 ± 0.020 | **+0.119** | 0.028 |
| 125 | 0.805 ± 0.036 | 0.543 ± 0.052 | **+0.262** | 0.048 |

Window-independent check on the final weights: **7 of 7 grokked matched pairs**, engine
sparser by +0.17 to +0.28, and the engine's `W_E` norm consistently smaller (11–13 against
15–19). Eliminated before precision was reached: the protocol (the same `tail_gini` and
`mult_amplitude` run on both sides), the hyperparameters (read from the k03 kernel source,
not assumed), the tail window, and the optimizer (the engine's AdamW is algebraically
`torch.optim.AdamW`).

> **`Gini_mult = 0.75` and `Gini_mult = 0.55` can be the same experiment.** A sparsity
> number from this project is uninterpretable without its precision regime, and the two must
> never be pooled into one mean.

**Consequences already visible in this file.** C4's Arm A replication matched the published
0.579 **because it was float32** — the arm that reproduced the literature is the arm that
shares its precision. C31's neuron-tuning percentages are float32 throughout, and the
float64 engine's only failed run (`n125_s1`, 6.8% tuned) is the sole float64 number quoted
beside them, labelled as such.

**❌ AND THE SUBSTRATE EXPLANATION IS RETRACTED TOO — O18, 2026-09-14.** After N9 killed
precision, the two candidates left were the execution substrate (Kaggle **T4 GPU** vs local
**CPU**) and the specific init draw. The k03 kernel's **own source**, patched only for
device and job list, was run on local CPU — 3 seeds × 40k steps, 108 min, zero quota:

| regime | tail Gini_mult | `‖W_E‖` |
|---|---|---|
| engine float64 | 0.7536 | 11.610 |
| torch **T4 GPU** | 0.5665 | 15.152 |
| **torch CPU** | **0.5791** | **14.296** |

**Torch on a CPU behaves like torch on a T4, not like the engine.** `f = 0.068` on
Gini_mult (Q1 NOT HELD, ≤ 0.30) and `f = 0.242` on `‖W_E‖` (Q2 NOT HELD — a quarter of the
*norm* gap does move with the substrate, reported as measured). So **every Kaggle number in
this project is NOT carrying a hardware artefact**, and the engine–torch gap is **in the
code**. What remains is the specific random **init draw** or a structural difference not yet
found; both implementations now run on the same box, so the next test starts the engine from
torch's own sampled weights. `experiments/PREREGISTER_o18_substrate.md`.

**❌ THE PRECISION EXPLANATION IS RETRACTED — N9, 2026-09-14.** The mechanism named above
(float32's larger gradient noise under wd = 1.0 leaving the model less feature-learned) was
put to a controlled A/B: the engine at `ENGINE_DTYPE=float32`, n=113, seeds 0–2, 40k steps,
one variable changed, criteria committed before the data existed.

| regime | tail Gini_mult | mean | `|W_E|` |
|---|---|---|---|
| engine float64 | 0.7623 · 0.7307 · 0.7679 | 0.7536 | 11.610 |
| **engine float32** | **0.7607 · 0.7785 · 0.7622** | **0.7671** | **11.642** |
| torch float32 | 0.5573 · 0.5370 · 0.5743 · 0.5812 · 0.5825 | 0.5665 | 15.152 |

Precision moved the statistic by **+0.014 — away from torch**. Fraction of the gap
explained: **f = −0.07** against a ≥ 0.70 threshold, and `f = 0.01` on the embedding norm.
Grokking time is precision-invariant (4,500/6,400/9,600 against 4,500/6,500/9,700), which
also kills "float64 groks faster" as an explanation for **O1b**.

> **The correlation was 7/7, the mechanism was plausible, and it was wrong.** That is what
> the controlled run was for. C30 is **RETRACTED**.

**So the section title is right and its explanation was not.** Two regimes do exist and
their numbers do disagree — but the axis is **which code trained it**, not the dtype. The
engine reads ~0.76 in *both* precisions. Read "precision regime" below as a label for
provenance, never as a mechanism.

**What is left (O18).** The ~0.19 gap survives, and the elimination list is now: protocol,
hyperparameters, tail window, optimizer (both reduce to `p(1−lr·wd) − lr·m̂/(√v̂+ε)`),
gradients (2.68e-15), minibatch noise (**both are full-batch**), initialisation
distribution (per-parameter norms agree to ~1%), the train/test split (**byte-identical —
the same 3,830 pairs**), and precision. Standing: the specific init draw, and Kaggle T4 GPU
vs local CPU.

A second fact any explanation must now fit: **neuron-level tuning runs the opposite way to
Gini.** The engine has the sparser embedding and the *less* pure neurons (70–93% tuned at
Gini 0.75–0.77); torch has the reverse (92–100% tuned at Gini 0.57). One "more or less
converged" axis cannot produce that — it is the signature of a *different solution*, which
is Clock-and-Pizza territory (2306.17844) rather than a sparsity-magnitude story.

**A caution earned on the way there.** N7's E5 pre-registered, in advance, that a systematic
engine-vs-torch disagreement would be "a finding about the science, not an engine bug",
citing C10's gradient verification to 2.3e-15. Taken at face value that sentence would have
shipped a precision artifact as a scientific claim. **C10 verifies the gradients — not the
dtype, not the optimizer, not the data split.** A pre-registration that pre-explains a
failure is not permission to skip the check.

### 8.1c O18: the engine and torch run the SAME update path, to 1e-15 (O18-T stage 1)

Pre-registered `experiments/PREREGISTER_o18_transplant.md` (committed `3ebbab6` before any
transplant number existed). `run_o18_transplant.py --stage1`, `logs/o18_transplant.log`,
`results/o18_transplant/stage1.json`, ~5 min, zero quota.

Torch's nine sampled init tensors were transplanted into the from-scratch engine. The
engine holds `W_Q/W_K/W_V` as `(d_model, H·d_head)` and `W_O` as `(H·d_head, d_model)`; the
kernel uses `(H, d_model, d_head)` and `(H, d_head, d_model)`. **The mapping was not
reasoned about** — it is a gate asserted by logit equality, after three bin-convention
arguments lost to reasoning in this project.

| check | result |
|---|---|
| T0 transplant exact — same weights, same function | max \|Δlogit\| **7.772e-16**, full grid, float64 |
| T1 split byte-identical between arms | HELD — 3,830 train / 8,939 test |
| T2/T3 at 500 float32 steps | ratio **0.56** — engine-vs-torch divergence is *smaller* than the engine's own float64-vs-float32 divergence |

**The decisive measurement is the deterministic one** (added, not pre-registered): both arms
at float64 from identical weights, comparing gradients and then weights after one optimiser
step — no float32 chaos for a difference to hide in.

| step | max \|Δloss\| | max \|Δgrad\| | max \|Δw\| after one AdamW step |
|---|---|---|---|
| 1 | 1.776e-15 | **2.082e-17** | **8.646e-15** |
| 2 | 0 | 1.821e-17 | 8.660e-15 |
| 3 | 0 | 9.758e-18 | 8.646e-15 |

**The autograd graph and the optimiser are the same computation.** C10 had verified
gradients to 2.3e-15, but against a torch reference written in the *engine's* layout; this
is against the kernel that actually produced the torch numbers, which is the comparison
O18 needs.

**The instrument is exonerated as well** (exploratory). All four artifacts through one
measurement path, `analyze_n7.mult_amplitude` + `gini`:

| arm | \|W_E\| | Gini_mult | PR_mult |
|---|---|---|---|
| engine n7 (float64) | 11.795 | **0.7630** | 3.890 |
| engine n9 (float32) | 11.801 | 0.7610 | 3.896 |
| torch o18 (CPU) | 13.323 | 0.5770 | 3.236 |
| torch k03 (T4) | 15.757 | 0.5383 | 4.394 |

**The gap is in the weights.** Eliminated to date: protocol · hyperparameters · tail window
· optimiser · gradients · minibatch noise · init *distribution* · train/test split ·
precision (N9) · hardware (Entry 32) · **update path (here)** · **measurement (here)**.

**What remains, and why it does not yet satisfy.** The only enumerated candidate is the
specific init **draw** — numpy `default_rng` against `torch.Generator`, the same
distribution, a different realisation. **A draw effect should widen each framework's band,
not shift one**, and the bands are tight and separated: engine 0.73–0.77 over 3 seeds,
torch 0.54–0.58 over 5. So either the draw differs systematically in a way nobody has
measured, or the cause is not on the list. **This is the fourth hypothesis about O18. The
first three — precision, hardware, update path — were each plausible, each tied up loose
ends, and each was wrong.** Stage 2 (the engine trained 40k steps from torch's own init)
tests it directly and is running as of 2026-09-15.

#### 8.1d STAGE 2 LANDED — the init draw is dead too (O18-T T4, 2026-09-15, 1 seed)

318 min local CPU, zero quota. `results/o18_transplant/WE_transplant_n113_s0.npz`,
`git_sha=b79288d`, `git_dirty=False`, grok at step 5,800. Tail window, 11 snapshots from
step 30,000 — never the last row (C27).

| arm | tail Gini_mult | Gini_add | rand-orth | \|W_E\| |
|---|---|---|---|---|
| engine, own init (float64) | 0.7536 | — | — | 11.610 |
| **engine, from TORCH's init (float64)** | **0.7827** (sd 0.0009) | 0.0104 | 0.1577 | **11.738** |
| torch, CPU (float32) | 0.5791 | — | — | 13.323 |
| torch, T4 (float32) | 0.5665 | — | — | 15.152 |

**f = +1.156** (+1.146 at tail mean − 2 sd). Torch's own starting weights, stepped by the
engine, produce a circuit **slightly sparser than the engine's own baseline**, and \|W_E\| is
engine-like as well. The two discriminators run in *opposite* directions in the original gap,
so this is not one axis read twice. **The init draw is eliminated. ⚠️ One seed.**

**The pre-registered rule's surviving arm is purely negative.** `f > 0.7` is labelled
"UPDATE PATH", and stage 1 killed the update path to 1e-15 before stage 2 launched — so the
branch that fired names a dead hypothesis. It eliminates; it explains nothing.

**All four enumerated hypotheses are now dead:** precision (N9, f = −0.07) · substrate
(Entry 32, f = 0.068) · update path (stage 1, 8.65e-15) · **init draw (stage 2, f = +1.156)**.

**O18 is now a contradiction and should be written as one.** Identical weights, data,
forward (7.77e-16), gradients (2.08e-17) and AdamW step (8.65e-15), both full-batch and
deterministic — different endpoints. Chaos would widen each band; the bands are tight
(engine 0.73–0.77 / 3 seeds, torch 0.54–0.58 / 5 seeds) and separated.

**The un-run cell.** Every torch run in this project is **float32** —
`kernels/k03_grid_acts/run.py:60` samples with `torch.randn` at the default dtype and nothing
calls `.double()`:

| | float32 | float64 |
|---|---|---|
| engine | 0.7671 (N9) | 0.7536 |
| torch | 0.5665 / 0.5791 | **NEVER RUN** |

N9 tested precision **inside the engine**, where it is inert. That is a main effect, and it
does **not** test whether float32 is doing something to torch. Stage 1's 1e-15 agreement was
engine-float64 against torch *promoted* to float64 — a torch that has never been trained.
~35 min local CPU, zero quota; needs its own pre-registration.

### 8.1e O21: the engine's dtype invariance has no surviving candidate (2026-09-15)

**Pre-registered** `experiments/PREREGISTER_o21_dtype.md`, committed `b7108d4` **before any
dtype was read**, with the follow-up's criteria added as an amendment **before that ran**.
`run_o21_probe.py`, `run_o21_numerics.py`. Under two minutes, zero quota.

**The question, narrowly.** O18 is closed (C34): precision moves torch's `Gini_mult`
0.5791 → 0.7728, `f = +1.110`. The residue is that **the engine sits at the float64 answer
at either dtype** — 0.7671 (f32) / 0.7536 (f64), Δ **0.0135** — while **torch moves 14×
further with the same variable**, Δ **0.1945**. Nothing explained that.

**The hypothesis was mechanical, not statistical.** `Tensor.__init__`
(`src/autograd/engine.py:63`) casts `.data` to `DTYPE` **on construction**, so an
intermediate numpy promoted to float64 would be computed wide and then *rounded down* —
better than true float32, and invisible to any check that inspects a finished tensor. That
is exactly why `test_autograd.py::test_dtype_flag` passes while O21 stayed open: it inspects
parameters, parameter gradients, Adam state and the loss, all downcast before it looks.

| criterion | what it tests | result |
|---|---|---|
| **4** negative control | a planted `float32 × int64` promotion must be **detected** | detected; clean op flagged 0; constructor confirmed to round the planted result back to float32 |
| **3** positive control | at `ENGINE_DTYPE=float64` the probe must read float64 | **100.00 %**, 115/115 calls |
| **1** PRIMARY | forward sites float64 under float32 (**pre-cast**) | **0** / 12 sites, 46 calls |
| **2** | backward `_accum` gradients float64 under float32 | **0** / 16 sites, 69 calls |

**⇒ candidate (c), residual NEP-50 promotion, is DEAD.** The engine computes in float32 end
to end; the `.astype(g.dtype)` fix in `max()`'s backward (`6991152`) was the only such site.

The pre-specified follow-up then tested the other two candidates by measuring each
implementation's float32 result against a float64 reference built from the **same** float32
inputs (the round trip is exact, so any difference is arithmetic, not representation):

| case | numpy f32 | torch f32 | ratio |
|---|---|---|---|
| matmul (3830,128)@(128,512) | 5.037e-07 | 5.037e-07 | **1.000** |
| matmul, real MLP (3830,512)@(512,128) | 4.547e-07 | 3.880e-07 | **1.172** |
| log_softmax, real logits | 1.206e-07 | 1.178e-07 | **1.024** |
| log_softmax, logits × 20 | 7.656e-08 | 7.365e-08 | **1.039** |

Every ratio is inside the pre-registered equivalence band [0.5, 2.0]. **Criterion A3 fires:
candidates (a) accumulation order and (b) softmax formulation are BOTH DEAD.** numpy and
torch are numerically indistinguishable at float32 on this model's own shapes and its own
activations, including a deliberately hard high-logit case.

#### What is claimed, and what is not

**Claimed:** the engine's dtype-invariance is not caused by hidden float64 arithmetic, by
matmul accumulation order, or by the softmax formulation. **All three enumerated candidates
are eliminated under pre-registered criteria with both a positive and a negative control.**

**NOT claimed:** that the engine is "more accurate". It is not — the numerics are identical
to within 17 %. And this does **not** re-open O18, which is closed on the torch side.

**O21 stays OPEN with no surviving candidate.** Cumulatively across O18 and O21 the
eliminated list is: protocol · hyperparameters · tail window · optimiser · gradients
(2.68e-15) · minibatch noise (both full-batch) · init distribution · the byte-identical
split · the substrate (`f = 0.068`) · the update path (one AdamW step, 8.65e-15) · the init
draw (`f = +1.156`) · precision inside the engine (`f = −0.07`) · **intermediate dtype** ·
**accumulation order** · **softmax formulation**. Precision *on torch* is the one that
answered (C34).

⚠️ **Paper treatment: a Limitations line and a methods appendix, never a claim.** Two
mathematically equivalent implementations agreeing to 1e-15 on gradients and 8.65e-15 on one
AdamW step converge to measurably different circuits, and we report that we cannot say why.

### 8.2 Critical dataset size — Power et al. reproduced

n=17, mul, 3000 steps: train_frac 0.4 → 0.144, 0.6 → 0.647, **0.8 → 1.000 (grok at step
500)**. 30% works at n=113 but is far below critical at n=17. **Local smoke tests need
train_frac ≥ 0.8, or a correct implementation looks broken.**

### 8.3 Engine correctness and performance

- **23/23 ops pass central-difference gradient checks.**
- **vs PyTorch with identical weights: forward 6.1e-16, gradients (all 9 tensors) 2.68e-15.**
  *(P18, 2026-09-29: `test_model.py`, unchanged since init, now prints forward **4.441e-16** and
  2.68e-15; the paper quotes those.)*
- **4.8× faster** — 2149 → 449 ms/step at n=113/d=128, via two fixes, both proven
  equivalent by the PyTorch cross-check:
  1. matmul backward materialised a **(3830,128,512) = 2 GB** temporary before summing over
     the batch; when the second operand is unbatched that sum *is* a contraction, so fold
     the batch into the row axis and let BLAS do one gemm. (2.06×)
  2. the MLP ran at all 3 positions but only position −1 reaches the logits; the others are
     computed, discarded, and carry identically zero gradient. (2.3×)
- **Memory: `backward()` now frees the graph.** Each node's `_backward` closure captures its
  own node, so the graph was a mass of reference cycles — nothing freed by refcounting, RSS
  climbing to **12 GB in 1000 steps**. Clearing `_prev`/`_backward` after the pass gives
  **395 MB flat, +0 MB over 30 further steps**. Consequence: `backward()` may be called once
  per graph. Guarded by `test_memory.py`.

### 8.4 Bugs the tests caught that inspection would not

- Ops set `_backward` unconditionally, so `requires_grad=False` nodes (a piecewise mask
  inside `stablemax`) ran with `grad=None`.
- `stablemax`'s inactive branch propagated `inf`: the forward masks `1/(1−x)` for x ≥ 0 but
  the gradient still evaluates it.

Neither changes a shape. Both would have produced silently wrong gradients.

---

## 9. Operational findings

- **Kaggle's default GPU is unusable with Kaggle's own PyTorch.** `enable_gpu: true` yields
  a **P100 (sm_60)**; preinstalled torch 2.10.0+cu128 dropped Pascal, so every CUDA op dies.
  Fix: `"machine_shape": "NvidiaTeslaT4"` — **case-sensitive**. `"T4x2"`, `"nvidiaTeslaT4"`
  and the `--accelerator` CLI flag are all **accepted without error and silently ignored**.
  Verified environment: 2× Tesla T4, 14.6 GB, sm_75, 30 ms per 4096² matmul.
- **Max 2 concurrent batch GPU sessions.**
- **Never report `Gini_mult` alone.** A purely *additive* planted signal scores Gini_mult
  0.53–0.63 — and 2606.17399's headline 0.58 sits inside that range. Always report the
  (mult, add, random-orthogonal) triple plus the key-frequency count, which separates
  cleanly (4 vs 18–25 on synthetic controls).
- **White noise gives Gini ≈ 0.06**, so their additive 0.07 is at the noise floor — which is
  consistent with their claim, but means "additive Gini is low" carries no extra information.
