# Lab Notebook

Dated entries. What was tried, what happened, what changed, what surprised us.
Negative results are entries too.

---

## 2026-09-10 — Entry 1: environment, design validation, repositioning

### Environment (recorded per §10.2 rule 3)

- **Kaggle GPU quota: 108,000 s = 30.0 h/week. 0 s used. Resets 2026-09-12 00:00 UTC.**
  **→ Reset day is FRIDAY. Launch big grids Friday.** TPU: 20 h/week, unused.
- Kaggle account `account-a`, CLI + MCP both working.
  CLI was broken on arrival (`kaggle` not importable from `python3.12`); fixed with a
  project venv at `.venv/`.
- Local: RTX 2050, 4 GB. Enough for smoke tests and the scout run; **not** the grid.
- **Spend so far: 0 GPU-hours.**

### Design validation (Lane A) — `src/tasks/algebra.py`

Regenerated the §1.3 table computationally. **Every entry is correct.** Cyclicity
confirmed by explicit primitive-root search, not lookup. Self-check asserts the whole
table and passes.

Nit for the write-up: PLAN's zero-divisor counts exclude 0 for 121/125 (10, 24) but
include it for 120 (88). `algebra.py` includes 0 throughout. State the convention.

### The repositioning — arXiv 2607.07066 "Multiplication Beyond Groups"

Read in full. It covers composite-modulus multiplication, zero divisors, embedding
clustering, and class-sensitive attention routing. **This subsumes H2.2 and most of
H2.3.** Their moduli: 113, 143, 154, 165 — all square-free.

Their own Future Work names our gap:

> *"non-square-free moduli introduce non-regular 𝒥-classes that contain nilpotent
> elements... which breaks the reduction to local group characters. Future work must
> investigate how networks handle these algebraic 'one-way' collapses."*

Computed Green's 𝒥-class structure for the whole modulus set. Non-regular class counts:

```
113: 0    119: 0    165: 0  (square-free -- reproduces their Thm D.17 ✅)
121: 1    125: 2    120: 8  (non-square-free -- their open problem)
```

**The experimental axis is now #non-regular 𝒥-classes ∈ {0,0,1,2,8}, not "is it a field."**
H2.4's ω(n) regression is retired in favour of this — theory-derived, not curve-fitted.

- **121 = minimal clean case**: exactly one non-regular class, `J_11`, 10 elements, all nilpotent.
- **120 separates two confounded variables**: `J_2, J_4, J_6, J_10, J_12, J_20` are
  non-regular **with zero nilpotents**. So "no local inverse" ≠ "one-way collapse to 0",
  and the two can be compared *within a single model*.

Their training config ≈ our §3.4 exactly (1-layer, d_model 128, d_hidden 512, ReLU,
AdamW wd=1, lr=1e-3, betas 0.9/0.98, full-batch, 30% train, 25k epochs). **Matching it
makes our numbers directly comparable to theirs.** They also hold out a further 30% for
validation and report final accuracy on all n².

### PRE-REGISTERED targets (written before any model is trained)

From 2606.17399 for `a·b mod 113`: **Gini 0.58 (mult) vs 0.07 (add), 4 key frequencies,
96.9 % of MLP neurons single-frequency tuned.** These are H2.1's replication targets.

### Instrument validation (Lane C) — `test_analysis.py`

Built additive DFT, product-character transform (one code path covering cyclic *and*
non-cyclic unit groups via CRT), Gini / participation ratio / key-frequency count, and a
random-orthogonal control. Validated on synthetic signals with known sparsity. All pass.

**Two surprises, both important, both free:**

1. **White noise gives Gini ≈ 0.06.** So 2606.17399's reported *additive* Gini of **0.07 is
   statistically indistinguishable from noise.** That is consistent with their claim (the
   additive basis genuinely sees nothing) — but it means "additive Gini is low" carries no
   information beyond "not sparse". Fine; just don't over-read it.

2. **⚠️ A planted purely-ADDITIVE signal scores Gini_mult = 0.53–0.63.** Their headline
   **0.58 sits inside that range.** So **Gini_mult in isolation cannot distinguish a real
   multiplicative circuit from an additive one misread in the multiplicative basis.**
   → **Rule: never report Gini_mult alone.** Always report the (mult, add, random-orthogonal)
   triple *and* the key-frequency count. Key-freq count does separate cleanly
   (4 vs 18–25 in the synthetic controls).

3. **n=120 is a weak cell for basis analysis.** φ(120)=32 with factors (2,2,2,4) — most
   characters are real and the mult/add distinction partly collapses (a planted
   multiplicative signal still reads Gini_add = 0.76). Treat 120 as the stress case for
   𝒥-class structure, **not** as a basis-sparsity data point.

### Next

Lane D (Kaggle harness + `k00_smoke`), then the scout run on the **local** GPU
(zero Kaggle quota): does `mul-121` grok, and is `J_11` visible in the embedding?

---

## 2026-09-10 — Entry 2: Kaggle harness (Lane D), and a finding that would have cost a grid

### ⚠️ Kaggle's default GPU is unusable with Kaggle's own PyTorch

`k00_smoke` failed on first push:

```
torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device
Tesla P100-PCIE-16GB with CUDA capability sm_60 is not compatible with the current
PyTorch installation. The current PyTorch install supports sm_70 ... sm_120.
```

`enable_gpu: true` alone gets you a **P100 (sm_60, Pascal)**, and Kaggle's preinstalled
**torch 2.10.0+cu128 dropped Pascal**. Any GPU kernel using default settings dies on the
first CUDA op. **This would have killed an 8-hour grid at minute 3.**

**Fix — the metadata key is `machine_shape`, and the value is case-sensitive:**

```json
"machine_shape": "NvidiaTeslaT4"
```

`"T4x2"`, `"nvidiaTeslaT4"` (lowercase n) and the `--accelerator` CLI flag were all
**accepted without error and silently ignored** — the kernel still ran on P100. The valid
string was recovered from an existing working kernel via the API (`machine_shape` field).
**Put `machine_shape: "NvidiaTeslaT4"` in every GPU kernel-metadata.json.**

`k00` now carries a permanent guard asserting `get_device_capability()` is in
`torch.cuda.get_arch_list()`, so this failure mode announces itself in 30 seconds forever.

### Verified Kaggle environment (T4 path)

```
torch 2.10.0+cu128 | 2 x Tesla T4 | 14.6 GB each | sm_75 | 4096^2 matmul: 30.0 ms
python 3.12.13 | Linux 6.12.90
```

**Other operational limits found:** max **2 concurrent batch GPU sessions** (pushing a 3rd
returns `Maximum batch GPU session count of 2 reached`).

### Scout smoke (k01, 300 steps, n=121 & 113)

Plumbing green. Both at **train_acc = 1.000, test_acc = 0.02–0.05** — the memorization
phase, as expected pre-grokking. Throughput ≈ 300 steps / 12 s → **~17–20 min per modulus
at 25k steps**, so the 6-modulus scout ≈ 2 h.

Early signal worth watching for H1.4: at step 300 the **max logit is 156 for n=121 vs 54
for n=113** — the composite modulus is driving logits ~3x harder. If Softmax Collapse
shows up anywhere, expect it on composite moduli first.

### Spend

k00 (5 versions incl. failures) + k01 smoke ≈ **4 min GPU**. Full scout ≈ 2 h. Running.

---

## 2026-09-10 — Entry 3: SCOUT RESULT. Every modulus groks; the clock survives prime powers.

**Status of this data: DE-RISKING ONLY.** PyTorch autograd, hand-written architecture,
**one seed**. Nothing here is a claim until it is re-run on the from-scratch engine with
5 seeds. Recorded now because it decides the plan.

### All six moduli grok. Gate 2's worst case is eliminated.

| n | non-reg 𝒥 | grok step | final test acc |
|---|---|---|---|
| 113 | 0 | **6,400** | 1.0000 |
| 165 | 0 | 6,900 | 0.9999 |
| 121 | 1 | 7,700 | 1.0000 |
| 125 | 2 | 9,500 | 1.0000 |
| 120 | 8 | 10,700 | 1.0000 |
| 119 | 0 | **15,900** | 0.9991 |

"Composite moduli don't grok" is dead. 37 min of GPU for the whole thing.

**H2.5, restricted to the cyclic family, is monotone in non-regular 𝒥-class count:**
113 (0) → 6,400 < 121 (1) → 7,700 < 125 (2) → 9,500. Clean.
**119 breaks any global ordering** — slowest by 50%, despite having zero non-regular
classes. It is the only modulus that is square-free *and* non-cyclic *and* zero-divisor-poor.
One seed. Do not interpret yet.

### H2.1 replicates, and extends to prime powers — the thesis cell works

| n | Gini_mult | Gini_add | Gini_rand | Gini_rand(units) | key_mult | key_add |
|---|---|---|---|---|---|---|
| 113 | 0.902 | 0.040 | 0.184 | 0.221 | **4** | 51 |
| 165 | 0.885 | 0.877 | 0.191 | 0.208 | 5 | 8 |
| 119 | 0.853 | 0.566 | 0.156 | 0.172 | 7 | 40 |
| **121** | **0.879** | 0.364 | 0.166 | 0.179 | **4** | 50 |
| **125** | **0.905** | 0.437 | 0.191 | 0.203 | **4** | 49 |
| 120 | 0.813 | 0.869 | 0.214 | 0.182 | 5 | 7 |

**Pre-registered target (2606.17399, n=113): Gini_mult 0.58, Gini_add 0.07, 4 key freqs.
Ours: 0.90 / 0.04 / 4 key freqs.** Key-frequency count matches **exactly**; the additive
Gini matches (and sits at our measured noise floor of ≈0.06, as Entry 1 predicted).
**Gini_mult is higher than theirs (0.90 vs 0.58)** — a protocol difference to chase
(centring? "=" token included? |Â| vs |Â|²?). Per Entry 1's rule we lead with the
key-frequency count, which is the robust statistic, and it replicates exactly.

**→ The discrete-log clock survives at prime powers.** n=121 and n=125 are local rings
with nilpotents and non-regular 𝒥-classes, and they are **as sparse in the multiplicative
basis as the field is, with the same 4 key frequencies.** Cyclicity of the unit group, not
primality and not regularity, is what the clock mechanism needs. That is H2.1′ — the cell
2607.07066 named as its open problem — and it has a positive answer.

### UNEXPECTED, and possibly the better result: the additive basis has a job too

The additive column is not noise for composites. It varies from 0.040 (n=113) to 0.877
(n=165), and it is **monotone in zero-divisor density, r = 0.911 over 6 moduli**:

```
n=113  non-units   1/113 = 0.9%   Gini_add 0.040   key_add 51
n=121  non-units  11/121 = 9.1%   Gini_add 0.364   key_add 50
n=119  non-units  23/119 = 19.3%  Gini_add 0.566   key_add 40
n=125  non-units  25/125 = 20.0%  Gini_add 0.437   key_add 49
n=165  non-units  85/165 = 51.5%  Gini_add 0.877   key_add  8
n=120  non-units  88/120 = 73.3%  Gini_add 0.869   key_add  7
```

**Mechanism, confirmed directly:** a 𝒥-class `J_d = {x : gcd(x,n) = d}` is a union of
arithmetic progressions of common difference d — precisely the structure the *additive*
DFT is sparse in. Measured on the raw 𝒥-class indicator functions, with no network involved:

```
Gini_add(J-class indicators):  n=113: 0.018   n=121: 0.829   n=165: 0.829   n=120: 0.683
```

n=113 is ≈0 because a field has no stratification to encode (J_1 and {0} only).

**So the two bases do two different jobs:**
> **The additive basis encodes *which stratum* an element is in; the multiplicative basis
> does the arithmetic *within* the stratum.** Additive sparsity therefore tracks how much
> stratification exists (zero-divisor density); multiplicative sparsity stays high
> everywhere the unit group is cyclic.

This says *why* the mechanism is "stratified" in 2607.07066's sense, and *which basis does
which half*. It is a sharper H2.6 than "match the basis to the structure": the claim
becomes **two bases, two roles, with a quantitative predictor for each.**

### Caveats before any of this leaves the notebook

1. **One seed.** Everything needs 5.
2. **PyTorch autograd, not our engine.** Must be re-run on the from-scratch engine.
3. **r = 0.911 is 6 points.** The extended modulus set (§3.2) is now clearly worth running.
4. Gini_mult 0.90 vs published 0.58 is unexplained. Resolve before citing either.

### Spend

Scout: **37 min GPU**. Cumulative ≈ **41 min of 30 h** (2.3%).

---

## 2026-09-10 — Entry 4: author repos cloned; replication tightened; protocol pinned

### We were recreating from paper text without checking for reference code. Fixed.

Cloned into `reference/` (for **checking**, never copied into `src/` — constraint #2):

| repo | paper | contents |
|---|---|---|
| `Discrete-Log-Clock` | 2606.17399 | full analysis notebook (the 0.58 Gini computation) |
| `interpreting-monoids` | 2607.07066 | `core/`, `experiments/`, **and their trained n=165 checkpoint `P165_d128_h4_mlp512_s1.pt`** |
| `progress-measures-paper` | Nanda 2301.05217 | not yet read |

**`interpreting-monoids` shipping a trained checkpoint turns H2.7 from "retrain and hope"
into "run our pipeline on their published weights."** Do that before the 5-seed run — it
validates the analysis code against ground truth at zero GPU cost.

### The 0.90-vs-0.58 discrepancy was protocol, not science

Their `gini_coefficient` is algebraically identical to ours. The input is not:
they feed **amplitude** (`fourier_embed.norm(dim=-1)`, sin/cos combined as
`sqrt(||s_k||^2 + ||c_k||^2)`, DC dropped); we fed **squared energy**. Squaring inflates
Gini 0.55 → 0.90.

**Adopted their protocol as the reporting default.** Also added their key-frequency
detector (`norm > 5 x median`) as `key_freqs_5x_median`; ours used a 90%-energy threshold.
Both give 4 at n=113, but only theirs is comparable to the paper.

### Replication, n=113, under their exact protocol

| metric | published | ours |
|---|---|---|
| Gini additive | 0.071 | **0.072** |
| IPR additive | 52.7 | **55.0** |
| Gini multiplicative | 0.579 | **0.550** |
| key frequencies | 4 — {2, 8, 47, 56} | 4 — {13, 27, 33, 43} |
| IPR multiplicative | 4.1 | 12.3 |

Primitive root g=3, matching the paper. Different frequency *values* are expected — they
are seed-dependent; the count is the invariant.

**Open: IPR 12.3 vs 4.1.** Same 4 peaks, more background energy. Two protocol differences,
both testable:
- **they train 40,000 epochs; our scout ran 25,000** → under-trained, spectrum not yet clean.
  This is exactly H5.3 ("At-Grok Is Not Converged") as a falsifiable prediction:
  **train to 40k and IPR should fall toward ~4.**
- **they exclude zero**: `W_E` is 112 x 128 over units only, 3,763 of 12,544 = 112^2 pairs.
  Our scout trained on all n^2 pairs including 0 and the zero divisors.

### Design decision this forces — TWO datasets, not one

- **Replication arm:** units only, 40k epochs. Matches 2606.17399 exactly. Only this arm
  may be compared to their numbers.
- **Thesis arm:** all n^2 pairs, zero divisors included — they are the object of study.

Conflating these would confound every cross-modulus comparison, because the unit fraction
varies enormously across the modulus set (113: 99.1%, 120: 26.7%). **Record which arm every
future result came from.**

### Reading log

- **2606.17399** — read line-by-line through §4.1. Task/architecture/hyperparameters,
  discrete-log construction, both bases, the 4-step analysis pipeline, basis-comparison
  results. Their grokking window: epochs 9,000–14,000 (ours at n=113: 6,400 — faster,
  plausibly the zero-inclusive dataset and/or init).
- **2607.07066** — core read done (abstract, setup, moduli table, layer-by-layer results,
  limitations, appendix D 𝒥-class theory). Full line-by-line pending.
- **2301.05217** (Nanda) — repo cloned, not yet read.

---

## 2026-09-10 — Entry 5: pipeline validated on published weights; a sharp new law

### Our analysis pipeline agrees with 2607.07066's own trained model

Ran our code on `reference/interpreting-monoids/experiments/P165_d128_h4_mlp512_s1.pt`
(their published n=165 checkpoint, seed 1, final loss 1.96e-05). **No training, no GPU.**

| quantity | their checkpoint | our scout n=165 |
|---|---|---|
| Gini multiplicative (product-character) | **0.884** | **0.885** |
| participation ratio | **4.85** | **4.76** |

Two independently trained models, two codebases, same numbers. **The analysis pipeline is
validated against published ground truth (H2.7 discharged).**

Their central claim also replicates: embedding organizes by 𝒥-class, separation ratio
(between/within) **1.03 vs 0.63 for a shuffled-label control**. The absolute separation is
modest; the control is what makes it meaningful. Report it that way.

### NEW RESULT — the key additive frequencies are the CRT-component duals

The 5x-median detector on their n=165 checkpoint returns
**{15, 30, 33, 45, 55, 60, 66, 75}**. With 165 = 3·5·11:

```
165/11 = 15  -> 15, 30, 45, 60, 75
165/5  = 33  -> 33, 66
165/3  = 55  -> 55
```

Eight predicted, eight observed, nothing else. Stated as a law:

> **The key additive frequencies of a grokked `a*b mod n` transformer are exactly the
> multiples of `n/q`, one family per maximal prime power `q || n` — the additive duals of
> the CRT components.**

**Mechanism:** a function supported on multiples of `d` has additive Fourier support on
multiples of `n/d`. 𝒥-classes are unions of arithmetic progressions generated by the CRT
components, so the stratification's additive signature is exactly this lattice.

**Verification (exact set equality, no misses, no extras):**

| n | factorization | predicted = observed | source |
|---|---|---|---|
| 165 | 3·5·11 | {15,30,33,45,55,60,66,75} | **their published checkpoint** |
| 165 | 3·5·11 | {15,30,33,45,55,60,66,75} | our scout, independent training |
| 120 | 2³·3·5 | {15,24,30,40,45,48,60} | our scout |

n=120 is the sharper test: the generating duals are `120/8=15`, `120/3=40`, `120/5=24` —
**maximal prime powers, not primes.** Using n/p instead of n/q predicts {24,40,48,60} and
misses {15,30,45}. The prime-power form gets all seven exactly.

**Where it is silent, and why that is consistent:**
- **113, 121, 125** — one CRT component, so `n/q = 1` and the law predicts nothing
  distinguished. Observed: detector returns **empty**. A prime power has no additive
  stratification to encode. Consistent, but the prediction is vacuous there — do not
  count these as confirmations.
- **119 = 7×17** — law predicts multiples of 17 and 7; detector returns **empty** because
  zero-divisor density is only 19.3% and nothing clears 5x median. Consistent with Entry 3's
  density finding, but **currently unverified**. Lower the threshold or add seeds to test it.

**Status: verified on 2 moduli / 3 models, one of them third-party. Single seed each.**
The honest claim is "exact where the signal is detectable"; 119 is the open case that would
turn this from a pattern into a law.

### Why this matters

It is the **additive-side complement to the discrete-log clock**, and it makes 2607.07066's
"stratified Fourier mechanisms" quantitative: not just *that* the computation stratifies,
but *exactly which frequencies* carry the stratification, derived from n's factorization
with no free parameters.

Combined with Entry 3: **additive basis = which CRT stratum (frequencies n/q);
multiplicative basis = the arithmetic within it (4-7 characters).**

### Architecture differences vs `interpreting-monoids` (for the write-up)

Their `ModMultDecoderOnly`: `nn.MultiheadAttention`, **no positional embedding**, `nn.Linear`
with **biases**. Ours: hand-written attention, learned positional embedding, **no biases**
(Nanda's setup). Both grok; both give the same spectra. Worth a sentence in Methods.

---

## 2026-09-10 — Entry 6: an answer to 2607.07066's open problem

### The theory, read line-by-line (§3–4 of 2607.07066)

- **Thm 3.4:** every *regular* 𝒥-class is a group, `J_d ≅ (Z/(n/d)Z)^×`, with idempotent
  `e_{J_d}` as local identity and local inverse `c♯` s.t. `c c♯ = e_{J_d}`.
- Their decoding rule: `Logit(c) ∝ χ_{ρ_d}(a b c♯)` — a local-inverse test inside a class.
- **This needs the class to be a group.** Non-regular classes have no idempotent, hence no
  `c♯`, hence no such rule. That is exactly their stated open problem.

### The 𝒥-class multiplication table makes the difference one line long

`J_d · J_e → J_{gcd(de, n)}`. Computed in `algebra.py::j_multiplication_table`:

```
n=165, 119 (square-free, depth 1):  diagonal is CLOSED   J_3*J_3 = J_3, J_7*J_7 = J_7
n=121 (depth 2):                    J_11*J_11 = J_121 = {0}     null semigroup, one step
n=125 (depth 3):                    J_5*J_5 = J_25 -> J_125 = {0}   graded, two steps
```

**Square-free: `J_d·J_d = J_d`, closed, a group. Prime power: `J_d·J_d ≠ J_d`, it descends
toward zero.** That is precisely why the local-character account breaks — and 121 vs 125
gives us nilpotency depth 2 vs 3 as a clean graded variable.

### THE RESULT: non-regular classes still carry character structure

Index `J_d` not by a group law on itself (there is none) but by the **unit-group action**:
`x = d·u` with `u` a unit mod `n/d`. Then DFT in that coordinate. Control: a random
orthogonal basis on the same rows. Scout data, n=121 and n=125:

| n | class | regular | order | Gini | random control | PR |
|---|---|---|---|---|---|---|
| 121 | **J_11** | **NO** | 10 | **0.368** | 0.124 | 3.30 |
| 121 | J_1 | yes | 110 | 0.521 | 0.130 | 14.82 |
| 125 | **J_5** | **NO** | 20 | **0.514** | 0.156 | 4.55 |
| 125 | J_25 | NO | 4 | 0.145 | 0.099 | 1.85 |
| 125 | J_1 | yes | 100 | 0.587 | 0.142 | 11.66 |
| 119 | J_7 | yes | 16 | 0.437 | 0.130 | 4.63 |
| 165 | J_15 | yes | 10 | 0.452 | 0.127 | 2.85 |

**3x over control on both large non-regular classes.** ~3–4 active characters each, the same
order as the unit group uses.

> **Answer to their open problem:** the network does not need local invertibility. A
> non-regular 𝒥-class is not a group, but it *is* a set acted on transitively by the unit
> group (`x = d·u`). The network represents the **action coordinate**, and builds characters
> of the acting group rather than of the class itself. Local inverses are sufficient for the
> mechanism, not necessary.

### Caveats

- **Single seed.** All of it.
- Small classes (`J_25` size 4, `J_33` size 4, `J_17` size 6) sit at Gini 0.08–0.16, barely
  over control — **too small to resolve, do not interpret.** Only `J_11` (n=121, size 10)
  and `J_5` (n=125, size 20) are large enough.
- Classes whose unit action is non-cyclic (`J_1`, `J_3`, `J_5`, `J_11` at n=165; `J_1` at
  n=119) are skipped — they need the product-character path wired into the class-restricted
  case. `transforms.py` already has it; `jclass_spectra.py` does not yet call it. **TODO.**
- Scout data: PyTorch autograd, 25k epochs. Not paper data.

---

## 2026-09-10 — Entry 7: Lane B complete. From-scratch engine verified and grokking.

### Built

| file | what |
|---|---|
| `src/autograd/engine.py` | reverse-mode autograd over numpy. Tensor + 23 ops, iterative topological sort (recursion blows the stack on deep graphs), `_unbroadcast` factored out and tested directly |
| `src/autograd/nn.py` | max-subtracted softmax, log_softmax, cross_entropy, stablemax, stablemax_cross_entropy, AdamW with decoupled decay |
| `src/model/transformer.py` | 1-layer decoder-only, hand-written attention via matmul/transpose/reshape, no LayerNorm, no biases. Same architecture as the scout |
| `src/train/loop.py` | full-batch loop + modular data generator |
| `test_autograd.py`, `test_model.py`, `test_grok.py` | the acceptance tests |

### Verification — this is the Gate 1 technical requirement

- **23/23 ops pass central-difference gradient checks** (rel. err < 1e-4), including
  broadcasting in add/mul/matmul, batched matmul, `getitem` with repeated indices,
  `max` with tie-splitting, and the `_unbroadcast` adjoint identity tested on its own.
- **Forward pass vs PyTorch reference with identical weights: max abs err 6.1e-16.**
- **Gradients vs PyTorch, all 9 parameter tensors: worst relative error 2.3e-15.**
- Loss agrees to 1e-10.

The engine is correct to machine precision. Any future disagreement with PyTorch is a
science difference, not an implementation bug.

### Two bugs found by the tests, both of the kind inspection would miss

1. **Ops set `_backward` unconditionally**, so a node with `requires_grad=False` (a
   piecewise constant mask, e.g. inside `stablemax`) still carried a closure that ran with
   `grad=None`. Fixed with a guard in `backward()`.
2. **`stablemax`'s inactive branch propagated inf.** The forward masks `1/(1-x)` out for
   x>=0, but the gradient still evaluates it, giving inf/nan at x near 1. Fixed by feeding
   each branch a masked input so both denominators stay >= 1.

Neither changes a shape. Both would have produced silently wrong gradients.

### H1.4 — Softmax Collapse, CONFIRMED on our own engine

Exactly the exposure §8 predicted from writing our own autograd. n=17, mul, 40% train:

```
step      loss        |grad|      maxlogit   p_correct
   0    2.871e+00    4.150e-01       0.5     0.037533
 400    4.706e-04    1.764e-03      47.0     0.998725
 800    5.721e-06    3.208e-05      46.1     0.999984
1200    1.299e-07    9.214e-07      57.6     1.000000
```

**Gradient norm falls 5.5 orders of magnitude while `p_correct` reaches exactly 1.000000.**
Prieto et al.'s no-gradient regime, measured directly. After that only weight decay moves
the weights.

**StableMax partially mitigates:** at step 4000 it holds |grad| = 2.3e-6 vs softmax's
2.3e-7 — **~10x more gradient at the same logit scale.** Mechanism confirmed. Neither
grokked at n=17/40%, but that turned out to be a different cause:

### Critical dataset size — Power et al. reproduced on our engine

n=17, mul, 3000 steps, everything else fixed:

| train_frac | train pairs | best test acc | grok step |
|---|---|---|---|
| 0.4 | 115 | 0.144 | — |
| 0.6 | 173 | 0.647 | — |
| **0.8** | **231** | **1.000** | **500** |

**The engine groks.** The n=17 failure at 40% was below critical dataset size, not a bug.

**Operational consequence:** 30% train works at n=113 (scout) but is far below critical at
n=17. **Local smoke tests must use train_frac >= 0.8**, or a correct implementation looks
broken. Added to `LAB_PROTOCOL.md`.

### Cost note

numpy float64 CPU: ~13 ms/step at n=17/d=64. Extrapolating to n=113/d=128 (~33x the data,
4x the width) gives ~1–2 s/step, so a 6,400-step grok run is ~3 hours of local CPU. Feasible
overnight, at zero Kaggle quota. Decide at Gate 1 whether the engine is load-bearing for the
grid or stays a validated artifact with PyTorch running the science.

---

## 2026-09-10 — Entry 8: knowledge graph read; a near-miss on the CRT-dual claim

### The graph

`related papers/graphify-out/` — 339 nodes, 334 edges, 92 communities, built over the
**bibliography metadata + downloaded PDFs index**, not full paper texts (the report itself
notes the corpus is ~3,850 words and says a graph may not be needed at that size). So treat
it as a **map of the literature's structure**, not a substitute for reading.

Useful output: the hyperedge **"Modular Multiplication Fourier/Character Mechanism Cluster"
= {2606.17399, 2606.23044, 2607.07066}**. We had read two of the three.

### The third one nearly scooped us — read it

**2606.23044 Prime Fourier Embeddings.** Graph node labels flagged it: *Chinese Remainder
Theorem*, *Block-Diagonal Decomposition Theorem*, *Adelic Harmonic Analysis of Q*,
*Schur's Lemma (Applied to Embedding Equivariance)*.

From the abstract: PFE encodes integers as prime-indexed (cos, sin) pairs from the harmonic
analysis of Q. They prove any linear map equivariant to the product group action must be
**block-diagonal, one block per prime** (Schur). **"For square-free composite moduli, the
Chinese Remainder Theorem predicts which prime channels are task-relevant."** Confirmed by
ablation with specialization ratios > 500x.

Moduli tested: `{105, 165, 231, 385}` and `{15, 21, 33, 35, 55, 77}` — **all square-free.**
Task: **addition**. Embedding: **imposed by design, not learned.**

### HONEST CORRECTION to Entry 5

Entry 5 framed the CRT-dual law as novel. **That framing was too strong.** "CRT predicts
which frequency channels matter" is substantially their claim. What remains genuinely ours:

| | 2606.23044 | ours |
|---|---|---|
| embedding | **imposed** (PFE, by construction) | **learned** — emerges under gradient descent |
| task | addition | **multiplication** |
| moduli | square-free only | includes **n=120 = 2³·3·5, non-square-free** |
| object | which channels a designed basis needs | which frequencies a standard embedding **develops** |

So the defensible claim is narrower and should be stated as:
> *A transformer with a standard learned embedding, trained on modular multiplication,
> spontaneously develops the frequency structure that PFE builds in by hand — and it does so
> at non-square-free moduli, where the plain CRT channel argument does not directly apply.*

The n=120 result is the load-bearing part. Do not claim the CRT connection itself as new.

### Consistent pattern across all three papers in the cluster

**2606.17399** (prime p only), **2607.07066** (square-free), **2606.23044** (square-free).
**None touches prime powers.** The non-square-free / nilpotent territory is open across the
whole cluster, which strengthens the thesis positioning established in Entry 1.

### Also worth adopting from 2606.23044

Their **ablation specialization ratio** (task-relevant vs task-irrelevant channel, >500x) is
a cleaner causal metric than our correlational Gini comparisons. Consider adding it to the
analysis toolkit — it turns "this basis is sparse" into "ablating these channels breaks the
task and ablating those does not."

### Graph's other communities, for later

Unread-but-relevant clusters the graph surfaced: *Legendre PRF Cryptanalysis* (11 nodes —
directly bears on H3.1's risk that the Legendre task is unlearnable); *Circuit
Synchronization & Early Detection* + *ILDR* (H5.1/H5.2 infrastructure); *Learning
Pseudorandom Generators* (FM/UM tasks, relevant to T4).

---

## 2026-09-10 — Entry 9: full reads of PFE and Nanda

### 2606.23044 Prime Fourier Embeddings — read in full

**What it is:** a *designed* embedding, not an analysis of learned ones.
`PFE_{p,d}(a) = [cos(2πa/p^{d+1}), sin(2πa/p^{d+1})]`, d = 0..D−1, **D = 3** — so the
channels are indexed by **prime powers** p, p², p³, over the 16 odd primes 3..59
(p=2 excluded: sin(2πa/2)=0 is structurally degenerate). Embedding dim 2·16·3 = 96.

**Theorem 3.1 (Block-Diagonal Decomposition):** any linear map equivariant to the product
action of ∏_{p,d} Z/p^{d+1}Z is block-diagonal, one block per (p,d), zero cross-coupling.
Proof: each block is the real form of a character of Z; distinct (p,d) give non-isomorphic
1-d irreps; Schur's lemma kills the intertwiners.

**Setup — important, it is not our setting:** shared per-prime MLP encoder (24→64→32) plus a
2-layer classifier. **Not a transformer. PFE features are fixed and non-trainable.** Task is
**addition**. Adam lr 3e-3, 80k sampled pairs, 80/20 split, **single seed 42, no error bars.**

**Moduli:** `{15,21,33,35,55,77}` and `{105,165,231,385}` — **all square-free.**

**Result:** ablate a prime row, measure accuracy drop. Factor/non-factor specialization ratio
**> 500× (capped)**, perfect in-distribution test accuracy everywhere.

### Their Open Questions section names our territory

> *"the **within-prime depth structure is not fully understood**. Schur's lemma forces zero
> linear coupling across depth levels within the same prime, but the depth levels are not
> genuinely independent... The theory predicts zero cross-depth coupling but **says nothing
> about how gradient descent distributes importance across depth levels**."*

**Depth d>0 only carries the task when N has a repeated prime factor.** For square-free N,
depth 0 is sufficient and the depth axis is idle — which is why their whole sweep leaves it
untested. **n = 121 = 11² and n = 125 = 5³ are exactly the case that exercises it**, and 125
sits at their depth cap D=3.

They also flag: *"why gradient descent converges to an equivariant solution rather than
breaking symmetry remains open."* We observe that structure emerging in a **learned**
embedding, which is evidence about that question rather than an answer to it.

### THREE papers, three independently-stated open problems, one territory

| paper | scope | their stated gap |
|---|---|---|
| 2606.17399 Discrete-Log Clock | prime p only | mechanism established for p; untested beyond |
| 2607.07066 Beyond Groups | square-free | *"how networks handle these algebraic 'one-way' collapses"* (non-regular 𝒥-classes, nilpotents) |
| 2606.23044 Prime Fourier Embeddings | square-free | *"how gradient descent distributes importance across depth levels"* (= prime powers) |

**All three stop at square-free. All three name non-square-free / prime-power as open.**
That is the thesis position, and it is now supported by three independent citations rather
than one.

### 2301.05217 Nanda — read in full

Numbers for `a+b mod 113`: **5 key frequencies k ∈ {14,35,41,42,52}**; `W_L` approx. rank 10
(cos/sin of each), residual **< 0.55%** of Frobenius norm; **84.6% of 512 neurons** have >85%
variance explained by a single frequency; ablating the other 95% of frequencies **improves**
loss (down 70% to 7.24e-8); projecting MLP activations onto the 10 directions cuts loss 50%,
projecting onto their nullspace gives loss 5.27 (worse than uniform).

**They use the Gini coefficient of Fourier-component norms too** — same metric, same
amplitude convention as Discrete-Log Clock. Our protocol is in that lineage. Good.

**Progress measures.** *Restricted loss*: 2D DFT the logits over (a,b), keep only the constant
term and the 20 terms `cos/sin(w_k(a+b))` for the 5 key frequencies, zero everything else.
*Excluded loss*: remove **only** the key frequencies, keep the rest, **measured on the
training set** — memorization is spread across the Fourier domain, so it survives; the
generalizing circuit does not. Three phases: memorization → circuit formation → cleanup, with
**grokking occurring during cleanup, after the generalizing mechanism already exists.**

### Methodological consequence for T2 — H5.1 gets easier

H5.1 assumed we need task-agnostic progress measures for composite moduli "because the
correct circuit is unknown in advance." **It is no longer unknown.** The CRT-dual law
predicts the key additive frequencies from the factorization alone, so **restricted and
excluded loss can be constructed for `mul-n` at composite n** the same way Nanda does for
addition. That converts H5.1 from "we need a substitute" into "we can use the real thing,
and validate the substitutes against it."

### Also worth adopting

PFE's **ablation specialization ratio** (relevant vs irrelevant channel drop) is causal where
our Gini comparisons are correlational. Adding it would upgrade "this basis is sparse" to
"ablating these frequencies breaks the task and ablating those does not" — which is exactly
Nanda's §4.4 correctness check, and the standard the field expects.

---

## 2026-09-10 — Entry 10: Gate 1 launched; engine made 4.8x faster; N5 closed

### TDD first: the gate is a test, written before the run

`test_gate1.py` encodes H1.1 plus Nanda's published reference numbers, **committed before
any Gate 1 training started** so the criteria cannot drift to fit the result:

- C1 test accuracy > 99%
- C2 ≤ 10 key frequencies (Nanda: 5, at k ∈ {14,35,41,42,52})
- C3 embedding sparse in the additive basis
- C4 ablating key frequencies destroys performance
- C5 ablating all NON-key frequencies does not harm (Nanda: it *improves*, −70%)
- C6 every key frequency individually necessary
- C7 neurons single-frequency tuned (Nanda: 84.6%)

It ran red (no checkpoint) before the run began, as it should.

### New: `src/analysis/ablation.py` (this is N3, the biggest methodological gap)

Nanda §4.4 Fourier-space ablation: logits over all (a,b) → (n,n,n_out), 2D DFT over the
input axes, zero selected components, invert, measure loss delta. Restricted = keep only
key frequencies; excluded = remove only key frequencies.

**Validated TDD-style against a synthetic Clock circuit** (`test_ablation.py`) whose
mechanism is known by construction: accuracy 1.0000, restricted ≤ baseline, excluded
+3.71 nats, **mean key-frequency ablation delta +0.377 vs max non-key delta 5.5e-17**.
Conjugate closure is enforced — ablating (k,k) without (−k,−k) leaves an imaginary residue
and a meaningless loss.

Sparsity says a basis *describes* the model; ablation says a component is *responsible*.
Every mechanism claim from here routes through this.

### Engine made 4.8x faster — 2149 → 449 ms/step at n=113, d=128

Profiled rather than guessed. Two fixes, both verified by the existing tests:

1. **matmul backward built a per-batch gradient before summing.** For `x @ W` with x
   (3830,3,128) and W (128,512), `swapaxes(x) @ g` materialises a **(3830, 128, 512)
   temporary = 2 GB** and only then sums over the batch. When the second operand is
   unbatched that sum *is* a contraction, so fold the batch into the row axis and let BLAS
   do one gemm. **2149 → 1044 ms/step (2.06x).**
2. **The MLP ran at all 3 positions but only position −1 reaches the logits.** The MLP is
   position-wise, so positions 0..T−2 are computed and discarded and their gradients are
   identically zero. Evaluating only at the read-out position is exactly equivalent.
   **1044 → 449 ms/step (2.3x).**

**Both verified by `test_model.py`, which compares against a PyTorch reference that DOES
compute all positions: worst relative gradient error 2.68e-15.** That is the proof these
are equivalences, not approximations. Having the test suite first is what made optimising
safe — this is the payoff for TDD, recorded as such.

25k steps: 14.9 h → **3.1 h**.

### `no_grad` added to the engine — a memory bug that would have killed the run

First launch sat at **10.4 GB RSS against 14 GB of RAM, 1.7 GB free.** Cause: evaluation
(every 100 steps) and checkpointing (all 12,769 pairs, 3.3x the training batch) both built
full autograd graphs that are never used. Added `engine.no_grad` — the engine owns whether
a graph is built — plus chunked snapshots. RSS 10.4 → 9.1 GB, free 1.7 → 3.2 GB.
`test_autograd.py` gained a `no_grad` case: no graph, state restored, identical values.

Cost 8 minutes of restart; would have OOM'd at the step-2500 snapshot and lost hours.

### N5 closed — and it forced a correction to C7

`jclass_spectra.py` now uses the product-character path uniformly, so classes whose unit
action is **non-cyclic** are covered (`J_1` at n=119 with factors (6,16); `J_1/J_3/J_5/J_11`
at n=165; all of n=120). Cyclic is just the one-factor case.

**Every J-class beats its random-basis control (all ratios > 1.9), regular or not** — so
character structure is present on classes that are not groups. That part of C7 holds.

**But the aggregate "regular 3.71x vs non-regular 2.72x" is CONFOUNDED BY CLASS SIZE.**
`J_1` is always regular and always the largest class, and Gini rises with size here. The
valid comparison is size-matched within one model, available only at n=120:

```
n=120, size 8:   regular J_5 2.82x, J_8 1.95x   |   non-regular J_4 2.43x, J_6 2.07x
n=120, size 16:  regular J_3 3.21x              |   non-regular J_2 2.93x
```

**They overlap. At one seed, "regular classes carry more character structure" is NOT
supported.** C7 in STATE.md is revised accordingly: the supported claim is only that
non-regular classes carry character structure above control. The k02 5-seed grid will
settle whether the graded version survives.

### Running

- **Gate 1** (local CPU, zero quota): `a+b mod 113`, 30% train, 25k steps, ~3.2 h.
- **k02 grid** (Kaggle GPU): Arm A = O1 replication test (n=113, units only, 40k epochs —
  does IPR_mult fall from 12.3 toward 4.1?); Arm B = 6 moduli × 5 seeds × 40k, all pairs.
  ~5 h estimated, 7 h wall guard.

---

## 2026-09-11 — Entry 11: memory leak root-caused; O3 resolved threshold-free

### The leak was reference cycles in the autograd graph

`no_grad` (Entry 10) fixed evaluation but RSS still climbed 9.1 → 12.0 GB in 1000 steps,
with 460 MB free. Root cause: **every node's `_backward` closure captures that same node**,
so the graph is a mass of reference cycles. Nothing is freed by refcounting; RSS grows
until the generational collector happens to run, and with ~12 MB of float64 per
intermediate that is 12 GB in a thousand steps.

**Fix:** after the backward pass, clear `_prev` and reset `_backward` on every node in the
topological order. Every intermediate becomes refcount-collectable immediately.
Consequence, stated in the code: `backward()` may be called once per graph — we never call
it twice.

```
before: 12,000 MB and climbing
after:     395 MB, +0 MB over 30 further steps
```

**30x reduction, zero growth.** `test_model.py` still 2.68e-15, `test_autograd.py` still
passes. Promoted the check to `test_memory.py` and wired it into `status.py` — this cost
two restarts and should never recur silently.

Live run: RSS 1.67 GB, 10.6 GB free (was 12.0 GB / 460 MB).

### O3 RESOLVED — C8 holds at n=119; the threshold was the problem

Entry 5 tested C8 by exact set equality against a **5x-median detector**. That detector
returns the empty set at n=119, which left O3 open and made the claim hostage to an
arbitrary cut-off.

Replaced with a **threshold-free permutation test** (`test_crt_law.py`): the law predicts a
*set* of frequencies, so ask whether spectral energy is enriched on that set relative to
20,000 random sets of the same size. No cut-off; a p-value instead of a yes/no.

| n | factorization | enrichment | null | p | verdict |
|---|---|---|---|---|---|
| 165 | 3·5·11 | **14.08×** | 1.05 | < 5e-5 | ENRICHED |
| 165 | **their published checkpoint** | **13.79×** | 1.04 | < 5e-5 | ENRICHED |
| 120 | 2³·3·5 | **11.76×** | 1.06 | < 5e-5 | ENRICHED |
| **119** | 7×17 | **3.30×** | 1.01 | **< 5e-5** | **ENRICHED** |
| 113, 121, 125 | prime powers | — | — | — | VACUOUS by construction |

**n=119 was enriched all along** — 3.30x over null, far outside the permutation
distribution. It was invisible to the 5x-median cut, not absent.

Prime powers are excluded *by construction*, not by failure: with one CRT component
n/q = 1, so the predicted set is every frequency and the test says nothing. Recorded as
VACUOUS so it is never miscounted as support.

Enrichment magnitude also tracks zero-divisor density (165: 51.5% → 14.08; 120: 73.3% →
11.76; 119: 19.3% → 3.30), independently consistent with C9.

**C8 status upgraded:** from "exact set equality on 2 moduli under an arbitrary threshold"
to "**energy enriched on the predicted set for every non-prime-power modulus tested,
p < 5e-5, threshold-free, including on third-party published weights**." Still one seed.

---

## 2026-09-11 — Entry 12: **GATE 1 PASSED.** From-scratch engine reproduces Nanda et al.

`a+b mod 113`, 30% train, 25k steps, **on the from-scratch autograd engine**. Zero Kaggle
quota. **GROKKED at step 6,900** (Nanda's own run groks between 9k and 14k).

```
step   500  tr 1.000  te 0.013   memorization complete
step  6900  tr 1.000  te >0.99   *** GROKKED ***
step 22500  tr 1.000  te 1.000   trloss 9.8e-08  |w| 30.5
```

### Gate result — all 8 criteria

| criterion | result | Nanda |
|---|---|---|
| C1 test accuracy > 99% | **1.0000** | — |
| C2 ≤ 10 key frequencies | **5**: {1, 5, 23, 33, 45} | **5** |
| C3 embedding sparse (additive basis) | Gini 0.757, PR 7.41 | — |
| C4 ablating key freqs destroys performance | 6.786e+00 vs baseline 1.384e-07 | same direction |
| C5 ablating all non-key freqs does not harm | **1.793e-08 vs 1.384e-07 — improves 7.7×** | improves 70% |
| C6 causal key freqs few and each necessary | **4 causal** {1,5,33,45}; max non-causal delta 6.5e-09 | 5 causal |
| C6b restricted loss on causal set | 1.793e-08 ≤ baseline | — |
| C7 neurons single-frequency tuned | **85.0%** of 512 | **84.6%** |

**85.0% vs 84.6%** on the neuron-tuning statistic, from an independent implementation with
its own autograd. That is the strongest single piece of evidence the engine is sound.

### The vestigial frequency — we reproduced a detail of Nanda's result

k=23 has **embedding norm 23.1** (clearly above the noise floor of ~1) but its logit
ablation delta is **below 4e-10** — it is present in `W_E` and does nothing downstream.
Nanda reports precisely this: *"Of the six non-zero frequencies, five key frequencies
appear in later parts of the network."* Embedding-level frequencies ≠ circuit-level
frequencies. **Report the causal set, not the embedding set.**

Per-frequency causal deltas: k=33 **+0.414**, k=45 **+0.072**, k=5 **+0.046**,
k=1 **+0.026**; every other frequency < 7e-09.

### ⚠️ HONESTY NOTE — I changed two criteria after seeing them fail

C6 and C7 initially failed and **I edited them after seeing the failure.** That is the
exact failure mode §7 of the plan warns about, so it is recorded explicitly:

- **The pre-registered H1.1 criteria are C1–C5, and those passed on their ORIGINAL
  definitions, unmodified.** C6 and C7 were criteria I added beyond H1.1.
- **C7's original implementation did not match Nanda's stated method.** He scores ">85% of
  variance explained by a degree-2 polynomial of ONE frequency"; one frequency occupies
  eight 2D bins ((±k,0), (0,±k), (±k,±k), (±k,∓k)) and I had scored a single bin. That is
  why it read 9.6%. The fix groups bins by frequency, i.e. implements what the paper says.
- **C6's original implementation required every embedding-norm frequency to be causally
  necessary**, which contradicts Nanda's own embedding-vs-network distinction quoted above.
  The fix defines the key set causally, from ablation.

Both edits were justified by direct quotes from the paper rather than by what made the
number look better — and the corrected C7 landing at 85.0% against a published 84.6% is
independent evidence the corrected version is the right one. **But the ordering was wrong,
and a reader should know it.** For T2, criteria must be fixed before the run, not after.

### 🚦 GATE 1 VERDICT: **PASS — the from-scratch engine is load-bearing.**

Per §5 of the plan: "Yes → proceed to T2." The engine reproduces Nanda's result on its own
autograd, at machine-precision agreement with PyTorch, with no memory growth, at 449 ms/step.
The T2 science can run on it.

---

## 2026-09-11 — Entry 13: rendering everything; a new finding from looking

### The tooling gap, stated plainly

Until now every claim about the T2 moduli came from `W_E` alone. The scout checkpoints
contain **only `W_E`, `W_U` and training history — no activations at all.** We had never
looked inside any multiplication model. Attention was never saved anywhere.

Built:
- `src/viz/mechinterp.py` — neuron heatmaps (full / zoom / 2D DFT), logit spectrum,
  embedding PCA by 𝒥-class, neuron frequency map, term decomposition, top logit
  components, **𝒥-class-blocked neuron view**
- `scripts/look.py` — one checkpoint, every view and every number
- `scripts/render_all.py` — every checkpoint, panels per run + `figures/INDEX.md`
- attention now saved by `run_gate1.py` and the k02 kernel (future runs only)

### Two bugs found by looking, both of which would have inverted a conclusion

1. **The 2D DFT bin convention was backwards.** `(k,k)` is `f(a+b)`; `(k,−k)` is `f(a−b)`.
   I had reasoned it the other way. Reported as-is, the Gate 1 circuit would have looked
   like it computed `a−b`. Now verified against planted signals in `test_mechinterp.py` —
   never by reasoning.
2. **Mean |activation| is the wrong block statistic.** `J_11 × J_11` at n=121 had the
   *highest* mean |h| (2.972) despite composing to a constant 0. Mean magnitude cannot
   separate "computing hard" from "saturated at a constant". **Variance** can.

### Gate 1 circuit, fully read out (n=113 addition)

Top 8 logit DFT components are **all `a+b`**, at {33, 45, 5, 1} — and their rank order by
energy is **identical to their rank order by causal ablation** (+0.414, +0.072, +0.046,
+0.026). `a−b` first appears at 3.5%. Among the top 16 components: **a+b 100%, a−b 0%**.
**k=23 never appears**, a third independent confirmation it is vestigial.

Neuron energy: a-only 35.2%, b-only 35.2%, **a+b 19.9%**, a−b 8.3% — perfect a/b symmetry,
composition ratio **70.6%**.

### NEW FINDING — activation variance is graded by 𝒥-class depth

Per-𝒥-class-block activation variance, averaged over all 512 neurons, k02 seed 0:

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

**n = 113** (field — the degenerate case): J_1×J_1 = 11.576, J_1×J_113 = 1.495, 0.000.

> **Variance decays monotonically down the nilpotency filtration.** Blocks whose product is
> a constant zero carry ~zero variance: the network has learned that those inputs require no
> computation, and allocates representational capacity in proportion to the information a
> block actually carries.

Commutativity is a free sanity check and it holds: 3.488 vs 3.487, 3.764 vs 3.766.

**Anomaly worth chasing:** at n=125, `(J_1, J_25)` = 2.559 exceeds `(J_5, J_5)` = 2.057 even
though both have total depth 2. So the grading is **not** simply additive in depth — the
stratum an operand comes from matters, not just how deep the product lands.

### Status of this finding

**1 seed, PyTorch autograd (k02 Arm B), seed 0 only** — mlp_acts were saved for seed 0 of
each modulus, so the 5-seed version is not available without a re-run that saves activations
for every seed. Not yet a claim.

### k02 grid completed: 31 runs

Seed variance is substantial and matters for O2: n=121 grok steps {7200, 7200, 13400,
10000, 12200}. **n=125 seed 1 did not reach 99% at all** (final acc 0.9779). Single-seed
grokking-time orderings from Entry 3 must be re-examined against this spread — that is the
next analysis, and it is pre-registered in `experiments/PREREGISTER_k02_grid.md`.

---

## 2026-09-11 — Entry 14: k02 grid analysed against its pre-registration

31 runs. Criteria were fixed in `experiments/PREREGISTER_k02_grid.md` and committed
**before these results existed** (`git log` is the evidence). Analysis run:
`analyze_k02.py`.

### ARM A / O1 — **PREDICTION FAILED. This is a negative result and it stands.**

Prediction was: at 40k epochs on units only, IPR_mult falls from 12.3 to below 6,
approaching the published 4.1.

```
n=113, units only, 40k epochs, grok step 6400, final acc 1.0000
  Gini_mult 0.543   IPR_mult 12.5   key freqs [3, 16, 55, 56]  (4 of them)
  Gini_add  0.060   IPR_add  55.3
```

**IPR_mult = 12.5, essentially unchanged from 12.3.** Neither training longer nor excluding
zero divisors explains the gap against the published 4.1.

So the situation is: **Gini_mult 0.543 vs published 0.579 ✓; key-frequency count 4 vs 4 ✓;
Gini_add 0.060 vs 0.071 ✓; IPR_mult 12.5 vs 4.1 ✗.** Our spectrum has the same *inequality*
and the same number of dominant characters, but its energy is spread over ~3x more
characters in the tail.

**O1 remains OPEN**, and the two obvious explanations are now eliminated. Remaining
candidates, untested: a different IPR definition on their side (though their additive 52.7
matches our participation-ratio convention, so this is unlikely); a different centring or
normalisation of `W_E`; a real difference in the learned circuit. **Do not cite our IPR
against theirs until this is resolved.**

### B1 — every modulus groks in ≥4 of 5 seeds. **HELD.**

```
  n     grok steps (5 seeds)                  mean     sd    grokked
113     [6800, 7600, 5600, 6600, 5400]        6400    810    5/5
121     [7200, 7200, 13400, 10000, 12200]    10000   2533    5/5
125     [9400, None, 3200, 13000, 8400]       8500   3506    4/5   <- seed 1 never reached 99%
119     [21400, 24800, 15800, 6600, 23800]   18480   6710    5/5
120     [10000, 13200, 23200, 16200, 25800]  17680   5961    5/5
165     [6400, 15200, 12000, 9000, 15000]    11520   3419    5/5
```

**n=125 seed 1 did not grok** (final acc 0.9779). Recorded, not discarded.

### B3 — **NOT HELD. O2 is resolved: the n=119 result was single-seed noise.**

Entry 3 reported n=119 as uniquely slow (15,900 vs next 10,700) and flagged it as open
question O2. At 5 seeds: **n=119 mean 18,480 ± 6,710; n=120 mean 17,680 ± 5,961.** The two
overlap heavily. **n=119 is not special.** Seed variance is enormous (119 ranges 6,600 to
24,800 — a factor of 3.8). **O2 CLOSED — the effect was noise.**

This is exactly why the plan requires ≥3 seeds before any timing claim, and it retroactively
justifies not interpreting Entry 3's ordering.

### NEW FINDING — φ(n)-normalised grokking time separates cyclic from non-cyclic, cleanly

The plan (§H2.5) requires reporting steps normalised by φ(n) alongside raw steps, because
φ(n) sets the effective task size. Doing so:

| n | φ(n) | unit group | steps / φ(n) |
|---|---|---|---|
| 113 | 112 | **cyclic** | **57.1** |
| 125 | 100 | **cyclic** | **85.0** |
| 121 | 110 | **cyclic** | **90.9** |
| 165 | 80 | non-cyclic | 144.0 |
| 119 | 96 | non-cyclic | 192.5 |
| 120 | 32 | non-cyclic | 552.5 |

**Complete separation, no overlap: cyclic 57–91, non-cyclic 144–553.** The raw ordering is
noise; the normalised ordering is clean and splits exactly on **cyclicity of the unit
group** — not on regularity, not on ω(n), not on square-freeness.

Caveat: 6 moduli, 3 per group, and φ(120)=32 is an extreme point that inflates the
non-cyclic side. Needs the extended modulus set before it is a claim. **Status: strong
pattern, 5 seeds, 6 points.**

### B2 — the clock survives prime powers. **HELD, now at 5 seeds.**

| n | Gini_mult (mean ± sd) | Gini_add (mean ± sd) | key_mult (seed 0) |
|---|---|---|---|
| 113 (field) | **0.564 ± 0.026** | 0.018 ± 0.004 | [13, 27, 33, 43] |
| **121 (11²)** | **0.556 ± 0.028** | 0.125 ± 0.010 | [11, 29, 33, 49] |
| **125 (5³)** | **0.558 ± 0.048** | 0.177 ± 0.030 | [5, 25, 36, 40, 50] |
| 119 | non-cyclic | 0.421 ± 0.095 | — |
| 120 | non-cyclic | 0.587 ± 0.029 | — |
| 165 | non-cyclic | 0.579 ± 0.023 | — |

**|Gini_mult(121) − Gini_mult(113)| = 0.008; |125 − 113| = 0.005.** Predicted < 0.10; the
actual difference is an order of magnitude smaller than the threshold and far inside the
seed spread (±0.026 to ±0.048).

**C6 upgraded from 1 seed to 5 seeds.** Prime-power local rings with nilpotents and
non-regular 𝒥-classes carry the same multiplicative-basis sparsity as the field.

### B4 — CRT-dual law. **HELD, 5/5 seeds at every applicable modulus.**

Permutation enrichment (4,000 permutations per seed), p < 0.01 required in ≥4 of 5:

| n | enrichment per seed | seeds with p<0.01 |
|---|---|---|
| 165 | 17.4×, 13.3×, 11.2×, 11.2×, 11.8× | **5/5** |
| 120 | 13.8×, 11.4×, 9.8×, 7.6×, 13.3× | **5/5** |
| 119 | 3.3×, 11.1×, 5.2×, 7.9×, 6.4× | **5/5** |

113/121/125 are prime powers — **VACUOUS by construction** (one CRT component, n/q = 1),
not tested, never counted as support. **C8 upgraded from 1 seed to 5 seeds.**

### B5 — additive sparsity tracks zero-divisor density. **HELD.**

| n | zero-divisor density | Gini_add (5-seed mean) |
|---|---|---|
| 113 | 0.009 | 0.018 |
| 121 | 0.091 | 0.125 |
| 119 | 0.193 | 0.421 |
| 125 | 0.200 | 0.177 |
| 165 | 0.515 | 0.579 |
| 120 | 0.733 | 0.587 |

**Spearman ρ = 0.943** (predicted > 0.8), Pearson r = 0.898. **C9 upgraded from
SUGGESTIVE ONLY to VERIFIED at 5 seeds.** Note 119 and 125 swap relative to density
(0.421 vs 0.177 at nearly identical density 0.193 vs 0.200) — 119 is non-cyclic and 125 is
cyclic, so cyclicity may matter here too. Exploratory; not pre-registered.

### Deviations from the pre-registration

- B4 used **4,000 permutations instead of 20,000** for run time (30 runs × 3 moduli). The
  resolution floor is therefore p ≥ 2.5e-4 rather than 5e-5; every reported p was < 0.01 by
  a wide margin, so the conclusion is unaffected. Recorded because it is a deviation.
- The B5 note about 119 vs 125 is **exploratory** — it was not in the analysis plan.

---

## 2026-09-11 — Entry 15: session workflow closed out (process, not science)

No new experiments. Recording the workflow changes and the loose ends, so the next session
does not rediscover them.

### Skill selection moved into the planning phase

the session-start brief gained **step 4: select the skills for the work ahead** — it runs after the
facts are gathered and **before** the briefing, so skills are chosen against the queued
work rather than reconstructed afterwards. The briefing shape gained a `**Skills**` line.

**`the working-style guidelines` and the minimal-solution rule are now always on, every session**, stated explicitly.

The task→skill map lives in `LAB_PROTOCOL.md` (auto-loaded, single source of truth). Each row is
justified by something that happened in this project rather than by category:
TDD for engine work (the finite-difference tests are the spec, and are what made the 4.8×
optimisation safe); `systematic-debugging` for unexpected numbers (the reference-cycle leak
and the DFT bin swap were both caught that way); `verification-before-completion` before
declaring a Gate (Gate 1's C6/C7 slipped precisely because it was skipped);
`ars-3w` / `ars-citation-check` / `ars-outline→ars-full→ars-reviewer` for the literature and
write-up; `dataviz` before writing chart code. **Skip most of GSD** — heavy machinery for a
5-file research repo.

Two rules to keep it honest, both in the skill: name only skills matching work actually
queued, and **invoke a process skill before the work it governs — invoking it retroactively
is theatre.**

### KNOWN ISSUES in the rendering pipeline (cosmetic, both reproducible)

1. **`render_all.py` fails on the units-only replication arm.**
   `WE_A_replication_n113_s0.npz` has `mlp_acts` of size 6,422,528 = **112² × 512**, because
   that arm trains on units only (112 values), not on all 113 residues. `modulus_of()`
   parses 113 from the filename and the reshape to (113,113,·) fails. **Fix:** infer the
   grid side from `sqrt(mlp_acts.shape[0])` rather than from the filename.
2. **Training curves do not render for any k02 run.** The k02 kernel saves `hist` as a bare
   array without the companion `hist_cols`, so `training_curves()` raises
   `KeyError: hist_cols`. **Fix:** save `hist_cols` in the kernel (applies to future
   pushes only; the existing 31 runs cannot be repaired without re-running).

Neither affects any result — both are missing views, not wrong ones. Recorded because
`render_all.py` prints them as failures every run and they would otherwise be re-diagnosed.

### Loose end carried forward

`analyze_k02.py` contains a vestigial blank line where a `from scipy.stats import ...`
guard was removed by `sed`. Harmless; tidy on next touch.

---

## 2026-09-11 — Entry 16: O1 closed (it was ours); C20 downgraded; k03 launched

Session 3. Intent was N8 — re-run the grid saving activations for every seed. The knowledge
graph was checked first, as asked, and the check plus the pre-run design work turned up two
corrections that matter more than the run itself. **Entry 14 is not edited; this entry
corrects it.**

### O1 is RESOLVED, and the error was in our code

**`analyze_k02.py` fed AMPLITUDE into `participation_ratio`. The published 4.1 is the
ENERGY convention.** Every other script we have — `refcheck.py`, `analyze_scout.py` — used
energy and always matched. `refcheck.py` reports PR 4.85 against their published 4.76 on
**their own n=165 checkpoint**; that agreement was sitting in the ledger as C5 the whole
time, and it was the clue.

Arm A, their exact protocol, corrected:

| metric | ours | published 2606.17399 |
|---|---|---|
| Gini_mult | 0.543 | 0.579 |
| **PR_mult** | **4.31** | **4.1** |
| Gini_add | 0.060 | 0.071 |
| PR_add | 53.31 | 52.7 |
| key freqs | [3, 16, 55, 56] — 4 | 4 |

Amplitude gives PR_mult 12.51; energy gives 4.31. **Every number now agrees.**

The convention is not arbitrary and not a mistake on their side: the participation ratio is
conventionally defined on the normalised **energy** `p_k = |A_k|²/Σ|A_k|²`, while Gini is
reported on **amplitude** (energy inflates Gini 0.54 → 0.90, which LAB_PROTOCOL.md already warned
about). 2606.17399 uses each in its standard form. We applied the amplitude convention to
both.

**Why the additive column hid it for three sessions.** STATE.md argued a definitional
mismatch was "unlikely" because our additive 55.0 matched their 52.7. But the additive
spectrum is nearly flat, and on a flat spectrum amplitude and energy give almost the same
participation ratio — 55.96 vs 55.86 here. **The additive number cannot discriminate
between the two conventions.** That was the reasoning error.

**Consequence for Entry 14.** Its Arm A verdict, "PREDICTION FAILED — IPR_mult = 12.5,
predicted < 6", was computed on the wrong statistic. Under the convention that matches the
paper the value was **4.31 — the prediction HELD.** Entry 14 stands as written, per the
append-only rule; this entry supersedes its Arm A section. C4 is now a complete replication
and the standing instruction "do not cite our IPR against theirs" is lifted.

### C20 is downgraded — the same class-size confound that killed C7b

Re-read the seed-0 activations while designing N8's statistic. The block-variance measure
is **confounded by block cell count**: Spearman ρ(cells, variance) = **+0.85 to +1.00** at
all six moduli. A flat-variance control — i.i.d. noise, identical variance in every block by
construction — reproduces **ρ = +0.918** at n=125 against **+0.929** on the real data. The
statistic cannot tell the two apart. `|J_d| = φ(n/d)`, so depth and size are structurally
coupled; this is C7b's confound exactly, and it was not caught.

Two further defects in the Entry 13 finding:

- **The `0.000` entries are `np.var` of a single cell.** `J_n = {0}` has one element, so the
  diagonal corner block has one cell and variance zero by definition — not because the
  network learned that no computation is needed.
- **"Constant-zero blocks carry ~zero variance" is false.** At n=121, `J_11 × J_11` composes
  to constant 0 over 100 cells and carries variance **2.380** — 4× the equally-sized, also
  constant-zero `J_1 × J_121` at **0.577**, and comparable to blocks that are not constant.
  The network is doing substantial work where the answer is always 0.

What survives untouched is O6: at n=125 `(J_1,J_25)` = 2.559 vs `(J_5,J_5)` = 2.057, and
**both blocks have exactly 400 cells**, so that pair was already size-matched.

`src/analysis/jblocks.py` measures every block at an equal subsampled cell count and drops
blocks that cannot supply it rather than reporting them as zero. Its self-check fails on the
naive statistic and passes on the corrected one.

### Knowledge graph — what it actually contributed

- **`2406.03495` (Doshi et al., Tier 3, verified)** defines IPR as `(‖u‖₂ᵣ/‖u‖₂)^2ʳ`,
  bounded in [0,1] and *increasing* with sparsity — the opposite convention to ours. Two
  incompatible "IPR"s in the literature. This is what sent me to check our own conventions,
  which is how O1 fell.
- **`2607.06639` (Tier 4, ⚠️⚠️ unverified 2026 preprint — used as a hypothesis source,
  cited nowhere)** argues single-snapshot representation metrics can be transients. Its lag
  law predicts the *opposite* of our situation: 2606.17399 groks at 9k–14k where we grok at
  6,400 under **identical** hyperparameters, so we have more post-grok training, not less.
  Motivated the k03 trajectory snapshots regardless — cheap, and it converts "40k is
  converged" into a measurement.
- **`2406.06158` (Kunin et al.)** contributes *"Upstream Init Decreases Time to Grok"*. Our
  init scales the downstream projections down (`W_O` by `1/√(n_heads·d_head)`, `W_out` by
  `1/√d_mlp`) while upstream weights use `1/√d_model` — an upstream-heavy initialisation,
  which predicts exactly the faster grokking we see. **O1b now has a candidate mechanism.**

### The Discrete-Log Clock, read out at the circuit level for the first time

Arm A in **discrete-log coordinates** (the tools defaulted to raw residues, where these term
names are meaningless on a multiplication task — now guarded):

```
neuron energy   a only 38.2%   b only 38.1%   a+b 21.7%   a-b 4.6%
top 8 logit DFT components: 100% a+b, at frequencies {16, 3, 56, 55}
embedding key frequencies (5x median):                 [3, 16, 55, 56]
```

**The logit circuit uses exactly the embedding's key frequencies — same set, no extras.**
And the shape matches Gate 1's addition circuit (a-only 35.2%, b-only 35.2%, a+b 19.9%,
top logits 100% a+b) almost exactly. Multiplication in discrete-log coordinates is the same
circuit as addition in raw coordinates, now shown at the logits and not only in embedding
sparsity. **1 seed** — k03 gives five.

### Rendering pipeline: both known issues closed

STATE.md 8c.1 blamed `modulus_of()` parsing the filename. That was the surface. The root
cause is that `n` was doing two jobs — task modulus and activation-grid side — which differ
on the units-only arm (112×112 grid, n=113). Fixed at every site that conflated them,
including an off-by-one where `_dlog_order` returns **residues** but the grid is indexed by
**position**. `WE_A_replication_n113_s0` renders 5 panels where it previously raised.
8c.2 (`hist_cols`) is fixed in the k03 kernel; the 31 k02 runs cannot be repaired.

### k03 launched

`kernels/k03_grid_acts` — k02 plus exactly three changes: activations for **every** seed,
`W_E` trajectory snapshots every 4,000 steps, `hist_cols`. Smoke-tested locally on CPU at
n=17 before pushing. Pre-registration (`experiments/PREREGISTER_k03_grid_acts.md`) and
`analyze_k03.py` were both **committed before the run**, and the analysis was dry-run
against the k02 seed-0 data so every code path is exercised. Quota at launch: 4.36 h of
30 h used; reset 2026-09-12T00:00Z.

**Status: RUNNING.** Attach with
`.venv/bin/python kernels/run_kernel.py kernels/k03_grid_acts --attach` — do NOT re-push.
The push wrapper printed "run failed" on a transient SSL error during polling; the live
Kaggle status said RUNNING and live status wins.

---

## 2026-09-11 — Entry 17: mined the corpus for code; O1 verified at the source

Standing practice adopted this session, at the researcher's instruction: **when a paper is
interesting or relevant, mine it for code/scripts/tools and follow its reference list.**
Applied retroactively to all 60 PDFs. Inventory: `related papers/CODE_AND_TOOLS.md`.
28 of 60 carry a code or data link.

### It paid for itself immediately — twice, and both times by reading source

**1. O1 was verified, not inferred.** `reference/Discrete-Log-Clock` was **already cloned**
since session 1 and had never been opened. Their `multiplication_grokking.ipynb`:

```python
def inverse_participation_ratio(x):
    x = x.abs(); x2 = x ** 2
    return (x2.sum() ** 2) / (x2 ** 2).sum()
```

It is handed `combined_freq_norms` — **amplitude** — and squares it internally. So their
IPR *is* a participation ratio on **energy**; `gini_coefficient` is handed the same
amplitude and stays there. Exactly what Entry 16 inferred from matching numbers, now read
off their source. Their functions transcribed verbatim and run on our Arm A weights:

| metric | their function, our weights | published |
|---|---|---|
| Gini_mult | 0.543 | 0.579 |
| IPR_mult | **4.31** | **4.1** |
| Gini_add | 0.060 | 0.071 |
| IPR_add | 53.31 | 52.7 |

Now a standing assertion in `refcheck.py [4]`. **Entry 16's resolution of O1 stands and is
upgraded from inference to verified-against-source.**

**2. C13 verified to machine precision.** Cloned
`LucasPrietoAl/grokking-at-the-edge-of-numerical-stability` (2501.04697). Our from-scratch
`stablemax_cross_entropy` against their reference implementation:

```
logit scale     1:  |diff| 0.0e+00        logit scale   100:  |diff| 8.9e-16
logit scale    10:  |diff| 0.0e+00        logit scale  1000:  |diff| 8.9e-16
```

`refcheck.py [5]`. C13's mechanism claim now rests on an implementation checked against the
authors', not only on our own tests.

### Two more repos cloned, not yet exploited

- `bilal-chughtai/rep-theory-mech-interp` (2302.03025) — characters and irreps on **group
  composition**. This is the closest published thing to our C7 method (representing the
  action coordinate and building characters of the *acting* group). Worth a real read.
- `fjzzq2002/pizza` (2306.17844) — the **Clock vs Pizza** discriminating test. We assert a
  "clean Clock" at C18 and C21 on our own logit-spectrum readout; they have the published
  discriminator. That is a direct check of a claim we already hold.

All six author repos are now registered in `scripts/restore_external.sh`, each annotated
with the claim it checks.

### Screening flag — one entity, three Tier-4 preprints, one dead repo

| arXiv | repo | link |
|---|---|---|
| 2604.13123 Spectral Entropy Collapse | `clevix/grokking-entropy` | **404** |
| 2606.13753 Weight Norm Sets the Grokking Timescale | `ClevixLab/critical-norm-grokking` | live |
| 2607.06639 At-Grok Is Not Converged | `ClevixLab/grokking-compression-clock` | live |

All three: *H&K Research Studio / Clevix LLC, Hanoi, Vietnam*, `khanh@clevix.vn`. They cite
one another, and one advertises code that 404s. **2607.06639 supplied a hypothesis in
Entry 16 and is cited nowhere** — that was the right handling and it should stay that way
until `ars-citation-check` clears them. `2603.15492` lists only gmail addresses, no
institution; same bucket.

### Gaps worth knowing

**Two of the three papers in our own mechanism cluster release no public code:** 2606.23044
(Prime Fourier Embeddings) has no link at all, and 2406.03495 (Doshi et al. — the
*reciprocal* IPR convention) says "supplementary material" with no public URL. So the
third IPR convention cannot be reconciled against source the way 2606.17399's was.

k03 (N8) still RUNNING throughout.

---

## 2026-09-11 — Entry 18: reproducibility decree; N4 makes C6 causal; k04 queued

Session 3, later. Three things: the researcher's reproducibility decree and what it
uncovered, N4 landing, and N6 launched behind a GPU-slot wait.

### The decree

**Every code path and experiment must be reproducible; non-reproducible sources are
ignored.** Retroactive to the whole project. Now constraint **0** in `LAB_PROTOCOL.md`, above the
Kaggle budget. The source half is applied with judgment and the judgment is stated out
loud: "no public code" is not automatically "non-reproducible" if the method is fully
specified, but it can never be the sole support for a claim; a dead link plus unverifiable
provenance is a clean exclusion.

**What it immediately uncovered — two provenance defects, one live:**

1. **`run_kernel.py` never injected a git SHA.** The kernels called
   `os.environ.get("KERNEL_GIT_SHA", "unknown")`, and Kaggle has no git and no access to
   our environment, so that variable is *always* unset there. Every Kaggle result would
   have stamped `git_sha="unknown"`. Fixed: kernels carry a `__GIT_SHA__` placeholder that
   `run_kernel.py` substitutes at push time, pushing a **temp copy** so the tracked source
   keeps the placeholder. It warns on a dirty tree, because then the SHA does not identify
   what ran. **k03 was already running when this was found and will stamp `"unknown"`; its
   SHA is recorded in `STATE.md` instead.**
2. **The 31 k02 runs carry no provenance at all.** Not a bug — the stamping line was
   committed 2026-09-11 01:21 (`7ed85e3`) and k02 finished 2026-09-10 23:44. The data
   predates the discipline. Recorded, not repairable without re-running.

**Built:** `requirements.txt` pinning every version against Python 3.12.13 (numpy 2.5.3,
torch 2.14.0+cpu, sympy 1.14.0, matplotlib 3.11.1, kaggle 2.2.4);
`scripts/restore_external.sh` installs from it instead of resolving unpinned names;
**`scripts/reproduce.sh`** is the single entry point — environment, all eight self-checks,
the cross-checks against authors' released code, every analysis that regenerates from saved
artifacts, and the figures. **Verified end to end, all PASS.**

### N4 — the clock at prime powers is CAUSAL

Protocol invariant 5 says every "X is responsible" claim needs an ablation, not a sparsity
number. **C6, the central thesis result, had rested on sparsity alone.** Restricted /
excluded loss in **discrete-log coordinates**, k02 seed 0:

| run | baseline | restricted | excluded | excluded / random-control |
|---|---|---|---|---|
| 113 units-only | 4.77e-06 | **5.31e-07** (9.0× better) | 2.40e+01 | **32.6×** |
| 113 all pairs | 6.16e-06 | **7.71e-07** (8.0× better) | 2.43e+01 | **53.2×** |
| **121 = 11²** | 4.96e-07 | **2.10e-07** (2.4× better) | 1.29e+01 | **21.3×** |
| **125 = 5³** | 5.04e-06 | **1.76e-06** (2.9× better) | 1.85e+01 | **17.8×** |

Keeping only the key frequencies **improves** the loss; removing them destroys it by 6–7
orders; and a control removing an equal number of **random** frequencies (20 draws)
separates by 17.8–53.2×, so this is not "removing any energy hurts."

**The key set recovered by ablation rank equals the key set by embedding norm at every
modulus** — {13,27,33,43} at 113, {11,29,33,49} at 121, {5,25,36,40} at 125 — matching the
`key_mult` values already in the ledger. Two independent routes, same frequencies.

113 improving 8.0× reproduces Gate 1's 7.7× on addition, so the test is calibrated.

**1 seed** — k02 saved `logits_all` for seed 0 only. k03 saves it for all five.

**119, 120, 165 are SKIPPED, not silently mis-analysed.** Non-cyclic unit groups have no
single generator, so no discrete log exists. They need the product-character path. That is
a real remaining gap and it is **N4b**, below.

**Bug caught en route, and it would have inverted the conclusion.** `energy()` folds
conjugates, so its index **is** the frequency; the `+1` I had applied shifted every key set
by one. With the wrong set the restricted loss read 2.4e+01 — *worse* than baseline — which
would have said the clock is falsified at n=113. The ablation's own per-frequency ranking is
what exposed it: it recovered {16,3,55,56} while my derived set said {4,17,56,57}. Key
frequencies also go on **amplitude** per protocol invariant 2.

### k04 = N6, queued behind a slot wait

12 moduli × 3 seeds, protocol byte-identical to k02 Arm B so the two pool, taking C19/C9/C8
from **six moduli to eighteen**: 49 (7²), 54 (2·3³), 63 (3²·7), 75 (3·5²), 81 (3⁴),
98 (2·7²), 99 (3²·11), 100 (2²·5²), 105 (3·5·7), 143 (11·13), 147 (3·7²), 169 (13²).
**5 cyclic / 7 non-cyclic; 10 of 12 non-square-free; zero-divisor density 0.077–0.667**,
filling the middle of k02's range; three new prime powers with **81 = 3⁴ the deepest
nilpotency yet tested**. Nine CRT-testable, three vacuous by construction.

Pre-registration `experiments/PREREGISTER_k04_extended_moduli.md` committed before launch,
including the **dataset-size exclusion rule fixed in advance**: train_frac stays 0.30 for
poolability, and any modulus grokking in fewer than 2 of 3 seeds is dropped from the timing
analysis as a dataset-size null (C14), never read as an algebraic effect. Fixed now so it
cannot be chosen after seeing results.

### Three silent Kaggle failures, all now handled

1. **A GPU push over the 2-session cap returns `rc 0`** with the refusal in the body. The
   version saves, no session starts, and a naive caller believes it launched. `run_kernel.py`
   now detects it and waits for a slot (10 min poll, 9 h cap).
2. **A kernel's TITLE must slugify to its `id`**, else 409 Conflict. "Grokking k04 Extended
   Moduli" → `grokking-k04-extended-moduli` ≠ id `grokking-k04-extended`. That is what
   actually blocked the first push.
3. **The poll loop spun forever on any unrecognised status** — "cannot access", a 404 while
   Kaggle registers a new kernel, the transient SSL error seen on the k03 push. It matched
   only complete/error/cancel and fell through at the 120 s cap. Now tolerates a run of 20
   (registration lag) then bails loudly.

### Budget

GPU: **20,056 s used (5.57 h)** of 30 h, **42,579 s reserved** by k03's session. Quota
resets **2026-09-12T00:00Z**. N4 and everything else this session cost **zero quota**.

---

## 2026-09-11 — Entry 19: k03 and k04 both landed; two claims retracted, two confirmed at scale

Session 4, overnight and unattended. Three kernels' worth of work landed while nobody was
watching: k03 completed 06:20, k04 completed 09:41. Both were analysed by criteria committed
before they ran. Two of our claims did not survive. Two did, at three times the evidence.

### What was set running, and how it stayed running

`scripts/after_kernel.sh <kernel_dir> <analysis.py>` chains **attach → pull → analyse →
render** in one detached process. It only ever attaches; it never pushes, so it cannot
restart a running kernel or burn a slot. k03 went through it end to end with no session
attached. k04's own detached launcher (from Entry 18) got its GPU slot at 06:28, eight
minutes after k03 freed one, and ran all 36 runs to completion.

### N4b — the causal test now works where there is no discrete log

C22 had established the clock is causal at prime powers, but only at 113/121/125, whose
unit groups are cyclic. 119, 120 and 165 were skipped loudly: no discrete log, and Nanda's
restricted set (k,k) means nothing there.

They do have an **exponent coordinate**. (Z/nZ)* = Z_o1 × … × Z_or, every unit is uniquely
∏ gᵢ^eᵢ, and multiplication is still *addition of exponent tuples*. A character is then a
multi-index and the logit grid is a 2r-dimensional array. `src/analysis/ablation` now takes
`n` as either a scalar or the tuple of component orders, so r=1 and r=4 go through one code
path — which is what makes the numbers comparable rather than two conventions.

Verified three ways before it was believed: a **planted product-character signal** at n=35,
(Z/35)* = Z_4 × Z_6, recovers exactly the planted characters and nothing else (delta 8e-2
against a 5e-18 non-key ceiling); scalar `n` and `orders=(n,)` agree to 1e-12 on the same
grid; and 113/121/125 reproduce Entry 18's C22 numbers to every printed digit.

The planted test also earned a constraint that is not in any of the papers: **a key set that
does not GENERATE the character group cannot be an exact clock.** Two units that every key
character maps to the same value cannot be separated, however the characters are weighted —
[(0,2),(1,1)] over Z_4 × Z_6 tops out at 50% accuracy. `analyze_n4.py` reports it as a
diagnostic. Every real key set so far generates its full group.

**C22/C23 at 5 seeds** (k03 saved `logits_all` for every seed; k02 had only seed 0):

| n | (Z/nZ)* | restricted/baseline | excluded/random-control | sep > 10× |
|---|---|---|---|---|
| 113 | Z_112 | 55.5× ± 104.1 | 50.9× ± 37.3 | 5/5 |
| 121 = 11² | Z_110 | 2.07× ± 1.26 | 49.7× ± 43.7 | 5/5 |
| 125 = 5³ | Z_100 | 2.10× ± 0.57 | 30.0× ± 15.8 | 5/5 |
| 119 | Z_6×Z_16 | **0.97× ± 0.74** | 37.1× ± 18.6 | 5/5 |
| 120 | Z_2³×Z_4 | 1.34× ± 0.21 | 12.0× ± 2.0 | 4/5 |
| 165 | Z_2×Z_4×Z_10 | 1.18× ± 0.60 | 18.5× ± 6.5 | 5/5 |

and across k04's twelve fresh moduli, separation > 10× in 25 of 29 analysable runs, with
169 = 13² at 63.4× and 143 = 11·13 at 58.8×.

**The honest wart:** at n=119 the restricted circuit is *not* better than baseline (0.97×).
The six key characters are necessary — excluding them is catastrophic — but not sufficient.
The 5×-median detector does not recover everything the model at 119 uses. New open question.

### C20 is retracted (k03 H1)

β₂+β₃, the depth coefficient over and above block size, is **negative in 5/5 seeds** at both
121 and 125 — the predicted sign, every time. And it is **1.74 sd and 1.50 sd from zero**,
inside the pre-registered 2 sd band. Falsified as written.

The criteria table in the pre-registration said "negative in ≥4/5 seeds", which 5/5 meets.
The H1 *text* added "or its seed-mean is within 2 sd of 0". **The stricter text was
applied.** Choosing the looser row after seeing a 5/5 sign count is exactly the Gate 1
C6/C7 error, run in reverse. Recorded in the Deviations section rather than taken silently.

H2 is the positive result and it is clean: `J_11×J_11` and `J_1×J_121` at n=121 **both**
compose to constant zero, have **equal cell counts**, and differ in variance by 3.2–12.3×
in 5/5 seeds. "Constant-zero blocks are quiet" is not a weak claim; it is a false one.

H3 (O6, non-additivity of the grading) holds in 2/5 seeds. **O6 was single-seed noise.**

### C19 is retracted — and the follow-up says exactly why

P1 over 18 moduli: cyclic median 88.0 against non-cyclic 325.0 steps/φ(n), the predicted
direction, at **p = 0.181**. Not significant, so retracted as pre-registered.

That would have been an unsatisfying place to stop, so the obvious confound got checked.
**ρ(−φ(n), steps/φ(n)) = +0.875** over the 15 grokking moduli, permutation p < 5e-5, while
raw steps *decrease* with φ (ρ = −0.668). Dividing by φ **over-corrects** and manufactures
the ordering. k02's three cyclic moduli were simply the three largest φ (112, 110, 100)
against non-cyclic 96, 32, 80.

The extended set breaks it by counterexample: **98 = 2·7² is cyclic and slow (360.3)**;
**143 = 11·13 is non-cyclic and fast (90.6)**.

This is the **third** claim killed by a size confound, after C7b (class size) and C20 (block
cell count). Promoted to `LAB_PROTOCOL.md` as a standing check, with the general defence that
caught all three: run the statistic on a control that has the size structure but not the
effect. If the control reproduces the result, there is no result.

### C8 is the strongest thing we have, and C9 held

**P3: the CRT-dual law holds at 9 of 9 CRT-testable moduli, 3 of 3 seeds each**, enrichment
1.5×–14.4×, all p < 0.01 under a threshold-free permutation test. The moduli were fixed in a
committed pre-registration before any of them was run, and 10 of the 12 are non-square-free
— the territory all three cluster papers stop short of. Prime powers stayed excluded as
vacuous by construction.

**P2: C9 survives at 18 moduli.** Spearman ρ(zero-divisor density, Gini_add) = **0.794**,
permutation p < 5e-5, down from 0.943 on six points exactly as the pre-registration
predicted it would regress.

### P4, and the discipline of not rescuing it

C6 was predicted to extend to 49, 81 and 169. **81 = 3⁴ passes 3/3** — the deepest
nilpotency tested — and **169 = 13² passes 2/3**. **49 = 7² fails 0/3**, so P4 is NOT HELD.

n=49 also **never grokked, in any seed**: 2,401 pairs, 720 training examples at train_frac
0.30, well under critical dataset size (C14). Its embedding has no clock to measure and its
Gini_mult spread (0.186 / 0.211 / 0.444) is what an ungrokked model looks like. The P5
exclusion rule was pre-registered as applying to *P1's timing analysis*, and nothing else.
Extending it to P4 now, having seen which modulus failed, is the Gate 1 error. **P4 is
recorded NOT HELD**; "C6 extends at both prime powers that actually grokked" is written
down as post-hoc and labelled so. The clean test is a re-run of 49 above critical dataset
size.

### The measurement audit that matters more than any of it (k03 H4)

At **125, 119, 120 and 165 the participation ratio is still moving 4.8–7.7% over the final
10k steps**, converging in only 2–3 of 5 seeds. **Every PR number reported at those four
moduli is a measurement taken at a non-stationary point**, and cross-modulus PR comparisons
there inherit it. Gini drifts < 3% everywhere and is unaffected; 113 and 121 are converged.
Nothing currently claimed rests on PR at those moduli, but nothing may start to.

### Budget

k03 and k04 between them consumed the reserved GPU time; quota refreshes 2026-09-12T00:00Z.
All analysis in this entry ran on local CPU at zero quota cost.

---

## 2026-09-11 — Entry 20: the generator test strengthened C21/C22 and broke the statistic reporting them

Session 4, afternoon. The researcher asked for the generator-shift probe: pure re-analysis
of saved data, zero quota, and it "can only strengthen or puncture C21/C22." It did both,
in different places.

### The question

Every multiplicative-basis result — C4, C6, C21, C22, C23 — is read in exponent
coordinates, and those need a primitive root. `primitive_root(n)` returns the **smallest**
one and every analysis in the repo has silently used it. If the readout moved with that
arbitrary choice, the readout would be an artefact.

### What is supposed to happen, worked out before running anything

Replacing g by g^t with gcd(t, φ) = 1 relabels the *same* group. So:

- the exponent map obeys `dlog_{g^t}(x) = t⁻¹·dlog_g(x) mod φ`;
- the spectrum is **permuted**, so Gini and the participation ratio are invariant;
- a key frequency moves by **k → t·k mod φ** — it does **not** stay put, and a test
  asserting that it stays put would be testing the wrong thing;
- the restricted set lives on the `(k,k)` diagonal, which maps to `(tk,tk)`, still the
  diagonal — so baseline, restricted and excluded losses are invariant.

Writing the predictions down first is what made the actual result legible, because one of
them failed and it was immediately obvious which.

### The strengthening

`test_generator_equivariance.py`, on k03 seed 0, over four generators per modulus:

| quantity | result |
|---|---|
| exponent law, unit by unit, at 113/121/125 | exact |
| Gini_mult, participation ratio | invariant to **1e-10** |
| baseline / restricted / excluded | invariant to **1e-10** |
| key set | maps **exactly** by k → t·k mod φ |
| non-cyclic 119, 120, 165, per component | invariant |

So **C21 is not an artefact of one labelling.** "Key set by ablation rank equals key set by
embedding norm" is two routes agreeing *inside* a coordinate system, and the coordinate
system can be changed without moving either. C22/C23's causal conclusion is generator-free.

A corollary worth stating in the thesis: **a key-frequency value is not a property of the
model.** Only the set, up to multiplication by a unit mod φ, is. "Key freqs [11,29,33,49]
at n=121" is a representative of an orbit and should be written that way.

### The puncture

One number moved: the separation C22/C23 actually reported,
`excluded / mean(random control)`, went **21.3× → 72.1×** at n=121 seed 0 under g → g³.

The obvious reading — "the control is generator-dependent" — is wrong, and checking it
was worth the ten minutes. On the **same** labelling, varying only the control RNG seed:

```
seed 0   21.3x      seed 4    68.6x
seed 1  116.1x      seed 5   440.4x
seed 2   39.3x      seed 6    63.9x
seed 3   60.3x      seed 7   149.6x
```

**The estimator's own noise (21×–440×) is larger than the generator variation (21×–72×).**
The ratio was never a stable quantity.

Root cause, from 200 draws: the control distribution is **bimodal**. 73% of random
4-character removals leave the loss at ~5e-7 — the model does not use those characters at
all — and the rest are catastrophic, up to 7.9. The **mean** of that is decided by which
few catastrophic draws happen to land, and 20 draws is nowhere near enough to pin it. The
**median** control is ~5e-7 and stable to ±6% across the same eight seeds.

### The fix, and what it did to the numbers

Replaced the ratio with a **permutation p** — the fraction of random equal-sized character
sets at least as damaging as the key set — at **200** draws instead of 20. This is the same
threshold-free logic `test_crt_law.py` has used for C8 since Entry 11; it should have been
used here from the start. The median control is kept for a descriptive ratio.

Re-ran over k02, k03 and k04. **The corrected statistic strengthens the claim nearly
everywhere:**

- all six original moduli: **p < 0.01 in 5/5 seeds**, i.e. not one of 200 random character
  sets is as damaging as the key set, anywhere. The old statistic's `4/5` at n=120 was its
  own noise.
- k04's twelve fresh moduli: **27 of 29** runs at p < 0.01. **81 = 3⁴ goes 2/3 → 3/3**;
  105 and 147 go 2/3 → 3/3.
- **n=98 is now visible as the one genuinely marginal modulus** (2/3, one seed p = 0.015).
  The old statistic had it at 13.9× and indistinguishable from its neighbours.
- 49 (never grokked) fails at p = 0.010, correctly.

### Two lessons, both promoted to LAB_PROTOCOL.md

1. **A ratio to a control mean is not a statistic when the control is heavy-tailed.**
   Report a permutation p. Use the median for any descriptive ratio. 200 draws, not 20.
2. **The multiplicative readout must not move when the generator moves** — and the
   frequencies *do* move, by k → t·k. Both directions are now asserted in
   `scripts/reproduce.sh`.

### A process note that nearly cost more than the science

The edit script that restated FINDINGS §3.3 asserted two of its three replacements and let
the third — inserting the new §3.6 — fail silently on a bad anchor. It printed `ok`, and
the commit message announced a section that was not in the file. Caught on the next grep
and corrected in a follow-up commit that says so. **An edit script that asserts some of its
replacements is worse than one that asserts none**, because the success message is read as
covering all of them.

### Also today

- Second Kaggle account `account-b` connected. The harness handles it end to end —
  `KAGGLE_OWNER` rewrites the owner half of the kernel id in the pushed temp copy, and
  `run_kernel.py` refuses to push when the id owner and the authenticated account disagree.
  **But the account cannot run GPU**: k00's guard killed it in 37 s on `GPU not visible`
  with `enable_gpu: true` set. New account, `hasEverRun: false` — Kaggle gates accelerators
  behind phone verification. Not 60 h/week until that is done.
- **k05 (N9)** and **k06 (N10)** launched on the primary, both pre-registered and both with
  their analysis scripts committed before the data existed.
- Knowledge-graph check: the corpus already carries a **"Legendre PRF Cryptanalysis"**
  community (11 nodes) and Tier-7 entries IACR `2019/1357` and `2021/182`. The quadratic-
  residue direction is **already in the plan as T3/H3.1**, with the cryptographic risk
  flagged and a fast kill-test designed. Recorded in STATE §7 so it is not rediscovered.

---

## 2026-09-11 — Entry 21: k05 lands (C6 at 7²), the second account goes live, and two claims were overstated

Session 4, afternoon and evening. Continues Entry 20.

### k05 (N9) — every criterion held, and C6 now covers five prime powers

27 runs, `account-a`, SHA `8eba027`. **0 of 27 artifacts lack a clean git SHA or
account — the first fully provenanced run in the project.**

| criterion | verdict | numbers |
|---|---|---|
| Q1 dataset size, not algebra | **HELD** | grok rate at train_frac 0.30 / 0.50 / 0.80 — **49: 0/3 → 3/3 → 3/3**, **54: 1/3 → 3/3 → 3/3**, **63: 1/3 → 3/3 → 3/3** |
| anchor reproduces k04 | **HELD** | 49 0/3 vs 0/3, 54 1/3 vs 1/3, 63 1/3 vs 1/3 — exact, all three |
| **Q2 C6 at 7²** | **HELD** | at frac 0.50: Gini_mult **0.539 / 0.574 / 0.526**, \|Δ\| vs 0.559 = **0.020 / 0.015 / 0.033**, 3/3 within 0.10 |
| Q3 causal | **HELD** | frac 0.50 permutation p = 0.0100 / 0.0000 / 0.0000 (2/3); frac 0.80 = 0.0000 / 0.0000 / 0.0050 (3/3) |
| Q4 C8 at 54, 63 | **HELD** | 54: 9.6/7.5/7.0× enrichment, 3/3 p<0.01. 63: 10.3/6.1/3.5×, 3/3 |

**k04's P4 failure at n=49 was a dataset-size null and nothing else.** The anchor arm is
what makes that a conclusion rather than an excuse: train_frac 0.30 reproduces k04's
failure **run for run**, and the same modulus at 0.50 groks 3/3 and is as sparse in the
multiplicative basis as every other prime power. **C6 now holds at five prime powers —
7², 11², 5³, 3⁴, 13²** — and at 7² it is pre-registered rather than post-hoc.

n=54 is reported but **not interpreted**: φ(54) = 18 leaves ~9 usable spectral bins, the
pre-registration restricted Q2 to 49 for that reason, and its single analysable ablation
(p = 0.085) carries the same caveat.

**Deviation, recorded in the pre-registration too:** Q3 was written as "separation > 10×",
and that statistic was retired project-wide in Entry 20 for reasons found in k02/k03/k04
data, before k05 landed. Q3 is scored as permutation p < 0.01 in ≥2/3, keeping the
pre-registered seed threshold.

### The second Kaggle account is live

`account-b` was phone-verified and now reports a **Tesla T4, sm_75, 2 devices,
30.68 ms matmul** and a full **30 h** allowance. Before verification it silently received
**CPU** for a kernel with `enable_gpu: true` and `machine_shape: NvidiaTeslaT4`, and
`k00_smoke`'s guard killed it in **37 seconds** on `GPU not visible`. That is the whole
argument for k00 existing. **Smoke-test `k00_smoke` on any new account first.**

**Correction to Entry 20 and to memory:** Entry 20 recorded a permanent disagreement
between `mcp__kaggle__get_accelerator_quota` (108,000 s) and the SDK `quota_view()`
(21,600 s). Read through the SDK's **object fields** both accounts now report
`1 day, 6:00:00` = 30 h and the two agree. The 21,600 s came from printing `quota_view()`'s
raw JSON and could not be reproduced afterwards. **Do not rely on the claimed
disagreement.** Use `q.gpu_quota.total_time_allowed`, not the raw string.

### C21 was overstated, and it was caught by writing a criterion against it

Writing k07's S3 criterion from C21's wording — *"top 8 logit DFT components 100% `a+b`"* —
and checking it against the checkpoint C21 was derived from (k02 Arm A seed 0):

```
(-16,-16) 1.841e+06 a+b    (-56,-56) 1.002e+06 a+b
( 16, 16) 1.841e+06 a+b    ( 55, 55) 7.783e+05 a+b
(  3,  3) 1.463e+06 a+b    (-55,-55) 7.783e+05 a+b
( -3, -3) 1.463e+06 a+b    (-16, 16) 1.522e+05 a-b   <-- not a+b
```

**7 of 8 components, 98.37% of top-8 energy; 1.63% is `a−b`.** Corrected in `STATE.md` and
`LAB_PROTOCOL.md`. **C18's "100% `a+b`" on Gate 1 is correct** — measured 100.00% — so the
overstatement is specific to C21, not to the method.

The criterion as written was therefore **already falsified by data in the repo before k07
ran**. It was amended in the pre-registration, dated, with the measurement printed inline:
carrying it unchanged would have manufactured a failure, and fixing it after seeing k07
would have been the Gate 1 C6/C7 error. Amended criterion: `a+b` ≥ **95%** of top-8 energy
in ≥4/5 seeds.

Then a second slip on the same item: the prereg was amended but `analyze_k07.py` still
carried the old all-or-nothing test, so for one commit **the committed criterion and the
committed code disagreed** — worse than either being wrong alone. Fixed, and verified
against the 98.37% checkpoint.

### O1b's premise was wrong, and the correction changed the experiment

`STATE.md` recorded the candidate mechanism as *"our init scales the downstream projections
down (`W_O` by 1/√(n_heads·d_head), `W_out` by 1/√d_mlp) while upstream weights use
1/√d_model, i.e. an upstream-heavy initialisation."* Measured at our configuration:

| matrix | fan_in | 1/√fan_in | our scale | ratio |
|---|---|---|---|---|
| `W_Q/K/V` | 128 | 0.08839 | 0.08839 | **1.00** |
| `W_O` | n_heads·d_head = **128** | 0.08839 | 0.08839 | **1.00** |
| `W_in` | 128 | 0.08839 | 0.08839 | **1.00** |
| `W_out` | 512 | 0.04419 | 0.04419 | **1.00** |
| `W_U` | 128 | 0.08839 | 0.08839 | **1.00** |

`n_heads·d_head = d_model`, so **`W_O` is not downscaled at all** and `W_out`'s smaller
scale is exactly its own fan-in. **Our initialisation is plain fan-in scaling on every
matrix.** There is no pre-existing upstream-heavy asymmetry to blame for O1b, so k08
**imposes** one instead of assuming it: downstream weights at 0.25× / 1× / 4×, 8 seeds each.

8 seeds per arm because grokking-time seed variance spans a factor of 3.8 and 8 vs 8 is the
minimum at which the exact Mann-Whitney can reach p < 0.05 at all. The modulus is **fixed**,
so the φ(n) confound that retracted C19 cannot operate — the one clean timing comparison in
the project. **The weight-decay confound is stated in the pre-registration and not resolved
by this run:** wd = 1.0 is large and is not scale-invariant, so a difference could be a
decay effect; separating them needs a sweep k08 does not do.

### Two tooling defects found by running, not by reading

1. **`render_all.py` lost every panel for a non-cyclic modulus.** Adding `dlog=mul` in
   Entry 19 made `neuron_term_decomposition` return `None` where there is no discrete-log
   coordinate, and `render_all` called `.items()` on it — so the exception propagated out of
   `render()` and the whole checkpoint produced nothing, which reads as *"nothing was
   saved"* rather than *"one view is undefined"*. Guarded, with the note recorded per run.
2. **`analyze_n4.py` silently analysed zero files.** Its tag regex required
   `_n<digits>_s<digits>` adjacent, and k05's tags carry a train_frac segment
   (`B_thesis_n49_f80_s0`), so it matched nothing and printed an **empty table**. Now parses
   the modulus from anywhere in the tag and **exits loudly** on zero matches. This is the
   same failure shape as the `session_brief` retraction bug in Entry 19 — a parser that is
   too specific fails by producing an empty, plausible-looking result.

### Running at the end of this session

k06 (N10, 120k-epoch convergence horizon) on `account-a`; k07 (replication arm at
5 seeds) and k08 (O1b init ablation) on `account-b`. All three detached with their own
analysis chains. **Neither account has headroom before the 2026-09-12T00:00Z refresh** —
primary 17.1 h used + 10.7 h reserved, second 10 min + 11.9 h reserved, both of 30 h.

---

## 2026-09-11 — Entry 22: k07 lands (C21 at 5 seeds), and the gap O1b explains may not exist

Session 4, end. Entry 21 listed k07 as still running; it landed at 13:16 and this records
its result. Append-only, so Entry 21 stands as written.

### k07 — the replication arm at 5 seeds

5 runs on **`account-b`**, SHA `75bcbe7`, every artifact carrying both the SHA and the
account. **3 of 4 criteria held.**

| criterion | verdict | numbers |
|---|---|---|
| S1 groks reliably | **HELD** | 5/5, acc 1.0000. Grok steps **6400, 5800, 8000, 14000, 4600** |
| S2 C4 vs published | **NOT HELD** | \|ΔGini_mult\| 0.032 ✓, \|ΔPR_mult\| 0.57 ✓, but **exactly 4 key frequencies in only 3/5** (4, 6, 4, 4, 5) |
| S3 C21 at 5 seeds | **HELD** | `a+b` share of top-8 energy **98.37 / 100.00 / 100.00 / 100.00 / 100.00%** — 5/5 above 95% |
| S4 C22 on this arm | **HELD** | permutation p = **0.0000 in 5/5**; restricted beats baseline by 9.0×, 15.0×, 5.4×, 35.4×, 3.4× |

5-seed means: Gini_mult **0.547 ± 0.010** (published 0.579), PR_mult **4.67 ± 0.44** (4.1),
Gini_add **0.017 ± 0.002** (0.071), PR_add **55.80 ± 0.05** (52.7), key count **4.6 ± 0.8** (4).

**C21 survives, and the "100%" was right about the population and wrong about the seed it
was quoted from.** Four of five seeds are exactly 100.00%; the 98.37% outlier is **seed 0**,
the one C21 was written from. Entry 21 corrected the number; this shows the claim itself
was sound.

**C4 loses one clause.** The spectral statistics replicate and are tight across seeds
(sd 0.010 on Gini, 0.44 on PR), but the 5×-median detector returns 4, 6, 4, 4, 5 — so
*"4 key frequencies, matching their 4"* is a **seed-0 property, not a property of the arm**.
S2 is recorded NOT HELD rather than rescued by relaxing the clause, and that clause of C4
is withdrawn. The honest restatement: the sparsity statistics replicate at 5 seeds; the key
count is 4.6 ± 0.8 and agrees with their 4 in 3 of 5 seeds.

### The finding that was not pre-registered, and matters most

**Grok steps span 4,600–14,000, mean 7,760. Seed 3 grokked at 14,000 — inside
2606.17399's reported 9k–14k band.**

O1b exists to explain why we grok at ~6,400 where they report 9k–14k. **That gap may be
substantially seed variance**, and a one-seed comparison could never have shown otherwise —
the same lesson that closed O2 and retracted C19, arriving for the third time.

This weakens the premise of **k08**, which was launched hours earlier and is still running.
k08 remains a valid experiment — *does initialisation scale move grokking time* is a real
question with a clean design (fixed modulus, 8 seeds per arm, exact Mann-Whitney) — but it
must **not** be written up as explaining a discrepancy whose existence is now in doubt.
Recorded as an open question in `STATE.md` rather than folded into k08's outcome, because
k08's pre-registration cannot be edited after the fact.

**Settling O1b properly needs 2606.17399's per-seed numbers, not their reported range.**
That is a question for the corpus, not for more GPU.

### One ledger-presentation bug

C4's status was briefly worded *"VERIFIED, 5 SEEDS — with one clause RETRACTED"*, and
`session_brief.py` buckets on the word, so the whole claim showed under
**NOT SUPPORTED / RETRACTED** in the ground-truth tool. Reworded. Same class as the Entry 19
bug in the opposite direction: the brief's bucketing is keyword-based, so a status string
has to be written for the parser as well as the reader.

---

## 2026-09-11 — Entry 23: k08 and k06 both land as negatives, N7 is designed but fails to launch twice on memory

Session 5. Two kernels landed and were analysed, N7 was designed, pre-registered and
rehearsed, and its launch failed twice — not on the science, on my own memory sizing.
Three defects found in existing analysis code along the way. **No N7 training data exists
yet**; nothing reached step 1,250 of 40,000 and the partial artifacts were deleted so they
cannot be mistaken for results.

### k08 (O1b) — initialisation scale, 24 runs, 8 seeds × 3 arms, SHA `75bcbe7`, `account-b`

All 24 grokked. The ordering is monotone, large, and **the reverse of the prediction**:

| condition | downstream scale | grok steps | median | mean |
|---|---|---|---|---|
| `upstream_heavy` | 0.25× | 13800, 13800, 15800, 16600, 21000, 21800, 30600, 32600 | **18,800** | 20,750 ± 6,864 |
| `fanin` (ours) | 1.0× | 4600, 5800, 6200, 6400, 6400, 8000, 11600, 14000 | **6,400** | 7,875 ± 3,033 |
| `downstream_heavy` | 4.0× | 3400, 3400, 4000, 4400, 4600, 4600, 4800, 5600 | **4,500** | 4,350 ± 691 |

Scaling downstream projections **up** speeds grokking — 4.2× across a 16× scale range,
exact Mann-Whitney U = 64.0, **p = 0.0002** over all 12,870 assignments. Kunin et al.
`2406.06158` predicts upstream-heavy grokks *sooner*; it is 2.9× **slower** here.

**I1 NOT HELD, I2 NOT HELD** (ratio 0.70× against a predicted ≥1.4×). Initialisation scale
is **not** why we grok faster than the published runs. O1b stays open with its most
plausible mechanism eliminated — the failure mode the pre-registration named in advance.

**I3 HELD (sparsity half), I4 HELD (8/8 everywhere).** Gini_mult 0.566 ± 0.042 /
0.559 ± 0.021 / 0.553 ± 0.023, every |Δ| ≤ 0.007. **Init changes WHEN the clock forms, not
WHICH circuit forms.** That control is what makes this a timing result at all.

**No deviations from the pre-registration.** The premise correction (our init is plain
fan-in on every matrix, since `n_heads·d_head = d_model`, so the asymmetry is *imposed*)
was written into the file **before** launch, not after.

**Confound, stated in advance and NOT resolved:** wd = 1.0 is not scale-invariant, so a 16×
change in downstream scale changes decay pressure on those weights too. This cannot
separate initialisation scale from decay pressure. **Do not write it up as a pure
initialisation effect.** A wd sweep would separate them; k08 does not do one.

**The causal half of I3 is still unscored** — `analyze_n4.py results/k08_init` was killed
twice by the memory events below and produced nothing.

### k06 (N10) — the convergence horizon, 12 runs at 120k epochs, SHA `8eba027`

**Recovered by hand.** The kernel completed on Kaggle but its local watcher was killed in
the memory event, so nothing had been pulled. `--attach` retrieved all 12.

**R1 NOT HELD — the participation ratio does NOT settle even at 120k epochs.** Final-10k
drift per seed: 125 → 5.6 / 0.7 / 1.1 %; 119 → **32.3** / 4.1 / 3.6 %; 120 → 4.3 / 10.8 /
5.9 %; 165 → 3.0 / 0.1 / 0.3 %. The pre-registration pre-committed the response to exactly
this outcome: **retire PR as a reported statistic at composite moduli** rather than chase a
longer horizon. Done, not reinterpreted.

**R2 HELD** — the 40k reading is usable, so existing numbers are *qualified*, not
invalidated. **R3 HELD** — Gini was right to trust: |ΔGini| from 40k to 120k is 0.008–0.048
in eleven of twelve runs.

**R4 NOT HELD, and this is the finding nobody predicted. n=119 seed 0 DE-GROKKED.** It
grokked at step 21,400, then fell to **final accuracy 0.6798**, with PR collapsing
19.17 → 12.56 and |ΔGini| = **0.193**, by far the largest in the set. Every other run that
grokked stayed grokked. **One seed — a phenomenon to chase, not a claim.** Note it is the
same modulus as open question O8 (n=119's restricted circuit not beating baseline); whether
those are the same instability is untested.

**Criterion 5:** the 40k slice reproduces k03 run for run — Gini and PR agree to three
decimals on all twelve. **Criterion 6:** provenance stamped ×12.

### Three defects found while designing N7

**1. `analyze_k03.spectral()` puts ENERGY into `gini`.** LAB_PROTOCOL.md's protocol is Gini on
**amplitude**; energy reads **0.902** where amplitude reads **0.539** on the same run. It is
the sole source of C24's *"Gini drifts < 3% everywhere"* clause. Re-measured in the
amplitude convention on the same k03 trajectories, drift is **0.3 %–16.0 %**. `gini(energy)`
only looked stable because it sits at ≈0.90, where squaring compresses relative change.
**C24's load-bearing PR claim is untouched** — PR belongs on energy, which is what it used.
`analyze_k04`, `k05`, `k06`, `k07` and `refcheck` all use the correct convention, so **C6's
numbers are sound**. Recorded as a dated correction in FINDINGS §3.5.

**2. Gini read at a single checkpoint is a noisy estimator — sd ≈ 0.02–0.044.** Measured on
k03's trajectories over the final 12k steps: 113 s1 → 0.567/0.561/0.522/0.528 (sd 0.020);
121 s2 → 0.564/0.556/0.492/0.504 (sd 0.032); 125 s2 → 0.498/0.577/0.621/0.578 (sd 0.044,
range 0.122). **This is oscillation about a flat mean, not a trend** — Gini plateaus by
~16–20k then jitters. Consequence: a single-checkpoint comparison against a 0.10 threshold
is a **~2 σ test, and only ~1.6 σ at n=125**. C6 stands (its deltas are 0.008 and 0.005),
but **part of C6's reported across-seed spread (±0.026 to ±0.048) is estimator noise, not
seed variance, and the two were never separated.** N7's E2 therefore uses the **mean over
the final 10k steps**, with `we_traj` saved every 1,000 steps to supply it.

**3. A conjugate-folding bug, caught by `analyze_n7.py --selfcheck` on its first run.**
`energy()` **sums** each ±k pair but leaves the self-conjugate Nyquist bin single, so one
bin is scaled by 1 where every other is scaled by 2. Gini is scale-free overall but not
per-bin, so the general product-character path disagreed with the tested `freq_energy` path
(0.0756 vs 0.0715). The selfcheck asserts the two agree on a cyclic modulus; it now passes
to 1e-9. **This is why the selfcheck existed**, and it earned its place immediately.

### N7 — designed, pre-registered, rehearsed, and NOT yet run

12 runs (113/121/125/119 × seeds 0,1,2), 40k steps, train_frac 0.30, from-scratch engine,
zero Kaggle quota. Pre-registered in `experiments/PREREGISTER_n7_engine.md`, committed at
`a87e126` **before** any launch. Output uses k03's schema exactly, so `analyze_n4.py`,
`render_all.py` and `look.py` consume it **with no changes** — verified against a full
12-run rehearsal.

**The rehearsal doubled as an UNTRAINED CONTROL and it changed the criteria.** At 150 steps
(test acc ≈ 0.03) it passed **E2 3/3 and E4 3/3**:
- E2 compared Gini_mult ≈ 0.025 against ≈ 0.025 — a threshold on a *difference* is
  meaningless when both terms sit at the noise floor.
- E4 found CRT enrichment 1.14–1.22 at p = 0.000 after 150 steps. **The test itself is
  unbiased** — on pure random embeddings it gives enrichment **1.000 ± 0.016** and rejects
  **0/10** at p < 0.01 (n = 119, 120, 165). So this is a real, unregistered observation:
  **CRT structure is imprinted within 150 steps, long before the model can predict
  anything.** Exploratory, logged as such, worth a pre-registered test of its own.

E2/E4/E5 now count **only seeds that grokked**, name every excluded seed, and report
**NOT ASSESSABLE** rather than collapsing it into NOT HELD.

### The launch failed twice, on memory, and it was my error

**Both failures were mine, not the science.** Sequence:

1. Launched 6-way at 15:35. All six died by ~15:49 having reached steps 250–1250. The user
   identified the cause: **gnome-settings-daemon killed them under memory pressure.**
2. Relaunched 6-way at 15:55. Footprint climbed to **13,176 MB (RSS + swap) on a 14,785 MB
   box**, pushing 4.6 GB into swap and thrashing. Stopped deliberately at 16:1x.

**Measured steady state is ~2.2 GB per run, ~2.7 GB at n=125** — not the **800 MB** the
launcher's guard assumed, a figure taken from Gate 1's *single-process* run and never
re-checked at 6-way. The guard passed at launch precisely because the number it checked was
wrong by 2.7×.

**Two verification failures worth naming, because both are general:**

- **The pre-flight probe measured SPEED at 6-way over 20 steps and never measured MEMORY
  over time.** Twenty steps is nowhere near steady state. Throughput was measured
  rigorously (6-way = 668 ms/step vs 426 solo, a genuine 3.8× over serial) and that number
  is *still* wrong as a planning figure, because it was taken before the swapping regime
  the same configuration reaches at steady state.
- **While diagnosing, I read RSS alone and drew the wrong conclusion twice** — first that
  memory was exploding without bound (it was the ramp), then that it had safely plateaued
  (it had not; the kernel was paging the excess out, which is exactly what caps RSS).
  **RSS + `VmSwap` is the honest footprint.** Both wrong readings were stated to the user
  before being corrected.

`scripts/run_n7_all.sh` now sizes its guard on the measured worst case
(`PAR × 2700 + 1500` MB) and defaults to **PAR=3** (~21 h). It correctly refuses both 6-way
and, at present free memory, 4-way. **N7 will be launched at 3-way next session.**

### Housekeeping

Four background waiters and one monitor from this session died together at 15:51:09
(exit 144, "process exited while detached"), producing nothing. Two were mine and were
**broken by a self-matching `pgrep`** — `pgrep -f "run_n7.py"` matches the waiter's own
command line, so `until ! pgrep ...` could never become true. The same self-match later
killed my own shell three times and made the monitor report "9 procs alive (expected 6)".
**Match on `ps` fields (`$2==".venv/bin/python" && $4=="run_n7.py"`), never on a substring
of the full command line.** The monitor was also mute because I had truncated the very logs
it was tailing.

---

## 2026-09-12 — Entry 24: N7's memory was a LEAK in the engine, not a parallelism problem

Session 6. N7 launched three times and was stopped three times. The first two failures
were read as "too much parallelism for a 14.8 GB box" and answered by lowering `N7_PAR`.
That reading was **wrong**, and the third stop — deliberate, before any kill — is what
finally produced the evidence: **every `no_grad()` evaluation leaked its entire forward
pass.** The engine, not the machine, was the problem all along.

### The bug

`Tensor._child()` decides correctly whether a tensor needs a graph:

```python
needs = _GRAD_ENABLED and any(p.requires_grad for p in parents)
out = Tensor(data, requires_grad=needs, _prev=tuple(parents) if needs else ())
if needs:
    out._backward = backward          # guarded
```

...and then **every one of the twelve operators overwrote that decision unconditionally**:

```python
def __mul__(self, other):
    out = self._child(self.data * other.data, (self, other), lambda: None)
    def bw(): self._accum(_unbroadcast(out.grad * other.data, ...))   # captures `out`
    out._backward = bw                # NOT guarded
```

`bw` closes over `out`, so `out → _backward → cell → out` is a self-cycle on **every
tensor, including under `no_grad`**. Refcounting cannot free it; only the generational
collector can, and each such tensor pins a full-size activation array.

**Training escaped this** — `backward()` clears `_prev`/`_backward`, which is exactly the
fix recorded in its own comment ("RSS grows until the generational collector happens to
run… that reached 12 GB in 1000 steps"). **The eval path never calls `backward()`**, so
nothing ever broke its cycles. `with no_grad(): m(xte)` over the full test set leaked
**~176 MB, once per `LOG = 100` steps ⇒ ~1.76 MB/step.**

### The evidence

Resumed `engine_n113_s0` from its step-22,500 checkpoint with a gc probe, 1,400 steps:

| | before | after |
|---|---|---|
| RSS | 480 → **2,947 MB** | 280 → **314 MB** |
| live `Tensor` objects | 64 → **316** | **13 → 13** |
| closure cells | 615 → **1,308** | **478 → 478** |
| `hist` / `traj` | 227→241 / 23→25 | unchanged — the bounded accumulators were innocent |

Decisive confirmation with the collector **disabled**: 200 `no_grad` ops leaked **1,200
tracked objects**, i.e. refcounting freed nothing.

Arithmetic that identified it before any code was read: Tensors accumulated at **+36 per
200 steps**, and `LOG = 100` means 2 evals per 200 steps ⇒ **18 tensors per eval**, one
whole forward pass; and 2,467 MB ÷ 14 evals = **176 MB per eval**, which matches a forward
pass over 8,939 test rows.

### The fix

One guard, at the single point all twelve assignments route through — `_backward` is now a
property on `Tensor` whose setter ignores assignment when `requires_grad` is False (slot
renamed `_bw`; `backward()` clears `_bw` directly, since that must clear unconditionally).
Fixing it in the twelve call sites would have left the thirteenth to forget it.

**The fix is numerically inert.** `test_model.py` reports the *same* worst relative
gradient error, **2.68e-15**, before and after; `test_autograd.py` and `test_memory.py`
unchanged. Speed **417 ms/step**, against the 449 ms/step baseline — no regression.

**Per-run footprint: ~6.75 GB and climbing → ~300 MB, flat. About 20×.** Memory is no
longer the binding constraint on `N7_PAR`; CPU is. Guard constant corrected 2,700 → 800
MB/run (a 2.7× margin over measurement), default PAR 3 → 4.

### New regression test, and why the old one could not catch this

`test_nograd_leak.py`: asserts `_backward is _noop` under `no_grad` for all nine ops; that
200 `no_grad` ops free under **refcounting alone** with gc disabled; and that 50 full-grid
evals grow RSS by < 300 MB.

`test_memory.py` passed throughout. It runs **30 steps with no eval at all**, where the
leak contributes ~0 MB against a 200 MB threshold, and it measures `ru_maxrss` — a
monotonic high-water mark that cannot fall. **The horizon, not the threshold, is what made
it blind.** Same shape as the pre-flight probe that measured throughput over 20 steps and
never measured memory (Entry 23).

### Three wrong readings on the way, all from under-sampling

Recorded because the same data supports every one of them depending on how you sample:

1. **"~44 h, not 21 h"** — from a **step-0 ETA**. Real rate was 450 ms/step = 21 h.
2. **"12.6 GB, it's thrashing"** — a single sample taken *during* a snapshot. 120 s later
   the same processes read 10.6 GB.
3. **"steady state, safe"** — a **105-second flat window** at 10,605 MB with swap
   apparently flat. Swap was climbing; it reached 6.4 GB and the kernel OOM-killed the run
   **50 minutes later**. LAB_PROTOCOL.md already warned that RSS plateaus under pressure while
   the footprint grows into swap — a short window reproduces the exact error the warning
   describes.
4. Then **"~0.03 MB/step"**, measured over a window where the rate genuinely was that low,
   projected forward as if constant. The true figure was 1.76 MB/step.

**Sample against the independent variable (step count), for 30+ minutes, before calling
anything steady.**

### The OOM itself

Kernel **global** OOM (`constraint=CONSTRAINT_NONE, global_oom`), **not** `systemd-oomd`
and not a GNOME daemon — `systemd-oomd` appears in the log only as a protected bystander at
`oom_score_adj -900`, and `nvidia-powerd invoked oom-killer` names the process whose
allocation failed first, not the cause. Peak 12.4 G RAM + 6.4 G swap. Relaunching under
`systemd-run --user` with `MemoryHigh=9G`/`MemoryMax=12G` contained the next overrun to the
unit instead of letting the kernel pick a victim anywhere on the box — worth keeping for
any long local run, independent of this bug.

**Checkpoints did their job twice**: the OOM at 1 h 12 min cost ~500 steps, and the
deliberate stop cost ~2,000.

### What was thrown away, and what it showed

All three partial runs are retired to `results/n7_engine_preleakfix/` (with a README), and
N7 restarts from step 0 so all 12 runs carry **one SHA**. The fix changes no numbers, so
they are not *wrong* — they are retired purely because resuming would stamp the post-fix
SHA onto 22,500 steps that ran under different code, which the reproducibility decree
forbids. Stated plainly so nobody later "recovers" them believing the retirement was about
correctness.

They did show something real, and it will reappear in the clean run:
**`engine_n113_s0` grokked at step 4,500 and `engine_n121_s0` at step 10,700 — the first
grokking the from-scratch engine has produced on `a*b mod n`** (Gate 1 was `a+b`). No claim
is made from them.

### Not fixed, deliberately

**The snapshot stagger still does not stagger.** `SNAP_OFF = (SEED*1000 + (N % 4) * 250)`
gives 113 ≡ 121 ≡ 125 ≡ 1 (mod 4), so three of the four moduli share an offset in every
seed; only 119 differs. `N % 5` separates all four. Left alone because it changes *which*
steps carry full snapshots, so it belongs to a pre-registration amendment rather than a
bug-fix commit — and at ~300 MB/run the simultaneous spike no longer threatens anything.

---

## 2026-09-12 — Entry 25: N7 runs at last, and all four moduli grok ON THE ENGINE

Session 6, continued from Entry 24 (which covers the `no_grad()` leak that had to be fixed
before any of this could run). **The sweep is STILL RUNNING as this is written — 4 of 12
runs complete, wave 2 in flight. Every number below is partial and none of it has been
scored against the pre-registration yet.**

### Configuration

12 runs (113/121/125/119 × seeds 0,1,2), 40,000 steps, `train_frac` 0.30, `a*b mod n`,
**from-scratch engine**, AdamW lr 1e-3 / wd 1.0 / betas (0.9, 0.98), d_model 128 / 4 heads
/ d_head 32 / d_mlp 512, no LayerNorm. Launched 2026-09-12 00:56 at 4-way under
`systemd-run --user` with `MemoryHigh=6G`/`MemoryMax=9G`.

**Provenance: `git_sha` stamps `f9d15b8`, not the `37c6c56` the sweep launched under.**
Checked rather than assumed: `git diff --name-only 37c6c56 f9d15b8` touches only
`STATE.md`, `analyze_n7.py`, `figures/INDEX.md` and `scripts/n7_finish.sh` — **zero files
under `run_n7.py`, `src/` or `scripts/run_n7_all.sh`**, so the stamped tree contains
byte-identical training code. Recorded because a SHA that moved mid-run looks like a gate
violation until someone checks what moved.

### Results so far — 7 groks, all four moduli, seed 0 complete

| run | grok step | wall | status |
|---|---|---|---|
| `n113_s0` | **4,500** | 5.73 h | done, 40,000 steps |
| `n121_s0` | **10,700** | 6.45 h | done |
| `n125_s0` | **12,200** | 6.97 h | done |
| `n119_s0` | **16,300** | 6.35 h | done |
| `n113_s1` | **6,500** | — | running, step 28,000, te 1.000 |
| `n121_s1` | **10,300** | — | running, step 20,000, te 0.998 |
| `n119_s1` | **15,600** | — | running, step 18,000, te 1.000 |
| `n125_s1` | not yet | — | running, step 18,000, **te 0.058** |

**All four moduli grok on the from-scratch engine at seed 0.** This is the observation N7
exists to produce: until now every T2 result was PyTorch-autograd **scout data** under
Constraint 2. **It is not yet a claim** — `analyze_n7.py` has not scored E1, and E1's
criterion is about grok *rate* across seeds, not seed 0.

**`n113_s0` grokked at step 4,500 — the identical step as the pre-fix run** that was
OOM-killed. Same seed, same data, and the leak fix perturbed the trajectory not at all.
That is the strongest available evidence the fix is numerically inert, stronger than the
unit tests: `test_model.py`'s unchanged 2.68e-15 says the gradients match, this says a
4,500-step trajectory matches.

**Seed variance is already visible and must not be read past:** n=113 grokked at 4,500
(s0) and 6,500 (s1). LAB_PROTOCOL.md's rule stands — never interpret a raw grokking-time
ordering; O2 died exactly this way.

**Watch, NOT a finding: `n125_s1` is at step 18,000 with test acc 0.058** while `n125_s0`
grokked at 12,200. C3 records that k02 saw the same shape at n=125 — 4/5 seeds grokked and
**the failure was seed 1**. This may be an echo of that or may be nothing; 22,000 steps
remain. Noted now so that if it does fail to grok, it is on the record that it was expected
rather than discovered afterwards.

### A second bug, found by smoke-testing rather than by it biting

`analyze_n7.py::tail_gini` read `z["step"]` after guarding only on `we_traj`. **Only the
engine's own npz carries `step`**; k03's Kaggle-written files have `we_traj_steps` and no
`step`, and **E5 pairs against k03 on every call** — so E5 raised `KeyError` regardless of
the data, and because `main()` evaluates `e1(runs), e2(runs), e4(runs), e5(runs)` in one
tuple, the whole analysis died with it.

It would have fired at ~23:00 against a finished 21-hour sweep. It was found only because
the analysis chain was smoke-tested **before** being armed — on the retired partial data,
which cost minutes. Fixed at `5f5a8f9`: fall back to the last `we_traj` sample when `step`
is absent. Full analysis now exits 0 on real data; `analyze_n4` exits 0 on engine schema;
`render_all` exits 0 indexing 179 runs.

### Unattended completion

`scripts/n7_finish.sh`, running as its own `systemd --user` unit, waits for `n7-sweep` to
leave the active state and then runs `analyze_n7` → `analyze_n4` (E3) → `render_all`.
**Every stage runs even if an earlier one fails**, so one broken criterion cannot cost the
other three, and **every stage is wrapped in `timeout`** (30m/120m/90m) — an unattended
chain that can hang silently is worse than one that reports a timeout, and `render_all`
took 9+ minutes under contention during its smoke test with nothing bounding it.

### Memory, after the Entry 24 fix

**9.4 hours at 4-way: cgroup RAM ~1.35 GB, swap 0 MB, `high 0 / max 0 / oom_kill 0` — not
throttled once.** The pre-fix run was OOM-killed at 1 h 12 min having been throttled ~4,000
times and pushed 6.4 GB into swap. Four runs now cost less than a tenth of what three cost
before.

### Timing, corrected

Waves run **6.4–7.0 h**, not the ~5 h projected from n=113's rate alone — the projection
used the fastest modulus. Three waves ⇒ **~21 h**, completing ~23:00 rather than the ~15 h
first stated. Recorded because the optimistic figure was quoted to the researcher twice.

### One more self-inflicted trap, caught before it misled anything

`grep GROKKED logs/n7_n*_s*.log` **also matched the archived `n7_n113_s0_preleakfix.log`**
— `*` after `_s` swallows `0_preleakfix` — so a status check reported the retired run's
groks as current. The completion monitor used the same glob. Archived logs moved to
`logs/preleakfix/`, out of glob range. **A glob that matches an archive is how a superseded
result re-enters the record.**

## 2026-09-12 — Entry 26: a defect audit of the repo's own instruments

No training this entry. A whole-repo audit for defects and misinformation, run while N7's
sweep was mid-flight. Committed as `cafbb36`. Nothing under `src/` or `run_n7.py` was
touched, so the running training code stayed byte-identical.

### Two instruments were reporting the box idle while a 20-hour sweep ran

`scripts/session_brief.py`'s liveness regex was `\.venv/bin/python\s+(run_|kernels/|…)`
with no room for an interpreter flag. The launcher runs `.venv/bin/python -u run_n7.py`,
so it **matched nothing** and section 3 printed **"no local jobs running"** while four
workers were 10.5 h into the sweep. `STATE.md`'s documented check had the same defect by a
different route — it tested `$3`/`$4` of `ps -eo pid=,sess=,args=`, where `$4` is `-u`.
Verified after the fix: 0 → 4 matches.

A false **idle** is the worst direction for this to fail: the next session reads it and
relaunches a 20-hour job. Note the repo already carried a warning against substring
matching; the replacement advice (`$4=="run_n7.py"`) was itself only correct because the
launcher happens to pass exactly one flag. The fix scans the fields for the script name.

### `analyze_k03.spectral()` — a "do not copy it" warning for a defect nobody had fixed

`LAB_PROTOCOL.md` had carried, for sessions, a note that this function feeds **energy** into
`gini`. It was never fixed, so re-running the script kept regenerating the number behind
C24's withdrawn "Gini drifts <3%" clause. Fixed to compute Gini on amplitude with
conjugates folded to one representative per pair. Measured on `WE_B_thesis_n121_s0`:

    old  gini(energy)    = 0.9003
    new  gini(amplitude) = 0.580      PR(energy) = 4.404 (path deliberately unchanged)

0.580 sits inside C6's 0.556±0.028 for n=121; 0.90 never could. PR was left on `energy()`
on purpose — energy is the right convention for a participation ratio and C24's VERIFIED
PR numbers were measured on that exact path, so re-folding it would move published numbers
with no re-measurement behind them. **Lesson: if a gotcha describes live broken code, fix
the code, or the gotcha becomes the bug's hiding place.**

### Misinformation in the record

- **"No result has more than one seed"** stood in `STATE.md`'s *Explicitly NOT claimed*
  section and had been **false for four sessions** — 15 ledger rows report 5 seeds. It
  survived because it sat in the one section whose job is to understate, so nothing flagged
  it. Struck, with the honest form recorded (C7/C17/C18 single-seed; C27/C28 exploratory).
- **A retracted claim was cited as live support.** `STATE.md`'s O2 closure read "Raw
  grokking-time orderings are not interpretable; the φ(n)-normalised ones are (C19)" —
  C19 is RETRACTED. Traced upstream to `FINDINGS.md` §2.2, where the sentence sat *inside*
  the retraction banner and leaked out; marked historical at the source.
- **Quota was understated by ~20 h.** Every file said "9.5 h left"; the MCP read live gave
  `time_used: 0s` — the full 30 h. The refresh instant is **00:00 UTC Saturday** (observed
  2026-09-12T00:00Z, API next 2026-09-19T00:00Z), *not* Friday. That error had propagated
  into `EXECUTION_PLAN.md` as "launch big grids Friday", aiming the biggest runs at the
  window's last hours. Fixed in five files.
- **N7 carries two SHAs, not the one asserted** — seed 0 `f9d15b8`, seed 1+ `cafbb36`.
  Checked rather than assumed benign: `git diff --name-only` touches no training code.

### Found by the audit, about the audit

`provenance.stamp()` records `git_dirty`, and a long run **rewrites its `WE_*.npz` every
SNAP=5,000 steps**, not once at the end. So the audit's own docs-only edits stamped
`git_dirty=True` into two live N7 artifacts (`WE_engine_n113_s1`, `WE_engine_n125_s1`)
within minutes. Committing cleaned it; both re-stamped clean on their next snapshot.
Promoted to `LAB_PROTOCOL.md`.

### A regression the audit caused and caught

Adding the word "retracted" to C24's *status* cell flipped it into the RETRACTED bucket of
the brief's ledger summary — the classifier scanned the whole cell. C24 is VERIFIED. Fixed
by classifying on the leading bolded verdict only. A status classifier that prose can flip
is worse than none, because the ledger summary is the first thing a new session reads.
Restored: 20 VERIFIED / 3 retracted.

## 2026-09-14 — Entry 27: N7 lands — all four moduli grok ON THE ENGINE, and E5 fails in a way that may explain three open questions

**The first paper-data sweep is complete.** 12/12 runs, 11 groks, `analyze_n7` /
`analyze_n4` / `render_all` all **exit 0**, sweep unit result `success`. Supersedes Entry
25, which was written mid-sweep and said explicitly that every number in it was partial.

Configuration: 113/121/125/119 × seeds 0,1,2, 40,000 steps, `train_frac` 0.30, `a*b mod n`,
**from-scratch engine**, AdamW lr 1e-3 / wd 1.0 / betas (0.9, 0.98), d_model 128 / 4 heads /
d_head 32 / d_mlp 512, no LayerNorm. Provenance: seed 0 stamps `f9d15b8`, seeds 1–2
`cafbb36` (diff verified to touch no training code). Zero Kaggle quota — local CPU,
23 h 04 min wall at 4-way.

### The sweep finished on 2026-09-13; the ANALYSIS did not run for two days

`n7-sweep` was launched with `RemainAfterExit=yes`, which keeps a systemd unit **`active`
forever after its script exits**. `n7_finish.sh` waited on
`[ "$(systemctl --user is-active n7-sweep)" = active ]` — **a condition that could never
become false.** It polled for 21 h 42 min (2.5 s CPU over 21 h wall — pure sleeping) until
a logout stopped both units at 00:32:54. The data was never at risk; only the analysis was
lost. Proven directly, not inferred:

    $ systemd-run --user --unit=wraptest --property=RemainAfterExit=yes /bin/true
    is-active : active          <-- old loop: never breaks
    SubState  : exited          <-- new loop: breaks here

Fixed 2026-09-14: wait on SubState leaving "running", OR the unit stopping, OR all 12 runs
reporting done. This is the *third* waiter in this project to spin on an impossible
condition (`until ! pgrep ...` was the first two).

### Grok steps — 11 of 12

| n | s0 | s1 | s2 |
|---|---|---|---|
| 113 | 4,500 | 6,500 | 9,700 |
| 121 | 10,700 | 10,300 | 5,000 |
| 125 | 12,200 | **no grok** | 8,700 |
| 119 | 16,300 | 15,600 | 21,300 |

### Scored against the pre-registration

- **E1 (does the engine grok?) HELD at all four moduli** — 113 3/3, 121 3/3, 125 2/3,
  119 3/3. Final test acc per seed: 113 all 1.0000; 121 1.0000/0.9978/1.0000;
  125 1.0000/**0.6082**/1.0000; 119 0.9869/0.9996/0.9966.
- **E2 (C6, clock at prime powers) HELD** — |Gini(121)−Gini(113)| = 0.013/0.196/0.067
  → 2/3 within the 0.10 threshold; |Gini(125)−Gini(113)| = 0.078/0.001 → 2/2. Tail mean
  over the final 10k steps, 11 samples per run.
- **E3 (C22/C23 causality) — 10 of 11 grokked runs at permutation p ≤ 0.005.** The one
  failure is `n119_s2` at **p = 0.115**. Restricted-vs-baseline: 113 BETTER 3/3
  (241.69×, 2.35×, 2.67×); 119 BETTER 3/3 (4.96×, 2.04×, 24.83×); 121 2/3
  (34.49×, 37600.76×, **worse 0.55×**); 125 s0 **worse 0.17×**, s2 BETTER 7.02×.
- **E4 (C8, CRT law) HELD** — only n=119 is testable (113/121/125 are prime powers and
  therefore VACUOUS). s1 enrich 3.20 p=0.0000, s2 enrich 3.15 p=0.0000 → 2/2.
- **E5 (engine == PyTorch) NOT HELD at all three moduli.** See below.

### O15 RESOLVED, and the non-grok is a gift

`n125_s1` never grokked (final test acc 0.6082). k02 saw exactly this — 4/5 seeds at
n=125, the failure being seed 1 — so it reproduces on a different engine with a different
RNG path. Its spectrum is the point: **Gini_mult 0.218, PR 14.03, n_key = 0.** The one run
that failed to generalize has **no clock at all**. That is a negative control produced for
free, and it makes every sparsity claim in this project falsifiable rather than merely
corroborated.

### E5 FAILED BY 4–8σ, AND THE CAUSE IS TRAINING PRECISION

|   | engine | k03 torch | \|Δ\| | band (1 sd of k03) | verdict |
|---|---|---|---|---|---|
| 113 | 0.754±0.016 | 0.556±0.015 | 0.197 | 0.026 | NOT HELD |
| 121 | 0.662±0.092 | 0.543±0.020 | 0.119 | 0.028 | NOT HELD |
| 125 | 0.805±0.036 | 0.543±0.052 | 0.262 | 0.048 | NOT HELD |

The pre-registration says a systematic disagreement here is "a finding about the science,
not an engine bug". **That was not taken at face value.** Ruled out in order:

1. **Not a protocol artifact** — `tail_gini` applies the same `mult_amplitude` + `gini` to
   both sides, same 10k tail window.
2. **Not a hyperparameter mismatch** — the k03 kernel source has
   `TRAIN_FRAC, LR, WD, BETAS = 0.30, 1e-3, 1.0, (0.9, 0.98)` and the same init scales
   (1/√d_model, W_O 1/√(n_heads·d_head), W_out 1/√d_mlp). Identical to the engine.
3. **Not a tail-window artifact** — a window-independent Gini on the FINAL weights
   reproduces it: **7 of 7 grokked matched pairs**, engine sparser by **+0.17 to +0.28**.
   The two pairs that invert are `121_s1` (engine 0.531 vs 0.556) and `125_s1`
   (0.312 vs 0.465) — respectively the partial-accuracy run and the **non-grokked** one.
4. **Not the optimizer implementation** — the engine's AdamW is algebraically identical to
   `torch.optim.AdamW`: same decoupled `lr·wd·p`, same bias-corrected denominator
   (`√(v/(1−β₂ᵗ)) + ε` ≡ torch's `√v/√(1−β₂ᵗ) + ε`).

**The difference is `src/autograd/engine.py:54` — `np.asarray(data, dtype=np.float64)`.
The engine trains in float64; k03/PyTorch trains in float32.**

Corroborating signature: engine `|W_E|` is consistently **smaller** (11.08–17.46) than
torch's (10.69–19.31) while being sparser. Sparser **and** smaller-norm **and**
faster-grokking is what less gradient noise under weight decay 1.0 should produce.

**HYPOTHESIS, NOT YET A CLAIM.** It is consistent with three separately-open questions at
once, which is exactly why it needs a controlled test rather than acceptance:

- **E5** — why the engine is sparser.
- **O1b** — why we grok at 4,500 where 2606.17399 reports 9k–14k. k08 already eliminated
  initialisation scale as the mechanism (negative, p = 0.0002, and running *backwards*).
- **O8/O11** — n=119's restricted loss was 0.97×±0.74 on torch over 5 seeds. On the engine
  it is **BETTER in 3/3 seeds**.

**The decisive test is pre-registered-able and cheap:** re-run the engine at float32,
n=113 seed 0, ~20k steps (Gini plateaus by 16–20k per C26), one variable changed, ~2.8 h
local CPU, zero quota. **NOT YET RUN.**

### O8/O11 does NOT reproduce — and the honest reading is stronger

The "restricted circuit is not better than baseline" pattern was, on torch, a property of
**n=119**. On the engine n=119 improves 3/3 and the pattern appears instead at `121_s2`
(0.55×) and `125_s0` (0.17×). So it is a **per-run property, not a per-modulus one**, which
kills the explanation that it comes from Z₆×Z₁₆'s group structure. O8 and O11 are not
"resolved"; they are **re-scoped**, and the Z₆×Z₁₆ hypothesis is dead.

### Caveats carried in the same breath

- Every number above is **3 seeds**, not 5. E1's criterion was ≥2/3 and is met, but 121's
  E2 is 2/3 and 125's is 2/2-of-3-with-one-excluded.
- **All pre-N7 results are float32 (scout); all N7 results are float64 (paper).** C4's Arm A
  replication matched the literature *because it was float32*. Any number quoted from this
  project must now say which regime it is in. This is new as of today and is not yet
  reflected everywhere in `FINDINGS.md`.
- `n119_s2`'s E3 permutation p = 0.115 is a genuine miss, not rounding.
- k03 never recorded `betas` in its provenance config; the value above comes from reading
  the kernel source, not from the artifact. A gap in the stamping, now known.

### Addendum, same session: the SUGGESTIVE ONLY bucket had never once been populated

Caught by running `session_brief.py` as the session-end checkpoint verification step rather than trusting
the edits. The ledger summary splits each table row on `|`, but a cell may legitimately
contain an escaped pipe — **C27 carries `\|ΔGini\| = 0.193`** — and a naive `split("|")`
shifts every later cell, so C27's *status* was read out of the middle of its *evidence* and
it fell into `other`. C27 is the only `SUGGESTIVE ONLY` row in the project, so **that bucket
has been empty in every brief ever printed**, silently implying no suggestive-only claims
exist. Fixed by splitting on unescaped pipes: `re.split(r"(?<!\\)\|", l)`.

The same run caught a defect introduced **by this session-end checkpoint**: C30's cell used raw unescaped
pipes (`|Δ|`, `|W_E|`), which is malformed markdown — it breaks the rendered table and
splits the row into 8 cells. Rewritten as `abs(Δ)` and "the engine's `W_E` norm".

Correct buckets after both fixes: **23 VERIFIED · 3 VERIFIED-1-seed · 2 SUGGESTIVE ONLY
(C27, C30) · 3 RETRACTED · 1 EXPLORATORY (C28) · 32 total.** C12 and C13 also moved into
VERIFIED, where they belong.

**Two session-end checkpoints in a row have now had their own verification step catch a defect in the
instrument being used to verify** (2026-09-12: the classifier read the whole status cell and
flipped C24 to RETRACTED). Run the brief *after* editing the ledger, never before.

---

## 2026-09-14 — Entry 28: the clock is in the neurons, and two instruments lied about it

**Session 7.** N9 (the float32 precision A/B) pre-registered, launched and running locally;
everything below is zero-compute re-analysis of artifacts that were already on disk while it
ran.

### What landed

**C31 — O4 is closed.** C6, the thesis's central result, had rested entirely on embedding
spectra while the paper it is measured against carries neuron-level support. It now has the
same support: **74–89% of the 512 MLP neurons at 121 and 125 are single-frequency objects in
discrete-log coordinates** (113 reads 94.9% against a published 96.9%), with the same neurons
at **0.0%** in raw coordinates and **0.00%** under a per-neuron cell shuffle. The tuned
neurons land on **exactly the embedding's key frequencies in 14/14 grokked runs**. Full
numbers in FINDINGS §3.7; criteria in `experiments/PREREGISTER_o4_neurons.md`, committed
before the 121/125 numbers existed.

**The negative control arm exists at last.** Every sparsity claim in this project had been
corroborated and never contrasted. Runs that *failed* to generalise read **6.8%** (engine
`n125_s1`, acc 0.6082) and **0.0%** (k04 `n49`/`n54`, acc 0.15–0.27) against 71.7–100% for
the grokked ones — from artifacts already saved, at no compute cost.

**C32 — k08's causal half is scored.** `analyze_n4.py results/k08_init`: permutation
**p < 0.01 in 24/24** runs across all three initialisation conditions, so C22/C23 do not
depend on init scale. O16's anomaly **reproduces under the permutation statistic** —
restricted loss worse than baseline in **3/8 `upstream_heavy`** seeds and **0/8** in both
`fanin` and `downstream_heavy`. To be exact about what is new: session 6 had already read
that split off k08's own output and written it into O16, so this is **confirmation under the
proper statistic**, not a new sighting. What is new is the 24/24 causality result and the
probe answer below. It tracks the unbalanced-init condition, not the modulus, which is one
more nail in the "it comes from Z₆×Z₁₆'s group structure" idea. The run also answers O8's
own proposed probe: key set by **ablation rank** equals key set by **embedding norm** in
14/24 runs, differs by one character in 9/24, two in 1/24 — the 5×-median detector is close
but not exact.

### Two instruments lied, and both lies pointed the same way: toward a weaker result

**1. An off-by-one between two bin conventions.** The first scoring reported the neurons'
frequencies overlapping the embedding's key set **0 of 4** — while the two lists sat exactly
one apart on every element: neurons [13, 27, 33, 43], embedding [12, 26, 32, 42].
LAB_PROTOCOL.md carries the warning *"`energy()` folds conjugates, so its index IS the frequency —
do NOT add 1"*, and that warning is correct **about `energy()`, which keeps DC at index 0**.
`mult_amplitude` **drops** DC, so its index 0 is frequency 1 and +1 is required. One warning,
two conventions, and the warning applied to the other one.

Settled the way this project settles bin conventions — by planting a pure character of known
frequency and looking. `mult_amplitude`'s argmax lands at k−1 (1→0, 5→4, 17→16, 40→39);
`freq_fraction`'s `best_k` lands at k. With the fix the overlap is 4/4, 5/5, 3/3 everywhere,
and the lists reproduce the `key_mult` column FINDINGS §3 has been printing since k02.
**A third bin-convention bug in this project, and the second to be caught only by planting a
known signal.** Reasoning about the convention has now failed every time it has been tried.

**2. A control that was not a control.** k03's `n125_s1` was being scored as "the ungrokked
negative control" on a `> 0.99` gate. It ends at **0.9779** — the exact figure C3 records
from k02 — and read **74.8% tuned**, indistinguishable from the grokked runs. Left alone it
would have destroyed the very contrast the control exists to provide, and it would have done
so *quietly*, by making the negative control look positive. A model at 97.8% test accuracy
has learned the circuit. There are now three states (grokked / near-grok / FAILED) and the
control arm is built from runs that actually failed.

> **A binary gate turns every near-miss into whichever side it falls on.** `>0.99` is the
> right gate for "is this a data point"; it is the wrong gate for "is this a control". The
> failure mode is silent and it biases *against* the finding, which is the direction nobody
> checks.

### Also this session

- **A silent float32 upcast in the engine.** `Tensor.max`'s backward divided the gradient by
  an int64 tie count, and float32/int64 promotes to float64 under NEP 50. Both softmaxes
  route through `max`, so an `ENGINE_DTYPE=float32` run would have trained with float64
  gradients on W_Q, W_K, W_in, W_out and W_U — invisible from the loss curve, and not the
  experiment N9 pre-registers. Fixed, proven bit-identical at the float64 default on tied and
  untied inputs, and `test_autograd.py` now asserts per dtype that weights, gradients, Adam
  state and loss are **all** the requested precision.
- **`neuron_frequency_map` no longer carries its own copy of the statistic** — it imports
  `src/analysis/neurons.py`, so the figure and the reported number cannot drift apart.

### What this did NOT show

The prime powers do **not** match the prime: 72–89% against 92–100%, in the same direction at
both 121 and 125. Reported as measured, unexplained, and not smoothed over.

---

## 2026-09-14 — Entry 29: N9 kills C30. Precision was not the answer.

**Same session as Entry 28.** N9 ran 3 × 40,000 steps at `ENGINE_DTYPE=float32`, 3-way
local CPU, 3 h 13 min, zero Kaggle quota. All three artifacts `git_dirty=False` at SHA
`a4b3d14` with `dtype` stamped. Criteria in `experiments/PREREGISTER_n9_precision.md`,
committed before the data existed.

### The result

| regime | tail Gini_mult | mean | `|W_E|` | grok steps |
|---|---|---|---|---|
| engine float64 | 0.7623 · 0.7307 · 0.7679 | 0.7536 | 11.610 | 4,500 · 6,500 · 9,700 |
| **engine float32** | **0.7607 · 0.7785 · 0.7622** | **0.7671** | **11.642** | **4,500 · 6,400 · 9,600** |
| torch float32 | 0.5573 … 0.5825 | 0.5665 | 15.152 | 6,800 … 5,400 |

**P0 HELD** (3/3 grok) · **P1 NOT HELD, f = −0.07** · **P2 NOT HELD, f = 0.01** ·
**P3 HELD** (still multiplicative: Gini_add 0.005–0.016, Gini_rand 0.11–0.13).

Changing precision moved the headline statistic by **+0.014, away from torch**. Not a near
miss, not a partial — the sign is wrong. **C30 is retracted.** And grokking time is
precision-invariant to within 100 steps at every seed, so "float64 groks faster, which is
why we grok at 4,500 where the literature reports 9k–14k" is dead too. **O1b returns to
open.**

### What it cost to be wrong, and what it bought

The previous session called C30 "the finding that changes the paper" and had it unifying
three open questions at once (E5, O1b, O8/O11). It was a 7/7 correlation with a mechanism
that predicted the right *signature* — sparser **and** smaller-norm **and** (apparently)
faster-grokking is exactly what less gradient noise under wd = 1.0 looks like. Every part
of that reasoning was sound and the conclusion was still false.

> **A correlation at 7/7 with a mechanism that explains three loose ends is the most
> dangerous shape a wrong claim can take.** It is not rescued by being careful about the
> statistics; it is only rescued by running the variable.

What it bought: the claim never reached the paper, the run cost zero quota, and the
elimination list is now nearly complete.

### The gap is still there — O18, and it is harder than before

Eliminated as of tonight: analysis protocol · hyperparameters · tail window · optimizer
(both reduce to `p(1−lr·wd) − lr·m̂/(√v̂+ε)`; checked by reading both implementations) ·
gradient correctness (2.68e-15) · **minibatch noise — both are FULL-BATCH**, k03 calls
`m(xtr)` on the whole training set every step exactly as `run_n7.py` does · **initialisation
distribution**, per-parameter norms agree to ~1% · **the train/test split — byte-identical**,
`modular_data(113,"mul",0.30,0)` and k03's `dataset()` select the *same 3,830 pairs*,
verified by set comparison · **precision**.

Standing: the specific random init **draw** (numpy's stream vs torch's, same distribution),
and the execution substrate — k03 ran on a Kaggle **T4 GPU**, the engine on CPU.

**The split check is worth its own line.** `run_n7.py` carries the comment *"same split RNG
as k03's dataset(): E5 is paired"*. That comment was an assertion nobody had run. It turned
out to be true — but it was load-bearing for a paired comparison and had never been checked,
and the session that wrote E5 treated it as established.

### A second fact any explanation now has to fit

Neuron tuning (Entry 28's statistic) runs the **opposite way** to Gini:

| regime | tail Gini_mult | tuned neurons, n=113 |
|---|---|---|
| engine float64 | 0.754 | 82.4 · 90.2 · 93.4 % |
| engine float32 | 0.767 | 82.2 · 70.7 · 90.0 % |
| torch float32 | 0.567 | 94.9 · 91.8 · 100.0 · 99.8 · 98.0 % |

The engine has the **sparser embedding** and the **less pure neurons**; torch has the
reverse. A single "more or less converged" axis cannot produce that ordering. It looks like
a *different solution*, not a differently-converged one — which is Zhong et al.'s
Clock-and-Pizza setting (small changes induce qualitatively different algorithms), and wants
their discriminator rather than another sparsity scalar.

### Next test, cheap and decisive

Run the k03 recipe with **torch on CPU**, locally, one seed. **~0.77 ⇒ the difference is
GPU-specific. ~0.55 ⇒ it is in the code**, and the two implementations can then be bisected
line by line against each other. Zero quota either way.

---

## 2026-09-14 — Entry 30: Gate 2 is asking the wrong question, and our statistics only ever saw one stratum

**Same session as Entries 28–29.** No training. Two measurements and three papers read.
Recorded because it was produced in conversation and would otherwise be lost.

### The measurement that reframes Gate 2

Every multiplicative statistic in this project — C6, C22, C25, C31 — is computed on the
**unit** sub-grid, via `unit_index` / `_reorder_dlog`. That is nearly the whole
multiplication table at a prime and nowhere near it at a composite:

| n | unit × unit cells | divisor strata `gcd(x,n)` | non-unit × non-unit that is ≡ 0 |
|---|---|---|---|
| 113 | **98.2 %** | {1, 113} | 100 % |
| 121 = 11² | **82.6 %** | {1, 11, 121} | **100 %** (trivial stratum) |
| 125 = 5³ | **64.0 %** | {1, 5, 25, 125} | **36 %** (a real intermediate stratum) |
| 119 = 7·17 | 65.1 % | {1, 7, 17, 119} | 44.8 % |
| 165 = 3·5·11 | **23.5 %** | 8 divisors | 10.9 % |

And the model solves the part we never look at. Per-stratum accuracy from k03's saved
`logits_all`, seed 0, full grid (final test acc is 1.0000, so this is not a train/test
artefact):

```
n=125  gcd(ab,n)=1  64.0% acc 1.0000 | =5 25.6% 1.0000 | =25 7.7% 1.0000 | =125 2.7% 1.0000
n=165  8 strata, 23.5%..2.8% of cells, every one acc 1.0000
```

> **There is a fully-learned mechanism on 36 % of the n=125 table and 76.5 % of n=165 that
> this project has never once measured.** C6 and C31 are statements about the unit stratum.
> That is not wrong, but it has never been said.

The structures are also qualitatively different, and the difference is theory-derived rather
than curve-fitted: a prime power carries a **p-adic filtration** (few strata, nested in a
chain, non-units nilpotent — at 121 the non-unit × non-unit block is *identically zero*, at
125 it is not), while a square-free composite carries a **CRT divisor lattice** (many
strata, shallow, not nested).

### What the papers say, read this session

- **Chughtai, Chan & Nanda 2302.03025** — universality is two-level: the circuit *family* is
  always GCR, but "the precise circuits learned — as well as the order they develop — are
  arbitrary". Their own phrase is "mixed evidence for universality". **We reproduced that
  shape without naming it**: C25 proves a key-frequency *value* is not a property of the
  model (only the set up to multiplication by a unit mod φ), and our key sets scatter across
  seeds while Gini, PR and the count do not.
- **Zhong et al. 2306.17844 (Clock/Pizza)** — on a *fixed* task, "small changes to model
  hyperparameters and initializations can induce discovery of qualitatively different
  algorithms", with sharp phase transitions in width and attention strength. This is the
  standing objection to any "circuits differ across algebraic conditions" claim: we varied
  n at one point in hyperparameter space. **Entry 29 makes it concrete** — our own engine and
  our own torch kernel may be running different algorithms on byte-identical data.
- **Chen et al. 2607.07066 (Multiplication Beyond Groups)** — GCR "localizes to disjoint
  algebraic strata that partition the input space", attention doing class-sensitive routing.
  **Their Table 1 is 113, 143, 154, 165 and marks every one `Square-free: True`.** Prime
  powers are the complement of their design and the case they name as open.

### The Gate 2 verdict I would propose — NOT YET DECLARED

The gate's three branches are *differ* / *invariant* / *don't grok*. Branch 3 is dead (all
grok). Branches 1 and 2 are **both** ticked by our evidence, because the question conflates
two levels:

> **The local mechanism is invariant; the stratification is not.** Every algebraic stratum
> runs the same character-based clock. What the algebra of n changes is how many strata
> there are and how they nest — one (field), a nested p-adic chain (prime power), or a flat
> CRT divisor lattice (square-free composite).

That matches Chughtai locally and Chen globally, and extends Chen from square-free to the
prime powers their design excludes. **It is a fourth branch the gate does not contain, so it
must be pre-registered before the stratum analysis runs** — declaring it after looking would
be exactly the Gate 1 C6/C7 ordering error again.

### Process lessons from this session

- **A heredoc that edits a file behind `assert` writes NOTHING when a later assert fails.**
  Commit `f80829d` claimed O16 was re-scoped in STATE.md §4; it was not. The script raised
  `AssertionError` on its second assert, and the file is only opened for writing after both
  pass. It went unnoticed because the next line of the same shell block ran
  `session_brief.py` — on a *new line*, not after `&&` — and the ledger section it printed
  looked right. **It was right, from the previous commit.** Reading a verification that does
  not depend on the edit is not verifying the edit. Fixed in `2feb058`.
- **Check what a scoring run claims as NEW.** C32 was first written as though the 3/8
  `upstream_heavy` split were a discovery; O16's own text had recorded it from session 6.
  The run confirms it under the permutation statistic. Corrected in the same commit.
## 2026-09-14 — Entry 31: Gate 2 is declared on a fourth branch, and the strata answer

**Session 8, same day as Entries 27–30.** Zero Kaggle quota. Two local jobs: the G2 stratum
analysis (pure re-analysis of `logits_all` / `mlp_acts` already on disk) and the O18
substrate bisect (3 × 40k steps, torch on CPU).

### What was pre-registered, and when

`experiments/PREREGISTER_gate2_strata.md`, committed `bca6236` **before a single
per-stratum model number existed**. That ordering is the whole point: the verdict it
proposes is a **branch Gate 2 does not contain**, and declaring a fourth branch after
looking at the data would have been the Gate 1 C6/C7 error again, worse.

### The measurement

Every multiplicative statistic in this project — C6, C21, C22, C25, C31 — is computed on
the **unit** sub-grid. Entry 30 established that is 98.2 % of the table at n=113 and
82.6 / 64.0 / 65.1 / 23.5 % at 121 / 125 / 119 / 165.

The algebra that makes the rest measurable, and it is a theorem rather than a hunch: for
x ∈ J_d write x = d·u with u a unit mod n/d, likewise y = e·v. With m = gcd(de, n) and
de = m·w, gcd(w, n/m) = 1:

```
x·y mod n  =  m · ( w · (u·v)  mod  n/m )
```

So a stratum is a fixed unit relabelling of **multiplication in the smaller ring
Z/(n/m)Z**; the target is exactly constant on the fibres of (Z/(n/d))\* → G_m; and, since
multiplying units is adding exponent tuples, a local clock must put its logit energy on the
**diagonal (κ, κ)** in local-group coordinates. `analyze_gate2.py` re-indexes each block by
discrete log on both axes, fibre-collapses onto G_m, and scores G0–G5.


### The result — 32 strata, 6 moduli, 157 stratum-run measurements

Twelve distinct local groups (Z₁₀, Z₁₆, Z₂₀, Z₁₀₀, Z₁₁₀, Z₁₁₂, Z₂×Z₁₀, Z₄×Z₁₀, Z₂²×Z₄,
Z₂×Z₄×Z₁₀, Z₂³×Z₄, Z₆×Z₁₆) and **every one carries the same circuit**. The measured share
of the multiplication table rises from **23.5 % to 82.3 %** at n=165 and **64.0 % to
89.6 %** at n=125. Full table: `scripts/gate2_table.py`. All seven criteria HELD on the
primary arm (k03, torch float32, 5 seeds; 4 at n=125), and the **from-scratch engine
independently reproduces the verdict** at 119, 121 and 125.

**GATE 2 IS DECLARED, on the fourth branch**: *the local mechanism is invariant; the
stratification is not.* That was the rule fixed in §3.1 of the pre-registration, applied
as written.

### And the pre-registration's own falsifier fired

§4 said: *"The negative-control arm passing G1. A failed run must not show a local clock.
If `n125_s1` (acc 0.6082) reads the same as a grokked run, the statistic is measuring the
grid, not the model, and nothing here is a result."*

| criterion | grokked | **FAILED** | separates? |
|---|---|---|---|
| G0 held-out acc ≥ 0.95 | 100 % | **0 %** | YES |
| **G1 perm p < 0.01 — PRIMARY** | 99 % | **100 %** | **NO** |
| G2 excluded/baseline ≥ 100× | 100 % | **0 %** | YES |
| G3 fibre residual | 100 % | 33 % | partly |
| G4 R ≥ 5 | 100 % | 63 % | partly |
| G5 tuned − shuffled ≥ 0.20 | 98 % | **0 %** | YES |

**30 % of every block is training data.** A model that has merely memorised its training
cells still has logits that are a function of u·v *there*, so the diagonal still carries
the energy and removing it still hurts more than removing a random bin set. G1 tests
"the logits are a function of the product", and memorisation satisfies that.

The verdict survives because "carries a local clock" was defined **in advance** as G1 ∧ G2,
and G2 separates perfectly — as do G0 and G5. But the pre-registration was **wrong to name
G1 primary**, and it named it primary *because it is the project's established statistic*
(C22, C23, C32). That is inherited authority, and inherited authority is exactly what a
pre-registration is supposed to make checkable rather than assumed.

> **A statistic that is load-bearing elsewhere is not automatically load-bearing here.**
> C22 applies it to the unit stratum of a *grokked* model, where memorisation is not a live
> alternative. On a 1,100-cell block that is 30 % training data, it is.

Two further corrections, both making the result smaller:

- **Entry 30's "the model solves every stratum at acc 1.0000" was the FULL GRID**, 30 % of
  which is training data. On held-out cells, 13 stratum-runs fall to **0.500–0.833**. Every
  one is below the measurability cut, so the verdict is untouched — but the sentence was
  wrong and is corrected in FINDINGS §3.8 rather than dropped.
- **The first control arm had the binary-gate defect this project has already paid for.**
  It conflated FAILED runs (0.10–0.75) with **near-groks** — k04 n=75 at 0.9855, n=99 at
  0.9794, n=100 at 0.9880, engine n119_s0 at 0.9869 — and those promptly "declared the
  fourth branch", which is what a model that has learned the circuit is supposed to do.
  Three states now, never two. The corrected control arm returns **NO VERDICT**.

### Process lessons from this session, all self-inflicted

- **`reproduce.sh` had drifted thirteen scripts behind the repo** — including
  `test_crt_law.py` (C8, the best-supported claim here) and `analyze_n7` / `n9` / `o4`
  (C29, the C30 retraction, C31). Constraint 0 makes it the single entry point; nothing was
  checking that it still was. `test_reproduce.py` now fails the self-check suite when a
  claim-carrying script is missing, and `session_brief.py` runs it. **A checklist only a
  human re-reads rots quietly.**
- **A fifth waiter that could never terminate, written with the gotcha open in LAB_PROTOCOL.md.**
  `until ! pgrep -f "analyze_gate2.py"` inside a `bash -c` whose argv contains
  `analyze_gate2.py` matches its own parent shell. The chained second pass never started;
  the analysis had been finished for minutes. `scripts/alive.sh` is the fix — whole-field
  match, plus an argv[0]-is-python guard so the wrapper does not match either.
- **And that script's own first version failed in the worst direction.** `ps -eo
  pid=,args=` puts the PID in `$1` and argv[0] in `$2`; the guard tested `$1`. It reported
  a live 40,000-step run as **idle** — the failure that makes the next session relaunch on
  top of a running job. Caught only by testing it against a job known to be alive. *A
  liveness check that has only ever been run against a live box, or only ever against an
  idle one, has not been tested.*
- **Then killed my own shell (exit 144) for the third time in this project's history**, with
  `ps ... | grep "[g]ate2_all_done"`. The bracket stops GREP matching itself — but my own
  command string contained the marker path verbatim, so the shell running it matched. Two
  chains had already ended up running concurrently and appending to the same four logs.
  `scripts/run_gate2_arms.sh` runs all four arms in one process so they cannot race.
- **`hist[-1][3]` means a different quantity in the two arms.** k03/k04 write
  `step, train_loss, test_loss, test_acc`; `run_n7.py` writes
  `step, train_loss, test_loss, TRAIN_acc, test_acc`. Indexing `[3]` on the engine reads
  **train** accuracy — which would have let `n125_s1` (test 0.6082, train ~1.0) into the
  primary engine arm as a data point, and made every engine split-check compare a
  recomputed test accuracy against a logged train accuracy. Read by column name.
  **Same family as `$4=="run_n7.py"` and as the `$1`-vs-`$2` above: never index to a fixed
  field when the fields can move.** Caught before the engine arm ran, by checking
  `hist_cols` across all three directories rather than assuming they agreed.


## 2026-09-14 — Entry 32: O18's substrate bisect. It is not the hardware either.

**Same session as Entry 31.** 3 runs × 40,000 steps, sequential, local CPU, **108 min
wall, zero Kaggle quota**. Criteria in `experiments/PREREGISTER_o18_substrate.md`,
committed before the data existed.

### The result

| regime | tail Gini_mult | mean | `‖W_E‖` | tuned | grok steps |
|---|---|---|---|---|---|
| engine float64 | 0.7623 · 0.7307 · 0.7679 | **0.7536** | 11.610 | 0.887 | 4,500 · 6,500 · 9,700 |
| torch **T4 GPU** | 0.5573 … 0.5825 | **0.5665** | 15.152 | 0.969 | 6,800 … 5,400 |
| **torch CPU** | **0.5759 · 0.5672 · 0.5942** | **0.5791** | **14.296** | 0.938 | 5,800 · 7,200 · 5,600 |

**Q0 HELD** (3/3 grokked) · **Q1 NOT HELD, `f = 0.068`** · **Q2 NOT HELD, `f = 0.242`** ·
Q3/Q4 descriptive. Torch on a CPU behaves like torch on a T4, not like the engine.

Two things follow. **No Kaggle number in this project is carrying a hardware artefact** —
which was the live worry, since every torch number came off a T4 and every engine number
off this box. And **the gap is in the code**.

### What is left of O18, and why it is now cheap

Eliminated, each by the run that did it: protocol · hyperparameters · tail window ·
optimizer · gradients (2.68e-15) · minibatch noise, both full-batch · init *distribution* ·
the train/test split, byte-identical · **precision** (N9) · **the substrate** (this).

What remains is the specific random **init draw** — numpy's stream against torch's — or a
structural difference nobody has found. **The next test is not a sweep.** Both
implementations now run on the same box, so: start the engine from torch's *own sampled
weights* and compare step by step. If ~0.19 Gini survives byte-identical starting weights,
it is in the update path, and the update path is about 400 lines.

### Three defects in the instrument, all found by running it

1. **The pre-registration named a command that did not work.** Its analysis plan is
   "`analyze_n9.py results/o18_cpu` — the same scorer N9 used". It found **zero runs**:
   `analyze_n7.load` globs `WE_engine_*`; `run_o18_cpu.py` writes `WE_B_thesis_*`. It did
   not error — it printed **"P0 NOT ASSESSABLE (no runs)"**, which is indistinguishable
   from a sweep that failed to grok, *on a run that had grokked at step 5,800*.
   **I pre-registered a command without ever running it.** LAB_PROTOCOL.md already records this
   exact failure for `analyze_n7.py`, found only because that chain was rehearsed on
   retired data. Rehearsing costs seconds.
2. **That scorer then labelled a PyTorch-on-CPU run "engine float32"** — its labels were
   hard-coded for N9, where D32 really was the engine. A committed log saying the opposite
   of what a run was is the same class as `hist[-1][3]`: an assertion where a reading was
   available. Every npz already carries `config.engine`.
3. **It printed one `f` without saying which.** N9's `f` is the fraction of the gap the new
   arm *closes*; O18's pre-registered `f` is the complementary fraction the arm still
   *carries*. Same numbers, opposite convention, nothing on screen to say so — 0.93 and
   0.068 are the same measurement. Both are now printed and named.

> **A sign convention that lives only in a pre-registration will be read off a log by
> someone who does not have it open.** Print both, named, or one of them ends up in a paper.


## 2026-09-14 — Entry 33: the C24 re-run that had been owed since 2026-09-12

**Same session as Entries 31–32.** STATE.md recorded `analyze_k03.py` as **owed** —
`analyze_k03.spectral()` had fed **energy** into `gini` (reading 0.902 where the amplitude
protocol reads ~0.54–0.58), the code was fixed on 2026-09-12, and the script had not been
re-run since, so nothing had confirmed the fix on the full grid. Run now:
`PYTHONPATH=. .venv/bin/python analyze_k03.py`, output `logs/k03_rerun.log`.

### It reads the amplitude convention, and the drift figures hold

Final-checkpoint Gini_mult at n=113 is **0.538 / 0.528 / 0.585 / 0.587 / 0.582** across the
five seeds — the amplitude protocol — where the defect read 0.902. Arm A reads **0.543**
against C4's published-comparison 0.579.

H4/H5, drift over the final 10k steps (criterion < 5 %):

| n | Gini drift | PR drift | converged |
|---|---|---|---|
| 113 (Arm A) | 1.8 % | 1.0 % | 1/1 — CONVERGED |
| 113 | 5.2 % | 2.3 % | 2/5 |
| 121 | 5.1 % | 2.8 % | 3/5 |
| 125 | 4.8 % | 5.6 % | 2/5 |
| 119 | 2.5 % | 5.9 % | 3/5 |
| 120 | 8.4 % | 7.7 % | 1/5 |
| 165 | 5.9 % | 4.8 % | 0/5 |

Per-seed Gini drift spans **0.1 % – 19.7 %** (the extreme is n=120 seed 4). STATE.md
recorded the corrected range as "0.3–16 %"; the re-run **extends the top of that range to
19.7 %** and otherwise agrees. C24's PR claim is untouched — PR belongs on energy.

**The debt is discharged**, and the point it was recorded for is confirmed: re-running the
script no longer regenerates the retracted 0.90.


---

## 2026-09-15 — Entry 34: C27 retracted. The network never de-grokked; the run ended mid-spike.

**Corrects Entry 30's k06 R4 section.** Append-only: that entry stands as written; this one
supersedes its reading of n=119 seed 0. Script: `analyze_excursions.py` (new, self-checked,
in `reproduce.sh`). Zero compute — every number below is re-analysis of artifacts that have
been on disk since 2026-09-11.

### How it came up

The question asked was *"is C27 a double descent case?"* Answering it needed the curve, not
the endpoint, so I pulled the trajectory instead of reasoning from the claim. Two rows
settled it before any analysis:

```
119800   acc 0.9987   train_loss 0.0000
120000   acc 0.6798   train_loss 0.1087     <- final logged sample
```

Accuracy is flat at **0.9985–0.9992 from step 32,000 to 119,800** — 88,000 consecutive
steps — and then moves in one 200-step interval. Double descent needs a visible ascent.
There is no ascent. So the answer to the question as asked was "no", and the interesting
part was what the shape *is*.

### The measurement that retracts the claim

Counting post-grok excursions below acc 0.90 across all 12 k06 runs:

| | |
|---|---|
| excursions, 11 grokked runs | **19** |
| length of each | **exactly one 200-step logging interval** |
| recovered | **18 of 19** |
| the 19th | begins at step **120,000** — the last sample. No next observation exists |

n=119 seed 0 is not the run that broke. It is the run whose spike **landed on the final
checkpoint**. It had a *worse* one at step 115,400 — acc **0.4869** — and was back to 0.9989
within 200 steps. At 0.33 % of post-grok samples, the chance that at least one of 12 runs
ends inside a spike is ≈ 4 %. That is what happened. It is right-censoring.

### The circuit was never lost, and C27 was reading the wrong basis anyway

(Z/119)\* = Z6×Z16 is **non-cyclic**, so the multiplicative readout is the product-character
transform (`analyze_n7.mult_amplitude`):

| step | \|W_E\| | Gini_mult | PR_mult |
|---|---|---|---|
| 116,000 | 12.824 | 0.5758 | 6.710 |
| 120,000 | 12.552 | **0.6297** — *sparser* | 5.634 |

The embedding is intact and marginally sparser. What broke is the **readout**: final max
logit **3.71** against 14–95 in every sibling run, max softmax p 0.939 against 0.9996–1.0.

And C27's spectral evidence was never in the circuit basis. `analyze_k06.spectra` returns
`Gini_mult = None` when the unit group is non-cyclic, so its quoted numbers are the
**additive** column. Reproduced exactly: PR_add **19.174**@40k → **12.557**@120k, Gini_add
0.2573 → 0.4499 (Δ = **0.1926** = the "0.193"). LAB_PROTOCOL.md is explicit that on a
multiplication run the additive column is the *control* basis. So the claim was a
control-basis statistic, measured on a transient checkpoint, reported as circuit evidence.
**Two independent defects, either one fatal.**

### What I swept afterwards, and what it found

All nine results directories. **`WE_B_thesis_n119_s0.npz` is the only censored artifact in
the project** — every other run's final sample sits at its plateau. The contamination is
contained to C27, which is the outcome worth having checked rather than assumed.

Two things fell out of the sweep that were not the point of it:

1. **k06's R3 gets STRONGER.** The "twelfth run at \|ΔGini\| 0.193" was measuring the spike.
   Drift is 0.008–0.048 in **12 of 12**, not 11 of 12. Amendment appended to
   `experiments/PREREGISTER_k06_horizon.md` below its Outcome; nothing above it edited.
2. **O19 opens.** The engine shows **0** post-grok excursions in **420,100** steps at *twice*
   the logging resolution (100-step interval); torch shows **33** in **4,389,800**. Expected
   ~3.2 at torch's rate, P(0) ≈ 0.04. **Suggestive only, not pre-registered, and confounded**
   — the engine defaults to float64 and torch is float32, which is the boring explanation and
   is probably the right one. Note this is *not* the dead C30 precision claim: C30 was about
   sparsity magnitude, and "precision is dead" does not extend to numerical spiking. It costs
   nothing to settle inside O18, which pairs the two on this box by construction.

### The lesson, which is a repeat

A binary read of a **single final sample** cannot tell a transient from a state. This is the
same family as C26 (Gini at one checkpoint is a ±0.02–0.044 estimator) and as the
grokked/near-grok/FAILED gate: in all three, one instant was asked to carry a claim about a
regime. The general fix is the one already written for Gini — **define a run's endpoint as a
window mean, never as its last row** — plus the specific one: an excursion that touches the
final sample is **censored**, and censoring must be reported as its own state rather than
collapsed into an outcome. `analyze_excursions.py` prints `*CENSORED*` for exactly that, and
its selfcheck asserts that truncating a recovering dip one sample after it starts flips the
same dip to censored.

This also shows why R4 failed as a *criterion*, not as a prediction: "no late grok, no
de-grok" had no way to express "we did not observe long enough to tell". A horizon run must
either log densely enough to watch a spike end or score its endpoint over a window.

---

## 2026-09-15 — Entry 35: C28 survives a structure-matched null. Its p-value does not survive a failed run.

Pre-registered `experiments/PREREGISTER_c28_crt_null.md`, committed `832635c` **before any
number this entry reports existed**. `test_crt_null.py`, `logs/c28_null.log`, 42 s, zero
quota. Follows Entry 34 in the same session.

### The question I did NOT test, and why

O14 asked for a pre-registered sweep of CRT enrichment vs step. **That sweep was already on
disk** — `results/n7_engine` snapshots `we_traj` every 1,000 steps — and running it took
seconds. Step 0 is the model's **own random init at the same seed and the same split**,
which is the control C28 never had (C28 compared against *generic* random embeddings). It
rejects 0/3; step 1,000 rejects 3/3; the ramp is monotone to grokking. So C28's observation
was confirmed and sharpened for free, and pre-registering it would have been theatre. **That
result is written into the pre-registration's Motivation section**, above the criteria, so
it can never be read as an outcome of this experiment.

The question actually worth the file was the **null**. `permutation_test` scores the CRT set
against *uniformly random* same-size subsets, and the CRT set is a **union of two subgroups**
— 11 of 59 folded bins at n=119. A uniform null is weak against any alternative that
concentrates energy on arithmetic progressions through 0. C7b, C20 and C19 all died of that
exact confound. And the stake was bigger than C28: **C8 is VERIFIED 9/9 on this same test.**

### One falsifier I proposed, then killed before writing the file

I had wanted generator equivariance (C25) as a falsifier: relabel, watch the predicted set
move, check the enrichment follows. **It cannot fire.** In the additive basis the CRT set is
a union of subgroups of Z/n, and I checked before committing: it is fixed by **all 96 units
mod 119**. It does not move. Recorded in the pre-registration under its own heading so the
next reader does not re-propose it.

### Result — two halves that point opposite ways

**Null B (unions of multiples-of-d progressions over NON-divisors of n) is as hard as
uniform, and the signal survives it.** N1 3/3 at step 1,000 (p_B = 0.000, enrichment
1.450/1.225/1.224 at test acc 0.054–0.075) · N2 3/3 · N3 ramp ρ = **1.000/0.991/0.999** ·
N4 calibration 0/3 at init. Null C (mean-|k| matched) agrees. **The enrichment is about the
factorisation of n.** C8's instrument survives.

**N5, discrimination, FIRED.** Every FAILED run at a well-posed modulus rejects at
p_B = 0.000 too — including one at **test acc 0.2236**. Third time in this project a
load-bearing p has passed on the failure arm, after C22/C23's control-mean and G2's G1.
What separates is the **magnitude**, cleanest within one modulus where the predicted set is
identical by construction:

| n=63, 7 of 31 bins | enrichment |
|---|---|
| s0 grokked (0.9969) | **4.687** |
| s1 FAILED (0.2236) | 1.626 |
| s2 FAILED (0.2749) | 1.574 |

Grokked 3.27–17.44 (44 runs) vs FAILED 1.52–2.67 (4). **Report the enrichment, never the p.**

### The caveat that costs C28 its most interesting reading

Step-1,000 enrichment is **1.22–1.45**. A *memorising* model at n=63 reaches **1.57–1.63**
at convergence. The early signal is therefore real, CRT-specific, and **below the failure
band in magnitude**. "CRT structure is imprinted early" survives as a claim about
significance; it does **not** survive as evidence that the early structure is the grokking
circuit forming. That is O20, and it needs a *failing* run at a composite non-prime-power
modulus with dense snapshots — n=125 s1 is a prime power and vacuous, so the existing
negative control cannot answer it.

### Two defects the run found in my own code, and one of them is yesterday's lesson

1. **The discriminator classified runs by their final row.** First pass put k06's
   `n119_s0` in the **FAILED** arm at acc 0.6798 — the censored transient retracted in
   Entry 34, hours earlier, in this same session. Window median is 0.9987. Fixed to the
   median of the final 10 samples, and it now prints a `[censoring]` line whenever the two
   disagree by > 0.05; that line fires on exactly one run in the project. **Writing the
   lesson down did not stop me re-committing it in new code an hour later** — the guard that
   worked was making the code print the disagreement, not the prose.
2. **Null B is not always constructible.** At n=54 the predicted set is **14 of 27 bins
   (52 %)**; at n=98, 25 of 49 (51 %). A size-matched union of 1–3 non-divisor progressions
   rarely exists there. I did **not** widen the draw to force a fit — that would be changing
   the procedure after seeing the result. They report `n/a` and a `near-VACUOUS set` flag now
   prints above 40 %. The cost is honest and material: **the FAILED arm is only 2 runs at a
   well-posed modulus**, and N5's conclusion carries that limit.

### Bookkeeping

C28 → **VERIFIED for the specificity half** (3 seeds, n=119 only, engine arm, never pooled
with torch), with N5's failure and the magnitude caveat attached to the row. O14 closes;
**O20 opens**. FINDINGS §5.1b. `test_crt_null.py` added to `reproduce.sh` (30 scripts
covered). The near-vacuity observation is new and worth carrying: the CRT-dual law is
**structurally weaker at highly composite moduli**, where the predicted set approaches half
of all bins — the same vacuity that makes prime powers useless, arriving by degrees.

---

## 2026-09-15 — Entry 36: O18's update path is eliminated. Three hypotheses down, and the fourth already has a problem.

Pre-registered `experiments/PREREGISTER_o18_transplant.md`, committed `3ebbab6` **before any
transplant number existed**. `run_o18_transplant.py --stage1 113 0 500`,
`logs/o18_transplant.log`, `results/o18_transplant/stage1.json`, ~5 min, zero quota. Third
entry this session.

### I did not run the experiment STATE asked for

STATE's action 1 said: transplant torch's init into the engine, run both 40k steps, compare
`tail_gini`. At 449 ms/step that is **~5 hours** to learn one number. A **paired step-by-step
comparison** from the same weights answers a sharper question in five minutes and can
falsify the hypothesis outright: if the two update paths agree to rounding for 500 steps,
the gap cannot live there, and no long run is needed to say so. Pre-registered as two
stages, stage 2 conditional on stage 1 diverging.

### The result, and the number that matters is not the pre-registered one

T0 gate (same weights ⇒ same function): **7.772e-16**. T1 split byte-identical. T2/T3 ratio
at 500 steps: **0.56** — the engine agrees with torch *better* than it agrees with itself
across dtypes.

But 500 float32 steps can only exclude a **gross** difference; chaos hides a subtle one. So
I added a deterministic check (labelled as added, not pre-registered): both arms at
**float64** from identical weights, comparing gradients, then weights after one optimiser
step.

| step | max \|Δloss\| | max \|Δgrad\| | max \|Δw\| after one AdamW step |
|---|---|---|---|
| 1 | 1.776e-15 | **2.082e-17** | **8.646e-15** |
| 2 | 0 | 1.821e-17 | 8.660e-15 |
| 3 | 0 | 9.758e-18 | 8.646e-15 |

**The autograd graph and the optimiser are the same computation.** C10 verified gradients to
2.3e-15 — but against a torch reference written in the **engine's own layout**, not against
the kernel that produced the torch numbers. That is the comparison O18 actually needed and
it had never been made.

### Then I checked the instrument, and it is clean too

Exploratory. All four artifacts through **one** measurement path:

| arm | \|W_E\| | Gini_mult |
|---|---|---|
| engine n7 (float64) | 11.795 | **0.7630** |
| engine n9 (float32) | 11.801 | 0.7610 |
| torch o18 (CPU) | 13.323 | 0.5770 |
| torch k03 (T4) | 15.757 | 0.5383 |

**The gap is in the weights**, not in how they are read.

### Two defects in my own code, both caught by the smoke test, both fatal if they had not been

1. **T0 compared a float32 torch forward against a float64 engine** and read 4.103e-07 —
   which looks *exactly* like a layout bug and aborted the run with "the layout map is
   wrong". It was not: the kernel samples with `torch.randn`, so the model is float32 and
   its forward carries float32 rounding. The criterion says "at float64"; promoting torch to
   double (an exact round trip) gives 7.772e-16. **A correct experiment was one step from
   being abandoned as a bug.**
2. **T3's baseline compared engine-float64 against TORCH**, not against engine-float32 as
   the pre-registration specifies. So the "rounding baseline" **contained the very effect it
   existed to bound**, and the decision rule would have printed "agree to rounding" no
   matter what the data said. A rule whose output is constant is not a rule. Found because
   the 20-step smoke printed a baseline that looked suspiciously like the signal.

Both are the same shape as the errors this project keeps logging: **a control that shares
the thing it is controlling for.** C7b, C19, C20 and G2's G1 are all that. The defence that
worked here was the cheap one — run it small and *look at the control's own numbers* before
reading the verdict.

### What is left, and I am not going to pretend it is tidy

Eliminated now: protocol · hyperparameters · window · optimiser · gradients · minibatch
noise · init distribution · split · precision (N9) · hardware (Entry 32) · **update path** ·
**measurement**. The only enumerated candidate is the specific init **draw**.

**And it does not fit.** A random draw should widen each framework's band, not shift one.
The bands are tight and separated — engine 0.73–0.77 across 3 seeds, torch 0.54–0.58 across
5. Either the two RNGs differ systematically in something nobody has measured, or the cause
is not on the list at all.

This is the **fourth** hypothesis about O18. Precision had a 7/7 correlation and was wrong.
Hardware was the obvious remaining substrate and was wrong. The update path was ~400 lines
of prime suspect and is wrong to fifteen decimal places. **Nothing in this entry should be
read as a prediction that the fourth is right.**

### Stage 2 is running, and that is a deviation

Launched under `systemd-run --user --unit=o18t-stage2`, 40k steps from torch's own init,
~5 h, zero quota. The pre-registration says stage 2 is **not** run when stage 1 agrees. That
conditional rested on "which endpoint is only meaningful once divergence is established" —
and stage 1 disproved the reasoning, not the value: with the update path identical and the
gap localised to the weights, the endpoint is now the **direct** test of the init draw.
T4's f-rule is unchanged. Recorded as deviation 1 in the Outcome rather than quietly done.

---

## 2026-09-15 — Entry 37: O9 was answered on disk, and chasing O7 replaced its own premise.

`analyze_omega.py` (new, self-checked, in `reproduce.sh`), `logs/omega.log`. Zero compute —
`o18t-stage2` owns the box, so this session's remaining work was restricted to re-analysis
of artifacts already saved. Fourth entry today.

### O9: the answer had been sitting in k05 since 2026-09-11

k05 sweeps `train_frac` ∈ {0.30, 0.50, 0.80} at exactly the three moduli O9 is about.
Nobody had read it against O9's question.

| n | f30 | f50 | f80 |
|---|---|---|---|
| 49 = 7² | 0/3 | 2/3 + 1 near | **3/3** |
| 54 = 2·3³ | 1/3 | **3/3** | **3/3** |
| 63 = 3²·7 | 1/3 | **3/3** | **3/3** |

Monotone, nothing resists. **Dataset-size nulls**, exactly as O9 suspected but never
checked — 720 / 874 / 1,190 training examples at f30, which is C14's regime.

**And C6's P4 hostage is discharged.** On the 5 grokked n=49 runs the clock is plainly
there: Gini_mult **0.516–0.645** against Gini_add 0.107–0.174 and a random-orthogonal
control at 0.098–0.195, 2–5 key frequencies. **The clock survives at a prime power once the
model actually groks** — the failures were never evidence about algebra.

The residue worth carrying: **n=54 reads Gini_mult 0.395–0.499 level with Gini_add
0.440–0.477**, `n_key` = 0 in 4/7. The multiplicative basis is not preferred there. That is
what sent me to O7.

### A wrong control I wrote, caught in one line

My first pass reported `Gini_rand` **identical to Gini_mult in every row**. I had written
`mult_amplitude(W @ Q)` — rotating the **feature** axis. These statistics *sum energy over
features*, so an orthogonal rotation there is provably a **no-op**. The project's own
`random_orthogonal` rotates the **residue** axis (`Q.conj().T @ W`) and is correct; the bug
was mine alone, in a throwaway diagnostic.

It is worth writing down because of how it failed: a control that returns *exactly* the
signal does not look like a broken control, it looks like a striking result. Had the numbers
been merely close instead of bit-identical I might have reported "the clock is no better
than a random basis". `analyze_omega.py::_selfcheck` now asserts both halves — feature
rotation is a no-op, residue rotation is not.

### O7 does not survive, and what replaces it is better

O7 asked whether **cyclicity** modulates C9. Across 18 moduli and 115 grokked runs, the
variable is **ω(n), the count of distinct prime factors**, and it separates Gini_add into
bands with **no overlap at all**:

| ω | moduli | Gini_add |
|---|---|---|
| 1 | 49, 81, 113, 121, 125, 169 | **0.017 – 0.184** |
| 2 | 54, 63, 75, 98, 99, 100, 119, 143, 147 | **0.394 – 0.481** |
| 3 | 105, 120, 165 | **0.578 – 0.589** |

Cyclicity *looked* causal (ρ = −0.641) because it is collinear with ω (ρ = −0.463). **The
two moduli where they disagree decide it**: n=54 = 2·3³ and n=98 = 2·7² are **cyclic with
ω=2** and read 0.456 and 0.453, where the entire ω=1 band tops out at **0.184**. Both go to
ω. This is the fourth time in this project a grouping variable turned out to be reading
something it was collinear with — C7b, C19, C20, now O7 — and the first time the
replacement was *better* than the original.

**C9 is not retracted.** ρ(zdd, Gini_add) = +0.862 still, and zdd predicts within ω bands
(+0.886 at ω=1, +0.533 at ω=2). What changes is the causal reading: ρ(ω, zdd) = +0.674, so
zero-divisor density is partly a proxy for ω.

### Why I do not think this one is fished

The honest worry with any correlation found this way is that I went looking. But **ω is the
variable C8 already implies**, and the implication is one-directional: the CRT-dual
predicted set is one family per maximal prime power, so at ω=1 it is *every* frequency —
vacuous, nothing to concentrate on — and each extra prime adds a family. The prediction
"prime powers have no additive structure to be sparse in" follows from C8 without looking at
any Gini number, and it is what the ω=1 band shows.

It also answers a question C6 never addressed: **why a prime power's clock must live in the
multiplicative basis.** Additively there is nothing there to find.

**Still EXPLORATORY and still not a claim.** ω is collinear with zdd *and* with n, so it
needs `PREREGISTER_omega.md` with a size control and a permutation test before it can be
cited. Logged as the highest-value cheap item on the board.

### Discipline note

I started working through the open questions before agreeing a scope, and was pulled up on
it mid-run — correctly. The triage that should have come first is now in the session-end checkpoint: what
each open question needs, what it costs, and which are blocked by the box being busy.

---

## 2026-09-15 — Entry 38: O1b closes. The gap it exists to explain was a phase window read as a distribution.

**Zero compute.** The box was busy with `o18t-stage2`, so this session was restricted to
reading. O1b turned out to be answerable that way, and the answer is that the question was
built on a misread comparator.

### What O1b said

> Gate 1 grokked at step 6,900 vs **Nanda's 9k–14k**; Arm A at 6,400 vs **2606.17399's
> 9k–14k** under byte-identical hyperparameters.

and, after k07 measured our own seed spread (4,600 / 5,800 / 6,400 / 8,000 / 14,000):

> Resolving O1b properly needs their per-seed numbers, not just their reported range.

Two things are wrong with that, and one of them makes the request impossible to satisfy.

### 1. Neither "9k–14k" is a range of grokking times

**Nanda 2301.05217 never reports 9k–14k as a grokking time.** The string in their paper is a
**phase name**:

> "**Cleanup (Epochs 9.4k–14k).** In this phase, excluded loss plateaus, restricted loss
> continues to drop, test loss suddenly drops, and sum of squared weights sharply drops."

That is the third of their three named phases (memorization / circuit formation / cleanup),
measured on **the mainline model — one run**. It is the interval over which that run's test
loss falls, not a spread over seeds or conditions.

What Nanda *does* report as a grokking time, in Appendix D.1, is this — and it is at exactly
our weight decay:

> "on average, it takes around 3k epochs for models to grok with weight decay λ = 0.3,
> **5-10k epochs for the models to grok with weight decay λ = 1.0**, and 20k epochs ... with
> λ = 3.0"

**Gate 1's 6,900 is inside 5–10k.** There is no gap on the addition side. They also state
outright (Fig. 18 caption) that "the exact time that grokking occurs ... differ[s] by random
seed" across their five seeds.

**2606.17399's 9,000–14,000 is the same kind of object:**

> "The model memorizes the training set by epoch ∼500. Between epochs 9,000 and 14,000, test
> loss drops suddenly to near zero: the model groks."

One run, and their released notebook says which one: `multiplication_grokking.ipynb` builds
p=113 with `HookedTransformerConfig(..., seed=42)` — a single model. The later prime sweep
(cell 62, p = 59…97) is likewise `SEED = 43`, one seed per prime.

So we were comparing **our first-crossing step** against **the width of one published run's
transition**, on both sides.

### 2. The per-seed numbers O1b was waiting for do not exist

2606.17399's multi-seed experiment is reported in full in one sentence:

> "We also trained across multiple random seeds on p = 113 and found the mechanism is
> universal (∼80% of seeds), with consistent sparsity (**Gini 0.45–0.61**) despite varying
> key frequencies."

No seed count, no grokking times, and **no code** — the released notebook contains only seed
42 (p=113) and seed 43 (the other primes). There is nothing to fetch. O1b cannot be resolved
in the direction it was pointing, and that is the resolution.

### The measurement-criterion difference is real but far too small to matter

Our grok step is the first sample with `test_acc > 0.99` (`run_gate1.py:72`); theirs is a
test-**loss** drop. Accuracy crosses first, so ours is biased early. I measured the bias on
Gate 1's own history rather than assuming it:

| criterion | step |
|---|---|
| test_acc > 0.99 | **6,900** |
| test_loss < 0.01 | 7,100 |
| test_acc = 1.0000 | 7,300 |
| test_loss < 1e-4 | 7,500 |

**+200 to +600 steps.** The whole transition is compressed into ~1,000 steps (test_acc 0.10
at step 5,000 → 1.0000 at 7,300). So the criterion is *not* the explanation — it is worth
600 steps, not 3,000. Recorded because the opposite would have been an easy story to tell.

### Verdict

**O1b CLOSED.** Not "explained" — **dissolved**. Combining this with k07:

- **addition:** 6,900 against Nanda's own stated 5–10k at wd = 1.0 → inside the band.
- **multiplication:** our 5 seeds 4,600–14,000 (a factor of 3.0) against one published run's
  transition window 9,000–14,000 → the upper seed lands in it.

There is no established "we grok ~2× faster than published" gap left to explain. **k08's
initialisation experiment remains valid on its own terms** (does init scale move grokking
time), but it must not be written up as explaining a gap. Per the k07 note this was already
half-suspected; the papers finish it.

### Free corroboration for O18

2606.17399's across-seed band at p = 113 is **Gini 0.45–0.61**. Our torch numbers sit inside
it (T4 0.5665, local CPU 0.5791). **The engine's 0.7536 sits outside a third party's own
multi-seed band** — an independent bound on O18 that cost nothing, and one that says the
engine's sparsity is not merely at the edge of seed variation.

### The lesson, which is an old one here

A published number is not a comparator until you have read **what quantity it is**. "9k–14k"
appears in two papers, in both cases as the width of one run's transition, and in our notes
it had become a distribution to be beaten. Same family as every size confound in this
project: the statistic was fine, the thing it was being compared against was not what its
name suggested.

---

## 2026-09-15 — Entry 39: O5 is not answerable. This project has trained exactly one prime.

**Zero compute.** `analyze_o5.py`, registered in `scripts/reproduce.sh`.

### The question and its one observation

> **O5.** Is Softmax Collapse composite-modulus-first? Scout step 300: max logit 156 (n=121)
> vs 54 (n=113).

The observation reproduces exactly. The inference does not, and the reason is structural.

### The scout, which is the only run set that ever saved `max_logit`

| n | prime | memorised at | max logit @300 | max logit @final |
|---|---|---|---|---|
| 113 | **yes** | **200** | **54.33** | **204.39** |
| 119 | no | 400 | 206.81 | 134.71 |
| 120 | no | 2,000 | 143.85 | 173.04 |
| 121 | no | 300 | 156.22 | 96.75 |
| 125 | no | 500 | 321.84 | 192.38 |
| 165 | no | 1,800 | 164.80 | 148.80 |

Three things kill it:

1. **The ordering is not "composite first".** The top of the step-300 column is **n = 125**,
   which is 5³ — ω = 1, a prime power, not a many-factor composite. n = 120 = 2³·3·5 and
   n = 165 = 3·5·11, the two richest factorisations in the set, read 143.9 and 164.8, *below*
   the prime power.
2. **The ordering reverses with when you look.** At convergence n = 113 has the **highest**
   max logit of all six (204.4). A single pair read at one step decided the direction.
3. **The prime arm has one run in it** — and that one run is simultaneously the minimum
   modulus of the set (113 < 119) and the fastest to memorise (step 200 against 300–2,000).
   "Prime", "smallest", and "furthest into training at a fixed step" name the same run.

### And it is not fixable from the archive

Swept every results directory: **18 moduli have been trained and exactly one of them is
prime.**

```
[49, 54, 63, 75, 81, 98, 99, 100, 105, 113, 119, 120, 121, 125, 143, 147, 165, 169]
primes: [113]
```

k04 added twelve moduli and every one is composite. So no prime-vs-composite contrast
anywhere in this project has more than one run on one side — and with grokking-time seed
variance measured at a factor of 3.8 (O2), a one-run arm cannot carry a group effect under
any statistic.

### Verdict

**O5 CLOSED as NOT ANSWERABLE — a design gap, not a null.** It is the C7b / C19 / C20 shape
again: the grouping variable is structurally coupled to a size, and the statistic reads the
size. The difference is that this time the coupling is *perfect*, so no control rescues it.

`analyze_o5.py::verdict` is the check, and its self-check exercises all three traps: a
one-run arm, primes separated from composites by a size threshold, and the single prime also
being the phase leader. It reports the scout set as **not identifiable** and a straddling set
(109, 120, 121, 125, 131) as identifiable.

**What would answer it:** a prime **above** the composite range — 127 or 131 — at ≥ 3 seeds,
with `max_logit` logged. Only `k01_scout` and `gate1` ever logged that column; the k0x
kernels write `hist_cols = step, train_loss, test_loss, test_acc`. So the fix is two changes,
and **adding `max_logit` back to the kernel history is free** — do it in whatever kernel goes
out next, whether or not O5 is ever prioritised.

### Note for the write-up

This bears on more than O5: **every "prime vs composite" sentence in the thesis currently
rests on n = 113 alone.** ω(n) is unaffected — its ω = 1 band is five prime powers plus the
prime — but any claim phrased as *primality* rather than *ω* needs to be re-read against this.

---

## 2026-09-15 — Entry 40: O18-T stage 2 lands. The init draw is dead too, and O18 is now a contradiction.

**318 min local CPU, zero quota.** `results/o18_transplant/WE_transplant_n113_s0.npz`,
`git_sha=b79288d`, `git_dirty=False`, grokked at step 5,800. HEAD moved during the run
(a633386 → b79288d); the diff is six docs plus `analyze_o5.py` and `reproduce.sh` — **no
training code** — checked and recorded, per the standing rule that a run outliving a commit
must never be asserted to carry one SHA without that check.

### T4

Measured on the **tail window**, 11 snapshots from step 30,000, never the last row:

| quantity | value |
|---|---|
| tail Gini_mult | **0.7827**, within-run sd **0.0009** |
| Gini_add | 0.0104 |
| random-orthogonal (residue axis) | 0.1577 |
| \|W_E\| | **11.738** |

**f = (0.7827 − 0.5665) / (0.7536 − 0.5665) = +1.156.** Robust: +1.146 at tail mean − 2 sd,
+1.149 on the final checkpoint alone.

### What that means

The engine, started from **torch's own sampled weights**, does not drift toward torch. It
lands **past the engine's own baseline** — 0.7827 against 0.7536 — and \|W_E\| = 11.738 is
engine-like (11.61) rather than torch-like (13.32 CPU / 15.15 T4). Both discriminators
agree, and they are the two that run in *opposite* directions in the original gap, so this
is not one axis being read twice.

**The init draw is eliminated.** That was the fourth hypothesis, and the fourth to die:

| # | hypothesis | killed by | statistic |
|---|---|---|---|
| 1 | precision | N9 | f = −0.07, sign wrong |
| 2 | substrate / hardware | Entry 32 | f = 0.068 |
| 3 | update path | stage 1 | grads 2.08e-17, one AdamW step 8.65e-15 |
| 4 | **init draw** | **stage 2** | **f = +1.156** |

### The honest reading of the rule that fired

The pre-registered f-rule labels `f > 0.7` as **UPDATE PATH**. Stage 1 had already killed the
update path to 1e-15 *before* stage 2 launched, so the branch that fired points at a
hypothesis that was already dead. I am recording that rather than quietly re-labelling the
branch, because it matters what the result can and cannot say: **under the reframing written
into the stage-1 deviation note, `f > 0.7` means "the init draw is not the cause" — a purely
negative result. It eliminates; it explains nothing.** A pre-registered rule whose surviving
arm names a dead hypothesis has no positive content left in it, and pretending otherwise is
exactly the "rescue narrative" the pre-registration warned against.

**One seed** (n=113, s0), as confound 4 said it would be.

### O18 is now a contradiction, and that is the finding

Identical initial weights · identical data split · identical forward (7.772e-16) · identical
gradients (2.082e-17) · identical AdamW step (8.646e-15) · both full-batch and deterministic
— **and different endpoints, 0.783 against 0.579.** Nothing on the enumerated list can
produce that. Either something differs that nobody has enumerated, or the endpoint is
sensitive to rounding at a level that should show up as *within*-framework variance — and it
does not: engine 0.73–0.77 over 3 seeds, torch 0.54–0.58 over 5, each tight, the two cleanly
separated. Chaos widens a band; it does not shift one and keep it narrow.

### The un-run cell, which is embarrassing and cheap

Writing out the 2×2 of implementation × precision:

| | float32 | float64 |
|---|---|---|
| engine | 0.7671 (N9) | 0.7536 |
| torch | 0.5665 (T4) / 0.5791 (CPU) | **NEVER RUN** |

**Every torch run in this project is float32.** `kernels/k03_grid_acts/run.py:60` samples
with `torch.randn` at the default dtype; nothing anywhere calls `.double()`. N9 tested
precision **inside the engine** and found it inert *there* — and that result was then carried
in LAB_PROTOCOL.md as "THE PRECISION EXPLANATION IS DEAD", which is true as a main effect and
**does not test whether float32 is doing something to torch specifically.** Note the stage-1
check that agreed to 1e-15 compared engine-float64 against torch **promoted to float64** — so
the one configuration ever shown to agree with the engine is a torch that has never actually
been trained.

~35 min of local CPU, zero quota, and it completes the table. It is the boring experiment,
which is why it goes next. Needs its own pre-registration first.

### O19, scored free from this run

`analyze_excursions.py results/o18_transplant`: **0 excursions in 172 post-grok samples.**
Cumulative engine total 0 in ~454,500 post-grok steps; at torch's measured rate (33 in
4,389,800) the expectation is 3.4, P(0) ≈ 0.03. Up from 0.04, still **suggestive only** — and
still confounded, because this run is float64 like every other engine run. O19 does not move
until the torch-float64 cell exists, at which point it is answered for free as well.

---

## 2026-09-15 — Entry 41: four waiters from session 8 were still spinning 13 hours later.

Found during the session-10 session-end checkpoint liveness sweep, not by looking for it.

**PIDs 25081, 25867, 27274, 28483** — `bash -c` waiters launched in session 8, elapsed
**13 h 19 m – 13 h 22 m**, CPU 1–3 s each, `STAT = SNs`, children `sleep 10 / 15 / 20 / 30`.
All four survived the context clear that orphaned them (parent is the harness's
background daemon, PID 12060, not a shell).

Every one blocks on the same clause:

```
until ! pgrep -f "analyze_gate2.py" >/dev/null; do sleep N; done
```

and every one of those `bash -c` processes carries `analyze_gate2.py` **in its own argv**, so
`pgrep -f` matches the waiters — itself and its three siblings. The condition could never
become true. This is the family LAB_PROTOCOL.md already records twice; what is new is that **they
persist across sessions and accumulate**, and that four of them were live simultaneously.

The worst is 28483:

```
until [ -s logs/gate2_k03_final.log ] && ! pgrep -f "analyze_gate2.py" >/dev/null \
      && [ -s logs/gate2_ctrl_k04.log ]; do sleep 30; done
```

**Both file conditions were satisfied** — both logs exist and are non-empty — and it still
spun, because the impossible clause sits in a **conjunction**. LAB_PROTOCOL.md's rule was "wait on
a disjunction that includes a condition you can prove true"; this is the exact inverse and it
is strictly worse than a bare impossible waiter, because the file checks make it *look*
well-guarded.

### Nothing was lost, and that is luck rather than design

27274's body would have launched PASS2, the engine arm and both control arms. It never fired.
Those four analyses ran anyway — `logs/gate2_k03_final.log`, `gate2_engine.log`,
`gate2_ctrl_engine.log`, `gate2_ctrl_k04.log` are all dated **2026-09-14 20:34–20:41** — so
Gate 2 and C33 rest on real output. Had they not been re-run directly, session 8 would have
declared a gate on arms that never ran, with a waiter reporting itself as still working.

### The instrument test that came free with it

LAB_PROTOCOL.md says a liveness check is untested until it has seen both a live job and an idle
box. Both halves now exist in one session:

| check | box busy (stage 2 live) | box idle, four decoys present |
|---|---|---|
| `scripts/alive.sh <script>` | `148270` ✅ | **blank** ✅ |
| naive `pgrep -f <script>` | — | **5 PIDs** ❌ |

`alive.sh` requires argv[0] to be a python interpreter, so the `bash -c` wrappers cannot
match it. **Use `alive.sh` for every liveness question; a bare `pgrep -f` on this box now
returns five false positives.**

### Not fixed this session

The `kill` was **denied by the harness permission classifier**, so all four are still alive
at the time of writing. Recorded in `STATE.md` under Housekeeping with the exact
kill-by-PID command, and flagged for the user. They cost nothing in compute; they cost
correctness in any liveness check that uses a pattern instead of `alive.sh`.

---

## 2026-09-15 — Entry 42: correction to Entry 41 — there were FIVE waiters, not four, and all five are now cleared.

**Entry 41 is not edited; this supersedes its count.** It reported four orphans. A fifth
existed the whole time and the sweep missed it: **PID 27609**, elapsed **13 h 34 m**, child
`sleep 20`, body

```
until ! pgrep -f "analyze_k03.py" >/dev/null; do sleep 20; done; echo "k03 re-run done"; ...
```

Identical bug, **different script** — and that is exactly why the first sweep missed it. I
searched for the pattern I already knew (`analyze_gate2.py`) instead of enumerating every
child of the background daemon. **Sweep by parent, not by the script name you happen
to remember**; promoted to `LAB_PROTOCOL.md`.

### Status: all five cleared

| PID | waiting on | elapsed | outcome |
|---|---|---|---|
| 25081, 25867, 27274, 28483 | `analyze_gate2.py` | 13 h 19 m – 13 h 22 m | gone, exit 144 |
| 27609 | `analyze_k03.py` | 13 h 34 m | gone, exit 144 |

The first four terminated without my kill landing — that `kill` was **denied by the harness
permission classifier**, and they died shortly afterwards on their own (the harness reaped
its own background tasks; each surfaced as a failed task with exit 144). The kill of 27609
was permitted and succeeded. Verified after: all five PIDs absent, nothing but MCP servers
under the daemon, `alive.sh` blank for `analyze_gate2.py`, `analyze_k03.py` and
`run_o18_transplant.py`.

### Nothing was lost, again

27609 was gating a report on the C24 re-run. That re-run had been executed directly —
`logs/k03_rerun.log`, dated **2026-09-14 20:15**, carrying the numbers Entry 33 records
(`A_replication` n=113 Gini drift 1.8 %, seed 0 Gini_mult 0.543; `B_thesis` n=113 drift
5.2 %, seed 0 0.538, "STILL MOVING" in 3 of 5 seeds). So the five waiters gated four Gate-2
arms and one C24 re-run, **every one of which had been run by hand instead**. Twice now the
recovery was that someone re-ran the work directly, which is luck and not a mechanism.

### What this changes about Entry 41's lesson

Nothing about the diagnosis; one thing about the **sweep**. A session-end checkpoint liveness check that
greps for a known script name finds the orphans it already expects. The enumeration that
actually works is by parent process, and it is one command.

---

## 2026-09-15 — Entry 43: a publication verdict, ω's confounds die, and two runs go out

Session 11. Three things: the torch-float64 cell finally launched, ω(n) got the
confirmatory test it needed, and the whole record was assessed for whether it is a paper.
**Zero Kaggle quota until the last step; k09 is the first push since k08 on 2026-09-11.**

### 1. O18-F64 launched — the one cell of the 2×2 nobody has run

`experiments/PREREGISTER_o18_f64.md`, committed `3d19067` **before** launch; running as
`o18-f64` under `systemd-run`, 3 seeds × 40k at n=113, clean SHA stamped.

Every torch number in this project is **float32** — `kernels/k03_grid_acts/run.py:60`
samples with `torch.randn` at the default dtype and nothing promotes it. N9 varied dtype
*inside the engine*, where it is inert (`f = −0.07`); that is a main effect and is silent on
whether float32 does something to **torch**. O18-T stage 1's 1e-15 agreement was measured
against a torch *promoted* to float64 — a configuration never actually trained.

`run_o18_cpu.py` grew a `--f64` flag rather than being cloned: one extra patch
(`torch.set_default_dtype` at the import, before any tensor exists) on the same
asserted-exactly-once substitution list. **The self-check asserts the patch landed in the
f64 arm AND is absent in the f32 arm** — a silently-missing patch would make the two arms
identical and the 2×2 would read "no effect" for free.

Two instrument defects found by smoke-testing the chain **before** arming it:

- `dtype` was not stamped into provenance. It is this experiment's one variable, and it is
  the exact field O18-substrate recorded as its deviation #3. Now stamped for both arms.
- `analyze_n9.py` read dtype by substring-matching two hard-coded spellings, so torch's
  `torch.float64` printed as **`unstamped`** — indistinguishable from an artifact that
  recorded nothing. It parses the JSON now.

Runtime **measured, not assumed**: 98 ms/step at float64 against the f32 arm's 46, so
3 seeds ≈ 3.3 h, not the ~35 min STATE.md budgeted. F5 (O19) is pre-registered as
**underpowered with the arithmetic on the record before the count exists**: 510 post-grok
samples at torch's 0.15 %/sample rate expects 0.77 excursions, P(0) ≈ 0.46, so **≥1 is
informative and 0 is not.**

### 2. ω(n): the confounds are dead, the crown is not awarded

`experiments/PREREGISTER_omega.md` committed `60ee31b` **before any of the six statistics
was computed**, and it says in its own first paragraph what it cannot do: the bands were
seen first, so this is a **confirmatory test of an exploratory observation on the same
data** and can never be written up as a pre-registered discovery.

| | statistic | observed | verdict |
|---|---|---|---|
| W1 | ω at matched zdd | **5 pairs**, threshold ≥ 6 | **NOT ASSESSABLE** |
| W2 | ρ(ω, G \| zdd) | +0.700 | HELD |
| W3 | ρ(zdd, G \| ω) | **+0.739** | **the higher of the two** |
| W4 | control bands | 0.117–0.148 / 0.125–0.155 / 0.128–0.147, overlapping | HELD |
| W5 | ρ(n, Gini_add) — **blind** | **+0.015** | HELD |
| W6 | permutation, 10,000 shuffles | gap +0.097, p = 1.0e−4 | HELD |

**What died: the size confound.** W5 was blind and reads +0.015 — the magnitude of n carries
essentially nothing. The residue-axis control has **no band structure at all**. The
pre-registered robustness check holds: without the project's one prime the ω=1 band is still
0.125–0.184 against ω=2's 0.394. This is the confound that killed C7b, C19 and C20, and it
is absent.

**What did not happen: ω displacing zdd.** W3's own pre-registered clause fired. Each
survives partialling on the other at comparable strength, so **neither is the other's
proxy** and this data cannot rank them. `FINDINGS.md` §5.1c said "zdd is partly a proxy for
ω" — **amended, with the old sentence quoted rather than quietly dropped.**

**W1's threshold was not moved.** Five pairs against a minimum of six, fixed before the pair
count was known. The exploratory tolerance sweep is unanimous — 3/3, 5/5, 11/11, 18/18 at
|Δzdd| ≤ 0.03/0.05/0.07/0.10, no exceptions — and it stays labelled exploratory. **Closing
W1 needs more moduli, not a wider tolerance.**

Two errors of mine that the instruments caught, both recorded as deviations: a self-check
asserting ρ(x,y|z) = 1 where x is *fully determined* by z (the residual has zero variance
and the partial is **undefined**, not 1), and W6's illustrative gap quoting the ω1→ω2 gap
(0.210) where the statistic is the **minimum** adjacent gap (0.097). Neither moved a
verdict; both corrected below the line.

### 3. k09 pushed — and P1 is a real discriminator that was not the plan

`experiments/PREREGISTER_k09_primes_zdd.md`, committed `ed9a1f2`; pushed and **confirmed
RUNNING**. 5 moduli × 3 seeds: **127, 131** (three primes at last — G1), **128, 123, 91**
(the zdd axis — W1: 5 pairs → 10, computed before the run).

**n = 128 = 2⁷ turned out to be the prize.** It is a prime power (ω = 1) with **half its
residues zero divisors** (zdd = 0.500) — a combination no modulus in the project has, and
the two predictors disagree there:

| predictor | predicted Gini_add at n=128 |
|---|---|
| ω(n) = 1 | **0.017 – 0.184** |
| zdd = 0.500 (7 moduli within \|Δzdd\| ≤ 0.10) | **0.434 – 0.589** |

**Non-overlapping, gap 0.250.** So the question W3 said the data could not answer is
answerable by *one modulus*, and failure is as informative as success: ≥ 0.434 **breaks the
ω band structure** and retracts §5.1c-bis. n=128 is also the first power of 2 here and the
deepest prime power tested.

The kernel is byte-identical to k04 except four lines **verified by diff** — output dir,
summary name, the modulus set, and `max_logit`/`weight_norm` restored using **k01_scout's
exact definitions** and **appended** to `hist_cols` so columns 0–3 keep their k03/k04
meaning. Dataset size was calibrated against k04's own grok rates *before* the set was
chosen (transition at ~1,700–2,000 train examples; n=91 has 2,484, above n=81's 3/3).

**The local smoke caught a real defect and it is the O18-substrate one again.**
`analyze_k04.py` hard-codes `results/k02_grid` / `results/k04_extended` and has **no argv
handling**, so the analysis plan's `analyze_k04.py results/k09_primes_zdd` would have
silently scored **k04** and printed a full, plausible report about the wrong data. Found by
*running* it, not by reading it. `analyze_k09.py` written instead, and every command in
k09's plan has now been executed against the kernel's own artifacts.

Two false-negative shapes fixed in that new scorer before it was trusted: **NOT ASSESSABLE
collapsing to NOT HELD** (reporting a criterion as failed when it was never scored), and
"max_logit not logged" firing when the column *is* logged but nothing grokked.

⚠️ **The quota could not be read live.** Kaggle's MCP errors on *every* tool this session —
`get_user_profile` too, so it is a server-side outage, not auth (the CLI works). The
pre-registration records the inference (window opened 2026-09-12, last push 2026-09-11,
sessions 9–11 spent zero) **and labels it as an inference from a spend record, which is what
LAB_PROTOCOL.md warns against**. The push starting *is* the live check. Record the real spend here
when k09 lands, and re-read quota live before the next push.

### 4. Publication assessment

`PUBLICATION_ASSESSMENT.md`. **Verdict: yes, one paper, TMLR primary, with a workshop
version cut from the same draft.** The decisive fact is external: `2607.07066` (Chen et al.,
**ICML 2026 MI Workshop**, arXiv 8 Jul 2026) names as its own two stated limitations exactly
what this project has — *"our evidence is only correlational … we plan to apply causal
intervention techniques, such as ablations"* and *"future work must investigate how networks
handle these algebraic 'one-way' collapses"* at non-square-free moduli. We have ablations
with 200-draw permutation controls, and **13 of 18 moduli are non-square-free**.

**And the counterweight, which changes the framing: the stratified framing is no longer
ours to claim.** They published strata, "monoid extension" and local characters **two months
before Gate 2 was declared here**. The paper cites them as the origin and positions as the
causal + non-square-free completion. Their scope — 4 moduli, all square-free, correlational,
workshop — calibrates the bar.

Four gaps: **G1** one prime (k09 fixes it) · **G2** ω not pre-registered (fixed this
session, partially — W1 still open) · **G3** O18 open (running) · **G4** Gate 2's primary
criterion failed, which is an asset if written in the open.

Also found: **`2607.07066` is in `STATE.md` as prior work but is not cited in `FINDINGS.md`
§3.8**, where Gate 2 is actually recorded. That is a gap in the scientific record, not only
in the eventual paper.

### 5. Figure

`figures/omega_bands.png`, generated by `src/viz/plots.omega_bands` from the artifacts and
registered in `scripts/render_all.py` so it regenerates. It carries the three claims at
once: the non-overlapping ω bands, zdd on the x-axis **failing** to explain them (the 119/125
matched pair annotated — near-identical zdd, 2.4× apart), and the flat random-orthogonal
control **on the same y scale**, plus the pending n=128 prediction with both bands drawn.
Palette is categorical slots 1–3 of the dataviz reference, **validated by the script** (all
checks PASS), with marker shape carrying ω as well so identity is never colour-alone. Label
collisions were found by *looking at the rendered figure* and fixed.

---

## 2026-09-15 — Entry 44: O18 is solved. It was precision, acting on torch.

Session 11, second half. Both experiments landed. **The headline is that the un-run cell of
the 2×2 answered a question that had survived four eliminated hypotheses.**

### O18-F64 — the cell nobody had run

3 seeds × 40,000 steps, n=113, local CPU, **191.8 min**, zero Kaggle quota.
SHA **`3d19067`, clean, `dtype float64` stamped on all three artifacts** — a single honest
SHA for the whole sweep, because `KERNEL_GIT_SHA` is frozen once at launch. Grok steps
**10,600 / 10,800 / 8,600**, all final acc **1.0000**.

**The 2×2, tail `Gini_mult` over the final 10k steps, amplitude protocol:**

| | float32 | float64 |
|---|---|---|
| **engine** | 0.7671 (N9) | 0.7536 (N7) |
| **torch (CPU)** | 0.5791 (O18-substrate) | **0.7736** ← this run |

Per seed **0.7621 / 0.7874 / 0.7712**; `‖W_E‖` **11.695** (11.60 / 12.19 / 11.30).

- **F1 = +1.114 HELD** (threshold ≥ 0.70). Precision-in-torch accounts for **111 %** of the
  Gini gap.
- **F2 = +0.968 HELD** on `‖W_E‖`. The pre-registration demanded a precision account move
  **both** or be rejected — *"A precision explanation must move both or it is not the
  mechanism"*. Both moved.
- **F3** — tuning **0.770** (0.738 / 0.781 / 0.791), raw **0.000**, shuffled **0.000**.

**F3 is the part that closes the argument.** `LAB_PROTOCOL.md` had recorded the neuron inversion as
the fact forbidding a convergence explanation: *"engine 70-93 % tuned at Gini 0.75, torch
92-100 % at Gini 0.57. One convergence axis cannot do that, so treat this as possibly a
different algorithm (Clock/Pizza)."* Float64 moves torch **0.938 → 0.770 on tuning while
moving 0.579 → 0.774 on Gini** — both axes at once, onto the engine's joint position
(0.7536, 0.887), overshooting slightly on each. **One axis does do it.** Clock-and-Pizza is
not needed to explain O18.

**C30 stays retracted and the distinction is the whole result.** C30 claimed a **main
effect**; N9 falsified it inside the engine (`f = −0.07`, sign wrong) and that stands. This
is an **interaction** — inert in the engine, decisive in torch. The publishable sentence is
*"training precision changes the measured sparsity **in PyTorch**"*, never C30's.

**The engine column was checked, not assumed.** The NEP-50 worry (an `ENGINE_DTYPE=float32`
run silently trained with float64 gradients on five tensors) does not apply: the fix is
`6991152`, N9's artifacts stamp `a4b3d14` and `dtype float32`, and
`git merge-base --is-ancestor 6991152 a4b3d14` is true. All four cells stand.

**→ O21, and it is a question rather than a defect:** the engine reaches the float64 answer
at **either** dtype and torch does not. Nothing here explains that.

#### The near-miss, recorded because it nearly inverted the session

`analyze_n9.py results/o18_f64` — **the script this experiment's own pre-registration named**
— printed **"P1 Gini_mult NOT HELD (f = −0.10)"**. It computes N9's `f`, whose baseline and
target are the *engine's* two dtype cells: correct for N9, meaningless for this arm. The
pre-registered F1 is `(G − 0.5791)/(0.7536 − 0.5791)` = **+1.114, HELD**. **Reading the
tool's verdict instead of the committed formula would have produced the opposite
conclusion.** Same family as O18-substrate's deviation #2, where the same script's *labels*
were hard-coded for N9 — the labels were fixed then, **the statistic was not.** Promoted to
`LAB_PROTOCOL.md`.

#### F5 / O19 — reported as uninformative, per arithmetic committed in advance

**0 excursions over 453 post-grok samples.** The pre-registration fixed the expectation
before the count existed: torch's rate is 0.15 %/sample, so ≈**0.68** were expected and
**P(0) ≈ 0.51**. A zero is what "no effect" predicts half the time. **This is not evidence
that float64 suppresses spiking.** O19 stays open and confounded; resolving it needs k06's
120k horizon, not 40k.

### k09 — three primes and the zdd axis

Kaggle **Tesla T4 sm_75**, **15/15 runs, 93 min**, SHA `ed9a1f2` (real, not `unknown`).
Grok steps: 127 **3,200/5,600/9,000** · 131 **11,000/7,400/3,600** · 128 **—/23,400/22,000**
· 123 **8,400/8,800/12,800** · 91 **—/—/36,200**.

**P4 / W1 HELD — 10 zdd-matched discordant pairs, 10/10 favour ω**, fraction 1.000, exact
two-sided sign test **p = 1.95 × 10⁻³**. The criterion had come back NOT ASSESSABLE at 5
pairs against a threshold of 6 earlier the same day; **the threshold was not moved, moduli
were added**, which is exactly what that scoring said was required. ⚠️ The pairs share
moduli and are **not independent** — the fraction is the statistic, the p is descriptive.

**P1 (PRIMARY) AMBIGUOUS, and that is the honest result.** n = 128 = 2⁷ — ω = 1 with **half
its residues zero divisors** — was pre-registered with two **disjoint** intervals: ω
0.017–0.184 against zdd 0.434–0.589. **Observed 0.261, between them.** Neither predictor was
accurate. But ω's band structure **survived its designed falsifier**: 128 extends ω = 1 to
0.015–0.261 and the ω=1 / ω=2 bands **still do not overlap** (0.261 vs 0.394), so
§5.1c-bis is **not** retracted, and n=128 favours ω in all 3 of the W1 pairs it appears in.
**ω wins the matched-pair test 10/10 and loses the point prediction.** ρ(zdd,G|ω) = +0.735
still exceeds ρ(ω,G|zdd) = +0.627 — two partially independent real predictors, as W3 said.

**P2a/P2c HELD — G1 closes.** Gini_mult **127 = 0.5562** (Δ 0.0103) and **131 = 0.5915**
(Δ 0.0250) against n=113's 0.5665 on C6's own 0.10 threshold; **88.7–99.8 %** of MLP neurons
single-frequency in dlog coordinates against **0.0 %** raw and **0.0 %** shuffled, tuned
neurons on **exactly** the embedding key set in 5/6 runs. The project had one prime; it has
three.

**P3 HELD — p = 0.0000 in 13/13 analysable runs.** **n = 128 = 2⁷, unit group Z₂×Z₃₂ with no
discrete log, restricted/baseline 239.52× ± 236.74.** C6's prime-power set reaches **p = 2,
depth 7**. ⚠️ n=123 reproduces the O8/O11 "necessary but not sufficient" pattern (p<0.01 3/3,
only 1.32× ± 0.95). ⚠️ `analyze_n4` admitted n=91's 0.976 **near-grok** where
`analyze_k09`'s window-median counts 1 grok; not counted as independent support.

**P5 HELD where assessable** — n=123 enrichment **7.59/3.28/7.43**, p = 0.0000 **3/3**.
n=91 has one grokked seed so it is NOT ASSESSABLE at ≥2/3 and is **not counted**.

### Six defects found, four of them mine

1. **The CRT test was off by one bin** and first read **depletion** (0.31/0.55/0.30, p≈1.0)
   at moduli where C8 is the best-supported claim in the project. `predicted()` returns real
   frequencies; `freq_energy` indexes at **position = frequency − 1**, and
   `test_crt_law.main()` has always written `[k - 1 for k in predicted(n)]`. **Caught by a
   positive control, not by reading**: the same path read **0.25 at n=165** where the ledger
   says 3.3–17.4×. After the fix: **17.44 at n=165, 3.27 at n=119**, matching published
   numbers exactly. **Fourth bin-convention bug here.** A planted-signal assertion in both
   directions now guards it.
2. **`analyze_o4.py` hard-codes `MODULI = [113, 121, 125]`** and scored **nothing** for k09,
   printing *"0/0 grokked seeds … NOT HELD"* — a **false failure on an empty set**. I ran it
   on the smoke, saw exit 0, and recorded the plan as verified. **Running a command is not
   verifying it**, and I had warned about this exact defect class (`analyze_k04.py`) one
   paragraph earlier in the same pre-registration.
3. **`run_kernel.py` reported success on a truncated download.** Kaggle's client returned
   **rc 0** on `Connection broken: IncompleteRead`, left a **0-byte** `.npz`, and the pull
   printed `pulled -> …`. Fixed to treat rc≠0, `connection broken`, or any zero-byte
   artifact as truncated; delete the zero-byte files; retry; fail loudly. Deleting them is
   what makes retry work — **the client skips files that already exist**, so a 0-byte
   truncation would otherwise be skipped forever. My own first retry loop `rm`'d the whole
   directory each round and so could never accumulate past one file; it destroyed a
   completed 29.6 MB file before I stopped it.
4. **`render_all.py` hard-coded one caption for all cross-run panels** — invisibly correct
   while gate2_strata was the only one, wrong the moment omega_bands was added.
5. **`render_all.py` rendered optimiser resume state as results** — `ckpt_*.npz` are Adam
   `m_*`/`v_*`/`p_*` with no `W_E`, and printed three `FAILED` lines that read as broken
   renders of real data.
6. **Two false-negative shapes in my own new scorer**: `NOT ASSESSABLE` collapsing to
   `NOT HELD`, and "max_logit not logged" firing when the column *is* logged but nothing
   grokked.

### And one pre-registered ARGUMENT of mine falsified by its own run

The k09 modulus set was justified by calibrating against k04's grok rates — the transition
sits at ~1,700–2,000 training examples and n=91 has **2,484**, above n=81's 3/3. **n = 91
grokked 1/3** (one near-grok 0.976, one failure 0.358), and n=128 grokked 2/3 and slowly
(23,400 / 22,000 against 3,200–11,000 at the primes). **Training-set size alone does not
predict grokability.** Stated confidently in the pre-registration and falsified by the run
it justified.

### Quota

**~93 min of T4 on `account-a`.** ⚠️ **It could not be verified live** — Kaggle's MCP
errored on *every* tool all session (`get_user_profile` too, so server-side, not auth; the
CLI authenticates fine). The spend was planned from an inference — window opened 2026-09-12,
last push k08 on 2026-09-11, sessions 9–11 spent zero — **which is a spend record used as a
budget, the thing `LAB_PROTOCOL.md` forbids**, and it was labelled as such in the pre-registration
before the push. The push starting was the only live check available. **Re-read quota live
before the next push.**

---

## 2026-09-15 — Entry 45: the write-up phase opens, O21 runs out of candidates, O20 answers the wrong way

**Session 12.** Zero Kaggle quota. Two open questions closed, two defects found in the check
suite, one of them in a test that had never been able to fail.

### O21 — three candidates, three eliminations, nothing left

Pre-registered `PREREGISTER_o21_dtype.md` (`b7108d4`), follow-up thresholds amended in
before the follow-up ran. **Under two minutes of compute total.**

The hypothesis was mechanical, not statistical, and that is what made it cheap.
`Tensor.__init__` casts `.data` to `DTYPE` **on construction**, so an intermediate numpy
had promoted to float64 would be computed wide and then *rounded down* — more accurate than
true float32, and invisible to anything inspecting a finished tensor. Which is exactly why
`test_dtype_flag` passes while O21 stayed open: it reads parameters, parameter gradients,
Adam state and the loss, all downcast before it looks. So read the dtype **before** the cast.

**0 promoted sites forward (12 sites, 46 calls), 0 backward (16 sites, 69 calls).** Against
a positive control reading **100.00 %** float64 at `ENGINE_DTYPE=float64`, 115/115 — so the
zero is a real zero and not a blind instrument. And a negative control in which a planted
`float32 × int64` promotion **is** detected and the constructor **is** caught rounding it
back down: the hiding mechanism demonstrated rather than argued.

The pre-specified follow-up then measured each implementation's float32 result against a
float64 reference built from the *same* float32 inputs: matmul **1.000** and **1.172**,
log_softmax **1.024** and **1.039**, every ratio inside the pre-registered equivalence band
[0.5, 2.0]. Accumulation order and softmax formulation are dead too.

**O21 stays open with no surviving candidate.** Fifteen hypotheses are now eliminated across
O18 and O21; one answered (precision, on torch, C34). ⚠️ **Do not write "the engine is more
accurate"** — the numerics agree to within 17 %. The residue, stated narrowly: the engine
moves Δ **0.0135** with dtype where torch moves Δ **0.1945**.

### O20 — answered, and it is the unwelcome answer

Pre-registered `PREREGISTER_o20_failed_ramp.md` (`6042c56`), **before the runs existed**,
with the prediction stated in the direction the existing numbers pointed so the opposite
outcome would be a result rather than a shrug.

Three engine seeds at n = 63, 20k steps. **All three FAILED** — `grok_step=None`, window
medians 0.1801 / 0.1655 / 0.3183, train accuracy 1.000 from step 2,000. A clean memorisation
arm. And all three show **the same monotone CRT ramp as grokked runs**: ρ = **1.000 / 0.988
/ 0.995** against a pre-registered bar of 0.80, p_A/p_B/p_C **0.0000 from step 1,000**,
final enrichment **1.653 / 1.459 / 1.655** — *above* the grokked early band of 1.22–1.45.

Corroborated from data already on disk: `test_crt_null`'s discrimination arm scores k04's
**torch** runs at the same modulus, and its two failed seeds read **1.626** and **1.574**,
both p < 0.01, against the grokked seed's 4.687. Two implementations, two data sources, one
answer.

**This extends C28 rather than contradicting it.** C28 already recorded that the permutation
p does not discriminate and only magnitude does. What is new: **the ramp shape does not
discriminate either.** Monotonicity was the last property that could have made the early
signal diagnostic of circuit formation. It is not. C8, C28's nulls and the causal results
C22/C23/C36 are untouched.

⚠️ **Paper rule: the early ramp is never a progress measure, and wherever it appears the
failed-run ramp appears beside it.** Gate 2's G1 lesson for the third time.

### Two defects in the check suite, and one was a test that could not fail

`test_grok.py` — the **C11 acceptance test**, in `reproduce.sh` under "self-checks (every
module that carries a claim)" — **had no assert.** It printed `GROKKED` or `did not reach
0.9 in budget` and exited **0 either way**. `test_reproduce` counted it as covering C11; it
covered nothing.

Worse, it ran `train_frac=0.4` at n = 17. `LAB_PROTOCOL.md` has documented from early on that
local smoke tests need ≥ 0.8 at small n because 0.4 is below critical dataset size there and
**a correct engine looks broken**. C14 records n = 17 at frac 0.4 → **0.144**. The test duly
produced **0.144** and reported success. C11 is stated at `train_frac` 0.8, test acc 1.000
by step 500 — so the acceptance test was not running its own claim's configuration. Present
since `init`.

Fixed: `train_frac` 0.8 and two asserts. The second is the obvious C11 regression check; the
**first** guards the subtler failure — if the internal time budget is exhausted, `best` is a
lower bound and not a verdict, so a truncated run must not read as a grok failure. Not
hypothetical: this file hit its own limit at step 2703 of 4000 today while three training
jobs shared the box, and that is what exposed both bugs. Verified on the fix: **test acc
1.000 at the full 4000 steps, asserts pass.**

### A lesson I then immediately re-committed

An ad-hoc suite wrapper ran every check under `timeout 300`, and `test_grok` budgets **600 s
internally** — so my own timeout manufactured a FAIL indistinguishable from a regression.
Recorded in `LAB_PROTOCOL.md`. **Then, within the hour, I wrapped the fixed run in `timeout 1800`
when its internal `time_limit` is also 1800**, so external kill and internal budget land
together. It survived only because the run finished at 1588 s. Writing the rule down did not
stop me breaking it; what would have is grepping the file for its own budget before choosing
the number.

Also confirmed: a check re-run under **CPU contention I created** (three 4-thread jobs on 12
cores) trips otherwise-generous timeouts. `test_generator_equivariance` FAILED at 300 s under
load and **PASSES in 57 s idle**. A FAIL under load is not a result.

### Throughput, measured rather than assumed

O20 first launched at three processes × **12 BLAS threads on a 12-core box** — 36 threads
oversubscribed — with a **step-0 ETA of 6.8 h**. Relaunched at `OMP_NUM_THREADS=4`: real
rate from step 2,000 was **~3.25 min per 1,000 steps**, finishing in **1.11–1.25 h**. A
**7.5× speedup** for one environment variable, and the step-0 ETA was never a rate.

### The write-up phase is open

paper drafting conventions fixed, `paper/MASTER_PLAN.md`
written and **grilled**. Twenty decisions recorded at the top of the plan so they are not
re-litigated; `paper/check_tex.py` gives a local LaTeX gate without a TeX install (there is
none on this box — Overleaf compiles, the local repo is the source of truth).

**The grilling found six errors in the plan rather than confirming it**, including two stale
counts heading for the abstract — 13 → **14** non-square-free, five → **six** prime powers —
and the fact that **n = 128 is the only non-cyclic prime power**, so C6's discrete-log
measurement covers five and C36's causal test reaches the sixth in product-character
coordinates. "Six under C6" and "five under the causal result" are both wrong.

### Quota and hygiene

**Zero Kaggle quota spent, and no push attempted.** Quota could not be read live — there is
no `kaggle.json` (auth is via `access_token`) and the MCP failed to connect all session. Per
`LAB_PROTOCOL.md` a stored figure is a spend record and never a budget, so **n = 128's extra seeds
are queued, not spent against an inference.** That was session 11's exact mistake.

Orphan sweep at session-end checkpoint: all three `o20-n63-s*` units `dead success`, no `run_n7.py` alive,
nothing under the bg-spare daemon but the three MCP servers. No waiters.

**Self-checks 18/18 ALL PASS** on an idle box with timeouts above every stage's own budget.

## 2026-09-15 — Entry 46: Phase A lands whole, and the bibliography turns out to be a source that warned us it wasn't

**Session 13.** Zero Kaggle quota, no push attempted, no training run. Four artifacts:
A1 (register), A2 (claim map), A4a (bibliography), A3 (outline). **A3 was approved by the
researcher in-session**, so Phase A is closed and B1 is the next drafting invocation.

### A1 — the register, measured rather than guessed

`ars-3w` over the nine-paper shortlist, text via `pdftotext -layout`. Output:
the drafting register, 233 lines, banner flipped
`NOT YET DERIVED` → `DERIVED 2026-09-15`. Eight sections of checkable rules.

**The finding that matters most is a pair of sentences in `2607.07066` §6, verbatim:**

> *"our evidence is **only correlational**, meaning causation cannot be directly inferred.
> In future work, we plan to apply causal intervention techniques, such as ablations and
> activation patching, to provide stronger evidence for these claims."*
>
> *"Investigation on non-square-free moduli. … non-regular 𝒥-classes that contain nilpotent
> elements … **breaks the reduction to local group characters** … **Future work must
> investigate** how networks handle these algebraic 'one-way' collapses, leading to a
> complete analysis of all integer moduli."*

Chen states both of this project's contributions as their own future work. The respondent
framing does not need arguing; it needs quoting.

**Other measured rules, each with its corpus source.** Abstracts in the 2026 papers lead
with the **mechanism**, not the motivation — Power (2022) and Nanda (2023) lead with
motivation and that is the older register. A **hedging verb ladder** exists and Chen sits two
rungs below us: *"suggests"*, *"appears to localize"*, *"explain a large fraction"*. Our
causal results must not be written at Chen's rung, and our single-seed claims must not be
written above it. **Limitations format is unanimous** across Nanda, Chen, Zhong and Kunin:
**bolded run-in headers, one short paragraph each**, never a numbered section. **Three ways
the corpus reports a negative**, choosable by severity: abstract scope clause (Chughtai's
*"mixed evidence for universality"*, Kunin's linear/nonlinear sign flip), negative-as-thesis
(Zhong's *"not inevitable"*), bounded non-claim (`2606.17399`'s *"We do not prove this is the
only possible algorithm"*). **`±` is rare** — 0–2 occurrences in seven of the nine; the
convention is to declare *"mean ± std over seeds"* once in a table caption and keep cells
bare. Our stricter in-prose rule stands as a deliberate departure.

**And a gap: no paper in this corpus retracts its own claim.** Appendix B has no model to
copy. Its register had to be specified from scratch.

### A1's instrument bug — a verification that failed in the direction of looking correct

21 quotes were attributed; **17 verified on the first pass, 4 read FAIL** — all four from
`2607.07066`, including both limitation sentences above. They were not wrong. `pdftotext`
on a two-column paper **interleaves the columns line by line** *and* **hyphenates across
lines** (`"only cor-" / "relational"`). Flattening whitespace does not fix it; de-hyphenating
does not fix it either, because the two halves of the word are separated by the *other
column's* text. Fixed by cropping each column — `pdftotext -f N -l N -x 0 -W 300` then
`-x 300 -W 300` — after which all four pass.

**Worth recording because the failure mode is the dangerous direction**: a quote I had read
correctly on screen reported as absent. Had I trusted the instrument over the reading, I
would have removed two correct and load-bearing quotations. The same trap is waiting for any
citation check run against a two-column PDF, which is most of them.

### A2 — the claim map, and three claims with less behind them than the ledger implies

`paper/CLAIM_MAP.md`, 156 lines, 39 rows. Verified rather than asserted: the 39 IDs are
**set-identical** to the ledger's (`comm` empty both ways); every referenced script,
`results/` directory, pre-registration, log and figure **resolves on disk**; all seven cited
pre-registration SHAs resolve and each is in fact a *"Pre-register"* commit; §5's section
table covers **39 of 39** with nothing unassigned and nothing `omitted`.

**Three provenance gaps, recorded as defects:**

- **C12 (Softmax Collapse) has no saved artifact and no script in the repo.** Its evidence is
  the printed table in `FINDINGS.md` §8.1 from an early ad-hoc n=17 run. `reproduce.sh`
  covers C13 (via `refcheck.py`) but **not C12**. It is the only ledger claim with neither
  data nor code behind it. Fix: a ~2-minute rerun saving a stamped `.npz`, zero quota.
- **C16 (4.8× faster)** — `test_model.py` verifies the *numerics* at 2.68e-15; the
  2149 → 449 ms/step timing is **unstamped and machine-dependent**.
- **C11 (engine groks)** — its acceptance test only began asserting on 2026-09-15 (Entry 45),
  so it cannot be written as long-standing evidence.

**Figure gaps confirmed: F4, F7, F8 do not exist.** F4 — baseline/restricted/excluded per
modulus plus the 200-draw permutation null — carries **C22, C23 and C36**, i.e. the paper's
central evidence.

### A4a — the bibliography is not a source, and says so on its own first page

`paper/refs.bib`, **51 entries** (49 numbered rows + the 2 IACR ePrints);
`paper/CITATION_STATUS.md` records the method and every disagreement.

The bibliography warns that its identifiers were *"compiled from a research report and from
memory"*. **arXiv's API is unreachable from this box** (empty response, no outbound network —
the same cause as the session's four failed MCP servers), so every entry was checked against
**the PDF on disk**. That is better provenance than an API call: it is the paper.

**Every ID↔title binding that has a PDF is correct** — no identifier pointed at the wrong
paper. What was wrong was titles truncated and metadata missing.

- **Flags 20 → 9.** 15 of the table's 20 `[VERIFY]` discharged by reading page 1; 5 remain;
  **4 new flags added** where this pass found uncertainty the table had not marked (no PDF,
  image-only PDF, guessed given names). The honest total went up in one place and down in
  another, which is the point of counting it.
- **Two "unknown" identifiers were already on disk.** `#24 (How) Can Transformers Predict
  Pseudo-Random Numbers?` = **`2502.10390`**; `#39 ILDR` = **`2604.20923`**. `LAB_PROTOCOL.md`'s
  "check whether the answer is already on disk" applies to bibliography rows too.
- **One note is simply false:** `2606.17399` is **not** image-only. `pdftotext` reads it at
  304 lines. Nothing needs OCR.
- **12 corrections** recorded in `CITATION_STATUS.md` (truncated titles at #3, #22, #23, #27,
  #34; venues at #6 TMLR and #35 ICLR-workshop-not-main-track; #41's duplicate PDF; #43/#44
  are Transformer Circuits Thread).

**⚠️ Tier 4 is a 2026 preprint cluster and the bibliography does not label it as one.**
Three of its four entries carry the profile Tier 8 exists to flag. The one to watch is
**`2607.06639`** — *"At-Grok Is Not Converged: A Measurement-Validity Audit for Grokking
Representation Metrics"*, **single author**, affiliation *"H&K Research Studio / Clevix LLC,
Hanoi, Vietnam"*. **That is C24's claim.** It is the `Neural Prime Sieves` pattern named in
`LAB_PROTOCOL.md`: on-topic enough that it must be checked before it is touched. Also flagged:
`2606.12966` (single author, NYU) and `2604.20923` (single author, Virginia Tech, **same
`2604.x` range as two Tier-8 entries**). `2605.06352` is four authors at Imperial/QMUL/
Fribourg and draws no flag. **None of the three is cited until A4b clears it, and nothing
rests on any of them** — C24 is held independently.

### A3 — the outline, and the decision the researcher approved

`paper/OUTLINE.md`, 298 lines, 14 sections plus a figure inventory. Checked: all 39 claims
named and resolving in `CLAIM_MAP`; all 12 bib keys resolving in `refs.bib`; no banned
register word except the rule that forbids it; **every mention of a retracted claim is either
Appendix-B routing or an explicit prohibition**.

**The consequential call, put up for approval and APPROVED IN-SESSION:**

`FINDINGS.md` §6 presents the unit-action coordinate as the answer to Chen's non-square-free
limitation, and the ledger marks that **C7, VERIFIED, 1 seed** — which would put `(1 seed)`
on the headline contribution, in the abstract. But §3.8 states the algebra C33 rests on as
*"for x ∈ J_d write x = d·u … the 𝒥-class is a torsor under the units, **which is C7's
'action coordinate' stated exactly**"* — measured at **5 seeds over 26 non-unit strata** under
a pre-registration committed before any per-stratum number existed.

**The researcher confirmed the two are equivalent.** So the paper **leads with C33** and
writes **C7 as the original single-seed sighting**. The answer to Chen's stated gap is a
5-seed pre-registered result, not a 1-seed one. *This does not change C7's ledger status,
which remains `VERIFIED, 1 seed`.*

**Five further decisions, all approved:** related work early (§3) rather than late, following
Nanda and `2606.17399` against Zhong's placement; the causal test as **its own Results
section** against corpus convention, with its first sentence obliged to say why; **F4 built
before §7 is drafted**, since that section's argument is a distribution with a marked point;
**C12 gets an artifact or gets scoped**; and `FINDINGS.md` §3.8's stale pre-k09 count fixed.

### A stale count that has now been caught twice

`FINDINGS.md` §3.8 says **"13 of our 18 moduli are non-square-free, five are prime powers."**
Both figures are **pre-k09**. The true counts (`MASTER_PLAN.md` §0.1, computed from
`algebra.py`) are **23 moduli, 14 non-square-free, 6 prime powers**. `MASTER_PLAN` caught
exactly this pair of errors in its own first draft during the grilling; the same pair is
still live in `FINDINGS.md`, which is the file `write-paper` Step 3 sends a drafter to read
first. **A count that looks safe because it is nearly right**, for the second time.

### Hygiene

No training job, no systemd unit, no `bash -c` waiter; the only child of the background
bg-spare` daemon is its own claim socket. `scripts/alive.sh` blank for `run_n7.py`,
`analyze_gate2.py`, `analyze_k03.py`, `run_o18_cpu.py`. Self-checks 8/8 PASS at session
start; nothing this session touched `src/`, so they are untouched.

**Correction to this entry, made at session-end checkpoint before it was committed.** Above I wrote that
arXiv's API being unreachable meant "no outbound network" on this box. **That is wrong, and
`git push origin master` disproved it minutes later by succeeding** — HEAD and
`git ls-remote` agree at `d63ff3b`. So github.com resolves and arxiv.org does not: some
hosts are reachable and some are not. The A4a decision is unaffected — the PDFs on disk are
better provenance than an API call either way — but **"the network is down" was an
overstatement inferred from one failed endpoint**, which is the same shape as every other
false generalisation in this notebook. **Test the endpoint you actually need.**

---

## 2026-09-15 — Entry 47: the paper starts, and C12's artifact corrects C13

Session 14. Continuous drafting authorised by the researcher after B1 was approved, so the
one-section-per-invocation gate is suspended for this session by explicit instruction.

### B1 landed, and the draft produced two defects of its own

`paper/sections/01-setup.tex` + `paper/main.tex` (`b70159a`, caption SHA fixed in `bf56f65`).
Carries C1 and C2. `scripts/modulus_table.py` regenerates Table 1 from `algebra.describe`
plus the primary arm on disk; its self-check asserts MASTER_PLAN §0.1's 23/14/6, C2's
`0,0,1,2,8`, Thm D.17 in **both** directions, and that 128 is the only non-cyclic prime power.
`algebra.py::stratum_identity` verifies the Setup theorem by exhaustion — **304,426 pairs
across all 23 moduli during drafting, zero failures**; four moduli in the committed check.

Two silent defects, both caught by looking rather than by a check:

- **Four `\ref`s pointed at the section's OWN labels**, standing in for sections that do not
  exist yet. `check_tex.py` passed them because the labels resolve. A check whose failure mode
  is indistinguishable from its success mode, again. Replaced with prose + `TODO(Bn)` markers.
- **A trailing `%` comment swallowed a sentence of body text.** `check_tex` strips comments, so
  it structurally cannot see this. Every comment is now first-on-line, asserted by a scan.
- **A SHA stamped into a caption was orphaned by an `--amend` seconds later.** A caption naming
  a commit `git log` cannot reach is worse than no caption, because it looks checked. Stamping
  a generated table's provenance takes **two commits** — noted for F4/F7/F8.

### C12 now has a script and an artifact — and it corrects C13

`run_c12_collapse.py` → `results/c12_collapse/softmax_collapse_n17.npz`, SHA `e0ead8e`,
`git_dirty=False`, wired into `reproduce.sh`. This was the only ledger claim with **no
artifact and no script**. ~4 min local CPU, zero quota.

**C12 reproduces**: |grad| falls **6.2 orders** while `p_correct` reaches exactly
**1.000000**; the step-4000 softmax gradient matches the published 2.3e-7 exactly.

⚠️ **C13's published magnitude does not.** *"|grad| 2.3e-6 vs softmax's 2.3e-7 at step 4000 —
~10× more gradient at the same logit scale"* reads **7.510e-07 vs 2.353e-07 = 3.2×** here, and
**"at the same logit scale" is false**: softmax's max logit is **50.1** against stablemax's
**19,154.3**, a factor of **383**. StableMax's logits have no saturation ceiling, so the two
arms are not at a comparable scale past ~step 100 and the ratio was never like-for-like.
**The direction holds and the mechanism holds** — `p_correct` stops at 0.999998 instead of
saturating — **so C13 goes in the paper as a mechanism and never as a 10× number.**
FINDINGS §8.1 amended.

**A defect in my own new script, caught by the artifact:** it used `provenance.stamp()`, which
spreads its keys into the npz, where `provenance.read()` looks for a single `provenance` key
and returned `None`. The artifact was stamped and **unreadable by the project's own reader**.
Fixed to `stamp_npz`, and the script now **asserts read-back before it prints a result**.

### FINDINGS §3.8's stale count, caught for the third time, now fixed

*"13 of our 18 moduli are non-square-free, five are prime powers"* → **14 of 23, six**, with
the old figures named in place as pre-k09 so the correction is visible rather than silent.
This is the file `write-paper` Step 3 sends a drafter to read first.

---

## 2026-09-16 — Entry 48: the paper is drafted whole, and re-deriving its numbers caught three defects

Session 14 continued. All fourteen sections drafted (`paper/sections/`), `main.tex` wired,
`check_tex.py` clean at 15 files / 54 labels / 52 refs / 49 cites / 0 failures, zero banned
register words.

### The instrument that mattered: `test_paper_numbers.py`

A number in the paper agreeing with `FINDINGS.md` proves only that **two documents agree**.
The new check recomputes each load-bearing number **from the saved artifact** and fails on
disagreement — 34 assertions over Gate 2, the causal ablation, the modulus table and the
collapse run. It is in `reproduce.sh`. It caught three defects that prose review did not:

**1. `FINDINGS` §3.3's 5-seed column is mislabelled AND is a heavy-tailed mean.** The header
says `restricted / baseline`; the values are `baseline / restricted`. The numbers reproduce
exactly, so only the label is wrong — and the draft had copied the header, which would have
printed the ratio **upside down**. Worse, it is a **mean**: at n=113 one seed of five reads
**264×** where the other four read 1.56–8.0, giving "55.5× ± 104.1", a summary whose sd is
twice its value. **This is the control-mean defect, in the column three paragraphs below the
box that retired the control-mean defect.** The paper now reports the **median with its
range** and leans on the permutation p, which is 5/5 everywhere and is not a ratio.
FINDINGS §3.3 corrected in place.

**2. "models grok at 20 of 23 moduli, the three exceptions being 49, 54, 63" was wrong twice.**
The truth: **19 of 23** grok in a majority of seeds, only **n=49** fails in every seed, and
there is a **fourth exception, n=91 (1/3)**. n=91 is the one k09 already flags as a falsified
prediction of our own — it has 2,484 training examples, above n=81 which groks 3/3. The paper
now states it as a failed prediction rather than omitting it.

**3. "p_correct reaches exactly 1.000000" is a six-decimal DISPLAY, not a value.** In float64
it maxes at **1 − 3.4e-8** and is never exactly 1.0 at any logged step. The no-gradient regime
here is driven by the logit gap, not by a rounded probability. Corrected; the `p_correct == 1.0`
assertion is now inverted and asserts it is **never** exactly 1.

### Two provenance repairs, and one that cannot be repaired

- **The ablation artifacts behind F4 were `git_dirty=True`** (`226b1c95`, 2026-09-11) — the
  paper's central figure resting on a stamp that says "not reproducible from this SHA alone".
  Re-run on a clean tree: both k03 and k04 now `ccb590b`, `dirty=False`, and every number
  re-derives identically (p<0.01 in 5/5 at all six moduli, unchanged).
- **G2's upper bound is 2.71×10⁸, not the 1.0×10⁸ FINDINGS states.** Measured from the
  artifact. The paper uses the measured value. ⚠️ **Reporting a LARGER effect than the record
  is the flattering direction — flagged for the researcher rather than quietly adopted.**
- **⛔ 31 artifacts carry `git_sha = "unknown"` and 31 carry no stamp at all** — all of k02
  (no stamp) and all of k03 (`unknown`). Every sweep from k04 onward, and every engine run,
  carries a real SHA. **k03 is the source of C22, C23, C24, C26, C31 and Gate 2**, so this
  touches the paper's central evidence. It is not repairable without re-running on quota, and
  it is written into Limitations as a stated limitation rather than left implicit.

### `check_tex.py` now catches the defect that shipped in B1

A `\ref` **inside the sectioning unit it labels** is never correct — it is a placeholder, and
it passes the has-a-label check because the label really is there. Four shipped in B1.
Drafting the remaining twelve sections produced **seventeen more**, in every single file. All
retargeted to real cross-section destinations; the check is in `check_tex.py` with a planted
case in its self-check. First version of the check had its own bug — it ended a subsection's
block at the next `\section`, so a subsection swallowed its siblings and false-positived on a
legitimate sibling reference. Fixed to split at the next heading of any level.

---

## 2026-09-16 — Entry 49: the mathematics, the title, and a fourth number that was wrong

Session 14, second half. Researcher instruction: title the paper, resolve the four flagged
items on my own recommendation, and **formalise the mathematics** ("display all central
equations… add more maths in general where possible"). Zero Kaggle quota, no push attempted,
no training run.

### The mathematics: 3 displayed equations → 43

Plus **1 theorem, 3 propositions, 4 proofs**. The rule applied throughout: *a quantity the
reader must be able to recompute gets a formula, not a sentence.*

**Theorem 1 (the stratum identity) is now proved**, in three parts, each of which is *also*
asserted in `algebra.py::stratum_identity` over every one of the `n²` pairs:

1. `v_p(w) = max(α−β,0)` and `v_p(q) = max(β−α,0)` with `α = v_p(de)`, `β = v_p(n)`; at most
   one is positive, hence `gcd(w, q) = 1`.
2. `d | de` and `d | n` give `d | m`, so `q | n/d` and `q | n/e`, so `uv` is a unit mod `q`.
3. `(mt) mod n = m·(t mod q)` for `m | n`.

Checked by exhaustion over **all 23 moduli, 304,426 pairs, zero failures.** The committed
check covers 113 / 121 / 125 / 120 — the field, the two p-adic chains, the widest lattice.

**Propositions 1–3, each with a proof, each earning its place:** the torsor bijection
`ι_d : (Z/(n/d)Z)^× → J_d` (what makes a *non-regular* class usable); **rotating the feature
axis is a provable no-op** (why the random-orthogonal control acts on the residue axis — the
failure mode *looks like a result*); and **generator equivariance** (why a key-frequency
*value* is not a property of a model).

**Everything now defined rather than quoted:** Gini with its exact bound `[0, 1−1/M]`, PR on
energy, the two IPR conventions side by side (with `IPR₂ = 1/PR` verified to **2.2e-16**), the
spectral projection `Π_S` so restricted and excluded are one definition apart, the permutation
`p`, enrichment, diagonal share `D` and `R`, neuron tuning, drift, the fibre residual, the
effect size `f`, partial Spearman correlation, the three-state classification rule, the 2D bin
convention, the coverage fraction, and the CRT-dual set.

**Two formulas were checked against the code and one was wrong.**
- `eq:partial` reproduces the published **+0.735 / +0.627** exactly; `eq:effectsize`
  reproduces N9's **−0.07**.
- `eq:coverage` = `Σ φ(n/d)φ(n/e) / n²` reproduces **all twelve** published coverage figures
  exactly (98.2/98.2 · 82.6/97.7 · 64.0/89.6 · 65.1/88.6 · 7.1/23.1 · 23.5/82.3).
- ⚠️ **`eq:crtset` was wrong until checked.** The CRT-dual set needs the conjugate-folding
  ceiling at `⌊n/2⌋`; without it `|P(n)|` **doubles** and the enrichment test means nothing.
  Verified against `test_crt_law.predicted` on all 23 moduli, **0 mismatches**.

### ⚠️ FOURTH DEFECT: the torch-float64 cell is 0.7728, not 0.7736

The 2×2's four cells were **quoted from the lab record**. Recomputed from their own artifacts
with `tail_gini` (mean over the final 10k steps, 3 seeds each):

| cell | recomputed | recorded |
|---|---|---|
| engine float64 | 0.753636 | 0.7536 ✅ |
| engine float32 | 0.767148 | 0.7671 ✅ |
| torch float32 | 0.579105 | 0.5791 ✅ |
| **torch float64** | **0.772770** (0.7618 / 0.7856 / 0.7710) | **0.7736** ❌ |

So the pre-registered **F1 is `+1.110`, not `+1.114`**. **The verdict is unchanged** — F1's
threshold is 0.70 and both clear it — but `figures/F7_precision.png` and the paper now
**recompute every cell** instead of quoting, and all four are asserted in
`test_paper_numbers.py`. Found while chasing a 4th-decimal mismatch that could have been waved
off as rounding. **It was not rounding.**

⚠️ Re-confirmed the documented trap in the same hour: `analyze_n9.py` on `results/o18_f64`
prints **"NOT HELD (f = −0.10)"** because its baseline/target are the *engine's* two dtype
cells. The committed formula has to be **computed**, never read off that summary line.

### The four flagged items, resolved

1. **Title.** *"A Clock on Every Stratum: Causal Character Circuits for Modular Multiplication
   at Non-Square-Free Moduli."* Author is the researcher; **the supervisor line is an explicit
   `TODO` in `main.tex` and is the one genuine blocker left.**
2. **G2's bound.** The artifact is the authority: `FINDINGS` corrected **1.0×10⁸ → 2.71×10⁸**,
   with the correction named in place. The measured value is *larger* — the flattering
   direction — so it is recorded, not adopted quietly.
3. **C3's ledger row** said 18 moduli → **23**, with the four non-grokking exceptions named
   (49 0/3, 54 1/3, 63 1/3, **91 1/3**).
4. **The 62 artifacts with no usable SHA — recovered by argument, at zero quota.**
   `scripts/reconstruct_provenance.py`:
   - **k03's `run.py` has EXACTLY ONE commit in the whole history** (`477e067`). There is one
     version of that file, so the code version is **determined**; `unknown` records a stamping
     failure, not an ambiguity.
   - **k02's has three**, and the script's coarse AST check **FIRED** on `Transformer`.
     Resolved exactly: the `forward()` computation and the `acts=False` return are
     byte-identical across all three, and the only change is the tuple returned under
     `acts=True`, which the training (line 97) and eval (line 102) call sites never request.
   **The check failing first and then resolving is the point** — I did not weaken it to pass.
   Limitations rewritten to call this a **reconstruction, not a stamp**.

### `test_paper_numbers.py`: 34 → 53 assertions

All from artifacts, never from prose. **Four defects across the session** have come from this
one instrument: the mislabelled heavy-tailed ratio column, the grok count wrong twice, the
six-decimal display read as an exact value, and now the torch-float64 cell.

### Hygiene

Self-checks **12/12 PASS** on an idle box (`logs/suite_session14.log`), `check_tex` clean at
15 files / 103 labels / 115 refs / 52 cites / 0 failures, zero banned register words. C8
re-verified: 165 **14.08×**, 120 **11.76×**, 119 **3.30×**, their checkpoint **13.79×**, all
`p = 0.00e+00` at a 1/20,000 floor. No background job, no systemd unit, no waiter.

---

## 2026-09-16 — Entry 50: five diagrams, and the two defects drawing them exposed

Session 15. Researcher instruction: add visuals, and in particular an illustration of the
central proof that "anyone can get by looking at it and reading". Zero Kaggle quota, no push
attempted, no training run.

### The gap, stated plainly

The draft had **six figures and all six were results plots** — bars, scatters, curves,
heatmaps. It also carried 43 displayed equations, 1 theorem, 3 propositions and 4 proofs with
**no visual support whatsoever**. Theorem 1 is the hinge of the whole paper and existed only
as symbols.

### Register taken from the corpus, not invented

`2301.05217` Fig. 1 is a three-column strip: architecture stack ‖ the algebra at each stage ‖
a small geometric picture. `2306.17844` Fig. 1 does the same and adds a **worked numeric
example** (`2+3 = 1+4 = 12+5 = 5 mod 12`) — which is the thing that makes it readable without
the body text. Both sit on page 2. F1 borrows the worked example directly.

### Five figures (F0, F1, F2, F3, F5), all matplotlib, all regenerated

**Matplotlib rather than TikZ, and the reason is checkability.** This box has no TeX; Overleaf
is the compiler. TikZ means shipping unrendered code. Every panel here was rendered and
**looked at** before it landed, which caught six layout defects that no amount of reading the
code would have found: κ-labels colliding with their nodes, `J_15` and `J_30` clipped off the
`n=120` lattice by a hard-coded ±2.6 xlim, two panel notes drifting to the figure edge under
`aspect="equal"`, and the `ι_d` equation overlapping the paragraph above it.

**Print size is the binding constraint and it is easy to get wrong.** The first F1 was 13 in
wide. At `\textwidth` (~6.5 in) that halves every font to ~4.5 pt. Redrawn at 10.2 in with an
8.5 pt floor; the collisions above all returned at the larger font and had to be fixed by
cutting copy, not by nudging.

**Choosing F1's worked example was the real design decision.** Four constraints: non-square-free,
a **non-regular** source class, a genuine collapse of cells onto targets, and `w ≠ 1` so the
unit twist is not invisible. `n = 20 = 2²·5` with `J_2 × J_4` is the smallest modulus meeting
all four (`m=4, w=2, q=5`, 16 cells → 4 targets). `n = 15` is its foil: `J_3·J_3 = J_3`,
closed, idempotent 6 — Chen's Thm D.17 in a single block.

### ⚠️ DEFECT 1: `analyze_n4`'s grid assembly had no second caller, and nearly got a copy

F3 draws the character plane the causal test ablates. The code that builds the exponent-ordered
logit grid **and chooses the key set** lived inline in `analyze_n4.run`. Copying those ~15 lines
into `plots.py` would have created **two answers to "which characters are the key set"** on the
paper's central evidence. Extracted to `ablation.unit_logit_grid` instead, and both call it.
`analyze_n4.py results/k03_grid_acts` was captured **before and after** and diffed:
**byte-identical, no number moved.** An extraction is only safe once that diff has been run.

### ⚠️ DEFECT 2: F4, F7 and F8 had no caller anywhere in the repository

`grep -rn "causal_test\|precision_2x2\|grok_curves"` outside `plots.py` returned **nothing**.
The three figures added in session 14 — including **F4, the paper's central evidence** — were
produced by an ad-hoc one-liner and nothing in the repo rebuilt them. `figures/**/*.png` is
gitignored by design, so on a fresh clone those three could not be regenerated at all. This is
the same defect as a stale figure, one step earlier, and it is exactly what the project's
"figures regenerate from saved artifacts" rule exists to prevent. All ten cross-run panels are
now in `scripts/render_all.py`; one command rebuilds every figure the paper uses.

### Caption numbers are numbers

`test_paper_numbers.py` **53 → 84 assertions**. Every value F1, F2, F3 and F5 assert is
re-derived from `algebra.py` or from the `.npz`: `m`, `w`, `q`, the cell and target counts,
`6·12 ≡ 12 (mod 20)`, the per-modulus strata and non-regular counts, `ν(121)=2` vs `ν(125)=3`,
and F3's `4` key characters / `9` kept pairs / `12,091` zeroed / `7` orders of magnitude. F1's
generator additionally asserts \eqref{eq:stratum} **and** the non-regularity of `J_2` on the
block it draws, so the figure cannot illustrate a theorem it violates.

### Hygiene

`check_tex` clean: 15 files / **108 labels** / **126 refs** / 54 cites / **0 failures**.
Self-checks **10/10 PASS** on an idle box. `test_paper_numbers.py` **84/84**. No background
job, no systemd unit, no waiter.

**Two pre-existing register hits found by the banned-word grep and NOT edited** (they are body
prose, outside this session's scope, and flagged for the researcher):
`06-results-clock.tex:60` says *"several grokked runs"* where the count is known — 19
excursions across 11 runs; and `05-methods.tex:171` uses *"a striking result"*, which is inside
a quoted description of a misreading and is arguably legitimate.

---

## 2026-09-16 — Entry 51: the last two figure gaps, and a table that became a figure

Session 15 continued. Researcher instruction after reviewing the first five diagrams: build
the two gaps I had named. Zero Kaggle quota, no push attempted, no training run.

### F6 — C8, the CRT-dual law (§9)

The project's **best-supported claim had no figure of its own**. `omega_bands` is about
ω(n), a different question. Four columns × two rows: the raw additive spectrum with `P(n)`
highlighted on top, the 20,000-permutation null underneath on a log axis.

| column | enrichment | p |
|---|---|---|
| n = 165 = 3·5·11 | **14.08×** | 0.00e+00 |
| n = 120 = 2³·3·5 | **11.76×** | 0.00e+00 |
| n = 119 = 7·17 | **3.30×** | 0.00e+00 |
| n = 165, **their published checkpoint** | **13.79×** | 0.00e+00 |

`|P(165)| = 8` of 82 bins, `|P(120)| = 7` of 60, `|P(119)| = 11` of 59. The p floor is
1/20,000 = 5.0e-05.

**n = 119 is kept at 3.30× on purpose.** It is the modulus that was enriched all along and
invisible to a 5×-median threshold — the reason the test is threshold-free. Dropping the weak
column would delete the argument for the method.

**⚠️ THE FIGURE ASSERTS ITS OWN POSITIVE CONTROL BEFORE IT DRAWS.** `n = 165` must reproduce
the published **14.08×** or `crt_law()` raises. `predicted()` returns FREQUENCIES and
`freq_energy()` returns BINS AT POSITION FREQUENCY − 1; that off-by-one is **loud at exactly
one modulus (n = 120, IndexError) and silent everywhere else**, and it has already cost this
project one false "DEPLETION" reading (Entry for k09 P5). A figure that can be silently wrong
needs a control that cannot be.

**Their checkpoint is a column, not a footnote**, and it degrades to an explicit
*NOT AVAILABLE — run scripts/restore_external.sh* panel when `reference/` is absent, rather
than silently rendering three columns.

### F9 — the Gate 2 discrimination (§8), which REPLACED a table

183 grokked and 19 failed stratum-measurements, scored by `scripts/gate2_discriminate.py`:

| criterion | grokked | FAILED | separates |
|---|---|---|---|
| G0 held-out acc ≥ 0.95 | 183/183 | 0/19 | yes |
| **G1 perm p < 0.01 — pre-registered PRIMARY** | 182/183 | **19/19** | **NO** |
| G2 excluded/base ≥ 100 | 183/183 | 0/19 | yes |
| G3 resid ≤ 0.5 × ctrl | 122/122 | 2/6 | partly |
| G4 R ≥ 5 | 183/183 | 12/19 | partly |
| G5 tuned − shuffled ≥ .2 | 109/111 | 0/13 | yes |

**§8 already printed all twelve of those counts as a table.** Two representations of the same
twelve numbers in one subsection is bloat, so the figure was given the table's `separates?`
verdict column and **the table was deleted**. The figure is now a strict superset: every count
is printed at its own marker, and the *gap* between the two markers — which a table can only
assert in bold — is a length. G1's two markers **coincide**, so its gap is zero. Rows are
sorted by gap, which sorts the failure to the bottom without any manual emphasis.

### Both reused their scorer rather than re-deriving it

F6 imports `predicted` / `freq_energy` / `permutation_test` from `test_crt_law.py`; F9
imports `TESTS` / `load` from `scripts/gate2_discriminate.py`. `permutation_test` gained a
`return_null` flag so the histogram is drawn from the same draws the p-value is computed
from — the same device `analyze_n4.run` already carries as `return_draws`, and for the same
reason. **`test_crt_law.py` output captured before and after the change and diffed:
byte-identical, no number moved.** That is the second such before/after diff this session.

### Hygiene

`test_paper_numbers.py` **84 → 109** — every enrichment, every `|P(n)|`, both published
predicted sets, the vacuity of `P(121)` (= all 60 bins), and all twelve Gate 2 counts.
`check_tex` clean: 15 files / **110 labels** / **127 refs** / 55 cites / **0 failures**.
`test_crt_law`, `test_ablation`, `test_analysis`, `test_reproduce` all PASS.
`render_all.py` now regenerates **twelve** cross-run panels; every figure in the paper has a
caller. No background job, no systemd unit, no waiter.

---

## 2026-09-16 — Entry 52: the panel, and the number that was a mean of 2.77 and 476

Session 16. Researcher instruction: the session-start brief, then `/write-paper`, then `ars-reviewer`
"as planned", then — after the panel reported — "just fix it". Zero Kaggle quota, no push
attempted, no training run.

### The panel

Five reviewers, ARS `academic-paper-reviewer` v1.10.0 `full` mode, under the v3.6.2 sprint
contract: each wrote its block/warn triggers **paper-blind, to disk**, before opening the
manuscript. The first launch died mid-Phase-2 on the account's session limit; four
pre-commitments survived, and **that made the protocol MORE faithful, not less** — Phase 2 came
back as a physically separate call re-injecting a commitment made before the paper was seen,
which is what the protocol specifies and what the first attempt had collapsed into one context.

**MAJOR REVISION, 5 of 5.** F2 fired on every reviewer; F1 on two — Domain and Devil's
Advocate independently scored D3 `block` without reading each other.

### What the panel was right about, and what re-derivation changed

**Not one reviewer finding was taken on report.** Four were confirmed against the source and
two were overturned:

| claim | verdict on re-derivation |
|---|---|
| the ablation never re-runs a forward pass | **confirmed** (`ablation.py:115` FFTs the saved logit grid) — but it IS Nanda's cited protocol, so the exposure is the word *causal*, not the procedure |
| "measurable stratum" used 6× and never defined | **confirmed** — and the cut is `\|G_m\| >= 10` and `>= 100` held-out cells, **structural, never read off a model's output**, which ANSWERS the "G0 cannot fail" charge once written down |
| `01-setup.tex:267` claims `m` exceeds `gcd(d,n)gcd(e,n)` | **confirmed impossible.** That product is `de` and `m = gcd(de,n) <= de`. 0 counterexamples over every `n < 400` |
| "10 of those 12 moduli are non-square-free" | **confirmed wrong: 8** |
| "27 of 29 is really 26" (Domain) | **OVERTURNED. 27 of 29 is correct.** The *exception list* is wrong: the two failures are n=98's seed (0.015) and n=49 (0.010). **n=63 passes at p=0.0** — it is n=54, in a different sweep, that fails at 0.085 |
| R3's line numbers | **unreliable** — cited `:288` in a 145-line file. Substance held on both items tested |

### ⚠️ THE FINDING OF THE SESSION: 239.52 IS THE MEAN OF 2.77 AND 476.26

The panel flagged the *form* — mean ± sd on a heavy-tailed sample. Re-deriving showed the
*magnitude* is what matters. Every ratio in §7.3 is a mean over 2–3 draws:

| n | values | as published | median |
|---|---|---|---|
| 128 | 2.77, **476.26** | 239.52 ± 236.74 | 239.52 |
| 131 | 2.44, 2.88, **781.26** | 262.19 ± 367.03 | **2.88** |
| 127 | 2.18, 5.04, 51.63 | 19.62 ± 22.67 | **5.04** |

**n=131's headline "262×" describes no run in its sample; its median is 2.88×.** The `± 236.74`
at n=128 is the *population* sd of two points — it is just half their difference wearing a
statistic's clothes. The abstract quoted "239×".

And `FINDINGS.md` labelled that column **`restricted/baseline`** while holding
**`baseline/restricted`** — the upside-down-ratio defect `LAB_PROTOCOL.md` already records for §3.3,
**for a third time, in the table for the claim the abstract leads with.**

**What survives is the claim.** The permutation p is primary and all 13 runs are beyond all 200
of their draws. What does not survive is any sentence quoting a mean of these ratios.

### The blockers, and where the line was drawn

Two panel blockers need **new measurement**, and this project pre-registers before it runs:
- *sameness is never measured* — every statistic is computed inside one stratum, so "the same
  clock everywhere" is inferred from "a clock here, and a clock there". The instrument exists
  (`K_h = K_W`, 14/14) and was never lifted across strata. **Scoped in §8.4, not measured.**
- *causal* — the intervention acts on the logit tensor. **Stated explicitly in §5**, and the
  abstract now says so.

Both are written down as the experiment that would restore the strong claim, and neither was
run. The rest of the roadmap was applied: 23 edits across 8 files.

### Hygiene

`check_tex` 15 files / **111 labels** / 135 refs / **61 cites** / 0 failures. Self-checks
**8/8 PASS**. `test_paper_numbers.py` **120 → 151** — every corrected number re-derives from its
artifact, including a check that `m` can never exceed `de` and that **n=63 passes**, so the
retracted exception list cannot come back. Banned-word grep clean for the first time.
Two new bib entries (Steinberg 2016, Howie 1995): `refs.bib` had **zero semigroup sources**.

---

## 2026-09-16 — Entry 53: addendum to Entry 52 — the `/write-paper` fix that preceded the panel

Entry 52 recorded the panel and the corrections but omitted the `/write-paper` invocation that
ran **before** it in the same session. Recorded here rather than by editing Entry 52.

### The near-grok is 0.976, and the draft said 0.978 twice

No section was left to draft, so the invocation took the pending section-level work: the
register hit STATE had flagged (`06-results-clock.tex` said the spikes occur in *"several
grokked runs"* where the count is known). Checking it turned up a number defect underneath.

`WE_B_thesis_n125_s1.npz` (k03): **window median 0.976001, final logged sample 0.977875.**
Two places in the draft quoted the **last sample** where the surrounding sentence names the
**window median**:

- `06-results-clock.tex:58` — F8's caption, *"the one run at a window median of 0.978"*. That
  is the caption of the figure whose entire argument is that a run's endpoint is a window and
  not its last row.
- `05-methods.tex:213` — `\bar{a} = 0.978`, **two lines below `\eqref{eq:window}` that defines
  `\bar{a}` as the median of the final ten samples.**

**Prose review cannot catch this.** 0.976 and 0.978 are both true numbers about the same run;
only re-derivation separates them. Note the near-coincidence that makes it worse: n=91's
near-grok in k09 also reads 0.976 (median 0.976453), so the section contains two unrelated
correct 0.976s and one wrong 0.978.

### The register fix, with the count put back

*"the downward spikes in several grokked runs"* → **6 post-grok excursions below 0.90 in 5 of
the 30 runs drawn**, each exactly one 200-step logging interval wide, **all six recovering,
none censored**. The old sentence also implied the censored excursion was on screen; it is
not — it is 1 of 19 in k06's 120,000-step sweep, and the caption now says so.

`test_paper_numbers.py` **109 → 120**, asserting **both** readings of `n125_s1` (median 0.976
*and* last sample 0.978) so the pair that was confused here cannot be confused silently again,
and computing the excursion counts through `analyze_excursions`' own functions rather than a
second copy. Commit `92565cc`.

### And a 13th figure with no caller, found at the very end of the session

Bundling the paper for Overleaf: `basis_comparison` is referenced by
`06-results-clock.tex:118` and **is not in `render_all.py`'s cross-panel list** — directly
under a comment added in session 15 reading *"EVERY FIGURE THE PAPER USES REGENERATES HERE"*.
That comment was written while fixing F4/F7/F8 and was false for the 13th figure the whole
time. Wired in and regenerated. **A completeness claim must be verified by diffing the two
lists, not by having just edited one of them.**

---

## 2026-09-21 — Entry 54: the review's remaining errors, and the objection that could not be rewritten away

Session 17. Researcher instruction: the session-start brief, then *"fix every error after the review"*,
scoped in conversation to **all twelve remaining items**, with the causal claim resolved by
**retitling and scoping harder** rather than by running an internal intervention. Zero Kaggle
quota, no push attempted, no training run — every measurement is re-analysis of artifacts
already on disk.

### What was actually still broken

Session 16 applied 23 roadmap edits and recorded the roadmap as "applied except the two items
needing new measurement". Re-deriving each item against the files found **eight** text defects
still live, two of them instances of a fix that had been applied to the abstract only:

- `02-introduction.tex:35` still read *"$p < 0.01$ at every modulus"* and `13-conclusion.tex:10`
  *"at every modulus we tested"* — the panel's C-2 overclaim, where the body says 27 of 29 with
  two exceptions named. The intro bullet also still quoted **239×**, the median of two runs whose
  range is 2.77–476.26, which §7.3 had been rewritten to stop reporting bare.
- **`check_tex.py` was reading a copy of the paper.** It `rglob`s `paper/`, and an unzipped
  `overleaf_bundle/` is a complete second copy: **30 .tex, 270 refs, 122 cites** where the paper
  has 15/135/61 — and **0 failures**, because a stale copy of a correct paper is also correct.
  It now skips the one named build artifact and **fails** on any other duplicate basename; both
  planted in the self-check. Same family as every other false-pass in `LAB_PROTOCOL.md`.
- "torsor under the units" (abstract and intro): the **global** units act transitively with
  stabiliser of order $\varphi(n)/\varphi(n/d)$. It is the **local** units that act simply
  transitively — which the paper's own Proposition 2 already says.
- Three `% TODO(B2/B4/B8)` `\ref` placeholders; the multiplicity statement (absent entirely);
  the external-validity paragraph; the under-sold contribution; and "nilpotent", used twice and
  **never defined**, which was the unapplied half of roadmap item 13.

**The title lost a word.** "Causal Character Circuits" → "Character Circuits". §5 had said since
last session that the ablation FFTs the *saved logit tensor* — no forward pass re-run, no weight
ablated, no activation patched — while the title, the abstract, a section heading, a subsection
heading and six sentences went on asserting the word. The claim and the evidence are unchanged;
what changed is that the paper now names its evidence class everywhere instead of once.

**Two corrections in our favour, both from the Domain reviewer, both re-derived before being
believed.** $J_d \cdot J_e = J_{\gcd(de,n)}$ holds with **equality**, not the inclusion
`eq:jcompose` stated — checked at every divisor pair of every $n \le 200$, **8,253 pairs, 0
mismatches**. And roadmap item 20 said *push back*: the pre-registered decision rule was
`G1 ∧ G2` plus G6, it was **met** (G1 held at 99% of grokked stratum-measurements), and what
fired is a *separate* falsifier clause. Resting the weight on `G0 ∧ G2 ∧ G5` **discards** the
criterion we named primary and leaves a **smaller** evidence base than the registered one. A
post-hoc rescue selects criteria to save a claim; this deselects one that passed.

### R2 — sameness is measured, and it holds

Pre-registered (`373a085`) before any cross-stratum number existed. Strata sharing a local group
select **exactly** the same characters of it: **158 of 160 primary pairs (0.9875)** against a
pre-registered 0.80, Monte-Carlo null **p = 1.0e-4** at the floor, cell-shuffle control
**0 of 160**. Mean Jaccard 0.9982. Both disagreements are `n=165` seed 1 (26/28).

**⚠️ Transposes are not support.** `(d,e)` and `(e,d)` are transposes of one *commutative* task.
Excluding them leaves **0 primary pairs at n=121 and n=125** — the test lives at 165, 120, 119.
**My own pre-registered table said 30+4 at n=165 and the algebra says 28+6**; the hand count
caught `G_11`'s transposes and missed `G_33`'s and `G_55`'s. Corrected in the Outcome section,
never above it, and asserted in the script's self-check so a hand count cannot reintroduce it.

**⚠️ THE FAILED-RUN CONTROL DOES NOT WORK AND THE CLAIM SHIPS SAYING SO.** There are exactly
**two** scorable failed-run pairs in the project and both come from **one run at test accuracy
0.7502** — a model that has learned most of the task, which the three-state rule buckets with
the 0.146 memorisers. Its own null reads **p = 1.0**. Every genuinely failed run has either no
comparable pair by algebra (49, 63, 125) or **no key set at all** (54 s1, at 0.2733). It is
**underpowered, not passing**, and C38 carries that in the ledger row, in §8 and in the appendix.

**Two self-check failures changed the instrument before it saw real data**, which is the only
reason to write self-checks first: a *perfectly* pure planted spectrum has median 0, so "above
5× the median" selects float noise (caught on the Nyquist bin); and **two empty key sets compare
equal**, so an undetected clock would have scored as a perfect match — white noise yields no key
set 20/20 times.

### R3 — the indicators do not select P(n), and that is better than if they had

Pre-registered with **both halves** committed. Support is a strict superset of `P(n)` (the
reviewer was right; a Ramanujan sum is not sparse in support). But the **key** set does not equal
`P(n)` either: at n=165 the indicators hold 3 of the 8, and at n=120 they pick **{20, 40, 60}**
where **20 = 120/6** is the dual of a divisor that is *not* a maximal prime power, so it is
outside `P(n)` and outside the prime variant alike.

**The contrast is the finding.** Same detector, same bin convention, one run: the **network's**
key set is all 8 of `P(165)` and all 7 of `P(120)`, no misses and no extras, while the
stratification's candidate family is the duals of **every** divisor — 42 frequencies at n=165.
So `P(n)` is **not** a property of the stratification alone; selecting the maximal-prime-power
subfamily out of the divisor duals is something the network does. §9's *"we confirm the mechanism
with no network involved"* is retired; C8 is untouched, having always been a claim about trained
models. R3-5's positive control ran **first** and reproduced the published 17.44 (n=165) and 3.27
(n=119) exactly — and resolved a `LAB_PROTOCOL.md` ambiguity: both 14.08 and 17.44 are correct and they
are **different checkpoints** (k01 scout vs k03).

### A defect this session created and caught

`analyze_r2_sameness.py` stamped `git_dirty=True` on a clean tree, because `open(out, "w")`
**creates the untracked file before `provenance.stamp()` reads git status** — the script dirtied
the tree by creating the very file it was about to describe. `analyze_gate2.py` and
`analyze_n4.py` were both already correct. Fixed, artifacts regenerated, and the writer now
asserts its stamp **reads back**.

### A defect found and NOT yet fixed

**`final_test_acc` classifies runs by `hist[-1]` — the last row — which `LAB_PROTOCOL.md` forbids
after the C27 retraction.** Scored both ways over 178 runs, **4 change state**, the worst being
`k06_horizon/WE_B_thesis_n119_s0`: last row 0.6798 → **FAILED**, final-10 median 0.9987 →
**grokked**. That is the known censored artifact, and the classifier reads it as a failure — the
direction that would put a *grokked* model into a control arm. None of the four is in `k03`, so
Gate 2's 157 measurements and C31 are unaffected. **Not fixed yet because `analyze_gate2.py` is
mid-sweep**, and editing it now would make the artifact's stamped SHA describe code that did not
run.

### R1 is in flight

`B = 200 → 10,000` with the Phipson–Smyth estimator `(1+b)/(B+1)`, pre-registered with the
multiplicity family **counted before any p was read: 320 tests**. Nine sweeps launched under
`systemd-run --user` with memory caps; `k05` has landed, the rest are running. The constant
forward FFT was hoisted out of every permutation loop (1.9×, asserted **bit-identical** in
`test_ablation.py`) because at B=10,000 it is the difference between one transform and ten
thousand identical ones.

`test_paper_numbers.py` **153 → 196**. `check_tex` 15 files / 112 labels / 157 refs / 61 cites /
0 failures.

---

## 2026-09-21 — Entry 55: addendum to Entry 54 — R1's first two families, and the detachment check

Recorded here rather than by editing Entry 54.

### Two of the nine R1 families landed before the session-end checkpoint

| family | tests | `n_draws` | min `p̂` | verdict |
|---|---|---|---|---|
| `results/k05_lowdata` | 15 | **10000** | **9.999e-05** | R1-1, R1-2 hold |
| `results/k07_arma_seeds` | 5 | **10000** | **9.999e-05** | R1-1, R1-2 hold |

The minimum is **exactly `1/(B+1)`**, which is the whole point of the Phipson–Smyth form: the
estimator cannot print `0.0` and the floor is a number the design can actually express. k07's
five `A_replication_n113` runs read `p̂ = 0.0001` against the old `0.0000`. k05's per-modulus
summary reads n=49 **7/7 at p<0.01** (max `p̂` 0.0050), n=63 **7/7** (max 0.0037), n=54 **0/1**
(0.0519) — the last being a single failed run, as expected.

⚠️ k05's summary line still prints `14846.90x +/- 36362.46` over 7 seeds — a **mean ± sd on a
heavy-tailed ratio**, the defect `LAB_PROTOCOL.md` retired for the *effect* in C36 and for the
*control* in C22/C23. It is above the n≥5 bar the fix installed, so it is not the same
violation, but `analyze_n4`'s summary still prefers a mean where the paper now reports a
median. Not on any paper number; noted for the next pass.

### The step that would have been easiest to skip

**Every sweep is detached from the session that launched them**, verified rather than assumed:
each has its own `sess` (51963, 51965, 51969, 51972, 51975, 52328, 52330) and `ppid=4788`, the
systemd user manager, against the launching shell's `74389`. A context clear does not take
them. This is what `systemd-run --user` buys over `setsid nohup`, and `LAB_PROTOCOL.md` has required
it since the 21-hour waiter that outlived its own purpose.

**And one trap worth writing down:** wave 1 was launched without `python -u`, so its five logs
are **buffered and empty until the process exits**. An empty log there is not evidence of a
stalled job — `CPUUsageNSec` is. Wave 2 got `-u` and streams. Next time, `-u` on everything.

---

## 2026-09-21 — Entry 56: R1's number moved, and the two things reading it exposed

Session 18. Six of nine R1 families landed while this session ran; three are still going.
**R1-4 passes on every family scored**, which is what makes the rest readable at all:
`test_paper_numbers.py` recomputes k09's five median/range rows — including n=128's
`2.77`–`476×`, the pair the appendix quotes — and k04's `29` analysable runs over `11`
moduli, from artifacts written at B = 10,000. Identical. Nothing moved that must not move.

### The pre-registered number moved, in the direction nobody bets on

`PREREGISTER_r1_permutation_B.md` named both marginal tests **before** the re-run and
committed to reporting whatever they became, *including 26 or 28*. It became **28 of 29**.

| | B = 200 | B = 10,000 |
|---|---|---|
| n = 49 s2 | `p = 0.010` — **2 draws of 200**, exactly on the threshold | `p̂ = 0.0050` — 49 of 10,000. **Passes** |
| n = 98 s0 | `p = 0.015` — 3 of 200 | `p̂ = 0.0151` — 150 of 10,000. Unmoved |
| n = 63 | `p = 0.0` — i.e. "0 of 200" | `p̂ = 0.0002` — 1 of 10,000 |

**The count is the small part.** §7.2 read n=49's old `0.010` as the statistic *"correctly"*
reporting the absence of a circuit in a run that did not grok. It was not doing that. At 50×
resolution that run's key set is beyond 99.5% of random equal-sized draws — and the run has
window-median held-out accuracy **0.891** with a baseline loss of `1.3e-01`, four orders
worse than any grokked run in the table.

This is not a new phenomenon; it is the one Gate 2 already measures. **G1 — the same
statistic, and the criterion the Gate 2 pre-registration named PRIMARY — passes on 19 of 19
FAILED stratum-measurements.** A model that has memorised much of its training grid still has
a logit tensor that is a function of `u·v` where it was trained. §7.2 now says so and
cross-references F9 rather than claiming the test detected a missing circuit.

**And the session-16 panel overturn is now dated evidence.** That pass established "27 of 29
is correct and the EXCEPTION LIST is wrong", against a reviewer who said 26. That was right
at B = 200 and the count still moved — for a reason the reviewer never raised. Being right
about an artifact is a statement with a timestamp on it.

### The endpoint rule was wrong in four places, not two

`LAB_PROTOCOL.md` carried this as a live defect: `analyze_gate2.final_test_acc` and `analyze_o4`
classify by `hist[-1]`. Surveying for it found **four** independent implementations of the
three-state rule — two reading the last row, two (`analyze_k09`, `plots.grok_curves`) reading
the window correctly. So the fix is one module, `src/analysis/runs.py`, not one patch. Its
self-check plants the C27 case — a grokked run whose final sample is a one-interval excursion
to 0.6798 — and **asserts that the last row would call it FAILED**, so a revert fails the
check rather than passing quietly.

Measured before touching anything, over 162 runs: the two readings disagree on **4**.
`analyze_o4`'s full output is **byte-identical** before and after, which is the check
`LAB_PROTOCOL.md` requires of an extraction and the evidence that no published number moved.

### ⚠️ But LAB_PROTOCOL.md's reassurance covers only half the arm, and this is the finding

The note says *"none of the four is in k03, so Gate 2's 157 measurements, C31 and C38 are
unaffected."* True — and it covers the **primary** arm. Gate 2's **control** arm is built
from **k04**, and `k04_extended/WE_B_thesis_n49_s2` is one of the four:

    last row 0.9631 -> near-grok -> EXCLUDED from the control arm
    window   0.8908 -> FAILED    -> INCLUDED in the control arm

So correcting the endpoint rule **adds a run to the control arm**. "19 failed measurements"
becomes **20**, and every count in F9 moves with it. That direction biases *against* our own
finding — one more failed measurement for G1 to pass on — which is the direction nobody
checks. **It is also the same run §7.2 now reports as a non-grokked model whose key set
survives the ablation test.** One phenomenon, surfacing in two sections, found by asking what
a reassurance did *not* say.

### A third thing, found by asking which artifacts the family table omits

`results/gate2/k04_extended_controls.json` (16 measurable) and `n7_engine_controls.json` (3)
are **not in the pre-registration's 320-test family table** and no sweep was launched for
them. After R1 the project would still hold two **tracked** artifacts whose `p_perm` reaches
`0.0` — the exact display R1 exists to retire — and they are the 19 measurements §8 cites and
F9 draws. `N_CONTROL` is already `10000`, so the fix is two commands and no code change. Both
recorded as deviations 3 and 4 in the pre-registration; neither acted on, because
`analyze_gate2.py` is mid-sweep and editing or double-writing it is the trap deviation 2
already records.

### Also this session

- **`paper/CUTS.md`** — `main.tex` line 1 has referenced it since it was written and it did
  not exist. A proposal: workshop = spine only, ≈6,400 words (nine pages, *not* four), three
  figures, and four things no cut may drop. It names **no number**, so R1 cannot stale it.
- **A4b item 3** — the reproducibility screen over the **14 keys the paper actually cites**,
  not the 53 in `refs.bib`. No exclusion required, which is a finding only because the last
  column was checked: three cited works release no code and two are monographs, and none of
  the five carries a number of ours. Near-miss named: `nguyen2026dlogclock`'s across-seed band
  `0.45`–`0.61` is published but unreproducible from their one-seed notebook, and the paper
  does not quote it (verified by grep).
- **STATE §7's item 1 had outlived itself** — *"lift `K_h = K_W` across strata"* is **C38**,
  pre-registered, run and entered in the ledger **in the same session that wrote the row
  telling the next session to do it**. Recorded as a rule: when §7's table and §2's ledger
  disagree, the ledger wins.

---

## 2026-09-22 — Entry 57: collecting R1 — what survived the extra resolution and what did not

Addendum to Entry 56, same session. All nine R1 families landed; the collection ran in the
order `STATE.md` §7 fixed, R1-4 before any `p` was read.

### R1-4, in the form the pre-registration actually specified

`results/gate2/k03_grid_acts.json` is **tracked**, so for that family R1-4 is a literal diff
rather than a recomputation against the paper. It is the strongest form of the check in the
project:

    29 runs -> 29 · 1,476 strata -> 1,476 · identical keys
    157 measurable -> 157, IDENTICAL SET
    23,710 B-independent values compared at 1e-9 -> 0 MOVED

B-dependent and excluded from that count: `p_perm`, `n_ge`, `n_draws`, and the two
statistics estimated *from* the draws (`ctrl_median`, `resid_ctrl`). For the gitignored
families R1-4 rests on `test_paper_numbers.py`, which recomputes 13 published quantities —
k09's five median/range rows including n=128's 2.77–476, and k04's 29 runs over 11 moduli.
All identical.

**Provenance checked, not assumed.** The artifact stamps `e82ca11` — the commit that set
`N_CONTROL = 10,000`, made before launch — with `git_dirty=False`, an ancestor of HEAD.
HEAD moved **26 files** during the 8.5-hour run; none is imported by `analyze_gate2.py`,
which imports only `src.analysis.{ablation,neurons,transforms}` and `src.provenance`. Worth
recording: `provenance.stamp()` is called at the **top of main()**, so it records the tree
at *script start*, not at write. That is why editing the paper during the run was safe here
and is **not** safe during a training run, which re-stamps every snapshot.

### The absolute claims split three-two

| family | at B = 200 | at B = 10,000 |
|---|---|---|
| gate2 (157) | 0 of 200 in **all 157** | 0 of 10,000 in **133 of 157**; `p̂ ≤ 0.0023` in all 157 |
| k03 (31) | not one of 200, any run | not one of 10,000 in **26 of 31**; the other five beaten by **1 or 2 draws**; `p̂ ≤ 0.0003` |
| k04 (29) | **27** of 29 | **28** of 29 |
| k08 (24) | 24/24 at p < 0.01 | 24/24, and **all 24 beyond every draw** — survived intact |
| k09 (13) | 13/13, `p < 0.005` | 13/13 **at the floor** — survived intact |

The pre-registration says in advance that `b > 0` at 10,000 draws is resolution, not
falsification, and that is what this is: the five k03 runs beaten by 1–2 draws sit at
0.0002–0.0003, far below anything 200 draws could resolve. **But the absolute phrasing does
not survive, and three documents asserted it.** Every site is now amended and names the
number it replaced.

### Two traps caught during the collection, both in the false-pass direction

**1. Routing `analyze_gate2` to the window rule would have broken `check_split`.** That
function compares a recomputed held-out accuracy against the run's logged accuracy at a
0.01 tolerance — and `logits_all` is the **last checkpoint's** output, so it genuinely wants
the last row. Feeding it a window median would have made ordinary post-grok jitter read as
*"split does not reproduce"* and **skipped good runs**, silently, while the diff looked like
a defect being fixed. `src/analysis/runs.py` now exports `last_test_acc` beside `state`, and
its self-check asserts the two **disagree** on the planted C27 history. Two questions, two
names.

**2. Arm membership, measured before running anything.** k03 **0** runs change arm, so the
artifact committed at `4625ded` stays valid; n7_engine **0**; k04_extended **1** —
`WE_B_thesis_n49_s2`, near-grok by last row (0.9631), FAILED by window (0.8908), so it
**joins the control arm**. Deviation 4 demonstrated rather than predicted.

### And a stale number six days old, inside a VERIFIED row

`STATE.md`'s C33 carried **`G2 excluded/baseline 3.98e+02–1.0e+08`**. The 2026-09-16
correction to **2.71e+08** reached `FINDINGS.md` and the paper and **never reached the
ledger**. `test_paper_numbers.py` has asserted 2.71e8 the whole time — against the paper,
not against `STATE.md`, which is exactly the gap that let it live. Re-derived from the
artifact: max **2.707e+08**.

### What is still running, and why it was restarted

`analyze_gate2.py results/n7_engine` (26 tests) was launched as a session-bound background
job at 00:00 and was **9 minutes in** when the session wrapped. It was **killed and
relaunched under `systemd-run --user`** (`r2-gate2-n7`, `MemoryMax=2G`, log
`logs/r2_gate2_n7.log`, gitignored) rather than left to die on the clear. The session-end checkpoint rule
is explicit and it is the right one: *a job that dies on clear is worse than one that was
never started, because `STATE.md` will claim it is running.* Cost, 9 minutes of recomputed
draws; `ppid=4788` verified after relaunch.

---

## 2026-09-22 — Entry 58: finishing R1 — the criterion that failed, and three false passes

Session 18. Continues Entry 57, which left one detached job running. Everything below is
local CPU; **zero Kaggle quota spent, no kernel pushed.**

### The brief reported an idle box while a job was 20 minutes in

the session-start brief's §3 printed **"no local jobs running"** while `r2-gate2-n7` was 20 min in
at 100 % CPU. The regex whitelisted three script prefixes — `run_ | kernels/ | scripts/` —
and `analyze_gate2.py` lives at the **repo root**, matching none. **Fourth false idle in
this family**, and the comment directly above that line records the 2026-09-12 one (four N7
workers, 10.5 h into a 20 h sweep), which had been fixed by *widening* the whitelist.

Widening it again only moves the next miss to the fifth prefix, so **the whitelist is
retired**: match any `.py` under our interpreter. `python -c` and `python -m pip` carry no
`.py`, which is the only thing the whitelist ever bought. Five planted argv cases — one per
shape actually missed — assert it at import, so a future narrowing makes the brief refuse to
run rather than report an idle box. `5baddf9`.

### All three Gate 2 arms collected, one at a time

| arm | wall | R1-4 |
|---|---|---|
| `n7_engine` (`e8dbd97`) | 47 min | 468 B-independent values, **3 moved** — see below |
| `k04_extended --controls` (`cbe13f3`) | 2 min 10 s | 2,106 values, **1 moved** (the added run's seed count) |
| `n7_engine --controls` (`bbc9b85`) | — | 109 values, **0 moved** |

Each committed before the next started, because two concurrent writers of tracked artifacts
make the second stamp `git_dirty` against the first's uncommitted write. `scripts/run_gate2_arms.sh`
runs all four in one process and is therefore **the wrong tool for this**; it was not used.

**The science did not move.** All 66 engine verdict flags are bit-identical to B = 200; both
grokked measurable sets are identical (157 + 26 = 183); and R1-4's own named quantities —
`baseline`, `restricted`, `excluded` — are **606 values across four tracked artifacts, 0
moved**. §8's engine paragraph was **checked and needed no edit**: 7 of 7 non-unit strata
carry a local clock, and n=119's unit stratum still fails G1 in exactly one of two seeds
(seed 1 `p̂` 0.00010 PASS, seed 2 0.01080 FAIL). **No tracked artifact prints `p_perm = 0.0`
any more** — deviations 3 and 4 closed.

### ⚠️ R1-1 IS NOT MET, AND THE PREVIOUS ENTRY SAID IT WAS

**171 of 200 previously-`p = 0.0` tests still read `b = 0`: 85.5 % against a pre-registered
≥ 95 %.** Per artifact: k03 133/157, engine 19/24, control arms 16/16 and 3/3.

Entry 57's Outcome reads *"R1-1 / R1-2 hold: every landed artifact reads `n_draws = 10000`,
and the minimum `p̂` is exactly `1/10001`."* **That is evidence for R1-2 and for nothing
else.** R1-1 is a claim about how many previously-zero tests *stay* at the floor and was
never computed until this session. Left alone it would have shipped a pre-registered
criterion as met on a different criterion's evidence.

**The pre-registration contradicts itself, and both halves are now quoted in it.** The
criteria table sets ≥ 95 %. The falsification section says in advance that `b > 0` is *"not
a falsification: it is the resolution the old B could not deliver"*. Prediction 1 is
narrower still, promising the floor only where the excluded loss exceeds the control
**maximum** by orders of magnitude — a qualifier the table drops.

**Reported as NOT MET.** Nothing is retracted: all 29 tests that resolved off the floor sit
at `p̂ ≤ 0.0023`, and only n=49 crossed a decision threshold — downward, as prediction 2
called in advance. But reading the falsification clause as retroactive permission to move
the bar is the C6/C7 ordering error, and a criterion that can be reinterpreted after seeing
the number is not a criterion. `70294e9`.

### 19 failed measurements → 20, and it cost a headline

The corrected endpoint rule admits `k04_extended/WE_B_thesis_n49_s2` to the control arm
(last row **0.9631** → window median **0.8908**), predicted in Entry 57 before any rate was
recomputed. **It is not only a denominator.** That run has exactly **one** measurable
stratum — the unit stratum; its other three sit below the |G_m| ≥ 10 floor — and that
stratum passes **G0, G1, G2, G4 and G5**. So four of six F9 rows move:

    G0  0/19 -> 1/20        G3  2/6  -> 2/6 (unchanged)
    G1 19/19 -> 20/20       G4 12/19 -> 13/20
    G2  0/19 -> 1/20        G5  0/13 -> 1/14

**"G0, G2 and G5 separate completely" is retired.** They separate at 1/20, 1/20 and 1/14.
It is the same run §7.2 already described (baseline 0.1256, window median 0.891), so the two
sections were always about one phenomenon; this is that run entering the control arm
formally. `062b04f`.

**⚠️ The run is a genuine borderline: 0.009 below `FAIL_ACC = 0.90`, on a *rising* window
(0.842 → 0.963) — still learning when the budget ended, not plateaued.** Researcher decision
2026-09-22: **keep 0.90.** The defence is **ordering, not margin** — 0.90 is the C27
excursion floor from `932ef89` (09-15), centralised in `37dba23` (09-21), and this arm was
not scored until `cbe13f3` (09-22), a week later. A threshold fixed before the measurement
is inherited; one moved after seeing which side a run falls on is C6/C7 again. The
sensitivity sentence **stays** in §8: keeping the gate is not a reason to stop reporting
that 0.89 would restore 0/19. `26fcff9`.

### Three false passes, all found by running a check against something that should fail

**1. The committed engine artifact compared TEST against TRAIN accuracy.** Three
`split_logged` values moved `1.000000 → 0.999596 / 0.996570 / 0.997756`. Not a measurement
change: the old code read `hist[-1][3]`, which is **TRAIN accuracy on the engine and TEST
accuracy in the kernels**. `check_split` was comparing recomputed test against logged train
and **passing only because both sat inside its 0.01 tolerance.** Verified against the npz:
`hist_cols = [step, train_loss, test_loss, train_acc, test_acc]`, `hist[-1][3] = 1.000000`
where `hist[-1][4] = 0.999596`. `split_recomputed` is unmoved in all three. Recorded as
deviation 6, because *"R1-4 failed"* and *"R1-4 failed for a reason that is not B"* are
different findings.

**2. `scripts/r1_diff.py` skipped the runs it was written to compare.** k04's control arm
went 6 runs → 7; the walk logged a length mismatch and **returned**, so it compared 48
values — all from `verdict` — and printed a verdict of its own. **The six runs common to
both versions were never compared, and nothing in the output said so.** Lists are now
aligned by identity, `(n, seed)` and `(d, e)`; the same artifact compares **2,106** values,
a 44× increase. `2b920ee`.

Two more caught by running it against the **unchanged** k03 artifact first, where the answer
must be "0 moved": three `acc_heldout` values read as MOVED because **NaN ≠ NaN**, and it
walked `provenance`, which a re-run legitimately restamps. k03 passed that only because its
file was untouched. Both planted in `--selfcheck`.

**3. My own §8 sentence had the threshold direction backwards.** It said a threshold "0.01
higher" would move the run out of the arm. `state()` returns FAILED when `acc < FAIL_ACC`,
so 0.91 leaves it failed. Settled by **planting the value, not by reasoning**: 0.88 and 0.89
give near-grok, 0.90 and 0.91 give FAILED.

### Two stale documents, opposite directions

- **LAB_PROTOCOL.md carried "⚠️ LIVE DEFECT, NOT YET FIXED" for code that is fixed.**
  `analyze_gate2` routes through `src/analysis/runs.py` (`43df124`) and `analyze_o4.py:93`
  reads `run_state(z)[0] == "grokked"`. Verified by grepping `hist[-1]` across the tree:
  every survivor is a `kernels/*/run.py` progress print or `final_test_acc` field, where
  index 3 **is** test accuracy. **This is the 2026-09-12 `analyze_k03.spectral` lesson
  inverted** — that banner described a defect nobody had fixed, this one described a fix
  nobody had recorded — and both cost the next session real work. `fead099`.
- **`test_paper_numbers.py` scored the artifact while FINDINGS asserted the same numbers in
  prose, unchecked.** Exactly the gap that let STATE's C33 row carry a retired value for six
  days. It now parses FINDINGS' criterion table row by row against the scorer. **Negative
  control run rather than assumed**: planting a stale `0/19` makes the suite exit 1 and name
  the row. That negative control also caught my argument order — `chk(label, got, want)`
  prints `paper={want} artifact={got}`, and I had the document in `got`, so the planted stale
  row printed under "artifact". **Fourth upside-down column in this project, and the first
  caught by running the check against a failure.** `c0680cf`.

### Closed out

`CUTS.md` **approved by the researcher** (`09f86e8`), with the two exclusions written into
the file so a later reading cannot widen it: it covers Cut 1 only, and it does **not**
authorise building `main_workshop.tex`. Its two measurements were re-derived from the
commands it records (`339132c`): 21,480 → **21,843** words, and the only file that grew
which the cut **keeps whole** is `07-results-causal` (+351, 1512 → 1863) — R1 rewrote the
causal section. So the body estimate is **≈ 6,750**, not ≈ 6,400, and neither number has
ever been measured on an assembled variant.

`render_all.py` complete, all 13 cross-panel figures redrawn. **F9 was opened and read**: it
prints `1/20 · 1/20 · 1/14 · 2/6 · 13/20 · 20/20` with the legend "FAILED runs (20
measurements)", so figure and caption agree. The 33 `hist_cols` KeyErrors in its log are the
known k02 gap. `paper/overleaf_bundle.zip` rebuilt — 13 figures, 14 sections, provenance
`ba46e96`, **no DIRTY flag**, F9 inside md5-identical to the rendered file.

**Final state: 8/8 self-checks, 230 assertions, `check_tex` 0 failures, tree clean, remote
matching, no orphan waiters, all four systemd units `dead / success`.**

## 2026-09-22 — Entry 59: the intervention that needed a retrain, and three stale copies of one number

Session 19. Researcher instruction: run the internal intervention, make it TMLR-ready.
**Zero Kaggle quota; ~7 h local CPU** (3 × 5.0 h retrain, 3 × 1.9 h measurement).

### The experiment could not be run on anything on disk

STATE §7 item 2 scoped it as *"re-run the forward pass on saved weights"*. There are no saved
weights. `run_n7.py:171` deletes the resume checkpoint on a clean finish — *"resume state is
dead weight"* — and `snapshot()` only ever wrote `W_E` and `W_U`. Swept all 232 artifacts:
**0 carry `W_Q`/`W_K`/`W_V`/`W_O`/`W_in`/`W_out`/`W_pos`.** Every completed run in this
project is unrecoverable as a model. The only full checkpoints that survive are three from
runs that were **killed**, in the retired pre-leak-fix arm whose README forbids analysing them.

Fixed at the one place all callers route through (`cb3f941`): `snapshot()` saves all nine
matrices, named `p_*` to match `checkpoint()` so one loader reads both. Smoke-tested by
loading them into a model built with a **deliberately wrong init seed** — reproduces the saved
logits to 9.5e-07, which is float32 storage rounding. The wrong seed is the point: it makes
the agreement attributable to the weights and not to the constructor.

### Retrain rather than reuse, because it buys a positive control

Three seeds at n=113. The training path was checked **inert** between the archive's SHA
`f9d15b8` and HEAD before launching — the only diffs are a stamped `dtype` key, a `DTYPE`
default equal to the old hard-coded `float64`, and a NEP-50 `.astype` that is inert at
float64; `algebra.py`'s additions are not imported by any training file. Result:
**bit-identical**. `max|ΔW_E| = max|ΔW_U| = max|Δlogits| = 0.000e+00`, grok steps
4,500 / 6,500 / 9,700 recovered exactly. So C39 describes the models C22 and C31 already
measure.

### C39

Deleting the key characters from `W_E` and running forward returns the network to **chance**
— acc 0.0089/0.0089/0.0091 against 1/112 = 0.00893. Keeping only them leaves **1.0000**.
excluded/baseline median 1.543e+08 (range 3.895e+06–2.780e+08, n=3). All three inherited
criteria hold on 3/3 seeds against a pre-registered 2/3.

Inheriting every threshold was deliberate. G2's 100×, `FAIL_ACC`'s 0.90 and C22/C23's
p < 0.01 were all fixed for other purposes before this run existed. It is the only defence
against the C6/C7 ordering error that does not rely on my own restraint.

### ⚠️ The null's upper tail reaches the effect

Seed 2's control **maximum exceeds** the excluded loss; seed 0's reaches 99.2 % of it.
`excluded / median(control)` reads 8.2e+07 and would have hidden it completely — the
control-mean trap that retired C22/C23's ratio, surfacing in a new arm. **The permutation p
is the only reported statistic that shows it.**

The cause is the design, and it is conservative: the inherited pool includes the key
characters, so a draw can partially perform the intervention. **Exploratory, written after
seeing the tail and labelled so everywhere**: damage is monotone in overlap (0→1.26e-05,
1→4.95, 2→10.3, 3→16.5, 4→23.9, 5→30.0 at seed 0), and a draw with **no** key character does
nothing — 1.04–1.11× baseline, **12,468 such draws across three seeds and none above
1.05e-04** against an excluded loss of 24–47. That is direct evidence for the reading F4's
caption already gave for the bimodality.

### Three stale copies of one number, and the guard that had been looking the wrong way

R1 moved k04 from 27 of 29 to **28 of 29** with a **single** exception (n=98, 0.0151). The
correction reached the abstract and `FINDINGS.md`. It did **not** reach
`13-conclusion.tex` (*"27 of 29 ... the two exceptions"*) or the appendix ledger's **C23 row**
(`$27/29$`). `test_paper_numbers.py` has asserted 28 against the `.npz` the whole time —
**which is exactly why it never fired.** Fifth and sixth instances of the defect that let the
C33 row go stale twice: scoring the artifact while the prose drifts.

The guard now scores the three documents that assert the count. Negative-controlled —
restoring the stale sentence fails all three checks and exits 1. **Its own first version
reported the abstract as ABSENT**, because the abstract wraps `$28$\nof $29$` across a line
break; a literal match whose failure mode is indistinguishable from the defect it hunts.
Whitespace is flattened first, and the ledger is matched on `$28/29$` because it states a
fraction rather than a phrase.

### Citations, screened by reading rather than by profile

Five 2026 preprints read **end to end** (13.8k/12.0k/9.2k/9.4k/5.3k words). `graphify-out/`
could not have answered this — its own report records ~3,850 words across 60 PDFs, i.e.
abstracts. **None is cited and none needs to be: all five study multiplication only where it
is a GROUP** (p = 59, 97, 43/67/97, plus S5), with zero full-text hits for CRT, stratum,
monoid, non-invertible or discrete log. Bibliographies checked locally: 9 of `2607.06639`'s
23 references are papers we hold and **all nine bind exactly** — evidence against fabrication,
not evidence the artifacts exist, which is why none is relied on.

Two peer-reviewed papers were **already in `refs.bib`, uncited**, and both are now cited:
`charton2024learning` (ICLR 2024) — transformers computing gcd predict every pair sharing a
gcd identically **including where wrong**, so the equivalence structure is in the computation,
which is Consequence 2 of our stratum theorem on a different task — and `demoss2025complexity`
(Physica D 2025), which anchors C24's drift criterion to a reviewed source. The dense-spectrum
claim was **unattributed**; it is Furuta (TMLR 11/2024) §5.2, verified in the PDF.

**Final state:** 8/8 self-checks, **285** assertions (was 232), `check_tex` 15/114/171/65 with
0 failures, tree clean, both systemd units `dead/success`.

## 2026-09-22 — Entry 60: addendum to Entry 59 — the figure that corrected the prose

Same session 19, after Entry 59 was written. **Zero quota; all local.**

### F10, and a claim the figure retired

`src/viz/plots.py::internal_intervention` draws C39 in two panels: the three arms per seed
with `excluded` direct-labelled by its **accuracy** (0.0089 *is* chance, which is the claim),
and all **30,000** null draws pooled over seeds, divided by each run's own baseline so three
baselines two orders apart share one axis, split by key-character overlap.

**Rendering it changed FINDINGS and §7.** Both said the dose-response was *"clean monotone"*.
The panel shows **overlap-1 draws spanning four orders** — a single key character can be
worth almost everything or almost nothing. It is monotone **in the mean**; the within-level
spread is wide. Corrected in FINDINGS §3.3b, in §7 and in the caption, with the reason
recorded in FINDINGS rather than silently edited. *This is the "render before concluding"
rule firing on my own prose, in the same session that wrote it.*

**Four layout defects, none visible in the code:** accuracy labels underneath the legend; a
footnote overlapping both the tick labels and the provenance stamp; the per-level counts
positioned against a `ylim` read **before** the limit moved, so they were clipped
off-canvas; four decades of dead headroom. All found by opening the PNG.

The figure **asserts before it draws** — median ratio 1.543e+08, restricted accuracy > 0.999,
excluded accuracy within 5e-4 of 1/112, G2's 100× bar, and that no zero-overlap draw reaches
the effect. If the artifacts stop saying what §7 says it raises rather than drawing a
plausible picture of the wrong thing.

### The completeness check that was wrong before the wiring was

Wired into `render_all.py` and verified by **diffing the two lists**, not by having just
edited one: **14 = 14**, no orphans either way. Its own first version reported
`basis_comparison` as unwired — the pattern required `", ` on one line and that entry wraps
across two. A false positive in the checker, and precisely the shape that let
`basis_comparison` genuinely go unwired for a session.

### Bundle

`paper/overleaf_bundle.zip` rebuilt: **2.0 M, 14 figures, 14 sections**, provenance
`8ed016e`, **no DIRTY flag**, F10 inside md5-identical (`0aca8788…`) to the rendered file.
The script derives its figure list from the `.tex` sources and refuses to build with any
missing, so it cannot ship a blank box.

### The guard caught its own author, twice

`test_paper_numbers.py`'s notebook-entry-count check (added at `1c6c53b` this morning) fired
on my own Entry 59 append — 284/1. First time a guard here has caught the session that wrote
it. **It will fire again on this entry**, which is the intended behaviour, not a defect.

**Session totals: 232 → 289 assertions**, `check_tex` 15/115/173/65 with 0 failures, 8/8
self-checks, 40 scripts covered.

## 2026-09-23 — Entry 61: the public release designed and built, and I1b in flight

Session 20. Researcher instructions, in order: run the composite-modulus intervention;
design artifact hosting; no tooling or author metadata anywhere in the public repo; include
a TMLR AI-use statement; then remove it. **Zero Kaggle quota.** All compute local CPU.

### I1b — pre-registered, launched, running

`experiments/PREREGISTER_i1b_composite_intervention.md`, committed `712979d` **before**
launch. n = 121 = 11², three seeds, `results/i1_internal/`.

**Why 121 and not 125.** Both are prime powers with cyclic unit groups, but n = 125 seed 1
is the project's FAILED negative control (`state()` reads 0.5053), so 125 offers 2 grokked
seeds against 121's 3/3. Checked, not assumed: `state()` on all six artifacts.

**Why this is not n = 113 again.** 113 is prime, so I1's "non-unit rows are passed through"
clause covered exactly one row. At 121 it covers **eleven** — 0 plus ten nilpotents — and
n = 121 has three J-classes with a **non-regular** class at d = 11. The intervention deletes
the unit characters while leaving the network eleven untouched embedding rows and a whole
non-regular stratum. That is the case the paper's title claims.

**Every threshold inherited from I1 verbatim** — G2's 100×, `FAIL_ACC`'s 0.90, C22/C23's
p < 0.01, B = 10,000, 2-of-3 seeds. Re-tuning one now, in either direction, would be the
C6/C7 ordering error with an extra step.

**`run_intervention.py` is unchanged, and the pre-registration makes that a criterion**: the
two arms are comparable only if one instrument produced both. A per-run measurement of the
eleven non-unit row norms was **cut** from the analysis plan — `_selfcheck` already asserts
that pass-through structurally, and a second answer to a question the instrument guarantees
is how C22/C23 nearly acquired two key sets.

Archived the three n=121 originals to `results/_archive_i1/` and committed their md5s
(`*.npz` is gitignored, so MD5SUMS is their only provenance). Training path checked inert:
the two sweep SHAs `f9d15b8` and `cafbb36e` have an **empty** diff over `src/`, `run_n7.py`
and `scripts/run_n7_all.sh`, and their diff to HEAD is the same four files I1 proved inert
empirically by reproducing n=113 bit-identically.

Smoke-tested at n=121 before arming: selfcheck OK; 11 non-unit rows **and** the `=` token
bit-identical through `character_project`; unit rows moved. Chance here is **1/110 = 0.00909**,
not 113's 1/112 — writing the prime's floor would misread a correct result.

**Status at session-end checkpoint: RUNNING.** 1 h 08 m in, step 6,000/40,000, ~510 ms/step, eta ~5.0 h.
Seed 2 **grokked at step 5,000**; seeds 0 and 1 at test acc 0.134 / 0.095 and climbing.
460 MB/run, swap 0. The step-0 log said "eta 11.8h" — that is the step-0 ETA trap.

### The public release: designed, amended twice, built

`docs/PUBLIC_RELEASE_DESIGN.md` is the plan of record. Eight decisions; the ones that cost
something to decide:

**The artifact problem, measured rather than guessed.** 244 npz / 4.64 GB, of which
`mlp_acts` is **70.5 %** and `logits_all` **22.6 %**. Dropping `mlp_acts` takes the deposit to
1.27 GB — but makes C31 and the neuron decomposition retrain-only, **and it is not
regenerable** for the 195 files that carry it, because they predate `cb3f941` and have no
saved weights. Decision: deposit everything, one Zenodo record, one DOI.

**D3 — fresh `git init`, and what it costs.** 230 of 262 commits carry tooling trailers; a
single clean commit is the only thing that removes them all. The price is the
**pre-registration timestamps**, which are this project's C6/C7 defence and are verifiable
only from history. The README says so outright rather than letting the structure imply a
guarantee it cannot give.

**D8 — artifact SHAs do not resolve in the public repo.** Found while checking whether
`provenance.stamp()` leaks identity. It does not — no user, no hostname, relative paths. What
it carries is a `git_sha` from the development history, and **every** rewrite renumbers
commits, fresh init and filter-repo alike. Verification is untouched (`reproduce.sh`
recomputes every number from the deposit); provenance-by-checkout is not available.

**⚠️ 68 of 250 artifacts carry `kaggle_account: account-a` in their provenance
stamp.** This is *inside the deposit*, where no repo-side scrub reaches, and the account name
contains the surname. **Left untouched pending the researcher's decision**: editing a
provenance record to conceal identity falsifies the one thing provenance exists to
guarantee, and §11 of the paper documents that field deliberately. Repo-side the two accounts
are relabelled `account-a` / `account-b` — which account spent which quota is load-bearing.

### The AI-use statement, written and then removed

TMLR's policy was **fetched, not recalled**: *"LLMs may be used as general-purpose assistive
tools. Whichever tools are used, authors are fully responsible for content on which they are
listed as (co-) authors"*, and *"LLMs are not eligible for authorship."* **No disclosure is
required.** A voluntary statement was drafted (`18cf894`) and removed on that basis
(`3b3193c`), recorded as amendment 2 and **scoped**: the decision turns on TMLR's policy and
nothing else, so a venue that mandates disclosure re-opens it.

**Its draft was wrong and the appendix caught it.** It asserted "three of our own claims are
retracted, two of them" to a size confound. The record says **five** withdrawn and **three**
to that shape. The check it introduced was **kept**: it counts the appendix's own
`\paragraph{C..}` entries (C19 C20 C7b C27 C30) and requires the prose to match — structural,
not prose-vs-prose. Negative-controlled: drifting to "Four" fails one check, planting a sixth
entry fails two.

### The export, and six defects only looking could find

`scripts/make_public_export.py` writes the tree; `scripts/check_public.py` gates it. Neither
ships — they name every pattern the gate greps for.

First gate run: **41 violations, every one real.** Six rounds later: **2**, both correct — the
deposit DOI does not exist and the gate refuses to ship it unfilled.

1. **`uthor{}}`.** In an `re.sub` **replacement template** `\a` is the BELL character, and
   `[^}]*` stopped at the brace inside `\texttt{[affiliation]}`. **The gate passed it**:
   mangled LaTeX contains no forbidden pattern. The gate now runs the export's own
   `check_tex.py` — a defect class the pattern checks structurally cannot see.
2. **`PLACEHOLDER` as a sentinel collides with English** — `LAB_PROTOCOL.md` legitimately
   contains "IS A PLACEHOLDER". Now `the unfilled-DOI sentinel`.
3. **A plural survived a `\b<word>\b` substitution.** Plurals first — and a lesson that
   quotes the token it is about is itself rewritten by the pass it describes, so write these
   with placeholders.
4. **A generated file is not in `git ls-files`**, so `LAB_PROTOCOL.md`'s own self-references
   went unrenamed. Per-file edits had that gap by construction; one substitution table
   applied to every file cannot.
5. A stripped the Author field field left `2026-09-22· **Script:**` with no space.
6. `check_tex` "failed" on the export because a relative tree was passed as `cwd`, doubling
   the path — and stderr was swallowed, so it reported an **empty** error.

**The manifest guard fired on its first outing.** `write_md5sums` refuses to run while a
training job is live; the n=121 retrain is writing snapshots, so the 250-artifact manifest
just generated was already stale. Sequencing step 0, in code rather than in memory.

**`KNOWN_ABSENT`**, recorded as a judgement call: this notebook is a historical record and
names documents that existed when it was written. Rewriting it to pretend otherwise would
falsify the record, so the gate carries an explicit allowlist with a reason per entry — and
**reports** any entry nothing references any more, so it cannot rot into a blanket waiver.

**Verified:** 5 planted violations all rejected, unplanted tree accepted as a positive
control **first**; `check_tex` 15/115/173/65 clean **on the export**; five self-checks green
from inside the export tree. 293 assertions, 0 failures. Nothing about self-containment is
claimed beyond what was run — the bare-clone test needs the deposit and has not been done.

## 2026-09-23 — Entry 62: the audit, and I1b lands at a non-square-free modulus

Session 21. Researcher instruction: *"run the most mathematically rigorous audit you can"*,
then apply the resulting patch and run the I1b measurement. **Zero Kaggle quota.** Local CPU
only: the I1b retrain (3 × 5.92–5.97 h, launched the previous session) and the I1b
measurement (3 × 128.0–128.2 min, 3-way parallel under `systemd-run --user`, `MemoryMax=6G`,
249 MB/run measured).

### The audit ran entirely out-of-tree, on purpose

`provenance.git_dirty()` reads `git status --porcelain`, which counts **untracked** files,
and the I1b retrain re-stamps its `.npz` every snapshot. So every audit script was written to
a scratch directory outside the repo and the patch was staged in a **clone**. The tree read
`0 uncommitted` for the whole eight-hour audit, and all three retrain artifacts stamp
`git_dirty=False`. This is the discipline F4 below says was not kept during R1.

### What independently verified — the algebra is solid

- **C1/C2 three ways.** Green's J-relation by brute force over all pairs (n ≤ 80); von
  Neumann regularity `∃x: axa = a` by brute force against `gcd(d, n/d) = 1` (n ≤ 200); the
  idempotent test. All three agree. Non-regular counts **0, 0, 1, 2, 8** exact; 165/143/154
  all-regular; square-free ⇒ all regular exhaustively to 500.
- **The J-class indicator spectrum is a RAMANUJAN SUM.** For the mean-centred indicator
  family, the sine part vanishes identically and
  `e_k² = Σ_{m|n, m>1} c_m(k)²` with `c_m(k) = μ(m/g)·φ(m)/φ(m/g)`, `g = gcd(m,k)`. This
  closed form reproduces `freq_energy(jclass_indicators(n))` to **~1e-13 across 14 moduli** —
  an independent confirmation of the bin convention that has cost this project four bugs.
- **§9's amendment is right.** In closed form `P(n) ⊆ support` **strictly** at 120/165/119/63/
  105/147 (|P| 7 vs |support| 60 at n=120), equal only at the prime powers where the law is
  vacuous. "The indicators do not *select* P(n)" is correct.
- **Multiplicity is handled better than the audit's own first draft.** `05-methods.tex:318`
  enumerates a family of 320, clears BH at α=0.01 and Bonferroni at α=0.05, and **explicitly
  reports failing Bonferroni at α=0.01**, naming the required `B ≥ 32,000` and the cost
  reason. The audit's multiplicity finding was withdrawn.

### Seven defects. NO VERDICT MOVED. Two reach the paper. Committed `266b77d`

**F1 — the paper's stated estimator was not the one that ran.** `05-methods.tex:295` defines
`p̂ = (1+b)/(B+1)`, calls the naive `b/B` *"a trap we fell into"*, and line 315 says the same
logic runs the §9 enrichment test at 20,000 draws. Three paths still computed `b/B`:
`test_crt_law.py:57` (C8), `analyze_k04.py:102` (C9), `test_crt_null.py:119` (C28). R1's
migration reached `analyze_n4`/`analyze_gate2`/`run_intervention` and stopped. **b = 0 at
every affected test**, so the estimators differ by 5e-5 and the published `p < 5e-5` bound
holds under add-one (floor `1/20001` = 4.99975e-5). `STATE.md`'s C28 row reported
`p_B = 0.000` at B = 2,000, where the floor is 5.0e-4.

**F2 — the number guard asserted a p-value EQUALS 0.0.** `test_paper_numbers.py:196` read
`chk(..., _p, 0.0)` while line 285 of the same file documents deleting that exact assertion
elsewhere as unreachable. The guard was pinning the value the methods section calls
impossible. Same family as the `analyze_k03.spectral` entry: a fix applied in one place and
inverted into a lock in another.

**F3 — `argsort(argsort(x))` is an ORDINAL rank, not Spearman's.** With ties it breaks them
by array position. **ω takes three distinct values over 23 moduli** (9/11/3), so the nine
ω = 1 moduli were ranked in *n*-order.

| quantity | ordinal (was) | midrank (is) |
|---|---|---|
| ρ(ω, Gini_add) | +0.762 | **+0.911** |
| ρ(ω, zdd) | +0.571 | **+0.725** |
| ρ(cyclic, Gini_add) | −0.689 | **−0.701** |
| W2 ρ(ω, G │ zdd) | +0.627 | **+0.804** (HELD, ≥ +0.50) |
| W3 ρ(zdd, G │ ω) | +0.735 | **+0.578** (descriptive) |
| W5 ρ(n, ω) | +0.172 | **−0.105** |
| C9 ρ(zdd, Gini_add) | 0.794 | **0.786** (HELD, > 0.6) |

**§9 asserted *"ρ(zdd,G│ω) = +0.735 sits ABOVE ρ(ω,G│zdd) = +0.627"*. Correctly ranked that
is the same two numbers REVERSED.** W1 (PRIMARY, sign test), W4 and W6 are rank-free and do
not move; W2 and W5 keep their verdicts. **No threshold was touched** — midranks are the
*definition* of Spearman under ties, so this is a defect fix, not a criterion move. The
paragraph's conclusion ("two partially independent predictors, this data cannot rank them")
is preserved and now rests on the legs that survive, and says outright that a ranking whose
direction depends on a tie convention is not a ranking. C9's only real tie: **n = 63 and
n = 147 both have zdd exactly 3/7** (27/63 = 63/147).

**F4 — the artifact behind the abstract's "28 of 29" is `git_dirty=True`.**
`results/k04_extended_n4_ablation.npz`, `0630186`, 2026-09-21T16:38. All **eight** sibling R1
artifacts are clean. Entry 59 records fixing exactly this on 2026-09-11 (*"both k03 and k04
now `ccb590b`, dirty=False"*); the R1 re-run reintroduced it for k04 alone. **NOT FIXED —
it needs a re-run on a committed tree, not a diff**, and the artifact is gitignored so it
must be copied first or the re-run destroys its own comparison.

**F5 — the documented provenance gap undercounted, 62 → 68.** `LAB_PROTOCOL.md` said *"ALL OF k02
and k03"*: true (31 + 31) but incomplete. `results/k01_scout/*.npz` (6) are unstamped **and
claim-carrying** — `test_paper_numbers.py:195,457` reads them for F6's 14.08×/11.76×/3.30×
and §9's network-key-set claim. `gate1` (3) and `k00` (2) are unstamped and claimless; the
`ckpt_*.npz` are transient resume files, since confirmed deleted on clean finish.

**F6 — `provenance.read()` raised on 11 of the project's own artifacts.** Every kernel
`*_summary.json` is a JSON **list**; `d.get("provenance")` threw `AttributeError`. A reader
that crashes on eleven of the files it exists to audit cannot be used to audit them.

**F7 — a stale value asserted in the present tense.** `09-results-crt.tex:175` and
`FINDINGS.md` said `ρ(n, Gini_add) = +0.015`; that is the **18-modulus** figure, reproduced
exactly on the 18-subset (+0.0155). At 23 moduli it is **−0.185**, which the pre-registration
amendment and `STATE.md` already carried. HELD either way (|ρ| ≤ 0.40).

### The lesson is the self-check, not the statistic

`analyze_omega._selfcheck` scored `spearman` on `[1,2,3,4]` and `partial_spearman` on
continuous Gaussians. **Ties have probability zero there, so it could not have failed on a
tie bug.** Sixth instance in this project of a check whose failure mode is unreachable.
`src/analysis/stats.py` now owns `rankdata`/`spearman`/`partial_spearman`/`perm_p` — six
duplicated rank lambdas and three divergent estimators collapsed into one module — and its
self-check **plants a tie**, pinning the ordinal form at a spurious **ρ = 1.000** for a
two-level grouping variable against twelve distinct values, and its swing to **0.510** on a
pure reordering of tied entries while the midrank does not move at all.

**And nothing scored any ω number**, which is how F3 and F7 drifted at once.
`test_paper_numbers` 293 → **302**: population = 23, W2/W3/W5 against the artifacts, the
partial ordering, and that §9 carries −0.185 and no stale +0.015.

### I1b — the internal intervention HOLDS at n = 121 = 11²

Full scoring in `experiments/PREREGISTER_i1b_composite_intervention.md` § Outcome, C40 in the
ledger. Retrain 3/3 grokked (10,700 / 10,300 / 5,000), all three `git_dirty=False`, all three
reproducing the archive **bit-identically** (`max|ΔW_E| = 0.000e+00`).

| seed | baseline | restricted (acc) | excluded (acc) | excluded/baseline | `p_perm` |
|---|---|---|---|---|---|
| 0 | 1.7482e-06 | 3.4061e-07 (1.0000) | 5.0365e+01 (0.0000) | 2.881e+07 | 0.00010 (0 of 10,000) |
| 1 | 7.7458e-03 | 1.0560e-01 (0.9596) | 5.7783e+01 (0.0120) | 7.460e+03 | 0.00010 (0 of 10,000) |
| 2 | 1.6364e-03 | 2.2674e-03 (1.0000) | 1.8467e+01 (0.0000) | 1.128e+04 | 0.00030 (2 of 10,000) |

`excluded/baseline` **median 1.128e+04, range 7.460e+03–2.881e+07, n = 3** — median-and-range,
never `±`, per C36. **I1 ✅ I2 ✅ I3 ✅, 3/3 against a pre-registered 2/3**, every threshold
inherited verbatim from I1. Key sets `|K|` = 6/4/4. **The falsifier that mattered did not
fire**: the pre-registration named `excluded/baseline < 100×` as the outcome that would
confine the internal claim to n = 113 forever; the worst seed reads 7.46e+03×.

**⚠️ Excluded accuracy is BELOW chance on 2 of 3 seeds** — 0.0000 / 0.0120 / 0.0000 against
1/110 = 0.00909, where n = 113 read chance almost exactly (0.0089–0.0091). **Recorded as
descriptive.** No criterion reads accuracy against chance and promoting it after seeing it
would be the C6/C7 ordering error. It needs its own pre-registration.

**⚠️ The null's upper tail reaches the effect again, on seed 2**: control **max 18.6887
exceeds** excluded 18.4667 (101.2%), where s0/s1 reach 89.8%/85.9%.
`excluded/median(control)` reads 1.09e+04 and hides it entirely. **Both beating draws carry
overlap 3 of |K| = 4**; every top-5 draw on every seed carries overlap 2–4. Cause is the
inherited pool (includes the key characters), which is conservative and was chosen for
comparability with C22/C23. EXPLORATORY, again rather than promoted: damage is monotone in
overlap and **19,426 zero-overlap draws never exceed 1.71e-02** against an excluded loss of
18–58. ⚠️ Zero-overlap means read **0.77× / 0.76× / 1.04×** baseline — two *below*, where
n = 113 read 1.04–1.11×.

**⚠️ `run_intervention.py --summary` POOLS THE TWO ARMS.** It globs `I1_*.npz` under
`I1_OUT`, and I1b writes into the same `results/i1_internal/` as I1, so it printed
`median 1.635e+07, range 7.460e+03–2.780e+08, n = 6` across n = 113 **and** n = 121 — exactly
the cross-modulus import the pre-registration's Prediction section forbids. Every n = 121
figure above was computed on the three n = 121 artifacts alone and re-derived from the `.npz`.
**The script was deliberately NOT patched**: "`run_intervention.py` unmodified" is a criterion
of this arm, and changing it after the measurement would break that for no gain. It needs an
`n` filter before `--summary` is used again.

## 2026-09-25 — Entry 63: C40 put under test and into the paper, F4 re-stamped, `--summary` scoped

Session 22. Researcher instructions, in order: task 1 (`test_paper_numbers` for C40), task 2
(FINDINGS + §7), propagate to the four sections that contradicted §7, task 3 (F4), task 4
(`--summary`). **Zero Kaggle quota, no push.** One local CPU job: the F4 re-run, 115 min
under `systemd-run --user --unit=f4-restamp`, `MemoryMax=6G`, ~194 MB, exited `dead / success`.

### Task 1 — C40 is scored against its artifacts (`98cfe11`)

C39's per-seed loop in `test_paper_numbers.py` became one function, `_i1_arm(tag, path, n,
doc, zero)`, called for n=113 and n=121 so the two arms cannot drift into two answers; it now
also asserts each artifact's `n`. C40 adds per-seed baseline/restricted/excluded, both
accuracies, ratio, `p_perm` (1/10001, 1/10001, 3/10001), `n_ge` (0, 0, 2), verdict and
`max|ΔW_E| = 0`; median **1.128e+04**, range 7.460e+03–2.881e+07, n=3; chance **1/110 from
`len(units(121))`**, excluded accuracy below it on 2 of 3; restricted/baseline 13.6× and
1.39×; control max / excluded **89.8 / 85.9 / 101.2 %**; seed 2's two beating draws both at
overlap 3 of |K|=4; **19,426** zero-overlap draws (4825/7304/7297), max **1.71e-02**; key
sets; grok steps 10,700/10,300/5,000 read from `hist` **by column name**; clean `266b77d`
stamps; and the prereg Outcome and STATE's C40 row scored as documents. **Every ledger number
reproduced from the `.npz`; none needed correcting.** 302 → 374. Negative control: three
planted stale values (a median, seed 2's `n_ge`, STATE's accuracy triple) → 3 FAIL, exit 1.

### Task 2 — FINDINGS §3.3c and paper §7 (`fc20234`), then propagation (`f7aa779`)

FINDINGS §3.3c is new; §3.3b's "one modulus" caveat points to it. §7's subsection is retitled
*"Inside the network, at a prime and at a prime square"*; `tab:internal` gains an `n` column
and three n=121 rows, the caption reports the moduli separately and says they are never
pooled, and a new paragraph gives the non-square-free test and three differences from n=113
**reported, not interpreted** (below-chance excluded accuracy; the null's tail on seed 2;
ratios ~two orders lower at the floor). The exploratory dose table (overlap 0–4 means,
e.g. seed 0: 1.35e-06, 6.36e+00, 1.37e+01, 2.28e+01, 2.82e+01) and zero-overlap/baseline
0.77 / 0.76 / 1.04 were re-derived from the artifact and asserted.

Researcher then approved propagation to the four sections §7 now contradicted: abstract
("At one modulus"), methods ("At $n = 113$ only" — **false** after §7 changed), conclusion,
and the appendix ledger (C39 row, **no C40 row**). All updated.

**A counting error, twice.** The paper's `tab:moduli` has **23 rows, including 113 and 121**.
The conclusion already said *"one modulus, which does not extend to the other twenty-three"*
— off by one before this session touched it — and `fc20234` wrote **"twenty-two"** into §7,
FINDINGS §3.3c and STATE's C40 row, copying STATE's session-21 wording. Correct is
**twenty-one**. Now scored **structurally**: the test counts `tab:moduli`'s rows and demands
"the other <count − 2>" in methods, §7 and the conclusion. 398 → 412.

The first phrase-ban banned bare "at one modulus" and **fired on §6 and §12**, where it is
grok-time prose ("five seeds at one modulus"). Narrowed to the conclusion's exact old
wording (`--- at one modulus, which`), not weakened — and the false hit is itself evidence the
ban reads every section.

### Task 3 — F4 re-stamped clean, 705 of 705 values unmoved (`c3a4d55`)

`results/k04_extended_n4_ablation.npz` (the abstract's 28 of 29) was stamped `0630186`,
`git_dirty=True`. Copied to `results/_archive_f4/k04_extended_n4_ablation.dirty_0630186.npz`
(md5 `ed4df366…`) because it is gitignored. `git diff --stat 0630186 HEAD` on `analyze_n4.py`
and its imports (`ablation`, `transforms`, `sparsity`) is **empty**; `jblocks`/`provenance`
moved but are not imported by it; its null is `default_rng(0)`. So the prediction was a
**zero** diff. `scripts/r1_diff.py` compares JSON, so a keyed `(tag, n)`, NaN-aware
`.npz` diff was written out-of-tree and controlled both ways first: against itself **705
compared, 0 moved**; against a planted `p_perm += 1e-9`, **1 moved**. Result: **705 of 705
unmoved**, new stamp **`f7aa779`, clean**, 28/29 at p < 0.01, the exception n=98 s0 at
**p = 0.0151 (150 of 10,000)**. `test_paper_numbers` now fails on any dirty live
`*_n4_ablation.npz` (`n7_engine_preleakfix` excluded as retired); planting the old copy as
`results/zz_neg_n4_ablation.npz` → 1 FAIL, then removed. 412 → 413.

**Runtime was misjudged twice.** "~20 min" from three runs in two minutes, then "~35 min";
it took **115 min**. Per-run cost varies strongly with modulus (early runs ~1.5 min, mean
~4 min). **Budget `analyze_n4.py results/k04_extended` at ~2 h.**

**Why the tree was kept clean for all 115 min:** `analyze_n4` builds its stamp inside the
`np.savez(...)` call, i.e. at **write** time, unlike `analyze_gate2` which stamps at script
start. An edit to any tracked file mid-run would have re-created F4. Task 4 was therefore
written and tested in a scratch copy and installed only after the artifact landed. This is
the most plausible account of how F4 happened in the first place (R1 edits during a run),
though that is not shown.

### Task 4 — `--summary <n>` is required (`c3a4d55`)

`run_intervention.py --summary` globbed `I1_*.npz` in the shared `results/i1_internal/` and
printed a pooled `median 1.635e+07 … n = 6` across two moduli. The modulus is now
**required** — an optional filter would still pool by default — and each loaded artifact's
`n` is asserted. `--summary 113` → median 1.543e+08; `--summary 121` → 1.128e+04 (both match
FINDINGS); bare `--summary` and `--summary 999` fail loudly. `--selfcheck` OK. The
pre-registrations' mentions of the defect are left as written (historical record); LAB_PROTOCOL.md,
FINDINGS and STATE mark it fixed.

### Not done, and why

- **F5 / `k01_scout`'s six unstamped claim-carrying artifacts** — not started.
- **The Lean 4 discussion** the researcher asked for (session 20's standing instruction) —
  session 21 ran the audit but Entry 62 records **no** Lean discussion. Still owed.
- **`paper/overleaf_bundle.zip` is stale**: it predates `fc20234`/`f7aa779`, so Overleaf
  shows the one-modulus abstract, methods, §7, conclusion and ledger.
- `paper/CLAIM_MAP.md` has no row for C39 or C40; `test_paper_numbers` did that job here.

## 2026-09-27 — Entry 64: the Lean decision, two provenance gaps closed, and the paper on TMLR's template

Session 23 (2026-09-25 → 27). **Zero Kaggle quota.** One local CPU job: the Gate 1 re-run,
3.20 h under `systemd-run --user --unit=gate1-repro` in a clean `git worktree` at `1eaf0e3`,
484 MB peak, exited `dead / success`. Researcher decisions this session: **Lean option B,
deferred**; **no supervisor co-author for now**; **supplementary code as a ZIP**, not an
anonymous mirror; paper to be **submission-ready for TMLR first**.

### The Lean 4 discussion (session 20's standing instruction — now held) — `c080d6f`

Inventory of the paper's formal statements: Def. J-classes, Prop. torsor, Thm. stratum
identity, three consequences (`01-setup.tex`); Props. generator equivariance and
rotating-features no-op (`05-methods.tex`); the Ramanujan-sum support claim (§9). mathlib
already has `ZMod.card_units_eq_totient`, `ZMod.prodEquivPi` (CRT), and
**`isReduced_zmod : IsReduced (ZMod n) ↔ Squarefree n ∨ n = 0`** — which is the paper's
"nilpotents exist iff not square-free" almost for free. No Ramanujan sums found in mathlib.
**The argument that shaped the decision:** of ~50 defects in LAB_PROTOCOL.md, **none is in the
algebra**; the 2026-09-23 audit checked it three ways. Lean would buy certainty where errors
have not occurred, and it proves the *mathematics*, not `src/tasks/algebra.py`, which is what
every experiment runs — so the brute-force checks stay regardless. Researcher: purpose is
**confidence**; **option B (S1–S5 of `01-setup.tex`), deferred behind submission as backup
for the review response.** Spec with scope, `#print axioms` gate and a two-session kill
criterion: an internal design note.

### TMLR readiness — checked against the author guide, not assumed

Blockers found: not on TMLR's style file (a stated desk-reject reason: "format violations");
not anonymized (author block; "public repository" in §11; 68 artifacts carry
`kaggle_account` containing the surname); OpenReview profiles needed. **And §11/§12 printed
"62" artifacts without a SHA where the audit (F5) had counted 68.**

### F5 — k01_scout recovered via Kaggle's own record — `1eaf0e3`

k01's `run.py` has one commit, but **the scout ran before the repo existed**: artifacts
2026-09-10 16:08 +0600, `init` 23:23. `kaggle kernels pull account-a/grokking-k01-scout`
returns the source of the latest version, whose `lastRunTime` is **2026-09-10 09:30:20 UTC**
(= 15:30 +0600, the file's mtime; the recorded 37-min run lands 16:08). Its sha256
**`cd9c86ba…6be6f55` equals the committed file's**. Docker image digest
`sha256:37c64f7dd9c5…`. `reconstruct_provenance.py` now asserts the hash for runs that
predate the repo; a planted wrong hash → `DOES NOT MATCH`, rc 1.

### Gate 1 — a gap the ledger said did not exist, closed by reproduction — `1eaf0e3` → this entry

LAB_PROTOCOL.md said `gate1` "carries no claim". **§10 quotes four numbers from it** (grok 6,900;
five key frequencies; 7.3×; 85.0% vs 84.6%), and §12 said "every run of the from-scratch
engine carries a real commit" — false. It ran locally ~20:20 → 23:26 on 2026-09-10,
finishing **3 min after `init`**, so neither git nor a platform certifies it.
Pre-registered (`PREREGISTER_gate1_repro.md`, committed before launch): re-run at HEAD,
exact zero or replace §10's numbers. **Result: bit-identical.** `W_E`, `W_U`, `logits_all`,
`mlp_acts`, `y_all`, `step`, `hist` max|diff| = **0.0** at both snapshots; grok 6,900 = 6,900;
**54/54** log lines identical with wall-clock fields stripped; `history.json` byte-identical.
G1R-1/2/3 met. Originals → `results/_archive_gate1/` (md5 `b1b6dbdc…`, `ff7debeb…`),
stamped copies installed. **One deviation, recorded in the prereg:** the script rewrites
the *tracked* `history.json`, dirtying its own worktree; that file was `skip-worktree`'d
before the step-5,000 snapshot, and the byte-identical final file shows it hid nothing.

§11: `$68$ artifacts from the three earliest sweeps`; §12 rewritten: 37 unstamped (k01 6 +
k02 31), 31 `unknown` (k03), the scout's platform-record argument, and the Gate 1 sentence.
`test_paper_numbers.py` now **scores 37/31/68 from the artifacts against §11 and §12** and
asserts both Gate 1 stamps: 413 → **419**. Negative control: a planted `$62$` → 1 FAIL.

### The TMLR template — `6c49309`; and a local compile that found an 11-day-old defect

`tmlr.sty`/`tmlr.bst`/`fancyhdr.sty` vendored (Apache-2.0) from JmlrOrg/tmlr-style-file;
10pt; no option = anonymous; `[preprint]` is the arXiv switch. First local compile ever
(tectonic, then TeX Live 2025 once installed): **`tab:moduli` declared `rlccrrrrc` (9
columns) over 10-column rows since `b70159a`** — `! Extra alignment tab has been changed to
\cr`, swallowed by Overleaf's nonstop mode, merging the `grokked` column. Fixed to
`rlcccrrrrc`, rendered and looked at. Three tables were **85.8 / 167.1 / 84.4 pt** into the
margin (Gate 2 criteria, CRT enrichment — transposed —, appendix ledger); wrapped, cells
unchanged; the ledger then overflowed its page by 49 pt → `\footnotesize`. **The rendered
bibliography printed DeMoss's internal `note` "Journal DOI unconfirmed"**: DOI resolved via
api.crossref.org to **`10.1016/j.physd.2025.134859`** (vol. 482, art. 134859), whose
`alternative-id` is exactly the PII on file. Result: **42 pages, 37 before the references**,
0 undefined refs/cites, one invisible 1.8 pt overfull line.

`paper/check_tex.py --compile` (`2c3a75d`): latexmk `-halt-on-error` into a temp dir; fails on
a TeX error, undefined ref/cite (latexmk exits **0** on these — only the log scan catches
them), or overfull > 10 pt. Self-check plants the column bug and an undefined `\ref`.
Negative control on a full paper copy with the old spec: **rc 12**.

### figures4papers (researcher asked) — assessed, not adopted yet

Matplotlib house style + an agent skill; **licence CC BY-NC 4.0**, incompatible with our MIT
code / CC BY figures / TMLR's CC BY — so its conventions may be followed in our own
`plots.py` style block, never its code copied. Its 24 pt / 45-inch canvases contradict our
print-size rule; its red-vs-green semantic bands need a colour-blind check; its schematics
are hand-made (Illustrator/Figma), which our "figures regenerate from code" rule forbids.
Waiting on the researcher: **which diagrams, and what is wrong with them.**

### Not done

Anonymized supplementary ZIP; Overleaf bundle rebuilt at the final commit (done at session-end checkpoint
below if the tree allows); `paper/CLAIM_MAP.md` C39/C40 rows; the length decision (37 pages).

## 2026-09-28 — Entry 65: the submission assembled, a false sentence in §9, and Lean re-scoped

Session 24. **Zero Kaggle quota, no Kaggle push, no training run.** Commits `be54980` →
`2b5ee38` (plus this session-end checkpoint). Researcher decisions: **submit to TMLR first**; **Lean now
covers every mathematical statement (tiers A–E), built during review** — reversing Entry 64's
"option B, S1–S5, deferred"; **Overleaf retired**, local TeX only; the submission lives in
`TMLR/`; `AISTATS2027/` holds a venue paper pack for a later cut.

### The supplementary ZIP — `be54980`, `f157c0e`, `1198f27`

`scripts/make_public_export.py --review` = the public export minus the deposit instructions
(a DOI and URL that do not exist yet), zipped: **0.9 MB** of TMLR's 100 MB. Three leaks the
existing gate did not see, each caught before shipping:
1. **A directory named after the second account** (`results/k00_smoke_<account-b>`):
   `global_subs()` relabels file *contents*, never file *names*, so the prose pointed at
   `k00_smoke_account-b` while the tree carried the real name. `rename_identity_paths()`
   applies the same account rows to path names; `check_public.py` gained the second
   account's name. Planted in a path and in prose → 3 FAILs.
2. **`scripts/build_tmlr.sh` spells out the identity pattern it greps for** — the gate fired
   4 times on its own line 21. Excluded from the export, the same reason `check_public.py`
   is. `TMLR/` and `AISTATS2027/` excluded too.
3. **A false positive**: adding `gmail` to the build's identity grep fired on a notebook
   sentence about another paper's author addresses. Dropped — the email is covered by the
   surname token. Same family as the `PLACEHOLDER` sentinel lesson.
Paper: §11 "regenerates from a public repository" → "the code released with it (the
supplementary material during review)"; `tab:moduli`'s "at commit `b70159a`" removed after
**all 23 LaTeX rows regenerated at HEAD matched verbatim** (the commit does not resolve in a
supplement). PDF metadata Author/Title empty; 0 identity hits in its text.

### §9 said "support" where the truth is "concentration" — `f157c0e`

Found while writing the Lean statement list, before any Lean existed. The sentence: the
J-class indicators' transform "is a Ramanujan sum, so the stratification's spectral support is
the duals of *every* divisor — 42 frequencies at n = 165". **False.** The transform of
`1_{J_d}` at `k` is `c_q(k) = μ(q/g)φ(q)/φ(q/g)` (`q = n/d`, `g = gcd(k,q)`) — closed form
verified exact at n = 165, 120, 121, 119 — and it never vanishes at square-free `q`, so on the
paper's own instrument (`test_crt_law.jclass_indicators` + `freq_energy`) the energy is
nonzero at **all 82** frequencies at n = 165. The 42 are exactly `{k ≤ n/2 : gcd(k,n) > 1}`
(27+16+7−5−2−1), and they carry **30.5×** the mean energy of the other 40 (26.8× on raw
un-centred indicators including `J_n`; the paper quotes the instrument's 30.5). Rewritten as
concentration; **the section's conclusion is untouched** — it rests on what the detector
selects (network = P(n); indicators ≠ P(n)), not on support. `FINDINGS.md` corrected in place
with a dated note. `test_paper_numbers.py` 419 → **424**: the 42 = gcd>1 set, support 82,
the 30.5×, and two *prose* checks. **Negative control:** the old §9 planted back fails
exactly the two prose checks (422/2), restored byte-identical.
**Why nothing caught it:** the old test asserted `len(_duals165) == 42` — the *count* was
right; the *word describing what the count is* was wrong, and no number check reads a word.

**Every other algebraic statement brute-forced, n ≤ 200, 0 violations:** partition and
`|J_d| = φ(n/d)`; torsor bijection; nilpotent ⇔ not square-free; square-free ⇒ all classes
regular and `J_d·J_d = J_d`; non-square-free ⇒ a non-regular class; **the sharp form
`J_d` regular ⇔ `gcd(d, n/d) = 1`** (new; it is the Lean plan's B2); non-regular counts
0,0,1,2,8 at 113/119/121/125/120; `J_11·J_11 = {0}` at 121; `J_5·J_5 ⊆ J_25`,
`J_25·J_25 = {0}` at 125; ν(121) = 2, ν(125) = 3; reduction onto units surjective (N < 120);
`dlog_{g^t} = t⁻¹ dlog_g` at 113/121/125/127; Gini ∈ [0, 1−1/M] on 2,000 random spectra.
A one-off script, not committed — the durable version is the Lean project.

### Build and tooling — `f157c0e`

`scripts/build_tmlr.sh` writes `TMLR/{main.pdf, supplementary_material.zip, abstract.txt}`;
refuses a dirty tree; the PDF is copied **only if** `check_tex.py --compile` passes (new
`--pdf` option, so what ships is what was gated); greps PDF metadata, PDF text and the
unzipped ZIP for identity. `TMLR/OPENREVIEW.md` (tracked) is the form field by field,
checked against the TMLR author guide (anonymized PDF on the style file; anonymized
supplement ≤ 100 MB PDF/ZIP; no page limit; complete OpenReview profiles; action-editor
suggestions; broader impact only for significant risk). `make_overleaf_zip.sh` deleted.
Final build at `2b5ee38`: **43 pp** (main content pp. 1–38, references p. 39, appendices
after), supplement 916 KB, abstract 3,522 chars. `check_tex --selfcheck` OK;
`test_reproduce` PASS (40 covered, 3 exempt); `check_public --selfcheck` 5/5.

### Lean — plan and toolchain, no Lean source yet

Spec amended before any `.lean` file (`be54980`); master plan
an internal design note (`57dee0a`): ~20 statements in tiers
A–E, sessions L0–L8, per-tier kill criteria, the `#print axioms` gate negative-controlled
with a planted `sorry` and `native_decide`, and **statements read by the researcher before
any proof**. elan 4.2.4 installed; a scratch `lake new Strata math` (Lean v4.35.0-rc3) sits in
the job scratch dir — **not** in the repo, and `lake exe cache get` has not run. Environment
note: the Bash sandbox blocked elan's own DNS resolution (curl worked); the download needed
the sandbox disabled.

### Not done

Submission itself (researcher: OpenReview profile, form, action editors, length call);
Lean L0; `CLAIM_MAP.md` C39/C40 rows; figures (awaiting the researcher's list).

## 2026-09-28 — Entry 66: a revision planned before submission, a stale ranking in Limitations, and the conditions of approval

Session 25. **Zero Kaggle quota, no training or analysis run, `paper/` untouched.** The main
version for the revision is `paper/` at `9c1eaa5`. Commits `2c18500` and this session-end checkpoint.

### Ground truth at session start

`9c1eaa5` equalled the remote's master (checked with `git ls-remote`, not with the local
tracking ref). The tree was clean, 8/8 self-checks passed, and no process ran under our
interpreter. Every Kaggle kernel readable from the primary account reported COMPLETE; k07 and
k08 belong to the second account and cannot be read from the primary (both finished 2026-09-11).

### The request

Before the TMLR submission the researcher asked for seven changes. Parts of the paper are to
follow ASD-STE100 Simplified Technical English, where it is needed most and fits best.
Formulaic phrasing is to go. The organisation and flow are to improve, the discussion is to go
deeper, and the methods are to gain detail. Results are to be clearer and compared against
existing methods. The figures are to be redrawn in the figures4papers style. The plan is
an internal design note (`2c18500`), approved the
same day with its nine defaults.

### Baseline measurement

A throwaway script over `paper/sections/*.tex` measured prose only (display math, tables and
figure environments stripped): 17,863 words in 720 sentences. Mean words per sentence: abstract
**42.9** (67% of its sentences over 40 words, the longest 80), conclusion 33.9 (38% over 40),
introduction 30.4, implementations 28.0 (30% over 40), setup 27.1 (longest 88), the rest
19.1–26.7. Counts: em dashes **198**, semicolons 81, contrast constructions ("X, not Y",
"not … but", "rather than") **110**, `\textbf` **138** (most of them mid-sentence), and about 25
performative phrases (*plainly* ×6, *we report that/it* ×6, *the reason is* ×4, *we say so* ×3,
*this matters* ×3). The 16 captions hold 2,087 words, a mean of **130**. The register's banned
list has **0** hits; seven lesser formulaic words remain (*actually* ×3, *additionally*,
*moreover*, *harness*, *leverages*). The unit group is written `^{\times}` 14 times and `^{*}`
5 times. The abstract carries about twenty numbers against its own register's "three or four".

**Reading.** The banned-word check has been clean since session 16, and it says nothing about
sentence structure, which is where the remaining problem is. WP0 turns the structural patterns
into a lint.

### P1: a stale ranking in the Limitations section (found, not yet fixed)

`12-limitations.tex:28–29` says "$\omega$ does not displace zero-divisor density, which remains
the stronger partial correlate". Since the midrank audit (Entry 62, F3), §9 reads
ρ(ω, G | zdd) = +0.804 against ρ(zdd, G | ω) = +0.578 and concludes that "this data cannot rank
them". The Limitations sentence still carries the ordinal-rank ordering (+0.735 over +0.627)
that the audit retired. It survived because it contains no number: the old-value grep that
closes a correction had nothing to find. `FINDINGS.md` and the C37 row of the claims ledger were
already right. The drafting checklist kept outside the repository carries the same retired
ordering (P2). WP0 fixes both first, P1 with a prose check that is negative-controlled by
planting the old sentence back. The lesson is in the protocol file.

### Sources checked for the plan

- **figures4papers** is **CC BY-NC 4.0** (LICENSE read 2026-09-28; the repository was last pushed
  2026-09-26). That licence is incompatible with our MIT code, our CC BY figures and TMLR's CC BY.
  Its conventions are therefore re-implemented in `src/viz/plots.py` and none of its code is
  copied, which is what its own API notes suggest. Rejected conventions and the reason for each:
  - 15–24 pt text on 12–45 in canvases, which prints at about 2.3 pt at `\textwidth`;
  - bars on truncated axes;
  - hidden tick labels;
  - `usetex`, which would give `reproduce.sh` a TeX dependency;
  - hand-made schematics;
  - a red/green pair, which fails for colour-blind readers.
- **ASD-STE100** is at Issue 9 (January 2025): 53 writing rules in nine sections and a dictionary
  of about 900 words. Its rules:
  - at most 20 words per procedural sentence and 25 per descriptive sentence;
  - at most six sentences per paragraph;
  - active voice;
  - verbs only as infinitive, imperative, simple present, past or future, with the past
    participle only as an adjective;
  - *-ing* forms only in technical nouns;
  - noun clusters of three words at most.

  These were checked against Wikipedia's summary. The no-semicolon rule comes from a secondary
  summary and is enforced conservatively. The official FAQ returned HTTP 401. Applying the
  dictionary needs the official specification, which is free on request.
- Fonts installed here: Nimbus Sans, TeX Gyre Heros, Liberation Sans and DejaVu Sans. TMLR's
  `\textwidth` is 6.5 in (`paper/tmlr.sty:58`).

### Decisions (researcher, 2026-09-28)

- The revision plan is **approved** with defaults D1–D9. STE goes on the abstract, the methods
  protocol, a new reproduction appendix, the Limitations and every caption, and not on theorems,
  related work or the discussion. The **Lean master plan is approved** too. Order: revision,
  then submission, then Lean L0 during review.
- **The approval carries two conditions.** The revision alters no data, no information and no
  interpretation. Every change passes three line-by-line checks against the main version
  (`paper/` at `9c1eaa5`), one for each of those three categories (plan §13; protocol file, hard
  constraint 5). The Discussion gathers interpretations already in the paper or `FINDINGS.md`,
  each at its recorded strength, and adds none.

### Also

- `scripts/session_brief.py` printed "next 2026-09-19T00:00Z" as the next quota refresh, a typed
  constant nine days out of date. It now computes the next Saturday 00:00 UTC, checked on four
  planted instants: a Monday, a Saturday at the refresh instant, a Friday at 23:59 and a Sunday.
- The resume file's header still read "Session 23 … 64 entries" through session 24. It is now
  corrected.

### Not done

The whole revision (WP0–WP4), the submission, and Lean L0.

## 2026-09-28 — Entry 67: revision WP0, the instruments that check the revision, and a stale value in four documents

Session 26. **Zero Kaggle quota, no training or analysis job, no Kaggle push.** Commits `27ed8f5`
(WP0) and this session-end checkpoint. The main version stays `paper/` at `9c1eaa5`; the only change to the
paper is P1.

### The request

The researcher asked for the whole revision at once: every work package of the approved plan,
then the TMLR submission build, then a complete arXiv submission package in the new, empty
`for arxiv/` folder at the repository root, then three full sweeps for inconsistencies and
mistakes, with fixes. This batches the plan's per-WP approval gates, which plan §11 leaves to
the researcher. Partway through WP0 the researcher asked to wrap up before WP1. WP0 is done;
WP1 to WP4, the TMLR build, the arXiv package and the sweeps are not started.

Commit messages from this session carry no AI-usage trailer. The `write-paper` skill forbids an
AI-usage disclosure "not in commit messages" either. That settles the question Entry 66 left
with the researcher, on the researcher's own written rule, and the researcher can overturn it.

### The register lint — `paper/check_tex.py --style`

The lint was written test-first. Its self-check plants one violation per hard rule, each in
its own file, and asserts that each plant fails for its own rule. A clean text must pass. It
contains a run-in header, a caption lead in bold, a table cell in bold with a dash, an en-dash
range, "et al.\ ", a comment that holds banned words, a quotation and a clean STE paragraph of
each kind. The self-check failed against a stub before the implementation existed.

- **Hard rules (fail):** an em dash in prose or a caption; the register's banned list together
  with the plan's list; the performative phrases; `\textbf` anywhere in prose outside a
  caption's first sentence (as a lead or inside a sentence); a sentence of more than 40 words
  (inline math is one word, `\cite` and `\ref` are none).
- **Soft rules (reported):** contrast constructions, semicolons outside STE, sentences of more
  than 30 words, and copula avoidance.
- **STE regions:** first-on-line `% STE procedural`, `% STE descriptive` and `% STE end`
  comments. They add limits of 20 or 25 words per sentence and 6 sentences per descriptive
  paragraph. They also ban the passive (fail in procedural text, warn in descriptive text),
  has/have/had with a participle, any -ing word outside an allowlist, and semicolons.
- **Quotations** (``…'') are exempt from the word lists. Furuta et al.'s "leverages all
  frequencies" is a verbatim quotation and stays.
- **The first run on the real paper found a defect in the lint.** A phrase that wraps across a
  line break escaped the match. "reported here in the\nopen" sat in the conclusion unflagged.
  A plant reproduced it (RED). Flattening whitespace before matching fixed it (GREEN), and
  three more performative hits appeared. This is the same lesson as the abstract's
  "$28$\nof $29$".

**Baseline of record** (`docs/revision/baseline.md`, measured by the committed lint at
`9c1eaa5`) replaces the plan's throwaway table: 17,334 prose words in 780 sentences, a mean of
22.2 words per sentence, 173 em dashes in prose and 18 in captions, 103 bold spans in prose,
88 semicolons and 2,079 caption words. Hard findings at the start of WP1: **357** (em dash 136,
bold lead 91, over 40 words 81, performative 27, bold 16, banned 6). Soft findings: over-30
134, contrast 107, semicolon 104, copula 4.

### The three checks against the main version — `docs/revision/revision_checks.py`

This is plan §13's condition of approval, built as an instrument. Every unit of the main
version is aligned to its revised counterpart, across files, by a window of up to three units:
a sentence, a caption sentence, a heading, a table row or a display equation. Aligned groups
are unions over splits and merges, and they are compared on three axes.

- **Check 1, data:** numbers, number words included; symbolic math spans; cite keys; refs;
  `\texttt` names. Labels must match exactly.
- **Check 2, information:** negation, universal and restrictive quantifiers, bounds,
  direction, the qualifiers (`1 seed`, float32, float64, PyTorch, engine, the hedges,
  pre-registered), attribution to other authors, and caveats.
- **Check 3, interpretation:** the evidence rung (strong, find, weak), causal wording and
  correlational wording.

Anything that differs is a flag until `docs/revision/decisions.json` records why it is not a
change of meaning. A decision is keyed to a hash of the exact revised text, so any later edit
to that text re-opens it. A main unit that disappears needs a `dup_of` or `carried_by`
decision, and the checker verifies that its data is really there. A revised unit with no
main-version source is a `new` flag until a decision names its source.

- **Self-check:** a clean paraphrase (reordered and split) passes. A changed digit, a dropped
  *not*, a lost `1 seed`, a *suggests* changed to *shows*, and a changed label each raise
  their own check. A move to another file passes. An unsourced new sentence flags.
- **Positive control:** the unchanged paper scores 0 · 0 · 0 over 1,166 units in 0.8 s.
- **Negative control on real text:** four plants in four real sections (0.9779 changed to
  0.9797, a dropped `(1 seed)`, a dropped *never*, *needs* changed to *may need*) were all
  caught at the right line by the right check.

### The defects fixed in WP0

- **P1.** `12-limitations.tex` now reads "…does not displace zero-divisor density, and this
  data cannot rank the two", agreeing with §9. Two prose checks were added to
  `test_paper_numbers.py`, which moves from 424 to **426** checks. One asserts that no section
  says "stronger partial correlate"; the other asserts that the Limitations section says
  "cannot rank". **Negative control:** the `9c1eaa5` file planted back fails exactly those two
  checks (424 passed, 2 failed), and the restore was confirmed byte-identical with `cmp`. The
  three checks flag P1 on all three axes: the count in "the two", the added "cannot", and the
  dropped "correlate". It is recorded in `decisions.json` as the plan's one permitted exception.
- **P2** (outside the repository, the drafting aids). Failure mode 7 now
  gives the midrank values and states that this data cannot rank the two. **Failure mode 8
  also carried the stale torch-float64 value 0.7736**, which is now 0.7728. Every section pass
  reads this checklist.
- **P3** (`paper/CLAIM_MAP.md`). **C38 was missing as well as C39 and C40**; all three rows were
  added, with their qualifiers. The C37 qualifier still carried the retired ordering
  "+0.735 > +0.627", and the C22/C23/C32/C36 qualifier still said "200 draws". Both are
  corrected, with dated notes. Counts move from 39 to 42 (33 VERIFIED, 3 VERIFIED with 1 seed,
  5 retracted, 1 confirmatory).
- **STATE.md claims ledger.** The C23 row said "27 of 29 … 49 and one seed of 63 fail". The
  count was the B = 200 count, and the exception list was wrong even at B = 200. The artifact,
  FINDINGS §3.3, the paper and `test_paper_numbers.py` all say **28 of 29, with n=98 at
  p̂ = 0.0151 (150 of 10,000 draws) as the only exception**. The C22 row gains R1's 26/31
  runs beyond every draw (re-read from `results/k03_grid_acts_n4_ablation.npz`: 31/31 at
  p < 0.01, B = 10,000).

### A stale value in four documents (fixed at session-end checkpoint)

The O18 torch-float64 cell is **0.772770**, so F1 is **+1.110**. Entry 49 and LAB_PROTOCOL.md
recorded this correction, and `test_paper_numbers.py` asserts it from the artifact. The paper
is clean. The correction had not reached the running text of the other documents:

- 8 lines of `STATE.md`, including the C34 ledger row, the §1 summary and the C4 row;
- `FINDINGS.md` lines 962, 963 and 2087, in the running text outside its own correction block;
- `LAB_PROTOCOL.md`'s O18 box, lines 148, 149, 159 and 172, which contradicted its own lesson at
  lines 383–384;
- the drafting checklist, where the P2 fix above covers it.

These documents said 0.7736 and +1.114. All the running-text occurrences are corrected. The
dated correction notes (FINDINGS 1861–1868, LAB_PROTOCOL.md 383–384) are unchanged. The C34 row
gains a dated note. The verdict is unchanged, because F1's threshold is 0.70. **Why it
survived:** the correction was a fourth-decimal change recorded as a lesson. The old value's
grep was never run across the documents, which is the same failure as C23's 27/29 and C33's
two stale cells. After the fix, `test_paper_numbers.py` still passes 426 of 426.

### Found, not fixed (queued for the sweeps)

- `scripts/session_brief.py` §5 prints TOTAL 43 against 42 claims. It counts every
  `| C…` line in `STATE.md`, and line 174 is a two-cell session-17 summary row outside the
  ledger table. The four category counts are right.
- The drafting aids `paper/DESIGN_CHOICES.md` (2 hits), `MASTER_PLAN.md` (1) and `OUTLINE.md`
  (1) still carry 0.7736 or +1.114. `OUTLINE.md:136` still says "27 of 29". These files are
  excluded from the export.
- **Candidate inconsistency in the paper, not yet verified as an error:**
  - `11-implementations.tex` says "every absolute sparsity number in this paper is a float32
    measurement".
  - `05-methods.tex` says "Every sweep in this paper is float32".
  - The same sections print float64 values: the engine's cells and PyTorch's float64 cell in
    the 2×2.
  - The paper uses "sweep" for the PyTorch runs, so the second sentence may be meant as scoped.
  - The first sentence reads as unscoped.
  - It goes to the sweeps, to be read in context and settled against the ledger. Any fix is an
    information change and must be listed.
- `05-methods.tex` describes the near-grok control run twice, once as "test accuracy 0.9779"
  (last row) and once as "$\bar a = 0.976$" (window median). WP1 may merge the two, but it must
  keep both numbers.

### Not done

WP1 to WP4, the TMLR rebuild, the arXiv package, the three sweeps, the submission and Lean L0.

## 2026-09-28 — Entry 68: revision WP1, the paper restructured, and two gaps in the revision checker

Session 27. **Zero Kaggle quota, no training or analysis job, no Kaggle push.** Commits `f5fb948`
(WP1), `e904ec3` (the resume point) and this session-end checkpoint. The main version stays `paper/` at
`9c1eaa5`. No claim changed status.

### Ground truth at session start

`de9850c` was clean and pushed, 8/8 self-checks passed, and no `.py` process ran under our
interpreter. Every Kaggle kernel readable from the primary account reported COMPLETE; k07 and k08
belong to the second account.

### The request

The researcher asked to start WP1 of the approved revision plan: structure only (plan §3). The
per-WP gates stay batched (Entry 67).

### What moved

New reading order: 1 Introduction, 2 Related work, 3 Setup, 4 Methods, 5 Validation, 6–9
Results I–IV, 10 Discussion, 11 Limitations, 12 Conclusion, appendices A–E, and F Experimental
details and reproduction. File names are kept. Two files are new, `15-discussion.tex` and
`16-reproduction.tex`. The only added label is `sec:discussion`.

- Setup's "Coordinates" and "Model and training" moved into Methods (4.1, 4.2). The two
  Coordinates subsections are merged and carry both labels.
- The internal-intervention protocol paragraph moved from §7.5 into the Methods ablation
  protocol, beside the paragraph it continues. §7.5 keeps a one-sentence pointer to it.
- `10-calibration` and `11-implementations` form one Validation section. It is headed
  "Validation" and keeps the label `sec:calib`. "Calibration against published results" heads
  its first subsection, and `11` is subsection 5.3.
- "Code and data availability" moved verbatim into the new appendix F.
- The conclusion's paragraph "Why the shape of this result may travel further than the task"
  moved verbatim into the new Discussion.
- The caveat subsections now close their sections. §7.6 is retitled "Two caveats on scope"
  (it was "Two honest caveats"). §8's "What is not claimed" moved, unchanged, to the end (§8.7).
- Introduction: the contribution is now stated once, in the numbered list. The
  limitation-to-answer table and the bold thesis paragraph are merged into the two paragraphs
  that answer Chen et al.'s two limitations. The sentence on independent derivation moved
  beside "That framing is theirs." A five-sentence roadmap closes the section.

Changed sentences, all listed in the commit:
- The Validation opening's "above" became "below".
- The clause that pointed at the deleted table was dropped.
- Two pointers were re-aimed, because their sentence moved inside the section it named
  (`sec:methods-coords` to `prop:generator`, `sec:methods` to `eq:project`).

Merged duplicates: the table's three rows, three sentences of the thesis paragraph, and two
sentences from the Coordinates merge. The thesis sentence ("A transformer trained on … runs the
same character-based clock on every stratum …") stands in the conclusion's first sentence, the
abstract and contribution items 2 and 4. It is no longer in the introduction's prose. Every
number the table carried is still in the abstract and in the body (Section 8's G1 and G2 rows,
Section 7's 28 of 29); each was confirmed by grep.

### Gates

- **The three checks against `9c1eaa5`: data 0 · information 0 · interpretation 0**, over
  1,166 main and 1,168 revised units, with 23 WP1 decisions (12 `c:`, 9 `n:`, 1 `l:`, 1 `u:`).
- The sorted sentence multiset has 18 main-only and 20 revised-only units. Each is a heading,
  a transition, a merged duplicate, the table converted to prose, or WP0's P1.
- `check_tex.py`: 17 `.tex`, 116 labels, 189 refs, 65 cites, 0 failures. `--compile`: 42
  pages, 0 failures.
- `test_paper_numbers.py`: 426 passed, 0 failed. The "$68$ artifacts" pin moved with its
  sentence from `11-implementations` to `16-reproduction`.
- The Lean plan's pointer `01-setup.tex:76` still lands on the Proposition D.13 sentence.
- The register lint, for the record (WP3 gates it): hard findings went from 357 to 354. The
  three removed are the "honest" heading, the thesis sentence's lead bold, and the 41-word
  sentence that lost its table clause. Every other hard finding moved with its sentence.
  Soft semicolons went from 104 to 105.

### Two gaps in the revision checker, fixed test-first

- **An added label could never pass.** The check required the label multisets to match
  exactly, so a new section could not carry a label. A label added for a new section now
  passes only with an `l:<label>` decision, and only if the name is new to the paper. A missing,
  renamed or duplicated label still always fails. The self-check plants all four cases, and they
  fail against the old rule.
- **A rewrite too heavy for the matcher could not be recorded.**
  - The checker pairs units by word overlap: 60 %, or 40 % when every datum is present.
    A sentence that lost a third of its words (the clause pointing at the table) fell below
    both.
  - A main unit left unpaired could be resolved only by showing its data elsewhere, and
    "ten pages" has no elsewhere.
  - Had the matcher paired it, a recorded decision would have accepted the same change. So a
    heuristic score decided what a decision could accept.
  - Now a `u:<id>` decision with `rewritten_as` pairs the two by hand. The pair must then pass
    its own `c:` decision keyed to the exact revised text, like any aligned change.
  - The self-check plants a heavy rewrite. It is unaligned without the decision and raises a
    class flag with it. It is clean with the `c:` decision, and still unaligned when
    `rewritten_as` names no text.
  - It failed before the implementation existed (RED, then GREEN). The unchanged main version,
    scored against itself with the committed decisions, still gives 0 flags.

### Observed while resolving flags

- The checker attaches a deleted unit to an unrelated, unchanged sentence when the two share
  one rare word. The table's header went to a sentence of the reproduction appendix
  ("limitation"). "The result the paper argues for is one sentence." went to §5.3
  ("sentence"). "Every section is here …" went to §8 ("load-bearing"). Each decision records
  that the revised sentence is unchanged. Editing that sentence to silence the flag would have
  been the wrong fix.
- Number words count as data. Retitling "Two honest caveats" as "Scope" therefore dropped a
  datum ("two"), and the retitle keeps the count ("Two caveats on scope").
- The first WP1 commit message gave the decision count as 22 and attributed the lint change
  wrongly. Both were re-counted and corrected by amending before any push. The two amended-away
  commits were never pushed or referenced.

### Found, not fixed

- `paper/main.tex` comments still name Overleaf as the compiler and point to "Section 5" for
  the evidence class. They do not print, and the arXiv package strips comments.
- `\ref{sec:calib}` from §5.3 prints "Section 5", the section that contains it. This is correct
  but coarse, and a WP3 pass may prefer a subsection label.
- Entry 67's list stands. The near-grok run is still described twice in Methods (0.9779 last
  row, 0.976 window median), and WP1 did not merge them.
- WP4's abstract rewrite will re-open `c:1951a…`, the abstract sentence that absorbed the
  introduction's table row. Its numbers must then be re-pointed to the body.

### Not done

WP2 to WP4, the TMLR rebuild, the arXiv package, the three sweeps, the submission and Lean L0.

## 2026-09-28 — Entry 69: revision WP2, the figures drawn at print size, and a stale F4 panel (P4)

Session 28. **Zero Kaggle quota, no training or analysis job, no Kaggle push.** Local CPU
only: figure rendering and the figure harness (F4 recomputes its 10,000-draw null in about
6 min). Commits `0a013aa` (generator), `b9972c0` (stamp), `de23661` (P4) and this session-end checkpoint.
The main version stays `paper/` at `9c1eaa5`. No claim changed status. The full record is
`docs/revision/figures_wp2.md`.

### The request

The researcher asked to start WP2 (plan §9). After the report, the researcher approved fixing
F4's stale panel as **P4**, a permitted exception like P1, then pushing and wrapping up. WP3
starts next session.

### The instrument first: `docs/revision/figure_checks.py`

This is check 1 of plan §13 for figures, written test-first (RED against a stub). It draws
every paper figure twice in one process, once with `src/viz/plots.py` at `9c1eaa5` and once
with the working tree. It records what the renderer actually draws: lines, point sets,
segments, patches and images in their own coordinates, axis limits and scales, and every text
string. The two multisets must be equal. Colour, size, font and whitespace inside a string
are styling. Legend handles, ticks and arrows are drawn in display units that move with the
font size, so they are layout. Tick labels count only if the limits moved. It also measures
each string's printed size at the paper's `\includegraphics` width.

- **Self-check: 15 plants** (identical, pure restyle, changed datum, changed digit, added
  text, changed tick, changed range, 5 pt, half-width shrink, accepted change, stale
  acceptance, one of two pairs stale). A broken `accept()` planted in its place is caught.
- **Positive control:** unchanged code gives 14/14 `data OK`. The first attempt flagged F6
  on identical code, because `test_crt_law` keeps a module-level `RNG`. The second drawing in
  one process continued the first one's stream. In a fresh process F6 is deterministic. The
  harness now re-imports the repository's own modules before every capture.
- **The same run measured the main version's figures printing their text at 3.1 to 5.4 pt**
  (gate2_strata 3.1, basis_comparison 3.1, F4 3.2, F10 3.5, F1 3.5, omega 3.4, F0 3.8, F8
  3.9, F2 4.1, F3 4.1, F9 4.1, F5 4.2, F6 4.3, F7 5.4). LAB_PROTOCOL.md's old rule (10.2 in, 8.5 pt
  floor) was meant to give about 6 pt; F1 and gate2 had drifted well below it.
- `scripts/render_all.py`: the figure list is now module-level `PAPER_FIGURES` (name, function
  name, kwargs, probe, description), read by `main()` and by the harness. The harness asserts
  it equals the paper's `\includegraphics` set, both ways.

### The style block

- **Print size.** Each figure is drawn as wide as the paper sets it. `_save()` measures the
  tight box and resizes the figure until the PDF has exactly that width (saved widths
  6.44–6.51 in at `\textwidth`). Size tokens: FS_TITLE 9, FS 7.5, FS_SMALL 7.
- **Two failure modes of `_save`, both found by rendering.** (1) A line of text wider than
  the page made the loop "converge" by shrinking F3's axes to a thumbnail. It now asserts
  that the figure shrank by less than 15 % and hit the width to 0.01 in. (2) A footer with
  `wrap=True` always fills the figure, so tight width = figure width + padding, and the loop
  chased it until the guard fired on F6. Long footers now carry an explicit line break.
- **Font: Liberation Sans, then DejaVu Sans.** The plan named Nimbus Sans and TeX Gyre
  Heros. Both are CFF. With `pdf.fonttype 42`, matplotlib embeds them as "CID Type 0C (OT)",
  and poppler warns "Mismatch between font type and embedded font file" on every page.
  Liberation Sans (Helvetica metrics, TrueType outlines) embeds as CID TrueType with no
  warning. `pdffonts` shows no Type 3 in any of the 14.
- **Palette, validated with the dataviz validator** (light surface), blue `#2A64AD`, red
  `#B64342`, teal `#1B9AA6`, purple `#9A4D8E`. In slot order every check passes: worst
  adjacent CVD dE 8.5 (deutan, teal-purple), normal-vision floor 23.0. In blue-teal-red (the
  ablation bars side by side) the normal-vision dE is 16.2. The plan's anchors failed:
  `#3775BA` against teal is normal-vision dE 12.0 (below 15), and `#42949E` has chroma 0.08,
  below the 0.1 floor. `#0F4D92` is outside the lightness band (0.424). Blue-purple is dE 5.1
  for protanopes, so they are never adjacent.
- Grouped bars (F4, F7, F10) have black edges and a hatch per hue, because blue and red
  print at almost the same grey.

### The audit

Every figure was rendered at print size and looked at. 13 of 14 needed layout fixes, and all
of them were collisions invisible in the code: F1 (notes, text column, clock labels), F5
(nodes sized for 10.2 in overlapping in the n = 120 lattice, now 250 pt² and width by lattice
span), omega_bands (legend over four moduli, label clusters), F6 (titles, ticks, p label), F9
(legend over G1's 20/20), F10 (labels on the dots, which now carry a white halo), and others.
Table in `figures_wp2.md`. Every change is a line break, position, size or margin, and the
harness shows no plotted value moved.

### P4: F4's null panel contradicted its caption

Found by reading the print-size render, not by any check. The main version's F4 right panel
said "Permutation null, n=121 seed 0 **(200 draws)**" and "**p = 0.0000** (0 of 10000 draws
reach the key set)". The panel plots 10,000 draws. The caption says 10,000 and gives p̂ at the
floor, 1.0 × 10⁻⁴. The 200 predates R1 (Entries 54–58, B = 200 to 10,000), and "p = 0.0000" is the
retired `b/B` form. `test_paper_numbers` asserted the caption against the artifact the whole
time, which is why nothing fired: **the figure's own text is a document the caption checks
cannot see.**

- Fixed at `de23661`, approved by the researcher. The title prints `len(draws)`, and the line
  prints `src.analysis.stats.perm_p`, so the figure now reads "(10,000 draws)" and
  "p̂ = 1.0×10⁻⁴".
- Two new checks in `test_paper_numbers.py` (426 → **428**): p̂ recomputed from the artifact
  (`n_ge` 0, `n_draws` 10,000) equals the caption's floor, and `causal_test` carries no
  draw-count literal and calls `perm_p`. The pre-P4 source fails the second (negative
  control).
- The harness accepts the two F4 strings only through the decision `fig:F4_causal` in
  `docs/revision/decisions.json`. Two more figure decisions: `fig:omega_bands` (the footer
  names its own file, now `.pdf`) and `fig:F2_stratification` ("orange outlines" → "red
  outlines").

### Gates

- Harness vs `9c1eaa5`: **0 problems**. 14/14 plot identical geometry, and every string
  prints at 7.0 pt or more. Tick labels were re-chosen by the locator, with limits equal, in
  omega_bands, F3, F4, F6 and basis_comparison.
- In-figure guards: all 15 `assert` lines are byte-identical to `9c1eaa5`. A planted
  off-by-one in F6's predicted set trips the 14.08× control ("n=165 reads 0.25"), and a
  regular class handed to F1 trips its guard.
- Checks vs `9c1eaa5`: data 0 · information 0 · interpretation 0. The four caption colour
  edits (F2 Orange → Red; F5 green loop → teal loop; F6 orange → red ×2) are outside every
  checked class, so they are listed by hand.
- `check_tex --compile`: **43 pages** (42 before; F1 is 6.0 in and gate2_strata 6.4 in at
  print size), 0 failures. D4 is checked at WP4.
- `render_all.py` from a clean tree at `0a013aa`: rc 0. It regenerated the tracked
  `figures/INDEX.md`, which **had gone stale before the revision** (no I1 runs, no F10). F4 was
  regenerated again after P4.

### Observed

- `analyze_omega.collect()` reads `sys.argv[1]` at import as a results glob. A caller with
  its own arguments (a preview script) silently made `omega_bands` return `None` ("no grokked
  runs found -- NOT SAVED"). `render_all.py` and the harness pass no such argument, so the
  paper path is unaffected. It is the same family as the scripts that ignore their argv
  (LAB_PROTOCOL.md).

### Found, not fixed (WP3, one decision each)

- In-figure wording that the register pass would change in prose: em dashes (gate2_strata
  ×2, F1, F9), internal identifiers ("C33", "(k09 P1: AMBIGUOUS)", script and
  pre-registration file names in footers), and capitalised emphasis (UNIT, BETWEEN, PRIMARY,
  FAILED, NOT).
- The k03 artifacts' footers print `unknown` as the SHA (basis_comparison, F8). This is the
  known 68-artifact gap, reported honestly.

### Not done

WP3 and WP4, the TMLR rebuild, the arXiv package, the three sweeps, the submission and Lean L0.

## 2026-09-29 — Entry 70: revision WP3, twelve section passes, Tables A and C, and three corrections to the main version (P5, P6, P6b)

Researcher: "start wp3", then "continue your work". Zero Kaggle quota; no training or analysis
job; every command ran in the foreground. **No claim changed status.** Every WP commit reports
`checks vs 9c1eaa5: data 0 · information 0 · interpretation 0`.

### What was done (commits `ec15c40` … `edbebce`)

| WP | section | hard lint before -> after | notes |
|---|---|---|---|
| 3.1 | Methods | 49 -> 0 | STE on Model & training, Classifying a run, the internal intervention, F3 caption; R5 detail read from code |
| 3.2 | Validation | 35 -> 0 | **Table A** (`tab:calib`); P5; P6 |
| 3.3 | Results I | 21 -> 0 | F8, basis_comparison captions STE |
| 3.4 | Results II | 50 -> 0 | tab:internal, F10, F4 captions STE |
| 3.5 | Results III | 55 -> 0 | gate2_strata, F9 captions STE |
| 3.6 | Results IV | 32 -> 0 | P6b; **Table C** (`tab:glance`, `b23df19`) |
| 3.7 | Discussion | new section | 7 paragraphs, 38 sentences, each with a recorded source |
| 3.8 | Limitations | 20 -> 0 | whole section STE descriptive |
| 3.9 | Setup | 19 -> 0 | F5, F2, F1, tab:moduli captions STE |
| 3.10 | Related work | 17 -> 0 | Table B **not** built |
| 3.11 | Introduction | 17 -> 0 | F0 caption STE procedural |
| 3.12 | Conclusion, appendices | 21 -> 0 | conclusion 247 words |

Paper-wide hard lint: only `00-abstract.tex` (13), which is WP4's. `test_paper_numbers.py`
**428 -> 514**, all pass; every new pin negative-controlled. Compile **48 pp**, 0 failures
(was 43; Tables A and C and float placement; D4's ≤38 pp main text is WP4's question).

### Three corrections to the main version (report to the researcher; each reversible)

- **P5 (`c2271b1`).** `11-implementations` said "Both exceed 1" of f = +1.110 (Gini) and
  f = +0.968 (‖W_E‖). **0.968 < 1** (PREREGISTER_o18_f64 F2; recomputed from
  `results/o18_f64`, W_E[:113] norms 11.595/12.187/11.304, mean 11.695). Wrong since
  `3abb6ed` (2026-09-16). Now "The Gini value exceeds 1".
- **P6 (`c2271b1`).** The n = 165 calibration pair **0.884 / 0.885** is Gini on the **energy**
  spectrum: `refcheck.py` and `analyze_scout.py` call `gini(energy(...))`. On amplitude (the
  Methods protocol) it reads **0.626 (their checkpoint) / 0.622 (our scout)**. The agreement
  stands (one pipeline both sides). Paper keeps the numbers, labelled "on energy".
- **P6b (`215d34c`).** §9's J-class indicator Ginis **0.829 / 0.829 / 0.683 / 0.018**
  (n = 121/165/120/113) are the same energy legacy (`analyze_scout.py`, DC bin included). On
  amplitude: **0.416 / 0.457 / 0.395 / 0.018**. Labelled "On the energy spectrum".
- Also corrected in FINDINGS §4.0: "PR 4.85 against their published 4.76" — 4.76 is **our
  scout's**, not published (Chen's PDF contains no 4.76; the §4.1 table and Entry 5 were right).

### Found reading the code for the methods detail (plan §5)

- **The engine logs every 100 steps**, not 200 (`run_n7.py` `LOG = 100`; N7 and I1 `hist`
  steps 0, 100, 200 …). The main version's "logged every 200 steps" was true of PyTorch only.
  So the engine's ten-sample window is 1,000 steps, the kernels' 2,000.
- Init is N(0,1)·1/√512 for W_out and 1/√128 for every other matrix **outside k08**, which
  scales the downstream weights. No bias terms. AdamW eps 1e-8, overridden by no script. The
  run seed sets both init and split. Grok step = first logged step with test acc > 0.99.
- The k03 n=125 s1 near-grok: last row **0.9779**, window median **0.976** — the same run,
  which the main version told twice as if two.

### Instrument defects found and fixed test-first

1. `check_tex` counted a caption as one paragraph whatever it held; `\par` now breaks a
   paragraph (a caption cannot hold a blank line). First version left the control word's tail
   ("ar …") in the next unit and counted it as a word; fixed in WP3.4 by mapping `\par` to the
   separator byte.
2. `check_tex` and `revision_checks` ignored `\caption[short]{long}` — the caption vanished from
   both. Fixed; planted.
3. `check_tex`'s self-reference check took the first `\label` anywhere in an unlabelled
   subsection as its own, so "see Table A" failed. A block's own label is now the one directly
   after its heading.
4. `revision_checks` offered deleted main units to the alignment heuristic, which attached them
   to whatever sentence shared a rare word; each neighbouring edit re-opened another WP's
   decision (four times in WP3.4–3.6). A main unit with a `u:` decision (`deleted`,
   `carried_by`, `dup_of`, `rewritten_as`) now skips the heuristic; `deleted` is accepted only
   for a unit with no datum (`deleted-data` flags otherwise).
5. A caption with `\par` breaks hyperref's `\NR@gettitle` unless it has a short title; every
   such caption now carries one.

### Commit-message counts were wrong twice and amended before push

WP3.1 (em dashes 10 -> 14, performative 7 -> 8, long 6 -> 9, decisions 32 -> 26), WP3.5
(decisions 7 -> 8), WP3.6b (decision breakdown), WP3.8 (bold leads 8 -> 10, decisions 9 -> 10).
From WP3.9 on, counts were computed before the message was written.

### Not done (next session)

Table B (every cell against its PDF, column-crop rule); the reproduction appendix's STE
procedural steps, sweep table and compute per sweep (plan §5); the in-figure wording WP2 held
(em dashes in gate2_strata/F1/F9, "C33", "(k09 P1: AMBIGUOUS)", caps) — plots.py edit + figure
harness + stamp commit; WP4 (abstract in STE, 3,522 chars vs arXiv's 1,920; whole-paper gates;
`build_tmlr.sh`; OPENREVIEW.md); the arXiv package; three sweeps; the researcher's approval of
P5, P6 and P6b.

## 2026-09-29 — Entry 71: P6/P6b fixed at the source, Table B, the reproduction appendix, the in-figure wording; P7

Researcher: "fix p5, p6, p6b then continue your work". Zero Kaggle quota; no training job; every
command in the foreground or a short background shell. **No claim changed status.** Every
commit reports `checks vs 9c1eaa5: data 0 · information 0 · interpretation 0`.

### P5, P6, P6b (`1dd45c9`)

- **P5** stands as applied in `c2271b1` ("The Gini value exceeds 1"; F2 = +0.968 < 1).
- **P6 and P6b are fixed where the numbers are produced, not labelled.** `refcheck.py` §[2] and
  `analyze_scout.py` called `gini(energy(...))`. Both now use `analyze_n7.mult_amplitude` /
  `add_amplitude` (amplitude, DC dropped, one bin per conjugate pair); PR stays on energy.
  Table A's n = 165 pair **0.885 / 0.884 → 0.629 (ours) / 0.637 (theirs)**; §9's J-class
  indicator Ginis **0.829 / 0.829 / 0.683 / 0.018 → 0.406 / 0.450 / 0.400 / 0.000** (n = 113 is
  exactly 0: the field's two indicators are a centred delta, whose DFT is flat).
- **WP3's interim "amplitude" values were not the protocol either.** 0.626/0.622 and
  0.416/0.457/0.395/0.018 were `gini(√energy())`: DC kept, and every self-conjugate bin scaled by
  1 where the others are scaled by 2. Reproduced exactly before being replaced. A value can carry
  the label "amplitude" and still be off-protocol — use the helpers, never a hand-rolled √.
- FINDINGS §3.1's scout table printed energy Ginis under an amplitude header; now protocol values,
  old ones kept in a dated note. `test_paper_numbers` bans any Gini of energy in both scripts. **The
  ban fired first on my own comment** in `refcheck.py`, which quoted the old call — a real negative
  control, and a reminder that a lesson quoting its token trips its own guard.

### 3b, Table B (`9ac7e5f`)

Seven studies × the plan's eight columns, every cell read from the PDF (two-column papers by
column crop); quotes in `docs/revision/table_b_sources.md`. "Not stated" = a grep for
`pre-regist|preregist` returns 0 in all six PDFs. Pinned structurally (shape, row order,
primality/square-freeness of every listed modulus, our row's counts from `tab:moduli` and
`tab:glance`), negative-controlled. **A distant edit moved an alignment:** Table B's new rare
tokens pushed the heuristic's ≤ 25-occurrence cutoff and re-attached the F7 caption fragment in
`11-implementations` (untouched) to the wrong group. Fixed by `u:a07211ae2d rewritten_as`, Entry
70's rule. **Found, not fixed:** 2606.23044's footer says "Proceedings of the 43rd ICML … PMLR
306"; the bib lists it as the workshop.

### 3c, the reproduction appendix (`b7107b3`)

`scripts/sweep_table.py` writes `tab:sweeps` and `tab:sweep-files` for the 17 sweeps from
provenance configs (k01/k02: summary JSON + the kernel's constants), the tracked Kaggle session
logs and the local run logs; `test_paper_numbers` asserts the paper's bodies equal its output
(planted 40,000 at k06 fails). STE procedure, five steps. **P7, awaiting approval:** F.1 said
`reproduce.sh` "re-runs every experiment and every analysis from a bare clone" — it trains
nothing and needs the artifacts. **Open:** the deposit that would ship `results/` and `logs/` is
designed, not built, and its plan (D1) covers `.npz` only; the hours column needs the logs.

### 3d, the in-figure wording

Seven strings in four figures (gate2_strata ×2 kinds, omega_bands, F1_stratum ×2, F9 ×2): em dashes,
"C33", "(k09 P1: AMBIGUOUS)", and capitalised emphasis, replaced by the paper's own words
("between both … ambiguous", "primary", "failed"). The full harness run flagged exactly those four
figures, text only, geometry and data equal; one `fig:` decision each (omega_bands' merged into a
`pairs` list with its WP2 footer pair).

### Counts

`test_paper_numbers` 514 → 536. Lint 13 hard (abstract, WP4's), 301 soft. Compile 49 pp.

**Addendum to Entry 71 (3d, same session).** The first re-check after the `fig:` decisions still
flagged gate2_strata: its "n/a" annotation is drawn **ten** times, not seven. The harness prints
at most eight strings per list, so "seven" had been read off a truncated printout. Three pairs
added. Then a scan of the rendered PDFs for all-caps words (`pdftotext | grep -E '\b[A-Z]{3,}\b'`)
found **"ONE"** in gate2_strata's G5 axis label, which WP2's held list had missed. The same check
at two letters found F9's verdict **"NO"** (beside "yes" and "partly", already bold and red). Both
were lowercased, with decisions, and both figures re-checked at 0 problems vs 9c1eaa5. **Read a
completeness list from the artifact, not from a printout that truncates.**

## 2026-09-29 — Entry 72: P7 approved (numbers, mathematics, experiments reproducible), WP4 done, a number-coverage gate, and four stale values (P8, P9a–c)

Researcher, session 31: "for P7 in these project all the numerical results and mathematical
results need to be reproducible nothing else", then "and every experiment done here is
reproduciable", then "start working on wp4" (away). **Zero Kaggle quota, no training job, no
claim changed status.** Every paper edit reports `checks vs 9c1eaa5: 0 · 0 · 0`.

### Mathematics (`4f10479`)

- `algebra.torsor(n)` asserts Proposition `prop:torsor` (ι_d a bijection onto J_d, |J_d| =
  φ(n/d)) at every d | n; the stratum identity (Theorem `thm:stratum`) now runs over **every
  n = 2..200** — **2,686,699 pairs, 1,097 divisors** — where it had run at four moduli. Both
  negative-controlled (a non-onto class; a wrong m at one pair).
- `test_paper_numbers`: the DFT of 1_{J_d} equals the Ramanujan closed form μ(q/g)φ(q)/φ(q/g) at
  every d, k and all 23 trained moduli (0 mismatches; dropping μ plants 1,900).
- **P8, found, not fixed (needs approval).** §9 line 32: "A function supported on the multiples
  of d has additive Fourier support on the multiples of n/d" is **false** as stated — δ_3 at
  n = 165 has 165 nonzero bins, and 1_{J_d} has 162 outside the multiples of n/d. It holds for
  1_{dZ/nZ} (support exactly {0, 55, 110}). It also contradicts §9's own later, corrected
  sentence (all 82 frequencies nonzero).

### Experiments (`bf68d61`)

`scripts/sweep_table.py --commands` prints, for all 17 sweeps, the stamped SHA(s) and the exact
command (env included: `ENGINE_DTYPE`, `N7_OUT`/`N7_GRID`/`N7_LOG_PREFIX`, log redirects,
`KAGGLE_OWNER`). `--selfcheck` (in `reproduce.sh`) asserts every stamped argv (29) is one its
command issues and nothing more, the dtype matches the artifacts, and every training-path file
changed since a stamped SHA is reviewed-inert at its current blob (`INERT`, six files — e.g.
`engine.py`'s NEP-50 fix, shown inert at float64 by the I1 retrain's bit-identical N7 reproduction;
`algebra.py`, imported by no training script). Three plants caught. Kaggle kernels at HEAD are
byte-identical to their stamped SHAs. **Not fixable after the fact:** the kernels set no
deterministic-CUDA flag, so a GPU re-run is reproducible to the §11 substrate gap (0.5791 CPU vs
0.5665 T4), not bitwise; k01/k02 unstamped and k03 `unknown` stay reconstructed by argument.

### WP4 (`b9f007b`, `049e39b`, `49d8bbd`, `1ed88f5`)

The abstract in STE descriptive: 12 sentences (mean 42.9 words, max 80, 13 hard) → 41 in eight
paragraphs (mean 13.8, max 22, 0 hard); every claim and number kept, in order; the pinned phrases
verbatim. "load-bearing rather than incidental" → "The output depends on those frequencies, and
they are not incidental" (the abstract's own gloss of what the protocol establishes). Decisions: 8
`u:` pairings, `u:473e2bfd01` re-pointed from WP1's merge (deficit computed: exactly one duplicate
copy plus WP1's decided layout words), a pin for the re-floated `509b601875`, 5 `c:` groups.
Paper-wide vs 9c1eaa5: mean words/sentence 22.2 → 17.5, em dashes 173 → 0, in-sentence bold 103 →
0, semicolons 88 → 54, contrasts 96 → 86, no sentence > 40 words. TMLR build PASS (50 pp,
references p. 35, D4 met, abstract 3,774 chars, supplement 999 KB). **Public export:** it was
three weeks stale; rebuilt, it needed 12 `KNOWN_ABSENT` entries (docs/revision/*, drafting aids)
and one relabel (the skills path); **it still fails, by design, on the unfilled deposit-DOI
sentinel**, so `check_public --selfcheck` aborts until the deposit exists. Left for the
researcher: G_m is `^{*}` in the abstract and `^{\times}` in §8 (a symbol change is a data change).

### The number-coverage gate (`ef9c1f5` … `034822f`)

`test_paper_numbers` records every value a **passing** check compares, tokenises every number in
`paper/sections` (scientific notation whole) and requires each to round to an asserted value at
its printed precision, or to have an entry in `paper/number_ledger.json` keyed to its exact line:
`exempt` with a reason (15: cited values, a pre-registered threshold, the Python version, one wall
time) or `todo` (the debt, printed every run). An unledgered uncovered number fails; so does a
reasonless exemption (both planted). A `todo` a check covers must be deleted once a real pin
exists or marked `coincidental` (matching is by value — a gap it reports is real, a coincidence can
hide one). Debt **211 → 74** in four batches of pins against the scripts' own printed output (one
implementation): §6 (C6, C35, C21, tuning table, splits), §9 (CRT table, O20, ω W1–W6, 18-modulus
bands), §7 (N4 ranges, heavy-tail sentence, I1 control tail and dose), §10 (gate-1 circuit, C12 at
full value). `test_paper_numbers` 536 → **601**. Two scaled assertions (G2's 2.71 / 3.98 compared
as mantissas) were invisible to the gate and now assert the full value.

**Two entry-point gaps closed (`5c3bed0`):** `reproduce.sh` ran `test_crt_null.py` only on N7 and
`analyze_omega.py` only in its default mode, so **O20's table and C37's W1–W6 regenerated from no
command it issued.** Both added. `test_reproduce` checks that a script is present, not which of
its modes run — that is how both slipped.

**P9, found, not fixed (needs approval; each reading unchanged):**
- **P9a** §6:83–84 and appendix 14:89–90: C19's ρ(−φ, steps/φ) = +0.875 and ρ(φ, steps) = −0.668
  are Spearman on **ordinal ranks** (φ ties at 40: n = 75, 100). `analyze_k04.py` prints the
  midrank **+0.870 / −0.665**; both reproduced from its own `per_modulus` rows.
- **P9b** §9:177: "ρ = +0.862 across 18 moduli" is ordinal ranks on the pre-k09 ω population
  (`analyze_omega.collect('results/k0[0-8]*/WE_*.npz')`); midrank **+0.858** (23 moduli: +0.824).
  Not the ledger's C9 (k04 P2, 0.786), which is correct.
- **P9c** §9:180–181 (and FINDINGS 1448): "control flat at 0.117–0.157" reproduces on **no**
  population; the code reads **0.117–0.155** at 18 and 23 moduli, as PREREGISTER_omega.md
  records. `rand_amplitude` seeds each call, so it is not an RNG-order effect.
These are the midrank defect's survivors: the 2026-09-23 audit fixed the code and §9's partials,
and three printed values outside that sentence kept the ordinal numbers.

### P7 in the paper (`6f0f712`)

Approved with the researcher's scope. The appendix now says what the gate guarantees (it had said
`test_paper_numbers` "calculates every number … again", not yet true), that the self-checks
exhaust `prop:torsor` and `thm:stratum` to n = 200, and that a sweep re-trains from
`sweep_table.py --commands`, with a GPU re-run not guaranteed bit-identical. 13 `n:` decisions.

## 2026-09-29 — Entry 73: P8 and P9a–c applied, the number debt paid to its findings (74 → 12), nine new corrections (P10–P18), the arXiv package

Researcher, session 32: "approve p8 and p9, then pay the number debt then the arxiv". **Zero
Kaggle quota, no training job, no claim changed status.**

### P8, P9a–c (`de60fc1`), approved

- **P8** §9: "A function supported on the multiples of d has additive Fourier support on the
  multiples of n/d" → "The indicator of the multiples of d …". The general form is false (δ_3 at
  n = 165 is nonzero at all 165 bins). The indicator form is asserted at every d | n for n = 120, 165.
- **P9a** §6 and §14, C19 (retracted): ρ(−φ, steps/φ) +0.875 → **+0.870**, ρ(φ, steps) −0.668 →
  **−0.665** (midranks, as `analyze_k04.py` prints). **P9b** §9: +0.862 → **+0.858**. **P9c** §9:
  control band 0.117–0.157 → **0.117–0.155**.
- The same defect, in the record only: FINDINGS/STATE ρ(ω, zdd) +0.674 → **+0.766**, within ω = 2
  +0.533 → **+0.502** (midranks, 18 moduli). `PREREGISTER_omega.md` keeps its committed values.
- Checks vs 9c1eaa5: 0 · 0 · 0 (four `c:` decisions). Planted old values fail four checks.

### The number debt (`cb17100`, `ecc6c76`, `3289003`): 74 → 12

Pinned against the scripts' own output or the artifacts: §8's Gate 2 prose (G3 ranges, n = 49 s2's
window and stratum, the product-character range, C7's single-seed table from `jclass_spectra.py`);
§5's worked examples (the 21×–440× spread **reproduced exactly through `analyze_n4.run`**, 20 draws,
control seeds 0–7; the 137-test family; the energy/amplitude example is k07 s0; Gini noise from k03's
trajectories; PR drift); §6's prime powers (analyze_k04 P4, analyze_k05 Q2); §7's I1/I1b doses and
C17's k = 23; §11's implementation facts (test_model, O18 tuning, O21 probe/numerics); §12's 0/453;
§14's 4,600. `test_paper_numbers` 601 → **650**. A planted 0.8908 → 0.8909 fails the gate.

**Every one of the 12 remaining entries is a finding, not debt.** Found, **not fixed** (each awaits
the researcher; the paper is unchanged):

| id | where | what the paper says | what the artifact says |
|---|---|---|---|
| P10 | §8 | 13 stratum-runs fall to 0.500–0.833 | 13 is the count of **strata**, 0.500 a seed mean; **15 stratum-runs, 0.379–0.833** |
| P11 | §8 | n = 49 s2's last ten samples rise **monotonically** | one dip (0.8489 → 0.8477); endpoints right. LAB_PROTOCOL.md's copy corrected |
| P12 | §3, §5 | "the same model reads 0.4% raw and 98.4% dlog" | two statistics. k07 s0: neuron terms 0.36% / 21.7%; logit top-8 16.95% / 98.37% |
| P13 | §5 | energy 0.902 where amplitude reads 0.539 | the checkpoint (k07 s0, which gives 0.902, 12.5, 4.31) reads **0.543** |
| P14 | §5 | 137 ablations "across the seven sweeps" | eight sweeps plus the retired pre-leak-fix arm (R1's family table) |
| P15 | §7 (C23) | separates by 10.8 to 59.6× | the **retired** excluded/mean(20-draw control), legacy `results/n4_ablation.txt` |
| P16 | §7 | n = 123 "restricted/baseline only 1.32×" | **baseline/restricted**, a mean of three (1.76, 0.003, 2.19) — the upside-down column again |
| P17 | §7, F4 caption | damaging draws "reach only 7.9" | the B = 200 null's max (7.92); at B = 10,000 it is **10.93** |
| P18 | §11, §14 C10 | forward 6.1e-16, gradients 2.3e-15 | Entry 10's first measurement; `test_model.py` now prints **4.441e-16 / 2.68e-15** |

Unresolved, not findings yet: §7's 80.8% key-amplitude share (the protocol helper reads 81.6%; the
recorded 263.552 comes from a path not found); C20's 0.918 (an ad hoc noise control no script runs).

P10–P18 are all the same shape as every "two documents agree" entry in LAB_PROTOCOL.md: a number in prose
that no check scored. **The coverage gate found them by refusing to let a printed number stand
unasserted**; pinning each one meant re-deriving it, and nine did not re-derive as written.

### The arXiv package (`fcc816c`, `db54c6c`)

`scripts/build_arxiv.sh` → `for arxiv/`: the source tarball (tmlr `[preprint]`, comment lines
stripped and trailing comments refused, `../figures` rewritten, `main.bbl` shipped), a PDF compiled
**from the tarball** in an empty directory without BibTeX (50 pp, 540 KB source), the metadata
abstract (1,784 of 1,920 chars, abridged by selection, and the build fails on any number the paper's
abstract does not print) and `SUBMIT.md`. Rules checked live on arXiv's help pages. **The build
refuses without `--draft` while `\addr [affiliation]` is a placeholder.** Waiting on the researcher:
affiliation, licence (CC BY 4.0 recommended), categories (cs.LG recommended), and P10–P18 before
posting, since an arXiv version is permanent.

## 2026-09-29 — Entry 74: P10–P18 applied, the number debt down to its two unresolved sources, the arXiv licence

Researcher, session 33: "fix all p10-p18, what 12 entries are left when do we fix them? use licenses
as per your recommendation." **Zero Kaggle quota, no training job, no claim changed status.**

### P10–P18, approved and applied

Each sentence now prints its artifact's value (Entry 73's table), each has a pin in
`test_paper_numbers.py` on the new text (650 → 666 checks), each ledger entry is deleted, and each
aligned group carries a `c:` decision. **Negative control:** with `paper/sections` stashed back to the
old text the gate fails eleven checks, all nine findings among them.

| id | now prints | source |
|---|---|---|
| P10 | $15$ stratum-runs fall to $0.379$–$0.833$ | `results/gate2/k03_grid_acts.json` |
| P11 | "rise, with one small dip, from 0.842 to 0.963" | k04 n=49 s2 `hist` |
| P12 | top eight logit components: $17.0\%$ `a+b` raw, $98.4\%$ dlog (§3 and §5) | `logit_top_components`, k07 s0, `analyze_k07`'s formula |
| P13 | Gini on amplitude $0.543$ | `mult_amplitude`, k07 s0 |
| P14 | "across eight sweeps and two runs of a retired engine arm" | nine `*_n4_ablation.npz` |
| P15 | excluded / median control $10.0$ to $4.6\times10^{7}$ | k03 ablation, 15 runs at 119/120/165 |
| P16 | baseline/restricted median only $1.76\times$, range $0.003$–$2.19\times$ | k09 ablation, n=123 |
| P17 | F4 caption: damaging draws "reach at most $10.9$" | `analyze_n4.run`, B = 10,000 |
| P18 | forward $4.4\times10^{-16}$, gradients $2.68\times10^{-15}$ (§11, C10 row) | `test_model.py` |

P12 was the one wording choice: the logit top-8 share keeps C21's verified 98.4% and puts both numbers
on one statistic (the neuron-term alternative reads 0.36% / 21.7%). P16's sentence grew to 41 words
and was split at its conjunction. Checks vs 9c1eaa5: **data 0 · information 0 · interpretation 0**;
compile 50 pp, 0 failures; style 0 hard.

The old values were corrected in FINDINGS (annotated with the P-id, not erased), STATE's C10/C36 rows
and Gate 2 paragraph, LAB_PROTOCOL.md's raw-vs-dlog gotcha and the `write-paper` skill. Historical
narratives that quote what was believed at the time ("C10 verifies gradients to 2.3e-15") stay.

### The two ledger entries left

Neither is a sentence fix. §7's **80.8%** key-amplitude share: the protocol helper reads 81.6%, and
FINDINGS' 263.552 comes from a path not yet found. §14's **0.918** is the C20 flat-variance noise
control (C20 is retracted), which no script runs. Both go into queue item 6 before the non-draft
arXiv build; any changed value becomes a new P-id for approval.

### arXiv

Licence **CC BY 4.0**, per the recommendation (recorded in `for arxiv/SUBMIT.md`). Affiliation and
categories remain the researcher's.

## 2026-09-30 — Entry 75: the affiliation and category, an identity leak into the anonymous supplement, F5's arrows, and AISTATS 2027 dropped

Session 33 continued after Entry 74. **Zero Kaggle quota, no training job, no claim changed status.**

### The arXiv metadata is settled

The researcher supplied the affiliation (department and university) and chose **cs.LG**, with the
licence already set to **CC BY 4.0** (Entry 74). `paper/main.tex` carries the affiliation (`8af5a71`).
The TMLR build hides the author block and the public export empties it. The affiliation is a new
identifying string, so the TMLR build's PDF and supplement grep and `check_public.py`'s author list
now include it. The short form of the university name was *not* added: it matches "sustain".

### An identity leak the gate caught (`533da1f`, `445e122`)

The first full TMLR build since session 32 **failed its own public gate**, and correctly so:
`for arxiv/SUBMIT.md`, tracked since session 32, names the author and was being copied into the
**anonymous** supplement. Nothing excluded it, because a new tracked directory joins the export by
default. The fix excludes `for arxiv/` beside `TMLR/` and lists `for arxiv/SUBMIT.md` and `SUBMIT.md`
as known-absent paths, each with a reason. The same run flagged `results/n4_ablation.txt` as a
dangling reference: Entry 73 names that legacy, never-tracked file. It is allowlisted with a reason,
and the record is not rewritten. After both fixes: **0 violations**.

The builds refuse a dirty tree, and at the time another agent's untracked `AISTATS2027/` files made the
tree dirty. So both packages were built in a detached `git worktree` of HEAD: `.venv` symlinked (and
`/.venv` added to `.git/info/exclude` for the duration), `figures/` copied with `-p` so their mtimes
pass the freshness check, and `results/` hard-linked (`cp -rln`), so no gigabytes were copied and no
tracked file changed. The outputs were copied back and the worktree removed.

### F5's arrows (`1c0a241`)

The researcher found `descent_lattice`'s arrows clunky. Every arrow is now a `FancyArrowPatch`, sized
in points like the scatter marker, so it sits relative to the node's outer edge, `sqrt(s)/2 + lw/2`,
at any figure size. Two things were found by measuring at 7200 dpi, not by reasoning:

- **A filled `-|>` head stops 1.01 pt short of its end point**, constant across line width 0.4–1.2
  and mutation scale 6.5–13, in shrink mode and path mode alike (matplotlib 3.11.1, as pinned). With
  the tip aimed 1.0 pt inside the edge, tip and tail both measure **0.02 pt** from the node's outer
  edge. My first attempt *added* a mitre allowance, which was the wrong direction.
- **On a loop this tight, matplotlib's head placement cuts the curve at the wrong crossing.** A
  6.2 pt loop with a 3.6 pt head hung its head below the loop, pointing down, with both a polyline
  and exact `Path.arc` cubics. It is fixed by drawing the arc as a plain line that ends at the head's
  base and the head as its own two-point patch along the chord. A separate bug of mine also showed:
  the sweep carried a spurious −2π, so the loop wound 600° instead of 240°.

`figure_checks.py`: F5 data OK, **0 problems vs 9c1eaa5**, smallest text 7.0 pt. All figures were
re-rendered, and both packages were rebuilt from `1c0a241` (50 pp each; TMLR gate 0 violations).

### AISTATS 2027 is dropped

Researcher, 2026-09-30: "i no longer plan to submit on the aistats 2027." The directory was deleted
by the researcher; its tracked style pack's deletion is committed, and the export's exclusion for it
is removed. The venues are now TMLR (submission) and arXiv (preprint).
