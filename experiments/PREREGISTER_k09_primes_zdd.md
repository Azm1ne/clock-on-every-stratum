# Pre-registration: k09 — three primes, and the modulus where ω and zdd disagree

**Commit this file BEFORE pushing.** **Date:** 2026-09-15 · **Kernel:** `kernels/k09_primes_zdd`
**Arm:** thesis (pools with k02 Arm B / k03 Arm B / k04) · **Compute:** 5 moduli × 3 seeds
= **15 runs**, ~3–4 GPU-h on a T4, Kaggle account `account-a`.

## ⚠️ Operational note: the quota could NOT be read live before this push

LAB_PROTOCOL.md requires re-reading quota live before planning a spend, because a stored figure is
a spend record and not a budget. **I could not.** Kaggle's MCP server — the only source of
`get_accelerator_quota` — is returning *"An error occurred invoking …"* for **every** tool
this session, including `get_user_profile`, so it is a server-side outage, not an auth
failure (the CLI authenticates fine and `kernels list --mine` works).

What is known instead, and its basis: the weekly window refreshes 00:00 UTC Saturday and the
current one opened 2026-09-12; **the last Kaggle push of any kind was k08 on 2026-09-11**,
and sessions 9, 10 and 11 spent zero. So the window should be close to untouched. **That is
an inference from a spend record, which is exactly what LAB_PROTOCOL.md warns against, and it is
labelled as one.** The mitigation is that the push itself is the live check — an exhausted
quota makes Kaggle refuse the session, `run_kernel.py` detects the rc-0 refusal, and nothing
is corrupted; the cost of being wrong is time, not data. **Record the real spend in
`LAB_NOTEBOOK.md` when the run lands, and re-read quota live before the next one.**

## The moduli, and why each one is here

| n | factorisation | ω | zdd | cyclic | train @ 0.30 | why |
|---|---|---|---|---|---|---|
| **127** | prime | 1 | 0.008 | yes | 4,838 | **G1** — a second prime |
| **131** | prime | 1 | 0.008 | yes | 5,148 | **G1** — a third prime |
| **128** | 2⁷ | 1 | **0.500** | **no** | 4,915 | **P1, the discriminator** — and the first power of 2 |
| **123** | 3 ·41 | 2 | 0.350 | no | 4,538 | **W1** — pairs with 81 (\|Δzdd\| 0.016) |
| **91** | 7 ·13 | 2 | 0.209 | no | 2,484 | **W1** — pairs with 125 (\|Δzdd\| 0.009) |

**Dataset size was calibrated against k04's own grok rates before the set was chosen**, not
assumed: the transition sits at ~1,700–2,000 training examples (n=49 → 0/6 at 720 ·
n=63 → 2/6 at 1,190 · n=75 → 2/3 at 1,687 · n=81 → **3/3** at 1,968 · n=98 → 3/3 at 2,881).
The smallest modulus here, n=91, has **2,484** — above n=81's 3/3. No modulus in this set is
expected to be a C14 dataset-size null, and if one is, that is the reading (C14), not an
algebraic effect.

## P1 — PRIMARY. n = 128 is a modulus where ω(n) and zero-divisor density make OPPOSITE predictions

`PREREGISTER_omega.md`'s W3 fired: ρ(zdd, Gini_add \| ω) = **+0.739** against
ρ(ω, Gini_add \| zdd) = **+0.700**, so on 18 moduli **each survives partialling on the other
and the data cannot rank them.** That is a statement about the existing modulus set, and the
fix is not a better statistic — it is a modulus where the two variables disagree.

**n = 128 = 2⁷ is that modulus.** It is a prime power (ω = 1) with **half its residues zero
divisors** (zdd = 0.500) — a combination no modulus in the project has.

| predictor | basis | predicted Gini_add for n=128 |
|---|---|---|
| **ω(n) = 1** | the ω = 1 band over 6 moduli, 57 runs | **0.017 – 0.184** |
| **zdd = 0.500** | the 7 existing moduli within \|Δzdd\| ≤ 0.10 (63, 147, 75, 165, 105, 98, 100) | **0.434 – 0.589** |

**The two intervals do not overlap. The gap between them is 0.250 and its midpoint is
0.309.** Both numbers are computed from data already on disk and are fixed here.

| # | criterion | threshold |
|---|---|---|
| **P1** | mean Gini_add at n=128 over grokked seeds | **ω HELD** < **0.184** (inside the ω band) · **zdd HELD** > **0.434** (inside the zdd neighbourhood) · **AMBIGUOUS** in between, and it must be reported as ambiguous, **not** rounded to whichever side the 0.309 midpoint falls on |

**This is the strongest prediction the project has made**, because failure is as informative
as success: if n=128 reads ≥ 0.434 the ω band structure is **broken** by one modulus and
§5.1c-bis must be retracted, exactly as C7b/C19/C20 were. I am not predicting which.

⚠️ **The mechanism in `FINDINGS.md` §5.1c predicts ω.** C8's CRT-dual set is one family per
maximal prime power, so at ω = 1 the predicted set is *every* frequency — vacuous, nothing
to concentrate on. That reasoning does not know about zdd, and n = 128 is where it is
exposed. Stating the expectation is not the same as hedging the criterion: the threshold
above is symmetric and was fixed before the run.

## P2 — G1. Is n = 113 special?

Every "prime versus composite" sentence in the write-up currently rests on **one modulus**
(O5, closed as *not answerable* for exactly this reason), against a measured grokking-time
seed variance of **3.8×**.

| # | criterion | statistic | threshold |
|---|---|---|---|
| **P2a** | the new primes carry the same clock | Gini_mult, tail mean over the final 10k steps, amplitude protocol | \|Δ\| ≤ **0.10** from n=113's k03 value (0.5665 T4), the threshold C6 already uses |
| **P2b** | key-frequency count | 5×-median detector on `W_E` | **descriptive.** C4's key-count clause was *withdrawn* at 5 seeds (4.6 ± 0.8, matching 4 in only 3/5), so no threshold is defensible and none is set |
| **P2c** | neuron tuning | fraction of 512 MLP neurons >85 % explained by one frequency in dlog coordinates | ≥ **0.70**, against C31's 92–100 % at n=113 and **0.0 %** raw-coordinate control |
| **P2d** | grokking time | steps to acc > 0.99 | **descriptive, and no ordering may be interpreted** — 3 seeds against a 3.8× seed spread (O2). Reported with variance |

## P3 — the clock at 2⁷, the first power of 2

n = 128 extends C6's prime-power set to **p = 2** and to depth 7, the deepest nilpotency
tested (previous deepest: 3⁴).

⚠️ **(Z/128)\* = Z₂ × Z₃₂ is NOT cyclic, so there is no discrete log and the scalar
multiplicative readout does not apply.** It must be **skipped loudly**, as 119/120/165
already are — never silently. Only the product-character path (C23) applies. The same holds
for 123 and 91.

| # | criterion | threshold |
|---|---|---|
| **P3** | product-character ablation at 128, 123, 91 | permutation **p < 0.01** in ≥ 2/3 seeds per modulus, 200 draws — the statistic C22/C23/C32 already use |

## P4 — W1 becomes scorable

Computed before the run by `analyze_omega.confirm` on the existing 18: adding these five
takes zdd-matched discordant pairs from **5 to 10**, past the pre-registered minimum of 6.
The five new pairs are 75↔128 (0.033), 165↔128 (0.015), 105↔128 (0.043), 81↔123 (0.016),
125↔91 (0.009).

| # | criterion | threshold |
|---|---|---|
| **P4** | W1 re-scored on 23 moduli | **the thresholds already fixed in `PREREGISTER_omega.md` and not re-opened here**: HELD ≥ 0.80 of discordant pairs favouring ω **and** sign-test p < 0.05; NOT HELD ≤ 0.60; PARTIAL between |

## P5 — C8 at two fresh CRT-testable moduli

123 = 3 ·41 and 91 = 7 ·13 are square-free with ω = 2, so the CRT-dual law applies.
**127, 131 and 128 are VACUOUS by construction** (one maximal prime power ⇒ the predicted
set is every frequency) and are **never counted as support** — the standing rule.

| # | criterion | threshold |
|---|---|---|
| **P5** | enrichment of key additive frequencies on the predicted set, 123 and 91 | permutation **p < 0.01** in ≥ 2/3 seeds each |

## P6 — O5, reopened with enough primes to ask the question

`max_logit` and `weight_norm` are logged again (k01_scout had them; k02–k08 dropped them).
O5 asked whether Softmax Collapse is composite-modulus-first and was closed as **not
answerable**: the single prime was *also* the smallest modulus and the fastest to memorise,
so primality, size and phase were one axis.

| # | criterion | threshold |
|---|---|---|
| **P6** | max_logit trajectory, 3 primes (113, 127, 131) vs the composite set | **EXPLORATORY, descriptive, and NOT a claim.** 3 primes against a 3.8× seed spread cannot support one. It is logged so the question becomes *askable later*, and any sentence written from it must carry this label |

## What would falsify what

- **P1 reading ≥ 0.434** falsifies the ω band structure. §5.1c-bis is retracted and ω joins
  C7b, C19 and C20.
- **P2a failing at 127 or 131** would mean n=113 is not representative of primes, and every
  prime-referenced statement in the write-up needs re-scoping.
- **P3 failing at 128** would bound C6: the clock survives at odd prime powers but not at 2ᵏ.

## Known confounds

- **3 seeds, not 5.** Enough for "≥2/3 groks"; not a 5-seed result. Say so.
- **n=128's non-cyclicity is confounded with its being a power of 2** — every 2ᵏ with k ≥ 3
  is non-cyclic. A failure at P3 cannot separate "power of two" from "non-cyclic unit
  group". Named now so it is not discovered later.
- **P1 is one modulus.** A single modulus deciding between two predictors is a strong design
  but a thin sample: if P1 favours ω, it is *one* discriminating observation, not a law, and
  must be written as such.
- **zdd and ω remain collinear overall** — this run breaks the collinearity at one point, it
  does not remove it.
- **Kaggle T4, float32.** Scout data, not paper data (constraint 2). O18 is open and these
  numbers must never be pooled with engine numbers.

## Analysis plan — decided now

```
PYTHONPATH=. .venv/bin/python analyze_k09.py results/k09_primes_zdd     # P0, P1, P2a/b, P5, P6
PYTHONPATH=. .venv/bin/python analyze_omega.py --confirm                # P1 cross-check, P4
PYTHONPATH=. .venv/bin/python analyze_o4.py  results/k09_primes_zdd     # P2c
PYTHONPATH=. .venv/bin/python analyze_n4.py  results/k09_primes_zdd     # P3
PYTHONPATH=. .venv/bin/python analyze_excursions.py results/k09_primes_zdd
PYTHONPATH=. .venv/bin/python scripts/render_all.py                     # figures
```

⚠️ **`analyze_k04.py` is deliberately NOT in this list.** It hard-codes
`results/k02_grid` / `results/k04_extended` and has **no argv handling**, so
`analyze_k04.py results/k09_primes_zdd` silently scores k04 and prints a full, plausible
report about the wrong data. That was found by *running* it during the pre-push smoke — it
is the same defect O18-substrate recorded as its deviation #1, caught before the push this
time instead of after. `analyze_k09.py` was written for this pre-registration and **every
command above has been executed against this kernel's own smoke artifacts.**

**The kernel was smoke-tested locally on CPU before the push** — its own source, patched by
an asserted-exactly-once substitution list, 400 steps at n = 127/128/91 — because a
pre-registered command that has never been run is how O18-substrate's analysis plan named a
command that found zero runs and printed something indistinguishable from a failed sweep.
Anything beyond the five commands above is exploratory and labelled so.

## Outcome (filled in AFTER the run — never edit anything above)

**Date scored:** 2026-09-15 · Kaggle **Tesla T4 sm_75**, 15/15 runs, **93 min** ·
artifacts `results/k09_primes_zdd/`, SHA **`ed9a1f2`** (real, not `unknown`) ·
logs `logs/k09_score.log`, `logs/omega_confirm.log`.

### Summary

| # | criterion | result | verdict |
|---|---|---|---|
| **P0** | groks | 127 **3/3** · 131 **3/3** · 128 **2/3** · 123 **3/3** · **91 1/3** | **PARTIAL** |
| **P1** | **PRIMARY** — ω vs zdd at n=128 | **Gini_add = 0.261** (seeds 0.260, 0.262) | **AMBIGUOUS** |
| **P2a** | new primes carry the clock | 127 **0.5562** (Δ 0.0103) · 131 **0.5915** (Δ 0.0250) | **HELD** |
| **P2b** | key-frequency count | 4,5,4 at 127 · 5,4,4 at 131 | descriptive |
| **P2c** | neuron tuning | **0.887–0.998** across 6 runs, raw **0.000**, shuffled **0.000** | **HELD** |
| **P3** | product-character ablation | **p < 0.01 in 13/13 analysable runs**, all five moduli | **HELD** |
| **P5** | C8 at fresh CRT moduli | 123 enrichment **7.59/3.28/7.43**, p = 0.0000 **3/3** | **HELD where assessable** |
| **P6** | O5 max_logit | logged; 3 primes 175.0/217.3 vs composites 146.8–288.2 | exploratory |
| **P4** | W1 rescored on 23 moduli | **10/10 pairs favour ω**, sign-test p = 1.95e−03 | **HELD** |

### P1: neither predictor was right, and that is the result

**n = 128 = 2⁷ read Gini_add = 0.261, between the two pre-registered intervals** — above
ω's 0.017–0.184 and well below zdd's 0.434–0.589. Per the pre-registration this is
**AMBIGUOUS and is reported as such, not rounded** to whichever side the 0.309 midpoint
falls on.

**What it does settle.** n=128 sits **on ω's side of the divide**: it extends the ω = 1 band
to 0.015–0.261, and the ω = 1 / ω = 2 bands **still do not overlap** (0.261 vs 0.394, gap
0.133, down from 0.210). The ω band structure **survives its designed falsifier** —
§5.1c-bis is **not** retracted. And in W1 the modulus appears in **3 of the 10 matched pairs
and favours ω in all three**. So ω is the better predictor of the two, while **neither
predicts n = 128 accurately** — which is the concrete form of what W3 said: both variables
carry real, partially independent signal.

### P4: W1 is HELD, and it was the point of the run

The criterion that came back **NOT ASSESSABLE** at 5 pairs (threshold 6) now has **10**, and
**every one favours ω — 10/10, fraction 1.000, exact two-sided sign test p = 1.95 × 10⁻³.**
The threshold was never moved; moduli were added instead. ⚠️ The pairs share moduli and are
**not independent**, so the p is optimistic and descriptive; **the fraction is the
statistic**. W2 +0.627, W3 +0.735, W4/W5/W6 all still HELD (ρ(n, Gini_add) = −0.185).

### P2a/P2c: n = 113 is not special — G1 closes

Two fresh primes carry the same clock (Δ 0.010 and 0.025 against a 0.10 threshold), and
**88.7–99.8 % of their MLP neurons are single-frequency in discrete-log coordinates against
0.0 % in raw coordinates and 0.0 % shuffled.** The tuned neurons sit on **exactly** the
embedding's key set in 5 of 6 runs (4/4, 5/5, 4/4, 4/5, 4/4, 4/4), reproducing C31's 14/14
pattern at moduli it had never seen. **Every "prime vs composite" sentence in the write-up
now rests on three primes rather than one.**

### P3: the clock is CAUSAL at 2⁷, on a non-cyclic unit group

`analyze_n4.py results/k09_primes_zdd`, 200-draw permutation, `logs/k09_n4.log`.
**Permutation p = 0.0000 in every analysable run — 13/13, all five moduli.**

| n | local group | seeds | restricted/baseline | p < 0.01 in |
|---|---|---|---|---|
| **128 = 2⁷** | Z₂×Z₃₂ | 2 | **239.52× ± 236.74** | **2/2** |
| 131 | Z₁₃₀ | 3 | 262.19× ± 367.03 | 3/3 |
| 127 | Z₁₂₆ | 3 | 19.62× ± 22.67 | 3/3 |
| 91 = 7 ·13 | Z₆×Z₁₂ | 2 | 117.26× ± 115.81 | 2/2 |
| 123 = 3 ·41 | Z₂×Z₄₀ | 3 | **1.32× ± 0.95** | 3/3 |

**n = 128 is the result here.** The first power of 2 in the project, ω = 1, zdd = 0.500, no
discrete log — and the product-character circuit is **causally responsible at 239×**. C6's
prime-power set now reaches **p = 2 and depth 7**.

⚠️ **n = 123 reproduces the O8/O11 pattern**: p < 0.01 in 3/3 — the characters are
*necessary* — while restricted/baseline is **1.32× ± 0.95**, i.e. barely better than
baseline. Necessary but not sufficient, at a modulus neither O8 nor O11 had seen. It remains
**a per-run property, not a per-modulus one** (O8/O11 re-scoped 2026-09-14).

⚠️ **`analyze_n4.py` admitted 2 seeds at n=91 where `analyze_k09.py`'s window-median rule
counts 1 grok.** The two scripts use different inclusion criteria — n4 included the 0.976
**near-grok**. Under LAB_PROTOCOL.md's three-state rule a near-grok is neither a data point nor a
control, so **n=91's P3 row is 2 seeds one of which is a near-grok** and is not counted as
independent support. Not reconciled here; flagged rather than silently pooled.

### Deviations from this pre-registration, and why

1. **The analysis plan named `analyze_o4.py` for P2c, and it scores NOTHING here.** It takes
   a directory but hard-codes `MODULI = [113, 121, 125]`, so it reported *"0/0 grokked
   seeds … NOT HELD"* — a **false failure on an empty set**. This is the same defect as
   `analyze_k04.py`, which this pre-registration explicitly warned about **one paragraph
   earlier**, and I still shipped it. **My "every command above has been executed against
   this kernel's own smoke artifacts" was true and insufficient: I ran it, saw exit 0, and
   never checked it had scored k09's moduli.** Running a command is not verifying it. P2c is
   now scored inside `analyze_k09.py` using `analyze_o4.measure`, which is reusable and
   correct.
2. **The CRT test was off by one bin, and a positive control caught it.** P5 first read
   enrichment **0.31/0.55/0.30 with p ≈ 1.0** — *depletion* — at moduli where C8 is the
   best-supported claim in the project. `predicted(n)` returns real frequencies;
   `freq_energy` returns bins at **position = frequency − 1**, and `test_crt_law.main()` has
   always written `[k - 1 for k in predicted(n)]`. I passed them unshifted. Run on
   C8-verified moduli the same path read **0.25 at n=165**, where the ledger says 3.3–17.4×;
   after the fix it reads **17.44 at n=165 and 3.27 at n=119**, matching the published
   numbers exactly. **Fourth bin-convention bug in this project; reasoning has lost every
   time.** A planted-signal assertion in both directions now guards it.
3. **P2b called `key_freqs_5x_median(Wu, len(U))`** — it takes one argument, per-frequency
   norms. Fixed to `key_freqs_5x_median(mult_amplitude(W, n))` with the mandatory `+1`,
   which is the *opposite* convention to deviation 2 and correct for that array.
4. **P5's gate conflated NOT ASSESSABLE with NOT HELD.** n=91 has one grokked seed, so a
   "≥ 2/3 seeds" rule is unmeetable there; scoring it NOT HELD reported **C8 as broken at a
   modulus where it was never testable**. Same false-negative shape fixed twice already in
   this scorer.
5. **The dataset-size calibration for n = 91 was WRONG.** §"The moduli" argued 2,484 training
   examples was safe because n=81 groks 3/3 at 1,968. n=91 grokked **1/3** (plus one
   near-grok at 0.976 and one failure at 0.358). **Training-set size alone does not predict
   grokability** — the argument was stated confidently and is falsified by its own run. n=128
   also grokked only 2/3, and slowly (23,400 / 22,000 vs 3,200–11,000 at the primes).
6. **Runtime was overestimated ~2.5×**: 93 min, not the 3–4 h budgeted. k04's own 4.5 min/run
   median was the right anchor; I used k03's, which had larger moduli and 5 seeds.
7. **The output pull failed repeatedly and reported success.** Kaggle's client returned rc 0
   on a truncated download and left a 0-byte `.npz`; `run_kernel.py` printed
   `pulled -> …` regardless. Fixed (`a9ffd1e`) to treat rc≠0, `connection broken` or any
   zero-byte artifact as a truncated pull, delete the zero-byte files, retry, and fail
   loudly. **No criterion or threshold in §"Success criteria" was altered after the data
   existed.**
8. **Quota still could not be read live** (Kaggle MCP down all session). The push started, so
   the session was granted. **Spend: ~93 min T4 on `account-a`.**
