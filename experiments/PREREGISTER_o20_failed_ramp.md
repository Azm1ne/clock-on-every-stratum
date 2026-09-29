# Pre-registration: O20 — is the early CRT structure the grokking circuit, or structure a memorising model also acquires?

**Commit this file BEFORE launching the run.**

**Date:** 2026-09-15 · **Script:** `run_n7.py` at n = 63,
scored by `test_crt_null.py` · **Arm:** thesis (from-scratch engine) · **Cost:** ~1 h local
CPU, 3 seeds in parallel, **zero Kaggle quota**

## Background

C28-NULL closed O14 and opened this. The CRT enrichment in `W_E` is **a monotone ramp**
(ρ ≈ 1.000), present from the first post-init snapshot, and it survives a structure-matched
null — so it is real and CRT-specific. But its **magnitude does not separate**: at step
1,000 a grokked run reads enrichment **1.22–1.45**, while a **FAILED** model at n = 63
reaches **1.57–1.63 at convergence** (k04). The early signal therefore sits *below the
failure band*, and nothing yet connects it to generalisation.

This is the same lesson Gate 2's G1 taught (`scripts/gate2_discriminate.py`): **a statistic
that also fires on failures is not evidence, whatever it does on successes.**

## Hypothesis

If the CRT ramp is the *grokking circuit forming*, a run that never generalises should
**stop climbing**. If the ramp is instead structure that any model fitting `a ·b mod n` on
30 % of the table acquires — memorising included — the failed run's ramp will look like the
grokked one.

## Prediction

n = 63 = 3² ·7 at `train_frac` 0.30 grokked **1/3 in k04** and is a C14 dataset-size null,
so most seeds fail. **We predict the FAILED runs keep climbing** (the ramp is not specific
to generalisation), because the enrichment magnitude already exceeds the grokked early band.
Stated as a falsifiable prediction so the opposite outcome is a real result, not a shrug.

## Success criteria — exact, implemented before the run

| # | criterion | threshold | meaning |
|---|---|---|---|
| **1 (PRIMARY)** | Spearman ρ(step, enrichment), 0–20k, on runs classified **FAILED** | **ρ ≥ 0.80 ⇒ the ramp is NOT generalisation-specific** (O20 answered: memorising models acquire it too). **ρ ≤ 0.30 ⇒ the ramp tracks generalisation.** 0.30 < ρ < 0.80 ⇒ AMBIGUOUS | `test_crt_null.py::_ramp` |
| **2** | final enrichment of the FAILED runs vs the grokked early band (1.22–1.45 at step 1,000) | failed final **> 1.45** ⇒ the early signal cannot be read as "the circuit forming" | same |
| **3 (CONTROL — the run must actually fail)** | test accuracy, **median of the final 10 logged samples**, never the last row | **< 0.90 ⇒ FAILED**; ≥ 0.99 ⇒ grokked and that seed is **excluded from criterion 1** and reported separately | `hist_cols` by name |
| **4 (POSITIVE CONTROL)** | the same scorer on a **grokked** run must reproduce the published ramp | ρ ≥ 0.80 at n = 119 in `results/n7_engine`, else the scorer is broken and 1–2 are void | `test_crt_null.py results/n7_engine` |

**Criterion 3 is not a formality.** k04 got 1/3 groks at this modulus, so a seed may
generalise. Classifying by the **last row** would import C27's censoring bug directly; the
window median is the project's standing rule.

**Criterion 4 exists because `test_crt_null.py` hard-codes `PRIMARY_N = 119`.** Pointed at
an n = 63 directory unmodified it finds nothing and prints a confident report about an empty
set — the "running a command is not verifying it" failure in `LAB_PROTOCOL.md`. The modulus is
made overridable by `CRT_NULL_N` in this same commit, and criterion 4 re-runs the original
n = 119 path to prove the override did not change the existing answer.

## What would falsify this

**ρ ≤ 0.30 on the failed runs.** The ramp would then be specific to runs that generalise,
the early CRT signal *would* be the circuit forming, and O20 closes the other way.

## Known confounds

- **n = 63 has no discrete log** — `(Z/63Z)* = Z₆ × Z₆`. Irrelevant here: the CRT law is
  about the **additive** basis (`freq_energy` on `W_E`), which needs no discrete log. The
  multiplicative readout is not used and must not be reported for this modulus.
- **Dataset size is the reason it fails** (C14, 1,190 training pairs), not algebra. This is
  a feature — it gives a failure that is not confounded with the modulus's structure.
- **A failed model still fits 30 % of the table**, and its logits are a function of `u ·v`
  there. That is precisely why the magnitude comparison (criterion 2) is needed alongside ρ.

## Analysis plan — decided now

`run_n7.py 63 {0,1,2} 20000`, `N7_OUT=results/o20_n63`, `TRAJ = 1000` (unchanged), under
`systemd-run --user` with a memory cap. Then `CRT_NULL_N=63 test_crt_null.py
results/o20_n63` for criteria 1–3, and `test_crt_null.py results/n7_engine` for criterion 4.
**20,000 steps, not 40,000** — criterion 1's window is 0–20k and the question is about the
ramp, not the endpoint.

## Outcome

*(to be filled after the run — nothing above this line is edited)*

---

# OUTCOME — 2026-09-15

**Run:** `run_n7.py 63 {0,1,2} 20000`, `N7_OUT=results/o20_n63`, engine float64, under
`systemd-run --user` with `MemoryMax=2500M` and `OMP_NUM_THREADS=4`. 1.11–1.25 h wall,
3 seeds in parallel, **zero Kaggle quota**. Scored by `CRT_NULL_N=63 test_crt_null.py
results/o20_n63`; log `logs/o20_score.log`.

## Controls first

| # | criterion | result | verdict |
|---|---|---|---|
| **3** | the runs must actually FAIL, by **median of the final 10 samples** | **3/3 FAILED** — medians **0.1801 / 0.1655 / 0.3183**; all three report `grok_step=None` | **PASS** |
| **4** | the scorer must reproduce the published ramp on **grokked** n = 119 | ρ = **1.000 / 0.991 / 0.999**, 3/3 above 0.8 | **PASS** — and the `CRT_NULL_N` override did not move the existing n = 119 answer |

⚠️ **Criterion 3 was nearly read wrong.** The engine's `hist_cols` is
`[step, train_loss, test_loss, train_acc, test_acc]`, so **index 3 is `train_acc` = 1.000**
— indexing positionally would have classified all three FAILED runs as grokked and inverted
the experiment. Read by column name.

The scorer's own null calibration also holds: at **step 0** enrichment is 0.981 / 0.987 /
0.979 at p_A ≈ 0.75–0.86, i.e. **not rejecting**. The instrument has a real null; it is not
firing on everything.

## Criterion 1 (PRIMARY) and criterion 2

| seed | final acc | enrich @1k | enrich @20k | ramp ρ (0–20k) |
|---|---|---|---|---|
| 0 | 0.1803 | 1.397 | **1.653** | **1.000** |
| 1 | 0.1655 | 1.339 | **1.459** | **0.988** |
| 2 | 0.3177 | 1.422 | **1.655** | **0.995** |

- **Criterion 1: ρ ≥ 0.80 in 3/3.** The pre-registered rule reads: **the ramp is NOT
  generalisation-specific.**
- **Criterion 2: final enrichment > 1.45 in 3/3** (1.653, 1.459, 1.655) against the grokked
  **early** band of 1.22–1.45. HELD.
- `p_A`, `p_B`, `p_C` are all **0.0000 from step 1,000 onward** in every failed seed.

## Independent corroboration, from data that already existed

`test_crt_null.py`'s own discrimination arm scores k04's **torch** runs at the same modulus:

| run | final acc | enrichment | p_A | p_B |
|---|---|---|---|---|
| `B_thesis_n63_s0` | 0.9969 **grokked** | 4.687 | 0.000 | 0.000 |
| `B_thesis_n63_s1` | 0.2236 **FAILED** | **1.626** | 0.000 | 0.000 |
| `B_thesis_n63_s2` | 0.2749 **FAILED** | **1.574** | 0.000 | 0.000 |

The FAILED arm passes null A **4/4** and null B **2/2**. Two independent implementations,
two independent data sources, same answer.

## ⇒ O20 IS ANSWERED, and the answer is the unwelcome one

**The early CRT structure is NOT the grokking circuit.** A model that never generalises —
train accuracy 1.000 from step 2,000, test accuracy 0.17–0.32 at step 20,000 — acquires
the **same monotone CRT-dual enrichment ramp** (ρ ≥ 0.988), at the **same significance**
(p < 0.01 from step 1,000), reaching a **higher** final enrichment (1.46–1.66) than grokked
runs show at step 1,000 (1.22–1.45).

**This extends C28 rather than contradicting it.** C28 already records that the permutation
p does not separate grokked from failed and only the **magnitude** does. O20 shows the
**ramp shape does not separate either** — monotonicity was the last property that might have
made the early signal diagnostic, and it is not.

**What survives, precisely:** the CRT-dual law is real, CRT-specific, survives a
structure-matched null, and is **causally** load-bearing in grokked models (C22/C23/C36,
p < 0.01). What does **not** survive is any reading of the *early* enrichment ramp as
"the generalising circuit forming". It is structure that a memorising model fitting 30 % of
the table acquires too.

⚠️ **Paper consequence:** the early ramp must never be presented as a progress measure or
as evidence of circuit formation. Where it appears, the failed-run ramp appears beside it.
This is the Gate 2 G1 lesson for the third time: **a statistic that fires on failures is not
evidence, whatever it does on successes.**
