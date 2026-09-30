# Pre-registration: O18-F64 — the un-run cell. Is *torch* at float64 engine-like?

**Commit this file BEFORE launching.** **Date:** 2026-09-15 · **Script:** `run_o18_cpu.py --f64` ·
**Arm:** thesis (diagnostic bisect) · **Compute:** **3 seeds × 40,000 steps ≈ 3.3 h local
CPU, measured not assumed** (400-step smoke: 98 ms/step at float64 against the float32
arm's 46 ms/step). **Zero Kaggle quota.**

## Hypothesis

The ~0.19 `Gini_mult` gap between the from-scratch engine and the k03 PyTorch kernel is
produced by **training precision acting on the PyTorch implementation** — an interaction
that N9 could not have seen, because N9 varied dtype inside the engine only.

## Why this cell, and why it is not a re-litigation of the dead C30

**Nothing in this repository has ever trained PyTorch at anything but float32.**
`kernels/k03_grid_acts/run.py:60` samples with `torch.randn` at the default dtype and no
call site promotes it. The 2×2 of {engine, torch} × {float32, float64} therefore stands at
three cells filled and one empty:

| | float32 | float64 |
|---|---|---|
| **engine** | 0.7671 (N9) | 0.7536 (N7) |
| **torch (CPU)** | 0.5791 (O18-substrate) | **never run** |

N9's result — `f = −0.07`, the sign wrong — is a statement about the **engine's** main
effect. It is silent on whether float32 does something to **torch**, and a main effect
being null in one implementation does not make the interaction null. O18-T stage 1's
agreement to 1e-15 was measured on a torch *promoted* to float64 for the comparison, which
is a configuration that was never actually **trained**.

This is not the retracted C30. C30 claimed "float64 makes circuits sparser" as a main
effect and was falsified by the controlled A/B. This tests a different, unexamined cell,
and its result cannot resurrect C30 — if float64 raises torch's Gini it is an
implementation-specific interaction, not the main effect C30 asserted.

All four enumerated O18 hypotheses are dead: precision-in-engine (N9, `f = −0.07`) ·
substrate (`f = 0.068`) · update path (stage 1, one AdamW step agreeing to 8.65e-15) ·
init draw (stage 2, `f = +1.156`). O18 is currently a **contradiction**. This run fills the
last cheap cell before the question has to be attacked structurally.

## Prediction

**No directional prediction is made, and that is deliberate** — the same stance as the
substrate pre-registration, and for the same reason: both branches are decisive and
choosing a story afterwards is precisely how C30 was lost.

- torch-f64 reads **≈ 0.75** (engine-like) → precision *does* act on torch, every torch
  number in this project is a float32 artifact, and so is every published float32 number
  it has been compared against (Nanda 2301.05217, 2606.17399);
- torch-f64 reads **≈ 0.58** (torch-like) → precision is inert on **both** sides of the
  2×2, the gap is purely implementational, and O18 stays a contradiction with no cheap
  cell left.

## Success criteria — fixed now, thresholds copied verbatim from N9 and O18-substrate

Reference values, all on disk before this run (tail means over the final 10k steps,
amplitude protocol): engine float64 **0.7536** · engine float32 **0.7671** ·
torch float32 **CPU 0.5791** · torch float32 T4 0.5665 · `‖W_E‖` engine **11.610**,
torch CPU **14.296**, T4 15.152 · tuned-neuron fraction engine **0.887**, torch CPU
**0.938**, T4 0.969.

**The baseline is torch-float32-on-CPU, not torch-on-T4.** Holding the substrate fixed is
the whole point of comparing against the arm that already varied it; using the T4 number
would fold O18-substrate's `f = 0.068` into this experiment's `f` and make the rule read
partly the substrate. (This is the C7b/C19/C20/G1 family — a baseline that contains the
effect it bounds.)

| # | criterion | statistic | threshold |
|---|---|---|---|
| **F0** | the float64 runs grok | `test_acc > 0.99`, classified on the **median of the final 10 samples**, never the last row (C27) | **≥ 2/3 seeds**, or F1–F3 are not scored |
| **F1** | **PRIMARY.** Fraction of the gap explained by precision-in-torch | `f = (G_torch_f64 − 0.5791) / (0.7536 − 0.5791)`, tail mean over the final 10k steps | **HELD** `f ≥ 0.70` · **NOT HELD** `f ≤ 0.30` · `0.30 < f < 0.70` is **PARTIAL** and must be reported as such, not rounded to either |
| **F2** | the embedding-norm gap moves with it | `f` on `‖W_E‖`: `(W − 14.296) / (11.610 − 14.296)` | same bands. A precision explanation must move **both** or it is not the mechanism |
| **F3** | neuron tuning moves the *right way* | tuned fraction at the 0.85 cut | **descriptive.** The engine is *less* tuned (0.887) at *higher* Gini (0.7536) than torch (0.938 at 0.5791). Any account of O18 must fit that inversion; a mechanism that raises Gini and tuning together does not |
| **F4** | grokking time | steps to acc > 0.99 | **descriptive, 3 seeds, variance reported.** Seed variance on grok time is a factor of 3.8 in this project (O2); no ordering at 3 seeds is interpretable |
| **F5** | **O19, and it is UNDERPOWERED BY CONSTRUCTION** | post-grok excursions below acc 0.90, `analyze_excursions.py` | **Asymmetric, stated now.** Torch's measured rate is 33 excursions in 21,949 post-grok samples = **0.15 %/sample**. This run yields ≈ **510** post-grok samples, so the expected count under "precision changes nothing" is **0.77** and **P(0) ≈ 0.46**. Therefore: **≥ 1 excursion is informative** — torch still spikes at float64, so precision does not explain the engine's zero and O19's confound is broken. **0 excursions is NOT evidence of anything** and must be reported as uninformative, not as support |

F5 is written this way because the alternative is the trap this project has hit repeatedly:
reading a null from an underpowered instrument as a result. The power calculation is on the
record **before** the count exists.

## What would falsify this

`f ≤ 0.30` on F1. Then precision is inert on both sides of the 2×2, there is no cheap cell
left, and O18 must be attacked as a structural difference — the remaining move is a
step-by-step divergence trace from identical weights over many steps rather than one.

## Known confounds

- **float64 changes torch's random draw from the same seed.** `torch.randn` at float64
  consumes the generator differently, so seed 0 here is not weight-identical to seed 0 in
  the float32 arm. **This confound is already controlled by a prior experiment:** O18-T
  stage 2 started the engine from torch's own sampled weights and the gap did not close
  (`f = +1.156`), so the init **draw** cannot carry a 0.17 Gini difference. Reported, not
  hand-waved.
- **Seed.** 3 seeds (0, 1, 2), matching N9, O18-substrate and k03's first three. `f` is a
  comparison of means; per-seed values are reported.
- **Only one modulus (113).** A bisect, not a claim about moduli. It inherits nothing about
  121/125/119 and must not be written up as if it did.
- **Tail window** fixed at the final 10k steps — the window C26 and N7's E2 already use.
- **BLAS threading** is not a scientific variable but is a numerical one; recorded in the
  log. Both arms are full-batch.
- **Storage precision is not the variable.** `mlp_acts`/`attn`/`logits_all` are cast to
  float32 on save in **both** arms, byte-identically, because that cast is in the kernel
  source this run execs unmodified. Only training precision differs.

## Analysis plan — decided now

1. `PYTHONPATH=. .venv/bin/python analyze_n9.py results/o18_f64` — F0–F4. The same scorer
   N9, O18-substrate and O18-T used, pointed at the new directory, so the comparison cannot
   drift.
2. `PYTHONPATH=. .venv/bin/python analyze_excursions.py results/o18_f64` — F5.

**Both were smoke-tested against a real 200-step artifact from this exact script before
this file was committed** (each exits 0 and correctly reports the un-grokked smoke as
excluded), because the substrate pre-registration named a command that had never been run
and it found zero runs while printing something indistinguishable from a failed sweep.
Anything beyond these two commands is exploratory and labelled so.

## Outcome (filled in AFTER the run — never edit anything above)

**Date scored:** 2026-09-15 · 3 seeds × 40,000 steps, local CPU, **191.8 min**, zero Kaggle
quota · artifacts `results/o18_f64/`, SHA **`3d19067`, clean, `dtype float64` stamped on all
three** · logs `logs/o18_f64.log`, `logs/o18_f64_score.log`.

### Result: the hypothesis is CONFIRMED. O18 is solved, and the cause is precision acting on PyTorch.

**The 2×2 is complete.** Tail `Gini_mult`, final 10k steps, amplitude protocol:

| | float32 | float64 |
|---|---|---|
| **engine** | 0.7671 (N9) | 0.7536 (N7) |
| **torch (CPU)** | 0.5791 (O18-substrate) | **0.7736** ← this run |

Per seed: **0.7621 / 0.7874 / 0.7712**. `‖W_E‖` **11.695** (11.60 / 12.19 / 11.30).

| # | criterion | observed | verdict |
|---|---|---|---|
| **F0** | groks | **3/3**, acc 1.0000, window median | **HELD** |
| **F1** | **PRIMARY.** `f = (G − 0.5791)/(0.7536 − 0.5791)` | **+1.114** | **HELD** (≥ 0.70) |
| **F2** | `f` on `‖W_E‖`: `(W − 14.296)/(11.610 − 14.296)` | **+0.968** | **HELD** |
| **F3** | neuron tuning | **0.770** (0.738/0.781/0.791), raw 0.000, shuffled 0.000 | see below — **it moves the right way** |
| **F4** | grokking time | 10,600 / 10,800 / 8,600 | descriptive |
| **F5** | O19 excursions | **0 over 453 post-grok samples** | **UNINFORMATIVE, as pre-registered** |

**Precision-in-torch accounts for 111 % of the Gini gap and 97 % of the embedding-norm
gap.** The pre-registration demanded both — *"A precision explanation must move **both** or it
is not the mechanism"* — and both moved, nearly exactly onto the engine's position.

### F3 dissolves the anomaly that no single axis was supposed to explain

`LAB_PROTOCOL.md` recorded this as the fact forbidding a convergence explanation:

> *"the two also differ in the opposite direction on neurons — engine 70-93 % tuned at Gini
> 0.75, torch 92-100 % at Gini 0.57. One convergence axis cannot do that, so treat this as
> possibly a different algorithm (Clock/Pizza, 2306.17844), not a sparsity magnitude."*

**Float64 moves both axes at once, in opposite directions, onto the engine's joint
position.** Torch goes Gini 0.5791 → **0.7736** (up) *and* tuning 0.938 → **0.770** (down),
against the engine's (0.7536, 0.887). It overshoots slightly on both. **The inversion was a
precision effect, not a different algorithm.** The Clock-and-Pizza discriminator is no longer
required to explain O18 — it may still be worth running, but not for this.

### This does NOT un-retract C30, and the distinction is the whole point

C30 claimed **"training precision changes the measured sparsity of a grokked circuit"** as a
**main effect** and was retracted by N9, which varied dtype *inside the engine* and found
`f = −0.07`, the sign wrong. **That retraction stands and N9's result is unchanged.**

What this run shows is an **interaction**: dtype is **inert in the engine** (0.7671 vs
0.7536) and **decisive in torch** (0.5791 vs 0.7736). A main effect and an interaction are
different claims, and only the interaction is supported. Write it as *"training precision
changes the measured sparsity **in PyTorch**"* — never as C30's sentence.

⚠️→✅ **Why the engine is already at the float64 answer even at float32 was a live question.
It is CHECKED and the engine column is VALID.** The candidate was NumPy's NEP-50 promotion,
which once silently trained an `ENGINE_DTYPE=float32` run with **float64 gradients** on five
tensors via `Tensor.max`'s backward dividing by an int64 tie count. **Resolved by ancestry,
not assumption:** the fix is commit `6991152`, N9's artifacts stamp SHA `a4b3d14` and
`dtype float32`, and `git merge-base --is-ancestor 6991152 a4b3d14` returns true — **the fix
predates the run.** N9's float32 arm is genuinely float32, so **all four cells of the 2×2
are valid** and the engine's dtype-invariance is a real measurement, not a mislabelling.

That leaves the *interesting* residue, which is no longer a defect but a question: **the
engine reaches the float64 answer at either dtype, and torch does not.** Nothing in this run
explains that, and it is the honest remaining open end of O18 — the *gap* is explained, the
engine's dtype-robustness is not.

### Consequence: every torch number in this project is a float32 measurement

All of k02–k09 is PyTorch float32. **At n=113 the same code, same data, same split, same
optimiser reads 0.5791 at float32 and 0.7736 at float64.** Any absolute sparsity number from
those runs is a property of the *precision regime* as much as of the circuit.

⚠️ **This reaches the literature, carefully stated.** C4 replicated `2606.17399`'s published
**0.579** and matched it — in float32, which is what their code uses. Our float32 agreement
with them is therefore real and their number is reproducible; what this run shows is that
**the quantity itself is precision-dependent**, so a published float32 Gini is not
comparable to a float64 one. It is **not** a claim that anyone's published number is wrong.
**Between-modulus and within-arm comparisons are untouched** — C6, C8, C22/C23, C31, C33 are
all comparisons within one precision regime.

### F5 (O19): reported as uninformative, per the arithmetic committed in advance

**0 excursions over 453 post-grok samples.** The pre-registration fixed the expectation
before the count existed: torch's rate is 0.15 %/sample, so ≈0.68 were expected and
**P(0) ≈ 0.51**. A zero here is what "no effect" predicts half the time. **It is not evidence
that float64 suppresses spiking, and O19 remains open and confounded.** Only ≥ 1 would have
been informative.

### Deviations from this pre-registration, and why

1. **⚠️ THE SCORER NAMED IN THE ANALYSIS PLAN PRINTS A DIFFERENT `f` THAN THE ONE
   PRE-REGISTERED, AND ITS VERDICT IS THE OPPOSITE.** `analyze_n9.py results/o18_f64`
   reports **"P1 Gini_mult NOT HELD (f = −0.10)"**. The pre-registered F1 is
   `(G − 0.5791)/(0.7536 − 0.5791)` = **+1.114, HELD**. The scorer computes N9's `f`, whose
   baseline and target are the *engine's* two dtype cells — correct for N9, meaningless for
   this arm. **Reading the tool's verdict instead of the committed formula would have
   inverted the conclusion of the session.** F1/F2 above are computed from the formulas in
   §"Success criteria", verbatim, and the scorer's own summary line must not be quoted for
   this experiment. Same family as O18-substrate's deviation #2, where the same script's
   labels were hard-coded for N9 — the labels were fixed then; **the statistic was not.**
2. **F3's threshold was left as "descriptive"** in §"Success criteria" and is scored that way.
   It is reported with an `f` (+3.291) only to show direction; no band was pre-registered and
   none is claimed.
3. **Runtime 191.8 min against the ~3.3 h predicted** — within the estimate. The 98 ms/step
   smoke measurement held.
4. **453 post-grok samples, not the ≈510 estimated.** The estimate assumed grokking at ~6,000
   steps; actual grok steps were 8,600–10,800. The power calculation is unchanged in
   substance (expected 0.68 vs 0.77, P(0) 0.51 vs 0.46) and the verdict —
   **uninformative** — is the same.
5. **No criterion or threshold in §"Success criteria" was altered after the data existed.**
