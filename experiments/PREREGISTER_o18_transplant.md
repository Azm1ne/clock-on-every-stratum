# Pre-registration: O18-T — is the engine/torch gap in the INIT DRAW or the UPDATE PATH?

**Commit BEFORE launching.** This is the **third** hypothesis about O18. The first two —
precision (C30, N9) and execution substrate (O18 bisect, Entry 32) — were both plausible,
both tied up loose ends, and **both were wrong**. LAB_PROTOCOL.md's standing warning applies
directly: *a 7/7 correlation with a mechanism that ties up three loose ends is the most
dangerous shape a wrong claim can take.*

**Date:** 2026-09-15 · **Script:** `run_o18_transplant.py`
**Arm:** thesis · **Compute:** local CPU, zero Kaggle quota

## The gap, restated

At byte-identical hyperparameters, architecture, data split and optimiser, at n=113 seed 0:

| | tail Gini_mult | \|W_E\| | neurons tuned |
|---|---|---|---|
| engine (float64) | **0.7536** | 11.610 | 70–93 % |
| engine (float32, N9) | 0.7671 | 11.642 | — |
| torch (Kaggle T4, float32) | **0.5665** | 15.152 | 92–100 % |
| torch (this CPU, O18 bisect) | 0.5791 | — | — |

Eliminated already, do not re-eliminate: protocol · hyperparameters · tail window ·
optimiser · gradients (2.68e-15) · minibatch noise (both full-batch) · init *distribution* ·
the train/test split (byte-identical) · precision (N9, sign wrong) · hardware (Entry 32).
**What remains: the specific init DRAW, and the ~400 lines of the update path.**

## Hypothesis

The gap is in the **update path**, not the init draw. Byte-identical starting weights will
not close it.

## Design — two stages, and stage 1 is the decisive one

STATE's action 1 proposed a 40k A/B. **That is the weaker experiment and it is slow**: the
engine runs ~449 ms/step (LAB_PROTOCOL.md), so 40k steps is **~5 h**, against 35 min for torch-CPU
(measured, `logs/o18_cpu.log`). A paired **step-by-step** comparison from identical weights
answers a sharper question in minutes, and it can falsify the hypothesis outright:

- **Stage 1 — paired trajectory, 500 steps, ~5 min.** Transplant torch's nine init tensors
  into the engine, then step both on *identical* batches, comparing every parameter after
  every step. **If the two update paths agree to rounding for 500 steps, the gap CANNOT be
  in the update path** — no 40k run is needed to establish that, and the hypothesis dies
  cheaply. If they diverge, stage 1 reports the **first step and the first tensor**, which
  is strictly more than a 40k Gini number can say.
- **Stage 2 — the 40k A/B, ~5 h, CONDITIONAL on stage 1 showing divergence.** Only then is
  "which endpoint does it land on" a meaningful question.

Running stage 2 unconditionally would spend 5 h to learn less than stage 1 learns in 5 min.

## Success criteria — exact, implemented before the run

| # | criterion | threshold | implemented in |
|---|---|---|---|
| **T0** | **GATE. The transplant is exact** — the two models compute the same function from the transplanted weights | max \|Δlogit\| < **1e-10** at float64 on the full grid, step 0 | `run_o18_transplant.py::check_transplant` |
| T1 | The data split is byte-identical between the arms | index arrays compare equal, asserted | `::dataset` |
| T2 | **Stage-1 PRIMARY.** Per-step max relative parameter divergence over 500 steps, engine-float32 vs torch-float32 | see decision rule below | `::stage1` |
| T3 | Rounding baseline: the same comparison engine-float64 vs engine-float32 | reported, bounds what T2's numbers can mean | `::stage1` |
| T4 | *(stage 2 only)* tail Gini_mult of the transplanted engine run | f-rule below | `analyze_n9.py`-compatible output |
| T5 | grok, and provenance stamped with a real SHA + dtype | test acc > 0.99; `git_dirty=False` | `::main` |

**T0 is a GATE, not a criterion.** The engine stores `W_Q/W_K/W_V` as `(d_model, H ·d_head)`
and `W_O` as `(H ·d_head, d_model)`; the k03 kernel uses `(H, d_model, d_head)` and
`(H, d_head, d_model)`. The reshape/transpose order between them **will not be reasoned
about** — this project has lost three bin-convention arguments to reasoning. It is asserted
by logit equality. **If T0 fails, nothing downstream is readable and the run does not start.**

**T2 decision rule, fixed now.** Let `r(t)` be the max over parameters of
`|w_eng - w_torch| / (|w_torch| + 1e-12)` after step t, both at float32.

- `r(500) < 100 × r_base(500)`, where `r_base` is the T3 float64-vs-float32 engine baseline
  ⇒ **the update paths agree to rounding. The hypothesis is FALSIFIED; the gap is the init
  draw or something not yet enumerated.** Stage 2 is not run.
- `r(t)` exceeds that at some step ⇒ **divergence is real.** Report the first such step and
  the tensor carrying it; run stage 2.

**T4 f-rule (stage 2 only), fixed now**, same form N9 used so the two are comparable:
`f = (G_transplant − G_torch) / (G_engine − G_torch)` with G_engine = 0.7536, G_torch = 0.5665.

- `f > 0.7` ⇒ **UPDATE PATH.** Torch's own weights still produce an engine-like circuit.
- `f < 0.3` ⇒ **INIT DRAW.** O18 closes.
- otherwise ⇒ **inconclusive / both**, reported as such and not rounded to a story.

## What would falsify this

Stage 1 showing the two update paths agreeing to rounding for 500 steps. That is a live and
arguably likely outcome — the optimisers are algebraically the same update
(`p(1-lr ·wd) - lr ·mh/(sqrt(vh)+eps)`, LAB_PROTOCOL.md) and C10 verifies gradients to 2.3e-15 — and
it would leave the **init draw** as the only enumerated candidate. Stage 2 with `f < 0.3`
falsifies it too.

**A pre-registration that pre-explains a failure is not permission to skip the check**
(LAB_PROTOCOL.md, on N7's E5). If stage 1 falsifies this, the answer is "the gap is the init draw
or something we have not enumerated" — **not** a rescue narrative.

## Required controls

| claim shape | control |
|---|---|
| "the update paths differ" | T3 rounding baseline, same engine, both dtypes — divergence must exceed it |
| "the transplant isolates the update path" | T0 logit equality gate; T1 byte-identical split |
| "X is engine-like / torch-like" | both reference endpoints measured on THIS box, not quoted from Kaggle |
| "sparser" | Gini on amplitude, reported with (mult, add, random-orthogonal) + key-frequency count |

## Known confounds

1. **Float32 chaos.** Two float32 trajectories diverge exponentially from rounding alone.
   T3 exists to bound that; T2's threshold is relative to it, never absolute.
2. **RNG consumption order.** Transplanting weights removes the init draw as a variable but
   not any *later* stochasticity. Both arms are full-batch with no dropout, so there is none
   after init — asserted by T1 rather than assumed.
3. **Engine default dtype is float64.** Stage 1's primary comparison runs the engine at
   `ENGINE_DTYPE=float32` to match torch; float64 is reported alongside as T3.
   `test_autograd.py::test_dtype_flag` guards that the flag is honoured through the whole
   graph (NEP 50 silently upcast it once already).
4. **n=113 seed 0 only.** One cell. This is a mechanism bisect, not a population estimate;
   a difference in *update rule* does not need seeds, but any Gini number from stage 2 does,
   and will be labelled 1-seed.
5. **Neuron tuning runs the OPPOSITE way to Gini** (engine 70–93 % tuned at Gini 0.75; torch
   92–100 % at 0.57). One convergence axis cannot produce that ordering, so a "the engine
   just converges further" reading is already excluded and must not be reintroduced.

## Analysis plan — decided now

`PYTHONPATH=. .venv/bin/python run_o18_transplant.py --stage1` writes
`results/o18_transplant/stage1.json` (per-step, per-tensor divergence) and prints T0–T3.
Stage 2, only if T2 shows divergence, writes `WE_transplant_n113_s0.npz` in k03's schema so
`analyze_n7.py` / `analyze_n9.py` consume it unchanged. **Any analysis not listed here is
exploratory and will be labelled so.**

## Outcome (filled in AFTER the run — never edit anything above)

**Stage 1 run 2026-09-15**, `PYTHONPATH=. .venv/bin/python run_o18_transplant.py --stage1 113 0 500`,
log `logs/o18_transplant.log`, artifact `results/o18_transplant/stage1.json`, ~5 min, zero quota.

- **Result: the hypothesis is FALSIFIED. The update path is NOT the source of the gap.**

  | # | criterion | verdict | numbers |
  |---|---|---|---|
  | T0 | transplant exact (GATE) | **HELD** | max \|Δlogit\| = **7.772e-16** on the full grid at float64 (logit scale 0.655) |
  | T1 | split byte-identical | **HELD** | engine `modular_data` == kernel `dataset`, 3,830 train / 8,939 test |
  | T2 | engine-f32 vs torch-f32 @500 | — | r = 1.082e+04 |
  | T3 | engine-f32 vs engine-f64 @500 (rounding baseline) | — | r = 1.948e+04 |
  | | **ratio** | **0.56** | rule was `< 100 ⇒ agree`. **The engine agrees with torch BETTER than it agrees with itself across dtypes.** |

  **The decisive number is not in the pre-registered table.** The added deterministic
  float64 single-step check (below) is far stronger than 500 chaotic float32 steps:

  | step | max \|Δloss\| | max \|Δgrad\| | max \|Δw\| after one AdamW step |
  |---|---|---|---|
  | 1 | 1.776e-15 | **2.082e-17** | **8.646e-15** |
  | 2 | 0.000e+00 | 1.821e-17 | 8.660e-15 |
  | 3 | 0.000e+00 | 9.758e-18 | 8.646e-15 |

  From identical weights, at float64: the **autograd graph** agrees to 2e-17 and the
  **weights after a full AdamW step** to 9e-15. The two update paths are not merely
  indistinguishable under chaos — they are the same computation. C10 had verified gradients
  to 2.3e-15, but against a torch reference written in the *engine's* layout; this is
  against the kernel that actually produced the torch numbers.

- **Where that leaves O18.** An exploratory follow-up (labelled as such) put all four
  artifacts through **one** measurement path (`analyze_n7.mult_amplitude` + `gini`):

  | arm | \|W_E\| | Gini_mult | PR_mult |
  |---|---|---|---|
  | engine n7 (float64) | 11.795 | **0.7630** | 3.890 |
  | engine n9 (float32) | 11.801 | 0.7610 | 3.896 |
  | torch o18 (CPU) | 13.323 | 0.5770 | 3.236 |
  | torch k03 (T4) | 15.757 | 0.5383 | 4.394 |

  So the gap is **in the weights, not the instrument** — the measurement protocol is
  exonerated on top of everything LAB_PROTOCOL.md already lists as eliminated. With data, init
  distribution, forward, gradients and optimiser all now shown identical, **the only
  enumerated candidate left is the specific init DRAW** (numpy `default_rng` vs
  `torch.Generator`, statistically the same distribution, different realisation).

  **And that candidate has a problem that must be stated rather than smoothed over.** A
  random draw should wash out across seeds; it does not. The engine reads 0.73–0.77 over
  3 seeds and torch 0.54–0.58 over 5 — each tight, the two separated by ~0.19. Chaos from
  the draw would widen each band, not shift one. **So either the draw is systematically
  different in a way we have not measured, or the cause is not enumerated at all.** This is
  the fourth hypothesis about O18; the first three were plausible and all three were wrong,
  and nothing here should be read as a prediction that the fourth is right.

- **Criteria met:** T0 HELD (gate) · T1 HELD · T2/T3 ratio 0.56 ⇒ **agree to rounding** ·
  T5 provenance stamped, real SHA, `git_dirty=False`. T4 pending stage 2.

- **Deviations from this pre-registration, and why:**
  1. **Stage 2 IS being run**, though the rule above says it is not run when stage 1 shows
     agreement. That conditional rested on the reasoning that "which endpoint" is only
     meaningful once divergence is established. **Stage 1 disproved the reasoning, not the
     value:** with the update path identical to 1e-15 and the gap localised to the weights,
     the endpoint question becomes the *direct* test of the init draw — the last enumerated
     candidate. **T4's f-rule is unchanged**; only the gate on reaching it is. Launched
     2026-09-15 under `systemd-run --user --unit=o18t-stage2`, 40k steps, ~5 h, zero quota.
  2. **T0's implementation was corrected before any conclusion was drawn.** The first run
     compared a **float32** torch forward against a float64 engine and read 4.103e-07 —
     which looks exactly like a layout bug and would have aborted a correct experiment. The
     criterion says "at float64"; promoting the torch model to double for the check (an
     exact round trip) gives 7.772e-16. The threshold was not moved.

---

### Stage 2 outcome — T4, filled in 2026-09-15 07:19 local (01:19 UTC)

`run_o18_transplant.py --stage2 113 0 40000` under `systemd-run --user --unit=o18t-stage2`,
**318 min**, exit 0, zero quota. Artifact `results/o18_transplant/WE_transplant_n113_s0.npz`,
provenance `git_sha=b79288d`, `git_dirty=False`. **HEAD moved during the run** (a633386 →
b79288d); `git diff --name-only a633386 b79288d` is FINDINGS.md, LAB_NOTEBOOK.md, STATE.md,
analyze_o5.py, CODE_AND_TOOLS.md, reproduce.sh — **no training code**, check recorded per
LAB_PROTOCOL.md. Grokked at step **5,800**.

**T4 measured on the tail window, not the last row (C27):** 11 snapshots from step 30,000.

| quantity | value |
|---|---|
| tail Gini_mult (mean of final 10k steps) | **0.7827**, within-run **sd 0.0009** |
| final checkpoint Gini_mult | 0.7815 |
| Gini_add | 0.0104 |
| random-orthogonal control (residue axis) | 0.1577 |
| \|W_E\| | **11.738** |

| f computed on | value |
|---|---|
| tail mean | **+1.156** |
| final checkpoint | +1.149 |
| tail mean − 2 sd (worst case for the conclusion) | +1.146 |

**T5 HELD** (test acc 1.0000, real SHA, `git_dirty=False`).

**Verdict. `f = +1.156` — not merely above the 0.7 threshold but above 1.0.** Giving the
engine torch's own sampled starting weights did not move it toward torch at all; it landed
**slightly past the engine's own baseline** (0.7827 vs 0.7536). And the second, independent
discriminator agrees: **\|W_E\| = 11.738 is engine-like (11.61), not torch-like** (13.32 CPU,
15.15 T4). Both axes say engine.

**The f-rule's `f > 0.7` branch is labelled "UPDATE PATH", and that label is now known to be
wrong** — stage 1 eliminated the update path to 1e-15 before stage 2 launched. Under the
reframing recorded in the stage-1 deviation note ("stage 2 becomes the direct test of the
init draw"), `f > 0.7` means **the init draw is NOT the cause**. Stating both readings rather
than silently re-labelling the branch: the pre-registered rule as written points at a
hypothesis its own stage 1 had already killed, so **the arm of the rule that fired carries no
positive information — only the negative one.** It eliminates; it does not explain.

**⚠️ ONE SEED**, as confound 4 said it would be. The direction is not marginal (f = 1.15, not
0.4) and is robust to ±2 sd, but this is n=113 seed 0.

**Where O18 now stands: all four enumerated hypotheses are dead.**

| # | hypothesis | killed by |
|---|---|---|
| 1 | precision | N9 — f = −0.07, sign wrong |
| 2 | execution substrate / hardware | O18 bisect, Entry 32 — f = 0.068 |
| 3 | update path | O18-T stage 1 — gradients 2.08e-17, one AdamW step 8.65e-15 |
| 4 | **init draw** | **O18-T stage 2 — f = +1.156** |

plus protocol, hyperparameters, tail window, optimiser, minibatch noise, init *distribution*,
the byte-identical split, and the measurement instrument.

**This is now a contradiction, and it should be written as one.** Identical initial weights,
identical data, identical forward (7.772e-16), identical gradients (2.082e-17), identical
AdamW step (8.646e-15), both full-batch and deterministic — and different endpoints. Nothing
on the enumerated list can produce that.

**The un-run cell.** The 2×2 of implementation × precision has a hole in it:

| | float32 | float64 |
|---|---|---|
| engine | 0.7671 (N9) | 0.7536 |
| torch | 0.5665 (T4) / 0.5791 (CPU) | **NEVER RUN** |

Every torch run in this project is float32 — `kernels/k03_grid_acts/run.py:60` samples with
`torch.randn` at the default dtype and nothing calls `.double()`. N9 tested precision *inside
the engine* and found it inert there; **that does not test whether float32 is doing something
to torch.** It is the boring, un-run experiment, and it is ~35 min of local CPU at zero quota
(torch-CPU measured, `logs/o18_cpu.log`). It needs its own pre-registration before it runs.

  3. **T3's implementation was corrected before any conclusion was drawn.** It compared
     engine-float64 against **torch**, not against engine-float32 as this file specifies —
     so the "rounding baseline" contained the very effect it existed to bound, and the
     decision rule would have returned "agree" no matter what. Found in the 20-step smoke
     test. The arms now run sequentially (ENGINE_DTYPE is read at import, so two dtypes
     cannot be live in one process) and are compared from snapshots.
  4. **Added, not pre-registered:** the deterministic float64 single-step check, and the
     four-artifact single-instrument comparison. Both are labelled exploratory here and in
     `FINDINGS`; neither was used to decide a pre-registered criterion.
  5. The pre-registered T2 metric (per-element relative with a 1e-12 floor) is dominated by
     elements where the reference weight is ≈0 and swings orders of magnitude between
     steps. It was **kept** as specified — the ratio construction against T3 makes it
     usable — and a per-tensor `max|Δ|/rms` is reported beside it as an addition.
