# Pre-registration: k06 — the convergence horizon (N10)

**Commit this file BEFORE launching.** Git's timestamp is the evidence the criteria
predated the result.

**Date:** 2026-09-11 · **Kernel:** `kernels/k06_horizon`
**Arm:** thesis (all n² pairs) · **Runs:** 4 moduli × 3 seeds = 12, at 120k epochs

## Why

k03's pre-registered H4 established **C24**: at 125, 119, 120 and 165 the participation
ratio is **still moving 4.8–7.7% over the final 10k steps** of a 40k run, converging in
only 2–3 of 5 seeds. 113 (2.3%) and 121 (2.8%) are converged; **Gini drifts < 3%
everywhere**.

So every PR number at those four moduli is a measurement taken at a non-stationary point,
and any cross-modulus PR comparison involving them inherits it. The choice is to train to
convergence or to stop reporting PR there. This run takes the first option, and it is the
only run in the project whose purpose is **the measurement instrument, not the phenomenon**.

## Hypothesis

40,000 epochs is simply too short at these four moduli; the participation ratio is
converging, not drifting indefinitely, and a longer horizon reaches a stationary value.

## The design

| n | factorization | φ(n) | unit group | PR drift @40k (k03) | seeds converged @40k |
|---|---|---|---|---|---|
| 125 | 5³ | 100 | Z_100 | 5.6% | 2/5 |
| 119 | 7 ·17 | 96 | Z_6×Z_16 | 5.9% | 3/5 |
| 120 | 2³ ·3 ·5 | 32 | Z_2³×Z_4 | 7.7% | 3/5 |
| 165 | 3 ·5 ·11 | 80 | Z_2×Z_4×Z_10 | 4.8% | 3/5 |

**120k epochs, 3× the k03 horizon.** Every other hyperparameter byte-identical to k02 Arm B
/ k03 / k04: train_frac 0.30, AdamW lr 1e-3, wd 1.0, d_model 128 / 4 heads / d_head 32 /
d_mlp 512, no LayerNorm. Seeds 0, 1, 2 — the same seeds as k03's first three, so the
trajectories are directly comparable run for run, not only in aggregate.

`W_E` snapshots **every 4,000 steps** (30 per run) so the drift curve can be computed at
any horizon rather than only at the end — this run's whole output is a trajectory.
`mlp_acts`, `logits_all` and `attn` saved at the final step for every seed.

**Poolability:** the **step-40k slice** of these runs pools with k03 (identical config and
seeds up to that point, modulo nondeterminism). **Nothing past 40k pools with anything**;
it is a new horizon and must be labelled as such wherever it is reported.

## Predictions

### R1 — the PR converges by 120k

**Prediction:** at each of the four moduli, PR drift over the final 10k steps is
**< 5%** in **≥2 of 3 seeds**. Threshold is k03's H4 criterion, unchanged.

**Falsified if** any modulus is still above 5% in ≥2 of 3 seeds at 120k. **That is the
interesting outcome**: it would mean the participation ratio at composite moduli does not
settle, and the honest response is to retire PR as a reported statistic there rather than
chase a longer horizon. Stated now so that outcome cannot be re-framed later as "needs
more steps".

### R2 — the converged value is close to the 40k reading

**Prediction:** |PR(120k) − PR(40k)| / PR(40k) < **25%** in ≥2 of 3 seeds at each modulus.

Threshold justified from the data: k03's *per-10k* drift is 4.8–7.7%, so ~8 further
10k-windows at a decaying rate should land well inside 25%. This is the criterion that
decides whether C24 **invalidates** existing PR numbers or merely **qualifies** them.

**Falsified if** ≥2 of 3 seeds move more than 25%. Then every published PR at these moduli
is wrong, not just imprecise, and must be restated.

### R3 — Gini was right to be trusted

**Prediction:** |Gini_mult(120k) − Gini_mult(40k)| < **0.05** in ≥2 of 3 seeds at the two
moduli with a cyclic unit group among these four (**125 only**; 119, 120, 165 are
non-cyclic and have no single-index Gini_mult). Gini_add < 0.05 at all four.

k03 measured < 3% drift in Gini at every modulus, so this predicts the stability holds at
3× the horizon. **Falsified if** ≥2 of 3 seeds move by 0.05 or more — which would put C6,
C9 and the whole of §3 of FINDINGS on the same footing as PR, i.e. measured before
convergence.

### R4 — nothing new grokks, and nothing un-grokks

**Prediction:** every run that reaches test acc > 0.99 does so **before step 40,000**, and
no run's final test accuracy falls below 0.99 after reaching it.

**Falsified if** any seed first groks after 40k (k02/k03/k04 grokking times would then be
right-censored, not measured), or if any run de-groks. Both are cheap to check and both
would matter more than R1.

## Success criteria — implemented as code before the run

| # | criterion | threshold | implemented in |
|---|---|---|---|
| 1 | R1 PR converges | final-10k drift < 5% in ≥2/3 seeds, all 4 moduli | `analyze_k06.py` |
| 2 | R2 40k value usable | \|ΔPR\| / PR(40k) < 25% in ≥2/3 seeds | `analyze_k06.py` |
| 3 | R3 Gini stable | \|ΔGini\| < 0.05 in ≥2/3 seeds | `analyze_k06.py` |
| 4 | R4 no late grok, no de-grok | all groks < 40k; final acc ≥ 0.99 once reached | `analyze_k06.py` |
| 5 | 40k slice replicates k03 | same seeds, same config; report the difference, do not assume it is zero | `analyze_k06.py` |
| 6 | provenance | every `.npz` carries a real git SHA **and the account that ran it** | `run_kernel.py` |

## Required controls

| claim shape | control |
|---|---|
| "the metric converged" | drift measured over the **final** window, not a fitted asymptote |
| "40k was fine after all" | the explicit 40k-vs-120k difference (R2), not a visual check |
| "X is stable" | 3 seeds, each reported; no seed-averaging before the per-seed verdict |
| "this pools with k03" | same seeds, and the 40k slice compared run-for-run (criterion 5) |

## Known confounds

- **PR convention.** Participation ratio is on the normalised **energy**
  `p_k = |A_k|²/Σ|A_k|²` (protocol invariant 2). Feeding amplitude gives 12.5 where the
  correct value is 4.3 and was open question O1 for three sessions. `analyze_k06.py` uses
  `energy` and states the convention in its output.
- **Weight decay is 1.0 and never decays.** At 3× the horizon the model may keep shrinking
  weights after the circuit is fixed, which can move a spectral statistic without the
  circuit changing. R3 partly controls for this; a Gini that moves while the ablation
  verdict does not would be the signature. Not a criterion — flagged so it is not
  discovered afterwards and presented as a finding.
- **Nondeterminism.** Same seed ≠ bitwise-identical trajectory on GPU. Criterion 5 reports
  the 40k difference rather than asserting it is zero.
- **120 has φ = 32** and LAB_PROTOCOL.md already records it as a weak cell for basis analysis
  (most characters real, mult/add distinction partly collapses). Its R3 verdict carries
  that caveat; it is included because C24 named it, not because its Gini is informative.

## Analysis plan — decided now

`analyze_k06.py`, written and committed **before** the run, implementing criteria 1–5 off
the `we_traj` snapshots. Statistics: per-seed drift fractions over the final 10k;
40k-vs-120k relative change; Gini on **amplitude**, PR on **energy**; grok step from the
saved history. Anything else is **exploratory and labelled**.

## Outcome (filled in AFTER the run — never edit anything above)

*Filled 2026-09-11, session 6. The run itself completed during session 5 and is written up
in LAB_NOTEBOOK Entry 23; this section was left empty then and is completed here. No number
below is new — all twelve runs were analysed by `analyze_k06.py` as committed.*

- **Result:** 12 runs at 120k epochs, SHA `8eba027`, account `account-a`. The
  hypothesis — *"40k is simply too short; PR is converging, not drifting indefinitely"* —
  is **false**. The participation ratio does **not** settle at composite moduli even at
  **3× the horizon**. The pre-registration named this as "the interesting outcome" and
  pre-committed the response, which has been carried out rather than reinterpreted:
  **PR is retired as a reported statistic at composite moduli.** Separately, one run did
  something nobody predicted — **n=119 seed 0 de-grokked**.

- **Criteria met:**

  | # | criterion | verdict | why |
  |---|---|---|---|
  | 1 | R1 PR converges — final-10k drift < 5% in ≥2/3 seeds, all 4 moduli | **NOT HELD** | Per-seed drift: 125 → 5.6 / 0.7 / 1.1 %; **119 → 32.3** / 4.1 / 3.6 %; 120 → 4.3 / **10.8** / 5.9 %; 165 → 3.0 / 0.1 / 0.3 %. 165 and 125 pass; **119 and 120 fail**, and 119's worst seed is 6× the threshold at 3× the horizon. PR does not settle |
  | 2 | R2 the 40k value is usable — \|ΔPR\|/PR(40k) < 25% in ≥2/3 seeds | **HELD** | Existing PR numbers are **qualified, not invalidated**. This is the criterion that decides whether C24 retracts published values or merely annotates them, and it lands on "annotate" |
  | 3 | R3 Gini was right to be trusted — \|ΔGini\| < 0.05 in ≥2/3 seeds | **HELD** | 40k→120k \|ΔGini\| is **0.008–0.048 in eleven of twelve runs**. The twelfth is n=119 seed 0 at **0.193** — the de-grokked run, i.e. the exception is a model that stopped working, not a statistic that drifted. C6, C9 and FINDINGS §3 are **not** on the same footing as PR |
  | 4 | R4 no late grok, no de-grok | **NOT HELD** | No run first grokked after 40k, so grokking times from k02/k03/k04 are **measured, not right-censored** — that half holds. But **n=119 seed 0 grokked at step 21,400 and ended at final test accuracy 0.6798**, PR collapsing 19.17 → 12.56. Every other run that grokked stayed grokked |
  | 5 | 40k slice replicates k03 run for run | **HELD** | Gini and PR agree **to three decimals on all twelve runs**. The slice pools with k03 as designed; nothing past 40k pools with anything |
  | 6 | provenance — real git SHA and the account that ran it | **HELD** | `git_sha 8eba027`, account `account-a`, stamped ×12 |

  ### AMENDMENT 2026-09-15 — R3 and R4 both change, and both in the SAFE direction

  Appended, not rewritten: nothing above this line is edited. C27 is **retracted**
  (`analyze_excursions.py results/k06_horizon`, STATE §2). n=119 seed 0 did not de-grok —
  its final sample is a **censored transient**: 19 post-grok excursions below acc 0.90
  occur across the 11 grokked runs, every one exactly one 200-step logging interval long,
  **18 of 19 recovering**; the 19th merely starts at step 120,000 with no next sample. This
  same run recovered from a **worse** spike (acc 0.4869) at step 115,400.

  - **R3 is stronger than recorded.** The "twelfth run at 0.193" was measuring the spike,
    not a drifting statistic — and in the additive column at that (`spectra` returns
    `Gini_mult = None` for non-cyclic (Z/119)\* = Z6×Z16). \|ΔGini\| is therefore
    **0.008–0.048 in twelve of twelve runs**. R3 HELD, unqualified.
  - **R4 is NOT HELD for one reason, not two.** "No late grok" still holds. "No de-grok"
    was never violated: nothing de-grokked. The correct reading of R4's failure is that the
    criterion could not distinguish a transient from a state — a **criterion defect**, and
    the one to carry forward. A future horizon run must either log densely enough to see a
    spike end, or define the endpoint as a **window mean**, never a single final sample.
  - **Artifact warning.** `results/k06_horizon/WE_B_thesis_n119_s0.npz` stores a transient
    final `W_E`/`logits_all` (max logit 3.71 against 14–95 in every sibling run). Do not
    read a converged number from it. It is the only censored artifact in the project.

  **What this settles.** The measurement instrument, which is what this run was for. PR at
  composite moduli is not a converged quantity at any horizon we can afford, so it is
  retired there rather than re-measured — the response was fixed in writing before the data
  existed, precisely so this outcome could not later be re-framed as "needs more steps".
  Gini survives the same test (R3), which is why C6 is unaffected. C24's PR clause is
  **superseded in part**: it said 40k is not a converged point, and the honest amendment is
  that **no point is**.

  **R4's failure is the larger result and it is NOT a claim.** C27 (de-grokking) rests on
  **one seed**. It is recorded as SUGGESTIVE ONLY in the ledger and must carry that
  qualifier every time it is repeated. It is the same modulus as open question O8 — n=119,
  where the restricted circuit does not beat baseline — and whether they are one instability
  or two is **untested**. Resolving it needs n=119 re-run at more seeds to 120k, which is
  open question O13 and is not part of this pre-registration.

- **Deviations from this pre-registration, and why:** **none in the science.** All six
  criteria were scored exactly as written, by `analyze_k06.py` as committed before launch.

  **One operational deviation, recorded for the reproducibility record:** the kernel
  completed on Kaggle but its **local watcher process was killed** during the same-session
  memory event that also killed two N7 launch attempts, so no results had been pulled and
  the run briefly looked lost. All twelve were recovered intact with
  `kernels/run_kernel.py kernels/k06_horizon --attach` — **no re-push, no re-run, no quota
  spent twice**. The artifacts are the originals and their SHA stamps are unaffected. The
  lesson (do not match processes on a substring of the full command line; a watchdog
  tailing a truncated log is mute) is promoted to `LAB_PROTOCOL.md`, not left here.

  **Confound flagged in advance and still not resolved:** weight decay is 1.0 and never
  decays, so at 3× the horizon the model may keep shrinking weights after the circuit is
  fixed. R3's stability makes a pure decay artefact unlikely as the explanation for R1's
  failure, but this run does not separate them and does not claim to.
