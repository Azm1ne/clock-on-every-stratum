# Grokking Thesis — lab protocol

Every constraint, convention and hard-won gotcha this project runs under. It is the
document the rest of the repository cross-references, and it is kept in the voice it
was written in: each entry exists because something went wrong once, and the entry is
the reason it did not go wrong twice.

## What this project is

Undergraduate thesis, multi-month. Mechanistic interpretability of grokked transformers on
`a*b mod n`, comparing circuits across the algebraic structure of `Z/nZ`.
The science is stated in the paper (`paper/`); every result is recorded in
`FINDINGS.md`.

## Document roles — do not blur these

| File | Role | Mutability |
|---|---|---|
| `FINDINGS.md` | **every result, organised by topic** — the scientific record | updated when a finding lands |
| `LAB_NOTEBOOK.md` | dated experimental record | **append-only, never rewrite history** |

## Hard constraints (from the researcher, not negotiable)

0. **REPRODUCIBILITY IS A GATE.** (Decree 2026-09-11, retroactive to the whole project.)
   Every code path and every experiment must be reproducible: pinned deps
   (`requirements.txt`), fixed seeds, a **real git SHA** stamped into every saved artifact,
   the exact command recorded, and `scripts/reproduce.sh` re-running everything from a bare
   clone. An unpinned dependency, an `unknown` SHA, or an unrecorded command is a **defect
   to fix, not a chore**. **And ignore non-reproducible sources** — no released code, a dead
   repo link, or unverifiable provenance means it is not cited and not built on. Judgment
   applies and must be stated out loud: "no public code" is not automatically
   "non-reproducible" if the method is fully specified, but it can never be the *sole*
   support for a claim. Record every exclusion in `related papers/CODE_AND_TOOLS.md`.
1. **Free Kaggle only.** 30 GPU-h/week, refreshing **00:00 UTC Saturday** (observed
   2026-09-12T00:00Z, API-reported next 2026-09-19T00:00Z — *not* Fridays; that error made
   the plan say "launch big grids Friday", aiming them at the window's last hours). Log
   every spend in `LAB_NOTEBOOK.md`, and **re-read quota live before planning a spend** —
   a stored figure is a spend record, never a budget (a pre-refresh "9.5 h left" understated
   the real budget by 20 h on 2026-09-12).
2. **Build from scratch.** The autograd engine and the transformer architecture are the
   researcher's own. The published author
   repositories and PyTorch autograd are for *checking*, not substituting.
   Anything trained on PyTorch autograd is **scout data, not paper data** — label it so.
3. **Debug locally, never on quota.** Smoke-test small before every push.
4. **Honesty over enthusiasm.** Negative results are results. Pre-register before running.
5. **The paper revision alters NO data, NO information and NO interpretation.** (Condition of
   approval, 2026-09-28.) Every revised passage passes **three line-by-line checks against the
   main version, `paper/` at `9c1eaa5`**: data, information and interpretation, defined in
   an internal design note §13. Text may move and
   exact duplicates may merge. The Discussion only gathers interpretations already in the
   paper or `FINDINGS.md`, at their recorded strength.

## Gotchas that have already cost time

- **`"machine_shape": "NvidiaTeslaT4"` in every GPU kernel-metadata.json.** Exact casing.
  Default `enable_gpu: true` gives a **P100 (sm_60)**, which Kaggle's own torch 2.10 cannot
  run — every CUDA op dies. `"T4x2"`, `"nvidiaTeslaT4"` and `--accelerator` are accepted
  and **silently ignored**.
- **Max 2 concurrent batch GPU sessions.** A push over the cap returns **rc 0** with the
  refusal in the body — the version saves, no session starts, and a naive caller thinks it
  launched. `run_kernel.py` now detects this and waits for a slot.
- **A kernel's TITLE must slugify to its `id`** or Kaggle rejects the push with a 409
  Conflict. "Grokking k04 Extended Moduli" ≠ id `grokking-k04-extended`.
- **A kernel id is OWNER-scoped** (`<account>/<slug>`), so running the same kernel from a
  second Kaggle account is not just a credentials swap. Do **not** hand-edit the five
  `kernel-metadata.json` files, and do **not** copy a token over `~/.kaggle/access_token`
  — that is how a push lands on an account whose quota and results nobody is tracking.
  ```bash
  KAGGLE_API_TOKEN=$HOME/.kaggle/<acct>_token KAGGLE_OWNER=<acct> \
    .venv/bin/python kernels/run_kernel.py kernels/kNN_whatever
  ```
  `run_kernel.py` rewrites only the **owner** half of the id in the temp copy it pushes
  (the slug never moves, so the title→slug rule still holds), and **refuses to push when
  the id owner and the authenticated account disagree**. Credentials resolve in the order
  `KAGGLE_API_TOKEN` (a token, or a path to a file holding one) → `~/.kaggle/access_token`
  → `kaggle.json` → `KAGGLE_USERNAME`/`KAGGLE_KEY`. A kernel that carries a
  `__KAGGLE_OWNER__` placeholder gets the account stamped into its results, which is what
  the reproducibility gate needs once results come from more than one.
  **Note the account is personal to its owner and its quota is theirs, not extra headroom
  for this project — log whose quota each run spent in `LAB_NOTEBOOK.md`.**
- **AN UNCOMMITTED TREE POISONS THE PROVENANCE OF A RUNNING SWEEP.** `provenance.stamp()`
  records `git_sha` **and `git_dirty`**, and a long local run **re-writes its `WE_*.npz`
  every snapshot — not once at the end**. So editing any tracked file while a sweep is in
  flight stamps `git_dirty=True` into artifacts written from that moment on. Caught on
  2026-09-12: a docs-only audit dirtied the tree and two of N7's live files
  (`WE_engine_n113_s1`, `WE_engine_n125_s1`) picked it up within minutes. It self-heals on
  the next snapshot **once you commit**, so: **commit before, or promptly during, any
  session that edits tracked files while a stamped run is writing.**
- **A SWEEP CAN CARRY MORE THAN ONE SHA, AND "it stamps X" AGES BADLY.** HEAD moving
  mid-run is picked up by later writes: N7's seed 0 stamps `f9d15b8`, seed 1 `70ebf2c`.
  That is acceptable **only** once `git diff --name-only <a> <b>` is shown to touch no
  training code — check it, record the check, and never assert a single SHA for a run that
  outlives a commit.
- **Kaggle has no git and no access to our env.** `os.environ.get("KERNEL_GIT_SHA")` is
  always unset there and silently stamps `"unknown"`. Kernels carry a `__GIT_SHA__`
  placeholder that `run_kernel.py` substitutes at push time, pushing a temp copy so the
  tracked source stays clean.
- **Sparsity protocol:** Gini on **amplitude** (`|Â|`), DC dropped, sin/cos combined — the
  Discrete-Log-Clock protocol — or numbers are not comparable to the literature. Squared
  energy inflates Gini ≈0.55 → ≈0.90.
- **But the PARTICIPATION RATIO goes on ENERGY**, `p_k = |Â_k|²/Σ|Â_k|²` — its standard
  definition. Amplitude gives 12.5 where the correct value is 4.3. Feeding amplitude into
  `participation_ratio` was open question O1 for three sessions and cost a failed
  pre-registered prediction (Entry 16). **The additive column cannot catch this** — a flat
  spectrum gives nearly the same PR either way (55.96 vs 55.86). Note `2406.03495`
  (Doshi/Gromov) uses a **third**, reciprocal convention, bounded in [0,1] and increasing
  with sparsity. Three "IPR"s in the literature; always state which.
- **`a+b`/`a−b` term names are meaningless on a multiplication run in raw coordinates.**
  Pass `dlog=True` to `neuron_term_decomposition` / `logit_top_components`. On k07 s0 the
  top-8 logit `a+b` share reads **17.0%** raw and **98.4%** in discrete-log coordinates (neuron
  terms: 0.36% raw, 21.7% dlog). **Never pair numbers from the two statistics** (P12).
- **A RATIO TO A CONTROL MEAN IS NOT A STATISTIC WHEN THE CONTROL IS HEAVY-TAILED.**
  C22/C23 reported `excluded / mean(random control)` from **20 draws**. The control
  distribution is bimodal: ~73% of random character removals leave the loss untouched
  (the model does not use those characters) and the rest are catastrophic, so the MEAN is
  decided by which few catastrophic draws happen to land. At n=121 seed 0 that ratio
  ranges over **21×–440× across control RNG seeds alone**, on identical data. Report a
  **permutation p** — the fraction of draws at least as damaging as the real set — the same
  threshold-free logic `test_crt_law.py` already uses for C8. Use the **median** control,
  never the mean, for any descriptive ratio, and 200 draws rather than 20.
- **GINI AT A SINGLE CHECKPOINT IS A NOISY ESTIMATOR — sd ≈ 0.02–0.044** (measured on k03,
  final 12k steps: 125 s2 wanders **0.498–0.621**, range 0.122). It is **oscillation about a
  flat mean, not a trend** — Gini plateaus by ~16–20k then jitters. So a single-checkpoint
  comparison against the usual **0.10** threshold is a **~2σ test, and only ~1.6σ at n=125**.
  **Average over the final ~10k steps** and report the within-run sd. Corollary: part of
  C6's reported across-seed spread (±0.026–0.048) is estimator noise, not seed variance.
- **`analyze_k03.spectral()` used to put ENERGY into `gini` — FIXED 2026-09-12.** It read
  0.902 where the amplitude protocol reads ~0.54–0.58 on the same run, and it was the sole
  source of C24's retired "Gini drifts <3%" clause. **A "do not copy it" warning had stood
  in this file for a defect nobody had actually fixed, so re-running the script kept
  regenerating the retracted number** — if a gotcha describes live broken code, fix the code
  or the gotcha becomes the bug's hiding place. Verified on `WE_B_thesis_n121_s0`:
  0.9003 → **0.580**, which lands inside C6's 0.556±0.028 for n=121 where 0.90 never could.
  `analyze_n7.mult_amplitude` is the canonical amplitude helper (self-checked);
  `analyze_k04/k05/k06/k07`, `refcheck` and `test_crt_law.freq_energy` are also correct.
- **`energy()` is the wrong folding for a Gini**: it *sums* each ±k pair but leaves the
  self-conjugate Nyquist bin single, so one bin is scaled by 1 where every other is scaled
  by 2. **Still live in `analyze_k03.spectral`'s PR path, deliberately** — energy is the
  right convention for a participation ratio and C24's VERIFIED PR numbers were measured
  on that exact path, so re-folding it would move published numbers with no re-measurement
  behind the change. Fix it only together with a re-run and a C24 amendment.
- **✅ O18 IS SOLVED (2026-09-15). PRECISION — ACTING ON TORCH, NOT ON THE ENGINE.**
  The 2×2 is complete. Tail `Gini_mult` at n=113: **engine f32 0.7671 · engine f64 0.7536 ·
  torch f32 0.5791 · torch f64 0.7728.** Precision is **inert in the engine** and
  **decisive in torch**: `f = +1.110` on Gini and `+0.968` on `|W_E|`
  (`PREREGISTER_o18_f64.md`, 3 seeds, 191 min local CPU).
  **⚠️ THIS DOES NOT UN-RETRACT C30.** C30 claimed a **main effect** and N9 falsified it
  (`f = −0.07`, sign wrong); that retraction stands. This is an **interaction**. Write
  *"training precision changes the measured sparsity **in PyTorch**"* — never C30's sentence.
  **The neuron inversion is explained too:** float64 moves torch's tuning 0.938 → **0.770**
  while moving Gini 0.579 → 0.774, i.e. **both axes at once, onto the engine's joint
  position** (0.7536, 0.887). The "one convergence axis cannot do that ⇒ maybe a different
  algorithm (Clock/Pizza)" reasoning below is **superseded** — one axis does do it.
  **CONSEQUENCE: every torch number here (k02–k09) is a float32 measurement**, and the same
  code reads 0.5791 vs 0.7728 at the two dtypes. Say which regime any absolute sparsity
  number came from. Between-modulus and within-arm comparisons (C6, C8, C22/C23, C31, C33)
  are untouched. C4 matched 2606.17399's published 0.579 **because both are float32** — that
  agreement is real; what is new is that the quantity is precision-dependent.
  **✅ THE 2×2's ENGINE COLUMN IS VALID — checked, not assumed.** The NEP-50 worry (an
  `ENGINE_DTYPE=float32` run trained with float64 gradients on five tensors) does not apply:
  the fix is `6991152`, N9 stamps `a4b3d14` + `dtype float32`, and
  `git merge-base --is-ancestor 6991152 a4b3d14` is true. **The fix predates the run.**
  **⚠️ WHAT REMAINS OPEN is no longer a defect but a question: the engine reaches the
  float64 answer at EITHER dtype and torch does not.** The gap is explained; the engine's
  dtype-robustness is not.
  **⚠️ `analyze_n9.py` PRINTS N9's `f`, NOT THE ONE A NEW ARM PRE-REGISTERS.** On
  `results/o18_f64` it says **"NOT HELD (f = −0.10)"** where the pre-registered F1 is
  **+1.110 HELD** — its baseline/target are the *engine's* two dtype cells. **Compute the
  committed formula; never quote that summary line for a non-N9 arm.**

  <details><summary>superseded framing, kept for the record (the question as it stood)</summary>

  **THE ENGINE AND THE KAGGLE KERNELS DISAGREE, AND ~~PRECISION~~ NOBODY KNOWS WHY (O18).**
  At byte-identical hyperparameters the engine reads **Gini_mult 0.75-0.84** where torch
  reads **0.54-0.58** — 7 of 7 grokked matched pairs, and the engine's `|W_E|` is also
  smaller. **Never pool engine and Kaggle numbers into one mean**, and always say which
  code trained a number. **C4's Arm A matched the published 0.579 because it is a torch
  arm.**
  **THE PRECISION EXPLANATION IS DEAD AS A MAIN EFFECT — but read what was actually tested.**
  N9 varied dtype **inside the engine only**. It says float64-vs-float32 does not move the
  *engine*. It does **not** test whether float32 does something to **torch**, and the half of
  the 2×2 that would (torch at float64) has never been run. Do not let this warning block it;
  what it forbids is reasserting "float64 makes circuits sparser", not completing the table.
  `engine.py` trained float64 and the kernels float32, so "float64 makes circuits sparser"
  was the obvious story and it had a 7/7 correlation behind it. **N9 (2026-09-14) ran the controlled A/B: the engine at
  float32 reads 0.7671 against its own float64 0.7536 and torch's 0.5665 — f = -0.07, the
  sign is WRONG.** Grokking time is precision-invariant too (4,500/6,400/9,600 vs
  4,500/6,500/9,700). Set `ENGINE_DTYPE=float32` to re-run that arm; **float64 stays the
  default**. (O1b no longer needs explaining — **CLOSED 2026-09-15, there was never a gap**:
  Nanda's "9.4k-14k" is their *Cleanup phase* of one run, and their stated grok time at
  wd=1.0 is "5-10k epochs", which contains our 6,900. Entry 38.)
  Already eliminated, so do not re-eliminate: protocol · hyperparameters · tail window ·
  optimizer (both reduce to `p(1-lr*wd) - lr*mh/(sqrt(vh)+eps)`) · gradients (2.68e-15) ·
  **minibatch noise, both are FULL-BATCH** · init distribution (norms agree to ~1%) · **the
  train/test split, byte-identical, the same 3,830 pairs**, **the substrate** (torch-on-CPU
  0.5791 vs T4 0.5665, f = 0.068, Entry 32), **the update path** (from identical weights at
  float64: gradients 2.08e-17, one AdamW step 8.65e-15, stage 1) and **the init DRAW**
  (the engine started from torch's own sampled weights lands at 0.7827, f = **+1.156**,
  stage 2, Entry 40). **ALL FOUR ENUMERATED HYPOTHESES ARE DEAD and O18 is a contradiction:**
  identical weights, data, forward, gradients and optimiser step, deterministic and
  full-batch, different endpoints.
  **⛔ THE ONE CELL NOBODY HAS RUN IS TORCH AT FLOAT64.** `kernels/k03_grid_acts/run.py:60`
  samples with `torch.randn` at the default dtype and nothing in this repo calls `.double()`,
  so **every torch number here is float32**. Read the next bullet before concluding that
  "precision is dead" forbids running it — it does not. ~35 min local CPU, zero quota.
  **And the two also differ in the opposite direction on neurons** — engine 70-93% tuned at
  Gini 0.75, torch 92-100% at Gini 0.57. One convergence axis cannot do that, so treat this
  as possibly a *different algorithm* (Clock/Pizza, 2306.17844), not a sparsity magnitude.

  </details>
- **A 7/7 CORRELATION WITH A MECHANISM THAT TIES UP THREE LOOSE ENDS IS THE MOST DANGEROUS
  SHAPE A WRONG CLAIM CAN TAKE.** C30 had all of that — sparser *and* smaller-norm *and*
  apparently faster-grokking is exactly the signature of less gradient noise under wd=1.0 —
  and it was false. Careful statistics do not rescue it. **Only running the variable does.**
- **A PRE-REGISTRATION THAT PRE-EXPLAINS A FAILURE IS NOT PERMISSION TO SKIP THE CHECK.**
  N7's E5 said in advance that a systematic engine-vs-torch disagreement would be "a finding
  about the science, not an engine bug" (C10 verifies gradients to 2.3e-15). Taken at face
  value that would have shipped a precision artifact as a scientific claim. **C10 verifies
  the GRADIENTS, not the optimizer, not the dtype, not the data split.** When a
  pre-registered criterion fails, enumerate and eliminate the boring causes **first** —
  protocol, hyperparameters, window, optimizer — and only then reach for the interesting one.
- **A RUN'S ENDPOINT IS A WINDOW, NEVER ITS LAST ROW — AND AN EXCURSION TOUCHING THE FINAL
  SAMPLE IS CENSORED, NOT AN OUTCOME.** C27 ("a network can de-grok") was k06's n=119 s0
  read at its final logged sample, 0.6798. It is a **transient**: 19 post-grok excursions
  below acc 0.90 occur across the 11 grokked runs, **every one exactly one 200-step logging
  interval long, 18 of 19 recovering**; the 19th merely starts at step 120,000 where no next
  observation exists. That run recovered from a **worse** spike (0.4869) 4,600 steps
  earlier. Use `analyze_excursions.py`, which prints `*CENSORED*` for exactly this. Post-grok
  spiking is ubiquitous under wd=1.0 and its rate rises with horizon (0.33 %/sample at 120k
  vs 0.09 % at 40k). ⚠️ `results/k06_horizon/WE_B_thesis_n119_s0.npz` therefore stores a
  **transient** final `W_E`/`logits_all` — the only censored artifact in the project (all
  nine results dirs swept). **Classify every run by the median of its final ~10 samples.**
  Writing this down did **not** stop me re-committing it in new code an hour later; what
  worked was making the code *print* the disagreement between last-row and window.
- **A RANDOM-ORTHOGONAL CONTROL MUST ROTATE THE RESIDUE AXIS, NOT THE FEATURE AXIS.**
  `mult_amplitude`/`energy` **sum energy over features**, so `W @ Q` for orthogonal Q is a
  provable **no-op** — it returns the signal *exactly*. `src.analysis.transforms.random_orthogonal`
  is correct (`Q.conj().T @ W`, rotating residues) and `analyze_n7.rand_amplitude` wraps it
  with the dimension matching; **use those, never a hand-rolled QR.** The failure mode is
  nasty: a control that returns exactly the signal does not look broken, it looks like a
  striking result ("the clock is no better than a random basis"). Asserted both ways in
  `analyze_omega.py::_selfcheck`.
- **`git rev-parse HEAD origin/master | uniq | wc -l` IS A FALSE-PASS WAITING TO HAPPEN.**
  If the ref does not resolve, the error goes to **stderr** and one line still reaches
  stdout, so the pipeline prints `1` — the pass value — whether or not the remote has the
  work. Verify a push against the **remote itself**:
  ```bash
  git ls-remote --heads origin | grep -w refs/heads/master   # compare to git rev-parse HEAD
  ```
  Same family as every other false-pass here: **a check whose failure mode is
  indistinguishable from its success mode is not a check.**
- **A BASELINE THAT CONTAINS THE EFFECT IT BOUNDS MAKES ITS RULE UNFALSIFIABLE.** O18-T's
  T3 was specified as engine-float64 vs engine-**float32** (dtype only) and implemented as
  engine-float64 vs **torch** — so the "rounding baseline" carried the very engine-vs-torch
  difference under test, and the decision rule would have printed "agree to rounding"
  whatever the data said. **Before reading a verdict, look at the control's own numbers**
  and ask whether the rule has any input that could flip it. Same family as C7b/C19/C20/G1.
- **CHECK WHETHER THE ANSWER IS ALREADY ON DISK BEFORE DESIGNING THE EXPERIMENT.** O9
  ("do 49/54/63 resist grokking?") was answered by `results/k05_lowdata`, which had swept
  `train_frac` at exactly those three moduli since 2026-09-11 — 3/3 groks at f80, monotone.
  O14's proposed enrichment-vs-step sweep was likewise sitting in `n7_engine`'s `we_traj`.
  Two open questions closed for zero compute because someone finally read existing artifacts
  against them. **`ls results/` and check `we_traj` density first; it costs one command.**
- **A FLOAT32 FORWARD COMPARED AGAINST A FLOAT64 ONE READS ~1e-7 AND LOOKS EXACTLY LIKE A
  LAYOUT BUG.** The kernels sample with `torch.randn`, so their models are **float32**.
  Promote with `.double()` before any cross-implementation equality check — the float32 →
  float64 → float32 round trip is exact, so the model is unchanged. This nearly aborted a
  correct experiment as "the layout map is wrong" (it was right, at 7.772e-16).
- **Never report `Gini_mult` alone.** A purely additive signal scores 0.53–0.63 in the
  multiplicative basis. Always report (mult, add, random-orthogonal) + key-frequency count.
- **A BINARY GATE TURNS EVERY NEAR-MISS INTO WHICHEVER SIDE IT FALLS ON.** `test_acc > 0.99`
  is the right gate for "is this a data point"; it is the **wrong** gate for "is this a
  control". k03's `n125_s1` ends at **0.9779** — a model that has plainly learned the
  circuit — and was being scored as the *ungrokked negative control*, reading 74.8% neuron
  tuning and quietly destroying the contrast the control exists to provide. Use three
  states: grokked / near-grok / FAILED, and build controls only from runs that **failed**
  (engine `n125_s1` at 0.6082, k04's 49/54 at 0.15-0.27). This biases **against** the
  finding, which is the direction nobody checks.
- **NUMPY PROMOTES float32/int64 BACK TO float64 (NEP 50), SILENTLY.** `Tensor.max`'s
  backward divided by an int64 tie count, and both softmaxes route through `max`, so an
  `ENGINE_DTYPE=float32` run trained with **float64 gradients** on W_Q, W_K, W_in, W_out
  and W_U — invisible from the loss curve. `test_autograd.py::test_dtype_flag` now asserts,
  per dtype in a subprocess, that weights, gradients, Adam state and loss are **all** the
  requested precision. Divide by `.astype(g.dtype)`, never by a raw integer count.
- **A `python - <<PY` HEREDOC THAT EDITS A FILE BEHIND `assert` WRITES NOTHING IF A LATER
  ASSERT FAILS** — the file is opened for writing only after every assert passes. And a
  following command on its **own line** runs regardless (no `&&`), so a verification printed
  after it can look fine while the edit never happened. This silently dropped a STATE.md
  edit on 2026-09-14 whose commit message claimed it. **Make the edit script print a unique
  success token, and verify something that DEPENDS on the edit** — not a neighbouring
  section that was already correct.
- **`hist[-1][3]` IS TEST ACCURACY IN THE KERNELS AND TRAIN ACCURACY ON THE ENGINE.**
  k03/k04 write `hist_cols = step, train_loss, test_loss, test_acc`; `run_n7.py` writes
  `step, train_loss, test_loss, TRAIN_acc, test_acc`. **The same index silently means a
  different quantity in the two arms.** Indexing `[3]` would have admitted the engine's
  `n125_s1` (test 0.6082, train ~1.0) to a PRIMARY arm as a data point, and made every
  engine split-check compare a recomputed test accuracy against a logged train accuracy.
  **Read by column name from `hist_cols`.** Same family as `$4=="run_n7.py"` — never index
  to a fixed field when the fields can move. Caught by printing `hist_cols` from all three
  results directories instead of assuming they agreed.
- **A LIVENESS CHECK HAS NOT BEEN TESTED UNTIL IT HAS SEEN BOTH A LIVE JOB AND AN IDLE
  BOX.** `scripts/alive.sh <script.py>` is the one to use; it matches the script name as a
  **whole field** (so `awk -v s=foo.py` cannot see its own argv) and requires **argv[0] to
  be a python interpreter** (so the `bash -c "... foo.py ..."` wrapper does not match).
  Its own first version put the interpreter guard on `$1` — which is the **PID** under
  `ps -eo pid=,args=` — and reported a live 40,000-step run as **idle**. That is the
  direction that makes the next session relaunch on top of a running job.
- **AN IMPOSSIBLE WAITER DOES NOT DIE WITH ITS SESSION — THEY ACCUMULATE, AND THEY OUTLIVE
  THE CONTEXT CLEAR THAT ORPHANED THEM.** On 2026-09-15 a session-end checkpoint liveness sweep found
  **FIVE** `bash -c` waiters from session 8 still spinning after **13 h 20 m - 13 h 34 m**,
  1-3 s of CPU each, children `sleep 10/15/20/30`. Four required `! pgrep -f "analyze_gate2.py"`
  and the fifth `! pgrep -f "analyze_k03.py"`; every one of those `bash -c` processes carries
  its own script name in its **own argv**, so pgrep matched the waiters themselves (and each
  other) — the condition could never become true. **The first sweep found only four; the
  fifth surfaced because it waited on a DIFFERENT script.** Enumerate by PARENT —
  `ps -eo pid=,ppid=,etime=,args= | awk '$2==<parent pid>'` — never by the script name you
  happen to remember. The worst was `until [ -s a.log ] && ! pgrep -f "..." && [ -s b.log ]`: **both file
  conditions were satisfied** and it still spun, because the impossible clause was in a
  **conjunction**. Their follow-on work never fired; it had been run directly instead, so
  nothing was lost — but that is luck, not design.
  **Three consequences.** (1) A naive `pgrep -f <script>` on an IDLE box returned **5 PIDs**;
  `scripts/alive.sh <script>` returned **blank**, correctly — the instrument is now tested
  against a live job *and* an idle box carrying decoys, in the same session. **Use
  `alive.sh`, never bare `pgrep -f`, for any liveness question.** (2) Sweep for orphans at
  session-end checkpoint and kill them **by explicit PID**, never by pattern:
  `ps -o pid=,ppid=,sess=,etime=,time=,stat= -p <pid>` to confirm, then `kill <pids>`.
  (3) **Wait on a disjunction containing a provable condition**, never a conjunction
  containing an unprovable one.
- **DO NOT PUT THE PATTERN YOU ARE GREPPING FOR LITERALLY IN THE SAME COMMAND.**
  `ps ... | grep "[g]ate2_all_done" | xargs kill` killed the shell running it (exit 144)
  on 2026-09-14 — the bracket stops *grep* matching itself, but the **shell running the
  command** has the marker path in its argv. Third occurrence of this family. Build the
  pattern from pieces (`P="gate2_all"; P="${P}_done"`), or collect PIDs into a file in one
  command and kill from the file in the next.
- **A STATISTIC THAT IS LOAD-BEARING ELSEWHERE IS NOT AUTOMATICALLY LOAD-BEARING HERE.**
  G2's pre-registration named the permutation p PRIMARY *because* it is the project's
  established statistic (C22/C23/C32) — and it passes on **100 % of FAILED** stratum
  measurements, because 30 % of every block is training data and a memorising model's
  logits are still a function of `u*v` there. C22 applies it to the unit stratum of a
  **grokked** model, where memorisation is not a live alternative. **Before reusing a
  statistic in a new setting, score it on a failed-run control and print the pass rate on
  both** — `scripts/gate2_discriminate.py` is the shape of that check.
- **`pdftotext` ON A TWO-COLUMN PAPER INTERLEAVES THE COLUMNS AND HYPHENATES ACROSS LINES,
  SO A CORRECT QUOTE READS AS ABSENT.** Verifying 21 quotes against the nine-paper corpus
  (2026-09-15), four came back FAIL — including both of `2607.07066`'s limitation sentences,
  the two the whole respondent framing rests on. They were right; the instrument was wrong.
  `-layout` puts the left column's line and the right column's line on **one** output line,
  and words break as `"only cor-" / "relational"` — so flattening whitespace does not fix it
  and de-hyphenating does not either, because the two halves are separated by the *other*
  column's text. Crop the columns instead:
  ```bash
  pdftotext -f $PG -l $PG -x 0   -W 300 -H 800 paper.pdf -   # left
  pdftotext -f $PG -l $PG -x 300 -W 300 -H 800 paper.pdf -   # right
  ```
  **The failure direction is the dangerous one**: had I trusted the grep over my own reading,
  I would have deleted two correct and load-bearing quotations as unverifiable. Any citation
  check against a two-column PDF — which is most of them — hits this.
- **A BIBLIOGRAPHY ROW SAYING "IDENTIFIER UNKNOWN" IS NOT EVIDENCE THE PAPER IS MISSING.**
  Two rows of `related papers/grokking_bibliography_dois.md` carried "arXiv ID unknown" while
  **the PDFs sat in `related papers/papers/`**: `#24` is `2502.10390`, `#39` is `2604.20923`.
  The same file's note that `2606.17399` is "image-only, will need OCR" is **false** —
  `pdftotext` reads it at 304 lines. That file warns on its own first page that it was
  "compiled from a research report and from memory", so **it is input, never a source**:
  check every identifier against the PDF. Same family as O9 and O14 — `ls` first.
- **A NUMBER QUOTED FROM `FINDINGS.md` IS A NUMBER NOBODY HAS CHECKED. RE-DERIVE IT FROM THE
  ARTIFACT.** Agreeing with the lab record proves only that **two documents agree**.
  `test_paper_numbers.py` recomputes every load-bearing number from its `.npz` and has found
  **four defects in one session**, none of which prose review caught:
  (1) FINDINGS §3.3's column labelled `restricted / baseline` holds **`baseline / restricted`**
  — copying the header prints the ratio **upside down** — and it is a **mean over a
  heavy-tailed distribution**: at n=113 one seed of five reads **264×** where four read
  1.56–8.0, giving "55.5× ± 104.1", a summary whose sd is twice its value. *That is the
  control-mean defect, in the column three paragraphs below the box that retired the
  control-mean defect.* **Report the median and the range.**
  (2) "grok at 20 of 23 moduli, three exceptions" → **19 of 23 in a majority of seeds**, only
  n=49 fails every seed, and there is a **fourth exception, n=91**.
  (3) "p_correct reaches **exactly** 1.000000" is a **six-decimal display**; in float64 it
  maxes at **1 − 3.4e-8** and never reaches 1.
  (4) the O18 2×2's torch-float64 cell is **0.772770**, not the recorded **0.7736**, so the
  pre-registered `f` is **+1.110** not +1.114 (verdict unchanged, threshold 0.70). **Three of
  the four cells matched to 4 dp and the fourth did not** — a 4th-decimal mismatch that is
  easy to wave off as rounding. It was not rounding.
  **Add every number a document asserts to `test_paper_numbers.py` before relying on it.**
- **`provenance.stamp()` SPREADS ITS KEYS; `provenance.read()` LOOKS FOR ONE `provenance` KEY
  AND RETURNS `None`.** An artifact stamped with `stamp()` is stamped and **unreadable by the
  project's own reader**, which looks identical to a correct stamp from the outside. Use
  **`stamp_npz`** (LAB_PROTOCOL.md already said so) and make the writer **assert the stamp reads
  back** before it prints a result.
- **STAMPING A GENERATED FIGURE OR TABLE'S PROVENANCE TAKES TWO COMMITS.** A caption cannot
  name the commit that contains it. The first attempt stamped `c787670`, which an `--amend`
  **orphaned seconds later** — a caption naming a commit `git log` cannot reach is worse than
  no caption, because it *looks* checked. Commit the generator, then stamp its SHA in a second
  commit, and verify with `git ls-tree -r <sha> --name-only | grep <script>`.
- **A `\ref` INSIDE THE SECTION IT LABELS IS A PLACEHOLDER, AND THE OBVIOUS CHECK CANNOT SEE
  IT.** It passes "does the label exist" because the label really is there. **Four shipped in
  the first drafted section; drafting the other twelve produced seventeen more, in every
  file.** `paper/check_tex.py` now fails on it, with a planted case in its self-check. Its own
  first version ended a subsection's block at the next `\section`, so a subsection swallowed
  its siblings and false-positived on a legitimate sibling reference — **split at the next
  heading of any level.**
- **OVERLEAF'S NONSTOP MODE RECOVERS FROM TeX ERRORS WITHOUT A RED BUILD — COMPILE LOCALLY
  WITH `-halt-on-error`.** `tab:moduli` declared `rlccrrrrc` (9 columns) over 10-column rows
  from `b70159a` (2026-09-15) until 2026-09-26: `! Extra alignment tab` was swallowed and the
  `grokked` column merged, on the paper's reference table, for eleven days. Found by the
  first local compile. TeX Live is installed now:
  `.venv/bin/python paper/check_tex.py --compile` fails on any TeX error, undefined
  ref/cite, or a box > 10 pt into the margin (three tables were 85-167 pt out), and its
  `--selfcheck` plants the column bug. **Overleaf is retired (2026-09-28): `bash scripts/build_tmlr.sh` builds `TMLR/` and runs it.**
- **A TRAILING `%` COMMENT IN LATEX SWALLOWS THE REST OF ITS LINE, AND A COMMENT-STRIPPING
  CHECKER STRUCTURALLY CANNOT SEE IT.** One ate a sentence of body text. Keep every comment
  **first-on-line** and scan for it:
  `re.search(r"(?<!\\)%", line) and not line.lstrip().startswith("%")`.
- **WHEN A CHECK YOU JUST WROTE FIRES, RESOLVE IT — DO NOT WEAKEN IT TO PASS.**
  `reconstruct_provenance.py`'s coarse AST check fired on k02's `Transformer`. Weakening the
  hash would have "passed"; instead it now reports *what* moved (the `acts=True` return tuple)
  and *whether the training call path can reach it* (it cannot — training and eval use the
  default `acts=False`). The verdict is stronger **because** the first check failed.
- **68 ARTIFACTS CARRY NO USABLE `git_sha` — k02 (31, no stamp), k03 (31, `unknown`) AND
  `k01_scout` (6, no stamp).** ⚠️ *This entry said 62 and named only k02 and k03 until the
  2026-09-23 audit swept every artifact. The scout six ARE claim-carrying:
  `test_paper_numbers.py` reads them for F6's 14.08×/11.76×/3.30× and for §9's
  network-key-set claim. `k00` (2) is also unstamped and carries no claim; the `ckpt_*.npz`
  are transient resume files, not results.*
  ⚠️ ***And "`gate1` carries no claim" was WRONG** (said here until 2026-09-26): §10 quotes
  four numbers from it (grok 6,900, five key freqs, 7.3×, 85.0% vs 84.6%). ✅ **Recovered by
  bit-identical re-run** at `1eaf0e3` (`PREREGISTER_gate1_repro.md`): the live
  `results/gate1/*.npz` are now stamped clean; unstamped originals in
  `results/_archive_gate1/`. **A "carries no claim" reassurance is a claim — grep the paper
  for the artifact's numbers before trusting it.***
  Everything from k04 on is stamped. **k03 is the source of C22, C23, C24, C26, C31 and Gate 2**
  — most of the paper. It is **recoverable**: k03's `run.py` has **exactly one commit** in
  history, and k02's three differ only in what is saved. **k01 ran BEFORE the repo's `init`
  commit**, so one version in history certifies nothing; Kaggle's stored source for the
  version that ran is byte-identical (sha256 asserted). `scripts/reconstruct_provenance.py`
  makes and checks all three arguments. **It is a reconstruction, not a stamp — say so.**
  `test_paper_numbers.py` scores the 37/31/68 counts against §11/§12.
- **`scripts/reproduce.sh` had silently fallen 13 scripts behind the repo**, including
  `test_crt_law.py` (C8). `test_reproduce.py` now fails the self-check suite when a
  claim-carrying script is missing from it, and `session_brief.py` runs it. Adding a new
  `analyze_*.py` / `test_*.py` means adding it there or to its `EXEMPT` map with a reason.
- Use `.venv/bin/python`, and `PYTHONPATH=.` for anything importing `src`.
- `kernels/run_kernel.py <dir> --attach` reattaches to a running kernel without re-pushing.

- **A FIGURE FUNCTION WITH NO CALLER IS A STALE FIGURE, ONE STEP EARLIER.** `causal_test`,
  `precision_2x2` and `grok_curves` were added in session 14 and
  `grep -rn "causal_test" --include=*.py --include=*.sh .` outside `plots.py` returned
  **nothing** — they had been run once by hand. `figures/**/*.png` is gitignored by design,
  so **F4, the paper's central evidence figure, could not be regenerated from a fresh
  clone at all.** Every figure the paper uses is now a row in `render_all.py`'s cross-panel
  list. **Adding a `plots.py` function means adding it there**, the same rule
  `test_reproduce.py` already enforces for `analyze_*.py`.
- **PRINT SIZE IS THE BINDING CONSTRAINT ON A FIGURE, AND IT IS INVISIBLE ON SCREEN.**
  A `figsize=(13, 9)` figure at `\textwidth` (~6.5 in) scales by **0.5**, so a 9 pt
  annotation prints at **4.5 pt**. ✅ **Since WP2 (2026-09-28) every paper figure is drawn AT
  print size** (`TEXTWIDTH` × its `\includegraphics` fraction) and `plots._save()` resizes it
  until the PDF is exactly that wide, so a `fontsize` in the code IS the printed size; floor
  **7 pt**. The old "10.2 in, 8.5 pt" rule had drifted to 3.1-5.4 pt printed.
  `docs/revision/figure_checks.py` measures printed size and proves plotted data unchanged
  against a main version — **run it after any `plots.py` edit.** The trap is the second
  half: shrinking the figure with the fonts fixed makes
  text occupy *more* relative space, so every collision you already fixed comes back — and
  the fix is **cutting copy, not nudging coordinates**. Six layout defects this session were
  found only by rendering the PNG and *looking at it*; none was visible in the code. WP2
  found thirteen more the same way.
  **Three traps in `_save`'s loop, all hit:** a text line wider than the page makes it
  "converge" by shrinking the axes to nothing (it now asserts); `wrap=True` text always
  fills the figure, so the loop chases it forever (break long lines by hand); and a CFF font
  (Nimbus Sans, TeX Gyre Heros) under `pdf.fonttype 42` embeds with a poppler "Mismatch
  between font type and embedded font file" — use **Liberation Sans** (TrueType).
- **A FIGURE'S OWN TEXT IS A DOCUMENT, AND NOTHING SCORED IT.** F4's panel said
  **"(200 draws)"** and **"p = 0.0000"** (the retired `b/B`) for six days after R1 moved the
  null to 10,000, directly above a caption saying 10,000 and p̂ = 1.0e-4.
  `test_paper_numbers` scored the caption against the artifact the whole time, which is why
  nothing fired, the same shape as the C33 ledger cell. Found only by reading the print-size
  render (P4, `de23661`). **Never hard-code a count or a statistic into a figure string —
  derive it from the data it labels** (`len(draws)`, `stats.perm_p`), and pin the derivation.
- **A MODULE-LEVEL RNG MAKES A FIGURE DEPEND ON CALL ORDER.** `test_crt_law.RNG` is seeded once
  at import, so drawing F6 twice in one process gives two different nulls from identical
  code. That looked exactly like a data change in the figure harness. A fresh process is
  deterministic; a harness must re-import the repo's modules per capture (it does).
- **`matplotlib` MATHTEXT IS NOT LaTeX, AND AN f-STRING EATS ITS BRACES.** `\bigl`, `\bigr`,
  `\underbrace`, `\pmod` and `\bmod` all raise `ParseFatalException`; use plain parentheses
  and `\;\mathrm{mod}\;`. Worse, in an f-string `\mathrm{mod}` is a **replacement field** —
  it fails as `NameError: name 'mod' is not defined`, which reads like a maths error and is
  a Python one. Double the braces (`\mathrm{{mod}}`) or drop the `f` prefix.
- **A FIGURE THAT CAN BE SILENTLY WRONG NEEDS A CONTROL THAT CANNOT.** `crt_law()` asserts
  the published **14.08×** at n=165 *before it draws anything*, because the
  `predicted`-returns-frequencies / `freq_energy`-returns-frequency−1 off-by-one is loud at
  exactly one modulus and silent at every other. `stratum_theorem()` likewise asserts the
  stratum identity **and** the non-regularity of the class it illustrates. A diagram cannot
  be allowed to illustrate a theorem it violates.
- **A CAPTION IS A NUMBER-CARRYING SENTENCE AND IS READ MORE OFTEN THAN THE BODY.** Every
  value a caption asserts goes into `test_paper_numbers.py` like any other
  (53 → 109 this session). Prose review does not catch a caption.
- **BEFORE COPYING CLAIM-CARRYING CODE INTO A FIGURE, EXTRACT IT — THEN DIFF THE OUTPUT.**
  `analyze_n4.run`'s grid assembly *and key-set choice* nearly got a second copy in
  `plots.py`: two answers to "which characters are the key set", on C22/C23. It now lives in
  `ablation.unit_logit_grid`. **An extraction is not safe until the caller's full output has
  been captured before and after and diffed** — done twice this session (`analyze_n4`,
  `test_crt_law`), byte-identical both times. When a figure needs a statistic's internals,
  add a `return_null` / `return_draws` flag to the one implementation; never write a second.
- **`alive.sh` MATCHES THE SCRIPT NAME AS A WHOLE FIELD, SO PASS THE SPELLING THAT IS IN
  ARGV.** `scripts/alive.sh render_all.py` returned **blank on a job that was running**
  (PID 103591, `RNl`, 4 min in) because argv holds `scripts/render_all.py` and the whole-field
  match cannot see a bare basename inside it. `scripts/alive.sh scripts/render_all.py` returns
  the PID correctly. The instrument is not broken — **the call was** — but the failure lands in
  the false-*idle* direction, which is the one that makes the next session relaunch on top of a
  running job. **Confirm any blank `alive.sh` against
  `ps -eo pid=,args= | awk '{for(i=2;i<=NF;i++) if($i=="<exact argv field>"){print;break}}'`
  before believing the box is idle.**
- **A MEAN OF TWO POINTS IS HALF THEIR DIFFERENCE WEARING A STATISTIC'S CLOTHES.** C36
  shipped as `239.52x +- 236.74` at n=128. The two runs are **2.77 and 476.26**; the `+-` is
  the *population* sd of two points, so it is arithmetically forced to equal half the gap and
  carries no information at all. Worse at n=131: `262.19x +- 367.03` is the mean of
  2.44/2.88/**781.26**, whose **median is 2.88x** — the published number describes **no run in
  the sample**, and the abstract led with it. `analyze_n4` prints mean+-sd for any seed count,
  including **two**. The control-mean box above retired a mean over a heavy-tailed *control*;
  this is the same defect over the heavy-tailed **effect**, three sections away, and it
  survived because the fix was filed as being about controls. **Report median + range + n for
  every ratio, and refuse to print `+-` below n=5.** Fixed 2026-09-16; asserted in
  `test_paper_numbers.py`.
- **AND THE COLUMN WAS LABELLED UPSIDE DOWN AGAIN — THIRD TIME.** `FINDINGS.md`'s C36 table
  said `restricted/baseline` and held `baseline/restricted` (0.36 vs 2.77 at the same cell).
  Same family as FINDINGS §3.3. **Before quoting a ratio column, recompute one cell both ways
  and see which matches the header.**
- **A REVIEWER'S FINDING IS A HYPOTHESIS, NOT A DEFECT — RE-DERIVE IT.** Of six substantive
  findings from the 2026-09-16 panel, four confirmed and **two were wrong**. A domain reviewer
  reported "27 of 29 is really 26"; the artifact says **27 of 29 is correct and the exception
  LIST is wrong** — the failures are n=98's seed (0.015) and n=49 (0.010), and **n=63 passes at
  p=0.0** (the 0.085 failure is n=54, in a different sweep). Acting on the report would have
  replaced a right number with a wrong one. Another reviewer's line numbers were fiction
  (cited `:288` in a 145-line file) while its substance held. **Check the file length before
  trusting a line number, and score the claim against the `.npz`, never against the report.**
  **⚠️ AND BEING RIGHT ABOUT AN ARTIFACT IS DATED EVIDENCE.** That overturn was correct at
  B = 200 and **the count still moved — to 28 — when R1 deepened the null**, for a reason the
  reviewer never raised (n=49 crossed the threshold *downward*, 0.010 from two draws of 200 →
  0.0050 from 49 of 10,000). A verdict scored against an artifact carries that artifact's
  timestamp. **Re-score it when the artifact is regenerated; do not inherit it.**
- **A 200-DRAW NULL CANNOT EXPRESS `p = 0.0000`.** Its floor is `1/201 ~ 0.005`. The paper
  printed four decimals of a precision the design cannot deliver, in eight places. Report the
  bound (`0 of 200 draws, p < 0.005`) or raise B. Same family as every other entry here: a
  display that cannot distinguish "very small" from "smaller than measurable".
- **`np.argsort(np.argsort(x))` IS AN ORDINAL RANK, NOT SPEARMAN'S — AND A SELF-CHECK ON
  CONTINUOUS DATA STRUCTURALLY CANNOT SEE IT.** It equals the Spearman rank only when nothing
  ties; with ties it breaks them **by array position**, so the statistic reads whatever the
  array happened to be sorted by. `omega(n)` takes **three distinct values over 23 moduli**, so
  the nine `omega = 1` moduli were ranked in *n*-order, and §9's *"rho(zdd,G|omega) = +0.735
  sits above rho(omega,G|zdd) = +0.627"* **inverted** under midranks (+0.578 vs +0.804) — the
  same two numbers, reversed. `analyze_omega._selfcheck` scored `[1,2,3,4]` and continuous
  Gaussians, where ties have probability **zero**, so it could not have failed. Sixth instance
  of a check whose failure mode is unreachable. **Use `src/analysis/stats.py`** (`rankdata`,
  `spearman`, `partial_spearman`, `perm_p`); its self-check **plants a tie** and pins the
  ordinal form at a spurious `rho = 1.000` for a two-level grouping variable against twelve
  distinct values, swinging to **0.510** on a pure reordering while the midrank does not move.
  **Midranks are the DEFINITION of Spearman under ties, so fixing this is a defect fix, not a
  criterion move** — no threshold was touched and W2/W5/C9 kept their verdicts. The only real
  tie in C9: **n=63 and n=147 both have zdd exactly 3/7.**
- **A FIX APPLIED TO PART OF A CODEBASE LEAVES THE REST ASSERTING THE OLD VALUE — AND THE TEST
  SUITE CAN END UP LOCKING IT IN.** R1 migrated `analyze_n4`/`analyze_gate2`/`run_intervention`
  to Phipson-Smyth `(1+b)/(B+1)` and **stopped**; `test_crt_law.py` (C8), `analyze_k04.py` (C9)
  and `test_crt_null.py` (C28) still computed `b/B` — the form `05-methods.tex` calls *"a trap
  we fell into"* while line 315 claims the same logic runs §9. Worse, `test_paper_numbers.py`
  **asserted `_p == 0.0`** for F6 while line 285 of the same file documents deleting that exact
  assertion elsewhere as unreachable. **When you retire a formula, grep for the FORMULA across
  every path, and then grep the guards for the value you just made impossible.**
- **A GLOB OVER A SHARED OUTPUT DIRECTORY POOLS ARMS, AND THE POOLED FIGURE LOOKS LIKE A
  RESULT.** ✅ *Fixed 2026-09-25: `--summary <n>` is now required and asserts each artifact's `n`.* `run_intervention.py --summary` globbed `I1_*.npz` under `I1_OUT`; I1b (n=121) writes
  into the same `results/i1_internal/` as I1 (n=113), so it printed
  `median 1.635e+07, range 7.460e+03-2.780e+08, **n = 6**` across two moduli — exactly the
  cross-modulus import that arm's pre-registration forbids. **Scope a cross-seed summary by the
  variable that defines the arm, not by the directory it happens to live in**, and read the `n`
  before quoting any median.
- **`provenance.stamp()` RECORDS THE TREE AT *SCRIPT START*, NOT AT WRITE — AND THAT CUTS
  BOTH WAYS.** `analyze_gate2.py` calls it at the top of `main()`, so an 8.5-hour run that
  wrote its artifact at 23:49 stamped the SHA and clean-tree of **15:26**. This is why
  editing tracked files during *that* analysis run is safe — ⚠️ **but not every analysis
  script stamps at start. `analyze_n4.py` builds its stamp INSIDE `np.savez(...)`, i.e. at
  WRITE time, so any tracked-file edit during its (~2 h on k04) run stamps `git_dirty=True`.**
  That is how F4's artifact went dirty, and on 2026-09-25 the re-run was kept clean only by
  staging the next task's edit in a scratch copy and installing it after the write. **Before
  editing during any run, grep the script for where `stamp`/`stamp_npz` is called.** This
  entry said "analysis runs are safe" unscoped until then — and why the LAB_PROTOCOL.md rule about
  dirtying a live sweep is about **training** runs, which re-stamp every snapshot. The other
  edge is sharper: **do not edit the analysis script itself mid-run**, because the artifact
  will claim a SHA whose code is not what executed. Check the same way every time —
  `git diff --name-only <stamped-sha> HEAD`, then confirm none of the changed files is
  imported by the script. `analyze_gate2` imports only
  `src.analysis.{ablation,neurons,transforms}` and `src.provenance`; 26 files moved under it
  and none of them mattered.
- **ONE FUNCTION ANSWERING TWO QUESTIONS IS HOW THE LAST-ROW DEFECT SURVIVED, AND FIXING IT
  NAIVELY BREAKS THE OTHER ONE.** `analyze_gate2.final_test_acc` was used by **both** the
  three-state classifier (which must read a **window** — C27) and `check_split` (which must
  read the **last row**, because `logits_all` is that checkpoint's output and it compares at
  a 0.01 tolerance). Substituting a window median would have made ordinary post-grok jitter
  read as *"split does not reproduce"* and **silently skipped good runs** — while the diff
  looked like a defect being fixed. `src/analysis/runs.py` now exports `last_test_acc`
  beside `state`, and its self-check asserts the two **disagree** on the planted C27
  history: if they ever agree, one is wrong. **Before replacing a function's body, list its
  callers and ask whether they are asking the same question.**
- **AND IT REACHED TWO DOCUMENTS OUT OF FOUR — count them.** R1 moved k04 from 27 of 29 to
  **28 of 29 with a SINGLE exception**. The correction landed in the abstract and
  `FINDINGS.md`; it did **not** reach `13-conclusion.tex` (*"27 of 29 ... the two
  exceptions"*) or the **appendix ledger's C23 row** (`$27/29$`), and both sat wrong for a
  day. `test_paper_numbers.py` asserted 28 against the `.npz` the whole time — **which is
  exactly why it never fired.** Scoring the artifact while the prose drifts is not a check on
  the prose. **Enumerate every document that states the number** (abstract, body, conclusion,
  appendix ledger, FINDINGS, STATE) and score each one; `grep -rn` the *old* value across
  `paper/`, `FINDINGS.md` and `STATE.md` before closing the task. Fifth and sixth instances.
- **A CORRECTION REACHES THE DOCUMENTS SOMEONE IS EDITING, NOT THE ONES THEY ARE NOT.**
  `STATE.md`'s C33 row carried `G2 excluded/baseline 3.98e+02-1.0e+08` for six days after
  the 2026-09-16 pass corrected it to **2.71e+08** in `FINDINGS.md` and in the paper — inside
  a row marked **VERIFIED**. `test_paper_numbers.py` asserted 2.71e8 the whole time, against
  the **paper**, which is precisely why it never fired on the ledger. **When a number is
  corrected, grep every document for the OLD value before closing the task** — and prefer a
  test that scores the artifact against *each* document that asserts it, not against one.
  **And a correction written down as a LESSON corrects nothing.** The O18 torch-float64 cell
  was re-derived as **0.772770 (f = +1.110)** on 2026-09-16 and recorded in this file as a
  lesson, while this file's own O18 box, `STATE.md` (8 lines, the C34 ledger row among them),
  `FINDINGS.md`'s running text and the `write-paper` checklist kept saying **0.7736 / +1.114**
  for twelve days. Fixed 2026-09-28 (Entry 67). The old-value grep must cover `STATE.md`,
  `FINDINGS.md`, `LAB_PROTOCOL.md` **and** the drafting aids.
- **A REASSURANCE IS SCOPED TO THE ARM IT WAS MEASURED ON, AND THE SENTENCE WILL NOT SAY SO.**
  This file recorded, correctly, that the `hist[-1]` endpoint defect changes 4 of 178 runs and
  *"none of the four is in k03, so Gate 2's 157 measurements, C31 and C38 are unaffected."*
  True — and it covers the **primary** arm. Gate 2's **control** arm is built from **k04**, and
  `k04_extended/WE_B_thesis_n49_s2` is one of the four: last row 0.9631 (near-grok, *excluded*
  from the controls) vs window median 0.8908 (FAILED, **included**). Fixing the rule therefore
  **adds a run to the control arm** — "19 failed measurements" becomes 20 and every F9 count
  moves. **Before trusting "X is unaffected", name every arm X has and check the one the
  sentence did not mention.** The tell is that the reassurance was measured on the arm someone
  happened to be looking at.
- **GREP FOR THE NUMBER, NOT FOR THE CONCLUSION — AND FOR WHAT A TABLE OMITS.** Updating the
  paper's permutation statements, `grep "p < 0.005"` found five and **missed two** that carry
  the same fact phrased as *"beyond all $200$ draws"* without the word `p`. Sweeping for the
  literal `200` found eleven, of which **three were a 200-STEP logging interval** and one was
  the methods section's own history of the defect — none of those four should move. The same
  hour, asking which artifacts the R1 family table did *not* list found the two **tracked**
  gate-2 control files, overlooked entirely, still at B = 200 and still printing `p_perm = 0.0`
  after the whole point of R1 was to retire that. **Enumerate by the quantity, then classify
  each hit; and read a table for its absences, not only its rows.**
- **"EVERY FIGURE THE PAPER USES IS NOW WIRED IN" WAS FALSE THE DAY IT WAS WRITTEN.**
  Session 15 fixed F4/F7/F8 and wrote that comment into `render_all.py`; **`basis_comparison`,
  the paper's 13th figure, was never added** and was still unreproducible from a bare clone a
  day later. Found only when bundling for Overleaf. **Verify a completeness claim by diffing
  the two lists** — `grep includegraphics paper/sections/*.tex` against `render_all.py`'s cross
  list — never by having just edited one of them.
- **A SECOND REPRESENTATION OF THE SAME NUMBERS IS BLOAT, NOT THOROUGHNESS.** §8 printed the
  Gate 2 discrimination as a six-row table; F9 draws the same twelve counts. The figure was
  given the table's `separates?` column and **the table was deleted**. Adding a figure beside
  a table that already says it means the next edit updates one of them.
- **A CHECKER THAT `rglob`s A DIRECTORY WILL EVENTUALLY READ A COPY OF ITS OWN INPUT.**
  `paper/check_tex.py` scanned `paper/` recursively, and an unzipped `overleaf_bundle/` is a
  complete second copy of the sources: it reported **30 .tex, 270 refs, 122 cites** where the
  paper has 15/135/61 — **and 0 failures**, because a stale copy of a correct paper is also
  correct. Fixed 2026-09-21: it skips the one named build artifact and **FAILS on any other
  duplicate basename**, both planted in `--selfcheck`. The bundle directory is gitignored
  beside its zip. **A count that silently doubles is the same false-pass family as every other
  entry here** — and the tell was the count, not a failure.
- **`open(out, "w")` BEFORE `provenance.stamp()` MAKES A SCRIPT DIRTY THE TREE IT IS ABOUT TO
  DESCRIBE.** On a NEW output path the open creates an untracked file, so
  `git status --porcelain` is non-empty and the artifact stamps `git_dirty=True` against a tree
  that was clean an instant earlier. `analyze_gate2.py` (stamp computed 30 lines before the
  open) and `analyze_n4.py` (`savez` evaluates `stamp_npz` first) were already correct;
  `analyze_r2_sameness.py` was not, and its first two artifacts carry the false flag in git
  history. **Compute the stamp first, and assert it reads back.** Corollary observed the same
  hour: regenerating artifact A then artifact B makes B stamp dirty because A is uncommitted —
  **commit between writers of tracked artifacts.**
- **RUN A LONG STAMPED JOB IN ITS OWN `git worktree` AND THE MAIN TREE IS FREE TO EDIT.**
  `git worktree add --detach <dir> HEAD`, launch with `WorkingDirectory=<dir>` and
  `PYTHONPATH=<dir>` under the main `.venv`; `git status` there sees only that checkout, so
  paper edits for 3 h did not touch Gate 1's re-run (2026-09-26). **But a script that writes
  a TRACKED output file dirties its own worktree**: `run_gate1.py` rewrites the tracked
  `results/gate1/history.json` every 500 steps, so its snapshots stamped `git_dirty=True`
  from step 500. Before launching, `git ls-files <output dir>`; if anything there is
  tracked, `git update-index --skip-worktree` it **in the worktree** and record that as a
  deviation. Then `cmp` the file against the tracked copy afterwards, so the skip is shown
  to have hidden no content change.
- **TWO EMPTY SETS COMPARE EQUAL, SO "NOTHING DETECTED" SCORES AS A PERFECT MATCH.** R2's
  agreement statistic compares key sets between strata; white noise yields **no** key set
  (20/20 in the self-check), and `frozenset() == frozenset()` is `True`. An undetected clock
  would therefore have read as agreement. `equal` is now false for an empty set **by
  construction**, and excluded pairs are counted. Same shape as every other "the failure mode
  looks like the success mode" entry — and it was caught by a self-check written before the
  statistic ever saw real data, which is the only reason to write them first.
- **A *PERFECTLY* PURE PLANTED SPECTRUM HAS MEDIAN 0, AND `key_freqs_5x_median` THEN SELECTS
  FLOAT NOISE.** `x > 5 * median` with `median == 0` is `x > 0`. Caught planting the Nyquist
  bin at `orders=(20,)`. Real blocks never have a zero median (amplitudes summed over `n_out`
  are strictly positive), so **plant the character WITH a broadband floor** — and assert the
  degenerate case on purpose rather than avoiding it. **Test the bin convention with `argmax`,
  which no threshold can rescue or break.**
- **`(d,e)` AND `(e,d)` ARE NOT INDEPENDENT EVIDENCE — THE TASK IS COMMUTATIVE.** Their
  diagonal spectra agree for a reason that has nothing to do with a shared circuit. Counting
  them in R2's agreement rate would have been the C7b/C19/C20 size confound in a new costume.
  Excluding them removes **every** primary pair at n=121 and n=125, which is worth knowing
  before designing the experiment, not after. **Split the statistic and report transposes
  descriptively.**
- **✅ THE LAST-ROW DEFECT IS FIXED (2026-09-22), IN ONE PLACE — and re-read this entry
  before re-fixing it.** It stood here as "⚠️ LIVE DEFECT, NOT YET FIXED" for
  `analyze_gate2.final_test_acc` **and** `analyze_o4.py`, both classifying by `hist[-1]`
  against this file's own C27 rule. Both now route through `src/analysis/runs.py`:
  `analyze_gate2` via `43df124`, and `analyze_o4.py:93` reads
  `run_state(z)[0] == "grokked"`. **A banner claiming live broken code that is actually
  fixed is the same hiding place as the reverse** — the 2026-09-12 `analyze_k03.spectral`
  entry — and it costs the next session a re-fix. Verified by
  `grep -rn "hist\[-1\]" --include=*.py .`: every surviving hit is a `kernels/*/run.py`
  progress print or `final_test_acc` field, where index 3 **is** test accuracy, plus
  `run_c12_collapse` (maxlogit) and one comment.
  **What the fix actually cost, now that it has been paid.** Scored both ways over 178 runs,
  **4 change state.** The reassurance printed here — *"none of the four is in k03, so Gate 2's
  157 measurements, C31 and C38 are unaffected"* — was true **and scoped to the primary arm**.
  Gate 2's **control** arm is built from k04, and `k04_extended/WE_B_thesis_n49_s2` is one of
  the four: last row 0.9631 (near-grok, excluded) vs window median 0.8908 (FAILED, included).
  Collecting it **moved four of the six F9 rows** — G0 `0/19 → 1/20`, G1 `19/19 → 20/20`,
  G2 `0/19 → 1/20`, G5 `0/13 → 1/14` — and retired *"G0, G2 and G5 separate completely"*.
  **One run, one measurable stratum, four published numbers.** ⚠️ Its window median clears
  `FAIL_ACC = 0.90` by **0.009** on a *rising* window (0.842 → 0.963; not monotone — one dip, 0.8489 → 0.8477, P11), so the
  FAILED-arm threshold is load-bearing on a hair. **DECIDED 2026-09-22: it stays at 0.90.**
  The defence is the ORDERING, not the margin — 0.90 is the C27 excursion floor from
  `932ef89` (09-15), centralised in `37dba23` (09-21), and the arm was not scored until
  `cbe13f3` (09-22). **A threshold fixed a week before the measurement is inherited; one
  moved after seeing which side a run falls on is C6/C7 again.** Settle any threshold question by planting the value, never by reasoning:
  0.89 gives near-grok, 0.91 gives FAILED, and my first draft of that sentence had the
  direction backwards.
- **A WHITELIST OF SCRIPT PREFIXES IS A FALSE-IDLE GENERATOR, AND WIDENING IT JUST MOVES THE
  MISS.** `session_brief.py` §3 matched `.venv/bin/python … (run_|kernels/|scripts/)` and
  printed **"no local jobs running"** while `analyze_gate2.py` — which lives at the **repo
  root** — was 20 min into a 47-min run at 100 % CPU. **Fourth false idle in this family**,
  and the comment directly above that line recorded the 2026-09-12 one (four N7 workers,
  10.5 h in), which had been "fixed" by adding a third prefix. Retired the whitelist: match
  any `.py` under our interpreter (`python -c` and `python -m pip` carry no `.py`, which is
  all the whitelist ever bought). Five planted argv cases assert it **at import**, so a
  narrowing makes the brief refuse to run rather than report an idle box. **A false idle is
  the one direction that makes the next session relaunch on top of a running job — and
  the start-of-session brief exists to prevent exactly that, so its own instrument
  failing is the worst case.** Confirm any "idle" against `systemctl --user show <unit> -p SubState`.
- **A STRUCTURAL MISMATCH THAT ABORTS A SUBTREE SKIPS THE COMPARISON YOU CAME FOR, AND STILL
  PRINTS A VERDICT.** `scripts/r1_diff.py` walked two JSON artifacts in parallel; k04's
  control arm went 6 runs → 7, the walk logged "length 6 -> 7" and **returned**. It compared
  **48** values, all from `verdict`, and reported R1-4 — while the **six runs common to both
  versions were never compared and nothing said so.** Keyed alignment by `(n, seed)` /
  `(d, e)` takes the same artifact to **2,106** values. **When a container's shape changes,
  diff the intersection and name the additions; never return.** Same family as every other
  entry here, and it also needs the other guard: **assert the walk compared more than zero
  values**, because two empty sets compare equal.
- **RUN A DIFF AGAINST AN ARTIFACT THAT HAS NOT CHANGED BEFORE TRUSTING IT ON ONE THAT HAS.**
  `r1_diff` against the untouched committed k03 artifact — where the only correct answer is
  "0 moved" — failed twice: three `acc_heldout` values read as MOVED because **NaN ≠ NaN**,
  and it walked `provenance`, which a re-run legitimately restamps (`git_sha`, timestamp), so
  the engine arm would have failed R1-4 on pure noise. **k03 passed only because its file was
  untouched.** A positive control costs one command and found both.
- **A CRITERION THAT CAN BE REINTERPRETED AFTER SEEING THE NUMBER IS NOT A CRITERION.** R1-1
  set ≥ 95 % of previously-`p = 0.0` tests staying at the floor; the measurement is **85.5 %
  (171/200)** and the same pre-registration's falsification section says, in advance, that
  `b > 0` is *"not a falsification"*. Both are in the document; the convenient one is not the
  verdict. **Reported NOT MET.** ⚠️ **And the prior session's Outcome had claimed "R1-1 /
  R1-2 hold" on evidence that `n_draws = 10000` everywhere — which is R1-2 and only R1-2.**
  R1-1 had never been computed. **Before recording a pre-registered criterion as met, compute
  *that* criterion**; adjacent evidence is not evidence. Same ordering error as C6/C7.
- **A LEDGER ROW THAT HAS ALREADY GONE STALE ONCE WILL GO STALE AGAIN, IN THE SAME CELL.**
  `STATE.md`'s C33 row carries a note that its `excluded/baseline` figure sat wrong for six
  days because the 2026-09-16 correction reached `FINDINGS.md` and the paper and not the
  ledger. **On 2026-09-22 the same row's `0 %/0 %/0 %` separation cell was stale for the same
  reason**, plus a second copy in §2's "never quote all six criteria" bullet. The fix that
  actually holds is a **test that scores each document that asserts a number**, not one that
  scores the artifact and trusts prose: `test_paper_numbers.py` now parses FINDINGS' criterion
  table row by row. **Negative-control it** — planting a stale `0/19` must exit 1 — which is
  also how the `chk(label, got, want)` argument order was caught printing the document under
  "artifact". **Fourth upside-down column in this project.**
- **BEFORE A RE-RUN OVERWRITES A GITIGNORED ARTIFACT, COPY IT.** R1 pre-registered "diff the
  B-independent quantities against the committed artifacts", and `results/*_n4_ablation.npz`
  are **gitignored** — so there was no committed version and the re-run destroyed the
  comparison it was supposed to be checked against. The two `results/gate2/*.json` are tracked
  and were checkable by `git diff`. **`git ls-files <artifact>` before you overwrite it**;
  if it comes back empty, `cp` it somewhere first.

- **✅ EVERY RUN NOW SAVES ITS WEIGHTS — and before `cb3f941` NONE DID.** `run_n7.py:171`
  deletes the resume checkpoint on a clean finish (*"resume state is dead weight"*) and
  `snapshot()` wrote only `W_E`/`W_U`, so **0 of 232 artifacts carried
  `W_Q`/`W_K`/`W_V`/`W_O`/`W_in`/`W_out`/`W_pos`** and **no completed run in this project
  was recoverable as a model.** Any experiment that re-runs the *forward pass* — rather than
  post-processing saved logits — was therefore impossible without retraining, which is not
  what STATE's one-line scoping ("re-run the forward pass on saved weights") implied.
  `snapshot()` now saves all nine, named `p_*` to match `checkpoint()` so one loader reads
  both. **Check what an artifact actually contains before designing an experiment around
  it** — `np.load(p).files`, one command. Pre-`cb3f941` artifacts still have no weights, so
  an intervention on an older run still means a retrain.
- **A RETRAIN IS NOT A SECOND-BEST — IT BUYS A POSITIVE CONTROL NOTHING ELSE CAN.** The
  engine is deterministic and full-batch, so re-running a published config at HEAD
  reproduces it **bit-identically**: `max|ΔW_E| = max|ΔW_U| = max|Δlogits| = 0.000e+00` on
  3/3 seeds, grok steps 4,500/6,500/9,700 recovered exactly. That turns "a model like the
  published one" into "the published one, now with weights", which is the difference between
  a new claim and a claim about existing results. **Check the training-path drift first** —
  `git diff --name-only <artifact SHA> HEAD` filtered to `src/{autograd,model,train,tasks}/`
  and the runner — and record that it is inert. **And `run_intervention.py` ABORTS unless the
  checkpoint reproduces its archive**; a guard that cannot fail is not a guard, so it was
  negative-controlled against the retired pre-leak-fix checkpoint (aborts, dW=2.6e-01).
- **REPORT A CONTROL'S MAXIMUM, NOT ONLY ITS MEDIAN — A NULL'S TAIL CAN REACH THE EFFECT.**
  I1's permutation null is significant on 3/3 seeds (`p̂` 1/10001, 1/10001, 5/10001) **and**
  seed 2's control **maximum exceeds** the excluded loss, with seed 0's at 99.2 % of it.
  `excluded / median(control)` reads 8.2e+07 and **hides this completely.** The cause was
  benign and *had to be shown to be*: the pool includes the key characters (inherited from
  `analyze_n4` for comparability), so a draw can partially perform the intervention — damage
  is monotone **in the mean** with overlap, and **12,468 zero-overlap draws never exceed
  1.05e-04** against an excluded loss of 24–47. **Look at the control's max and its shape
  before believing a ratio to its centre.** Same family as the control-mean defect, one
  statistic over.
- **INHERIT EVERY THRESHOLD YOU CAN — IT IS THE ONLY C6/C7 DEFENCE THAT DOES NOT RELY ON
  YOUR OWN RESTRAINT.** I1's three criteria were Gate 2's G2 (100×), `FAIL_ACC` (0.90) and
  C22/C23 (p < 0.01) — all fixed earlier, for other purposes, before the experiment existed.
  A threshold chosen for the run it judges is arguable however carefully it is justified; an
  inherited one cannot be tuned to the answer because it predates the question. Same
  reasoning that defended `FAIL_ACC = 0.90` on 2026-09-22: the defence is the **ordering**.
- **A COUNT WRITTEN IN WORDS ("THE OTHER TWENTY-THREE") IS A DERIVED NUMBER — SCORE IT FROM
  THE TABLE IT IS DERIVED FROM.** The paper's `tab:moduli` has 23 rows *including* the two
  intervention moduli. The conclusion said "one modulus … the other twenty-three" (wrong
  before anyone touched it), and on 2026-09-25 "twenty-two" was copied from STATE's C40 row
  into §7 and FINDINGS (also wrong — **twenty-one**). No prose check catches a spelled-out
  count. `test_paper_numbers` now counts the table's rows and demands "the other
  <count − 2>" in every section that says it. And a phrase-ban needs a needle **specific to
  the sentence it retires**: bare "at one modulus" fired on grok-time prose in §6 and §12.
- **A COMPLETENESS DIFF CAN FALSE-POSITIVE ON A LINE WRAP, AND SO CAN A PROSE GREP.** The
  figure check (`grep -oP '\("\K[A-Za-z0-9_]+(?=", )'`) reported `basis_comparison` as
  unwired because its entry wraps after `",`; the paper-prose check reported the abstract as
  missing `$28$ of $29$` because the abstract wraps between `$28$` and `of`. **Flatten
  whitespace before matching prose, and never require punctuation-plus-space on one line.**
  Both failures land in the direction that looks exactly like the defect being hunted — the
  same shape as `basis_comparison` genuinely going unwired for a session.

- **`\a` IN AN `re.sub` REPLACEMENT IS THE BELL CHARACTER, AND IT MANGLES SILENTLY.** The
  replacement template is *parsed*, so `"\\author{}"` becomes `<BEL>uthor{}`. Emptying
  `paper/main.tex`'s author block produced **`uthor{}}`** — also one brace short, because
  `[^}]*` stops at the `}` inside `\texttt{[affiliation]}`. **Pass a lambda as the
  replacement** (it is returned verbatim, no template parsing) and match one level of
  nesting: `\{(?:[^{}]|\{[^{}]*\})*\}`. The sibling trap is already in this file: a
  `\a`/`\b`/`\n` in a *pattern* is fine; it is the **replacement** that bites.
- **A PATTERN GATE CANNOT SEE STRUCTURAL CORRUPTION — RUN THE FORMAT'S OWN CHECKER TOO.**
  `uthor{}}` passed every content check in `scripts/check_public.py`, because mangled LaTeX
  contains no forbidden *pattern*. It was caught by **opening the file**, and the gate now
  shells out to the export's own `paper/check_tex.py`. Any transform of a structured format
  needs the format's validator in the same gate, or the transform's worst failure mode is
  invisible to it.
- **A SENTINEL THAT COLLIDES WITH ORDINARY PROSE IS A BAD SENTINEL.** The unfilled-DOI check
  grepped for `PLACEHOLDER` and fired on `LAB_PROTOCOL.md`'s own sentence *"A `\ref` INSIDE
  THE SECTION IT LABELS IS A PLACEHOLDER"*. A gate that fires on vocabulary is one its reader
  learns to wave through — which is worse than no gate. Use a token that cannot occur
  naturally, e.g. a double-underscored sentinel. Same hour: **`\b<word>\b` does not match
  `<word>s`** — when substituting words, put the plural first. And a lesson that QUOTES the
  token it is about cannot itself survive the substitution: write it with a placeholder.
- **A GENERATED FILE IS NOT IN `git ls-files`, SO A SWEEP DRIVEN BY THE TRACKED LIST SKIPS
  IT.** `LAB_PROTOCOL.md` is written by the export builder, so the `LAB_PROTOCOL.md` → `LAB_PROTOCOL.md`
  rename ran over 38 tracked files and **not over the new file's own self-references**. Per-file
  edits had that gap by construction. **One substitution table applied to every file in the
  output tree cannot have it** — sweep the OUTPUT, never the input list.
- **SCRUBBING IDENTITY FROM A REPO DOES NOT SCRUB IT FROM THE ARTIFACTS.** `provenance.stamp()`
  leaks nothing by itself — no user, no hostname, relative paths — but **68 of 250 `.npz`
  carry `kaggle_account`**, and that account name contains the researcher's surname. No
  repo-side transform reaches inside a 4.6 GB deposit. **Enumerate every field of every
  stamped artifact before claiming a release carries no identity**
  (`provenance.read(p)` over `results/**/*.npz`, one loop). And note the direction of the
  fix: **editing a provenance record to conceal identity falsifies the one thing provenance
  exists to guarantee**, so the choice is "document it" or "do not deposit it", never
  "rewrite it quietly".
- **A HISTORICAL RECORD LEGITIMATELY NAMES FILES A RELEASE DOES NOT CARRY.** `LAB_NOTEBOOK.md`
  is append-only and refers to `STATE.md` and the drafting aids; a dangling-reference check
  fires on all of them, and "fixing" the notebook would falsify the record. The answer is an
  **explicit allowlist with a reason per entry**, which also **reports entries nothing
  references any more** — otherwise the allowlist rots into a blanket waiver and the check
  stops meaning anything. An allowlist without a staleness report is a disabled check.

- **A NUMBER CHECK CANNOT SEE THE WORD THAT SAYS WHAT THE NUMBER IS.** §9 called the 42
  divisor-dual frequencies at n=165 the J-class indicators' **support**; they are where the
  energy **concentrates** (30.5×) — the support is all 82, because a Ramanujan sum never
  vanishes at square-free q. `test_paper_numbers.py` asserted `== 42` the whole time and was
  right; the sentence around it was wrong. Found only by writing the claim as a formal
  statement. **When a number is asserted to BE something (a support, a bound, "exactly",
  "every"), check that property, not just the count** — and add a prose check for the word.
  **And a retired claim can survive with NO number in it.** On 2026-09-28 the Limitations
  section still said zero-divisor density "remains the stronger partial correlate", five days
  after the midrank audit made §9 say the data "cannot rank them". The old-value grep that
  closes a correction found nothing, because the stale sentence carries no number. **When a
  correction retires an ordering or a verdict, grep for its WORDS** (*stronger, above,
  displaces, exceeds*), not only its values.
- **A SCRUB OF FILE CONTENTS DOES NOT SCRUB FILE NAMES.** `make_public_export.py` relabelled
  the second Kaggle account in every file's text and shipped a **directory** carrying the
  real name; the prose then pointed at a path the tree did not have. Anonymization must walk
  **path names** too (`rename_identity_paths`), and the gate must match paths
  (`check_public.py` does). Corollary, same hour: a build script that greps for an identity
  pattern **contains** that pattern — exclude it from the export like `check_public.py`.
- **THE REVISION CHECKER (`docs/revision/revision_checks.py`) FLAGS BY RULE, NOT BY MEANING —
  RESOLVE EACH FLAG FOR WHAT IT IS (earned in WP1, Entry 68).**
  - **Number words are data.** Retitling "Two honest caveats" as "Scope" dropped a datum
    ("two"). Keep the count in the retitle, or pair the two units by hand. WP3's heading
    pass meets this again ("Three primes, not one").
  - **An unrelated-looking MAIN/REV pair is a deleted unit that the matcher attached to an
    unchanged sentence by one shared rare word** ("limitation", "sentence", "load-bearing").
    The decision says the REV sentence is unchanged. Editing that sentence to silence the
    flag is the wrong fix.
  - **A rewrite that falls below the word-overlap threshold** (60 %, or 40 % with every datum
    present) is paired with `u:<id>` `rewritten_as`. The pair then needs its own `c:`
    decision keyed to the exact revised text, so pairing by hand accepts nothing on its own.
  - **Count decisions and lint deltas before writing a commit message.** Both first-draft
    figures in WP1's message were wrong, and both were caught only by re-counting.

- **✅ `refcheck.py` SECTION [2] AND `analyze_scout.py` PUT ENERGY INTO `gini` — FIXED
  2026-09-29 (P6/P6b).** Both now call `analyze_n7.mult_amplitude` / `add_amplitude`, and
  `test_paper_numbers` bans any Gini of energy in either script. The paper's n = 165 pair is
  **0.629 / 0.637** (was 0.885/0.884 on energy) and §9's indicator Ginis **0.406 / 0.450 /
  0.400 / 0.000** (was 0.829/0.829/0.683/0.018). **`gini(np.sqrt(energy(...)))` IS NOT THE
  AMPLITUDE PROTOCOL EITHER** — WP3's interim "amplitude" values 0.626/0.622 and
  0.416/0.457/0.395/0.018 were exactly that: it keeps DC and scales each self-conjugate bin
  by 1 where every other is scaled by 2 (`energy()` sums ±k pairs). A value can be labelled
  "amplitude" and still be off-protocol. **Use the helpers, never a hand-rolled √.**
  `test_analysis.py::report` still uses energy on purpose: a planted-signal ordering check,
  not a measurement.
- **A CAPTION THAT BREAKS WITH `\par` NEEDS A SHORT TITLE (`\caption[short]{long}`)** or
  hyperref's `\NR@gettitle` aborts the build ("Paragraph ended before ..."). STE captions over
  six sentences need `\par`, so they need the short title too. `check_tex` and
  `revision_checks` read `\caption[...]{`; before 2026-09-29 such a caption was invisible to
  both, which is the direction that hides a caption from every check.
- **IN THE REVISION CHECKER, A DELETED OR RETITLED MAIN UNIT WITHOUT A `u:` DECISION FLOATS.**
  The heuristic attaches it to any sentence sharing its rarest word, so a neighbour's edit
  re-opens an unrelated WP's decision. Give it a `u:` decision (`deleted` if it carries no
  datum, else `rewritten_as`), which now removes it from the heuristic entirely.
- **A TABLE THAT RESTATES OTHER TEXT NEEDS ITS COUNTS TIED TO THE ARTIFACT, NOT ONLY TO THE
  PROSE.** Table C's "every outcome number appears in the cited section" check passed a planted
  stale "27 of 29", because §7 tells that count's history. The count cells are now recomputed
  from their artifacts. Same family as every "two documents agree" entry above.
- **COUNT BEFORE YOU WRITE THE COMMIT MESSAGE.** Four WP3 messages carried wrong lint or
  decision counts and were amended before push. Compute them (`_findings_for` on
  `git show HEAD:<file>`, and the `wp` field in `decisions.json`) first.
- **`figure_checks.py` PRINTS AT MOST EIGHT STRINGS PER LIST, SO A COUNT READ OFF ITS OUTPUT
  CAN BE SHORT.** 3d accepted "seven" n/a annotations in gate2_strata; there are **ten**. The
  stale-free re-check caught it. Build `fig:` pairs from a count taken from the figure itself,
  and after any wording pass **scan the rendered PDFs** (`pdftotext | grep -E '\b[A-Z]{2,}\b|—'`):
  that scan found "ONE" and "NO", which WP2's hand-made held list had missed.
- **A TABLE ADDED IN ONE SECTION CAN RE-OPEN A DECISION IN ANOTHER.** Table B's new rare tokens
  crossed `revision_checks`' ≤ 25-occurrence cutoff and re-attached an F7 caption fragment in
  `11-implementations` (untouched) to the wrong group. Pin any decided pairing with a `u:`
  `rewritten_as` so a distant edit cannot move it (Entry 70's rule, now seen in the wild).

- **EVERY NUMBER THE PAPER PRINTS IS GATED (P7, 2026-09-29).** `test_paper_numbers.py` fails on a
  printed number that no passing check asserts, unless `paper/number_ledger.json` records it —
  `exempt` with a reason (a cited value, a threshold) or `todo` (pinning debt, printed every run).
  **Adding a number to `paper/` means adding its pin in the same commit.** Pin against the
  claim-carrying script's *own printed output* (`_script(...)` in that file), never a second copy
  of its computation. Three traps, all hit: (1) a check that compares a **scaled** value
  (`round(x/1e8, 2) == 2.71`) is invisible to the gate — assert the printed magnitude; (2)
  matching is by value, so an unrelated equal value can "cover" a number — a covered `todo` must
  be deleted only when a real pin exists, else marked `coincidental`; (3) the gate found **four
  stale printed values** the day it ran (P9a–c are the midrank defect's survivors: the 2026-09-23
  fix reached the code and §9's partials, and **three printed ρ/bands outside that sentence kept
  the ordinal numbers**). When a statistic is fixed, recompute every printed value it produced.
- **PINNING A PRINTED NUMBER MEANS RE-DERIVING IT, AND ONE IN SEVEN DID NOT RE-DERIVE (session 32).**
  Paying 62 `todo` entries found **nine** stale or mislabelled numbers (P10–P18): a strata count
  printed as stratum-runs, a retired statistic still quoted (C23's 10.8–59.6×), a B = 200 maximum
  under a 10,000-draw figure, an upside-down column (fifth time), a first measurement never
  refreshed (C10). **Never pin by copying the paper's value into the check; find the script or
  artifact, reproduce it, and only then assert.** Where the paper is wrong, assert the TRUE value
  as a history check and leave the ledger entry `todo` with the P-id, so the gate keeps it visible.
  Three mechanics in `test_paper_numbers.py`: (1) module-level names are REUSED further down
  (`d`, `_grok`, `_t7`, `_i1` are rebound or defined later) — load explicitly in a new block;
  (2) a number LaTeX wraps across lines (`2.881 \times` / `10^{7}`) is tokenised as its mantissa,
  so assert the mantissa too; (3) `analyze_n4.run` hard-codes `default_rng(0)` and `N_CONTROL` —
  history runs patch both in a `try/finally`, never a second copy of the null.
- **`reproduce.sh` COVERAGE IS PER SCRIPT, NOT PER MODE — AND THAT HID TWO RESULTS.** It ran
  `test_crt_null.py` only on N7 and `analyze_omega.py` without `--confirm`, so O20's table and
  C37's W1–W6 regenerated from no command it issued, while `test_reproduce` passed. When a
  script has modes or argument-selected arms, check the invocation that produces **the paper's**
  number is in `reproduce.sh`. Experiments re-train from `scripts/sweep_table.py --commands`,
  whose `--selfcheck` ties each command to its stamped argv and fails on unreviewed training-code
  drift (`INERT` is keyed by blob, so editing a reviewed file re-opens it).

- **A MATPLOTLIB `-|>` HEAD STOPS 1.0 pt SHORT OF ITS END POINT, AND ON A TIGHT LOOP IT LANDS
  ON THE WRONG PART OF THE CURVE.** Measured at 7200 dpi (matplotlib 3.11.1): constant across lw
  and mutation scale, in shrink and path mode. Aim the tip 1.0 pt *inside* the target; adding a
  mitre allowance is the wrong direction. On a self-loop whose radius is under about 2 head
  lengths, draw the arc as a plain line and the head as its own two-point patch.
  `descent_lattice` does both. **Measure, do not reason.** The first two attempts reasoned and
  were wrong. Re-measure if matplotlib is upgraded.
- **A NEW TRACKED DIRECTORY JOINS THE ANONYMOUS EXPORT BY DEFAULT.** `for arxiv/SUBMIT.md` names
  the author and sat in the TMLR supplement's input for a session, until the first full
  `build_tmlr.sh` failed its gate. **Adding a tracked file that names the author means adding
  its path to `make_public_export.EXCLUDE` in the same commit.** Never write the affiliation into
  `LAB_NOTEBOOK.md`, which ships.
- **TO BUILD FROM A TREE ANOTHER PROCESS HAS DIRTIED, BUILD IN A WORKTREE; DO NOT CLEAN THEIR
  FILES.** Run `git worktree add --detach <d> HEAD`, then `ln -s $PWD/.venv <d>/.venv` (and add
  `/.venv` to `.git/info/exclude` while it exists), `cp -rp figures/. <d>/figures/` (with `-p`,
  or the freshness check calls every figure stale) and `cp -rln results/. <d>/results/` (hard
  links, so no gigabytes are copied). Copy the outputs back, then `git worktree remove --force`.

## Non-negotiable process (added after Gate 1)

- **Pre-register before every run.** Copy `experiments/PREREGISTER_TEMPLATE.md`, fill it in,
  and **commit it before launching**. Git's timestamp is the evidence the criteria predated
  the result. Gate 1's C6/C7 were edited after failing — defensible but the wrong ordering,
  and it must not recur.
- **Stamp provenance on every saved result** — `src/provenance.py`; `stamp_npz(config)` into
  every `np.savez`. Recover with `provenance.read(path)`. A number with no code version
  attached is not usable in a thesis.
- **Figures regenerate from saved artifacts** (`src/viz/plots.py`), never hand-made. Each
  carries its source file and git SHA in the caption.

- **RUNNING A COMMAND IS NOT VERIFYING IT — CHECK THAT IT SCORED YOUR DATA.** Several
  `analyze_*.py` scripts **accept a directory argument and ignore it**, or accept it and then
  filter to a hard-coded modulus list: `analyze_k04.py` hard-codes `results/k02_grid` /
  `results/k04_extended` and has **no argv handling at all**; `analyze_o4.py` takes a
  directory but hard-codes `MODULI = [113, 121, 125]`. Pointed at k09 they exit **0** and
  print a full, plausible report — about the wrong data, or about nothing
  (*"0/0 grokked seeds … NOT HELD"*, a **false failure on an empty set**). On 2026-09-15 I
  smoke-tested both before a push, saw exit 0, and wrote "every command has been executed
  against this kernel's own artifacts" — true, and insufficient. **Check that the output
  names YOUR moduli/seeds**, not just that the exit code is 0. Which take a directory:
  `analyze_n4`, `analyze_o4` (filtered), `analyze_excursions`, `analyze_omega`, `analyze_k05`,
  `analyze_k09`. Which do **not**: `analyze_k04`, `test_crt_law`.
- **A DOWNLOAD THAT REPORTS SUCCESS ON A TRUNCATED PULL.** `kaggle kernels output` returns
  **rc 0** on `Connection broken: IncompleteRead`, leaving a **0-byte** `.npz`; `run_kernel.py`
  used to print `pulled -> …` regardless. Fixed — it now treats rc≠0, `connection broken`, or
  **any zero-byte artifact** as truncated, deletes the zero-byte files, retries, and fails
  loudly. **Deleting them is what makes the retry work: the client SKIPS files that already
  exist**, so a 0-byte truncation is otherwise "already downloaded" forever. Corollary —
  **never `rm` the output directory between retries**; that is resumption, and my first
  ad-hoc loop destroyed a completed 29.6 MB file by doing it.

## Analysis invariants (earned, session 2)

- **Seed variance is enormous on grokking time** — n=119 spanned 6,600–24,800 steps across
  5 seeds, a factor of 3.8. **Never interpret a raw grokking-time ordering.** One
  single-seed "finding" (O2) died this way. Report φ(n)-normalised times alongside raw.
- **Render before concluding.** Two conclusions were nearly inverted by looking:
  the 2D DFT bin convention (`(k,k)` is `f(a+b)`, `(k,−k)` is `f(a−b)` — verify empirically,
  never by reasoning), and using mean |activation| where **variance** was the right block
  statistic (mean cannot separate computation from saturation at a constant).
- **`scripts/render_all.py` after every run.** Panels per checkpoint plus `figures/INDEX.md`.
  When a view cannot be produced it says **"NOT SAVED"** — a gap in what was saved is never
  to be read as absence of a phenomenon.
- **Save `mlp_acts`, `logits_all` and `attn` for EVERY seed**, not just seed 0. The entire
  neuron-level story currently rests on one seed because of this.
- **THERE ARE TWO BIN CONVENTIONS AND THE RULE BELOW IS ONLY ABOUT ONE OF THEM.**
  `energy()` **keeps DC at index 0**, so its index IS the frequency — do not add 1.
  `mult_amplitude()` **DROPS DC**, so its index 0 is frequency **1** — you MUST add 1
  before comparing it to anything that reports a real frequency (e.g.
  `neurons.freq_fraction`'s `best_k`). Mixing them made the neuron/embedding key overlap
  read **0/4 when it is 4/4** (2026-09-14). **Settle a bin convention by planting a pure
  signal of known frequency and looking at where it lands — never by reasoning.** Three
  bin-convention bugs so far; reasoning has lost every time.
- **AND A THIRD ARRAY WITH A THIRD RULE: `test_crt_law.predicted()` RETURNS FREQUENCIES,
  `freq_energy()` RETURNS BINS AT POSITION = FREQUENCY − 1. YOU MUST SUBTRACT 1.**
  `test_crt_law.main()` has always written `[k - 1 for k in predicted(n)]`. Passing them
  unshifted (2026-09-15, k09 P5) put every predicted frequency one bin off the key set and
  read **DEPLETION** — enrichment 0.31/0.55/0.30 at p ≈ 1.0 — on moduli where C8 is the
  best-supported claim in the project. **It fails LOUDLY at exactly one modulus** (n=120,
  `IndexError`, because `predicted` includes n/2) **and silently everywhere else.**
  **Fourth bin-convention bug. Caught by a POSITIVE CONTROL, never by reading**: the same
  path scored **0.25 at n=165** where the ledger says 3.3–17.4×; after the fix, **17.44 at
  n=165 and 3.27 at n=119**, matching published numbers exactly. **Before trusting any
  spectral statistic on a new modulus, run it on one where the answer is already published.**
- **`energy()` folds conjugates, so its index IS the frequency** (index 0 = DC). Do **not**
  add 1. A `+1` shifted every key set by one in N4 and made the restricted loss look
  catastrophic (2.4e+01) instead of 8× better — it would have read as the clock being
  falsified at n=113.
- **The multiplicative readout must not move when the GENERATOR moves.** Exponent
  coordinates need a choice of primitive root and `primitive_root(n)` silently returns the
  smallest. Replacing g by g^t (gcd(t,φ)=1) relabels the same group, so Gini, PR and the
  baseline/restricted/excluded losses are **exactly invariant**, and a key frequency moves
  by **k → t·k mod φ** — it does not stay put. `test_generator_equivariance.py` asserts all
  of it (and is what caught the control-mean defect above). Non-cyclic groups: the same,
  per cyclic component, via `unit_index(n, powers=[...])`.
- **Ablate in the coordinate where the task is addition.** Nanda's restricted set puts
  f(a+b) at bins (k,k); for `a*b mod n` that holds only after relabelling by discrete log.
  Non-cyclic unit groups have no discrete log — skip them loudly, never silently.
- **Prime powers are VACUOUS for the CRT-dual law** (one CRT component, n/q = 1, so the
  predicted set is every frequency). Never count them as support.
- **A SIZE CONFOUND HAS NOW KILLED THREE CLAIMS. Check for it first, every time.**
  C7b (𝒥-class *class size*), C20 (*block cell count*), C19 (*φ(n)*). The pattern is
  identical each time: the grouping variable is structurally coupled to a size, and the
  statistic reads the size.
  - **Blocks/classes:** `|J_d| = φ(n/d)`, so depth and class size are coupled.
    ρ(block cells, variance) = +0.856..+0.949, and a flat-variance **noise control** scores
    ρ = +0.942 (midranks, P20; the ordinal ranks read +0.85..+1.00 and +0.918). Use `src/analysis/jblocks.py`, which measures every block at an equal
    subsampled cell count and **drops** blocks that cannot supply it rather than reporting
    them as 0. `np.var` of a single cell is 0 by definition, not by learning.
  - **Normalisation is not neutral.** C19 divided grokking time by φ(n). Raw steps
    *decrease* with φ (ρ = −0.665), so dividing by φ over-corrects and manufactures an
    ordering: ρ(−φ, steps/φ) = **+0.870** over 15 moduli (midranks; P9a). k02's cyclic moduli were simply
    the three largest φ. **Before normalising by a quantity, check the correlation of the
    normalised statistic with that quantity** — and report raw and normalised together.
  - The cheap general defence is the one that caught all three: **run the statistic on a
    control that has the size structure but not the effect** (shuffled labels, flat
    variance, permuted grouping). If the control reproduces the result, there is no result.

## Engine memory: the `no_grad()` leak (earned 2026-09-12, one OOM and three misdiagnoses)

- **`_backward` is a GUARDED PROPERTY on `Tensor`. Do not turn it back into a plain slot.**
  `_child()` decides whether a tensor needs a graph; every operator used to overwrite that
  decision with `out._backward = bw`, and `bw` closes over `out`, so **every tensor
  self-referenced and refcounting could not free it**. Training escaped it because
  `backward()` clears the edges — **the eval path never calls `backward()`**, so each
  `with no_grad(): m(xte)` leaked its whole forward pass: ~176 MB at n=113, once per
  `LOG` steps ⇒ **~1.76 MB/step**. Guarded by `test_nograd_leak.py`. The fix is numerically
  inert (`test_model.py` unchanged at 2.68e-15) and cut per-run footprint **~6.75 GB → ~300
  MB**. Adding a new operator? It may assign `_backward` freely; the property ignores the
  assignment when `requires_grad` is False.
- **`test_memory.py` DID NOT CATCH THIS and cannot.** It runs **30 steps with no eval**,
  where the leak contributes ~0 MB against a 200 MB threshold, and reads `ru_maxrss`, a
  monotonic high-water mark that cannot fall. **The HORIZON, not the threshold, is what
  makes a memory test blind.** Same shape as the pre-flight probe that measured throughput
  over 20 steps and never measured memory.
- **A MEMORY PROBLEM THAT SCALES WITH PARALLELISM IS NOT NECESSARILY ABOUT PARALLELISM.**
  Two launches were "fixed" by lowering `N7_PAR` before anyone read the engine. Every
  footprint figure in this project's history — "2.2 GB/run", the 13.2 GB 6-way thrash, the
  17 GB that OOM-killed a 3-way run — was this leak. **Ask what grows per STEP before
  tuning how many processes run.**

## Measuring a long run (earned the same day, four wrong answers from one dataset)

- **SAMPLE AGAINST THE INDEPENDENT VARIABLE (step count), FOR 30+ MINUTES, BEFORE CALLING
  ANYTHING STEADY.** The same run supported all of: *"44 h not 21 h"* (a **step-0 ETA**;
  real rate 450 ms/step ⇒ 21 h) · *"12.6 GB, thrashing"* (**one sample taken during a
  snapshot**; 120 s later, 10.6 GB) · *"steady state, safe"* (a **105-second** flat window
  — swap was climbing too slowly to see and the kernel OOM-killed the run **50 minutes
  later**) · *"0.03 MB/step"* (measured where the rate genuinely was that low, then
  projected as constant; the truth was 1.76).
- **`step 0` in a log is not a stall.** `run_n7.py` prints on `step % 2000`, **not** on
  `LOG`. Check `ps -o stat=,time=,pcpu=` — `Rl` at ~170% CPU is healthy, `D` is IO wait.
- **Run long local jobs under `systemd-run --user` with a cgroup cap**, never `setsid
  nohup` (which dies with the terminal scope). `MemoryHigh`/`MemoryMax` turn a runaway into
  a contained kill instead of a **kernel global OOM that can take out the desktop session**.
  Read `memory.events` (`high`/`max`/`oom_kill`) and `memory.current`/`memory.swap.current`
  — `memory.max` bounds RAM only, swap is accounted separately.
- **An OOM in the journal: read `constraint=` and the victim.** `global_oom` means the
  whole box, not a cgroup limit. `systemd-oomd` being active does **not** mean it did it,
  and `nvidia-powerd invoked oom-killer` names the process whose allocation failed first,
  **not the cause**. Chasing either wastes an hour.
- **A `timeout` SHORTER THAN THE STAGE'S OWN INTERNAL BUDGET MANUFACTURES A FAILURE, AND IT
  LOOKS EXACTLY LIKE A REGRESSION.** `test_grok.py` calls `train(..., time_limit=600)`; an
  ad-hoc suite wrapper ran every check under `timeout 300` and it came back **FAIL** on
  2026-09-15. Nothing was broken — the file is untouched since `init`, and
  `scripts/reproduce.sh` runs these with no timeout at all, so the project's real entry
  point was never affected. **Before wrapping a check in a timeout, grep it for its own
  `time_limit` / `timeout` / step budget**, and set yours above it. Corollary, learned the
  same hour: a check re-run under **CPU contention you created** (three 4-thread training
  jobs on a 12-core box) can trip a timeout that is otherwise generous — so a FAIL under
  load is not a result until it has been re-run on an idle box. Same family as every other
  entry here: **a check whose failure mode is indistinguishable from its success mode — or
  from a real regression — is not a check.**
- **SMOKE-TEST AN UNATTENDED CHAIN BEFORE ARMING IT, AND PUT A `timeout` ON EVERY STAGE.**
  `analyze_n7.py` crashed on **every** call — `tail_gini` read `z["step"]`, which only the
  engine's npz carries while E5 always pairs against k03's — and `main()` evaluates
  `e1..e5` in one tuple, so it took the whole analysis down. Found in minutes on retired
  data; it would have fired at 23:00 against a finished 21-hour sweep.
- **NEVER GLOB OVER A DIRECTORY THAT ALSO HOLDS ARCHIVES.** `logs/n7_n*_s*.log` matched
  `n7_n113_s0_preleakfix.log` — `*` after `_s` swallows `0_preleakfix` — and a status check
  reported a **retired run's groks as current**. Archives go in a subdirectory.

## Running local compute (earned 2026-09-11, two failed launches)

- **MEASURE MEMORY OVER TIME BEFORE COMMITTING TO A LONG PARALLEL RUN, NOT JUST SPEED.**
  N7's pre-flight probe measured throughput at 6-way over **20 steps** and never measured
  memory. Steady state turned out to be **~2.2 GB/run and ~2.7 GB at n=125**, so six runs
  reach a **13.2 GB footprint on a 14.8 GB box**, push 4.6 GB into swap and thrash. Twenty
  steps is nowhere near steady state, and the throughput figure measured there is also
  wrong as a planning number because it predates the swapping regime.
- **RSS ALONE LIES UNDER MEMORY PRESSURE.** RSS *plateaus* while the real footprint keeps
  growing, because the kernel pages the excess out. Reading RSS alone produced two wrong
  conclusions in ten minutes — first "unbounded leak", then "safely plateaued". The honest
  number is **RSS + `VmSwap`**:
  ```bash
  awk '/^VmRSS|^VmSwap/{printf "%s ", $2}' /proc/<pid>/status
  ```
  Also watch `free -m` **Swap used** — swap climbing while RSS is flat *is* the symptom.
- **A memory guard is only as good as its constant.** `run_n7_all.sh` let an impossible
  configuration start because it assumed 800 MB/run, a figure lifted from Gate 1's
  *single-process* run and never re-checked. Size guards on a **measured** worst case.
- **NEVER MATCH PROCESSES ON A SUBSTRING OF THE FULL COMMAND LINE.** `pgrep -f "run_n7.py"`
  and `awk '$0 ~ /run_n7/'` match **the matching command's own argv**. This killed the
  the shell running it three times (exit 144), made two `until ! pgrep ...` waiters spin
  forever because the condition could never become true, and made a watchdog report
  "9 alive (expected 6)". Match on **fields**:
  ```bash
  ps -eo pid=,args= | awk '$2==".venv/bin/python"{for(i=3;i<=NF;i++)if($i=="run_n7.py"){print $1;break}}'
  ```
- **BUT A FIXED FIELD INDEX IS ITS OWN BUG — AN INTERPRETER FLAG SHIFTS EVERY FIELD.**
  The old form above hard-coded `$4=="run_n7.py"`, which is only right because the launcher
  passes exactly one flag (`python -u run_n7.py`); it is one `-X` away from silently
  matching nothing. Two instruments had the shifted variant and **both reported the box
  idle while a 20 h sweep was 10.5 h in** (2026-09-12): `STATE.md`'s documented check tested
  `$3`/`$4` of `pid=,sess=,args=` where `$4` is `-u`, and `session_brief.py`'s regex was
  `\.venv/bin/python\s+(run_|…)` with no room for a flag, so §3 printed **"no local jobs
  running"**. A false *idle* is the worst direction for this bug to fail in — the next
  session reads it and relaunches. **Scan the fields for the script name; never index to
  it, and never trust a liveness check that has only ever been run against a live job.**
- **`RemainAfterExit=yes` MAKES `is-active` A CONDITION THAT CAN NEVER BECOME FALSE.** A
  systemd unit launched with it stays **`active`** forever after its script exits — only
  `SubState` moves to `exited`. `n7_finish.sh` waited on `is-active != active`, so when the
  N7 sweep finished all 12 runs the waiter polled for **21 h 42 min** (2.5 s CPU over 21 h
  wall) until a logout killed both units, and **the analysis never ran — two days lost with
  the results sitting on disk**. This was the **third** waiter in this project to spin on an
  impossible condition. Check it in five seconds before trusting one:
  ```bash
  systemd-run --user --unit=t --property=RemainAfterExit=yes /bin/true
  systemctl --user is-active t                      # active   <- the trap
  systemctl --user show t -p SubState --value       # exited   <- what to wait on
  ```
  **Wait on a disjunction that includes a condition you can prove true** — SubState leaves
  "running", OR the unit stops, OR the work's own completion marker appears (e.g. 12 logs
  saying `done.`). Never a single systemd predicate.
- **A watchdog tailing a log you later truncate is mute.** Do not clear logs a monitor is
  following, and make the filter cover failure signatures, not just success.
- **Local background jobs die with the session.** `setsid nohup ... & disown` still sits in
  the terminal's systemd scope (`ptyxis-spawn-*.scope`); when that scope goes, so does the
  job. A long unattended local run needs either a live session or `systemd-run --user`
  (and note `Linger=no` means user units die at full logout anyway).

## Working style

Surface assumptions, build the simplest thing that answers the question, keep
diffs surgical, and define verifiable success criteria before the run. One
runnable check per non-trivial module (`test_analysis.py`,
`algebra.py::_selfcheck`).

## More gotchas (earned)

- **Local smoke tests need `train_frac >= 0.8` at small n.** 30% works at n=113 but is far
  below critical dataset size at n=17 — a correct implementation will look broken.
- **The from-scratch engine is verified against PyTorch to ~1e-15** (`test_model.py`).
  If it disagrees with PyTorch, suspect the science, not the engine — but re-run
  `test_autograd.py && test_model.py` first.
- **Softmax Collapse is real here** (H1.4 confirmed): |grad| drops 5.5 orders while
  p_correct hits exactly 1.0. `stablemax_cross_entropy` holds ~10x more gradient.

## Engine performance and safety (earned the hard way)

- **449 ms/step** at n=113/d=128 after two fixes. If it regresses, check: (1) matmul
  backward must not materialise a per-batch gradient before summing — fold the batch into
  the row axis when the second operand is unbatched; (2) the MLP runs only at the read-out
  position.
- **`engine.no_grad()` around every eval and checkpoint.** Forward passes over the full
  grid are 3.3× the training batch and need no graph.
- **`backward()` frees the graph** (clears `_prev`/`_backward`). Without it the closures
  form reference cycles and RSS climbs to 12 GB in 1000 steps. **`backward()` may be
  called once per graph.** `test_memory.py` guards this.
- **Define acceptance criteria BEFORE the run, and implement them against the paper's
  stated method.** Gate 1's C6/C7 had to be edited after failing — defensible (they did not
  match Nanda's stated criterion) but the wrong ordering. See LAB_NOTEBOOK Entry 12.
