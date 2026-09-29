# Pre-registration: gate1-repro — recover Gate 1's code version by reproduction

**Commit this file BEFORE launching the run.**

**Date:** 2026-09-26 · **Kernel/script:** `run_gate1.py` at
HEAD, unmodified, in a clean git worktree · **Arm:** replication (provenance)

## Why

`results/gate1/add113_{atgrok,final}.npz` carry **no provenance stamp** and §10 of the paper
quotes them: grok at step 6,900, five key frequencies, 7.3× loss improvement from ablating
non-key frequencies, 85.0% single-frequency neurons (vs Nanda's 84.6%). The run went
2026-09-10 ~20:20 → 23:26 +0600; the repo's `init` commit is 23:23. **It ran before the
repository existed, locally, so neither git nor a platform record certifies what ran.**
(LAB_PROTOCOL.md's "gate1 … carry no claim" is wrong: §10 carries four numbers from it.)

## Hypothesis

The engine is deterministic and full-batch, and at HEAD it has already reproduced a
published configuration bit-identically (C39's retrain, `max|ΔW_E| = 0` on 3/3 seeds). So
HEAD's `run_gate1.py` reproduces the saved Gate 1 artifacts exactly.

## Prediction

`max|ΔW_E| = max|ΔW_U| = max|Δlogits_all| = max|Δmlp_acts| = 0.0` at both snapshots, and the
same grok step, 6,900.

## Success criteria

| # | criterion | threshold | implemented in |
|---|---|---|---|
| G1R-1 | final snapshot `W_E`, `W_U`, `logits_all`, `mlp_acts` identical to `results/gate1/add113_final.npz` | max abs diff **exactly 0** | comparison in the Outcome, run as a one-off `np.load` diff over the listed keys |
| G1R-2 | at-grok snapshot identical likewise, and its `step` equal | exactly 0; equal | same |
| G1R-3 | new artifacts stamp a real `git_sha` with `git_dirty=False` | both | `provenance.read` |

Exact zero is the threshold because the claim is identity, not agreement; a float
tolerance would turn "the same code" into "similar code".

## What would falsify this

Any nonzero difference. Then the saved artifacts' code version stays **unrecoverable**, and
§10's four numbers are re-derived from the new stamped run and replace the old ones in the
paper — reported whether they move or not.

## Known confounds

The engine changed after 2026-09-10 (leak fix — numerically inert per `test_model.py`; the
float32 NEP-50 fix — this run is float64; the matmul-backward optimisation, whose timing
the 188-minute log already reflects). A mismatch would therefore not by itself say *which*
change moved it; that is not this experiment's question.

## Analysis plan

1. Diff the listed keys, both snapshots. 2. `provenance.read` the new files. 3. If G1R-1/2
hold, copy the stamped artifacts over the unstamped ones (keeping the unstamped originals in
`results/_archive_gate1/`) and state "recovered by bit-identical reproduction" in §12.
4. If they fail, recompute §10's numbers from the new artifacts with the scripts that
produced them and update §10/§12.

## Outcome (filled in AFTER the run — never edit anything above)

- **Result (2026-09-27, run 2026-09-26 02:23 → 05:35 +0600, 3.20 h local CPU, zero quota,
  `systemd-run --user --unit=gate1-repro`, clean worktree at `1eaf0e3`):** `done.
  grok_step=6900`. **Bit-identical at both snapshots**: `W_E`, `W_U`, `logits_all`,
  `mlp_acts`, `y_all`, `step`, `hist` all max|diff| = **0.0** (atgrok step 6,900 = 6,900;
  final 25,000). All **54 of 54** logged lines match the 2026-09-10 `run.log` once the
  wall-clock fields (`[Nm, eta Xh]`, total hours) are stripped, and the
  rewritten `history.json` is byte-identical to the tracked one.
- **Criteria met: G1R-1 ✅ G1R-2 ✅ G1R-3 ✅** — new artifacts stamp `1eaf0e3`,
  `git_dirty=False`. Gate 1's code version is **recovered by reproduction**. Per the
  analysis plan, the unstamped originals are in `results/_archive_gate1/`
  (`*.unstamped_20260910.npz`, md5 `b1b6dbdc…` atgrok / `ff7debeb…` final) and the stamped
  files replace them in `results/gate1/`. §12 says so; `test_paper_numbers.py` asserts
  both stamps.
- **Deviations from this pre-registration, and why:** one. `run_gate1.py` rewrites the
  **tracked** `results/gate1/history.json` every 500 steps, so from step 500 the worktree
  read dirty and the rolling step-2,500 snapshot stamped `git_dirty=True` — the script
  dirtying the tree it describes (LAB_PROTOCOL.md's `open(out,"w")` family), not a code change.
  At ~step 2,600, before the step-5,000 snapshot and the step-6,900 at-grok one, that one
  output file was marked `git update-index --skip-worktree` **inside the isolated
  worktree only**; `git diff --name-only HEAD -- src run_gate1.py` was 0 files at the time.
  The step-2,500 snapshot was overwritten at 5,000. The skip hid no content change: the
  final `history.json` is byte-identical to the committed one. No criterion was touched.
