# A Clock on Every Stratum — code, checks and proofs

Reproduction material for *A Clock on Every Stratum: Character Circuits for Modular
Multiplication at Non-Square-Free Moduli*. This repository exists for one purpose: so that
every number in the paper can be checked by someone who did not run the experiments.

It carries no authorship metadata. The paper that links here identifies its authors.

## Two ways in

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
```

**(A) From scratch: train the headline run yourself.**

```bash
scripts/train_headline.sh       # n=113, seed 0, 40,000 steps, on the from-scratch engine
```

It calls `run_n7.py`, the script that produced the deposited run, with that run's arguments.
Measured: 0.42 s per step on one process with 2 BLAS threads, so about 5 hours wall clock
(about 8 CPU-hours), 450 MB peak memory. Training is deterministic and full-batch, so if the
deposit is present the script compares its result against the deposited run and requires a
bit-identical match. Every other sweep and seed is one command each, printed with the commit
that produced it by:

```bash
scripts/sweep_table.py --commands
```

The GPU sweeps (`kernels/`) ran on free-tier Kaggle T4s and are relaunched through
`kernels/run_kernel.py`; the engine sweeps run on a CPU.

**(B) From the deposit: regenerate every number, table and figure.**

```bash
scripts/fetch_artifacts.sh      # ~4.8 GB from the Zenodo deposit (forthcoming), verified against results/MD5SUMS
scripts/reproduce.sh            # every self-check, then every claim-carrying analysis and figure
```

`test_paper_numbers.py` is the one to read first. It re-derives every load-bearing number in
the manuscript — captions included — from the saved artifacts rather than from the prose,
which is the only arrangement under which agreement means anything.

Training runs are reproducible but not cheap, which is why the deposit exists: nobody has to
repeat the sweeps to check the analysis.

## What is here

| path | what it is |
|---|---|
| `src/` | the from-scratch autograd engine, the transformer, and the analysis toolkit |
| `run_*.py` | the local training and measurement runs (the engine sweeps, Gate 1, the interventions) |
| `kernels/` | the GPU kernels that produced each sweep, and the push/poll/pull harness |
| `analyze_*.py`, `test_*.py` | the analyses, and the checks that guard them |
| `experiments/` | the pre-registrations: criteria, predictions, falsifiers, outcome; `MANIFEST.md` dates each |
| `paper/` | the manuscript sources |
| `lean/` | the paper's mathematics in Lean 4, indexed statement by statement in `lean/README.md` |

Every script here is run by `scripts/reproduce.sh`, cited by the paper, pinned by
`test_paper_numbers.py`, part of a training path, or imported by one of those.

## What this repository does not give you, and why

**Lab tooling and the working record.** Status pollers, one-off diffs, orchestration and
submission-build scripts, the lab record (findings ledger, notebook, protocol) and the planning documents are
not here: none of
them is needed by either path above, and none produces a number in the paper.

**Pre-registration timestamps are not verifiable here.** `experiments/MANIFEST.md` gives the
commit and date at which each pre-registration was first committed, and each file's Outcome
section is visibly separate from the criteria above it. But this repository's history begins
at a single import commit and the development repository is private, so you cannot check out
those commits to confirm that the criteria predated the run. We say this rather than let the
manifest imply a guarantee it cannot give.

**Artifact `git_sha` stamps do not resolve here.** Every saved artifact records the commit
that produced it, and those commits belong to the development history, which this repository
does not carry. The stamps remain useful as identifiers — they tell you which artifacts came
from the same code — but you cannot check one out. What you can do instead is run
`scripts/reproduce.sh`, which recomputes every published number from the deposited artifacts
using the code in front of you. A related gap is documented in the paper: 68 artifacts from the three earliest sweeps carry no usable SHA at all, and their code version is recovered by argument rather than by stamp — see `scripts/reconstruct_provenance.py`, which makes and checks that argument.

**The deposited artifacts name the compute accounts; this repository does not.** Every saved
artifact carries a provenance stamp, and 68 of them record a `kaggle_account` field naming the
free-tier account whose GPU quota produced them. Those stamps are left exactly as they were
written. They could have been rewritten to match the neutral `account-a` / `account-b` labels
used in the prose here — and deliberately were not, because a provenance record edited after
the fact guarantees nothing, which is the whole of what a provenance record is for. So if you
read `account-a` in this repository and find a different string inside an `.npz`, that is the
expected state and not a discrepancy: the prose is relabelled, the evidence is not.

## Licence

Code: MIT (`LICENSE`). Manuscript sources under `paper/` and all figures: CC BY 4.0.
The artifact deposit: CC BY 4.0. It is not yet published; its DOI will be added here and to `scripts/fetch_artifacts.sh` when it is.
