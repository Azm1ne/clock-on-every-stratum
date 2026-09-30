# A Clock on Every Stratum: code, checks and proofs

This repository contains the code, the checks and the Lean 4 proofs for the paper *A Clock on
Every Stratum: Character Circuits for Modular Multiplication at Non-Square-Free Moduli*. Its
purpose is to let a reader check each number in the paper without running the experiments
again.

The repository contains no author information.
The paper identifies its authors.

## How to use this repository

First, install the dependencies:

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Then use one of two paths.

### Path A: train the main run from the start

```bash
scripts/train_headline.sh       # n=113, seed 0, 40,000 steps, on this project's autograd engine
```

This script runs `run_n7.py` with the arguments of the run in the deposit. It does not change
`run_n7.py`. With one process and 2 BLAS threads, one step takes 0.42 s. The full run takes
approximately 5 hours (approximately 8 CPU-hours) and uses approximately 450 MB of memory.

Training is deterministic and uses the full batch. If the deposit is present, the script
compares its result with the run in the deposit. The two results must be identical, bit for bit.

To get the command for each of the other sweeps and seeds, run:

```bash
scripts/sweep_table.py --commands
```

The output gives one command for each sweep, and the commit that produced the sweep. The GPU
sweeps in `kernels/` ran on free Kaggle T4 GPUs. To run one of them again, use
`kernels/run_kernel.py`. The engine sweeps run on a CPU.

### Path B: calculate each number, table and figure again from the deposit

```bash
scripts/fetch_artifacts.sh      # ~4.8 GB from the Zenodo deposit (not published yet), verified against results/MD5SUMS
scripts/reproduce.sh            # all self-checks, then each analysis that supports a claim, then the figures
```

Read `test_paper_numbers.py` first. It calculates each important number in the paper, including
the numbers in the captions, from the saved artifacts. It does not copy the numbers from the
text. A match therefore shows that the artifacts support the number, and not only that two
texts agree.

Training takes a long time. With the deposit, you can check the analysis without training the
models again.

## Contents

| Path | Contents |
|---|---|
| `src/` | The autograd engine and the transformer, both written for this project, and the analysis tools |
| `run_*.py` | The local training and measurement runs: the engine sweeps, Gate 1 and the interventions |
| `kernels/` | The GPU kernels that produced the sweeps, and the script that sends, monitors and downloads them |
| `analyze_*.py`, `test_*.py` | The analyses, and the checks for them |
| `experiments/` | The pre-registrations: criteria, predictions, falsification conditions and outcome. `MANIFEST.md` gives the date of each |
| `paper/` | The source files of the paper |
| `lean/` | The mathematics of the paper in Lean 4. `lean/README.md` lists each statement |

Each script in this repository has at least one of these properties: `scripts/reproduce.sh`
runs it, the paper cites it, `test_paper_numbers.py` checks its output, it is part of a
training path, or one of these scripts imports it.

## What this repository does not contain, and why

### Laboratory tools and working records

This repository does not contain the status tools, the one-time comparison scripts, the
orchestration scripts, the submission build scripts, the laboratory record (the findings, the
notebook and the protocol), the session notes or the planning
documents.
Neither path needs them, and they produce no number in the paper.

### Proof that the pre-registrations came before the runs

`experiments/MANIFEST.md` gives the commit and the date at which each pre-registration was
first committed. In each file, the Outcome section is separate from the criteria above it.
But the history of this repository starts at one import commit, and the development repository
is private. Thus you cannot examine those commits to confirm that the criteria were written
before the run. The manifest does not give that guarantee.

### The commits recorded in the artifacts

Each saved artifact records the commit that produced it (`git_sha`). These commits are in the
development history, which this repository does not contain. A stamp shows which artifacts came
from the same code, but you cannot check out its commit. Instead, run `scripts/reproduce.sh`.
It calculates each published number again from the deposited artifacts with the code in this
repository.

The paper also records a related gap. 68 artifacts from the three earliest sweeps have no
usable commit stamp. An argument, not a stamp, identifies their code version.
`scripts/reconstruct_provenance.py` makes this argument and checks it.

### The names of the compute accounts

Each saved artifact has a provenance stamp. In 68 artifacts, the field `kaggle_account` gives
the name of the free Kaggle account that supplied the GPU time. The text in this repository
uses the neutral labels `account-a` and `account-b` instead of these names. We did not change
the stamps to agree with the labels, because a provenance record that is changed after it is
written is not reliable. If the text here says `account-a` and an `.npz` file contains
a different name, this is correct: only the text uses the labels, and the artifacts keep the
original names.

## Licence

The code has the MIT licence (`LICENSE`). The paper sources in `paper/` and all figures have
the CC BY 4.0 licence.
The artifact deposit has the CC BY 4.0 licence. It is not published yet. Its
DOI will be added here and in `scripts/fetch_artifacts.sh` when it is published.
