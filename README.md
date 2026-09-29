# A Clock on Every Stratum — code and lab record

Reproduction material for *A Clock on Every Stratum: Character Circuits for Modular
Multiplication at Non-Square-Free Moduli*. This repository exists for one purpose: so that
every number in the paper can be checked by someone who did not run the experiments.

It carries no authorship metadata. The paper that links here identifies its authors.

## What is here

| path | what it is |
|---|---|
| `src/` | the from-scratch autograd engine, the transformer, and the analysis toolkit |
| `kernels/` | the GPU kernels that produced each sweep, and the push/poll/pull harness |
| `experiments/` | a pre-registration per experiment: criteria, predictions, falsifiers, outcome |
| `analyze_*.py`, `test_*.py` | the analyses, and the checks that guard them |
| `paper/` | the manuscript sources |
| `FINDINGS.md` | every result, organised by topic — the scientific record |
| `LAB_NOTEBOOK.md` | the dated experimental record, append-only |
| `LAB_PROTOCOL.md` | the conventions and the accumulated gotchas the work runs under |

## Reproducing the numbers

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
scripts/fetch_artifacts.sh      # ~4.8 GB from the Zenodo deposit (forthcoming), verified against results/MD5SUMS
scripts/reproduce.sh            # every self-check, then every claim-carrying analysis
```

`test_paper_numbers.py` is the one to read first. It re-derives every load-bearing number in
the manuscript — captions included — from the saved artifacts rather than from the prose,
which is the only arrangement under which agreement means anything.

Training runs are reproducible but not cheap: a single engine run is roughly five CPU-hours,
and the GPU sweeps were run on free-tier Kaggle T4s. The deposit exists so that nobody has to
repeat them to check the analysis.

## What this repository does not give you

**Pre-registration timestamps are not verifiable here.** Each `experiments/PREREGISTER_*.md`
states its own date, and its Outcome section is visibly separate from the criteria above it —
but this repository's history begins at a single import commit, so you cannot confirm from it that the
criteria predated the run. In the private development repository they are separate commits and
they do. We say this rather than let the structure imply a guarantee it cannot give.

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
read `account-a` in the lab notebook and find a different string inside an `.npz`, that is the
expected state and not a discrepancy: the prose is relabelled, the evidence is not.

## Licence

Code: MIT (`LICENSE`). Manuscript sources under `paper/` and all figures: CC BY 4.0.
The artifact deposit: CC BY 4.0. It is not yet published; its DOI will be added here and to `scripts/fetch_artifacts.sh` when it is.
