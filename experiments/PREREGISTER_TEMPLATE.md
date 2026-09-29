# Pre-registration: <experiment id>

**Commit this file BEFORE launching the run.** Git's timestamp is the evidence that the
criteria predated the result — which is the thing a reviewer will ask about, and the thing
we got wrong once already (Gate 1 C6/C7, LAB_NOTEBOOK Entry 12).

**Date:** · **Kernel/script:** · **Arm:** replication | thesis

## Hypothesis

One sentence. What do we believe, and about what.

## Prediction

The specific, quantitative thing we expect to observe. Numbers where possible, with the
source of the reference number if there is one.

## Success criteria — exact, and implemented as code before the run

| # | criterion | threshold | implemented in |
|---|---|---|---|
| 1 | | | |

Thresholds are chosen **now** and justified — from a published number, from a control, or
from a pre-computed noise floor. "We'll see what looks reasonable" is not a threshold.

## What would falsify this

Be concrete. If no observation could falsify it, it is not a hypothesis yet.

## Required controls

| claim shape | control |
|---|---|
| "sparse in basis X" | ≥2 alternative bases including a random orthogonal one |
| "component C is responsible" | ablate C **and** a random equal-sized set |
| "the model encodes P" | probe baseline on raw input features, not activations |
| "X groks faster than Y" | ≥3 seeds; report variance |

## Known confounds

What could produce this result without the hypothesis being true? (Class size, dataset
fraction, φ(n), unit-fraction differences across moduli, seed.)

## Analysis plan — decided now

Which script, which statistic, which comparison. **Any analysis not listed here is
exploratory and must be labelled as such in the write-up.**

## Outcome (filled in AFTER the run — never edit anything above)

- Result:
- Criteria met:
- **Deviations from this pre-registration, and why:**
