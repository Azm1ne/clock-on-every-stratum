# N7, pre-leak-fix partial runs — SUPERSEDED, kept as evidence only

These are the partial artifacts from N7's first two launch attempts on 2026-09-11/12,
before the `no_grad()` tensor leak was found and fixed (`src/autograd/engine.py`, the
`_backward` property; regression test `test_nograd_leak.py`).

**Do not analyse these and do not pool them with the real N7 run.** They are here because
they contain real observations that cost GPU-free compute to obtain, not because they are
usable results.

## What they contain

| run | reached | grok step |
|---|---|---|
| `engine_n113_s0` | step 22,500 | **4,500** |
| `engine_n121_s0` | step 20,000 | **10,700** |
| `engine_n125_s0` | step 7,500 | not yet |

Both groks are genuine and were observed at test accuracy 1.000. They are **the first
grokking ever produced by the from-scratch engine on `a*b mod n`** (Gate 1 was `a+b`).

## Why they are superseded, given the fix changes no numbers

The leak fix is **numerically inert** — it only stops `no_grad()` tensors from
self-referencing. `test_model.py` reports the same worst relative gradient error
(2.68e-15) before and after, and `test_autograd.py` is unchanged. So the numbers in these
files are not *wrong*.

They are unusable for a different reason: **provenance**. These steps ran under the
pre-fix engine. Resuming them would stamp the post-fix SHA onto an artifact whose 22,500
steps came from different code, and the reproducibility decree requires the stamped SHA to
identify the code that actually ran. A clean restart gives all 12 runs one SHA. That is
the only reason these were retired, and it is worth stating plainly so nobody later
"recovers" them thinking the retirement was about correctness.

## The measurement that justified the restart

Resuming `engine_n113_s0` from its step-22,500 checkpoint, instrumented, over 1,400 steps:

| | before fix | after fix |
|---|---|---|
| RSS | 480 → 2,947 MB | 280 → 314 MB |
| live `Tensor` objects | 64 → 316 | 13 → 13 |
| closure cells | 615 → 1,308 | 478 → 478 |

~1.76 MB/step leaked, entirely from the eval path: ~176 MB per `with no_grad(): m(xte)`,
once every `LOG = 100` steps.
