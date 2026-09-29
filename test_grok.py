"""End-to-end: does the from-scratch engine actually grok? The Lane B acceptance test."""
import numpy as np
from src.model.transformer import Transformer
from src.train.loop import modular_data, train

# train_frac 0.8, NOT 0.4. At n=17, 0.4 is below critical dataset size (C14: 0.4 -> 0.144)
# and a CORRECT engine looks broken there -- which is what this file used to assert nothing
# about. C11 is stated at train_frac 0.8, test acc 1.000 by step 500, so that is what the
# acceptance test must run.
N, TRAIN_FRAC, THRESHOLD = 17, 0.8, 0.9
xtr, ytr, xte, yte = modular_data(N, "mul", train_frac=TRAIN_FRAC, seed=0)
print(f"n={N} mul: {len(ytr)} train / {len(yte)} test pairs")
m = Transformer(N + 1, N, d_model=64, n_heads=4, d_head=16, d_mlp=256, seed=0)
print(f"params: {sum(p.data.size for p in m.parameters()):,}")
h = train(m, xtr, ytr, xte, yte, steps=4000, log_every=250, time_limit=1800)
best = max(r["test_acc"] for r in h)
print(f"\nbest test acc {best:.3f}  final {h[-1]['test_acc']:.3f}")

# Two outcomes fail this check:
#   1. the time budget was exhausted, so `best` is only a lower bound, not a verdict;
#   2. the engine ran to completion and did not grok.
# (1) is separate because CPU contention can end the run early (for example at step 2703
# of 4000 with other jobs on the machine), and a truncated run is not a result.
ran_out = h[-1]["step"] < 4000
assert not ran_out or best > THRESHOLD, (
    f"TRUNCATED at step {h[-1]['step']}/4000 without grokking (best {best:.3f}). "
    "Not a verdict -- re-run on an idle box before believing it.")
assert best > THRESHOLD, (
    f"C11 REGRESSION: engine did not grok at n={N}, train_frac={TRAIN_FRAC} "
    f"(best test acc {best:.3f}, threshold {THRESHOLD})")
print(f"GROKKED -- C11 holds (best {best:.3f} > {THRESHOLD}, step {h[-1]['step']})")
