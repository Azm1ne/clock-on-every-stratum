"""O21 — is the engine secretly computing in float64 when ENGINE_DTYPE=float32?

Pre-registered in experiments/PREREGISTER_o21_dtype.md.

Tensor.__init__ casts .data to DTYPE on construction, so an intermediate that numpy
promoted to float64 is computed wide and then rounded down -- more accurate than a true
float32 computation, and invisible to any check that inspects a finished Tensor. That is
why test_autograd.py::test_dtype_flag passes while O21 stays open: it inspects parameters,
parameter gradients, Adam state and the loss, all of which are downcast before it looks.

This probe reads the dtype BEFORE the cast, at every operator site, for one real step.

Usage:  ENGINE_DTYPE is set from argv, so do NOT export it.
    .venv/bin/python run_o21_probe.py float32
    .venv/bin/python run_o21_probe.py float64      # criterion 3, the positive control
    .venv/bin/python run_o21_probe.py --selfcheck  # criterion 4, planted promotion
"""
import os
import sys

WANT = "float32"
for a in sys.argv[1:]:
    if a in ("float32", "float64"):
        WANT = a
os.environ["ENGINE_DTYPE"] = WANT          # must precede the engine import

import numpy as np                                                   # noqa: E402
from src.autograd import engine as E                                 # noqa: E402
from src.autograd.nn import AdamW, cross_entropy                     # noqa: E402
from src.model.transformer import Transformer                        # noqa: E402
from src.train.loop import modular_data                              # noqa: E402

FWD, BWD = [], []


def instrument():
    """Record the PRE-CAST dtype at every operator site. Returns an undo callable."""
    orig_child, orig_accum = E.Tensor._child, E.Tensor._accum

    def child(self, data, parents, backward):
        d = np.asarray(data)
        f = sys._getframe(1)
        FWD.append((f.f_code.co_name, f.f_lineno, d.dtype.name,
                    tuple(p.data.dtype.name for p in parents)))
        return orig_child(self, data, parents, backward)

    def accum(self, g):
        f = sys._getframe(1)
        BWD.append((f.f_code.co_name, f.f_lineno, np.asarray(g).dtype.name, ()))
        return orig_accum(self, g)

    E.Tensor._child, E.Tensor._accum = child, accum

    def undo():
        E.Tensor._child, E.Tensor._accum = orig_child, orig_accum
    return undo


def sites(records, bad_only, want):
    """Collapse per-call records to distinct (function, line, dtype) sites."""
    seen = {}
    for name, line, dt, ops in records:
        if bad_only and dt == want:
            continue
        seen.setdefault((name, line, dt, ops), 0)
        seen[(name, line, dt, ops)] += 1
    return seen


def report(tag, records, want):
    total = len(records)
    bad = sites(records, True, want)
    n_bad_calls = sum(bad.values())
    print(f"\n{tag}: {total} calls, {len(sites(records, False, want))} distinct sites, "
          f"{n_bad_calls} calls NOT {want} across {len(bad)} sites")
    for (name, line, dt, ops), k in sorted(bad.items(), key=lambda kv: -kv[1]):
        print(f"    {dt:>8}  x{k:<6} engine.py:{line} in {name}()   operands={ops}")
    return len(bad), total, n_bad_calls


def _selfcheck():
    """Criterion 4: a planted promotion MUST be detected, or the probe is blind."""
    assert WANT == "float32", "run the selfcheck at float32"
    undo = instrument()
    try:
        FWD.clear()
        x = E.Tensor(np.ones((4, 4)), requires_grad=True)
        # int64 array operand: float32 * int64 -> float64 under NEP 50. This is the exact
        # shape of the bug the max() backward already carries a comment about.
        planted = x * E.Tensor(np.ones((4, 4)))       # control: must stay float32
        clean_bad = len(sites(FWD, True, "float32"))
        FWD.clear()
        _ = (x.data * np.ones((4, 4), dtype=np.int64))
        forced = E.Tensor._child(x, x.data * np.ones((4, 4), dtype=np.int64), (x,),
                                 lambda: None)
        planted_bad = len(sites(FWD, True, "float32"))
    finally:
        undo()
    assert clean_bad == 0, f"clean float32 op flagged {clean_bad} sites -- false positive"
    assert planted_bad >= 1, "planted float64 promotion NOT detected -- probe is blind"
    assert forced.data.dtype.name == "float32", "constructor should still cast down"
    print(f"selfcheck OK: clean={clean_bad} flagged, planted={planted_bad} flagged "
          f"(detected), constructor still casts to {forced.data.dtype.name}")


def main():
    n, seed = 113, 0
    xtr, ytr, _, _ = modular_data(n, "mul", train_frac=0.30, seed=seed)
    m = Transformer(n_vocab=n + 1, n_out=n, seed=seed)
    opt = AdamW(m.parameters(), lr=1e-3, weight_decay=1.0)

    undo = instrument()
    try:
        opt.zero_grad()
        logits = m(xtr)
        loss = cross_entropy(logits, ytr)
        loss.backward()
        opt.step()
    finally:
        undo()

    print(f"=== O21 dtype probe · ENGINE_DTYPE={WANT} · n={n} seed={seed} · one step ===")
    print(f"engine DTYPE  : {E.DTYPE}")
    print(f"loss          : {float(loss.data):.6f}  dtype {loss.data.dtype.name}")
    nf, tf, cf = report("FORWARD (_child, pre-cast)", FWD, WANT)
    nb, tb, cb = report("BACKWARD (_accum gradients)", BWD, WANT)

    # Criterion 3: at float64 essentially every site must READ float64, else the
    # instrument cannot see dtypes at all and criteria 1-2 are void.
    frac_wide = 1.0 - (cf + cb) / max(tf + tb, 1)
    print(f"\nfraction of calls at {WANT}: {frac_wide:.4f}  ({tf + tb} calls total)")

    print("\n--- verdict ---")
    if WANT == "float64":
        ok = frac_wide >= 0.90
        print(f"CRITERION 3 (positive control): {'PASS' if ok else 'FAIL'} "
              f"-- {frac_wide:.2%} of calls are float64, threshold 90%")
        if not ok:
            print("  ** instrument cannot read dtypes; criteria 1-2 are VOID **")
    else:
        print(f"CRITERION 1 (forward promotion) : {nf} float64 sites -> "
              f"{'CONFIRMED' if nf else 'NOT FOUND'}")
        print(f"CRITERION 2 (backward promotion): {nb} float64 sites -> "
              f"{'CONFIRMED' if nb else 'NOT FOUND'}")
        if nf == 0 and nb == 0:
            print("  => candidate (c), residual NEP-50 promotion, is DEAD.")
            print("     The engine computes in float32 end to end. The residue is")
            print("     (a) accumulation order or (b) softmax formulation.")
        else:
            print("  => the engine is NOT computing at ENGINE_DTYPE. O21 explained.")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()
