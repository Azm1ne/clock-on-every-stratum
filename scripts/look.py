#!/usr/bin/env python
"""Show EVERYTHING about one checkpoint: every panel, every number, one command.

    PYTHONPATH=. .venv/bin/python scripts/look.py <checkpoint.npz> <n>

Built for pattern-spotting by eye. Scalars say a structure exists; the panels say what it
is. Every mechanism in this literature -- the Clock, the Pizza, the Discrete-Log Clock,
J-class stratification -- was found by looking at one of these views.
"""
import sys
from math import gcd
import numpy as np

sys.path.insert(0, ".")
from src.tasks.algebra import describe
from src.analysis.sparsity import gini, participation_ratio, key_freqs_5x_median
from src.analysis.transforms import unit_index
from src.viz import mechinterp as M
from src.viz.plots import training_curves, basis_comparison
from src.provenance import read as read_prov


def freq_norms(W, n):
    k = np.arange(1, n // 2 + 1)[:, None]; t = np.arange(n)[None, :]
    s = np.sin(2 * np.pi * k * t / n) @ W
    c = np.cos(2 * np.pi * k * t / n) @ W
    return np.sqrt((s ** 2).sum(1) + (c ** 2).sum(1))


def main(path, n):
    z = np.load(path, allow_pickle=False)
    d = describe(n)
    prov = read_prov(path)
    print("=" * 74)
    print(f"INSPECT  {path}   n={n}")
    print("=" * 74)
    print(f"algebra : {d['factorization']}  phi={d['phi']}  cyclic={d['cyclic']}  "
          f"{d['n_jclasses']} J-classes, {d['n_nonregular']} non-regular")
    print(f"contents: {', '.join(z.files)}")
    if prov:
        print(f"code    : {prov.get('git_sha','?')[:8]}  {prov.get('timestamp_utc','?')}")
    else:
        print("code    : NO PROVENANCE STAMP (pre-dates src/provenance.py)")

    missing = [k for k in ("mlp_acts", "logits_all", "attn") if k not in z.files]
    if missing:
        print(f"\n!! NOT SAVED: {missing}  -- those views cannot be produced for this run")

    # ---- embedding ----
    W = z["W_E"][:n].astype(float); W -= W.mean(0, keepdims=True)
    fn = freq_norms(W, n)
    print(f"\n[embedding]  additive basis: Gini {gini(fn):.3f}  PR {participation_ratio(fn):.2f}"
          f"  key {[k+1 for k in key_freqs_5x_median(fn)]}")
    idx, orders, _ = unit_index(n)
    U = np.array(sorted(idx, key=lambda x: idx[x]))
    if len(orders) == 1:
        fm = freq_norms(W[U], len(U))
        print(f"             mult. basis   : Gini {gini(fm):.3f}  PR {participation_ratio(fm):.2f}"
              f"  key {[k+1 for k in key_freqs_5x_median(fm)]}")
    else:
        print(f"             mult. basis   : unit group non-cyclic {orders}; "
              f"see jclass_spectra.py for the product-character view")

    # ---- what is the circuit computing? ----
    if "mlp_acts" in z.files:
        print()
        M.neuron_term_decomposition(path, n)
        frac = M.neuron_term_decomposition(path, n, verbose=False)
        comp = frac["a+b"].mean() / max(frac["a+b"].mean() + frac["a-b"].mean(), 1e-12)
        print(f"     composition ratio  a+b / (a+b + a-b) = {comp:.1%}"
              f"   (task term vs its mirror; 50% = no preference)")

    # ---- panels ----
    print("\n[panels]")
    for p in M.report(path, n):
        print(f"   {p}")
    for f, args in [(training_curves, (path,)), (basis_comparison, (path, n))]:
        try:
            print(f"   {f(*args)}")
        except Exception as e:
            print(f"   {f.__name__}: unavailable ({e})")
    print("\nOpen the PNGs in figures/. Read them in this order:")
    print("   1. training_curves   -- did it grok, and when")
    print("   2. embedding_n*      -- is the embedding organised by J-class")
    print("   3. neurons_n*        -- what each neuron computes (zoom + DFT columns)")
    print("   4. neurons_n*_dlog   -- the same under discrete log; periodic here => clock")
    print("   5. logit_spectrum    -- block structure of the output")
    print("   6. neuron_freqs      -- single-frequency tuning")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
