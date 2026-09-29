"""I1 -- the internal character intervention.

Pre-registered in `experiments/PREREGISTER_i1_internal_intervention.md`. Commit that file
before this script runs; the criteria must predate the numbers.

    PYTHONPATH=. .venv/bin/python run_intervention.py <n> <seed> [--draws B]
    PYTHONPATH=. .venv/bin/python run_intervention.py --summary <n>

Edits `W_E` in the multiplicative character basis and runs the UNMODIFIED forward pass.
Every other causal number in this project comes from FFT-ing the saved logit tensor,
which cannot distinguish "the output has character structure" from "the network computes
with characters". Softmax attention and the ReLU MLP sit downstream of this edit, so
nothing here is algebraically forced.

The run ABORTS unless the checkpoint reproduces its archived counterpart: a model that is
not the published one cannot carry a claim about the published one.
"""
import json, os, sys, time
from math import gcd

import numpy as np

from src.analysis.ablation import (ablation_report, cross_entropy_np, unit_logit_grid)
from src.analysis.intervention import character_project, random_character_sets
from src.autograd.engine import no_grad
from src.model.transformer import Transformer
from src.provenance import stamp_npz

if "--summary" in sys.argv:
    # Cross-seed report + the key-overlap decomposition of the null.
    # THE DECOMPOSITION IS EXPLORATORY: the pre-registration's analysis plan lists seven
    # steps and this is not one of them, so it is labelled here and in every write-up.
    # It exists because the null has a heavy upper tail that reaches the effect, and the
    # honest question is WHY -- the answer being that the inherited pool includes the key
    # characters, so a draw can partially reproduce the intervention.
    import glob, json as _json
    import numpy as _np
    from src.analysis.intervention import random_character_sets as _rcs
    # The modulus is REQUIRED: I1 (n=113) and I1b (n=121) share one directory, and a glob
    # over it printed a pooled `median ... n = 6` across two moduli -- the cross-modulus
    # import I1b's pre-registration forbids. An optional filter would still pool by default.
    _i = sys.argv.index("--summary") + 1
    assert _i < len(sys.argv) and sys.argv[_i].isdigit(), "usage: run_intervention.py --summary <n>"
    _n = int(sys.argv[_i])
    paths = sorted(glob.glob(f"{os.environ.get('I1_OUT', 'results/i1_internal')}/I1_*_n{_n}_s*.npz"))
    assert paths, f"no I1_*_n{_n}_s*.npz found"
    rows = []
    for _p in paths:
        z = _np.load(_p, allow_pickle=True)
        assert int(z["n"]) == _n, f"{_p} carries n={int(z['n'])}, not {_n}"
        rows.append((int(z["seed"]), z))
    print(f"{'seed':>4} {'baseline':>11} {'restricted':>11} {'excluded':>10} {'acc_r':>6} "
          f"{'exc/base':>10} {'p_perm':>8} {'n>=':>4} {'ctrl med':>11}")
    for s, z in rows:
        print(f"{s:>4} {float(z['internal_baseline']):11.4e} {float(z['internal_restricted']):11.4e} "
              f"{float(z['internal_excluded']):10.4e} {float(z['acc_restricted']):6.4f} "
              f"{float(z['ratio_excluded_baseline']):10.3e} {float(z['p_perm']):8.5f} "
              f"{int(z['n_ge']):>4} {float(z['control_median']):11.4e}")
    r = _np.array([float(z["ratio_excluded_baseline"]) for _, z in rows])
    # C36: median + range + n, and NO +- below n = 5.
    print(f"\nn = {_n}: excluded/baseline  median {_np.median(r):.3e}  range {r.min():.3e}-{r.max():.3e}  n={len(r)}")
    print("\n-- EXPLORATORY (not pre-registered): null damage vs key-character overlap --")
    for s, z in rows:
        kf = [k[0] for k in _json.loads(str(z["key_freqs"]))]
        c, exc = z["control"], float(z["internal_excluded"])
        draws = _rcs(kf, int(z["n"]), int(z["draws"]), seed=0)
        ov = _np.array([len(set(kf) & set(d)) for d in draws])
        base = float(z["internal_baseline"])
        z0 = c[ov == 0]
        print(f"  seed {s}: |K|={len(kf)}  zero-overlap draws {len(z0)}  mean {z0.mean():.4e} "
              f"(baseline {base:.4e}, ratio {z0.mean()/base:.2f}x)  max {z0.max():.4e}")
        print("           dose: " + "  ".join(
            f"{o}:{c[ov==o].mean():.2e}(n={int((ov==o).sum())})"
            for o in range(ov.max() + 1) if (ov == o).sum()))
        assert (c[ov == 0] < exc).all(), f"seed {s}: a ZERO-overlap draw reached the effect"
    print("\n  assert: no zero-overlap draw reaches `excluded`, on any seed -- held")
    raise SystemExit(0)

if "--selfcheck" in sys.argv:
    # The instrument's own checks: bin convention settled by PLANTING a signal, the
    # unmasked round trip, the keep/remove partition on unit rows, pass-through of
    # non-units, the conjugate-closure requirement asserted in BOTH directions, keep_dc,
    # and the null's label matching. Two of these exist because a mutation test showed
    # the suite passed without them.
    from src.analysis.intervention import _selfcheck
    _selfcheck()
    raise SystemExit(0)

N = int(sys.argv[1])
SEED = int(sys.argv[2])
DRAWS = int(sys.argv[sys.argv.index("--draws") + 1]) if "--draws" in sys.argv else 10_000
SRC = os.environ.get("I1_SRC", "results/i1_internal")
ARCH = os.environ.get("I1_ARCHIVE", "results/_archive_i1")
OUT = os.environ.get("I1_OUT", "results/i1_internal")
TAG = f"engine_n{N}_s{SEED}"

# Inherited thresholds -- see the pre-registration. Not one of these was chosen for this
# run: G2's 100x is Gate 2's necessity bar, 0.90 is FAIL_ACC (the C27 excursion floor),
# and p < 0.01 is C22/C23's. A threshold fixed before the measurement is inherited; one
# picked after seeing the number is the C6/C7 ordering error.
I1_RATIO, I2_ACC, I3_P = 100.0, 0.90, 0.01


def build(z):
    """Rebuild the trained model from the saved parameters."""
    m = Transformer(N + 1, N, d_model=128, n_heads=4, d_head=32, d_mlp=512, seed=0)
    names = [k[2:] for k in z.files if k.startswith("p_")]
    assert len(names) == 9, f"{TAG}: {len(names)} p_* keys, expected 9 -- old artifact?"
    for k in names:
        t = getattr(m, k)
        assert t.data.shape == z[f"p_{k}"].shape, (k, t.data.shape, z[f"p_{k}"].shape)
        t.data = z[f"p_{k}"].astype(t.data.dtype)
    return m


def forward(m, x_all, chunk=4000):
    with no_grad():
        return np.concatenate([m(x_all[i:i + chunk]).data
                               for i in range(0, len(x_all), chunk)])


def main():
    t0 = time.time()
    src = f"{SRC}/WE_{TAG}.npz"
    z = np.load(src, allow_pickle=True)

    # --- provenance stamp FIRST, before any output file exists. Opening the output on a
    # --- new path creates an untracked file, which makes git_dirty True against a tree
    # --- that was clean an instant earlier.
    prov = stamp_npz(dict(experiment="I1", n=N, seed=SEED, draws=DRAWS,
                          source=src, i1_ratio=I1_RATIO, i2_acc=I2_ACC, i3_p=I3_P))

    m = build(z)
    a, b = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    x_all = np.stack([a.ravel(), b.ravel(), np.full(a.size, N)], 1)
    y_all = (a.ravel() * b.ravel()) % N
    # The character basis is defined on units, so the reported loss is the unit sub-grid.
    # CE is a mean over cells, so a boolean mask and the exponent-ordered grid give the
    # same number -- no need to re-derive the ordering here.
    um = np.array([gcd(int(i), N) == 1 and gcd(int(j), N) == 1
                   for i, j in zip(a.ravel(), b.ravel())])
    W0 = z["p_W_E"].astype(float)

    L0 = forward(m, x_all)

    # ---- POSITIVE CONTROL: this must be the published model, or it claims nothing. ----
    arch = f"{ARCH}/WE_{TAG}.npz"
    if os.path.exists(arch):
        za = np.load(arch, allow_pickle=True)
        dW = float(np.abs(za["W_E"].astype(float) - W0).max())
        dL = float(np.abs(za["logits_all"].astype(float) - L0).max())
        print(f"  archive check: max|dW_E| = {dW:.3e}   max|dlogits| = {dL:.3e}")
        assert dW < 1e-5 and dL < 1e-4, (
            f"{TAG} does NOT reproduce {arch} (dW={dW:.3e}, dL={dL:.3e}). "
            "A checkpoint that is not the published model cannot carry a claim about it.")
        repro = dict(archive=arch, max_dW_E=dW, max_dlogits=dL, reproduced=True)
    else:
        print(f"  !! no archive at {arch} -- reproduction NOT verified")
        repro = dict(archive=arch, reproduced=False)

    # ---- the key set, from the ONE function the published causal test uses ----
    G, y_u, orders, kf = unit_logit_grid({"logits_all": L0, "W_E": W0}, N)
    print(f"  key characters ({len(kf)}): {kf}")
    assert kf, f"{TAG}: empty key set -- nothing to intervene on"

    # ---- the published logit-space test on this same model, for the side-by-side ----
    q = int(np.prod(orders))
    logit_space = ablation_report(G, y_u, kf, orders)
    logit_space.pop("per_freq", None)

    def score(W):
        mm = build(z)
        mm.W_E.data = W.astype(mm.W_E.data.dtype)
        L = forward(mm, x_all)
        return (cross_entropy_np(L[um], y_all[um]),
                float((L[um].argmax(1) == y_all[um]).mean()),
                cross_entropy_np(L, y_all))

    base_ce, base_acc, base_full = score(W0)
    res_ce, res_acc, _ = score(character_project(W0, N, keep=kf, keep_dc=True))
    exc_ce, exc_acc, _ = score(character_project(W0, N, remove=kf))
    print(f"  internal: baseline {base_ce:.6e} (acc {base_acc:.4f}) | "
          f"restricted {res_ce:.6e} (acc {res_acc:.4f}) | "
          f"excluded {exc_ce:.6e} (acc {exc_acc:.4f})")

    # ---- the null: same label count, same closure, drawn the way analyze_n4 draws ----
    draws = random_character_sets(kf, N, DRAWS, seed=0)
    ctrl = np.empty(DRAWS)
    for i, dr in enumerate(draws):
        ctrl[i] = score(character_project(W0, N, remove=dr))[0]
        if (i + 1) % 500 == 0:
            el = time.time() - t0
            print(f"    null {i+1}/{DRAWS}  [{el/60:.1f}m, eta "
                  f"{el/(i+1)*(DRAWS-i-1)/60:.1f}m]", flush=True)

    # PHIPSON-SMYTH, as R1 pre-registered: p = (1+b)/(B+1), never b/B. A Monte-Carlo p
    # cannot claim more resolution than its draw count.
    n_ge = int((ctrl >= exc_ce).sum())
    p_perm = (1.0 + n_ge) / (DRAWS + 1.0)
    ratio = exc_ce / base_ce

    v = dict(I1_necessity=bool(ratio >= I1_RATIO),
             I2_sufficiency=bool(res_acc >= I2_ACC),
             I3_specificity=bool(p_perm < I3_P))
    print(f"  excluded/baseline = {ratio:.3e}x   "
          f"p_perm = {p_perm:.5f} ({n_ge} of {DRAWS}, floor {1/(DRAWS+1):.2e})")
    print(f"  control median {np.median(ctrl):.6e}  max {ctrl.max():.6e}")
    print(f"  I1 {'PASS' if v['I1_necessity'] else 'FAIL'} | "
          f"I2 {'PASS' if v['I2_sufficiency'] else 'FAIL'} | "
          f"I3 {'PASS' if v['I3_specificity'] else 'FAIL'}")

    out = dict(prov,
               n=N, seed=SEED, key_freqs=json.dumps([[int(x) for x in np.atleast_1d(k)] for k in kf]),
               n_key=len(kf), draws=DRAWS,
               internal_baseline=base_ce, internal_restricted=res_ce,
               internal_excluded=exc_ce, internal_baseline_full=base_full,
               acc_baseline=base_acc, acc_restricted=res_acc, acc_excluded=exc_acc,
               ratio_excluded_baseline=ratio, p_perm=p_perm, n_ge=n_ge,
               control=ctrl, control_median=float(np.median(ctrl)),
               control_max=float(ctrl.max()),
               logit_baseline=logit_space["baseline"],
               logit_restricted=logit_space["restricted"],
               logit_excluded=logit_space["excluded"],
               repro=json.dumps(repro), verdict=json.dumps(v))
    os.makedirs(OUT, exist_ok=True)
    dst = f"{OUT}/I1_{TAG}.npz"
    np.savez_compressed(dst, **out)
    # The project's own reader must be able to read the stamp back. An artifact that is
    # stamped and unreadable looks identical to a correct one from the outside.
    from src import provenance
    assert provenance.read(dst) is not None, f"{dst}: stamp does not read back"
    print(f"  -> {dst}   [{(time.time()-t0)/60:.1f} min]")


if __name__ == "__main__":
    main()
