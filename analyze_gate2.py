"""G2 -- Gate 2: does every algebraic stratum run the same clock, on its own local group?

Scores the criteria pre-registered in `experiments/PREREGISTER_gate2_strata.md`, committed
at bca6236 before any per-stratum number existed.

  PYTHONPATH=. .venv/bin/python analyze_gate2.py [results_dir]
  PYTHONPATH=. .venv/bin/python analyze_gate2.py --selfcheck

Every multiplicative statistic in this project -- C6, C21, C22, C25, C31 -- is computed on
the unit sub-grid. That is 98.2% of the multiplication table at n=113 and 82.6% / 64.0% /
23.5% at 121 / 125 / 165. This measures the rest.

The algebra (pre-registration section 1.1). For x in J_d write x = d*u with u a unit mod
n/d; likewise y = e*v. With m = gcd(de, n) and de = m*w, gcd(w, n/m) = 1:

    x*y mod n  =  m * ( w * (u*v) mod n/m )

so the stratum is a fixed unit relabelling of multiplication in the smaller ring Z/(n/m),
the target is exactly constant on the fibres of (Z/(n/d))* -> (Z/(n/m))*, and a local clock
must put its logit energy on the diagonal (kappa, kappa) in local-group coordinates.

No key-frequency detector is used. The ablated set is the whole diagonal, fixed by
the algebra above. That is deliberate: choosing the set by ablation rank and then asking a
permutation test whether that set is unusually damaging is circular.
"""
import functools
import json
import os
import re
import sys
from math import gcd

import numpy as np
from sympy import divisors

from src.analysis.ablation import (conjugate_closure, cross_entropy_np, fourier_ablate,
                                  fourier_spectrum)
from src.analysis.neurons import freq_fraction, freq_fraction_nd, shuffle_control
from src.analysis.transforms import unit_index
from src.analysis.runs import last_test_acc, state as endpoint_state
from src import provenance

D = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "results/k03_grid_acts"

# --- thresholds, all fixed in the pre-registration -------------------------------------
G_MIN = 10            # S-b: below this the 8-bin family cannot discriminate
HELDOUT_MIN = 100     # S-c: fewer unseen cells cannot separate clock from memorisation
N_CONTROL = 10000     # R1 (PREREGISTER_r1_permutation_B.md): 200 has a floor of 1/201 and
                      # cannot express the p it was being printed at. 20 was far too few for
                      # a heavy-tailed control; 200 was too few for a 320-test family.
TRAIN_FRAC = 0.30     # kernels/k03_grid_acts/run.py and run_n7.py
G0_ACC = 0.95         # G0 held-out accuracy
G1_P = 0.01           # G1 permutation p
G2_EXCL = 100.0       # G2 excluded / baseline
G3_RATIO = 0.5        # G3 residual <= this * control residual
G4_R = 5.0            # G4 diagonal share / chance
G5_MARGIN = 0.20      # G5 tuned - shuffled
TUNE_THRESH = 0.85    # C31 / 2606.17399 / Nanda
SEED_MIN = 4          # ">= 4/5 seeds" on the primary arm


# --- algebra ----------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def jclass_dlog(n, d):
    """J_d = {x : gcd(x,n) = d}, listed in discrete-log order of its action coordinate.

    x = d*u with u a unit mod n/d, and u -> d*u mod n is a bijection (Z/(n/d))* -> J_d
    because gcd(d*u, d*(n/d)) = d*gcd(u, n/d) = d. Returns (residues, units, orders).
    """
    M = n // d
    idx, orders, inverse = unit_index(M)
    E = list(np.ndindex(*orders)) if orders else [()]
    us = np.array([inverse[e] for e in E], dtype=np.int64)
    return (d * us) % n, us, orders


@functools.lru_cache(maxsize=None)
def strata(n):
    """Every (d,e) block with its local group, size, and measurability verdict."""
    out = []
    for d in divisors(n):  # cached: measure() deep-copies before mutating
        for e in divisors(n):
            if d == n or e == n:
                continue                                  # J_n = {0}: block identically 0
            m = gcd(d * e, n)
            L = n // m
            orders_m = unit_index(L)[1] if L > 1 else ()
            g = int(np.prod(orders_m)) if orders_m else 1
            rows, cols = len(jclass_dlog(n, d)[0]), len(jclass_dlog(n, e)[0])
            cells = rows * cols
            heldout = int(round(cells * (1 - TRAIN_FRAC)))
            why = None
            if g < G_MIN:
                why = f"|G_m|={g} < {G_MIN}"
            elif heldout < HELDOUT_MIN:
                why = f"held-out ~{heldout} < {HELDOUT_MIN}"
            out.append(dict(d=d, e=e, m=m, local=L, orders_m=orders_m, g=g, rows=rows,
                            cols=cols, cells=cells, cyclic=len(orders_m) == 1,
                            measurable=why is None, why=why))
    return out


@functools.lru_cache(maxsize=None)
def fibre_map(n, d, m):
    """Position in J_d's dlog order -> flat position in G_m's dlog order, via u -> u mod n/m.

    n/m divides n/d because d | gcd(de,n) = m, so the reduction is defined and surjective.
    """
    L = n // m
    _, us, _ = jclass_dlog(n, d)
    idx_L, orders_L, _ = unit_index(L)
    E_L = {e: i for i, e in enumerate(np.ndindex(*orders_L))}
    return np.array([E_L[idx_L[int(u) % L]] for u in us], dtype=np.int64), orders_L


# --- block extraction and collapse ------------------------------------------------------

def block(flat, n, d, e):
    """(n*n, K) in row-major (a,b) -> (|J_d|, |J_e|, K) in dlog order on both axes."""
    xa = jclass_dlog(n, d)[0]
    xb = jclass_dlog(n, e)[0]
    G = np.asarray(flat).reshape(n, n, -1)
    return G[np.ix_(xa, xb)].astype(float), xa, xb


def collapse(B, rf, cf, nr, nc):
    """Average a block onto the local-group grid: fibres of the two reduction maps."""
    R, C, K = B.shape
    out = np.zeros((nr, nc, K))
    cnt = np.zeros((nr, nc))
    np.add.at(out, (rf[:, None].repeat(C, 1), cf[None, :].repeat(R, 0)), B)
    np.add.at(cnt, (rf[:, None].repeat(C, 1), cf[None, :].repeat(R, 0)), 1)
    assert (cnt > 0).all(), "a local-group cell had no preimage -- reduction is not onto"
    return out / cnt[..., None], cnt


def fibre_residual(B, rf, cf, nr, nc, seed=0):
    """Share of block variance destroyed by fibre-averaging, and a permuted-fibre control.

    The control is what makes this informative. Averaging always smooths, so a small
    residual means nothing without a control that has the same fibre sizes and no algebraic
    meaning. C7b, C19 and C20 were each retracted for a size confound of this kind.
    """
    tot = float(((B - B.mean((0, 1), keepdims=True)) ** 2).sum())
    if tot <= 0:
        return None
    def resid(r, c):
        C, _ = collapse(B, r, c, nr, nc)
        return float(((B - C[np.ix_(r, c)]) ** 2).sum()) / tot
    rng = np.random.default_rng(seed)
    return resid(rf, cf), resid(rng.permutation(rf), rng.permutation(cf))


# --- spectral and causal ----------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def _bins(orders):
    """Conjugate classes of non-DC (kappa_a, kappa_b) bins, and the diagonal's classes."""
    r = len(orders)
    allk = list(np.ndindex(*orders))
    seen, classes, diag = set(), [], []
    for ka in allk:
        for kb in allk:
            if (ka, kb) in seen or (ka + kb) == (0,) * (2 * r):
                continue
            cl = tuple(sorted(conjugate_closure([(ka, kb)], orders)))
            seen.update(cl)
            classes.append(cl)
            if ka == kb:
                diag.append(cl)
    return classes, diag


def diag_share(C, orders):
    """Diagonal share of non-DC energy, and its ratio to the exact chance level.

    Chance is (|G|-1)/(|G|^2-1) = 1/(|G|+1) exactly, so R = D*(|G|+1) is size-normalised in
    closed form -- no cross-stratum comparison of a raw number anywhere.
    """
    r = len(orders)
    g = int(np.prod(orders))
    A = C.reshape(orders + orders + (-1,))
    F = np.abs(np.fft.fftn(A, axes=tuple(range(2 * r)))) ** 2
    E = F.sum(-1)
    tot = E.sum() - E[(0,) * (2 * r)]
    dg = sum(E[k + k] for k in np.ndindex(*orders)) - E[(0,) * (2 * r)]
    Dv = float(dg / max(tot, 1e-30))
    return Dv, Dv * (g + 1)


def ablate(C, targets, orders, seed=0):
    """Baseline / restricted / excluded on the whole diagonal, plus the permutation p.

    The control draws random conjugate-closed bin sets of the same bin count from all
    non-DC bins, not from the diagonal. The question is whether the diagonal is special
    among bin sets of its size, which is what "the clock is the mechanism" asserts.
    """
    r = len(orders)
    g = int(np.prod(orders))
    A = C.reshape(orders + orders + (-1,))
    flat = lambda G: G.reshape(g * g, -1)
    classes, diag = _bins(orders)
    keep = sorted({b for cl in diag for b in cl})
    target_n = len(keep)
    base = cross_entropy_np(flat(A), targets)
    # The grid is constant across the whole null, so its forward transform is computed once.
    F = fourier_spectrum(A, r)
    restricted = cross_entropy_np(flat(fourier_ablate(A, keep=keep, spectrum=F)), targets)
    excluded = cross_entropy_np(flat(fourier_ablate(A, remove=keep, spectrum=F)), targets)

    rng = np.random.default_rng(seed)
    ctrl = []
    for _ in range(N_CONTROL):
        picked, size, order = [], 0, rng.permutation(len(classes))
        for i in order:                      # greedy to an EXACT equal bin count
            cl = classes[i]
            if size + len(cl) <= target_n:
                picked.append(cl); size += len(cl)
            if size == target_n:
                break
        rem = sorted({b for cl in picked for b in cl})
        ctrl.append(cross_entropy_np(flat(fourier_ablate(A, remove=rem, spectrum=F)), targets))
    ctrl = np.array(ctrl)
    # PHIPSON-SMYTH, PRE-REGISTERED IN R1: p_hat = (1+b)/(B+1), never b/B.
    n_ge = int((ctrl >= excluded).sum())
    return dict(baseline=base, restricted=restricted, excluded=excluded,
                p_perm=(1.0 + n_ge) / (len(ctrl) + 1.0), n_ge=n_ge, n_draws=int(len(ctrl)),
                ctrl_median=float(np.median(ctrl)), n_bins=target_n)


def lut_r2(B, Y):
    """How much of the block's logit tensor is just "large at the correct token"?

    Exploratory: added after the pre-registration. On a stratum, any correct function is a
    function of dlog(u) + dlog(v), so a perfectly confident lookup table already reads
    D = 1.000, permutation p = 0 and fibre residual 0 (selfcheck 2 plants exactly that).
    G1, G2 and G4 are therefore partly entailed by correctness: they are necessary
    conditions, not evidence of a clock. G5 (single-frequency MLP neurons) is not entailed
    by correctness, and is where the mechanism claim rests.

    This is the number that says how close to vacuous the logit tests are for a given block:
    R^2 of the block logits on {1, onehot(target)}. Near 1.0 means the logits carry almost
    nothing beyond the answer, and G1/G2/G4 should be read as consistency checks only.
    """
    R, C, K = B.shape
    X = np.zeros((R, C, K))
    np.put_along_axis(X, Y[:, :, None], 1.0, axis=2)
    b, x = B.ravel(), X.ravel()
    bc, xc = b - b.mean(), x - x.mean()
    beta = float(xc @ bc) / max(float(xc @ xc), 1e-30)
    return 1.0 - float(((bc - beta * xc) ** 2).sum()) / max(float((bc ** 2).sum()), 1e-30)


# --- the split (reproduced offline, never trusted without the assert) --------------------

def heldout_mask(n, seed):
    perm = np.random.default_rng(seed).permutation(n * n)
    cut = int(TRAIN_FRAC * n * n)
    mask = np.zeros(n * n, bool)
    mask[perm[cut:]] = True
    return mask


def check_split(z, n, seed):
    """The recomputed split must reproduce the run's OWN final test accuracy.

    Pre-registration section 6.7. A split that is merely 'the same code' is an assertion;
    O18 found a comment claiming exactly that which nobody had ever run.
    """
    if "logits_all" not in z.files or "hist" not in z.files:
        return None
    te = heldout_mask(n, seed)
    pred = z["logits_all"].argmax(-1)
    acc = float((pred[te] == z["y_all"][te]).mean())
    return acc, last_test_acc(z)   # the LAST ROW: logits_all is that checkpoint's


# --- one run ----------------------------------------------------------------------------

def measure(f, n, seed):
    z = np.load(f, allow_pickle=False)
    if "logits_all" not in z.files:
        return dict(skip="logits_all NOT SAVED -- a gap in what was saved is not absence")
    # --controls scores ONLY the FAILED runs; the primary arm scores ONLY the grokked ones;
    # a NEAR-GROK is in neither.
    state, acc = run_state(D, n, seed, z)
    want = "FAILED" if "--controls" in sys.argv else "grokked"
    if state != want:
        return dict(skip=f"{state} (final test acc {acc:.4f}) -- this arm scores {want}")
    chk = check_split(z, n, seed)
    if chk and abs(chk[0] - chk[1]) > 0.01:
        return dict(skip=f"split does not reproduce: recomputed {chk[0]:.4f} vs logged {chk[1]:.4f}")
    L, y_all = z["logits_all"], z["y_all"]
    H = z["mlp_acts"] if "mlp_acts" in z.files else None
    te = heldout_mask(n, seed)
    rows = []
    for s in [dict(x) for x in strata(n)]:
        d, e, m = s["d"], s["e"], s["m"]
        xa, xb = jclass_dlog(n, d)[0], jclass_dlog(n, e)[0]
        cell = (xa[:, None] * n + xb[None, :]).ravel()
        s["heldout"] = int(te[cell].sum())
        if s["measurable"] and s["heldout"] < HELDOUT_MIN:
            s["measurable"], s["why"] = False, f"held-out {s['heldout']} < {HELDOUT_MIN}"
        sel = cell[te[cell]]
        s["acc_heldout"] = (float((L.argmax(-1)[sel] == y_all[sel]).mean())
                            if len(sel) else float("nan"))
        s["acc_all"] = float((L.argmax(-1)[cell] == y_all[cell]).mean())
        if not s["measurable"]:
            rows.append(s); continue

        rf, orders_m = fibre_map(n, d, m)
        cf, _ = fibre_map(n, e, m)
        g = int(np.prod(orders_m))
        B, _, _ = block(L, n, d, e)
        C, cnt = collapse(B, rf, cf, g, g)

        # targets: constant on fibres by the algebra of section 1.1 -- ASSERTED, not assumed
        Y = y_all.reshape(n, n)[np.ix_(xa, xb)]
        T = np.full((g, g), -1, np.int64)
        for i in range(len(xa)):
            for j in range(len(xb)):
                t = T[rf[i], cf[j]]
                assert t < 0 or t == Y[i, j], (
                    f"n={n} ({d},{e}): target is NOT constant on the fibre -- the local-ring "
                    f"reduction is wrong")
                T[rf[i], cf[j]] = Y[i, j]
        s["lut_r2"] = lut_r2(B, Y)
        s["Dshare"], s["R"] = diag_share(C, orders_m)
        s.update(ablate(C, T.ravel(), orders_m))
        fr = (fibre_residual(B, rf, cf, g, g)
              if (len(xa) > g or len(xb) > g) else None)
        s["resid"], s["resid_ctrl"] = fr if fr else (None, None)

        if H is not None and g >= G_MIN:
            HB, _, _ = block(H, n, d, e)
            HC, _ = collapse(HB, rf, cf, g, g)
            if s["cyclic"]:
                # G5 as pre-registered: the cyclic 8-bin family.
                s["tuned"] = float((freq_fraction(HC)[0] > TUNE_THRESH).mean())
                s["tuned_sh"] = float((freq_fraction(shuffle_control(HC, seed=seed))[0]
                                       > TUNE_THRESH).mean())
            else:
                # EXPLORATORY, beyond the pre-registration, which skips a non-cyclic G_m on
                # the neuron endpoint. Added because G5 is the ONE endpoint not entailed by
                # correctness (see lut_r2), and skipping it removed the mechanism evidence
                # from exactly the moduli whose stratification is a LATTICE rather than a
                # chain -- 119, 120 and most of 165. Reported separately, never as G5.
                A = HC.reshape(orders_m + orders_m + (-1,))
                s["tuned_nd"] = float((freq_fraction_nd(A, orders_m)[0] > TUNE_THRESH).mean())
                sh = shuffle_control(HC, seed=seed).reshape(A.shape)
                s["tuned_nd_sh"] = float((freq_fraction_nd(sh, orders_m)[0] > TUNE_THRESH).mean())
        rows.append(s)
    return dict(n=n, seed=seed, rows=rows, split_ok=chk)


# --- self-checks: settle conventions by PLANTING, never by reasoning --------------------

def _selfcheck():
    # 1. the algebra: x*y is a relabelling of u*v in the local ring, at every modulus
    for n in (113, 121, 125, 119, 120, 165):
        for s in strata(n):
            d, e, m = s["d"], s["e"], s["m"]
            if s["local"] == 1:
                continue
            rf, om = fibre_map(n, d, m)
            cf, _ = fibre_map(n, e, m)
            xa, xb = jclass_dlog(n, d)[0], jclass_dlog(n, e)[0]
            g = int(np.prod(om))
            seen = {}
            for i in range(len(xa)):
                for j in range(len(xb)):
                    k = (rf[i], cf[j])
                    v = int(xa[i]) * int(xb[j]) % n
                    assert seen.setdefault(k, v) == v, (n, d, e, "target not fibre-constant")
            assert len(seen) == g * g, (n, d, e, len(seen), g * g)
    print("  selfcheck 1 PASS  targets are constant on fibres at 6 moduli, all strata")

    # 2. a PLANTED local clock must read D ~ 1, R ~ |G|+1, p = 0, residual ~ 0
    n, d, e = 125, 1, 5
    m = gcd(d * e, n); g = int(np.prod(unit_index(n // m)[1]))
    rf, om = fibre_map(n, d, m); cf, _ = fibre_map(n, e, m)
    xa, xb = jclass_dlog(n, d)[0], jclass_dlog(n, e)[0]
    lg = np.zeros((len(xa), len(xb), n))
    for i in range(len(xa)):
        for j in range(len(xb)):
            lg[i, j, int(xa[i]) * int(xb[j]) % n] = 12.0      # a perfect local clock
    C, _ = collapse(lg, rf, cf, g, g)
    Dv, R = diag_share(C, om)
    T = np.array([[int(xa[i]) * int(xb[j]) % n for j in range(len(xb))]
                  for i in range(len(xa))])
    Tc = np.zeros((g, g), np.int64)
    for i in range(len(xa)):
        for j in range(len(xb)):
            Tc[rf[i], cf[j]] = T[i, j]
    ab = ablate(C, Tc.ravel(), om)
    res, ctl = fibre_residual(lg, rf, cf, g, g)
    assert R > 0.5 * (g + 1), f"planted clock reads R={R:.2f}, expected ~{g+1}"
    # (1+b)/(B+1) can never read 0; a planted clock must sit at the FLOOR, i.e. b = 0.
    assert ab["n_ge"] == 0, ab["n_ge"]
    assert ab["p_perm"] == 1.0 / (ab["n_draws"] + 1), ab["p_perm"]
    assert res < 1e-9 < ctl, (res, ctl)
    print(f"  selfcheck 2 PASS  planted local clock: D={Dv:.3f} R={R:.1f}/{g+1} "
          f"p={ab['p_perm']:.3f} resid={res:.2e} vs control {ctl:.3f}")

    # 3. a PLANTED non-clock (white noise) must read R ~ 1 and p ~ 1
    rng = np.random.default_rng(0)
    nz = rng.standard_normal((len(xa), len(xb), n))
    C2, _ = collapse(nz, rf, cf, g, g)
    D2, R2 = diag_share(C2, om)
    ab2 = ablate(C2, Tc.ravel(), om)
    assert R2 < 3.0, f"white noise reads R={R2:.2f}, chance is 1.0"
    assert ab2["p_perm"] > 0.10, ab2["p_perm"]
    print(f"  selfcheck 3 PASS  white noise: R={R2:.2f} (chance 1.0) p={ab2['p_perm']:.3f}")

    # 4. the FIBRE map: a signal depending only on u mod n/m collapses; one depending on u
    #    itself does not. This is what settles the reduction -- three bin-convention bugs in
    #    this project and reasoning lost every time.
    loc = np.zeros((len(xa), len(xb), 1))
    glb = np.zeros((len(xa), len(xb), 1))
    _, us_a, _ = jclass_dlog(n, d)
    _, us_b, _ = jclass_dlog(n, e)
    L = n // m
    for i in range(len(xa)):
        for j in range(len(xb)):
            loc[i, j, 0] = np.cos(2 * np.pi * (int(us_a[i]) % L) / L)
            glb[i, j, 0] = np.cos(2 * np.pi * int(us_a[i]) / n)
    rl = fibre_residual(loc, rf, cf, g, g)[0]
    rgl = fibre_residual(glb, rf, cf, g, g)[0]
    assert rl < 1e-12, rl
    assert rgl > 0.10, rgl
    print(f"  selfcheck 4 PASS  fibre map: local signal resid {rl:.2e}, "
          f"u-dependent signal resid {rgl:.3f}")

    # 5. the equal-bin-count control really is equal
    classes, diag = _bins((10,))
    assert len({b for cl in diag for b in cl}) == 9, "diagonal of Z_10 should be 9 non-DC bins"
    print("  selfcheck 5 PASS  diagonal bin count = |G|-1")
    print("analyze_gate2 selfcheck PASS")


# --- reporting -------------------------------------------------------------------------

MODULI = [113, 121, 125, 119, 120, 165]
SEEDS = [0, 1, 2, 3, 4]


def discover(d):
    """(n, seed) pairs actually present in a results directory.

    The control arm the pre-registration requires lives in results/k04_extended at n=49 and
    n=54, which a hard-coded modulus list can never reach -- so the arm would have been
    'required' and unscorable. Discovery is also what lets the engine arm (results/n7_engine,
    WE_engine_*) run through the same path as the primary arm rather than a parallel one.
    """
    pat = re.compile(r"WE_(?:B_thesis|engine)_n(\d+)_s(\d+)\.npz$")
    found = sorted({(int(m.group(1)), int(m.group(2)))
                    for f in os.listdir(d) if (m := pat.match(f))})
    return found


# Pre-registration section 2: a near-grok is neither a data point nor a control
# (three states: grokked / near-grok / failed). k03's n125_s1 ends at 0.9779, a model that
# has learned the circuit; scoring it as a negative control would erase the contrast.
NEAR_GROK = {("results/k03_grid_acts", 125, 1)}
# The two thresholds live in src/analysis/runs.py and are not duplicated here.
# Between them is a near-grok, which is neither. A binary gate turns every near-miss into
# whichever side it falls on, in either direction: near-grok runs such as k04's n=75
# (0.9855), n=99 (0.9794), n=100 (0.9880) and engine n119_s0 (0.9869) behave like grokked
# models and do not belong in the control arm.


# Both readings now come from src/analysis/runs.py, which owns them for the whole project.
# They are deliberately two names because they answer two questions, and this file is where
# conflating them did damage: `check_split` wants the LAST ROW, because that is the
# checkpoint `logits_all` was saved at, while classification wants the WINDOW, because a
# run's endpoint is never its last row (C27). Reading by column name matters for the engine
# arm, where hist[-1][3] is TRAIN accuracy: that would have let n125_s1 (test 0.6082, train
# ~1.0) into the PRIMARY arm and made every split check compare test against train.


def run_state(d, n, s, z):
    """grokked / near-grok / FAILED. Three states, never two; endpoint is a window."""
    if (d.rstrip("/"), n, s) in NEAR_GROK:
        return "near-grok", last_test_acc(z)
    return endpoint_state(z)


def path(d, n, s):
    for pat in (f"WE_B_thesis_n{n}_s{s}.npz", f"WE_engine_n{n}_s{s}.npz"):
        f = os.path.join(d, pat)
        if os.path.exists(f):
            return f
    return None


def fmt(v, w=8, p=3, e=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return " " * (w - 1) + "-"
    return f"{v:>{w}.{p}{'e' if e else 'f'}}"


def run_table(res):
    print(f"\n  n={res['n']} seed={res['seed']}   split check: recomputed "
          f"{res['split_ok'][0]:.4f} vs logged {res['split_ok'][1]:.4f}")
    # D and R are both printed ON PURPOSE. R = D*(|G_m|+1) is size-normalised, but its
    # CEILING is |G_m|+1, so "R=108 at the unit stratum vs R=10.5 at J1xJ11" must never be
    # read as ten times more clock-like -- both are ~97% of their own maximum. D is the
    # comparable number; R is the one with a fixed chance level of 1.0.
    print(f"    {'stratum':>14} {'m':>4} {'G_m':>10} {'cells':>7} {'held':>6} "
          f"{'acc_ho':>7} {'D':>6} {'R=D/ch':>7} {'Rmax':>5} {'perm p':>7} {'excl/base':>10} "
          f"{'resid':>7} {'ctrl':>6} {'tuned':>6} {'shuf':>6} {'lutR2':>6} {'*tunND':>7} {'*shND':>6}")
    for s in sorted(res["rows"], key=lambda r: -r["cells"]):
        g = "x".join(f"Z{o}" for o in s["orders_m"]) or "triv"
        tag = f"J{s['d']}xJ{s['e']}"
        if not s["measurable"]:
            print(f"    {tag:>14} {s['m']:>4} {g:>10} {s['cells']:>7} {s['heldout']:>6} "
                  f"{fmt(s['acc_heldout'],7,4)}   NOT MEASURABLE: {s['why']}")
            continue
        er = s["excluded"] / max(s["baseline"], 1e-30)
        print(f"    {tag:>14} {s['m']:>4} {g:>10} {s['cells']:>7} {s['heldout']:>6} "
              f"{fmt(s['acc_heldout'],7,4)} {fmt(s['Dshare'],6,3)} {fmt(s['R'],7,2)} "
              f"{s['g']+1:>5} {fmt(s['p_perm'],7,4)} "
              f"{fmt(er,10,2,e=True)} {fmt(s.get('resid'),7,3)} {fmt(s.get('resid_ctrl'),6,3)} "
              f"{fmt(s.get('tuned'),6,3)} {fmt(s.get('tuned_sh'),6,3)} "
              f"{fmt(s.get('lut_r2'),6,3)} {fmt(s.get('tuned_nd'),7,3)} "
              f"{fmt(s.get('tuned_nd_sh'),6,3)}   (* = exploratory)"
              if s.get("tuned_nd") is not None else f"{fmt(s.get('lut_r2'),6,3)}")


def score(all_res, label):
    """G0-G5 per stratum, pooled over seeds. A criterion is met at >= SEED_MIN/5 seeds."""
    print(f"\n{'='*100}\nCRITERIA -- {label}\n{'='*100}")
    print(f"  {'n':>4} {'stratum':>12} {'unit?':>6} {'seeds':>6} "
          f"{'G0 acc':>18} {'G1 p<0.01':>10} {'G2 excl':>10} {'G3 fibre':>10} "
          f"{'G4 R>=5':>10} {'G5 tuned':>10}")
    verdict = {}
    for n in sorted({r["n"] for r in all_res}):
        keys = {}
        for r in all_res:
            if r["n"] != n:
                continue
            for s in r["rows"]:
                if s["measurable"]:
                    keys.setdefault((s["d"], s["e"]), []).append(s)
        for (d, e), ss in sorted(keys.items(), key=lambda kv: -kv[1][0]["cells"]):
            k = len(ss)
            need = min(SEED_MIN, k)
            g0 = sum(s["acc_heldout"] >= G0_ACC for s in ss)
            g1 = sum(s["p_perm"] < G1_P for s in ss)
            g2 = sum(s["excluded"] / max(s["baseline"], 1e-30) >= G2_EXCL for s in ss)
            g3n = [s for s in ss if s.get("resid") is not None]
            g3 = sum(s["resid"] <= G3_RATIO * s["resid_ctrl"] for s in g3n)
            g4 = sum(s["R"] >= G4_R for s in ss)
            g5n = [s for s in ss if s.get("tuned") is not None]
            g5 = sum(s["tuned"] - s["tuned_sh"] >= G5_MARGIN for s in g5n)
            ok = lambda c, tot: f"{c}/{tot} {'PASS' if tot and c >= min(SEED_MIN,tot) else 'FAIL'}"
            acc = np.mean([s["acc_heldout"] for s in ss])
            print(f"  {n:>4} {f'J{d}xJ{e}':>12} {str(d==1 and e==1):>6} {k:>6} "
                  f"{ok(g0,k)+f' ({acc:.4f})':>18} {ok(g1,k):>10} {ok(g2,k):>10} "
                  f"{(ok(g3,len(g3n)) if g3n else 'N/A'):>10} {ok(g4,k):>10} "
                  f"{(ok(g5,len(g5n)) if g5n else 'skip'):>10}")
            verdict[(n, d, e)] = dict(unit=(d == 1 and e == 1), seeds=k,
                                      G0=g0 >= need, G1=g1 >= need, G2=g2 >= need,
                                      G3=(not g3n) or g3 >= min(SEED_MIN, len(g3n)),
                                      G4=g4 >= need, G5=(not g5n) or g5 >= min(SEED_MIN, len(g5n)))
    return verdict


def lattice(ns=tuple(MODULI)):
    """G6: the stratum structure, pure algebra. Chain (prime power) vs lattice (square-free)."""
    print(f"\n{'='*100}\nG6 -- STRATIFICATION (pure algebra, no model)\n{'='*100}")
    for n in ns:
        ms = sorted({s["m"] for s in strata(n) if s["local"] > 1})
        gs = [f"m={m}: {'x'.join(f'Z{o}' for o in unit_index(n//m)[1])}" for m in ms]
        chain = all(a % b == 0 or b % a == 0 for a in ms for b in ms)
        print(f"  n={n:>4}  {len(ms)} non-trivial local group(s), "
              f"{'NESTED CHAIN' if chain else 'NON-CHAIN LATTICE'}")
        print(f"          {' | '.join(gs)}")


def gate2_verdict(v):
    """The verdict rule, fixed in the pre-registration section 3.1 before the data existed."""
    print(f"\n{'='*100}\nGATE 2 VERDICT (rule fixed in pre-registration section 3.1)\n{'='*100}")
    nonunit = {k: r for k, r in v.items() if not r["unit"]}
    if not nonunit:
        print("  NO VERDICT -- no measurable non-unit stratum"); return
    clock = lambda r: r["G1"] and r["G2"]
    bad_g0 = [k for k, r in nonunit.items() if not r["G0"]]
    no_clock = [k for k, r in nonunit.items() if r["G0"] and not clock(r)]
    moduli_ok = {k[0] for k, r in nonunit.items() if clock(r)}
    for k, r in sorted(nonunit.items()):
        print(f"    n={k[0]} J{k[1]}xJ{k[2]}: "
              f"G0 {'ok' if r['G0'] else 'FAIL'}  clock {'YES' if clock(r) else 'NO'}  "
              f"(G1 {r['G1']}, G2 {r['G2']}, G3 {r['G3']}, G4 {r['G4']}, G5 {r['G5']})")
    print()
    if bad_g0 and len(bad_g0) == len(nonunit):
        print("  --> NO VERDICT. The non-unit strata are not generalised; the mechanism "
              "question is unanswerable from this data and Gate 2 stays open.")
    elif no_clock:
        print(f"  --> BRANCH (a): circuits genuinely DIFFER within a modulus. "
              f"{len(no_clock)} generalised stratum/strata carry no local clock: {no_clock}. "
              "Report as branch (a), NOT smoothed into the fourth branch.")
    elif len(moduli_ok) >= 2:
        print("  --> FOURTH BRANCH DECLARED: the local mechanism is invariant; the "
              "stratification is not.")
        print(f"      Every measurable non-unit stratum carries a local clock, at "
              f"{len(moduli_ok)} moduli {sorted(moduli_ok)}, and G6 shows the stratum sets "
              "differ by the algebra of n.")
    else:
        print(f"  --> INSUFFICIENT: local clocks found at only {len(moduli_ok)} modulus. "
              "The rule requires >= 2.")


def main():
    # Constraint 0: a number with no code version attached is not usable in a thesis, and
    # that applies to a log as much as to an .npz.
    prov = provenance.stamp(config=dict(results_dir=D, controls="--controls" in sys.argv,
                                        G_MIN=G_MIN, HELDOUT_MIN=HELDOUT_MIN,
                                        N_CONTROL=N_CONTROL, TRAIN_FRAC=TRAIN_FRAC,
                                        thresholds=dict(G0=G0_ACC, G1=G1_P, G2=G2_EXCL,
                                                        G3=G3_RATIO, G4=G4_R, G5=G5_MARGIN,
                                                        tune=TUNE_THRESH, seeds=SEED_MIN)))
    print(f"G2 -- Gate 2 stratum analysis · {D}")
    print(f"criteria: experiments/PREREGISTER_gate2_strata.md (committed bca6236)")
    print(f"git_sha={prov['git_sha']} git_dirty={prov['git_dirty']} "
          f"utc={prov['timestamp_utc']} argv={prov['argv']}")
    lattice(sorted({n for n, _ in discover(D)}) or MODULI)
    res = []
    for n, s in discover(D):
        if True:
            f = path(D, n, s)
            if f is None:
                continue
            r = measure(f, n, s)
            if "skip" in r:
                print(f"\n  n={n} seed={s}: SKIPPED -- {r['skip']}")
                continue
            res.append(r)
            run_table(r)
    if res:
        v = score(res, D)
        gate2_verdict(v)
        out = os.path.join("results", "gate2",
                           os.path.basename(D.rstrip("/")) +
                           ("_controls" if "--controls" in sys.argv else "") + ".json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as fh:
            json.dump(dict(provenance=prov,
                           runs=[dict(n=r["n"], seed=r["seed"],
                                      split_recomputed=r["split_ok"][0],
                                      split_logged=r["split_ok"][1],
                                      strata=r["rows"]) for r in res],
                           verdict={f"{k[0]}_J{k[1]}xJ{k[2]}": val
                                    for k, val in v.items()}), fh, indent=1, default=str)
        print(f"\n  saved {out}  (provenance stamped; figures regenerate from this, "
              f"never from the log)")
    else:
        print("\n  no runs measured")
    return res


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck(); sys.exit(0)
    _selfcheck()
    main()
