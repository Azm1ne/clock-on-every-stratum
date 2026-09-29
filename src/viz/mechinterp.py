"""Render the model's internals. Every panel here exists because a paper in this field
found its mechanism by LOOKING at this specific view.

- neuron activation heatmaps h(a,b)          Nanda Fig 4 (centre) -- how the Clock was found
- the same, reordered by discrete log        2606.17399 -- how the Discrete-Log Clock was found
- 2D DFT of the logits                       Nanda Fig 4 (right) -- the 5-block structure
- embedding PCA coloured by algebra          2607.07066 -- J-class organisation
- per-neuron frequency assignment            Nanda Fig 5 -- single-frequency tuning

Scalars (Gini, participation ratio) say a structure EXISTS. These say what it IS.
"""
import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from math import gcd

from src.analysis.neurons import freq_fraction

FIG = "figures"


def _grid(x, n):
    """(cells, neurons) -> (side, side, neurons), side inferred from the data.

    The units-only replication arm trains on the 112 units of Z/113Z, so its grid is
    112x112 while `n` is 113 -- reshaping by the filename modulus raises. Infer the side.
    """
    x = np.asarray(x)
    side = math.isqrt(x.shape[0])
    if side * side != x.shape[0]:
        raise ValueError(f"activation grid of {x.shape[0]} cells is not square")
    return x.reshape(side, side, -1)


def _dlog_order(n):
    """Elements ordered by discrete log. Returns (order, labels) or (None, None) when the
    unit group is non-cyclic -- then use the CRT/product coordinate instead."""
    from src.analysis.transforms import unit_index
    idx, orders, _ = unit_index(n)
    if len(orders) != 1:
        return None, orders
    return np.array(sorted(idx, key=lambda x: idx[x])), orders


def _reorder_dlog(H, n, order):
    """Reorder a grid into discrete-log coordinates.

    _dlog_order returns RESIDUES; the grid is indexed by POSITION, and on the units-only
    arm position i holds the i-th unit, not the residue i.
    """
    vals = (np.arange(n) if H.shape[0] == n
            else np.array([x for x in range(n) if gcd(x, n) == 1]))
    pos = {v: i for i, v in enumerate(vals)}
    idx = np.array([pos[x] for x in order if x in pos])
    return H[np.ix_(idx, idx)]


def neuron_heatmaps(npz, n, k=4, dlog=False, crop=32, out=None):
    """h_n(a,b) for the k highest-variance neurons: full view, zoomed crop, and 2D DFT.

    THE diagnostic view -- how the Clock and the Discrete-Log Clock were both found.
    The crop matters: a frequency-33 neuron on a 113 grid has a ~3-pixel period, which
    aliases into meaningless texture at full extent. The DFT panel makes the frequency
    content unambiguous regardless.
    """
    z = np.load(npz, allow_pickle=False)
    H = _grid(z["mlp_acts"], n)
    if dlog:
        order, orders = _dlog_order(n)
        if order is None:
            return None
        H = _reorder_dlog(H, n, order)
    v = H.reshape(-1, H.shape[-1]).var(0)
    top = np.argsort(-v)[:k]
    fig, axes = plt.subplots(k, 3, figsize=(9.5, 3.0 * k))
    axes = np.atleast_2d(axes)
    for row, j in enumerate(top):
        h = H[:, :, j]
        lim = np.abs(h).max()
        axes[row, 0].imshow(h, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
        axes[row, 0].set_ylabel(f"neuron {j}", fontsize=9)
        axes[row, 0].set_title("full  h(a,b)" if row == 0 else "", fontsize=9)
        c = h[:crop, :crop]
        axes[row, 1].imshow(c, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
        axes[row, 1].set_title(f"zoom {crop}x{crop}" if row == 0 else "", fontsize=9)
        F = np.fft.fftshift(np.abs(np.fft.fft2(h - h.mean())))
        axes[row, 2].imshow(np.log10(F + 1e-9), cmap="magma")
        axes[row, 2].set_title("log|2D DFT|" if row == 0 else "", fontsize=9)
        for a in axes[row]:
            a.set_xticks([]); a.set_yticks([])
    lab = "reordered by discrete log" if dlog else "natural order"
    fig.suptitle(f"n={n}: MLP neuron activations, {lab}", fontsize=11)
    out = out or f"{FIG}/neurons_n{n}{'_dlog' if dlog else ''}.png"
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return out


def logit_spectrum(npz, n, out=None):
    """2D DFT of the logits over (a,b). Nanda: 20 significant components = 5 key
    frequencies x 4, appearing as 2x2 blocks."""
    z = np.load(npz, allow_pickle=False)
    L = _grid(z["logits_all"], n).astype(float)
    L -= L.mean((0, 1), keepdims=True)
    F = np.linalg.norm(np.fft.fft2(L, axes=(0, 1)), axis=-1)
    F = np.fft.fftshift(F)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
    # Clip to the top 3 decades. Without this the log scale spans the whole dynamic range
    # and every cell lands mid-palette, hiding the block structure entirely.
    L10 = np.log10(F + 1e-9)
    im = a1.imshow(L10, cmap="magma", vmin=L10.max() - 3, vmax=L10.max())
    a1.set_title("log|2D DFT of logits|  (norm over output axis)", fontsize=10)
    a1.set_xlabel("frequency in b"); a1.set_ylabel("frequency in a")
    plt.colorbar(im, ax=a1, fraction=0.046)
    flat = np.sort(F.ravel())[::-1]
    a2.semilogy(flat[:80], "o-", ms=3, lw=0.8)
    a2.set_title(f"top 80 components, sorted  "
                 f"({(F > 0.1 * F.max()).sum()} above 10% of max)", fontsize=10)
    a2.set_xlabel("rank"); a2.set_ylabel("norm")
    a2.grid(alpha=0.3); a2.spines[["top", "right"]].set_visible(False)
    out = out or f"{FIG}/logit_spectrum_n{n}.png"
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    return out


def embedding_structure(npz, n, out=None):
    """PCA of W_E coloured by algebraic role. 2607.07066's central claim is that the
    embedding organises by J-class; this is that claim, rendered."""
    z = np.load(npz, allow_pickle=False)
    W = z["W_E"][:n].astype(float)
    W -= W.mean(0, keepdims=True)
    U, S, _ = np.linalg.svd(W, full_matrices=False)
    P = U[:, :4] * S[:4]
    g = np.array([gcd(x, n) for x in range(n)])
    classes = sorted(set(g))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4)) 
    for ax, (i, j) in zip(axes[:2], [(0, 1), (2, 3)]):
        for c in classes:
            m = g == c
            ax.scatter(P[m, i], P[m, j], s=26 if c > 1 else 9,
                       label=f"J_{c} ({m.sum()})", alpha=0.85,
                       edgecolors="k" if c > 1 else "none", linewidths=0.4)
        ax.set_xlabel(f"PC{i+1}"); ax.set_ylabel(f"PC{j+1}")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_title("PC1–PC2, coloured by 𝒥-class", fontsize=10)
    axes[1].set_title("PC3–PC4", fontsize=10)
    if len(classes) <= 10:
        axes[0].legend(fontsize=7, frameon=False)
    axes[2].plot(S[:40] / S[0], "o-", ms=3, lw=0.9)
    axes[2].set_title("singular values of W_E (normalised)", fontsize=10)
    axes[2].set_xlabel("index"); axes[2].grid(alpha=0.3)
    axes[2].spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"n={n}: embedding structure", fontsize=11)
    out = out or f"{FIG}/embedding_n{n}.png"
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    return out


def neuron_frequency_map(npz, n, dlog=False, out=None):
    """Per-neuron: which single frequency explains it, and how well. Nanda Fig 5."""
    z = np.load(npz, allow_pickle=False)
    H = _grid(z["mlp_acts"], n).astype(float)
    if dlog:
        order, orders = _dlog_order(n)
        if order is None:
            print(f"  n={n}: unit group is non-cyclic {orders}; no discrete-log coordinate")
            return None
        H = _reorder_dlog(H, n, order)
    s = H.shape[0]                       # grid side; != n on the units-only arm
    # The statistic lives in src/analysis/neurons.py so that this picture and the number
    # analyze_o4.py reports are the same computation, not two implementations of it.
    frac, best_k = freq_fraction(H)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4))
    a1.hist(frac, bins=40, color="#4C72B0")
    a1.axvline(0.85, color="crimson", ls="--", lw=1)
    a1.set_title(f"variance explained by best single frequency ({'discrete-log' if dlog else 'raw integer'} coords)\n"
                 f"{(frac>0.85).mean():.1%} above 0.85   (a+b, Nanda: 84.6%; a*b in dlog, 2606.17399: 96.9%)",
                 fontsize=8)
    a1.set_xlabel("fraction"); a1.spines[["top", "right"]].set_visible(False)
    sel = frac > 0.85
    a2.hist(best_k[sel], bins=np.arange(1, s // 2 + 2) - 0.5, color="#C44E52")
    a2.set_title(f"frequency assignment of the {sel.sum()} tuned neurons", fontsize=9)
    a2.set_xlabel("frequency"); a2.spines[["top", "right"]].set_visible(False)
    out = out or f"{FIG}/neuron_freqs_n{n}{'_dlog' if dlog else ''}.png"
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    return out, frac, best_k


def report(npz, n):
    """Every panel, plus the numbers, for one checkpoint."""
    made = []
    for f in (embedding_structure, logit_spectrum):
        try:
            made.append(f(npz, n))
        except Exception as e:
            made.append(f"{f.__name__}: unavailable ({e})")
    for dl in (False, True):
        try:
            r = neuron_heatmaps(npz, n, dlog=dl)
            made.append(r or f"neuron_heatmaps(dlog={dl}): non-cyclic unit group, skipped")
        except Exception as e:
            made.append(f"neuron_heatmaps(dlog={dl}): unavailable ({e})")
    # BOTH coordinates, as neuron_heatmaps above already does. On a MULTIPLICATION run the
    # raw-integer panel is the uninformative one -- it reads 0.0% tuned at every modulus and
    # every seed (C31) -- so rendering only it meant the "render before concluding" invariant
    # was satisfied by a picture of nothing for every run in the project. The dlog panel is
    # where the 74-100% lives.
    for dl in (False, True):
        try:
            r = neuron_frequency_map(npz, n, dlog=dl)
            made.append(r[0] if r else
                        f"neuron_frequency_map(dlog={dl}): non-cyclic unit group, skipped")
        except Exception as e:
            made.append(f"neuron_frequency_map(dlog={dl}): unavailable ({e})")
    return made


def neuron_term_decomposition(npz, n, verbose=True, dlog=False):
    """Split each neuron's spectral energy into the terms a circuit could be computing.

    Within one frequency k, the 2D DFT bins say WHICH function of (a,b) the neuron holds:
      (k,0),(0,k)   -> depends on a alone / b alone      (pre-composition, "input" terms)
      (k, k)        -> a function of (a + b)             <- the Clock's output term
      (k,-k)        -> a function of (a - b)
    Nanda's neurons compute cos(w(a+b)) via trig identities, so a grokked additive circuit
    should carry substantial (k,-k) energy. A neuron holding only axis terms has not yet
    composed its inputs.

    Sign convention verified empirically in test_mechinterp.py, NOT by reasoning -- the
    first version of this function had (a+b) and (a-b) swapped, which inverts the
    conclusion. cos(2*pi*k*(a+b)/n) peaks at (k,k) and (-k,-k); cos(...(a-b)...) at (k,-k).
    """
    z = np.load(npz, allow_pickle=False)
    H = _grid(z["mlp_acts"], n).astype(float)
    if dlog:
        order, orders = _dlog_order(n)
        if order is None:
            print(f"  n={n}: unit group is non-cyclic {orders}; no discrete-log coordinate")
            return None
        H = _reorder_dlog(H, n, order)
    s = H.shape[0]                       # grid side; != n on the units-only arm
    H -= H.mean((0, 1), keepdims=True)
    F = np.abs(np.fft.fft2(H, axes=(0, 1))) ** 2
    tot = F.sum((0, 1))

    terms = {"a only": 0.0, "b only": 0.0, "a+b": 0.0, "a-b": 0.0}
    per_neuron = {k: np.zeros(F.shape[-1]) for k in terms}
    for k in range(1, s // 2 + 1):
        p, q = k % s, (-k) % s
        per_neuron["a only"] += F[p, 0] + F[q, 0]
        per_neuron["b only"] += F[0, p] + F[0, q]
        per_neuron["a+b"] += F[p, p] + F[q, q]      # diagonal: f(a+b)
        per_neuron["a-b"] += F[p, q] + F[q, p]      # anti-diagonal: f(a-b)
    frac = {k: v / np.maximum(tot, 1e-30) for k, v in per_neuron.items()}
    if verbose:
        coord = "DISCRETE-LOG coordinate (a*b -> alpha+beta)" if dlog else "RAW residue coordinate"
        print(f"  n={n}: neuron energy by term (mean over {F.shape[-1]} neurons)")
        print(f"        basis: {coord}")
        if not dlog:
            print("        WARNING: on a MULTIPLICATION run these term names are "
                  "meaningless -- pass dlog=True.")
        for k in ["a only", "b only", "a-b", "a+b"]:
            print(f"     {k:<8} {frac[k].mean():6.1%}   "
                  f"(neurons where it dominates: {(np.argmax(np.stack([frac[t] for t in terms]),0)==list(terms).index(k)).sum():>4})")
        resid = 1 - sum(frac[k] for k in terms)
        print(f"     {'other':<8} {resid.mean():6.1%}   (cross-frequency / higher order)")
    return frac


def logit_top_components(npz, n, top=16, verbose=True, dlog=False):
    """The dominant logit frequency components, named. This is the definitive readout of
    what the output layer computes -- eyeballing the heatmap is not a substitute."""
    z = np.load(npz, allow_pickle=False)
    L = _grid(z["logits_all"], n).astype(float)
    if dlog:
        order, orders = _dlog_order(n)
        if order is None:
            print(f"  n={n}: unit group is non-cyclic {orders}; no discrete-log coordinate")
            return None
        L = _reorder_dlog(L, n, order)
    s = L.shape[0]                       # grid side; != n on the units-only arm
    L -= L.mean((0, 1), keepdims=True)
    F = np.linalg.norm(np.fft.fft2(L, axes=(0, 1)), axis=-1)
    flat = np.argsort(-F.ravel())[:top]
    rows = []
    for i in flat:
        ka, kb = np.unravel_index(i, F.shape)
        sa, sb = (ka + s // 2) % s - s // 2, (kb + s // 2) % s - s // 2   # signed
        if sa == 0 and sb == 0:
            kind = "DC"
        elif sb == 0:
            kind = "a only"
        elif sa == 0:
            kind = "b only"
        elif sa == sb:
            kind = "a+b"
        elif sa == -sb:
            kind = "a-b"
        else:
            kind = "mixed"
        rows.append((sa, sb, F[ka, kb], kind))
    if verbose:
        coord = "DISCRETE-LOG coordinate (a*b -> alpha+beta)" if dlog else "RAW residue coordinate"
        print(f"  n={n}: top {top} logit DFT components   [basis: {coord}]")
        if not dlog:
            print("        WARNING: on a MULTIPLICATION run these term names are "
                  "meaningless -- pass dlog=True.")
        print(f"     {'(f_a, f_b)':>12}  {'norm':>10}  {'% of max':>8}  term")
        mx = rows[0][2]
        for sa, sb, v, kind in rows:
            print(f"     {f'({sa:+d}, {sb:+d})':>12}  {v:10.3e}  {v/mx:>7.1%}  {kind}")
        kinds = {}
        for _, _, v, k in rows:
            kinds[k] = kinds.get(k, 0) + v ** 2
        tot = sum(kinds.values())
        print("     energy share among these: " +
              ", ".join(f"{k} {v/tot:.0%}" for k, v in
                        sorted(kinds.items(), key=lambda x: -x[1])))
    return rows


def jclass_blocked_view(npz, n, k=4, out=None, verbose=True):
    """h(a,b) with BOTH axes ordered by J-class, then by the unit-action coordinate.

    The discrete-log view drops every non-unit, which at a prime power is exactly the
    thesis question. Here the whole monoid is present and the block structure is explicit:

        J_1 x J_1   -> the unit group; a clock lives here
        J_1 x J_d   -> units acting on a zero-divisor stratum
        J_d x J_d   -> at a prime power this composes to constant 0 -- but note it is
                       NOT featureless: at n=121, J_11 x J_11 carries variance 2.380,
                       4x the equally-sized, also-constant-zero J_1 x J_121

    Block boundaries are drawn, so "does the network treat the strata differently" becomes
    a question you answer by looking.
    """
    from math import gcd
    from src.analysis.transforms import unit_index
    z = np.load(npz, allow_pickle=False)
    H = _grid(z["mlp_acts"], n)
    if H.shape[0] != n:
        print(f"  J-class blocking needs the full {n}x{n} grid; this run is "
              f"{H.shape[0]}x{H.shape[0]} (units only). SKIPPED.")
        return None

    order, bounds, labels = [], [], []
    for d in sorted({gcd(x, n) for x in range(n)}):
        members = [x for x in range(n) if gcd(x, n) == d]
        m = n // d
        if m > 1:
            try:
                idx, _, _ = unit_index(m)
                members.sort(key=lambda x: idx.get((x // d) % m, 0))
            except Exception:
                pass
        order += members
        bounds.append(len(order))
        labels.append((len(order) - len(members) / 2, f"J_{d}", len(members)))
    order = np.array(order)
    Hb = H[np.ix_(order, order)]
    v = Hb.reshape(-1, Hb.shape[-1]).var(0)
    top = np.argsort(-v)[:k]

    fig, axes = plt.subplots(1, k, figsize=(3.1 * k, 3.5))
    for ax, j in zip(np.atleast_1d(axes), top):
        h = Hb[:, :, j]; lim = np.abs(h).max()
        ax.imshow(h, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
        for b in bounds[:-1]:
            ax.axhline(b - 0.5, color="k", lw=0.8)
            ax.axvline(b - 0.5, color="k", lw=0.8)
        ax.set_title(f"neuron {j}", fontsize=9)
        ax.set_xticks([p for p, _, _ in labels]); ax.set_yticks([p for p, _, _ in labels])
        ax.set_xticklabels([l for _, l, _ in labels], fontsize=6, rotation=90)
        ax.set_yticklabels([l for _, l, _ in labels], fontsize=6)
    fig.suptitle(f"n={n}: neurons with axes ordered by 𝒥-class "
                 f"({', '.join(f'{l}:{c}' for _, l, c in labels)})", fontsize=10)
    out = out or f"{FIG}/jclass_blocks_n{n}.png"
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)

    # Quantitative: VARIANCE per block, not mean magnitude. A block whose output is a
    # constant (J_d x J_d at a prime power composes to 0) still has a large mean |h| while
    # carrying no information; only the variance distinguishes computation from saturation.
    if verbose:
        starts = [0] + bounds[:-1]
        V = Hb.var(-1) if False else Hb
        print(f"  n={n}: per-𝒥-class-block activation VARIANCE across inputs, "
              f"averaged over {H.shape[-1]} neurons")
        print("        " + "".join(f"{l:>9}" for _, l, _ in labels))
        for i, (s0, e0) in enumerate(zip(starts, bounds)):
            row = ""
            for s1, e1 in zip(starts, bounds):
                blk = Hb[s0:e0, s1:e1, :]                    # (rows, cols, neurons)
                # variance over the (a,b) cells within the block, then mean over neurons
                row += f"{blk.reshape(-1, blk.shape[-1]).var(0).mean():9.3f}"
            print(f"  {labels[i][1]:>6}" + row)
        print(f"  (J_d x J_d at a prime power composes to a constant 0 -- low variance "
              f"there means the network has learned it needs no computation)")
    return out
