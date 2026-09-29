"""Figures generated from saved artifacts, so they regenerate rather than go stale.

Every figure takes a path to a result file and reads its provenance stamp, so the caption
can state which code version produced it. One style block below sets the look of all of
them; a figure function sets only what is particular to it.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.provenance import read as read_provenance

FIGDIR = "figures"

# --- style: one block for every figure --------------------------------------------------
# Every paper figure is drawn at print size: as wide as it is set in the paper (TMLR
# \textwidth = 6.5 in, times its \includegraphics fraction), so a font size here is the
# printed size, and 7 pt is the floor.
# Saved as vector PDF with TrueType fonts (fonttype 42): no Type 3 fonts, sharp at any zoom.
TEXTWIDTH = 6.5
FS_TITLE, FS, FS_SMALL = 9, 7.5, 7
plt.rcParams.update({
    "font.family": "sans-serif",
    # Liberation Sans: Helvetica's metrics in TrueType outlines. Nimbus Sans and TeX Gyre
    # Heros are CFF, which fonttype 42 embeds inside a TrueType wrapper (poppler: "Mismatch
    # between font type and embedded font file").
    "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
    "font.size": FS, "axes.titlesize": FS_TITLE, "axes.labelsize": FS,
    "xtick.labelsize": FS, "ytick.labelsize": FS, "legend.fontsize": FS,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
})
# Categorical hues, fixed order, never cycled: blue, red, teal, purple. Validated with the
# dataviz validator on the light surface: every check PASSES in slot order (worst adjacent
# CVD dE 8.5 deutan, teal-purple; normal-vision floor 23.0), and in the order the ablation
# figures put side by side, blue-teal-red (normal-vision dE 16.2). Blue and purple are
# never adjacent: that pair fails for protanopes (dE 5.1). The plan's #3775BA and #42949E
# failed (blue-teal normal-vision dE 12; teal below the chroma floor), so both moved one
# step along their own hue. Neutrals are for what should recede.
BLUE, RED, TEAL, PURPLE = "#2A64AD", "#B64342", "#1B9AA6", "#9A4D8E"
INK, DARK, MUTE, FAINT = "#272727", "#4D4D4D", "#767676", "#CFCECE"
# Grouped bars carry black edges and a hatch per hue, because blue and red print at almost
# the same grey: a bar's role must survive a black-and-white printer.
HATCH = {BLUE: "", TEAL: "////", RED: "\\\\\\"}
# A white halo keeps a label legible where it has to cross data.
from matplotlib import patheffects
HALO = [patheffects.withStroke(linewidth=2.2, foreground="white")]


def _save(fig, out, frac=1.0):
    """Save at exactly the width the paper sets the figure (`frac` of \\textwidth).

    The tight bounding box is measured and the figure widened or narrowed until the two
    agree, so the PDF is never scaled on the page and every font size above is the
    printed size. Text is fixed in inches, so this converges in a step or two."""
    pad, w0 = plt.rcParams["savefig.pad_inches"], fig.get_figwidth()
    for _ in range(5):
        fig.canvas.draw()
        w = fig.get_tightbbox(fig.canvas.get_renderer()).width + 2 * pad
        if abs(w - frac * TEXTWIDTH) < 0.002:
            break
        fig.set_size_inches(fig.get_figwidth() + frac * TEXTWIDTH - w, fig.get_figheight())
    # A line of text wider than the page makes the loop shrink the axes to nothing, which
    # still "converges". Refuse it: the fix is a line break, not a smaller figure.
    assert fig.get_figwidth() > 0.85 * w0 and abs(w - frac * TEXTWIDTH) < 0.01, (
        f"{out}: some text is wider than {frac * TEXTWIDTH:.2f} in; wrap it")
    fig.savefig(out, bbox_inches="tight", pad_inches=pad)
    plt.close(fig)


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=FS_TITLE)
    ax.set_xlabel(xlabel, fontsize=FS)
    ax.set_ylabel(ylabel, fontsize=FS)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=FS)
    ax.grid(alpha=0.25, linewidth=0.5)


def _caption(fig, path):
    p = read_provenance(path)
    if p:
        fig.text(0.005, 0.005, f"{path} · {p.get('git_sha','?')[:8]} · "
                 f"{p.get('timestamp_utc','?')}", fontsize=FS_SMALL, color="0.45")


def training_curves(npz_path, out=f"{FIGDIR}/training_curves.png"):
    """Accuracy and loss against step, with the grokking point marked."""
    z = np.load(npz_path, allow_pickle=False)
    h, cols = z["hist"], [str(c) for c in z["hist_cols"]]
    # Kernels differ in what they log: k02 saved no hist_cols at all, k03/k04 log
    # test_acc but not train_acc. Render what is there and SAY what is not, rather
    # than failing the whole panel -- a missing curve is a gap in what was saved.
    col = lambda k: h[:, cols.index(k)] if k in cols else None
    step = col("step")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))
    for name in ("train_acc", "test_acc"):
        v = col(name)
        if v is not None:
            a1.plot(step, v, label=name.split("_")[0], lw=1.4)
    missing = [k for k in ("train_acc", "test_acc", "train_loss", "test_loss")
               if col(k) is None]
    if missing:
        a1.text(0.02, 0.04, "not logged: " + ", ".join(missing), transform=a1.transAxes,
                fontsize=FS_SMALL, color="crimson")
    _style(a1, "Accuracy", "step", "accuracy"); a1.legend(fontsize=FS, frameon=False)
    for name in ("train_loss", "test_loss"):
        v = col(name)
        if v is not None:
            a2.semilogy(step, np.maximum(v, 1e-12), label=name.split("_")[0], lw=1.4)
    _style(a2, "Loss (log)", "step", "cross-entropy"); a2.legend(fontsize=FS, frameon=False)
    te = col("test_acc")
    if te is not None and (te > 0.99).any():
        g = step[np.argmax(te > 0.99)]
        for a in (a1, a2):
            a.axvline(g, color="crimson", ls="--", lw=0.9)
        a1.annotate(f"grok @ {int(g)}", (g, 0.5), fontsize=FS, color="crimson",
                    rotation=90, va="center", ha="right")
    _caption(fig, npz_path)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)
    return out


def basis_comparison(npz_path, n, out=f"{FIGDIR}/basis_comparison.pdf"):
    """Embedding spectrum in the additive vs multiplicative basis -- the core figure."""
    from src.analysis.transforms import unit_index
    z = np.load(npz_path, allow_pickle=False)
    W = z["W_E"][:n].astype(float); W -= W.mean(0, keepdims=True)
    idx, orders, _ = unit_index(n)
    U = np.array(sorted(idx, key=lambda x: idx[x]))

    def norms(M, m):
        k = np.arange(1, m // 2 + 1)[:, None]; t = np.arange(m)[None, :]
        s = np.sin(2 * np.pi * k * t / m) @ M; c = np.cos(2 * np.pi * k * t / m) @ M
        return np.sqrt((s ** 2).sum(1) + (c ** 2).sum(1))

    add, mult = norms(W, n), norms(W[U], len(U))
    from src.analysis.sparsity import gini
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(0.78 * TEXTWIDTH, 2.1), sharey=True)
    a1.bar(np.arange(1, len(add) + 1), add, width=1.0, color=BLUE)
    _style(a1, f"Additive basis  (Gini {gini(add):.3f})", "frequency", "norm")
    a2.bar(np.arange(1, len(mult) + 1), mult, width=1.0, color=RED)
    _style(a2, f"Multiplicative basis  (Gini {gini(mult):.3f})", "character", "")
    fig.suptitle(f"n = {n}: sparsity is basis-dependent", fontsize=FS_TITLE)
    _caption(fig, npz_path)
    fig.tight_layout(); _save(fig, out, 0.78)
    return out


def ablation_bars(per_freq, key_freqs, out=f"{FIGDIR}/ablation.png"):
    """Per-frequency loss delta; key frequencies highlighted. Nanda Figure 6."""
    ks = sorted(per_freq)
    v = np.array([per_freq[k] for k in ks])
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    ax.bar(ks, np.maximum(v, 1e-12),
           color=["#C44E52" if k in key_freqs else "#B0B0B0" for k in ks], width=1.0)
    ax.set_yscale("log")
    _style(ax, "Loss increase when ablating each frequency", "frequency", "Δ loss (log)")
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)
    return out


def gate2_strata(json_path="results/gate2/k03_grid_acts.json",
                 out=f"{FIGDIR}/gate2_strata.pdf"):
    """Per-stratum measurement against its OWN control, grouped by modulus (C33).

    FORM. The job is "a measurement against its control, per stratum" -- magnitude with two
    identities -- so this is a paired dot plot, one row per stratum, two marks joined by a
    rule. Not a bar chart: the quantity of interest is the GAP between the pair, and a bar
    encodes distance-from-zero instead.

    COLOUR. Two identities only, measured and control: blue for measured, the neutral for
    the control, which is *supposed* to recede. Marker SHAPE differs as well (filled vs
    open), so identity is never colour-alone.

    Panel B is a residual: LOWER is better, and the control sits to the right.
    """
    import collections
    d = json.load(open(json_path))
    by = collections.defaultdict(list)
    for run in d["runs"]:
        for s in run["strata"]:
            if s["measurable"]:
                by[(run["n"], s["d"], s["e"])].append(s)

    rows = []
    for (n, dd, e), ss in sorted(by.items(), key=lambda kv: (kv[0][0], -kv[1][0]["cells"])):
        grp = "x".join(f"Z{o}" for o in ss[0]["orders_m"])
        key = "tuned" if ss[0].get("tuned") is not None else "tuned_nd"
        shf = key + ("_sh" if key == "tuned" else "_sh")
        tun = [s[key] for s in ss if s.get(key) is not None]
        tsh = [s[shf] for s in ss if s.get(shf) is not None]
        res = [s["resid"] for s in ss if s.get("resid") is not None]
        rsh = [s["resid_ctrl"] for s in ss if s.get("resid_ctrl") is not None]
        rows.append(dict(label=f"{n}  J{dd}×J{e} → {grp}", n=n,
                         unit=(dd == 1 and e == 1), tun=tun, tsh=tsh, res=res, rsh=rsh))

    MEAS, CTRL = BLUE, MUTE
    y = np.arange(len(rows))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, max(3.0, 0.16 * len(rows) + 1.2)),
                             sharey=True)

    def panel(ax, a_key, b_key, title, xlabel, lo_is_good):
        for i, r in zip(y, rows):
            a, b = r[a_key], r[b_key]
            if not a or not b:
                # A blank row must not read as absence. These strata have fibre size 1 on
                # both axes, so there is nothing to collapse and G3 is n/a, not zero.
                ax.annotate("n/a: no fibre to collapse", xy=(0.02, i), fontsize=FS_SMALL,
                            color="0.6", va="center")
                continue
            ma, mb = float(np.mean(a)), float(np.mean(b))
            ax.plot([ma, mb], [i, i], color="0.75", lw=1.0, zorder=1)
            ax.scatter([mb], [i], s=34, facecolors="none", edgecolors=CTRL, lw=1.4, zorder=2)
            ax.scatter([ma], [i], s=34, color=MEAS, zorder=3)
            if len(a) > 1:
                ax.errorbar([ma], [i], xerr=[np.std(a)], color=MEAS, lw=1.0,
                            capsize=2, zorder=3)
        _style(ax, title, xlabel, "")
        ax.set_xlim(-0.04, 1.04)
        if lo_is_good:
            ax.annotate("lower = the block collapses to the local ring",
                        xy=(0.02, -0.9), fontsize=FS_SMALL, color="0.45")

    panel(axes[0], "tun", "tsh", "G5  single-frequency MLP neurons",
          "share of 512 neurons explained\nby one local character", False)
    panel(axes[1], "res", "rsh", "G3  fibre-collapse residual",
          "share of block variance\nsurviving the collapse", True)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels([r["label"] for r in rows], fontsize=FS_SMALL,
                            fontfamily="monospace")
    for i, r in zip(y, rows):
        if r["unit"]:
            axes[0].get_yticklabels()[list(y).index(i)].set_color("0.35")

    from matplotlib.lines import Line2D
    fig.legend(handles=[
        Line2D([], [], marker="o", ls="", color=MEAS, markersize=6, label="measured"),
        Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=CTRL,
               markersize=6, label="its own control (shuffled neurons / permuted fibres)")],
        fontsize=FS, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=2, frameon=False)
    fig.suptitle("Every algebraic stratum runs the same clock, on its own local group"
                 "\n(rows greyed = the unit stratum, the only one measured before this)",
                 fontsize=FS_TITLE)
    _caption(fig, json_path)
    fig.tight_layout(rect=(0, 0.01, 1, 0.935))
    fig.subplots_adjust(top=0.85)
    _save(fig, out)
    return out


# omega(n) takes slots 1-3 of the palette. Every point is directly labelled with its
# modulus and marker SHAPE carries omega too, so identity is never colour-alone.
OMEGA_COLOR = {1: BLUE, 2: RED, 3: TEAL}
OMEGA_MARKER = {1: "o", 2: "s", 3: "^"}


def omega_bands(out=f"{FIGDIR}/omega_bands.pdf", predict_n=128):
    """Additive sparsity against zero-divisor density, grouped by omega(n).

    The figure has to carry three things at once, because the claim is exactly that the
    third defeats the second:
      * Gini_add separates into non-overlapping bands by omega  (left, y separation)
      * zdd is the confound and does NOT explain it             (left, x position)
      * the residue-axis random-orthogonal control is FLAT      (right, same y scale)

    It also plots the PRE-REGISTERED, UNRESOLVED prediction for n=128 = 2^7, the one
    modulus where omega and zdd disagree (PREREGISTER_k09_primes_zdd.md P1).
    """
    import analyze_omega as A
    d = A.collect()
    if not d:
        print("omega_bands: no grokked runs found -- NOT SAVED"); return None

    fig, (ax, axc) = plt.subplots(1, 2, figsize=(TEXTWIDTH, 3.7), sharey=True,
                                  gridspec_kw={"width_ratios": [2.05, 1]})

    # omega bands as recessive horizontal spans -- the claim, drawn once
    for w in sorted({x["omega"] for x in d}):
        g = [x["ga"] for x in d if x["omega"] == w]
        ax.axhspan(min(g), max(g), color=OMEGA_COLOR[w], alpha=0.07, zorder=0)

    for w in sorted({x["omega"] for x in d}):
        g = [x for x in d if x["omega"] == w]
        ax.scatter([x["zdd"] for x in g], [x["ga"] for x in g], s=40,
                   marker=OMEGA_MARKER[w], color=OMEGA_COLOR[w],
                   edgecolor="white", linewidth=0.7, zorder=3,
                   label=f"$\\omega(n)={w}$  ({len(g)} moduli)")
        axc.scatter([x["zdd"] for x in g], [x["rand"] for x in g], s=40,
                    marker=OMEGA_MARKER[w], color=OMEGA_COLOR[w],
                    edgecolor="white", linewidth=0.7, zorder=3)
        for x in g:                                   # direct labels: the contrast relief
            # Collisions found by LOOKING at the rendered figure, not by assuming:
            # the 125 label sat under the matched-pair arrow and 63/147 sat on top of
            # each other (both at zdd 0.429).
            off = {125: (0, -12), 63: (0, -13), 147: (0, 7), 119: (13, 2),
                   169: (-7, 6), 121: (8, -10), 165: (-8, 6), 105: (7, 6),
                   # the three primes all sit at zdd ~ 0.008 and their labels collided
                   # into unreadable overlap once 127 and 131 landed (k09).
                   113: (14, 8), 127: (0, 8), 131: (14, -1),
                   128: (-16, 2)}
            ax.annotate(str(x["n"]), (x["zdd"], x["ga"]), fontsize=FS_SMALL,
                        xytext=off.get(x["n"], (0, 7)), textcoords="offset points",
                        ha="center", color="0.25")

    # the decisive matched pair: near-identical zdd, 2.4x the sparsity
    p = {x["n"]: x for x in d}
    if 119 in p and 125 in p:
        dx = 0.008   # nudge clear of the point labels
        ax.annotate("", xy=(p[119]["zdd"] + dx, p[119]["ga"]),
                    xytext=(p[125]["zdd"] + dx, p[125]["ga"]),
                    arrowprops=dict(arrowstyle="<->", color="0.35", lw=1.0,
                                    shrinkA=9, shrinkB=9))
        ax.annotate("matched on zdd\n(0.193 vs 0.200)\n2.4$\\times$ apart", fontsize=FS_SMALL,
                    color="0.3", xy=(p[125]["zdd"] + 0.012, (p[119]["ga"] + p[125]["ga"]) / 2),
                    ha="left", va="center")

    # The pre-registered k09 P1 prediction AND its outcome. The two intervals were fixed
    # in PREREGISTER_k09_primes_zdd.md from data that existed BEFORE n=128 was trained:
    # omega=1 said 0.017-0.184, the zdd~0.500 neighbourhood said 0.434-0.589, disjoint.
    # n=128 landed at 0.261 -- BETWEEN them, nearer omega. Drawing the prediction next to
    # the outcome is the honest rendering; drawing only the outcome hides the test.
    if predict_n == 128 and 128 in {x["n"] for x in d}:
        OM_LO, OM_HI, ZD_LO, ZD_HI = 0.017, 0.184, 0.434, 0.589
        obs = [x["ga"] for x in d if x["n"] == 128][0]
        xb = 0.5
        for lo, hi, col, lab, dx, rot in (
                (OM_LO, OM_HI, OMEGA_COLOR[1], "$\\omega$ predicted", -40, 0),
                (ZD_LO, ZD_HI, "0.45", "zdd predicted", 8, 90)):   # upright: 98/100 crowd it
            ax.plot([xb, xb], [lo, hi], color=col, lw=3.4, alpha=0.5,
                    solid_capstyle="butt", zorder=2)
            ax.annotate(lab, (xb, (lo + hi) / 2), fontsize=FS_SMALL, color=col, ha="center",
                        xytext=(dx, 0), textcoords="offset points", va="center", rotation=rot)
        ax.annotate(f"observed {obs:.3f},\nbetween both:\nambiguous",
                    (xb, obs), fontsize=FS_SMALL, color="0.15", ha="left", fontweight="bold",
                    xytext=(14, -22), textcoords="offset points", va="top",
                    arrowprops=dict(arrowstyle="->", color="0.3", lw=0.9, shrinkB=8))

    _style(ax, "Additive sparsity separates by $\\omega(n)$,\nnot by zero-divisor density",
           "zero-divisor density  $1-\\varphi(n)/n$", "Gini$_{\\rm add}$ of $W_E$")
    _style(axc, "Random-orthogonal control\n(residue axis)",
           "zero-divisor density", "")
    # below the axes: at upper left the legend covered four labelled moduli
    ax.legend(fontsize=FS, frameon=False, loc="upper center", ncol=3,
              bbox_to_anchor=(0.5, -0.2), handletextpad=0.2, columnspacing=1.0)
    axc.set_ylim(ax.get_ylim())
    axc.annotate("no band structure:\nthe separation is not\nan artifact of spectrum length",
                 (0.5, 0.97), xycoords="axes fraction", fontsize=FS_SMALL, color="0.35",
                 ha="center", va="top")
    fig.text(0.005, 0.005,
             f"figures/omega_bands.pdf · analyze_omega.py --confirm · "
             f"{len(d)} moduli, {sum(x['runs'] for x in d)} grokked runs ·\n"
             f"per-modulus means · PREREGISTER_omega.md",
             fontsize=FS_SMALL, color="0.45")
    fig.tight_layout(rect=(0, 0.05, 1, 1)); _save(fig, out)
    return out


# --- palette -------------------------------------------------------------------
# Three roles, the same hues in every figure that shows them: baseline, restricted,
# excluded. Every panel direct-labels or legends its marks.
C_BASE, C_REST, C_EXCL = BLUE, TEAL, RED
C_MUTED, C_GRID = MUTE, "#B0B0B0"


def _seed_rows(npz_path):
    rows = [json.loads(str(r)) for r in np.load(npz_path, allow_pickle=True)["rows"]]
    return [r for r in rows if r["tag"].startswith("B_thesis")]


def causal_test(ablation_npz="results/k03_grid_acts_n4_ablation.npz",
                draws_tag="B_thesis_n121_s0", draws_n=121,
                draws_dir="results/k03_grid_acts", out=f"{FIGDIR}/F4_causal.pdf"):
    """F4 -- the causal test (C22, C23, C36) and the permutation null it rests on.

    LEFT   absolute baseline / restricted / excluded loss per modulus, the convention
           2302.03025 uses. Bars are the median over seeds; every seed is drawn as a dot,
           so the spread is visible rather than summarised away.
    RIGHT  the 200-draw permutation null for one run, with the real key set marked. This
           panel is the statistic: `p` is the fraction of the histogram at or beyond the
           marked line. The control is BIMODAL -- most random character removals do
           nothing -- which is exactly why a ratio to its MEAN was retired.

    The draws are recomputed through analyze_n4.run, so the figure and the claim come
    from one implementation of the ablation rather than two.
    """
    import collections
    import analyze_n4

    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in _seed_rows(ablation_npz):
        for k in ("baseline", "restricted", "excluded"):
            per[r["n"]][k].append(r[k])
    ns = sorted(per)

    fig, (ax, axh) = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.6),
                                  gridspec_kw={"width_ratios": [1.45, 1]})

    w, x = 0.26, np.arange(len(ns), dtype=float)
    for off, key, col, lab in ((-w, "baseline", C_BASE, "baseline"),
                               (0.0, "restricted", C_REST, "restricted (keep key only)"),
                               (+w, "excluded", C_EXCL, "excluded (remove key)")):
        med = np.array([np.median(per[n][key]) for n in ns])
        ax.bar(x + off, med, width=w * 0.88, color=col, label=lab, zorder=2,
               edgecolor=INK, linewidth=0.5, hatch=HATCH[col])
        for i, n in enumerate(ns):                       # every seed, not just the median
            v = per[n][key]
            ax.plot(np.full(len(v), x[i] + off), v, ".", ms=3.4, color="0.15",
                    alpha=0.75, zorder=3, clip_on=False)
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels([str(n) for n in ns])
    _style(ax, "Ablating the key characters, per modulus", "modulus n", "cross-entropy (log)")
    ax.legend(fontsize=FS, frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), columnspacing=0.9, handlelength=1.2,
              handletextpad=0.4)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(top=ax.get_ylim()[1] * 6)

    row, draws = analyze_n4.run(draws_tag, draws_n, verbose=False, d=draws_dir,
                                return_draws=True)
    lo = max(min(draws.min(), row["baseline"]) * 0.5, 1e-12)
    bins = np.geomspace(lo, max(draws.max(), row["excluded"]) * 2.0, 34)
    axh.hist(draws, bins=bins, color=C_MUTED, edgecolor="white", linewidth=0.5, zorder=2)
    axh.axvline(np.median(draws), color=C_BASE, lw=2.0, zorder=4)
    axh.axvline(row["excluded"], color=C_EXCL, lw=2.0, zorder=4)
    axh.set_xscale("log")
    axh.set_ylim(top=axh.get_ylim()[1] * 1.38)
    top = axh.get_ylim()[1]
    axh.annotate("median\ncontrol", (np.median(draws), top * 0.58), fontsize=FS_SMALL,
                 color=C_BASE, ha="left", va="center",
                 xytext=(7, 0), textcoords="offset points")
    axh.annotate(f"key set\n{row['excluded']:.3g}", (row["excluded"], top * 0.58),
                 fontsize=FS_SMALL, color=C_EXCL, ha="right", va="center",
                 xytext=(-7, 0), textcoords="offset points")
    # P4: the title hard-coded a 200-draw count and the line below printed b/B, 0.0000, after
    # R1 moved the null to 10,000 draws; both are now read from the draws themselves.
    _style(axh, f"Permutation null, n={draws_n} seed 0\n({len(draws):,} draws)",
           "excluded loss of a random equal-sized set (log)", "draws")
    n_ge = int((draws >= row["excluded"]).sum())
    from src.analysis.stats import perm_p
    mant, ex = f"{perm_p(n_ge, len(draws)):.1e}".split("e")
    axh.text(0.5, 0.97, rf"$\hat{{p}} = {mant}\times10^{{{int(ex)}}}$" "\n"
             f"({n_ge} of {len(draws)} draws reach the key set)",
             transform=axh.transAxes, fontsize=FS, va="top", ha="center", color="0.25",
             path_effects=HALO, zorder=5)

    _caption(fig, ablation_npz)
    fig.tight_layout(rect=(0, 0.02, 1, 1)); _save(fig, out)
    return out


def internal_intervention(res_dir="results/i1_internal", n=113, seeds=(0, 1, 2),
                          out=f"{FIGDIR}/F10_internal.pdf"):
    """F10 -- C39. The intervention on W_E, and why the null's tail is benign.

    LEFT   baseline / restricted / excluded loss per seed, the same three roles and the
           same three hues as F4, so the eye carries the meaning across figures. Excluded
           is direct-labelled with its ACCURACY, because that is the claim: 0.0089 is
           chance at n = 113, not "degraded".
    RIGHT  the null decomposed by how many key characters a draw happens to remove. This
           panel exists because the null's upper tail REACHES the effect -- on seed 2 the
           control max exceeds excluded -- and a ratio to the control median would hide
           that. Plotted as loss/baseline so the three seeds, whose baselines differ by
           two orders, share one axis; that ratio is the statistic the paper reports.

    ASSERTS the published numbers before it draws anything. A figure that can be silently
    wrong needs a control that cannot: if the artifacts ever stop saying what Section 7
    says, this raises instead of drawing a plausible picture of the wrong thing.
    """
    import os
    paths = [f"{res_dir}/I1_engine_n{n}_s{s}.npz" for s in seeds]
    for q in paths:
        assert os.path.exists(q), f"{q} missing -- run run_intervention.py first"
    Z = [np.load(q, allow_pickle=True) for q in paths]

    base = np.array([float(z["internal_baseline"]) for z in Z])
    rest = np.array([float(z["internal_restricted"]) for z in Z])
    excl = np.array([float(z["internal_excluded"]) for z in Z])
    acc_r = np.array([float(z["acc_restricted"]) for z in Z])
    acc_e = np.array([float(z["acc_excluded"]) for z in Z])
    ratio = excl / base

    # --- the control, asserted against Section 7 -------------------------------------
    assert abs(np.median(ratio) - 1.543e+08) < 1.0e+06, np.median(ratio)
    assert (acc_r > 0.999).all(), acc_r
    assert (abs(acc_e - 1.0 / 112) < 5e-4).all(), acc_e      # chance, units-only grid
    assert (excl / base > 100.0).all()                       # G2's inherited bar

    fig, (ax, axd) = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.7),
                                  gridspec_kw={"width_ratios": [1.0, 1.25]})

    w, x = 0.26, np.arange(len(seeds), dtype=float)
    # Short labels so the legend fits ONE row under a narrow panel; the caption carries
    # what each arm means. Three rows of legend made the two panels visibly unbalanced.
    for off, vals, col, lab in ((-w, base, C_BASE, "baseline"),
                                (0.0, rest, C_REST, "restricted"),
                                (+w, excl, C_EXCL, "excluded")):
        ax.bar(x + off, vals, width=w * 0.88, color=col, label=lab, zorder=2,
               edgecolor=INK, linewidth=0.5, hatch=HATCH[col])
    for i in range(len(seeds)):
        ax.annotate(f"acc {acc_e[i]:.4f}", (x[i] + w, excl[i]), fontsize=FS,
                    color=C_EXCL, ha="center", va="bottom",
                    xytext=(0, 3), textcoords="offset points")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylim(top=ax.get_ylim()[1] * 8)
    _style(ax, f"Editing $W_E$, then running forward (n={n})", "", "cross-entropy (log)")
    ax.tick_params(labelsize=FS)
    # Legend BELOW the axes: at `upper left` it sat on top of the acc labels, which is
    # invisible in the code and obvious in the PNG.
    ax.legend(fontsize=FS, frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.06), columnspacing=1.4, handlelength=1.3)
    ax.grid(axis="x", visible=False)

    # --- right: null damage as a dose-response in key-character overlap --------------
    from src.analysis.intervention import random_character_sets
    ov_all, rel_all = [], []
    for s, z in zip(seeds, Z):
        kf = [k[0] for k in json.loads(str(z["key_freqs"]))]
        draws = random_character_sets(kf, n, int(z["draws"]), seed=0)
        ov = np.array([len(set(kf) & set(d)) for d in draws])
        ov_all.append(ov)
        rel_all.append(z["control"] / float(z["internal_baseline"]))
    ov_cat = np.concatenate(ov_all); rel_cat = np.concatenate(rel_all)

    # THE control that cannot be silently wrong: zero-overlap draws must do nothing, and
    # must never reach the effect. If that stops holding, the panel's whole reading fails.
    z0 = rel_cat[ov_cat == 0]
    assert z0.max() < ratio.min(), (z0.max(), ratio.min())
    assert z0.mean() < 1.2, z0.mean()

    levels = sorted(set(int(v) for v in ov_cat))
    ramp = plt.cm.YlOrBr(np.linspace(0.28, 0.92, len(levels)))   # sequential, one hue
    for i, lv in enumerate(levels):
        v = rel_cat[ov_cat == lv]
        jitter = (np.random.default_rng(lv).random(len(v)) - 0.5) * 0.52
        axd.plot(np.full(len(v), i) + jitter, v, ".", ms=2.0, color=ramp[i],
                 alpha=0.45, zorder=2, rasterized=True)
        axd.plot([i - 0.34, i + 0.34], [np.median(v)] * 2, "-", lw=2.0,
                 color="0.15", zorder=4, solid_capstyle="butt")
        axd.text(i, 0.02, f"{len(v)}", transform=axd.get_xaxis_transform(),
                 fontsize=FS, color="0.45", ha="center", va="bottom", path_effects=HALO)
    axd.axhspan(ratio.min(), ratio.max(), color=C_EXCL, alpha=0.16, zorder=1)
    axd.axhline(1.0, color=C_BASE, lw=1.6, ls="--", zorder=3)
    axd.set_yscale("log")
    axd.set_xticks(range(len(levels)))
    axd.set_xticklabels([str(l) for l in levels])
    _style(axd, "Null damage is a dose-response in key-character overlap",
           "key characters the random draw happened to remove",
           "control loss / that run's baseline (log)")
    axd.tick_params(labelsize=FS)
    top = axd.get_ylim()[1]
    # Both labels sit where no draw lands: level 0 never reaches the band, and levels
    # 4-5 never come near baseline. Found by looking at the print-size render.
    axd.annotate("the real intervention\n(removes all of them)",
                 (-0.42, np.median(ratio)), fontsize=FS, color=C_EXCL,
                 ha="left", va="center", path_effects=HALO)
    axd.annotate("baseline", (len(levels) - 0.58, 1.0), fontsize=FS, color=C_BASE,
                 ha="right", va="bottom", path_effects=HALO)
    axd.set_ylim(top=top * 6)
    axd.grid(axis="x", visible=False)
    axd.text(0.5, 1.01, "bar = median;  count above axis = draws at that overlap",
             transform=axd.transAxes, fontsize=FS, color="0.45", va="bottom", ha="center")
    axd.set_title(axd.get_title(), pad=14)

    _caption(fig, paths[0])
    fig.tight_layout(rect=(0, 0.045, 1, 1)); _save(fig, out)
    return out


def precision_2x2(out=f"{FIGDIR}/F7_precision.pdf"):
    """F7 -- C34. Tail Gini_mult at n=113 across {engine, torch} x {float32, float64}.

    The numbers are the pre-registered outcome of PREREGISTER_o18_f64.md and are quoted
    here rather than recomputed: each is a 3-or-5-seed tail mean over a 40k run, and the
    runs are the artifact. Every cell is direct-labelled so the figure carries its own
    values -- no reading them off an axis.
    """
    import glob
    from analyze_n7 import tail_gini
    SRC = {("engine", "float64"): "results/n7_engine/WE_engine_n113_s*.npz",
           ("engine", "float32"): "results/n9_f32/WE_*n113*.npz",
           ("PyTorch", "float32"): "results/o18_cpu/WE_*n113*.npz",
           ("PyTorch", "float64"): "results/o18_f64/WE_*n113*.npz"}

    def _cell(pat):
        gs = [tail_gini(np.load(f, allow_pickle=True), 113) for f in sorted(glob.glob(pat))]
        gs = [float(g[0]) for g in gs if g]
        if not gs:
            raise SystemExit(f"F7: no tail-Gini trajectories under {pat}")
        return float(np.mean(gs))

    # Recomputed from the artifacts, never quoted from the lab record: three of the four
    # cells agreed with the record to 4 dp and the fourth (torch float64) did not --
    # 0.7728 measured against 0.7736 written down.
    cells = {k: _cell(v) for k, v in SRC.items()}
    impls, dtypes = ["engine", "PyTorch"], ["float32", "float64"]
    fig, ax = plt.subplots(figsize=(0.62 * TEXTWIDTH, 2.6))
    w, x = 0.32, np.arange(len(impls), dtype=float)
    for off, dt, col in ((-w / 2, "float32", C_BASE), (+w / 2, "float64", C_EXCL)):
        v = [cells[(i, dt)] for i in impls]
        ax.bar(x + off, v, width=w * 0.9, color=col, label=dt, zorder=2,
               edgecolor=INK, linewidth=0.5, hatch=HATCH[col])
        for xi, vi in zip(x + off, v):
            ax.annotate(f"{vi:.4f}", (xi, vi), textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=FS, color="0.2")
    ax.set_xticks(x); ax.set_xticklabels(impls)
    ax.set_xlim(-0.55, len(impls) - 0.45); ax.set_ylim(0, 1.06)
    _style(ax, "Tail $\\mathrm{Gini}_{\\mathrm{mult}}$ at $n=113$", "", "Gini (amplitude)")
    ax.legend(fontsize=FS, frameon=False, title="training precision", title_fontsize=FS,
              loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("dtype is inert in the engine and decisive in PyTorch:\n"
                  "an interaction, not a main effect", fontsize=FS, color="0.35")
    fig.tight_layout(); _save(fig, out, 0.62)
    return out


def grok_curves(results_dir="results/k03_grid_acts", out=f"{FIGDIR}/F8_grok.pdf",
                grok=0.99, fail=0.90):
    """F8 -- test accuracy against step, every run classified by its WINDOW MEDIAN.

    The point of the figure is the classification rule, not the curves. A run's endpoint
    is a window, never its last logged row: post-grok excursions below 0.90 are common
    under weight decay 1.0 and one of them landing on the final sample once produced a
    retracted claim ("a network can de-grok"). Each curve is therefore coloured by the
    median of its final ten samples, and that median is drawn as a dot at the right edge
    beside the final sample, so a disagreement between the two is visible.
    """
    import glob, os
    runs = []
    for f in sorted(glob.glob(f"{results_dir}/WE_B_thesis_n*_s*.npz")):
        n = int(os.path.basename(f).split("_n")[1].split("_")[0])
        z = np.load(f, allow_pickle=True)
        cols = [str(c) for c in z["hist_cols"]]          # by NAME: index 3 differs by arm
        h = z["hist"]
        runs.append((n, h[:, cols.index("step")], h[:, cols.index("test_acc")]))
    if not runs:
        raise SystemExit(f"no B_thesis runs under {results_dir}")

    # THREE states, not two. A binary gate turns every near-miss into whichever side it
    # falls on: k03's n=125 s1 ends at a window median of 0.978 -- a model that has plainly
    # learned the circuit -- and scoring it as a failure destroys the contrast a control
    # exists to provide. grokked > 0.99, FAILED < 0.90, near-grok in neither arm.
    fig, ax = plt.subplots(figsize=(0.86 * TEXTWIDTH, 2.9))
    cnt = {"grokked": 0, "near-grok": 0, "failed": 0}
    for n, step, acc in runs:
        med = float(np.median(acc[-10:]))
        state = "grokked" if med > grok else ("failed" if med < fail else "near-grok")
        cnt[state] += 1
        col = {"grokked": C_BASE, "near-grok": C_REST, "failed": C_EXCL}[state]
        ax.plot(step, acc, lw=1.0, alpha=0.5 if state == "grokked" else 1.0,
                color=col, zorder=2 if state == "grokked" else 3)
        ax.plot([step[-1]], [med], "o", ms=4.2, mfc="none", mew=1.3, color=col,
                zorder=4, clip_on=False)
    ax.axhline(grok, color=C_GRID, lw=1.0, ls="--", zorder=1)
    ax.axhline(fail, color=C_GRID, lw=1.0, ls=":", zorder=1)
    ax.set_ylim(-0.03, 1.10)
    ax.annotate("grokked > 0.99", (ax.get_xlim()[0], grok), fontsize=FS_SMALL, color="0.45",
                va="bottom", ha="left", xytext=(2, 2), textcoords="offset points")
    ax.annotate("failed < 0.90", (ax.get_xlim()[0], fail), fontsize=FS_SMALL, color="0.45",
                va="top", ha="left", xytext=(2, -2), textcoords="offset points")
    _style(ax, f"Generalisation at 6 moduli: {cnt['grokked']} grokked, "
              f"{cnt['near-grok']} near-grok, {cnt['failed']} failed "
              f"({len(runs)} runs)", "step", "test accuracy")
    for lab, col in (("grokked", C_BASE), ("near-grok", C_REST), ("failed", C_EXCL)):
        ax.plot([], [], color=col, lw=1.6, label=f"{lab} ({cnt[lab]})")
    ax.plot([], [], "o", ms=4.2, mfc="none", mew=1.3, color="0.3",
            label="median of final 10 samples")
    ax.legend(fontsize=FS, frameon=False, loc="lower right", ncol=2)
    _caption(fig, sorted(glob.glob(f"{results_dir}/WE_B_thesis_n*_s*.npz"))[0])
    fig.tight_layout(); _save(fig, out, 0.86)
    return out


# ---------------------------------------------------------------------------
# Explanatory diagrams. These carry no experimental data: every number in them is
# computed from src/tasks/algebra.py, so they are regenerated like any other figure
# and cannot drift from the mathematics they illustrate.
# ---------------------------------------------------------------------------

D_BLUE, D_TEAL, D_RED, D_PURPLE = BLUE, TEAL, RED, PURPLE
D_TARGET = [D_BLUE, D_RED, D_TEAL, D_PURPLE]      # the palette's slot order
D_INK, D_MUTE, D_FAINT = INK, MUTE, "0.80"


def _j_order(n):
    """Residues sorted so that each J-class is a contiguous run, plus the cut points.

    The strata are only visible as rectangles once the axes are ordered by
    gcd(x, n); in the natural order they are scattered lattices.
    """
    from math import gcd
    from sympy import divisors
    ds = list(divisors(n))
    order = sorted(range(n), key=lambda x: (ds.index(gcd(x, n)), x))
    cuts, seen = [], []
    for i, x in enumerate(order):
        d = gcd(x, n)
        if d not in seen:
            seen.append(d)
            cuts.append(i)
    return order, ds, cuts


def _table_panel(ax, n, block=None, title="", note="", mark_nonregular=None):
    """The n x n multiplication table, J-ordered, each cell shaded by the J-class of
    its product.  `block` = (d, e) outlines one stratum."""
    from math import gcd
    order, ds, cuts = _j_order(n)
    idx = {x: i for i, x in enumerate(order)}
    img = np.zeros((n, n))
    for i, x in enumerate(order):
        for j, y in enumerate(order):
            img[i, j] = ds.index(gcd(x * y % n, n))
    ax.imshow(img, cmap="Blues", vmin=-0.8, vmax=len(ds) - 0.5,
              interpolation="nearest", origin="upper")
    for c in cuts[1:]:
        ax.axhline(c - 0.5, color="white", lw=1.1)
        ax.axvline(c - 0.5, color="white", lw=1.1)
    # J-class tick labels at the centre of each run
    bounds = cuts + [n]
    mids = [(bounds[k] + bounds[k + 1] - 1) / 2 for k in range(len(ds))]
    ax.set_xticks(mids); ax.set_yticks(mids)
    sizes = [bounds[k + 1] - bounds[k] for k in range(len(ds))]
    lab = [f"$J_{{{d}}}$" if sz > 1 else "" for d, sz in zip(ds, sizes)]
    ax.set_xticklabels(lab, fontsize=FS); ax.set_yticklabels(lab, fontsize=FS)
    ax.tick_params(length=0)
    for dd in (mark_nonregular or []):
        rs = [idx[x] for x in range(n) if gcd(x, n) == dd]
        ax.add_patch(plt.Rectangle((min(rs) - 0.5, min(rs) - 0.5), len(rs), len(rs),
                                   fill=False, edgecolor=D_RED, lw=1.3, zorder=4))
    if block is not None:
        d, e = block
        rs = [idx[x] for x in range(n) if gcd(x, n) == d]
        cs = [idx[y] for y in range(n) if gcd(y, n) == e]
        ax.add_patch(plt.Rectangle((min(cs) - 0.5, min(rs) - 0.5), len(cs), len(rs),
                                   fill=False, edgecolor=D_RED, lw=2.2, zorder=5))
    ax.set_title(title, fontsize=FS, pad=6)
    if note:
        ax.set_xlabel(note, fontsize=FS, color=D_MUTE, labelpad=6)
    for s in ax.spines.values():
        s.set_color(D_FAINT)


def _cellgrid(ax, vals, rowlab, collab, colour_by, palette, title, xlabel, ylabel,
              ring=None, fmt="{}"):
    """A small labelled table: text in every cell, background keyed to the target."""
    k = len(np.unique(colour_by))
    lut = {v: palette[i % len(palette)] for i, v in enumerate(sorted(np.unique(colour_by)))}
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            ax.add_patch(plt.Rectangle((j, -i), 1, -1, facecolor=lut[colour_by[i, j]],
                                       edgecolor="white", lw=1.6, alpha=0.85))
            ax.text(j + 0.5, -i - 0.5, fmt.format(vals[i, j]), ha="center", va="center",
                    fontsize=FS, color="white", fontweight="bold")
    if ring is not None:
        i, j = ring
        ax.add_patch(plt.Rectangle((j, -i), 1, -1, fill=False, edgecolor=D_INK,
                                   lw=2.4, zorder=6))
    ax.set_xlim(-0.05, vals.shape[1] + 0.05); ax.set_ylim(-vals.shape[0] - 0.05, 0.05)
    ax.set_xticks(np.arange(vals.shape[1]) + 0.5)
    ax.set_yticks(-np.arange(vals.shape[0]) - 0.5)
    ax.set_xticklabels(collab, fontsize=FS_SMALL); ax.set_yticklabels(rowlab, fontsize=FS_SMALL)
    ax.tick_params(length=0)
    ax.set_title(title, fontsize=FS, pad=6)
    ax.set_xlabel(xlabel, fontsize=FS); ax.set_ylabel(ylabel, fontsize=FS)
    ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_visible(False)
    return k


def stratum_theorem(out=f"{FIGDIR}/F1_stratum.pdf", sf=15, nsf=20, block=(2, 4)):
    """Theorem 1, and the limitation it answers, as one picture.

    Top: the multiplication table of a square-free and a non-square-free modulus,
    ordered so the strata are rectangles. Chen et al.'s Thm D.17 makes every class
    of the square-free table regular -- a group, with its own characters. The
    non-square-free table has a non-regular class, which is where their reduction
    stops. Bottom: the outlined stratum unfolded by Theorem 1 into multiplication
    on a local group, with one cell carried through all three panels.

    SIZE. Drawn at 10.2in so that at \\textwidth (~6.5in) nothing prints below ~6pt.
    That is the binding constraint on how much copy each panel can carry.
    """
    from math import gcd
    from sympy import totient
    d, e = block
    m, q = gcd(d * e, nsf), nsf // gcd(d * e, nsf)
    w = (d * e) // m
    xs = [x for x in range(nsf) if gcd(x, nsf) == d]
    ys = [y for y in range(nsf) if gcd(y, nsf) == e]
    us, vs = [x // d for x in xs], [y // e for y in ys]
    raw = np.array([[x * y % nsf for y in ys] for x in xs])
    loc = np.array([[w * u * v % q for v in vs] for u in us])
    # The figure asserts the theorem it draws: if the identity ever failed, no
    # amount of looking at the picture would reveal it.
    assert np.array_equal(raw, m * loc), "Theorem 1 fails on the illustrated block"
    assert not any(z * z % nsf == z for z in xs), f"J_{d} is regular; pick a non-regular class"

    ri, ci = 1, 2                                   # the cell carried through all panels
    x0, y0, u0, v0 = xs[ri], ys[ci], us[ri], vs[ci]

    fig = plt.figure(figsize=(TEXTWIDTH, 6.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.12], width_ratios=[1, 1, 1.3],
                          hspace=0.78, wspace=0.5)

    # ---- top: the two regimes -------------------------------------------------
    _table_panel(fig.add_subplot(gs[0, 0]), sf, block=(3, 3),
                 title=f"$n={sf}$  square-free",
                 note="every class is regular\n(their Thm D.17): "
                      "$J_3\\cdot J_3=J_3$\nis closed, and $J_3$ is "
                      "a group,\nwith identity $6$.")
    _table_panel(fig.add_subplot(gs[0, 1]), nsf, block=block,
                 title=f"$n={nsf}=2^2\\!\\cdot\\!5$  not square-free",
                 note="$J_2$ has no idempotent,\nso it is not "
                      "regular, and\n$J_2\\cdot J_4=J_4$ leaves it.\n"
                      "This is the case they exclude.")

    axt = fig.add_subplot(gs[0, 2]); axt.axis("off")
    axt.set_xlim(0, 1); axt.set_ylim(0, 1)
    T = lambda y, t, **kw: axt.text(0.0, y, t, va="top", **{"fontsize": FS,
                                    "color": "0.25", "linespacing": 1.35, **kw})
    T(1.10, "The reduction does not break.\nIt relocates.",
      fontsize=FS_TITLE, color=D_INK, fontweight="bold", linespacing=1.3)
    T(0.86, "A regular class is a group and has\ncharacters. "
            "A non-regular class is\nneither, yet the units\n"
            "act on it simply transitively,\nso it has a coordinate:")
    T(0.38, r"$\iota_d:(\mathbb{Z}/(n/d)\mathbb{Z})^{\times}\!\to J_d,\;\; u\mapsto du$",
      fontsize=FS_TITLE, color=D_BLUE)
    T(0.25, "In that coordinate, Theorem 1 says\nevery "
            "stratum is multiplication in a\nsmaller ring:")
    T(-0.06, r"$x\cdot y \;=\; m\,(\,w\,uv \;\mathrm{mod}\; q\,)$",
      fontsize=FS_TITLE, color=D_INK)
    T(-0.20, rf"with $m=\mathrm{{gcd}}({d}\!\cdot\!{e},{nsf})={m}$, $w={w}$," "\n"
      rf"$q=n/m={q}$,",
      fontsize=FS, color=D_MUTE)
    T(-0.43, rf"so the outlined block is a clock on $(\mathbb{{Z}}/{q}\mathbb{{Z}})^{{\times}}$.",
      fontsize=FS, color=D_MUTE)

    # ---- bottom: the unfold ---------------------------------------------------
    axa = fig.add_subplot(gs[1, 0])
    _cellgrid(axa, raw, [f"${x}$" for x in xs], [f"${y}$" for y in ys], raw, D_TARGET,
              f"1.  the stratum $J_{{{d}}}\\times J_{{{e}}}$",
              f"$y\\in J_{{{e}}}$\n{raw.size} cells, {len(np.unique(raw))} distinct products",
              f"$x\\in J_{{{d}}}$", ring=(ri, ci))
    axb = fig.add_subplot(gs[1, 1])
    _cellgrid(axb, loc, [f"$u={u}$" for u in us], [f"$v={v}$" for v in vs], raw, D_TARGET,
              "2.  the same cells,\nin torsor coordinates",
              f"$y={e}v$\n" + rf"cell $={w}uv\;\mathrm{{mod}}\;{q}$;  $\times{m}$ gives panel 1",
              f"$x={d}u$", ring=(ri, ci))

    # ---- the clock ------------------------------------------------------------
    axc = fig.add_subplot(gs[1, 2])
    G = [g for g in range(1, q) if gcd(g, q) == 1]
    gen = next(g for g in G if len({pow(g, k, q) for k in range(len(G))}) == len(G))
    elems = [pow(gen, k, q) for k in range(len(G))]
    lut = {t: D_TARGET[i % len(D_TARGET)] for i, t in enumerate(sorted(np.unique(raw)))}
    ang = [np.pi / 2 - 2 * np.pi * k / len(G) for k in range(len(G))]
    axc.add_patch(plt.Circle((0, 0), 1.0, fill=False, edgecolor=D_FAINT, lw=1.2))
    for k, (a, el) in enumerate(zip(ang, elems)):
        axc.scatter([np.cos(a)], [np.sin(a)], s=360, color=lut[m * el], zorder=3,
                    edgecolor="white", linewidth=1.6)
        axc.text(np.cos(a), np.sin(a), f"${el}$", ha="center", va="center",
                 fontsize=FS_TITLE, color="white", fontweight="bold", zorder=4)
        ha = "left" if np.cos(a) > 0.1 else ("right" if np.cos(a) < -0.1 else "center")
        va = "bottom" if np.sin(a) > 0.1 else ("top" if np.sin(a) < -0.1 else "center")
        axc.text(1.46 * np.cos(a), 1.48 * np.sin(a), f"$\\kappa={k}$", ha=ha, va=va,
                 fontsize=FS, color=D_MUTE)
    k_u, k_v, k_w = (elems.index(z % q) for z in (u0, v0, w))
    k_t = (k_u + k_v + k_w) % len(G)
    axc.annotate("", xy=(0.72 * np.cos(ang[k_t]), 0.72 * np.sin(ang[k_t])), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color=D_INK, lw=2.2), zorder=5)
    axc.set_xlim(-2.3, 2.3); axc.set_ylim(-3.3, 3.1); axc.set_aspect("equal")
    axc.axis("off")
    axc.set_title(f"3.  a clock on $G_m=(\\mathbb{{Z}}/{q}\\mathbb{{Z}})^{{\\times}}$",
                  fontsize=FS_TITLE, pad=8)
    axc.text(0, 3.05, f"index each unit by its exponent $\\kappa$ in "
             f"$G_m=\\langle {gen}\\rangle$;\nmultiplying adds $\\kappa$",
             ha="center", va="top", fontsize=FS, color="0.25", linespacing=1.35)
    axc.text(0, -2.05, rf"$x={x0}=({d})({u0})$,   $y={y0}=({e})({v0})$",
             ha="center", va="center", fontsize=FS_TITLE, color=D_INK)
    axc.text(0, -2.6, rf"$\kappa_u\!+\!\kappa_v\!+\!\kappa_w={k_u}\!+\!{k_v}\!+\!{k_w}"
             rf"\equiv{k_t}$, landing on ${elems[k_t]}$",
             ha="center", va="center", fontsize=FS, color=D_MUTE)
    axc.text(0, -3.15, rf"$x\,y={m}\cdot{elems[k_t]}={x0 * y0 % nsf}$"
             rf"  $(\mathrm{{mod}}\;{nsf})$",
             ha="center", va="center", fontsize=FS_TITLE, color=D_INK)

    fig.text(0.005, 0.005, "src/viz/plots.py::stratum_theorem · every value computed "
             "from src/tasks/algebra.py", fontsize=FS_SMALL, color=D_MUTE)
    _save(fig, out)
    return out


def stratification(out=f"{FIGDIR}/F2_stratification.pdf",
                   moduli=(113, 121, 125, 119, 120)):
    """The same operation, five algebras: what n changes is the stratification.

    This is Gate 2's verdict as one image. Every panel is the multiplication table
    of Z/nZ under the same ordering, so the only thing that differs between them is
    the algebra of n -- how many strata there are and how they nest.
    """
    from sympy import factorint, totient
    from src.tasks.algebra import j_structure, nilpotency_depth

    fig, axes = plt.subplots(1, len(moduli), figsize=(TEXTWIDTH, 2.3))
    for ax, n in zip(axes, moduli):
        js = j_structure(n)
        nonreg = [j["d"] for j in js if not j["regular"]]
        f = factorint(n)
        fac = r"\cdot".join(f"{p}^{{{k}}}" if k > 1 else f"{p}" for p, k in sorted(f.items()))
        head = f"$n={n}$" if len(f) == 1 and max(f.values()) == 1 else f"$n={n}={fac}$"
        kind = ("field" if len(f) == 1 and max(f.values()) == 1 else
                "prime power" if len(f) == 1 else
                "square-free" if max(f.values()) == 1 else "composite")
        _table_panel(ax, n, title=f"{head}\n{kind}", mark_nonregular=nonreg,
                     note=f"{len(js)} classes, {len(js) ** 2} strata\n"
                          f"{len(nonreg)} non-regular,  $\\nu={nilpotency_depth(n)}$\n"
                          f"$\\varphi(n)={int(totient(n))}$")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("The operation is the same in every panel; only the algebra of $n$ differs.\n"
                 "Cell shade is the $\\mathcal{J}$-class of the product.",
                 fontsize=FS_TITLE, y=0.99)
    fig.tight_layout()
    fig.text(0.005, -0.1, "src/viz/plots.py::stratification · computed from "
             "src/tasks/algebra.py  ·  red outlines mark non-regular classes",
             fontsize=FS_SMALL, color=D_MUTE);
    _save(fig, out)
    return out


def ablation_scheme(npz_path="results/k03_grid_acts/WE_B_thesis_n121_s0.npz", n=121,
                    out=f"{FIGDIR}/F3_ablation_scheme.pdf"):
    """What the causal test actually does to the model -- the procedure, not the result.

    F4 reports the losses; nothing showed the reader what "restricted" and "excluded"
    mean. Each panel is the SAME character plane under a different projection Pi_S, with
    the loss that projection produces underneath. The spectrum, the key set and the
    losses all come from `ablation.unit_logit_grid` / `ablation_report`, the same code
    the claim rests on.
    """
    from src.analysis.ablation import (unit_logit_grid, ablation_report, key_freq_pairs,
                                       fourier_ablate, cross_entropy_np)
    z = np.load(npz_path, allow_pickle=False)
    G, y, orders, kf = unit_logit_grid(z, n)
    assert len(orders) == 1, "this panel is drawn for a cyclic unit group"
    q = orders[0]
    F = np.fft.fftn(G, axes=(0, 1))
    S = np.linalg.norm(F, axis=-1)                       # magnitude per character pair
    keep = key_freq_pairs(kf, orders)
    rep = ablation_report(G, y, kf, orders)

    mask = np.zeros((q, q), bool)
    for ka, kb in keep:
        mask[ka[0], kb[0]] = True
    mask[0, 0] = True                                    # DC is kept by both projections

    sh = lambda M: np.fft.fftshift(M)                    # DC to the centre, so the
    ext = [-q // 2, q // 2, -q // 2, q // 2]             # diagonal reads as a diagonal
    logS = np.log10(np.maximum(sh(S), 1e-6))
    panels = [
        ("baseline", "$\\Pi_S$ = identity: nothing removed", logS,
         None, rep["baseline"]),
        ("restricted", "keep only $\\Delta$, the key characters", np.where(sh(mask), logS, np.nan),
         "keep", rep["restricted"]),
        ("excluded", "remove only $\\Delta$, keep all the rest", np.where(sh(mask), np.nan, logS),
         "remove", rep["excluded"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.8))
    vmin, vmax = np.percentile(logS, 2), logS.max()
    for ax, (name, sub, img, _, loss) in zip(axes, panels):
        cmap = plt.get_cmap("Blues").copy(); cmap.set_bad("#f2f2f0")
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, extent=ext, origin="lower",
                  interpolation="nearest")
        ax.plot([-q // 2, q // 2], [-q // 2, q // 2], color=D_RED, lw=0.7, ls=(0, (4, 4)),
                zorder=3, alpha=0.8)
        # Nine kept pixels out of 110^2 are invisible at print size, and "invisible"
        # would read as "nothing there" -- exactly the wrong reading. Ring them.
        if name != "baseline":
            c = np.array([((int(ka[0]) + q // 2) % q) - q // 2 for ka, _ in keep])
            ax.scatter(c, c, s=52, facecolors="none", zorder=4, linewidth=1.3,
                       edgecolors=D_BLUE if name == "restricted" else D_RED)
        ax.set_title(f"{name}\n{sub}", fontsize=FS, pad=6)
        ax.set_xlabel("$\\kappa_b$", fontsize=FS); ax.tick_params(labelsize=FS)
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect("equal")
        col = D_BLUE if name == "restricted" else (D_RED if name == "excluded" else D_MUTE)
        ax.annotate(f"cross-entropy  {loss:.3g}", xy=(0.5, -0.33), xycoords="axes fraction",
                    ha="center", fontsize=FS_TITLE, color=col, fontweight="bold")
    axes[0].set_ylabel("$\\kappa_a$", fontsize=FS)
    decades = np.log10(rep["excluded"] / rep["baseline"])
    axes[1].annotate("at or below baseline:\nthey are sufficient", xy=(0.97, 0.05),
                     xycoords="axes fraction", ha="right", fontsize=FS, color=D_BLUE)
    axes[2].annotate(f"{decades:.0f} orders of magnitude\nworse: they are necessary",
                     xy=(0.97, 0.05), xycoords="axes fraction", ha="right",
                     fontsize=FS, color=D_RED)
    axes[0].annotate("the diagonal $\\Delta=\\{(\\kappa,\\kappa)\\}$", xy=(q // 4, q // 4),
                     xytext=(-q // 2.4, q // 3.2), fontsize=FS, color=D_RED,
                     arrowprops=dict(arrowstyle="->", color=D_RED, lw=1.0))
    fig.suptitle(f"$n={n}$, seed 0: the character plane of the logits, and the two "
                 f"projections the causal test applies to it\n"
                 f"($|\\Delta|$ = {len(kf)} key characters of {q // 2})",
                 fontsize=FS_TITLE, y=0.995)
    fig.tight_layout()
    fig.text(0.005, -0.11, f"{npz_path} · src/viz/plots.py::ablation_scheme ·\n"
             "spectrum, key set and losses from src/analysis/ablation.py",
             fontsize=FS_SMALL, color=D_MUTE, va="top")
    _save(fig, out)
    return out


def _box(ax, x, y, w, h, text, face, edge, fs=FS_SMALL, tc=D_INK, bold=False):
    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=face, edgecolor=edge, lw=1.3,
                               zorder=2, joinstyle="round"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, zorder=3, linespacing=1.45,
            fontweight="bold" if bold else "normal")


def _arrow(ax, x0, y0, x1, y1, label="", fs=FS):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=1,
                arrowprops=dict(arrowstyle="-|>", color="0.45", lw=1.4,
                                shrinkA=2, shrinkB=2))
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.035, label, ha="center", va="bottom",
                fontsize=fs, color=D_MUTE)


def pipeline(out=f"{FIGDIR}/F0_pipeline.pdf"):
    """What the paper does, end to end, in one strip.

    Nanda's Figure 1 is the register here: the stack on the left, what happens to the
    representation in the middle, and the coordinate it happens in on the right. The
    point this figure has to carry is that the measurement is made in exponent
    coordinates, PER STRATUM -- everything else is standard.
    """
    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 1.7))
    ax.set_xlim(0, 10.2); ax.set_ylim(0.52, 3.15); ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    LIGHT, RING = "#eef4fc", D_BLUE
    y, h = 1.62, 0.86
    cols = [
        (0.05, 1.42, "$a\\cdot b \\;\\mathrm{mod}\\; n$\n$23$ moduli,\n$5$ seeds",
         "#f4f4f2", "0.72"),
        (1.72, 1.62, "1-layer\ntransformer\n$d=128$, 4 heads", LIGHT, RING),
        (3.59, 1.62, "train to grok\nfull batch, AdamW\nwd $=1.0$", LIGHT, RING),
        (5.46, 1.72, "relabel units by\nexponent tuple\n$u=\\prod g_i^{e_i}$", "#E4F3F5", D_TEAL),
        (7.43, 1.42, "2D character\ntransform\nper stratum", "#E4F3F5", D_TEAL),
        (9.00, 1.15, "measure\n+\nablate", "#F6E3E2", D_RED),
    ]
    for x, w, t, fc, ec in cols:
        _box(ax, x, y, w, h, t, fc, ec)
    for (x0, w0, *_), (x1, *_) in zip(cols, cols[1:]):
        _arrow(ax, x0 + w0, y + h / 2, x1, y + h / 2)

    ax.annotate("", xy=(5.46, y - 0.08), xytext=(10.15, y - 0.08), zorder=1,
                arrowprops=dict(arrowstyle="-", color=D_TEAL, lw=1.1,
                                connectionstyle="bar,fraction=-0.07"))
    ax.text((5.46 + 10.15) / 2, y - 0.46,
            "the only non-standard step: multiplication becomes addition\n"
            "only in these coordinates, and every stratum $J_d\\times J_e$ gets\n"
            "its own, on its own local group $G_m$",
            ha="center", va="top", fontsize=FS, color=D_TEAL, linespacing=1.55)
    ax.text(0.05, y + h + 0.30, "Sparsity says a basis describes the model; ablation says a "
            "component is responsible for it.", fontsize=FS, color=D_INK, va="bottom")
    fig.text(0.005, 0.005, "src/viz/plots.py::pipeline", fontsize=FS_SMALL, color=D_MUTE)
    _save(fig, out)
    return out


def descent_lattice(out=f"{FIGDIR}/F5_descent.pdf", moduli=(121, 125, 120)):
    """How the strata nest: the divisor lattice with the composition map on it.

    Theorem 1 says what happens inside a stratum. This says what happens between them:
    J_d . J_d lands in J_gcd(d^2, n), which for a square-free modulus is J_d itself and
    for a prime power walks down to zero. nu(n) is the length of that walk, and it is
    the graded variable separating n = 121 from n = 125 at nearly equal phi(n).
    """
    import collections
    from math import gcd
    from sympy import divisors, factorint, totient
    from src.tasks.algebra import j_structure, nilpotency_depth

    # width follows each lattice's widest level, so the nodes are the same size in all three
    widest = [max(collections.Counter(sum(factorint(d).values()) for d in divisors(n)).values())
              for n in moduli]
    fig, axes = plt.subplots(1, len(moduli), figsize=(TEXTWIDTH, 2.6),
                             gridspec_kw={"width_ratios": [w + 1 for w in widest]})
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.path import Path
    from matplotlib.transforms import Affine2D, ScaledTranslation
    # Everything below is in points. A scatter marker of area s pt^2 has radius sqrt(s)/2,
    # and its edge stroke adds half its width, so R_NODE is the circle's OUTER edge.
    NODE_S, NODE_LW, ARROW_LW, D_EDGE = 250, 1.3, 0.8, "0.62"
    R_NODE = np.sqrt(NODE_S) / 2 + NODE_LW / 2
    HL, HW = 0.55, 0.2                               # head length, half-width (x mutation)
    arrow = dict(arrowstyle=f"-|>,head_length={HL},head_width={HW}", mutation_scale=6.5,
                 lw=ARROW_LW, joinstyle="miter", capstyle="butt", zorder=1, clip_on=False)
    # A filled "-|>" head stops 1.0 pt short of its end point at any lw and scale (measured
    # at 7200 dpi on the pinned matplotlib 3.11.1, both shrink and path modes; re-measure
    # if it is upgraded). Aiming the tip 1.0 pt inside the edge lands it ON the edge.
    tip = -1.0
    LOOP_C, LOOP_R = R_NODE + 3.4, 6.2               # the self-loop's circle (centre on +x)

    def _meet(r):
        """Upper point where the loop's circle crosses the circle of radius r."""
        x = (r * r - LOOP_R ** 2 + LOOP_C ** 2) / (2 * LOOP_C)
        return x, np.sqrt(r * r - x * x)

    for ax, n in zip(np.atleast_1d(axes), moduli):
        ds = list(divisors(n))
        reg = {j["d"]: j["regular"] for j in j_structure(n)}
        # Depth = number of prime factors with multiplicity, so the lattice is drawn by
        # its grading rather than by divisor size: J_1 at the top, {0} at the bottom.
        depth = {d: sum(factorint(d).values()) for d in ds}
        maxd = max(depth.values())
        byd = {}
        for d in ds:
            byd.setdefault(depth[d], []).append(d)
        pos = {}
        for lev, row in sorted(byd.items()):
            for i, d in enumerate(sorted(row)):
                pos[d] = ((i - (len(row) - 1) / 2) * 1.55, maxd - lev)
        for d in ds:                                   # the squaring map J_d . J_d
            t = gcd(d * d, n)
            if t != d:
                # shrink is in points, like the marker, so tail and tip land on the node's
                # outer edge at any figure size
                ax.add_patch(FancyArrowPatch(pos[d], pos[t], shrinkA=R_NODE,
                                             shrinkB=R_NODE + tip, color=D_EDGE,
                                             connectionstyle="arc3,rad=0.16", **arrow))
            else:
                # the loop, drawn in points about the node's centre: a circle of radius
                # LOOP_R centred LOOP_C to the right, from the node edge back to it
                x, y = pos[d]
                a0, a1 = _meet(R_NODE), _meet(R_NODE + tip)
                # Clockwise over the outer side, from the upper meeting point to the lower.
                # The head is its own straight patch: on a loop this tight, matplotlib's
                # head placement cuts the curve at the wrong crossing.
                p_end = np.arctan2(-a1[1], a1[0] - LOOP_C)
                p_base = p_end + HL * arrow["mutation_scale"] / LOOP_R   # one head length back
                on = lambda p: np.array([LOOP_C + LOOP_R * np.cos(p), LOOP_R * np.sin(p)])
                arc = Path.arc(np.degrees(p_base), np.degrees(np.arctan2(a0[1], a0[0] - LOOP_C)))
                trans = (Affine2D().scale(1 / 72) + fig.dpi_scale_trans
                               + ScaledTranslation(x, y, ax.transData))
                for pth, style in ((Path(arc.vertices[::-1] * LOOP_R + [LOOP_C, 0], arc.codes), "-"),
                                   (Path([on(p_base), (a1[0], -a1[1])]), arrow["arrowstyle"])):
                    ax.add_patch(FancyArrowPatch(path=pth, color=D_TEAL, transform=trans,
                                                 shrinkA=0, shrinkB=0,
                                                 **{**arrow, "arrowstyle": style}))
        for d in ds:
            x, y = pos[d]
            closed, ok = gcd(d * d, n) == d, reg[d]
            ax.scatter([x], [y], s=NODE_S, zorder=3, linewidth=NODE_LW,
                       color="#E4F3F5" if ok else "#F6E3E2",
                       edgecolor=D_TEAL if ok else D_RED)
            ax.text(x, y, "$\\{0\\}$" if d == n else f"$J_{{{d}}}$", ha="center",
                    va="center", fontsize=FS, color=D_INK, zorder=4)
        span = max(abs(x) for x, _ in pos.values()) + 0.92
        ax.set_xlim(-span, span); ax.set_ylim(-0.78, maxd + 0.55)
        # Equal aspect, or the self-loop offset means something different in a
        # one-column lattice (121) than in a four-column one (120).
        ax.set_aspect("equal"); ax.axis("off")
        fac = r"\cdot".join(f"{p}^{{{k}}}" if k > 1 else f"{p}" for p, k in
                            sorted(factorint(n).items()))
        nr = sum(1 for d in ds if not reg[d])
        ax.set_title(f"$n={n}={fac}$", fontsize=FS_TITLE, pad=4)
        ax.text(0, -0.64, f"$\\nu(n)={nilpotency_depth(n)}$,  {nr} non-regular,  "
                f"$\\varphi(n)={int(totient(n))}$", ha="center", va="center",
                fontsize=FS, color=D_MUTE)
    from matplotlib.lines import Line2D
    fig.legend(handles=[
        Line2D([], [], marker="o", ls="", markerfacecolor="#E4F3F5",
               markeredgecolor=D_TEAL, markeredgewidth=1.6, markersize=9,
               label="regular: a group, with its own characters"),
        Line2D([], [], marker="o", ls="", markerfacecolor="#F6E3E2",
               markeredgecolor=D_RED, markeredgewidth=1.6, markersize=9,
               label="non-regular: no idempotent, no local inverse"),
        Line2D([], [], color=D_EDGE, lw=ARROW_LW, label="$J_d\\cdot J_d$ leaves the class"),
        Line2D([], [], color=D_TEAL, lw=ARROW_LW, label="$J_d\\cdot J_d=J_d$: closed"),
    ], loc="lower center", ncol=2, frameon=False, fontsize=FS,
        bbox_to_anchor=(0.5, -0.14))
    fig.suptitle("Squaring walks a class down the divisor lattice.\nHow far it walks is "
                 "$\\nu(n)$, and it is $1$ exactly when $n$ is square-free.",
                 fontsize=FS_TITLE, y=0.99)
    fig.tight_layout()
    fig.text(0.005, -0.175, "src/viz/plots.py::descent_lattice · computed from "
             "src/tasks/algebra.py", fontsize=FS_SMALL, color=D_MUTE)
    _save(fig, out)
    return out


def gate2_discrimination(out=f"{FIGDIR}/F9_discrimination.pdf"):
    """Which Gate 2 criteria separate a grokked model from a failed one -- including the
    one that does not.

    The pre-registration named G1 PRIMARY because it is the project's established
    statistic. It passes on 100% of FAILED stratum-measurements, because 30% of every
    block is training data and a memorising model's logits are still a function of u*v
    there. This figure is that falsifier firing, drawn.

    FORM. Two rates per criterion, and the quantity of interest is the GAP between them,
    so this is the same paired dot plot as `gate2_strata` rather than a bar chart: a bar
    would encode distance from zero, and zero is not the reference here.
    """
    import sys
    sys.path.insert(0, ".")
    from scripts.gate2_discriminate import TESTS, load

    grokked = load("results/gate2/k03_grid_acts.json") + load("results/gate2/n7_engine.json")
    failed = load("results/gate2/*_controls.json")
    assert grokked and failed, "run analyze_gate2.py <dir> --controls first"

    rows = []
    for name, fn in TESTS.items():
        g = [x for x in (fn(s) for s in grokked) if x is not None]
        b = [x for x in (fn(s) for s in failed) if x is not None]
        if not g or not b:
            continue
        gr, br = sum(g) / len(g), sum(b) / len(b)
        rows.append(dict(name=name, gr=gr, br=br,
                         gn=(sum(g), len(g)), bn=(sum(b), len(b)),
                         primary=name.startswith("G1"),
                         # the same thresholds gate2_discriminate.py prints
                         verdict=("yes" if gr > 0.9 and br < 0.25 else
                                  "partly" if gr > 0.9 and br < 0.75 else "no")))
    rows.sort(key=lambda r: r["gr"] - r["br"])

    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 3.3))
    y = np.arange(len(rows))
    for i, r in zip(y, rows):
        ax.plot([r["br"], r["gr"]], [i, i], color="0.78", lw=2.0, zorder=1)
        ax.scatter([r["gr"]], [i], s=40, color=D_BLUE, zorder=3)
        ax.scatter([r["br"]], [i], s=40, facecolors="none", edgecolors=D_RED,
                   linewidth=1.4, zorder=3)
        ax.annotate(f"{r['gn'][0]}/{r['gn'][1]}", (r["gr"], i), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=FS, color=D_BLUE)
        # When a criterion does not separate, the two labels land on the same point --
        # and the row where that happens is the one the figure exists to show.
        dy = 6 if abs(r["gr"] - r["br"]) > 0.06 else -13
        ax.annotate(f"{r['bn'][0]}/{r['bn'][1]}", (r["br"], i), xytext=(0, dy),
                    textcoords="offset points", ha="center", fontsize=FS, color=D_RED)
        if r["primary"]:
            ax.axhspan(i - 0.42, i + 0.42, color=D_RED, alpha=0.08, zorder=0)
            ax.annotate("pre-registered primary, and it does not separate",
                        (-0.04, i - 0.52), fontsize=FS, color=D_RED, ha="left",
                        va="top")
        ax.annotate(r["verdict"], (1.20, i), fontsize=FS, va="center", ha="center",
                    color=D_RED if r["verdict"] == "no" else "0.30",
                    fontweight="bold" if r["verdict"] == "no" else "normal")
    ax.annotate("separates?", (1.20, len(rows) - 0.55), fontsize=FS, ha="center",
                va="center", color=D_MUTE)
    ax.set_yticks(y); ax.set_yticklabels([r["name"] for r in rows], fontsize=FS)
    ax.set_xlim(-0.06, 1.32); ax.set_ylim(-1.75, len(rows) - 0.30)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xticklabels([f"{v:.0%}" for v in np.linspace(0, 1, 6)], fontsize=FS)
    _style(ax, "", "fraction of stratum-measurements passing the criterion", "")
    ax.grid(axis="y", visible=False)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color=D_BLUE, markersize=8,
               label=f"grokked runs ({len(grokked)} measurements)"),
        Line2D([], [], marker="o", ls="", markerfacecolor="none",
               markeredgecolor=D_RED, markeredgewidth=1.8, markersize=8,
               label=f"failed runs ({len(failed)} measurements)")],
        loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=FS)
    ax.set_title("A criterion that also passes on failed runs is a necessary condition, "
                 "not evidence", fontsize=FS_TITLE, pad=8)
    fig.tight_layout()
    fig.text(0.005, -0.03, "results/gate2/*.json · src/viz/plots.py::gate2_discrimination "
             "· scored by scripts/gate2_discriminate.py", fontsize=FS_SMALL, color=D_MUTE)
    _save(fig, out)
    return out


def crt_law(out=f"{FIGDIR}/F6_crt.pdf", moduli=(165, 120, 119)):
    """C8: the energy sits on the frequency set the CRT predicts, before any threshold.

    The law names a set of frequencies from the factorisation of n alone. The top row
    asks whether the observed spectrum is concentrated there; the bottom row calibrates
    that against 20,000 random sets of the same size. Prime powers are omitted because
    one CRT component makes the predicted set every frequency (vacuous by construction,
    not by failure), and the caption says so.

    Positive control: n = 165 must reproduce the published 14.08x. A bin-convention slip
    here is silent at every modulus but one (`predicted` returns frequencies,
    `freq_energy` returns bins at position frequency - 1), so the figure asserts the
    published value before it draws anything.
    """
    import sys
    sys.path.insert(0, ".")
    from sympy import factorint
    from test_crt_law import predicted, freq_energy, permutation_test, N_PERM

    def case(W, n, label):
        W = W[:n].astype(float)
        W = W - W.mean(0, keepdims=True)
        e = freq_energy(W, n)
        pred = [k - 1 for k in predicted(n)]          # frequencies -> bins. Never omit.
        obs, p, nullmean, null = permutation_test(e, pred, return_null=True)
        return dict(n=n, label=label, e=e, pred=pred, obs=obs, p=p, null=null)

    cases = [case(np.load(f"results/k01_scout/scout_n{n}_seed0.npz")["W_E"], n, "ours")
             for n in moduli]
    try:
        import torch
        ck = torch.load("reference/interpreting-monoids/experiments/"
                        "P165_d128_h4_mlp512_s1.pt", map_location="cpu",
                        weights_only=False)
        sd = ck.get("model_state_dict", ck)
        cases.append(case(sd["embed.weight"].numpy(), 165, "their checkpoint"))
    except Exception as exc:                          # a gap must never read as absence
        cases.append(dict(n=165, label="their checkpoint", missing=str(exc)[:60]))

    ctl = next(c for c in cases if c["n"] == 165 and c["label"] == "ours")
    assert abs(ctl["obs"] - 14.08) < 0.01, (
        f"positive control failed: n=165 reads {ctl['obs']:.2f}, published 14.08 -- "
        "suspect the bin convention before believing any other panel")

    fig, axes = plt.subplots(2, len(cases), figsize=(TEXTWIDTH, 3.9),
                             gridspec_kw=dict(height_ratios=[1.0, 0.82], hspace=0.75,
                                              wspace=0.4))
    for col, c in enumerate(cases):
        top, bot = axes[0][col], axes[1][col]
        fac = "\\cdot".join(f"{q}^{{{k}}}" if k > 1 else str(q)
                            for q, k in sorted(factorint(c["n"]).items()))
        if c.get("missing"):
            for ax in (top, bot):
                ax.axis("off")
            top.text(0.5, 0.5, f"$n={c['n']}$, {c['label']}\n\nNOT AVAILABLE\n"
                     "run scripts/restore_external.sh", ha="center", va="center",
                     fontsize=FS, color=D_RED, transform=top.transAxes)
            continue
        m = np.zeros(len(c["e"]), bool); m[c["pred"]] = True
        k = np.arange(1, len(c["e"]) + 1)
        top.bar(k[~m], c["e"][~m], width=1.0, color=FAINT, zorder=2)
        top.bar(k[m], c["e"][m], width=1.0, color=D_RED, zorder=3)
        head = f"$n={c['n']}={fac}$" + ("" if c["label"] == "ours" else f"\n({c['label']})")
        _style(top, f"{head}\n{m.sum()} of {len(c['e'])} frequencies predicted",
               "frequency $k$", "energy" if col == 0 else "")
        top.set_title(top.get_title(), fontsize=FS_SMALL)

        lo = min(c["null"].min(), 0.6)
        bins = np.logspace(np.log10(lo * 0.95), np.log10(c["obs"] * 1.35), 44)
        bot.hist(c["null"], bins=bins, color=D_MUTE, edgecolor="white", linewidth=0.4,
                 zorder=2)
        bot.axvline(c["obs"], color=D_RED, lw=2.0, zorder=4)
        bot.set_xscale("log"); bot.set_xlim(bins[0], bins[-1])
        # A narrow log range (n=119 spans 0.6 to 4.5) puts five auto-formatted decade
        # labels on top of each other. Name the ticks.
        from matplotlib.ticker import FixedLocator, NullLocator
        tk = [t for t in (1, 2, 10, 20, 50) if bins[0] <= t <= bins[-1]]
        bot.xaxis.set_major_locator(FixedLocator(tk))
        bot.xaxis.set_minor_locator(NullLocator())
        bot.set_xticklabels([f"{t:g}$\\times$" for t in tk], fontsize=FS)
        bot.annotate(f"{c['obs']:.2f}$\\times$", (c["obs"], 0.86), xycoords=("data",
                     "axes fraction"), ha="right", va="top", fontsize=FS,
                     color=D_RED, fontweight="bold", rotation=90)
        bot.annotate(f"$p<${1 / N_PERM:.0e}".replace("e-05", "\\times10^{-5}$").replace(
            "$p<$", "$p<"), (0.02, 1.03), xycoords="axes fraction", fontsize=FS,
            color=D_INK, va="bottom")
        _style(bot, "", "enrichment (log)", "permutations" if col == 0 else "")
        bot.set_yticks([])

    fig.suptitle("The predicted set is fixed by the factorisation of $n$ before any "
                 "model is trained;\nthe energy is on it.", fontsize=FS_TITLE, y=1.0)
    fig.subplots_adjust(top=0.8)
    fig.text(0.005, -0.035, f"results/k01_scout/*.npz · src/viz/plots.py::crt_law · "
             f"set, energy and null from test_crt_law.py ({N_PERM:,} permutations);\n"
             "prime powers omitted as vacuous", fontsize=FS_SMALL, color=D_MUTE)
    _save(fig, out)
    return out
