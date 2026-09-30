"""Re-derive the paper's load-bearing numbers from the ARTIFACTS, not from prose.

  PYTHONPATH=. .venv/bin/python test_paper_numbers.py

A number in the paper that agrees with FINDINGS.md proves only that two documents agree.
This recomputes each load-bearing number from the saved artifact and fails if the paper
and the data disagree. It has already caught three:

  - FINDINGS 3.3's 5-seed column is labelled `restricted / baseline` and holds
    `baseline / restricted`; the draft copied the label and would have printed the ratio
    backwards. It is also a MEAN over a heavy-tailed distribution -- at n=113 one seed of
    five reads 264x where the others read 1.56-8.0 -- so the paper reports the MEDIAN.
  - "models grok at 20 of 23 moduli, the three exceptions being 49, 54 and 63" was wrong
    twice: 19 of 23 grok in a majority of seeds, and there is a FOURTH exception, n=91.
  - "p_correct reaches exactly 1.000000" is a six-decimal DISPLAY. In float64 it maxes at
    1 - 3.4e-8 and never reaches 1 in the arithmetic.
"""
import json, glob, collections, math, os, re, pathlib
import numpy as np
from src.analysis.sparsity import key_freqs_5x_median

ok = fail = 0
_ASSERTED = []   # every value a PASSING check compared; the coverage gate at the end reads it
_SKIPPED = []    # every value a check would compare whose input does not ship (public repository)


def _collect(x):
    if isinstance(x, (bool, np.bool_)):
        return
    if isinstance(x, (int, float, np.integer, np.floating)):
        _ASSERTED.append(float(x))
    elif isinstance(x, str):
        for t in re.findall(r"\d[\d,]*\.?\d*(?:e[-+]?\d+)?", x.replace("{,}", ",")):
            try:
                _ASSERTED.append(float(t.replace(",", "").rstrip(".")))
            except ValueError:
                pass
    elif isinstance(x, dict):
        for k, v in x.items():
            _collect(k); _collect(v)
    elif isinstance(x, (list, tuple, set, frozenset, np.ndarray)):
        for v in x:
            _collect(v)


def chk(label, got, want, tol=0):
    global ok, fail
    good = (abs(got - want) <= tol) if isinstance(want, (int, float)) else got == want
    print(f"  [{'PASS' if good else 'FAIL'}] {label:52s} paper={want}  artifact={got}")
    ok, fail = ok + good, fail + (not good)
    if good:
        _collect((got, want))

print("GATE 2 (C33) -- results/gate2/k03_grid_acts.json")
d = json.load(open("results/gate2/k03_grid_acts.json"))
meas = [s for r in d["runs"] for s in r["strata"] if s.get("measurable")]
chk("32 strata", len(d["verdict"]), 32)
chk("26 non-unit strata", sum(1 for v in d["verdict"].values() if not v["unit"]), 26)
chk("157 stratum-run measurements", len(meas), 157)
chk("12 distinct local groups",
    len({tuple(s["orders_m"]) for s in meas}), 12)
chk("6 moduli", len({r["n"] for r in d["runs"]}), 6)
chk("G2 min 3.98e2", float(f'{min(s["excluded"]/s["baseline"] for s in meas):.2e}'), 3.98e2)
chk("G2 max 2.71e8", float(f'{max(s["excluded"]/s["baseline"] for s in meas):.2e}'), 2.71e8)
chk("G4 Dshare 0.837-0.997",
    (round(min(s["Dshare"] for s in meas), 3), round(max(s["Dshare"] for s in meas), 3)),
    (0.837, 0.997))
chk("lut_r2 0.012-0.074",
    (round(min(s["lut_r2"] for s in meas), 3), round(max(s["lut_r2"] for s in meas), 3)),
    (0.012, 0.074))
# R1 (2026-09-21): `p_perm == 0.0` is unreachable under (1+b)/(B+1), and at 50x the
# resolution the old "0 of 200 draws in ALL 157" resolves into 133 at the floor and 24
# with 1-22 damaging draws. That is not a falsification (the pre-registration says so in
# advance); it is resolution the 200-draw null could not deliver. All 157 still pass.
chk("G1: all 157 pass p<0.01", sum(s["p_perm"] < 0.01 for s in meas), 157)
chk("G1: beyond EVERY draw in 133 of 157", sum(s["n_ge"] == 0 for s in meas), 133)
chk("G1: the worst is 22 draws of 10,000", max(s["n_ge"] for s in meas), 22)
chk("G1: so the largest p is 0.0023", round(max(s["p_perm"] for s in meas), 4), 0.0023)
chk("G1: null drawn at B = 10,000", sorted({s["n_draws"] for s in meas}), [10000])

print("\nCAUSAL (C22/C23) -- results/k03_grid_acts_n4_ablation.npz")
rows = [json.loads(str(r)) for r in
        np.load("results/k03_grid_acts_n4_ablation.npz", allow_pickle=True)["rows"]]
b = [r for r in rows if r["tag"].startswith("B_thesis")]
per = collections.defaultdict(list)
for r in b: per[r["n"]].append(r)
for n, med in ((113, 2.34), (121, 1.85), (125, 1.98), (119, 0.67), (120, 1.25), (165, 1.42)):
    got = float(np.median([r["baseline"] / r["restricted"] for r in per[n]]))
    chk(f"n={n} median baseline/restricted", round(got, 2), med, 0.01)
    chk(f"n={n} p<0.01 in 5/5", sum(r["p_perm"] < 0.01 for r in per[n]), 5)
chk("restricted improves in 25/30 runs",
    sum(r["baseline"] / r["restricted"] > 1 for r in b), 25)

print("\nMODULUS TABLE (C1/C2) -- scripts/modulus_table.py")
import sys; sys.path.insert(0, "scripts")
import modulus_table as mt
rs = mt.rows()
chk("23 moduli", len(rs), 23)
chk("14 non-square-free", sum(1 for r in rs if not r["sf"]), 14)
chk("6 prime powers", sum(1 for r in rs if r["omega"] == 1 and not r["sf"]), 6)
chk("C2 counts 0,0,1,2,8",
    [{r["n"]: r["nr"] for r in rs}[n] for n in (113, 119, 121, 125, 120)], [0, 0, 1, 2, 8])
chk("19 of 23 grok in a majority of seeds",
    sum(1 for r in rs if r["grok"] * 2 > r["seeds"]), 19)
chk("only n=49 fails every seed", [r["n"] for r in rs if r["grok"] == 0], [49])
import io as _io, contextlib as _cl
with _cl.redirect_stdout(_io.StringIO()) as _buf:
    mt.latex(rs)
_tabM = pathlib.Path("paper/sections/01-setup.tex").read_text() \
    .split("\\label{sec:setup-moduli}", 1)[1].split("\\midrule", 1)[1].split("\\bottomrule", 1)[0]
_norm = lambda t: [re.sub(r"\s+", " ", r).strip() for r in t.split("\\\\") if r.strip()]
chk("tab:moduli body equals modulus_table.latex() (every cell)", _norm(_tabM), _norm(_buf.getvalue()))

print("\nSOFTMAX COLLAPSE (C12/C13) -- results/c12_collapse/")
z = np.load("results/c12_collapse/softmax_collapse_n17.npz", allow_pickle=True)
C = [str(c) for c in z["hist_cols"]]; s, t = z["softmax"], z["stablemax"]
g, m = C.index("grad_norm"), C.index("max_logit")
chk("6.2 orders of gradient decay", round(float(np.log10(s[0, g] / s[-1, g])), 1), 6.2, 0.05)
chk("p_correct rises to 1 - 3.4e-8",
    round((1 - float(s[:, C.index("p_correct")].max())) * 1e8, 1), 3.4, 0.05)
chk("p_correct is NEVER exactly 1.0", bool((s[:, C.index("p_correct")] == 1.0).any()), False)
chk("stablemax/softmax |grad| = 3.2x", round(float(t[-1, g] / s[-1, g]), 1), 3.2, 0.05)
chk("logit scales 50.1 vs 19154.3",
    (round(float(s[-1, m]), 1), round(float(t[-1, m]), 1)), (50.1, 19154.3))

print("\nCOVERAGE (eq:coverage) -- recomputed from the J-class sizes")
from sympy import totient
from src.tasks.algebra import j_class
by_n = {}
for r in d["runs"]:
    S = by_n.setdefault(r["n"], set())
    for st in r["strata"]:
        if st.get("measurable"):
            S.add((st["d"], st["e"]))
for n, (want_unit, want_cov) in {113: (98.2, 98.2), 121: (82.6, 97.7), 125: (64.0, 89.6),
                                 119: (65.1, 88.6), 120: (7.1, 23.1), 165: (23.5, 82.3)}.items():
    cov = sum(len(j_class(n, dd)) * len(j_class(n, ee)) for dd, ee in by_n[n]) / n ** 2
    unit = int(totient(n)) ** 2 / n ** 2
    chk(f"n={n} unit-stratum coverage", round(unit * 100, 1), want_unit, 0.05)
    chk(f"n={n} measured coverage", round(cov * 100, 1), want_cov, 0.05)

print("\nPRECISION 2x2 (C34) -- the four cells, each recomputed from its own artifact")
import glob
from analyze_n7 import tail_gini
SRC = {("engine", "float64"): ("results/n7_engine/WE_engine_n113_s*.npz", 0.7536),
       ("engine", "float32"): ("results/n9_f32/WE_*n113*.npz", 0.7671),
       ("torch", "float32"): ("results/o18_cpu/WE_*n113*.npz", 0.5791),
       ("torch", "float64"): ("results/o18_f64/WE_*n113*.npz", 0.7728)}
cells = {}
for (impl, dt), (pat, want) in SRC.items():
    gs = [tail_gini(np.load(f, allow_pickle=True), 113) for f in sorted(glob.glob(pat))]
    gs = [float(g[0]) for g in gs if g]
    cells[(impl, dt)] = float(np.mean(gs))
    chk(f"{impl} {dt} tail Gini", round(cells[(impl, dt)], 4), want, 1e-9)
# The pre-registered F1, computed from the COMMITTED formula. analyze_n9.py prints a
# DIFFERENT f on this arm -- its baseline/target are the engine's two dtype cells.
f1 = (cells[("torch", "float64")] - cells[("torch", "float32")]) / \
     (cells[("engine", "float64")] - cells[("torch", "float32")])
chk("pre-registered F1 effect size", round(f1, 3), 1.110, 0.001)
chk("F1 HELD (f >= 0.70)", f1 >= 0.70, True)
chk("dtype is inert in the engine (|delta| < 0.02)",
    round(abs(cells[("engine", "float32")] - cells[("engine", "float64")]), 4), 0.0135, 0.001)

# ---------------------------------------------------------------------------------
# Figure captions. Every number a caption asserts is re-derived here from the same
# source the figure draws from.
# ---------------------------------------------------------------------------------
print("\nFIGURE CAPTIONS -- numbers asserted by the diagrams (F1, F2, F3, F5)")
from math import gcd as _gcd
from sympy import divisors as _divisors, totient as _totient
from src.tasks.algebra import j_structure as _jstruct, nilpotency_depth as _nu

# F1: the worked block J_2 x J_4 at n = 20.
_n, _d, _e = 20, 2, 4
_m = _gcd(_d * _e, _n); _q = _n // _m; _w = (_d * _e) // _m
_xs = [x for x in range(_n) if _gcd(x, _n) == _d]
_ys = [y for y in range(_n) if _gcd(y, _n) == _e]
_raw = np.array([[x * y % _n for y in _ys] for x in _xs])
chk("F1: m = gcd(2*4, 20)", _m, 4)
chk("F1: w = de/m", _w, 2)
chk("F1: q = n/m", _q, 5)
chk("F1: cells in the block", int(_raw.size), 16)
chk("F1: distinct products in the block", int(len(np.unique(_raw))), 4)
chk("F1: |G_m| = phi(5)", int(_totient(_q)), 4)
chk("F1: the ringed cell 6*12 mod 20", int(6 * 12 % _n), 12)
chk("F1: J_2 is NOT regular at n=20",
    any(j["regular"] for j in _jstruct(_n) if j["d"] == _d), False)
chk("F1: J_3 IS regular at n=15 (identity 6)",
    [j["idempotents"] for j in _jstruct(15) if j["d"] == 3][0], [6])
chk("F1: J_3 * J_3 = J_3 at n=15 (closed)", _gcd(3 * 3, 15), 3)

# F2: strata counts and non-regular counts, per panel.
for _N, _cls, _nr in [(113, 2, 0), (121, 3, 1), (125, 4, 2), (119, 4, 0), (120, 16, 8)]:
    _js = _jstruct(_N)
    chk(f"F2: n={_N} strata", len(_js) ** 2, _cls ** 2)
    chk(f"F2: n={_N} non-regular classes", sum(1 for j in _js if not j["regular"]), _nr)

# F5: the graded pair, and the width of the n=120 lattice.
chk("F5: nu(121)", _nu(121), 2)
chk("F5: nu(125)", _nu(125), 3)
chk("F5: phi(121)", int(_totient(121)), 110)
chk("F5: phi(125)", int(_totient(125)), 100)
chk("F5: n=120 classes", len(_divisors(120)), 16)

# F3: the projection's own arithmetic at n = 121, plus the three losses it prints.
from src.analysis.ablation import unit_logit_grid, ablation_report, key_freq_pairs
_z = np.load("results/k03_grid_acts/WE_B_thesis_n121_s0.npz", allow_pickle=False)
_G, _y, _orders, _kf = unit_logit_grid(_z, 121)
_keep = key_freq_pairs(_kf, _orders)
_qq = _orders[0]
_kept = len({(int(a[0]), int(b[0])) for a, b in _keep} | {(0, 0)})
chk("F3: key characters at n=121 s0", len(_kf), 4)
chk("F3: character pairs in the plane", _qq ** 2, 12100)
chk("F3: pairs kept by Pi_K (with DC)", _kept, 9)
chk("F3: pairs zeroed by Pi_K", _qq ** 2 - _kept, 12091)
_rep = ablation_report(_G, _y, _kf, _orders)
chk("F3: restricted is at or below baseline",
    _rep["restricted"] <= _rep["baseline"], True)
chk("F3: excluded, orders of magnitude above baseline",
    int(round(np.log10(_rep["excluded"] / _rep["baseline"]))), 7)

print("\nFIGURE CAPTIONS -- F6 (C8) and F9 (the falsifier)")
# F6: the enrichment figures the caption quotes, and the positive control.
import test_crt_law as _crt
for _N, _want in [(165, 14.08), (120, 11.76), (119, 3.30)]:
    _W = np.load(f"results/k01_scout/scout_n{_N}_seed0.npz")["W_E"][:_N].astype(float)
    _W = _W - _W.mean(0, keepdims=True)
    _e = _crt.freq_energy(_W, _N)
    _obs, _p, _ = _crt.permutation_test(_e, [k - 1 for k in _crt.predicted(_N)])
    chk(f"F6: n={_N} enrichment", round(_obs, 2), _want, 1e-9)
    # R1 retired `p_perm == 0.0` for the ablation family and left this assertion,
    # on the enrichment family, pinning the exact value methods calls impossible.
    chk(f"F6: n={_N} p is the floor, not zero",
        round(_p, 12), round(1 / (_crt.N_PERM + 1), 12))
chk("F6: |P(165)|", len(_crt.predicted(165)), 8)
chk("F6: |P(120)|", len(_crt.predicted(120)), 7)
chk("F6: |P(119)|", len(_crt.predicted(119)), 11)
chk("F6: P(165) is the published set", _crt.predicted(165),
    [15, 30, 33, 45, 55, 60, 66, 75])
chk("F6: P(120) is the published set", _crt.predicted(120), [15, 24, 30, 40, 45, 48, 60])
chk("F6: a prime power is vacuous -- P(121) is every bin",
    len(_crt.predicted(121)), 121 // 2)

# F9: the six pass rates the figure prints, from the same scorer the script uses.
from scripts.gate2_discriminate import TESTS as _TESTS, load as _load
_grok = _load("results/gate2/k03_grid_acts.json") + _load("results/gate2/n7_engine.json")
_fail = _load("results/gate2/*_controls.json")
_rate = lambda ss, fn: (lambda v: (sum(v), len(v)))([x for x in (fn(s) for s in ss)
                                                     if x is not None])
# FAILED counts moved when the corrected endpoint rule (window median, not last row)
# admitted k04_extended/WE_B_thesis_n49_s2 to the control arm: last row 0.9631 vs window
# median 0.8908. Its one measurable stratum passes G0, G2, G4 and G5, so four of the six
# rows move and three "0 of 19" claims become "1 of 20". Was, before 2026-09-22:
#   G0 (0,19)  G1 (19,19)  G2 (0,19)  G3 (2,6)  G4 (12,19)  G5 (0,13)
_WANT = {"G0 held-out acc >= 0.95":   ((183, 183), (1, 20)),
         "G1 perm p < 0.01":          ((182, 183), (20, 20)),
         "G2 excluded/base >= 100":   ((183, 183), (1, 20)),
         "G3 resid <= 0.5 x ctrl":    ((122, 122), (2, 6)),
         "G4 R >= 5":                 ((183, 183), (13, 20)),
         "G5 tuned - shuffled >= .2": ((109, 111), (1, 14))}
for _name, (_wg, _wb) in _WANT.items():
    chk(f"F9: {_name} grokked", _rate(_grok, _TESTS[_name]), _wg)
    chk(f"F9: {_name} FAILED", _rate(_fail, _TESTS[_name]), _wb)
chk("F9: G1 passes on EVERY failed measurement -- the falsifier fired",
    _rate(_fail, _TESTS["G1 perm p < 0.01"])[0] == _rate(_fail, _TESTS["G1 perm p < 0.01"])[1],
    True)

print("\nFIGURE CAPTIONS -- F8 (the classification rule and the excursions)")
# The figure classifies by the WINDOW MEDIAN, so its caption must quote one too. The
# caption said 0.978 -- which is that run's LAST logged sample, not its window median --
# inside the figure whose whole argument is that the two differ. Same for methods' \bar{a}.
import analyze_excursions as _exc
_runs = sorted(glob.glob("results/k03_grid_acts/WE_B_thesis_n*_s*.npz"))
chk("F8: runs drawn", len(_runs), 30)
_z125 = np.load("results/k03_grid_acts/WE_B_thesis_n125_s1.npz", allow_pickle=True)
_s125, _a125 = _exc.hist_of(_z125)
chk("F8: the near-grok's WINDOW MEDIAN", round(float(np.median(_a125[-10:])), 3), 0.976)
chk("F8: its last sample differs (the defect the figure exists to show)",
    round(float(_a125[-1]), 3), 0.978)
chk("F8: logging interval", int(np.unique(np.diff(_s125))[0]), 200)
_ex = [e for f in _runs for e in
       ((_exc.excursions(*_exc.hist_of(np.load(f, allow_pickle=True))) or {}).get("ex") or [])]
_nrun = sum(1 for f in _runs
            if ((_exc.excursions(*_exc.hist_of(np.load(f, allow_pickle=True))) or {})
                .get("ex") or []))
chk("F8: post-grok excursions below 0.90", len(_ex), 6)
chk("F8: runs containing one", _nrun, 5)
chk("F8: each is one logging interval wide", {e["samples"] for e in _ex}, {1})
chk("F8: none is censored here", sum(e["censored"] for e in _ex), 0)
_h6 = [_exc.excursions(*_exc.hist_of(np.load(f, allow_pickle=True)))
       for f in sorted(glob.glob("results/k06_horizon/WE_B_thesis_n*_s*.npz"))]
_ex6 = [e for r in _h6 if r for e in r["ex"]]
chk("F8: horizon sweep excursions", len(_ex6), 19)
chk("F8: exactly one of them is censored", sum(e["censored"] for e in _ex6), 1)
chk("F8: horizon length",
    int(max(_exc.hist_of(np.load(f, allow_pickle=True))[0][-1]
            for f in glob.glob("results/k06_horizon/WE_B_thesis_n*_s*.npz"))), 120000)

# ---------------------------------------------------------------------------------
# Post-review corrections (2026-09-16). Every number the review moved is re-derived
# here; four of these replaced a value that was wrong in the draft.
# ---------------------------------------------------------------------------------
print("\nPOST-REVIEW -- section 7.3, the prime-power ratios as median + range")
_k09 = [json.loads(str(r)) for r in
        np.load("results/k09_primes_zdd_n4_ablation.npz", allow_pickle=True)["rows"]]
_by = collections.defaultdict(list)
for _r in _k09: _by[_r["n"]].append(_r["baseline"] / _r["restricted"])
# (n, runs, median, lo, hi) -- the table that replaced mean +- sd
for _n, _runs, _med, _lo, _hi in [(128, 2, 239.52, 2.77, 476.26),
                                  (131, 3, 2.88, 2.44, 781.26),
                                  (127, 3, 5.04, 2.18, 51.63),
                                  (91, 2, 117.26, 1.45, 233.07),
                                  (123, 3, 1.76, 0.00, 2.19)]:
    _v = sorted(_by[_n])
    chk(f"7.3: n={_n} runs", len(_v), _runs)
    chk(f"7.3: n={_n} median", round(float(np.median(_v)), 2), _med, 0.01)
    chk(f"7.3: n={_n} range", (round(_v[0], 2), round(_v[-1], 2)), (_lo, _hi))
# R1 (2026-09-21): this used to assert `p_perm == 0.0`, which the Phipson-Smyth
# estimator (1+b)/(B+1) can never produce -- so the old form would pass only while
# the artifact carried a p-value its own draw count could not express. Assert the
# two things that are meaningful: no draw reached the key set, and the reported p is
# the floor. The pre-registration names exactly this replacement.
chk("7.3: 13 of 13 runs beyond every draw", sum(r["n_ge"] == 0 for r in _k09), 13)
chk("7.3: ...and the null was drawn at B = 10,000", sorted({r["n_draws"] for r in _k09}), [10000])
chk("7.3: ...so each reports the floor, not zero",
    sorted({round(r["p_perm"], 12) for r in _k09}), [round(1 / 10001, 12)])
chk("7.3: n=131's mean would be 262x where its median is 2.88x",
    round(float(np.mean(_by[131])), 2), 262.19, 0.01)

print("\nPOST-REVIEW -- section 7.2, the 28/29 count AND its exception")
_k04 = [json.loads(str(r)) for r in
        np.load("results/k04_extended_n4_ablation.npz", allow_pickle=True)["rows"]]
chk("7.2: analysable runs", len(_k04), 29)          # R1-4: B-independent, unchanged
# F4: the abstract's 28/29 sat on a git_dirty=True artifact for four days. Every ablation
# artifact the paper reads must stamp clean; `n7_engine_preleakfix` is retired, not read.
from src import provenance as _prov
chk("every live *_n4_ablation.npz is stamped clean",
    [f for f in sorted(glob.glob("results/*_n4_ablation.npz"))
     if "preleakfix" not in f and (_prov.read(f) or {}).get("git_dirty", True)], [])
chk("7.2: eleven further moduli, not twelve", len({r["n"] for r in _k04}), 11)
# R1 (2026-09-21): 27 -> 28 at B = 10,000. The pre-registration named both marginal tests
# in advance and committed to reporting whatever they became, including 26 or 28.
chk("7.2: pass p<0.01", sum(r["p_perm"] < 0.01 for r in _k04), 28)
_fail = sorted((r["n"], round(r["p_perm"], 4)) for r in _k04 if not r["p_perm"] < 0.01)
chk("7.2: the sole exception is n=98", _fail, [(98, 0.0151)])
# AND score every section that ASSERTS the count, not just the artifact. The conclusion
# carried "27 of 29 ... the two exceptions" for a day after R1 moved it to 28 with ONE
# exception, while this file happily passed against the .npz. Fifth stale-prose defect here.
# Flatten whitespace first: the abstract wraps "$28$\nof $29$" across a line break, so a
# literal match reported ABSENT on text that is present. Same shape as any other check
# whose failure mode is indistinguishable from the defect it hunts.
_pro = {f: " ".join((pathlib.Path("paper/sections") / f).read_text().split())
        for f in ("00-abstract.tex", "13-conclusion.tex", "14-appendices.tex")}
for _f, _txt in _pro.items():
    chk(f"7.2: {_f} carries no stale 27", ("$27$ of $29$" in _txt) or ("$27/29$" in _txt), False)
    chk(f"7.2: {_f} says ONE exception, not two", "two exceptions" in _txt, False)
    _want = "$28/29$" if _f == "14-appendices.tex" else "$28$ of $29$"
    chk(f"7.2: {_f} states the count", _want in _txt, True)
chk("7.2: ...from 150 of 10,000 draws",
    [(r["n_ge"], r["n_draws"]) for r in _k04 if r["n"] == 98 and r["p_perm"] >= 0.01],
    [(150, 10000)])
# n=49 crossed the threshold DOWNWARD and now passes -- on a run that did not grok. That is
# the caveat section 7.2 now carries, so it is asserted here rather than left as prose.
chk("7.2: n=49 now PASSES at 0.0050, on 49 of 10,000 draws",
    [(round(r["p_perm"], 4), r["n_ge"]) for r in _k04 if r["n"] == 49], [(0.0050, 49)])
chk("7.2: ...and that n=49 run did NOT grok (window median, never the last row)",
    round(float(np.median(
        (lambda z: z["hist"][:, [str(c) for c in z["hist_cols"]].index("test_acc")])(
            np.load("results/k04_extended/WE_B_thesis_n49_s2.npz", allow_pickle=True))[-10:])), 3),
    0.891, 0.001)
# the draft named "one seed of 63" as a failure; n=63's run passes -- at 1 draw of 10,000,
# where the 200-draw null could only say "0 of 200".
chk("7.2: n=63 PASSES -- the draft named it as a failure",
    [round(r["p_perm"], 4) for r in _k04 if r["n"] == 63], [0.0002])

print("\n7.1 -- k03 at B = 10,000: the headline that sharpened rather than held")
_k03p = [json.loads(str(r)) for r in
         np.load("results/k03_grid_acts_n4_ablation.npz", allow_pickle=True)["rows"]]
chk("7.1: 31 runs", len(_k03p), 31)
chk("7.1: 31/31 pass p<0.01", sum(r["p_perm"] < 0.01 for r in _k03p), 31)
# "not one of 200 is as damaging" becomes "not one of 10,000, in 26 of 31" -- the
# other five are beaten by 1-2 draws, which 200 draws could not have resolved.
chk("7.1: beyond EVERY draw in 26 of 31", sum(r["n_ge"] == 0 for r in _k03p), 26)
chk("7.1: the other five are beaten by 1 or 2 draws",
    sorted({r["n_ge"] for r in _k03p if r["n_ge"]}), [1, 2])
chk("7.1: so every run reads p <= 0.0003", round(max(r["p_perm"] for r in _k03p), 4), 0.0003)
chk("7.1: F4's panel run n=121 s0 is at the floor, 0 of 10,000",
    [(r["n_ge"], round(r["excluded"], 4)) for r in _k03p if r["tag"] == "B_thesis_n121_s0"],
    [(0, 12.8785)])
# P4: F4's own panel said "(200 draws)" and "p = 0.0000" (the retired b/B) above a caption
# saying 10,000 draws and the 1.0e-4 floor. The figure now derives both from the draws.
from src.analysis.stats import perm_p as _pp
_f4 = [r for r in _k03p if r["tag"] == "B_thesis_n121_s0"][0]
chk("P4: F4's p-hat = the caption's floor 1.0e-4, over 10,000 draws",
    (f"{_pp(_f4['n_ge'], _f4['n_draws']):.1e}", _f4["n_draws"]), ("1.0e-04", 10000))
import inspect as _insp, src.viz.plots as _plots
_src = _insp.getsource(_plots.causal_test)
chk("P4: causal_test derives its draw count and p (no literal)",
    ("(200 draws)" in _src, "len(draws):,} draws" in _src, "perm_p(n_ge, len(draws))" in _src),
    (False, True, True))

print("\n7.4 -- the initialisation sweep (k08), which nothing asserted until R1")
_k08 = [json.loads(str(r)) for r in
        np.load("results/k08_init_n4_ablation.npz", allow_pickle=True)["rows"]]
chk("7.4: 24 runs, three conditions x eight seeds", len(_k08), 24)
chk("7.4: three initialisation conditions",
    len({r["tag"].rsplit("_n", 1)[0] for r in _k08}), 3)
chk("7.4: 24/24 pass p<0.01", sum(r["p_perm"] < 0.01 for r in _k08), 24)
chk("7.4: ...and all 24 are at the floor, 0 of 10,000 draws",
    (sum(r["n_ge"] == 0 for r in _k08), sorted({r["n_draws"] for r in _k08})), (24, [10000]))

print("\nPOST-REVIEW -- section 8, the measurability cut and the G5 count")
import analyze_gate2 as _g2
chk("8: cut is |G_m| >= 10", _g2.G_MIN, 10)
chk("8: cut is >= 100 held-out cells", _g2.HELDOUT_MIN, 100)
_m2 = [(r, s2) for r in d["runs"] for s2 in r["strata"] if s2.get("measurable")]
_g5 = [(r, s2, s2["tuned"] - s2["tuned_sh"]) for r, s2 in _m2
       if s2.get("tuned") is not None and s2.get("tuned_sh") is not None]
chk("8: G5 is scored on 87 of the 157", len(_g5), 87)
chk("8: G5 passes 85 of 87", sum(x >= 0.20 for _, _, x in _g5), 85)
chk("8: the two below threshold are n=165's J5xJ15 and J15xJ5",
    sorted((r["n"], s2["d"], s2["e"], round(x, 4)) for r, s2, x in _g5 if x < 0.20),
    [(165, 5, 15, 0.1934), (165, 15, 5, 0.1934)])
chk("8: 157 + 26 engine = 183",
    len(_m2) + len([s2 for r in json.load(open("results/gate2/n7_engine.json"))["runs"]
                    for s2 in r["strata"] if s2.get("measurable")]), 183)

print("\nPOST-REVIEW -- section 1, the corrected inequality, and section 9's count")
# The draft said nilpotents make m EXCEED gcd(d,n)gcd(e,n) = de. It cannot: m = gcd(de,n) <= de.
_viol = [(n, dd, ee) for n in range(2, 200) for dd in _divisors(n) for ee in _divisors(n)
         if _gcd(dd * ee, n) > dd * ee]
chk("1: m = gcd(de,n) never exceeds de", len(_viol), 0)
chk("1: and it does reach de (the ceiling the text now claims)",
    any(_gcd(dd * ee, n) == dd * ee for n in range(2, 60)
        for dd in _divisors(n) for ee in _divisors(n) if dd * ee > 1), True)
from sympy import factorint as _factorint
_crt12 = [54, 63, 75, 98, 99, 100, 105, 143, 147, 165, 120, 119]
chk("9: 8 of those 12 moduli are non-square-free",
    sum(1 for n in _crt12 if any(e > 1 for e in _factorint(n).values())), 8)

# Section 1 now states J_d . J_e = J_gcd(de,n) with EQUALITY, not inclusion (the panel's
# Domain reviewer: the theorem proves more than the draft claimed). Equality is the half a
# reader cannot check by eye, so it is checked here over every divisor pair of every n <= 200.
_pairs = _mismatch = 0
for n in range(2, 201):
    _J = {}
    for x in range(n):
        _J.setdefault(_gcd(x, n), set()).add(x)
    for dd in _divisors(n):
        for ee in _divisors(n):
            _pairs += 1
            if {(x * y) % n for x in _J[dd] for y in _J[ee]} != _J[_gcd(dd * ee, n)]:
                _mismatch += 1
chk("1: J_d . J_e = J_gcd(de,n) pairs checked", _pairs, 8253)
chk("1: ...with EQUALITY, no mismatches", _mismatch, 0)

# ---------------------------------------------------------------------------------
# R2 -- cross-stratum sameness (section 8, the panel's load-bearing objection).
print("\nR2 -- the same clock, or one clock each")
_r2 = json.load(open("results/gate2/r2_sameness_k03_grid_acts.json"))
_s2 = _r2["summary"]
chk("8: primary pairs", _s2["n_pairs"], 160)
chk("8: exactly equal", _s2["n_equal"], 158)
chk("8: agreement rate", round(_s2["rate"], 4), 0.9875, 1e-4)
chk("8: Monte-Carlo null at the floor", _s2["p_null"], 1 / 10001, 1e-9)
chk("8: cell-shuffle control", _s2["rate_shuffled"], 0.0)
chk("8: transpose pairs (descriptive)", (_s2["n_transpose_equal"], _s2["n_transpose"]), (54, 54))
chk("8: median |K|", _s2["median_K"], 2)
chk("8: no pair excluded for an empty key set", _s2["n_empty"], 0)
_jac = [x["jaccard"] for r in _r2["runs"] for x in r["primary"]]
chk("8: mean Jaccard", round(float(np.mean(_jac)), 4), 0.9982, 1e-4)
# the two disagreements are BOTH n=165 seed 1, which the section names
_bad = [(r["n"], r["seed"]) for r in _r2["runs"] for x in r["primary"] if not x["equal"]]
chk("8: both disagreements are n=165 seed 1", sorted(set(_bad)), [(165, 1)])
chk("8: ...and there are two of them", len(_bad), 2)
# the structural counts, which are algebra and must not depend on any model
import analyze_r2_sameness as _r2m
for _n, _pt in [(165, (28, 6)), (120, (2, 2)), (119, (2, 1)), (121, (0, 1)), (125, (0, 1)),
                (113, (0, 0))]:
    _pp, _tt = _r2m.pairs_for(_n)
    chk(f"8: n={_n} structural pairs (primary, transpose)", (len(_pp), len(_tt)), _pt)
# chance agreement at G_11 is 1/C(5,2) = 0.10, which is why the null is not a formality
chk("8: G_11 folded bins", len(_r2m.folded_labels((10,))), 5)
chk("8: chance of two independent 2-sets coinciding at G_11",
    round(1 / math.comb(5, 2), 2), 0.10)

# R2's control: 2 scorable pairs, both from ONE run at accuracy 0.7502.
_r2c = json.load(open("results/gate2/r2_sameness_controls_k04_extended.json"))
chk("8: control scorable pairs", _r2c["summary"]["n_pairs"], 2)
chk("8: control rate (the criterion that FAILED)", _r2c["summary"]["rate"], 1.0)
chk("8: control's own null is uninformative", _r2c["summary"]["p_null"], 1.0)
_acc = sorted(round(r["acc"], 4) for r in _r2c["runs"] if r["primary"])
chk("8: the control's pairs come from runs at these accuracies", _acc, [0.2733, 0.7502])
chk("8: and the 0.2733 run has NO key set at all",
    [x["empty"] for r in _r2c["runs"] if round(r["acc"], 4) == 0.2733 for x in r["primary"]],
    [True, True])

# ---------------------------------------------------------------------------------
# R3 -- the indicators are enriched on P(n) but do not select it (section 9).
print("\nR3 -- what the stratification alone does and does not do")
import test_crt_law as _crt
for _n, _npred, _nkey, _key in [(165, 8, 3, [33, 55, 66]), (120, 7, 3, [20, 40, 60])]:
    _r = _crt.indicator_setequality(_n)
    chk(f"9: |P({_n})|", len(_r["predicted"]), _npred)
    chk(f"9: indicator key set at n={_n}", _r["key"], _key)
    chk(f"9: ...does NOT equal P({_n})", _r["equal"], False)
    _W = np.load(f"results/k01_scout/scout_n{_n}_seed0.npz")["W_E"][:_n].astype(float)
    _W = _W - _W.mean(0, keepdims=True)
    _kW = sorted(int(i) + 1 for i in key_freqs_5x_median(_crt.freq_energy(_W, _n)))
    chk(f"9: the NETWORK's key set at n={_n} IS P(n)", _kW, _r["predicted"])
chk("9: 20 = 120/6 is the dual of a non-maximal-prime-power divisor",
    (120 // 6, 6 in [d for d in range(2, 120) if 120 % d == 0],
     20 in _crt.predicted(120)), (20, True, False))
_duals165 = sorted({m for d in range(2, 165) if 165 % d == 0
                    for m in range(165 // d, 165 // 2 + 1, 165 // d)})
chk("9: the stratification's candidate family at n=165 is the duals of EVERY divisor",
    len(_duals165), 42)
# ...and it is where the energy concentrates, not its support (the sentence said "support"
# until 2026-09-28; a Ramanujan sum never vanishes at square-free q, so all 82 are nonzero).
_e165 = _crt.freq_energy(_crt.jclass_indicators(165), 165) ** 2
_share = np.array([np.gcd(k, 165) > 1 for k in range(1, 165 // 2 + 1)])
chk("9: the 42 are exactly the frequencies sharing a factor with 165",
    sorted(int(k) + 1 for k in np.flatnonzero(_share)), _duals165)
chk("9: indicator energy is nonzero at all 82 frequencies (support is NOT the 42)",
    int((_e165 > 1e-9).sum()), 82)
# The closed form the paragraph states, c_q(k) = mu(q/g) phi(q)/phi(q/g), q = n/d,
# g = gcd(k, q), is the DFT of 1_{J_d} -- checked at every d | n, every k, every trained modulus.
from sympy import mobius as _mu, totient as _phi
_rbad = _rmax = 0
for _n in sorted(mt.grok_counts()):
    for _d in [d for d in range(1, _n + 1) if _n % d == 0]:
        _q = _n // _d
        _ft = np.fft.fft([1.0 if _gcd(x, _n) == _d else 0.0 for x in range(_n)])
        for _k in range(_n):
            _g = _gcd(_k, _q)
            _c = int(_mu(_q // _g)) * int(_phi(_q)) // int(_phi(_q // _g))
            _rbad += abs(_ft[_k] - _c) > 1e-6
            if _gcd(_k, _n) == 1:
                _rmax = max(_rmax, abs(_c))
chk("9: 1_{J_d} transform == Ramanujan closed form, all d|n, k, 23 moduli", int(_rbad), 0)
chk("9: ...magnitude at most 1 wherever k is coprime to n", _rmax <= 1, True)
chk("9: the 42 carry 30.5x the mean energy of the other 40",
    round(_e165[_share].mean() / _e165[~_share].mean(), 1), 30.5)
for _n, _e in [(165, 4.87), (120, 3.32), (119, 7.15)]:
    _obs, _p, _ = _crt.permutation_test(_crt.freq_energy(_crt.jclass_indicators(_n), _n),
                                        [k - 1 for k in _crt.predicted(_n)])
    chk(f"9: indicator enrichment at n={_n}", round(_obs, 2), _e, 0.01)
# R3-5's positive control, on the artifacts the published values came from
for _n, _want in [(165, 17.44), (119, 3.27)]:
    _W = np.load(f"results/k03_grid_acts/WE_B_thesis_n{_n}_s0.npz")["W_E"][:_n].astype(float)
    _W = _W - _W.mean(0, keepdims=True)
    _obs, _, _ = _crt.permutation_test(_crt.freq_energy(_W, _n),
                                       [k - 1 for k in _crt.predicted(_n)])
    chk(f"9: positive control, k03 n={_n}", round(_obs, 2), _want, 0.01)

# ---------------------------------------------------------------------------------
# R1 -- the multiplicity paragraph's arithmetic (section 5).
print("\nR1 -- the permutation family and what it clears")
_B = 10000
chk("5: the floor a B-draw null can express", round(1 / (_B + 1), 6), round(1e-4, 6), 1e-6)
chk("5: Bonferroni at alpha=0.05 clears", 0.05 / 320 > 1 / (_B + 1), True)
chk("5: Bonferroni at alpha=0.01 does NOT clear", 0.01 / 320 > 1 / (_B + 1), False)
chk("5: ...and would need B >= 32,000", math.ceil(320 / 0.01) - 1 >= 32000 - 1, True)

# Section 1 now DEFINES nilpotent and asserts when they exist. The clause is checkable.
from sympy import factorint as _fi
_bad = [n for n in range(2, 500)
        if any(any(pow(x, k, n) == 0 for k in range(2, 20)) for x in range(1, n))
        == all(e == 1 for e in _fi(n).values())]
chk("1: nilpotents exist in Z/nZ iff n is NOT square-free", len(_bad), 0)

# ---------------------------------------------------------------------------
# F9's table is asserted above against the ARTIFACT. It is also written out, in
# prose, in FINDINGS.md -- and a test that scores one document while another
# asserts the same number is exactly how STATE's C33 row stayed stale for six
# days after the correction reached FINDINGS and the paper. So score the
# DOCUMENT against the scorer too, row by row.
# FINDINGS.md and the pre-registrations are not in the public repository, so the checks
# that score them are skipped (and say so) where they are absent, as for STATE.md below.
_FINDINGS = pathlib.Path(__file__).with_name("FINDINGS.md")
if _FINDINGS.exists():
    print("\nF9 -- FINDINGS.md's table scored against the scorer, not against the paper")
    _find = _FINDINGS.read_text()
    _row = re.compile(r"^\|\s*\*\*(G\d)\*\*[^|]*\|[^|]*?(\d+)/(\d+)[^|]*\|[^|]*?(\d+)/(\d+)[^|]*\|",
                      re.M)
    _doc = {m.group(1): ((int(m.group(2)), int(m.group(3))),
                         (int(m.group(4)), int(m.group(5)))) for m in _row.finditer(_find)}
    chk("F9: FINDINGS' table has all six criterion rows", sorted(_doc), ["G0","G1","G2","G3","G4","G5"])
    for _name, (_wg, _wb) in _WANT.items():
        _g = _name.split()[0]
        # chk(label, got, want) prints paper={want} artifact={got}, so the SCORER value is
        # `got` and the DOCUMENT value is `want`. Passing them the other way round printed
        # the planted stale row under "artifact" -- the fourth upside-down column here.
        chk(f"F9: FINDINGS row {_g} grokked matches the scorer", _wg, _doc.get(_g, (None, None))[0])
        chk(f"F9: FINDINGS row {_g} FAILED matches the scorer",  _wb, _doc.get(_g, (None, None))[1])
else:
    print("\nFINDINGS.md absent (public repository): its F9 table checks are skipped")
# And the two population totals, which the paper states in prose in two places.
_tex = (pathlib.Path(__file__).parent / "paper/sections/08-results-strata.tex").read_text()
_ng, _nf = _WANT["G1 perm p < 0.01"][0][1], _WANT["G1 perm p < 0.01"][1][1]
chk(f"F9: section 8 body states the {_ng}/{_nf} populations", (True, True),
    (f"${_ng}$ grokked and ${_nf}$ failed stratum-measurements" in _tex,
     f"(open, ${_nf}$)" in _tex))

# The working repo's resume file asserts a notebook entry count; the count is derivable
# from the notebook itself, so it is checked here when that file is present.
# STATE.md is the working repo's resume file and is not part of the public export, so the
# two checks that score it are skipped (and say so) where it is absent.
_STATE = pathlib.Path(__file__).with_name("STATE.md")
if _STATE.exists():
    print("\nSTATE.md header -- the entry count scored against the notebook")
    _state = _STATE.read_text()
    _nb = pathlib.Path(__file__).with_name("LAB_NOTEBOOK.md").read_text()
    _real = len(re.findall(r"^## 20\d\d-\d\d-\d\d — Entry ", _nb, re.M))
    _m = re.search(r"`LAB_NOTEBOOK\.md` has the \*\*(\d+)\*\* dated entries", _state)
    chk("STATE header states a notebook entry count", _m is not None, True)
    chk("STATE's entry count matches LAB_NOTEBOOK", _real, int(_m.group(1)) if _m else None)
else:
    print("\nSTATE.md absent (public export): its checks are skipped")


# C39 (I1) -- the INTERNAL intervention. Every value FINDINGS 3.3b asserts is recomputed
# from the artifact here, including the two that are easiest to get wrong: the chance level
# the excluded accuracy is compared against, and the zero-overlap null, which is the whole
# reason the heavy tail is not a problem.
from src.analysis.intervention import random_character_sets as _rcs

def _i1_arm(tag, path, n, doc, zero):
    """Score one I1 arm: every per-seed value against `doc`, and the zero-overlap null
    (what makes the heavy tail benign) against `zero`. One body for both moduli, so C39
    and C40 cannot drift into two answers. Returns the per-seed excluded/baseline ratios
    and each seed's key-overlap count per null draw."""
    ratios, overlaps = [], {}
    for _s, (_b, _r, _e, _ar, _ae, _rt, _pp, _ng) in doc.items():
        _z = np.load(path % _s, allow_pickle=True)
        chk(f"{tag} s{_s} modulus", int(_z["n"]), n)
        chk(f"{tag} s{_s} baseline",   float(_z["internal_baseline"]),   _b,  abs(_b) * 5e-4)
        chk(f"{tag} s{_s} restricted", float(_z["internal_restricted"]), _r,  abs(_r) * 5e-4)
        chk(f"{tag} s{_s} excluded",   float(_z["internal_excluded"]),   _e,  abs(_e) * 5e-4)
        chk(f"{tag} s{_s} restricted accuracy", float(_z["acc_restricted"]), _ar, 1e-4)
        chk(f"{tag} s{_s} excluded accuracy",   float(_z["acc_excluded"]),   _ae, 1e-4)
        chk(f"{tag} s{_s} excluded/baseline",   float(_z["ratio_excluded_baseline"]), _rt, abs(_rt) * 1e-3)
        chk(f"{tag} s{_s} p_perm",              float(_z["p_perm"]), _pp, 1e-9)
        chk(f"{tag} s{_s} draws >= excluded",   int(_z["n_ge"]), _ng)
        chk(f"{tag} s{_s} verdict all three held",
            json.loads(str(_z["verdict"])), {"I1_necessity": True, "I2_sufficiency": True,
                                             "I3_specificity": True})
        chk(f"{tag} s{_s} reproduces archive exactly",
            json.loads(str(_z["repro"]))["max_dW_E"], 0.0)
        ratios.append(float(_z["ratio_excluded_baseline"]))
    # Regenerating the draws is deterministic (seed=0), exactly as run_intervention does.
    for _s, _want_zero in zero.items():
        _z = np.load(path % _s, allow_pickle=True)
        _kf = [k[0] for k in json.loads(str(_z["key_freqs"]))]
        _dr = _rcs(_kf, n, int(_z["draws"]), seed=0)
        _ov = np.array([len(set(_kf) & set(d)) for d in _dr])
        _c, overlaps[_s] = _z["control"], _ov
        chk(f"{tag} s{_s} zero-overlap draw count", int((_ov == 0).sum()), _want_zero)
        chk(f"{tag} s{_s} zero-overlap draws never reach excluded",
            bool((_c[_ov == 0] < float(_z["internal_excluded"])).all()), True)
        chk(f"{tag} s{_s} zero-overlap mean is within 1.2x of baseline",
            bool(_c[_ov == 0].mean() / float(_z["internal_baseline"]) < 1.2), True)
    return ratios, overlaps

print("\nC39 / I1 -- results/i1_internal/I1_engine_n113_s*.npz")
_I1 = "results/i1_internal/I1_engine_n113_s%d.npz"
if os.path.exists(_I1 % 0):
    _ratios, _ = _i1_arm("I1", _I1, 113, {  # exactly as FINDINGS 3.3b's table prints them
        0: (1.2102e-05, 2.0775e-06, 4.7136e+01, 1.0000, 0.0089, 3.895e+06, 1/10001, 0),
        1: (1.2743e-07, 3.2178e-07, 3.5425e+01, 1.0000, 0.0089, 2.780e+08, 1/10001, 0),
        2: (1.5480e-07, 2.2235e-07, 2.3883e+01, 1.0000, 0.0091, 1.543e+08, 5/10001, 4),
    }, {0: 2651, 1: 4892, 2: 4925})
    # median + range + n, never a mean and never a +- at n = 3 (C36)
    chk("I1 excluded/baseline median over 3 seeds", float(np.median(_ratios)), 1.543e+08, 1.543e5)
    chk("I1 excluded/baseline min",                 float(min(_ratios)),       3.895e+06, 3.895e3)
    chk("I1 excluded/baseline max",                 float(max(_ratios)),       2.780e+08, 2.780e5)
    # The chance level the excluded accuracy is claimed to equal. FINDINGS says 1/112; the
    # grid is over units only, so 112 is the right denominator and 113 would be wrong.
    chk("I1 chance level is 1/112", round(1.0 / 112, 5), 0.00893)
    chk("I1 zero-overlap draws total across seeds", 2651 + 4892 + 4925, 12468)
    # F10's caption asserts these; a caption is a number-carrying sentence and is read
    # more often than the body, so it is scored like any other.
    _cap = " ".join((pathlib.Path("paper/sections") / "07-results-causal.tex").read_text().split())
    chk("F10 caption: 30,000 total draws", "$30{,}000$ null draws" in _cap, True)
    chk("F10 caption: 12,468 zero-overlap", "$12{,}468$ draws that remove" in _cap, True)
    chk("F10 caption: band is 3.895e6 to 2.780e8",
        ("$3.895 \\times 10^{6}$" in _cap) and ("$2.780 \\times 10^{8}$" in _cap), True)
    chk("I1 total draws across seeds is 30000", sum(int(np.load(_I1 % s, allow_pickle=True)["draws"])
                                                    for s in (0, 1, 2)), 30000)
else:
    print("  [SKIP] I1 artifacts absent")


# C40 (I1b) -- the same instrument at n = 121 = 11^2, the first non-square-free arm. Scored
# against the pre-registration's Outcome table AND STATE's C40 row, because those are the
# only two documents that carry it and an unasserted number is how F7 sat stale.
print("\nC40 / I1b -- results/i1_internal/I1_engine_n121_s*.npz")
_I1b = "results/i1_internal/I1_engine_n121_s%d.npz"
if os.path.exists(_I1b % 0):
    _zb = [np.load(_I1b % s, allow_pickle=True) for s in (0, 1, 2)]
    _rb, _ovb = _i1_arm("I1b", _I1b, 121, {  # exactly as the pre-registration's Outcome prints them
        0: (1.7482e-06, 3.4061e-07, 5.0365e+01, 1.0000, 0.0000, 2.881e+07, 1/10001, 0),
        1: (7.7458e-03, 1.0560e-01, 5.7783e+01, 0.9596, 0.0120, 7.460e+03, 1/10001, 0),
        2: (1.6364e-03, 2.2674e-03, 1.8467e+01, 1.0000, 0.0000, 1.128e+04, 3/10001, 2),
    }, {0: 4825, 1: 7304, 2: 7297})
    # C36: median + range + n = 3, never a mean or a +-
    chk("I1b excluded/baseline median over 3 seeds", float(np.median(_rb)), 1.128e+04, 1.128e1)
    chk("I1b excluded/baseline min",                 float(min(_rb)),       7.460e+03, 7.460e0)
    chk("I1b excluded/baseline max",                 float(max(_rb)),       2.881e+07, 2.881e4)
    chk("I1b every seed clears I1's 100x",          min(_rb) >= 100, True)
    # CHANCE is 1/110, not 1/120 or 1/121: the grid is the 110 x 110 UNITS of Z/121.
    from src.tasks.algebra import units as _units
    chk("I1b chance level is 1/|units(121)| = 1/110", len(_units(121)), 110)
    chk("I1b excluded acc BELOW chance on 2 of 3 seeds",
        sum(float(z["acc_excluded"]) < 1 / 110 for z in _zb), 2)
    # necessary-not-sufficient signature: restricted LOSS worse than baseline on s1, s2
    chk("I1b restricted/baseline s1 (13.6x)",
        round(float(_zb[1]["internal_restricted"]) / float(_zb[1]["internal_baseline"]), 1), 13.6)
    chk("I1b restricted/baseline s2 (1.39x)",
        round(float(_zb[2]["internal_restricted"]) / float(_zb[2]["internal_baseline"]), 2), 1.39)
    # The null's tail: report the control maximum, not only the median. Seed 2's exceeds the
    # effect (101.2%) and excluded/median(control) hides it completely.
    chk("I1b control max / excluded, per seed (%)",
        [round(100 * float(z["control_max"]) / float(z["internal_excluded"]), 1) for z in _zb],
        [89.8, 85.9, 101.2])
    chk("I1b s2 control max", float(_zb[2]["control_max"]), 18.6887, 1e-4)
    chk("I1b excluded/median(control), per seed",
        [float(f"{float(z['internal_excluded']) / float(z['control_median']):.3g}") for z in _zb],
        [3.12e+06, 8.84e+03, 1.09e+04])
    chk("I1b s2: both beating draws overlap 3 of |K|=4",
        sorted(_ovb[2][_zb[2]["control"] >= float(_zb[2]["internal_excluded"])].tolist()), [3, 3])
    chk("I1b zero-overlap draws total across seeds", 4825 + 7304 + 7297, 19426)
    chk("I1b zero-overlap max over all seeds (1.71e-02)",
        round(max(float(z["control"][_ovb[s] == 0].max()) for s, z in enumerate(_zb)), 4), 0.0171)
    chk("I1b key sets", [[k[0] for k in json.loads(str(z["key_freqs"]))] for z in _zb],
        [[2, 22, 40, 44, 48, 55], [3, 11, 18, 42], [22, 44, 48, 55]])
    # grok step from the retrained run's own history, read BY COLUMN NAME (hist[-1][3] trap)
    _gs = []
    for s in (0, 1, 2):
        _w = np.load(f"results/i1_internal/WE_engine_n121_s{s}.npz", allow_pickle=True)
        _h, _cols = _w["hist"], list(_w["hist_cols"])
        _te = _h[:, _cols.index("test_acc")]
        _gs.append(int(_h[np.argmax(_te > 0.99), _cols.index("step")]))
    chk("I1b grok steps", _gs, [10700, 10300, 5000])
    chk("I1b artifacts stamped clean at 266b77d",
        {(json.loads(str(z["provenance"]))["git_sha"][:7], json.loads(str(z["provenance"]))["git_dirty"])
         for z in _zb}, {("266b77d", False)})
    # THE PROSE: the two documents that carry C40 today, scored against the artifact.
    _PRE = pathlib.Path("experiments/PREREGISTER_i1b_composite_intervention.md")
    _pre = " ".join(_PRE.read_text().split()) if _PRE.exists() else None
    _c40 = " ".join(next(l for l in _STATE.read_text().splitlines()
                         if l.startswith("| **C40**")).split()) if _STATE.exists() else None
    for _doc_name, _txt in ((("prereg", _pre),) if _pre else ()) + ((("STATE C40 row", _c40),) if _c40 else ()):
        chk(f"I1b {_doc_name}: median 1.128e+04", "1.128e+04" in _txt, True)
        chk(f"I1b {_doc_name}: range 7.460e+03-2.881e+07", "7.460e+03–2.881e+07" in _txt, True)
    if _c40:
        chk("I1b STATE: excluded acc 0.0000 / 0.0120 / 0.0000", "0.0000 / 0.0120 / 0.0000" in _c40, True)
        chk("I1b STATE: restricted acc 1.0000 / 0.9596 / 1.0000", "1.0000 / 0.9596 / 1.0000" in _c40, True)
        chk("I1b STATE: p 1/10001, 1/10001, 3/10001", "1/10001, 1/10001, 3/10001" in _c40, True)
        chk("I1b STATE: 19,426 zero-overlap draws", "19,426 zero-overlap" in _c40, True)
        chk("I1b STATE: grok 10,700/10,300/5,000", "10,700/10,300/5,000" in _c40, True)
    if _pre:
        chk("I1b prereg: grok 10,700 / 10,300 / 5,000", "10,700 / 10,300 / 5,000" in _pre, True)
    # EXPLORATORY dose table (FINDINGS 3.3c, prereg Outcome): mean null damage by overlap
    chk("I1b dose table, mean damage by key overlap",
        [[f"{float(z['control'][_ovb[s] == o].mean()):.2e}" for o in range(_ovb[s].max() + 1)]
         for s, z in enumerate(_zb)],
        [["1.35e-06", "6.36e+00", "1.37e+01", "2.28e+01", "2.82e+01"],
         ["5.92e-03", "1.13e+01", "2.70e+01", "2.96e+01"],
         ["1.70e-03", "5.47e+00", "9.28e+00", "1.63e+01"]])
    chk("I1b zero-overlap mean / baseline per seed",
        [round(float(z["control"][_ovb[s] == 0].mean()) / float(z["internal_baseline"]), 2)
         for s, z in enumerate(_zb)], [0.77, 0.76, 1.04])
    # FINDINGS 3.3c and the paper's section 7 -- the documents a reader actually opens.
    if _FINDINGS.exists():
        _f33c = " ".join(pathlib.Path("FINDINGS.md").read_text().split("### 3.3c", 1)[1]
                         .split("\n### ", 1)[0].split())
        for _needle in ("1.128e+04, range 7.460e+03–2.881e+07, n = 3", "0.0000 / 0.0120 / 0.0000",
                        "1.0000 / 0.9596 / 1.0000", "19,426 zero-overlap", "1.71e-02",
                        "10,700 / 10,300 / 5,000", "101.2 %", "13.6× and 1.39×", "0.77× / 0.76× / 1.04×"):
            chk(f"I1b FINDINGS 3.3c: {_needle}", _needle in _f33c, True)
    _s7 = " ".join(pathlib.Path("paper/sections/07-results-causal.tex").read_text().split())
    for _needle in ("$1.128 \\times 10^{4}$ (range $7.460 \\times 10^{3}$--$2.881 \\times 10^{7}$)",
                    "$7.46 \\times 10^{3}$", "$0.0000$, $0.0120$ and $0.0000$", "$1/110 = 0.00909$",
                    "$0.9596$", "$18.6887$", "$18.4667$", "$19{,}426$", "$1.71 \\times 10^{-2}$",
                    "$10{,}700$, $10{,}300$ and $5{,}000$", "$13.6\\times$ and $1.39\\times$",
                    "& $3/10001$"):
        chk(f"I1b section 7: {_needle}", _needle in _s7, True)
    # PROPAGATION: a correction reaches the section someone is editing, not the others.
    # The modulus count is read from the setup table itself, so "the other N" is scored
    # structurally -- "the other twenty-three" sat in the conclusion against a 23-row table.
    _setup = pathlib.Path("paper/sections/01-setup.tex").read_text()
    _tab = _setup.split("\\label{sec:setup-moduli}", 1)[1].split("\\bottomrule", 1)[0]
    _nmod = len(re.findall(r"^\s*\d+ & ", _tab, re.M))
    chk("the paper's modulus table has 23 rows", _nmod, 23)
    _words = {21: "twenty-one", 22: "twenty-two", 23: "twenty-three"}
    _secs = {f.name: " ".join(f.read_text().split())
             for f in pathlib.Path("paper/sections").glob("*.tex")}
    for _f in ("05-methods.tex", "07-results-causal.tex", "13-conclusion.tex"):
        chk(f"I1b {_f}: 'the other {_words[_nmod - 2]}'",
            f"the other {_words[_nmod - 2]}" in _secs[_f], True)
    for _stale in ("At one modulus we do claim more", "At $n = 113$ only",
                   "the other twenty-three", "the other twenty-two",
                   "--- at one modulus, which"):  # bare "at one modulus" is grok-time prose in 6 and 12
        chk(f"I1b no section says '{_stale}'",
            [f for f, t in _secs.items() if _stale in t], [])
    chk("I1b appendix ledger has a C40 row",
        "C40 & the same at a non-square-free modulus" in _secs["14-appendices.tex"], True)
    for _needle in ("$0.0000$--$0.0120$ at $n = 121 = 11^{2}$ (chance $1/110$)",
                    "$0.0089$--$0.0091$ at the prime $n = 113$ (chance $1/112$)",
                    "accuracy at $0.9596$ or above"):
        chk(f"I1b abstract: {_needle}", _needle in _secs["00-abstract.tex"], True)
    chk("I1b abstract/conclusion bound: max excluded acc <= 0.0120",
        round(max(float(z["acc_excluded"]) for z in _zb), 4), 0.0120)
    chk("I1b section 7: excluded range 18--58 matches artifact",
        (round(min(float(z["internal_excluded"]) for z in _zb)),
         round(max(float(z["internal_excluded"]) for z in _zb))), (18, 58))
else:
    print("  [SKIP] I1b artifacts absent")


# ---------------------------------------------------------------------------
# The retraction count, scored structurally and then against the prose that states
# it. The appendix's own \paragraph entries are the source here: a prose-vs-prose check
# would only prove two sentences agree. Add a retraction and this fires until the prose
# moves with it.
print("\nOMEGA (C37) -- section 9's correlations, scored against the artifacts")
# Score the documents, not just the artifact: FINDINGS and section 9 once carried the
# 18-modulus rho(n, Gini_add) = +0.015 after k09 took the population to 23 and the value
# to -0.185.
import analyze_omega as _ao
from src.analysis.stats import spearman as _sp, partial_spearman as _psp
_d = _ao.collect()
_N_ = [x["n"] for x in _d]; _W_ = [x["omega"] for x in _d]
_Z_ = [x["zdd"] for x in _d]; _G_ = [x["ga"] for x in _d]
chk("9: the omega population is 23 moduli", len(_d), 23)
chk("9: W5 rho(n, Gini_add)", round(_sp(_N_, _G_), 3), -0.185, 0.0015)
chk("9: W5 holds, |rho| <= 0.40", abs(_sp(_N_, _G_)) <= 0.40, True)
chk("9: W2 rho(omega, G | zdd)", round(_psp(_W_, _G_, _Z_), 3), 0.804, 0.0015)
chk("9: W2 holds, >= +0.50", _psp(_W_, _G_, _Z_) >= 0.50, True)
chk("9: W3 rho(zdd, G | omega)", round(_psp(_Z_, _G_, _W_), 3), 0.578, 0.0015)
# the ordering the prose asserts -- under MIDRANKS omega's partial is the higher one
chk("9: omega's partial exceeds zdd's", _psp(_W_, _G_, _Z_) > _psp(_Z_, _G_, _W_), True)
_pro9 = " ".join(pathlib.Path("paper/sections/09-results-crt.tex").read_text().split())
chk("9: section 9 carries no stale +0.015", "+0.015$" in _pro9, False)
chk("9: section 9 states -0.185", "-0.185" in _pro9, True)
chk("9: section 9 no longer calls the divisor duals the SUPPORT",
    "spectral support is the duals" in _pro9, False)
chk("9: section 9 states the 30.5x concentration", "$30.5\\times$" in _pro9, True)
# P1 (revision WP0, 2026-09-28): the Limitations section still said zdd "remains the stronger
# partial correlate" five days after the midrank audit made section 9 say this data "cannot
# rank them". The sentence carries no number, so the old-value grep that closes a correction
# had nothing to find. Score the words, in every section.
_secs9 = {f.name: " ".join(f.read_text().split())
          for f in pathlib.Path("paper/sections").glob("*.tex")}
chk("P1: no section calls either predictor the stronger partial correlate",
    [f for f, t in _secs9.items() if "stronger partial correlate" in t], [])
chk("P1: the Limitations section says this data cannot rank the two",
    "cannot rank" in _secs9["12-limitations.tex"], True)

print("\nRETRACTIONS -- the count, from the appendix's structure")
_app = (pathlib.Path("paper/sections") / "14-appendices.tex").read_text()
_retr = _app.split("\\label{app:retractions}", 1)[1].split("\n\\section{", 1)[0]
_ids = re.findall(r"\\paragraph\{(C[0-9]+[a-z]?)", _retr)
chk("appendix has 5 retraction entries", len(_ids), 5)
chk("appendix retraction ids", sorted(_ids), sorted(["C19", "C20", "C7b", "C27", "C30"]))
# Whitespace is flattened first: the abstract's "$28$\nof $29$" wrap taught us that a
# literal match fails in the direction that looks exactly like the defect being hunted.
_flat = lambda q: " ".join(q.split())
chk("appendix prose says five were withdrawn",
    "Five claims were made and withdrawn" in _flat(_retr), True)
chk("appendix prose says three died to one shape",
    "Three of the five died to the same shape" in _flat(_retr), True)


print("\nPROVENANCE GAP -- the counts the availability appendix and section 12 assert, from the artifacts")
from src import provenance as _prov
_nostamp, _unknown = 0, 0
for _d in ("k01_scout", "k02_grid", "k03_grid_acts"):
    for _p in sorted(pathlib.Path("results", _d).glob("*.npz")):
        if _p.name.startswith("ckpt_"):
            continue                      # transient resume files, not results
        _r = _prov.read(str(_p))
        _nostamp += _r is None
        _unknown += _r is not None and _r.get("git_sha") == "unknown"
chk("unstamped artifacts, k01+k02", _nostamp, 37)
chk("'unknown'-SHA artifacts, k03", _unknown, 31)
# Revision WP1 (2026-09-28) moved "Code and data availability" from 11-implementations into
# the new reproduction appendix, 16-reproduction; the pin moved with the sentence.
_s16 = " ".join(pathlib.Path("paper/sections/16-reproduction.tex").read_text().split())
_s12 = " ".join(pathlib.Path("paper/sections/12-limitations.tex").read_text().split())
chk("68 artifacts from the three earliest sweeps carry no usable SHA", _nostamp + _unknown, 68)
chk("availability appendix says $68$ from the three earliest sweeps",
    f"${_nostamp + _unknown}$ artifacts from the three earliest sweeps" in _s16, True)
chk("section 12 says $37$ unstamped and $31$ unknown",
    f"${_nostamp}$ artifacts carry no stamp at all and ${_unknown}$ carry" in _s12, True)
# Gate 1 (section 10's calibration numbers) was recovered by bit-identical reproduction
# (PREREGISTER_gate1_repro.md); the live artifacts must carry that clean stamp.
for _t in ("atgrok", "final"):
    _r = _prov.read(f"results/gate1/add113_{_t}.npz") or {}
    chk(f"gate1 {_t} stamped clean at 1eaf0e3",
        (str(_r.get("git_sha", ""))[:7], _r.get("git_dirty")), ("1eaf0e3", False))


print("\nMETHODS (revision WP3.1) -- the training details section 4 states, from the code and artifacts")
# Plan section 5: every methods detail is read from code or an artifact, never from memory.
import inspect as _insp
import torch as _torch
from src.autograd.nn import AdamW as _EAdamW
from src.model.transformer import Transformer as _ETr
from src.analysis.runs import state as _state, last_test_acc as _last
_s5 = " ".join(pathlib.Path("paper/sections/05-methods.tex").read_text().split())
chk("engine AdamW default eps 1e-8",
    _insp.signature(_EAdamW.__init__).parameters["eps"].default, 1e-8)
chk("torch AdamW default eps 1e-8",
    _insp.signature(_torch.optim.AdamW.__init__).parameters["eps"].default, 1e-8)
_kern = sorted(pathlib.Path("kernels").glob("k0[1-9]*/run.py"))
chk("no training script overrides AdamW eps",
    [str(k) for k in _kern + [pathlib.Path("run_n7.py")]
     if re.search(r"AdamW\([^)]*eps", k.read_text())], [])
chk("every training script's grok rule is test acc > 0.99",
    [str(k) for k in _kern + [pathlib.Path("run_n7.py")]
     if not re.search(r"(?:te|te_acc) > 0\.99", k.read_text())], [])
_em = _ETr(114, 113, seed=0)
_nm = {k: v for k, v in vars(_em).items() if hasattr(v, "requires_grad") and v.requires_grad}
chk("engine: weight matrices only, no bias terms", sorted(k[:2] for k in _nm), ["W_"] * len(_nm))
chk("engine init: W_out sd 1/sqrt(512)", round(float(_nm["W_out"].data.std()) * 512 ** 0.5, 2), 1.0, 0.02)
chk("engine init: W_E sd 1/sqrt(128)", round(float(_nm["W_E"].data.std()) * 128 ** 0.5, 2), 1.0, 0.03)
chk("engine init: W_in sd 1/sqrt(128)", round(float(_nm["W_in"].data.std()) * 128 ** 0.5, 2), 1.0, 0.02)
chk("torch kernels outside k08 scale W_out by 1/sqrt(D_MLP), the rest 1/sqrt(128)",
    [str(k) for k in _kern if "k08" not in str(k) and "k01" not in str(k)
     and not ("sc=1 / math.sqrt(D_MLP)" in k.read_text() and "1 / math.sqrt(D_MODEL)" in k.read_text())], [])
# "The run seed sets both the initial weights and the random split": the init generator and
# the split permutation are both seeded by the run's own seed, in every training script.
chk("every kernel seeds its init and its split with the run seed",
    [str(k) for k in _kern if not ("manual_seed(seed)" in k.read_text()
     and re.search(r"default_rng\(seed\)(?:\.permutation|\s*\n\s*perm = rng\.permutation)", k.read_text()))], [])
chk("the engine seeds its init and its split with the run seed",
    ("default_rng(seed)" in pathlib.Path("src/model/transformer.py").read_text(),
     "default_rng(seed).permutation" in pathlib.Path("src/train/loop.py").read_text()), (True, True))
chk("PyTorch logs every 200 steps (k03)",
    sorted(set(np.diff(np.load("results/k03_grid_acts/WE_B_thesis_n113_s0.npz")["hist"][:, 0]).astype(int))), [200])
chk("the engine logs every 100 steps (N7)",
    sorted(set(np.diff(np.load("results/n7_engine/WE_engine_n113_s0.npz")["hist"][:, 0]).astype(int))), [100])
_z125 = np.load("results/k03_grid_acts/WE_B_thesis_n125_s1.npz", allow_pickle=True)
chk("the near-grok run: window median 0.976", (_state(_z125)[0], round(_state(_z125)[1], 3)), ("near-grok", 0.976))
chk("the near-grok run: the same run ends at 0.9779", round(_last(_z125), 4), 0.9779)
for _needle in ("$\\epsilon = 10^{-8}$", "every $200$ steps, and the engine logs them every $100$ steps",
                "$1/\\sqrt{512}$ for $W_{\\text{out}}$ and $1/\\sqrt{128}$", "test accuracy $0.9779$",
                "The run above ends at $\\bar{a} = 0.976$", "above $0.99$"):
    chk(f"section 4 states: {_needle}", _needle in _s5, True)


print("\nTABLE A (revision WP3.2) -- tab:calib, every cell from its artifact or script")
import subprocess as _sp_
import analyze_k07 as _k07
_s10 = " ".join(pathlib.Path("paper/sections/10-calibration.tex").read_text().split())
_tab = _s10.split("\\label{tab:calib}", 1)[1].split("\\bottomrule", 1)[0]
_v = np.array([_k07.spectra(_s) for _s in range(5)], dtype=float)
_m, _sd = _v.mean(0), _v.std(0)
for _i, (_lab, _fmt, _pub) in enumerate([("Gini_{\\text{mult}}", "{:.3f}", 0.579),
                                         ("PR}_{\\text{mult}", "{:.2f}", 4.1),
                                         ("Gini}_{\\text{add}", "{:.3f}", 0.071),
                                         ("PR}_{\\text{add}", "{:.2f}", 52.7)]):
    _cell = f"${_fmt.format(_m[_i])} \\pm {_fmt.format(_sd[_i])}$ & ${_pub}$"
    chk(f"Table A, 2606.17399 row {_i}: {_cell}", _cell in _tab, True)
chk("Table A: published values are analyze_k07.PUB (checked against 2606.17399 Table 1)",
    (_k07.PUB["gini_mult"], _k07.PUB["pr_mult"], _k07.PUB["gini_add"], _k07.PUB["pr_add"],
     _k07.PUB["n_key"]), (0.579, 4.1, 0.071, 52.7, 4))
chk("Table A: key frequencies 4.6 +- 0.8", (round(_m[4], 1), round(_sd[4], 1)), (4.6, 0.8))
chk("Table A: 4 key frequencies in 3 of 5 seeds", int((_v[:, 4] == 4).sum()), 3)
chk("Table A: row reads $4.6 \\pm 0.8$ & $4$", "$4.6 \\pm 0.8$ & $4$" in _tab, True)
_REF = pathlib.Path("reference/interpreting-monoids")   # the authors' repo; not shipped
_rc = _sp_.run([".venv/bin/python", "refcheck.py"], capture_output=True, text=True,
               env={**os.environ, "PYTHONPATH": "."}).stdout if _REF.is_dir() else None
_sc = _sp_.run([".venv/bin/python", "analyze_scout.py"], capture_output=True, text=True,
               env={**os.environ, "PYTHONPATH": "."}).stdout
_r165 = re.search(r"^\s*165\s+\d+\s+\d+ \|\s+([\d.]+).*\|\s+([\d.]+)\s*$", _sc, re.M)
chk("Table A: our scout n=165 Gini_mult (amplitude) and PR", (_r165.group(1), _r165.group(2)), ("0.629", "4.76"))
for _f in ("refcheck.py", "analyze_scout.py"):   # P6: no Gini on energy may come back
    chk(f"P6: {_f} takes no Gini on energy",
        bool(re.search(r"gini\((energy|e_mult|e_add|np\.sqrt\(energy)", pathlib.Path(_f).read_text())), False)
chk("Table A: row reads Gini_mult & $0.629$ & $0.637$",
    "$\\mathrm{Gini}_{\\text{mult}}$ & $0.629$ & $0.637$" in _tab, True)
chk("Table A: no energy label left", "on energy" in _tab or "energy spectrum" in _tab, False)
chk("Table A: row reads $4.76$ & $4.85$", "$4.76$ & $4.85$" in _tab, True)
if _rc is None:
    print(f"  {_REF}/ absent: the three checks on the authors' checkpoint are skipped")
    _SKIPPED.extend([0.637, 4.85, 1.03, 0.63])
else:
    _g = re.search(r"multiplicative \(product-character\) Gini=([\d.]+)\s+PR=([\d.]+)", _rc)
    chk("Table A: their checkpoint Gini_mult (amplitude) and PR", (_g.group(1), _g.group(2)), ("0.637", "4.85"))
    chk("Table A: J-class separation 1.03 vs shuffled 0.63",
        (re.search(r"separation ratio \(between/within\) = ([\d.]+)", _rc).group(1),
         re.search(r"shuffled-label control ratio\s+= ([\d.]+)", _rc).group(1)), ("1.03", "0.63"))
    chk("Table A: P(n) exact on their checkpoint", "EXACT  [their ckpt]" in _rc, True)
_g1 = _sp_.run([".venv/bin/python", "test_gate1.py", "results/gate1/add113_final.npz"],
               capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "."}).stdout
chk("Table A: Gate 1 grok step 6,900",
    json.load(open("results/gate1/history.json"))["grok_step"], 6900)
chk("Table A: Gate 1 five key frequencies", re.search(r"(\d+) found:", _g1).group(1), "5")
chk("Table A: Gate 1 85.0% of 512 neurons", re.search(r"([\d.]+)% of (\d+) neurons", _g1).groups(), ("85.0", "512"))
_b, _r = (float(x) for x in re.search(r"restricted ([\d.e+-]+) vs baseline ([\d.e+-]+)", _g1).groups()[::-1])
chk("Table A: restricted improves 7.3x", round(_b / _r, 1), 7.3)
for _needle in ("& $6{,}900$ & $5$--$10$k", "& $5$ & $5$", "& improves $7.3\\times$ & improves",
                "& $85.0\\%$ & $84.6\\%$"):
    chk(f"Table A row: {_needle}", _needle in _tab, True)
print("\nP5 (revision WP3.2) -- only the Gini effect size exceeds 1")
_f2 = (float(np.mean([np.linalg.norm(np.load(f)["W_E"][:113].astype(float))
                      for f in sorted(glob.glob("results/o18_f64/WE_*n113*.npz"))])) - 14.296) / (11.610 - 14.296)
chk("F2 on |W_E| is +0.968", round(_f2, 3), 0.968)
chk("F2 is below 1", _f2 < 1, True)
_s11 = " ".join(pathlib.Path("paper/sections/11-implementations.tex").read_text().split())
chk("section 5 no longer says both effect sizes exceed 1", "Both exceed $1$" in _s11, False)
chk("section 5 says the Gini value exceeds 1", "The Gini value exceeds $1$" in _s11, True)


print("\nP6b -- section 9's J-class indicator Ginis on the amplitude protocol")
from math import gcd as _gcd
from src.tasks.algebra import describe as _describe
from analyze_n7 import add_amplitude as _add_amp
from src.analysis.sparsity import gini as _gini
_s9 = " ".join(pathlib.Path("paper/sections/09-results-crt.tex").read_text().split())
for _n, _want in [(121, 0.406), (165, 0.450), (120, 0.400), (113, 0.0)]:
    _js = _describe(_n)["jclasses"]
    _ind = np.zeros((_n, len(_js)))
    for _j, _c in enumerate(_js):
        _ind[[x for x in range(_n) if _gcd(x, _n) == _c["d"]], _j] = 1
    _ind -= _ind.mean(0, keepdims=True)
    chk(f"9: indicator Gini_add at n={_n}, amplitude (analyze_scout.py)",
        abs(round(_gini(_add_amp(_ind, _n)), 3)), _want)
    chk(f"9: analyze_scout prints {_want:.3f} at n={_n}",
        f"n={_n:<4} Gini_add(J-class indicators) = {_want:.3f}" in _sc, True)
chk("9: the sentence carries the amplitude values",
    "$\\mathrm{Gini}_{\\text{add}}$ is $0.406$ at $n = 121$, $0.450$ at $165$, $0.400$ at $120$, and $\\mathbf{0.000}$ at $n = 113$" in _s9, True)
chk("9: no energy label left", "energy spectrum" in _s9, False)


print("\nTABLE C (revision WP3.6) -- tab:glance restates the ledger and the results sections")
# Structural, both ways: a row's status must equal the appendix ledger's status for its claim,
# and every number in its outcome cell must appear in the section its \ref names. A table that
# restates other text is the easiest place for a stale copy to hide.
_secsC = {f.name: f.read_text() for f in pathlib.Path("paper/sections").glob("*.tex")}
_labfile = {lab: f for f, t in _secsC.items() for lab in re.findall(r"\\label\{([^}]+)\}", t)}
_ledger = {}
for _row in _secsC["14-appendices.tex"].split("\\label{app:ledger}", 1)[1].split("\\bottomrule", 1)[0].split("\\\\"):
    _c = [x.strip() for x in _row.split("&")]
    if len(_c) == 4 and re.fullmatch(r"C\d+[a-z]?", _c[0].split()[-1] if _c[0] else ""):
        _ledger[_c[0].split()[-1]] = re.sub(r"\\textbf\{([^}]*)\}", r"\1", _c[3])   # emphasis is not status
_glance = _secsC["06-results-clock.tex"].split("\\label{tab:glance}", 1)[1].split("\\bottomrule", 1)[0]
_rows = [[x.strip() for x in r.split("&")] for r in _glance.split("\\midrule", 1)[1].split("\\\\")]
_rows = [r for r in _rows if len(r) == 6]
chk("Table C has 14 rows", len(_rows), 14)
for _r in _rows:
    _ids = [x.strip() for x in _r[0].split(",")]
    chk(f"Table C {_r[0]}: status = the ledger's", [_ledger.get(i) for i in _ids], [_r[5]] * len(_ids))
    _lab = re.search(r"\\ref\{([^}]+)\}", _r[2]).group(1)
    _body = " ".join(_secsC[_labfile[_lab]].split("\\label{tab:glance}", 1)[-1].split())
    _nums = re.findall(r"(?<![\w.])\d+(?:\.\d+)?", re.sub(r"\\times 10\^\{-?\d+\}|\^\{\d+\}", " ", _r[4]))
    _outc = {"C3": f"${sum(1 for r in rs if r['grok'] * 2 > r['seeds'])}$ of ${len(rs)}$ moduli",
             "C23": f"${sum(r['p_perm'] < 0.01 for r in _k04)}$ of ${len(_k04)}$ runs",
             "C32": f"${sum(r['p_perm'] < 0.01 for r in _k08)}/{len(_k08)}$",
             "C38": f"${_s2['n_equal']}/{_s2['n_pairs']}$ pairs"}.get(_r[0])
    if _outc:   # a count the table restates is also scored against the artifact itself
        chk(f"Table C {_r[0]}: outcome = the artifact's count", _r[4], _outc)
    chk(f"Table C {_r[0]}: outcome numbers appear in {_labfile[_lab]}",
        [x for x in _nums if not re.search(rf"(?<![\w.]){re.escape(x)}(?![\d])", _body)], [])


print("\nTABLE B (revision 3b) -- tab:methods, every cell from each study's PDF")
# The cited PDFs are not redistributed, so the cells were read once from them and are
# pinned here structurally: shape, row order, and every cell that can be
# recomputed from something in the repo.
from sympy import factorint as _fiB, isprime as _ipB
_rel = pathlib.Path("paper/sections/03-related.tex").read_text()
_tb = _rel.split("\\label{tab:methods}", 1)[1].split("\\bottomrule", 1)[0].split("\\midrule", 1)[1]
_rowsB = [[x.strip() for x in r.split("&")] for r in _tb.split("\\\\") if r.strip()]
chk("Table B has 7 rows of 8 cells", [len(r) for r in _rowsB], [8] * 7)
chk("Table B row order", [re.sub(r"\\citet\{([^}]+)\}", r"\1", r[0]) for r in _rowsB],
    ["nanda2023progress", "zhong2023clock", "chughtai2023toy", "nguyen2026dlogclock",
     "chen2026multiplication", "hwang2026pfe", "this work"])
chk("Table B: every study but ours is 'not stated' for pre-registration",
    [r[7] for r in _rowsB[:6]], ["not stated"] * 6)
_sf = lambda n: all(e == 1 for e in _fiB(n).values())
chk("Table B: Nanda's and Zhong's moduli are prime", all(map(_ipB, (113, 53, 109, 59))), True)
chk("Table B: C_118 is square-free, 113 prime", (_sf(118), _ipB(113)), (True, True))
chk("Table B: Chen's four moduli, as the prose states them",
    ("$113$, $143$, $154$, $165$ (all square-free)" in _rowsB[4][3],
     "four moduli, $113$, $143$, $154$ and $165$" in " ".join(_rel.split()),
     all(map(_sf, (113, 143, 154, 165)))), (True, True, True))
_pfe = (15, 21, 33, 35, 55, 77, 105, 165, 231, 385)   # 2606.23044 section 4.2
chk("Table B: PFE's ten composites, 15 to 385, all square-free",
    (len(_pfe), min(_pfe), max(_pfe), all(map(_sf, _pfe)), "ten composites from $15$ to $385$ (all square-free)" in _rowsB[5][3]),
    (10, 15, 385, True, True))
_setup_tab = pathlib.Path("paper/sections/01-setup.tex").read_text() \
    .split("\\label{sec:setup-moduli}", 1)[1].split("\\bottomrule", 1)[0]
_modB = [int(m) for m in re.findall(r"^\s*(\d+) & ", _setup_tab, re.M)]
chk("Table B: our moduli cell = tab:moduli's count and non-square-free count",
    f"${len(_modB)}$ moduli, ${sum(not _sf(n) for n in _modB)}$ not square-free", _rowsB[6][3])
chk("Table B: tab:moduli's sq.-free column agrees with factorint",
    [n for n in _modB if ("\\yes" in re.search(rf"^\s*{n} & .*", _setup_tab, re.M).group(0).split("&")[3]) != _sf(n)], [])
_gl = [r for r in ([x.strip() for x in r.split("&")] for r in _glance.split("\\midrule", 1)[1].split("\\\\")) if len(r) == 6]
chk("Table B: our pre-registration cell = Table C's pre-reg. rows",
    f"${sum('pre-reg.' in r[3] for r in _gl)}$ of the ${len(_gl)}$ rows of Table~\\ref{{tab:glance}}", _rowsB[6][7])
chk("Table B: the weight intervention is at the two I1 moduli", "weight intervention at $n = 113$, $121$" in _rowsB[6][5], True)

print("\nSWEEP TABLES (revision 3c) -- tab:sweeps and tab:sweep-files equal scripts/sweep_table.py")
sys.path.insert(0, "scripts")
import sweep_table as _swt
_confS, _filesS = _swt.rows()
_rep = pathlib.Path("paper/sections/16-reproduction.tex").read_text()
def _tbody(lab):
    t = _rep.split(f"\\label{{{lab}}}", 1)[1].split("\\bottomrule", 1)[0].split("\\midrule", 1)[1]
    return [[c.strip() for c in r.split("&")] for r in t.split("\\\\") if r.strip()]
chk("tab:sweeps equals sweep_table.rows()", _confS, _tbody("tab:sweeps"))
_filesT = _tbody("tab:sweep-files")
if any(r[-1] is None for r in _filesS):   # the public repository ships no run logs
    print("  logs/ absent: tab:sweep-files is compared without its compute-hours column")
    _SKIPPED.extend(float(t[-1].strip("$")) for r, t in zip(_filesS, _filesT) if r[-1] is None)
    chk("tab:sweep-files compute hours, where the logs ship",
        [r[-1] for r in _filesS if r[-1] is not None],
        [t[-1] for r, t in zip(_filesS, _filesT) if r[-1] is not None])
    _filesS, _filesT = [r[:-1] for r in _filesS], [r[:-1] for r in _filesT]
chk("tab:sweep-files equals sweep_table.rows()", _filesS, _filesT)
_allc = [c for _, d, pat, _, _ in _swt.SWEEPS for c in _swt.configs(d, pat)]
chk("tab:sweeps caption: every run is lr 1e-3, wd 1.0",
    sorted({(c["lr"], c["wd"]) for c in _allc}), [(0.001, 1.0)])
chk("sweep tables name every training and analysis script that exists",
    [x for _, _, _, t, a in _swt.SWEEPS for x in (t.split()[0], a) if not pathlib.Path(x).exists()], [])

print("\nPINS FROM THE ANALYSIS SCRIPTS (P7) -- the paper's printed cells against each script's printed cells")
# The claim-carrying analysis script is the ONE implementation of each number; these pins score
# the paper against what it prints, at the precision it prints. Only scripts that write no
# artifact are run here (analyze_n7 and run_c12_collapse restamp theirs, so they are not).
import subprocess as _sp
_SCRIPT_OUT = {}
def _script(*argv, env=None):
    key = (argv, tuple(sorted((env or {}).items())))
    if key not in _SCRIPT_OUT:
        _SCRIPT_OUT[key] = _sp.run([sys.executable, *argv], capture_output=True, text=True, check=True,
                                   env={**os.environ, "PYTHONPATH": ".", **(env or {})}).stdout
    return _SCRIPT_OUT[key]
_sec = lambda f: (pathlib.Path("paper/sections") / f).read_text()
_rows = lambda t: [[c.strip() for c in r.split("&")] for r in t.split("\\\\")]
_nums = lambda c: re.findall(r"\d+\.\d+|\d+", c.replace("{,}", ""))

# 6, C6 table (k02, five seeds): Gini_mult and Gini_add mean +- sd at 113, 121, 125.
_k02 = _script("analyze_k02.py")
_c6 = {int(m[0]): m[1:] for m in re.findall(r"^\s+(113|121|125)\s+([\d.]+)\+-([\d.]+) ([\d.]+)\+-([\d.]+)", _k02, re.M)}
_t6 = _sec("06-results-clock.tex").split("$|\\Delta|$ vs $113$", 1)[1].split("\\bottomrule", 1)[0]
for _r in _rows(_t6):
    if len(_r) == 4 and _nums(_r[0]):
        _n = int(_nums(_r[0])[0])
        chk(f"6 C6 table n={_n}: Gini_mult, Gini_add mean +- sd", _nums(_r[1] + _r[2]), list(_c6[_n]))
_d = re.search(r"\(113\)\| = ([\d.]+)\s+\|125 - 113\| = ([\d.]+)", _k02).groups()
chk("6 C6 table: |Delta| vs 113 at 121, 125", [_nums(r[3])[0] for r in _rows(_t6) if len(r) == 4 and "---" not in r[3] and _nums(r[3])], list(_d))
_g119 = [int(x) for x in re.search(r"^\s+119 \[([\d, ]+)\]", _k02, re.M).group(1).split(",")]
chk("6: n=119 grokking time spans 6,600 to 24,800 (k02, 5 seeds)", (min(_g119), max(_g119)), (6600, 24800))

# 6, C35 table (k09): the two new primes.
_k09 = _script("analyze_k09.py")
_t35 = _sec("06-results-clock.tex").split("neurons tuned (dlog) & raw & shuffled", 1)[1].split("\\bottomrule", 1)[0]
for _r in _rows(_t35):
    if len(_r) == 6:
        _n = int(_nums(_r[0])[0])
        _g, _dl = re.search(rf"n={_n}\s+Gini_mult ([\d.]+)\s+delta ([\d.]+)", _k09).groups()
        _tu = re.findall(rf"n={_n}\s+s\d\s+tuned ([\d.]+)\s+raw ([\d.]+)\s+shuffled ([\d.]+)", _k09)
        chk(f"6 C35 table n={_n}: Gini_mult, |Delta|", _nums(_r[1] + " " + _r[2]), [_g, _dl])
        chk(f"6 C35 table n={_n}: tuned per seed, raw, shuffled",
            (_nums(_r[3]), _nums(_r[4])[0], _nums(_r[5])[0]),
            ([t[0] for t in _tu], max(t[1] for t in _tu), max(t[2] for t in _tu)))
chk("6 C35 row: Delta <= 0.0250 at 127 and 131",
    max(re.findall(r"n=1(?:27|31)\s+Gini_mult [\d.]+\s+delta ([\d.]+)", _k09)), "0.0250")

# 6 C21 and 10 Table A (k07, Arm A, five seeds).
_k07 = _script("analyze_k07.py")
_ab = re.findall(r"a\+b carries\s+([\d.]+)% of top-8 energy", _k07)
chk("6 C21: a+b share of top-8 logit energy per seed", _ab, ["98.37", "100.00", "100.00", "100.00", "100.00"])
_mean = re.search(r"^\s+mean\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", _k07, re.M).groups()
_sd = re.search(r"^\s+sd\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", _k07, re.M).groups()
_tA = _sec("10-calibration.tex").split("units only, their protocol, five seeds}", 1)[1].split("\\midrule", 1)[0]
chk("10 Table A: this-work column = analyze_k07 mean +- sd",
    [_nums(r[2]) for r in _rows(_tA) if len(r) == 4],
    [[m, s_] for m, s_ in zip(_mean, _sd)][:4] + [[_mean[4], _sd[4].rstrip("0").rstrip(".")]])

# 6 neuron-tuning table and its controls (analyze_o4, C31).
_o4 = _script("analyze_o4.py")
_tun = {}
for _run, _st, _pct in re.findall(r"^\s+(n\d+_s\d)\s+\d+\s+(grokked|near-grok)\s+([\d.]+)%", _o4, re.M):
    _tun.setdefault((_run.split("_")[0][1:], _st), []).append(_pct)
_tN = _sec("06-results-clock.tex").split("tuned neurons, per seed &", 1)[1].split("\\bottomrule", 1)[0]
for _r in _rows(_tN):
    if len(_r) == 3 and _nums(_r[0]):
        _n = _nums(_r[0])[0]
        chk(f"6 tuning table n={_n}: grokked seeds", _nums(_r[1]), _tun[(_n, "grokked")])
chk("5/6: the near-grok n=125 s1 reads 74.8% tuned", sorted(set(_tun[("125", "near-grok")])), ["74.8"])   # o4 prints its table twice
chk("6: grokked n=113 spans 92-100% (rounded)", (round(min(map(float, _tun[("113", "grokked")]))), round(max(map(float, _tun[("113", "grokked")])))), (92, 100))
chk("6: matched-failure control, engine n=125 reads 6.8%",
    re.search(r"n7_eng/n125\s+\d+\s+FAILED\s+([\d.]+)%", _o4).group(1), "6.8")
from src.analysis import runs as _runs
_z125 = np.load("results/n7_engine/WE_engine_n125_s1.npz", allow_pickle=True)
chk("6: ...whose final (last-row) accuracy is 0.6082, FAILED by the window rule",
    (round(_runs.last_test_acc(_z125), 4), _runs.state(_z125)[0]), (0.6082, "FAILED"))

# 6: training-set sizes at train_frac 0.30, from the task's own split (src/train/loop.py).
from src.train.loop import modular_data as _md
chk("6: training examples at 0.30 for 49, 54, 63, 91", [len(_md(n)[1]) for n in (49, 54, 63, 91)], [720, 874, 1190, 2484])
chk("9: at n=113 the CRT prediction is all 56 bins", len(_crt.predicted(113)), 56)

# 9, the CRT-dual law on k04's nine CRT-testable moduli (analyze_k04 P3), n = 123 (analyze_k09),
# and the five-seed moduli (analyze_k02).
_k04o = _script("analyze_k04.py")
_p3 = {m[0]: [x.rstrip("x") for x in re.findall(r"'([\d.]+x)'", m[1])]
       for m in re.findall(r"^\s+(\d+)\s+(\[[^\]]*\])\s+3/3", _k04o.split("P3 --", 1)[1], re.M)}
_t9 = _sec("09-results-crt.tex")
_t9rows = {r[0]: r[1].split("/") for r in (x.replace("\\midrule", "").strip().split(" & ") for x in _t9.split("\\\\")) if len(r) == 3 and r[0].strip().isdigit()}
chk("9 CRT table: per-seed enrichment at k04's nine moduli = analyze_k04 P3", _t9rows, _p3)
chk("9: n = 123 enrichment per seed (analyze_k09)", re.search(r"n=123\s+enrichment \[([\d. ]+)\]", _k09).group(1).split(), ["7.59", "3.28", "7.43"])
_k02e = {m[0]: [x.rstrip("x") for x in re.findall(r"'([\d.]+x)'", m[1])] for m in re.findall(r"^\s+(165|120|119)\s+(\[[^\]]*\])\s+5/5", _k02, re.M)}
_t9f = " ".join(_t9.split())
for _n in ("165", "120", "119"):
    chk(f"9: n={_n} reads its five seeds' enrichments (analyze_k02)",
        re.search(rf"\${_n}\$ reads \$([\d., $]+)\$", _t9f).group(1).replace("$", "").replace(" ", "").split(","), _k02e[_n])

# 9, O20: the failed-ramp arm at n = 63 (test_crt_null.py under CRT_NULL_N=63), its early grokked
# band (the N7 engine's n = 119, same scorer), and the PyTorch corroboration.
_o20 = _script("test_crt_null.py", "results/o20_n63", env={"CRT_NULL_N": "63"})
_n7c = _script("test_crt_null.py", "results/n7_engine")
_seeds = re.findall(r"--- seed (\d)\s+final acc ([\d.]+) ---(.*?)ramp rho\(step, enrich\) over 0-20k = ([\d.]+)", _o20.split("PRE-REGISTERED")[0], re.S)
_o20rows = [[s_, a_, re.search(r"^\s+1000\s+([\d.]+)", b_, re.M).group(1), re.search(r"^\s+20000\s+([\d.]+)", b_, re.M).group(1), r_] for s_, a_, b_, r_ in _seeds]
_tO = _t9.split("ramp $\\rho$ \\\\", 1)[1].split("\\bottomrule", 1)[0]
chk("9 O20 table: seed, final acc, enrichment at 1k and 20k, ramp rho", [_nums(" ".join(r)) for r in _rows(_tO) if len(r) == 5], _o20rows)
chk("9 O20: window-median test accuracy of the three failed runs",
    [f"{_runs.final_acc(np.load(f'results/o20_n63/WE_engine_n63_s{i}.npz', allow_pickle=True)):.4f}" for i in range(3)],
    _nums(re.search(r"window medians\s+\$([^)]*)", " ".join(_t9.split())).group(1))[:3])
_early = [float(x) for x in re.findall(r"^\s+1000\s+([\d.]+)", _n7c.split("PRE-REGISTERED")[0], re.M)]
chk("9 O20: the grokked early band at step 1,000 (N7, n = 119)", (f"{min(_early):.2f}", f"{max(_early):.2f}"), ("1.22", "1.45"))
_pt = [t for t in re.findall(r"B_thesis_n63_s\d\s+n=63\s+acc ([\d.]+)\s+set.*?enrich\s+([\d.]+)", _o20)
       if float(t[0]) < 0.90]   # the failures only: FAIL_ACC, src/analysis/runs.py
chk("9 O20: PyTorch n = 63 corroboration (enrichment, accuracy)", sorted(_pt, key=lambda t: -float(t[1])), [("0.2236", "1.626"), ("0.2749", "1.574")])
_fail_top = max([float(r[3]) for r in _o20rows] + [float(e) for _, e in _pt])
_grok = [float(e) for e in re.findall(r"B_thesis_n\d+_s\d\s+n=\d+\s+acc (?:0\.99\d\d|1\.0000)\s+set.*?enrich\s+([\d.]+)", _n7c)]
chk("9: grokked runs reach 3.3 to 17.4, failures top out at 1.66", (f"{min(_grok):.1f}", f"{max(_grok):.1f}", f"{_fail_top:.2f}"), ("3.3", "17.4", "1.66"))

# 9, omega (C37): the confirmatory W criteria (analyze_omega --confirm) and the 18-modulus
# snapshot the section's first sentence describes (analyze_omega.collect on the population
# before k09 added 5, 91, 123, 127, 128 and 131).
_om = _script("analyze_omega.py", "--confirm")
chk("9 W1: 10 of 10 discordant pairs favour omega, sign test p", re.search(r"favouring omega = (\d+)/(\d+) = [\d.]+\s+sign test \(exact, two-sided\) p = ([\d.e+-]+)", _om).groups(), ("10", "10", "1.95e-03"))
chk("9 W6: 10,000 shuffles put the min adjacent gap at p = 1.0e-4", re.search(r"min adjacent band gap = \+[\d.]+\s+p = ([\d.e+-]+)", _om).group(1), "1.00e-04")
chk("9: with n = 128 the omega = 1 band is 0.015-0.261, omega = 2 starts at 0.394",
    (re.search(r"with n=113\s+: ([\d.]+) - ([\d.]+)", _om).groups(), re.search(r"omega=2 band starts at ([\d.]+)", _om).group(1)), (("0.015", "0.261"), "0.394"))
_pre = [x for x in _ao.collect() if x["n"] not in (5, 91, 123, 127, 128, 131)]
chk("9: the omega snapshot is 18 moduli", len(_pre), 18)
_band = lambda w, key="ga": (f"{min(x[key] for x in _pre if x['omega'] == w):.3f}", f"{max(x[key] for x in _pre if x['omega'] == w):.3f}")
chk("9: 18-modulus omega bands 1, 2, 3", [_band(1), _band(2), _band(3)], [("0.017", "0.184"), ("0.394", "0.481"), ("0.578", "0.589")])
# P9b/P9c (approved 2026-09-29): the paper printed +0.862 (ordinal ranks) and a control band
# 0.117-0.157 that reproduces on no population. Midrank rho and the code's own band, here.
from src.analysis.stats import spearman as _sp_mid
_t9w = " ".join(_t9.split())
_rz = f"{_sp_mid([x['zdd'] for x in _pre], [x['ga'] for x in _pre]):+.3f}"
chk("9 P9b: rho(zdd, Gini_add) over the 18 moduli, midrank", _rz, "+0.858")
chk("9 P9b: ...and the paper prints it", f"across $18$ moduli at $\\rho = {_rz}$" in _t9w, True)
_cb = f"${min(x['rand'] for x in _pre):.3f}$--${max(x['rand'] for x in _pre):.3f}$"
chk("9 P9c: the residue-axis control band over the 18 moduli", _cb, "$0.117$--$0.155$")
chk("9 P9c: ...and the paper prints it", f"flat at {_cb} throughout" in _t9w, True)
# P9a (approved 2026-09-29): C19's confound rho, midrank, as analyze_k04 prints it, in both places.
_c19 = re.search(r"rho\(-phi, steps/phi\) = ([+-][\d.]+).*?rho\( phi, raw steps\) = ([+-][\d.]+)", _k04o, re.S).groups()
chk("6/14 P9a: analyze_k04 prints rho(-phi, steps/phi), rho(phi, steps)", _c19, ("+0.870", "-0.665"))
for _f in ("06-results-clock.tex", "14-appendices.tex"):
    _tf = " ".join(_sec(_f).split())
    chk(f"{_f[:2]} P9a: the paper prints both midrank values",
        (f"$\\rho = {_c19[1]}$" in _tf, f"\\text{{steps}}/\\varphi) = {_c19[0]}$ over $15$ moduli" in _tf), (True, True))
# P8 (approved 2026-09-29): "a function supported on the multiples of d has additive Fourier support
# on the multiples of n/d" is false (delta_3 at n=165 is nonzero at all 165 bins); it holds for the
# indicator of dZ/nZ, whose support is exactly the multiples of n/d. Asserted at every d | n.
for _n in (120, 165):
    for _d in [d for d in range(1, _n + 1) if _n % d == 0]:
        _ind = np.zeros(_n); _ind[::_d] = 1
        _supp = set(np.flatnonzero(np.abs(np.fft.fft(_ind)) > 1e-9).tolist())
        assert _supp == set(range(0, _n, _n // _d)), (_n, _d)
_dl = np.zeros(165); _dl[3] = 1
chk("9 P8: a function on the multiples of 3 (delta_3, n=165) has 165 nonzero bins",
    int((np.abs(np.fft.fft(_dl)) > 1e-9).sum()), 165)
chk("9 P8: the paper states the indicator form, not the false general one",
    ("The indicator of the multiples of $d$ has additive Fourier support" in _t9w,
     "A function supported on the multiples of" in _t9w), (True, False))

# 8, number debt (session 32): Gate 2's prose against results/gate2/*.json and the runs' own history.
_t8 = " ".join(_sec("08-results-strata.tex").split())
_g2k = _load("results/gate2/k03_grid_acts.json") + _load("results/gate2/n7_engine.json")
_g2d = json.load(open("results/gate2/k03_grid_acts.json"))
_g3 = [s_ for s_ in _g2k if s_.get("resid") is not None]
chk("8 G3 row: real residual range, permuted control range (grokked arm)",
    (f"{min(x['resid'] for x in _g3):.3f}", f"{max(x['resid'] for x in _g3):.3f}",
     f"{min(x['resid_ctrl'] for x in _g3):.3f}", f"{max(x['resid_ctrl'] for x in _g3):.3f}"),
    ("0.000", "0.132", "0.396", "0.929"))
_z49 = np.load("results/k04_extended/WE_B_thesis_n49_s2.npz", allow_pickle=True)
_h49 = _z49["hist"][-10:, [str(c) for c in _z49["hist_cols"]].index("test_acc")]
chk("8: n=49 s2 window median, first and last of its final ten samples",
    (f"{_runs.final_acc(_z49):.4f}", f"{_h49[0]:.3f}", f"{_h49[-1]:.3f}"), ("0.8908", "0.842", "0.963"))
# P11 (approved 2026-09-29): the paper said those ten samples "rise monotonically"; sample 3 dips.
chk("8 P11: the final ten samples of n=49 s2 are NOT monotone (one dip)", int((np.diff(_h49) < 0).sum()), 1)
chk("8 P11: ...and the paper says one dip, not monotone",
    ("rise, with one small dip, from $0.842$ to $0.963$" in _t8, "monotonically" in _t8), (True, False))
_s49 = [s_ for r in json.load(open("results/gate2/k04_extended_controls.json"))["runs"]
        if (r["n"], r["seed"]) == (49, 2) for s_ in r["strata"] if s_["measurable"]]
chk("8: n=49 s2 has one measurable stratum: held-out acc, excluded/baseline",
    [(f"{s_['acc_heldout']:.3f}", f"{s_['excluded'] / s_['baseline']:.0f}") for s_ in _s49], [("0.959", "302")])
chk("8: the lowest-accuracy R2 control run (the 0.146 memoriser)",
    f"{min(r['acc'] for r in _r2c['runs']):.3f}", "0.146")
_nd = [s_ for s_ in _g2k if s_.get("tuned_nd") is not None]
chk("8: product-character tuning range, and its shuffle control",
    (f"{min(x['tuned_nd'] for x in _nd):.3f}", f"{max(x['tuned_nd'] for x in _nd):.3f}", f"{max(x['tuned_nd_sh'] for x in _nd):.4f}"),
    ("0.182", "0.877", "0.0000"))
# P10 (approved 2026-09-29): the paper said "13 stratum-runs fall to 0.500-0.833" -- 13 is the number
# of strata (gate2_table.py groups by (n, d, e)) and 0.500 is a seed mean; the stratum-runs number 15, 0.379-0.833.
_bad = [(r["n"], s_["d"], s_["e"], s_["acc_heldout"]) for r in _g2d["runs"] for s_ in r["strata"]
        if s_["acc_heldout"] == s_["acc_heldout"] and s_["acc_heldout"] < 0.95]
_bm = collections.defaultdict(list)
for _n, _d_, _e_, _a in _bad:
    _bm[(_n, _d_, _e_)].append(_a)
chk("8 P10: stratum-runs below 0.95 (k03): count and range", (len(_bad), f"{min(a for *_, a in _bad):.3f}", f"{max(a for *_, a in _bad):.3f}"), (15, "0.379", "0.833"))
chk("8 P10: ...grouped by stratum: count and range of seed means",
    (len(_bm), f"{min(np.mean(v) for v in _bm.values()):.3f}", f"{max(np.mean(v) for v in _bm.values()):.3f}"), (13, "0.500", "0.833"))
chk("8 P10: ...and the paper prints the stratum-run count and range",
    f"${len(_bad)}$ stratum-runs fall to ${min(a for *_, a in _bad):.3f}$--${max(a for *_, a in _bad):.3f}$" in _t8, True)
# 8.C7 (1 seed): jclass_spectra.py prints the single-seed table.
_js = _script("jclass_spectra.py")
_jrow = lambda n, J: re.search(rf"=== n={n} .*?^\s+{J}\s+\d+\s+NON-REG\s+\(.*?\)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", _js, re.S | re.M).groups()
for _n, _J in ((121, "J_11"), (125, "J_5"), (120, "J_2")):
    _g, _c, _r = _jrow(_n, _J)
    chk(f"8 C7: {_J} at n={_n} Gini, control, ratio (jclass_spectra)",
        f"${_J.replace('_', '_{')}}}$ at $n = {_n}$ scored ${_g}$ against ${_c}$ (${_r}\\times$)" in _t8, True)
    _collect((_g, _c, _r))

# 5, number debt (session 32): the methods section's worked examples, each re-derived.
_t5 = " ".join(_sec("05-methods.tex").split())
chk("5: a 200-draw null's floor is 1/201", (200 + 1, f"{1 / 201:.3f}"), (201, "0.005"))
# The retired excluded/mean(control) at 20 draws, n=121 s0, control RNG seeds 0-7, through
# analyze_n4.run itself (only N_CONTROL and the rng seed are swapped for the history).
import analyze_n4 as _an4
_rng0, _nc0 = np.random.default_rng, _an4.N_CONTROL
_sm = []
try:
    _an4.N_CONTROL = 20
    for _s in range(8):
        np.random.default_rng = lambda _x, _s=_s: _rng0(_s)
        _sm.append(_an4.run("B_thesis_n121_s0", 121, verbose=False, d="results/k03_grid_acts")["sep_mean_UNSTABLE"])
        np.random.default_rng = _rng0
finally:
    np.random.default_rng, _an4.N_CONTROL = _rng0, _nc0
chk("5: excluded/mean(20-draw control) over control seeds 0-7 spans 21x to 440x",
    (f"{min(_sm):.0f}", f"{max(_sm):.0f}"), ("21", "440"))
_fam = sum(len(np.load(f, allow_pickle=True)["rows"]) for f in glob.glob("results/*_n4_ablation.npz"))
chk("5: 137 single-modulus ablations (R1's family table: nine _n4_ablation files)", _fam, 137)
chk("5: Bonferroni per-test levels at alpha 0.05 and 0.01 over 320",
    (f"{0.05 / 320:.1e}", f"{0.01 / 320:.1e}"), ("1.6e-04", "3.1e-05"))
chk("5: ...and 1/(B+1) <= 0.01/320 needs B + 1 >= 32,000", math.ceil(320 / 0.01), 32000)
# The energy-vs-amplitude example is k07 (Arm A) seed 0: Gini on energy() 0.902, PR on
# amplitude 12.5, PR on energy 4.31 (analyze_k07 prints the last).
from analyze_n7 import mult_amplitude as _mamp
from src.analysis.sparsity import gini as _gini, participation_ratio as _pr
from src.analysis.transforms import energy as _energy, character_transform as _ct
_W7 = np.load("results/k07_arma_seeds/WE_A_replication_n113_s0.npz", allow_pickle=True)["W_E"][:113].astype(float)
_a7 = _mamp(_W7, 113)
chk("5: k07 s0 Gini on energy (the retired convention), PR on amplitude, PR on energy",
    (f"{_gini(_energy(_ct(_W7 - _W7.mean(0, keepdims=True), 113))):.3f}", f"{_pr(_a7):.1f}", f"{_pr(_a7 ** 2):.2f}"),
    ("0.902", "12.5", "4.31"))
chk("5: ...and analyze_k07 prints seed 0's PR_mult as 4.31",
    re.search(r"^\s+0\s+[\d.]+\s+([\d.]+)", _k07, re.M).group(1), "4.31")
# P13 (approved 2026-09-29): the paper paired those with "Gini on amplitude reads 0.539"; the
# protocol reads 0.543 on that checkpoint (analyze_k07's seed-0 row says the same).
chk("5 P13: k07 s0 Gini on amplitude", f"{_gini(_a7):.3f}", "0.543")
chk("5 P13: ...and the paper prints it", f"Gini on amplitude reads ${_gini(_a7):.3f}$" in _t5, True)
# P12 (approved 2026-09-29): "0.4% raw, 98.4% dlog" paired two statistics (neuron terms raw, logit
# top-8 dlog). Now one statistic: k07 s0's a+b share of its top-8 logit components, analyze_k07's formula.
from src.viz import mechinterp as _M12
def _ab8(dl):
    _r = _M12.logit_top_components("results/k07_arma_seeds/WE_A_replication_n113_s0.npz", 113, top=8, verbose=False, dlog=dl)
    return 100 * sum(v for *_, v, k in _r if k == "a+b") / sum(v for *_, v, k in _r)
_p12 = (f"{_ab8(False):.1f}", f"{_ab8(True):.1f}")
chk("5 P12: k07 s0 top-8 logit a+b share, raw and dlog", (f"{_ab8(False):.2f}", f"{_ab8(True):.2f}"), ("16.95", "98.37"))
for _f in ("03-related.tex", "05-methods.tex"):
    _tf = " ".join(_sec(_f).split())
    chk(f"{_f[:2]} P12: one statistic for both numbers",
        (f"top eight logit components read ${_p12[0]}\\%$ ``$a+b$''" in _tf,
         f"${_p12[1]}\\%$ in discrete-log coordinates" in _tf, "$0.4\\%$" in _tf), (True, True, False))
# P14 (approved 2026-09-29): "across the seven sweeps" -- the 137 are eight sweeps plus the retired
# pre-leak-fix engine arm's two runs (R1's family table).
_fams = {f: len(np.load(f, allow_pickle=True)["rows"]) for f in glob.glob("results/*_n4_ablation.npz")}
chk("5 P14: nine ablation files, the retired arm holding two",
    (len(_fams), _fams["results/n7_engine_preleakfix_n4_ablation.npz"]), (9, 2))
chk("5 P14: ...and the paper says so", "$137$ single-modulus ablations across eight sweeps and two runs of a retired engine arm" in _t5, True)
# Gini noise: final 12k steps of k03's trajectories, FINDINGS 3.5's three runs.
import analyze_k03 as _ak3
def _tailg(n, s):
    z = np.load(f"results/k03_grid_acts/WE_B_thesis_n{n}_s{s}.npz", allow_pickle=True)
    tr, st = z["we_traj"], z["we_traj_steps"]
    return np.array([_ak3.spectral(W, n)[0] for W in tr[st >= st[-1] - 12000]])
_tg = {k: _tailg(*k) for k in ((113, 1), (121, 2), (125, 2))}
chk("5: within-run sd over the final 12k steps, 113 s1 / 121 s2 / 125 s2",
    [f"{v.std():.3f}" for v in _tg.values()], ["0.020", "0.032", "0.044"])
chk("5: n=125 s2 wanders over 0.498-0.621, a range of 0.122",
    (f"{_tg[(125, 2)].min():.3f}", f"{_tg[(125, 2)].max():.3f}", f"{np.ptp(_tg[(125, 2)]):.3f}"), ("0.498", "0.621", "0.122"))
# PR drift over the final 10k steps (analyze_k03 H4/H5), and the per-seed PR-only counts.
_k03o = _script("analyze_k03.py")
_pd = {m[0]: m[1] for m in re.findall(r"B_thesis n=(\d+): Gini drift [\d.]+%\s+PR drift ([\d.]+)%", _k03o)}
chk("5: PR drift at 113, 121, 125, 119, 120, 165 (analyze_k03)", [_pd[k] for k in ("113", "121", "125", "119", "120", "165")],
    ["2.3", "2.8", "5.6", "5.9", "7.7", "4.8"])
_blk = {m[0]: m[1] for m in re.findall(r"B_thesis n=(\d+):.*?\n((?:\s+seed \d.*\n)+)", _k03o)}
_cnt = {n: sum(float(x) < 5.0 for x in re.findall(r"PR_mult [\d.]+ \(drift ([\d.]+)%\)", b)) for n, b in _blk.items()}
chk("5: seeds with PR drift < 5%: 113, 121, 125, 119, 120, 165", [_cnt[k] for k in ("113", "121", "125", "119", "120", "165")], [5, 4, 2, 3, 3, 3])
chk("5: ...as the paper prints them",
    all(s_ in _t5 for s_ in ("($2.3\\%$ participation-ratio drift, $5/5$ seeds)", "($2.8\\%$, $4/5$)", "$n = 125$ at $5.6\\%$ ($2/5$ seeds)",
                              "$n = 119$ at $5.9\\%$ ($3/5$)", "$n = 120$ at $7.7\\%$ ($3/5$)", "$n = 165$ at $4.8\\%$ ($3/5$)")), True)

# The ordinal-rank partials the section reports as history (retired 2026-09-23): the same two
# numbers in the opposite order. Ordinal ranks deliberately, to reproduce the retired value.
_ordr = lambda v: np.argsort(np.argsort(np.asarray(v, float))).astype(float)
_ords = lambda x, y: float(np.corrcoef(_ordr(x), _ordr(y))[0, 1])
def _opart(x, y, z):
    rxy, rxz, rzy = _ords(x, y), _ords(x, z), _ords(z, y)
    return (rxy - rxz * rzy) / math.sqrt((1 - rxz ** 2) * (1 - rzy ** 2))
chk("9: under ordinal ranks the two partials read +0.627 and +0.735 (history)",
    (round(_opart(_W_, _G_, _Z_), 3), round(_opart(_Z_, _G_, _W_), 3)), (0.627, 0.735))
chk("9 P1 (analyze_k09): n = 128 reads 0.261 between the omega band and the zdd neighbourhood",
    re.search(r"Gini_add = ([\d.]+)\s+over 2 grokked seeds.*?omega=1 band ([\d.]+)-([\d.]+)\s+\|\s+zdd~0.500 neighbourhood ([\d.]+)-([\d.]+)", _k09, re.S).groups(),
    ("0.261", "0.017", "0.184", "0.434", "0.589"))

# 7, the N4 table's ranges and the heavy-tail sentence (k03, five seeds; the `per` rows above).
_t7 = _sec("07-results-causal.tex")
_tN4 = _t7.split("range over seeds & $p < 0.01$ in", 1)[1].split("\\bottomrule", 1)[0]
_r4 = {int(_nums(r[0])[0]): r[3] for r in _rows(_tN4) if len(r) == 5 and _nums(r[0])}
for _n, _cell in _r4.items():
    _q = [x["baseline"] / x["restricted"] for x in per[_n]]
    chk(f"7 N4 table n={_n}: range over seeds", _nums(_cell), [f"{min(_q):.3g}", f"{max(_q):.3g}"])
_q113 = sorted(x["baseline"] / x["restricted"] for x in per[113])
chk("7: at 113 one seed reads 264x, the other four 1.56 to 8.0, mean 55.5 +- 104.1",
    (f"{_q113[-1]:.0f}", f"{_q113[0]:.2f}", f"{_q113[-2]:.1f}", f"{np.mean(_q113):.1f}", f"{np.std(_q113):.1f}"),
    ("264", "1.56", "8.0", "55.5", "104.1"))
_t7w = " ".join(_t7.split())
# P15 (approved 2026-09-29): C23 printed "separates by 10.8 to 59.6x", the retired
# excluded/mean(20-draw control). Now excluded/median(control), 10,000 draws, k03's 119/120/165.
_sm23 = [x["sep_median"] for n_ in (119, 120, 165) for x in per[n_]]
_p15 = (f"{min(_sm23):.1f}", f"{max(_sm23):.1e}")
chk("7 P15: excluded / median control at 119, 120, 165 (15 runs): min, max", (len(_sm23), *_p15), (15, "10.0", "4.6e+07"))
chk("7 P15: ...and the paper prints it, not the retired ratio",
    (f"by ${_p15[0]}$ to ${_p15[1][:3]} \\times 10^{{{int(_p15[1][4:])}}}\\times$" in _t7w, "59.6" in _t7w), (True, False))
# P16 (approved 2026-09-29): n=123 printed "restricted/baseline only 1.32x", the mean of three
# baseline/restricted values under an upside-down label. Median and range, k09's three seeds.
_q123 = [x["baseline"] / x["restricted"] for x in
         (json.loads(str(r)) for r in np.load("results/k09_primes_zdd_n4_ablation.npz", allow_pickle=True)["rows"]) if x["n"] == 123]
_p16 = (f"{np.median(_q123):.2f}", f"{min(_q123):.3f}", f"{max(_q123):.2f}")
chk("7 P16: n=123 baseline/restricted median, min, max over 3 seeds", (len(_q123), *_p16), (3, "1.76", "0.003", "2.19"))
chk("7 P16: ...and the paper prints median and range under the right label",
    (f"baseline/restricted median only ${_p16[0]}\\times$, range ${_p16[1]}$--${_p16[2]}\\times$" in _t7w, "1.32" in _t7w), (True, False))
_r121 = [x for x in per[121] if x["tag"].endswith("_s0")][0]
chk("7: n=121 seed 0 baseline / restricted / excluded loss",
    tuple(f"{_r121[k]:.2e}" for k in ("baseline", "restricted", "excluded")), ("4.96e-07", "2.10e-07", "1.29e+01"))

# 7, I1 (run_intervention.py): the table's p-hat denominator, the retrain, the control tail.
_i1 = {s_: np.load(f"results/i1_internal/I1_engine_n113_s{s_}.npz", allow_pickle=True) for s_ in range(3)}
_i1b = {s_: np.load(f"results/i1_internal/I1_engine_n121_s{s_}.npz", allow_pickle=True) for s_ in range(3)}
chk("7 I1/I1b table: p-hat is over B + 1 = 10,001", sorted({int(z["draws"]) + 1 for z in (*_i1.values(), *_i1b.values())}), [10001])
chk("7: the I1 retrain groks at 4,500 / 6,500 / 9,700",
    [json.loads(_prov.read(f"results/i1_internal/WE_engine_n113_s{s_}.npz")["config"])["grok_step"] for s_ in range(3)], [4500, 6500, 9700])
chk("7: seed 2's control maximum 2.4464e1 exceeds its excluded loss",
    (f"{float(_i1[2]['control'].max()):.4e}", bool(_i1[2]["control"].max() > _i1[2]["internal_excluded"])), ("2.4464e+01", True))
chk("7: seed 0's control maximum reaches 99.2% of its excluded loss",
    f"{100 * float(_i1[0]['control'].max() / _i1[0]['internal_excluded']):.1f}", "99.2")
chk("7: excluded / median(control) would read 8.2e7 (median over seeds)",
    f"{np.median([float(z['internal_excluded'] / z['control_median']) for z in _i1.values()]):.1e}", "8.2e+07")

# 6, number debt: the added prime powers (analyze_k04 P4 at 81 and 169; analyze_k05 Q2 at 49).
_p4 = {m[0]: (re.findall(r"'([\d.]+)'", m[1]), re.findall(r"'([\d.]+)'", m[2]))
       for m in re.findall(r"^\s+(81|169)\s+(\[[^\]]*\])\s+(\[[^\]]*\])", _k04o.split("P4 --", 1)[1], re.M)}
_t6f = " ".join(_sec("06-results-clock.tex").split())
chk("6: 81 and 169 |delta| per seed (analyze_k04 P4)", (_p4["81"][1], _p4["169"][1]),
    (_nums(re.search(r"81 = 3\^\{4\}\$ at \$3/3\$ seeds within the threshold \(\$\\Delta = ([^)]*)\)", _t6f).group(1)),
     _nums(re.search(r"169 = 13\^\{2\}\$ at \$2/3\$ \(([^)]*)\)", _t6f).group(1))))
_k05 = _script("analyze_k05.py")
_q2 = [re.findall(r"'([\d.]+)'", re.search(rf"{k}\s*: (\[[^\]]*\])", _k05).group(1)) for k in ("Gini_mult per grokked seed", r"\|delta\| vs 0.559")]
chk("6: 49 at train_frac 0.50, Gini_mult and |delta| per seed (analyze_k05 Q2)", _q2,
    [_nums(x) for x in re.search(r"\\mathrm\{Gini\}_\{\\text\{mult\}\} = ([\d., ]+)\$ \(\$\\Delta = ([\d., ]+)\$\)", _t6f).groups()])
# 7, number debt.
chk("7 I1b: seed 0's excluded/baseline 2.881e+07 (the top of the n=121 range)",
    (f"{float(_i1b[0]['ratio_excluded_baseline']):.3e}", f"{float(_i1b[0]['ratio_excluded_baseline']) / 1e7:.3f}"), ("2.881e+07", "2.881"))   # the body wraps it as "2.881 \\times" / "10^{7}"
_dose = lambda z: float(z["internal_restricted"] / z["internal_baseline"])
chk("7: restricted worse than baseline on seeds 1 and 2: 2.5x, 1.4x (113), 13.6x, 1.39x (121)",
    (f"{_dose(_i1[1]):.1f}", f"{_dose(_i1[2]):.1f}", f"{_dose(_i1b[1]):.1f}", f"{_dose(_i1b[2]):.2f}"), ("2.5", "1.4", "13.6", "1.39"))
# C17 (1 seed): the vestigial k = 23 in Gate 1's addition model.
import test_gate1 as _tg1
_zg = np.load("results/gate1/add113_final.npz", allow_pickle=True)
_Wg = _zg["W_E"][:113].astype(float)
_fg = _tg1.freq_norms(_Wg - _Wg.mean(0, keepdims=True), 113)
_yg = (np.arange(113)[:, None] + np.arange(113)[None, :]).reshape(-1) % 113
_d23 = ablation_report(_zg["logits_all"].reshape(113, 113, 113), _yg, [1, 5, 23, 33, 45], 113)["per_freq"][23]
chk("7 C17: k = 23 embedding norm, and its logit ablation delta is below 4e-10",
    (f"{_fg[22]:.1f}", bool(abs(_d23) < 4e-10), "4e-10"), ("23.1", True, "4e-10"))
# P17 (approved 2026-09-29): F4's caption said the damaging draws "reach only 7.9". That was the
# B = 200 null's maximum; at the 10,000 draws the figure plots, the maximum is 10.93.
_rng0, _nc0 = np.random.default_rng, _an4.N_CONTROL
try:
    _an4.N_CONTROL = 200
    _, _d200 = _an4.run("B_thesis_n121_s0", 121, verbose=False, d="results/k03_grid_acts", return_draws=True)
finally:
    _an4.N_CONTROL = _nc0
chk("7 P17: the caption's 7.9 is the B = 200 null's maximum", f"{_d200.max():.2f}", "7.92")
_, _d10k = _an4.run("B_thesis_n121_s0", 121, verbose=False, d="results/k03_grid_acts", return_draws=True)
chk("7 P17: F4's null at B = 10,000: draws, maximum", (len(_d10k), f"{_d10k.max():.2f}"), (10000, "10.93"))
chk("7 P17: ...and the caption prints that maximum", (f"reach at most ${_d10k.max():.1f}$" in _t7w, "reach only" in _t7w), (True, False))

# 11, 12, 14: implementation facts, each from the script that measured it.
_tm = _script("test_model.py")
_fw, _gw = re.search(r"forward max \|ours - torch\| = ([\d.e+-]+)", _tm).group(1), re.search(r"worst relative gradient error ([\d.e+-]+)", _tm).group(1)
chk("11/14: worst gradient error vs PyTorch (test_model.py)", _gw, "2.68e-15")
# P18 (approved 2026-09-29): C10's "forward 6.1e-16, gradients 2.3e-15" (section 11 and the
# appendix row) was the first measurement (LAB_NOTEBOOK Entry 10). test_model.py, unchanged since
# init, now prints 4.441e-16 and 2.68e-15, and both places print those.
chk("11 P18: test_model.py's forward error now", _fw, "4.441e-16")
_m18, _e18 = f"{float(_fw):.1e}".split("e")
chk("11/14 P18: section 11 and the C10 row print test_model.py's values",
    (f"${_m18} \\times 10^{{{int(_e18)}}}$ on the forward pass and ${_gw[:4]} \\times 10^{{{int(_gw[5:])}}}$ on gradients" in " ".join(_sec("11-implementations.tex").split()),
     f"forward ${_m18}$e--${-int(_e18)}$, gradients ${_gw[:4]}$e--${-int(_gw[5:])}$" in _sec("14-appendices.tex")), (True, True))
chk("14 P18: the C10 row's mantissas (it splits them from their exponents)", (_m18, _gw[:4]), ("4.4", "2.68"))
chk("11: the O18 split is the same 3,830 training pairs at n = 113, train_frac 0.30", len(_md(113)[1]), 3830)
chk("11: T4 tail Gini_mult at n = 113, k03's five seeds", f"{np.mean([tail_gini(np.load(f, allow_pickle=True), 113)[0] for f in sorted(glob.glob('results/k03_grid_acts/WE_B_thesis_n113_s*.npz'))]):.4f}", "0.5665")
import analyze_o4 as _ao4
_tun18 = {d_: np.mean([_ao4.measure(f"results/{d_}/WE_B_thesis_n113_s{s_}.npz", 113)["tuned"] for s_ in range(3)]) for d_ in ("o18_cpu", "o18_f64")}
chk("11: float64 moves PyTorch's neuron tuning 0.938 -> 0.770 (analyze_o4.measure, O18 arms)",
    (f"{_tun18['o18_cpu']:.3f}", f"{_tun18['o18_f64']:.3f}"), ("0.938", "0.770"))
_o21p = _script("run_o21_probe.py", "float32")
chk("11: O21 probe -- 115 forward intermediates, 0 promoted to float64",
    (re.search(r"\((\d+) calls total\)", _o21p).group(1), re.search(r"CRITERION 1 \(forward promotion\) : (\d+) float64 sites", _o21p).group(1)), ("115", "0"))
_o21n = [float(x) for x in re.findall(r"ratio\s+([\d.]+)\s+equivalent \(A3\)", _script("run_o21_numerics.py"))]
chk("11: O21 numerics ratios span 1.000-1.172, all inside the band", (f"{min(_o21n):.3f}", f"{max(_o21n):.3f}", len(_o21n)), ("1.000", "1.172", 4))
chk("12: O18-F64's F5 -- 0 excursions over 453 post-grok samples (analyze_excursions)",
    re.search(r"(\d+) excursions over (\d+) post-grok samples", _script("analyze_excursions.py", "results/o18_f64")).groups(), ("0", "453"))
_c6 = [r for r in _h6 if r and any(e["censored"] for e in r["ex"])][0]["ex"]
_cz = [e for e in _c6 if e["censored"]][0]
_deep = min((e for e in _c6 if not e["censored"]), key=lambda e: e["min_acc"])
chk("14 C27: the censored run recovered from a deeper excursion 4,600 steps earlier",
    (_cz["start"] - _deep["start"], bool(_deep["min_acc"] < _cz["min_acc"])), (4600, True))
chk("7: n = 113's multiplicative embedding spectrum has 56 bins", len(_mamp(_W7, 113)), 56)
_i1s = _script("run_intervention.py", "--summary", "113")
_dose = re.search(r"seed 0:.*?dose: 0:([\d.e+-]+)\(n=\d+\)\s+1:([\d.e+-]+)\(n=\d+\)\s+2:([\d.e+-]+)\(n=\d+\)\s+3:([\d.e+-]+)", _i1s, re.S).groups()
chk("7: seed 0 damage by key-character overlap 0..3", [f"{float(x):.3g}" for x in _dose], ["1.26e-05", "4.95", "10.3", "16.5"])
chk("7: zero-overlap draws never exceed 1.05e-4",
    f"{max(float(x) for x in re.findall(r'zero-overlap draws \d+\s+mean [\d.e+-]+ \(.*?\)\s+max ([\d.e+-]+)', _i1s)):.2e}", "1.05e-04")

# 10, the gate-1 circuit (1 seed) and Softmax Collapse (C12/C13), at full value, not scaled.
from src.viz import mechinterp as _M
_top = _M.logit_top_components("results/gate1/add113_final.npz", 113, top=16, verbose=False)
_amp = [float(t[2]) for t in _top]
chk("10: gate-1 components (+-45), (+-5), (+-1) at 31.9 / 28.0 / 24.0% of (+-33)",
    [f"{100 * _amp[i] / _amp[0]:.1f}" for i in (2, 4, 6)] + [str(abs(int(_top[i][0]))) for i in (0, 2, 4, 6)],
    ["31.9", "28.0", "24.0", "33", "45", "5", "1"])
_e2 = {k: sum(float(t[2]) ** 2 for t in _top if t[3] == k) / sum(float(t[2]) ** 2 for t in _top) for k in ("a+b", "a-b")}
chk("10: top-16 energy share 100% a+b, 0% a-b", (f"{100 * _e2['a+b']:.0f}", f"{100 * _e2['a-b']:.0f}"), ("100", "0"))
_cz = np.load("results/c12_collapse/softmax_collapse_n17.npz", allow_pickle=True)
_C = [str(c) for c in _cz["hist_cols"]]; _sm, _st = _cz["softmax"], _cz["stablemax"]
_gi, _pi, _mi = _C.index("grad_norm"), _C.index("p_correct"), _C.index("max_logit")
chk("10: softmax gradient 3.41e-1 -> 2.35e-7; stablemax 7.51e-7 at the same step",
    (f"{_sm[0, _gi]:.2e}", f"{_sm[-1, _gi]:.2e}", f"{_st[-1, _gi]:.2e}", int(_sm[-1, _C.index('step')])),
    ("3.41e-01", "2.35e-07", "7.51e-07", 4000))
chk("10: correct-class p reaches 1 - 3.4e-8 (softmax) and 0.999998 (stablemax)",
    (f"{1 - _sm[:, _pi].max():.1e}", f"{_st[:, _pi].max():.6f}"), ("3.4e-08", "0.999998"))
chk("10: the logit scales differ by a factor of 383", f"{_st[-1, _mi] / _sm[-1, _mi]:.0f}", "383")

# 16: the appendix says the self-check exhausts the Setup proposition and theorem up to n = 200.
import inspect as _insp
from src.tasks import algebra as _alg
_rg = re.search(r"for n in range\(2, (\d+)\):\s+torsor\(n\)\s+assert stratum_identity\(n\)", _insp.getsource(_alg._selfcheck))
chk("16: algebra.py's self-check exhausts torsor + stratum identity up to n = 200", int(_rg.group(1)) - 1, 200)

# P19 (researcher, 2026-09-30: "fix the 2 entries"). Section 7's key-amplitude share, on the
# protocol helper. The paper printed 80.8% from a path no longer in the repo; the helper
# reads 81.6%, and restricted/excluded are checked against the same spectrum.
import analyze_n7 as _an7
from src.analysis.intervention import character_project as _cproj
from src.analysis.sparsity import key_freqs_5x_median as _k5
_iz = np.load("results/i1_internal/I1_engine_n113_s0.npz", allow_pickle=True)
_ik = [k[0] for k in json.loads(str(_iz["key_freqs"]))]
_iw = np.load("results/i1_internal/WE_engine_n113_s0.npz")["W_E"]
_ia = _an7.mult_amplitude(_iw, 113)
_im = np.zeros(len(_ia), bool); _im[[k - 1 for k in _ik]] = True
chk("7 P19: key set holds 81.6% of embedding amplitude, seed 0", f"{_ia[_im].sum() / _ia.sum():.1%}", "81.6%")
chk("7 P19: ...in 8 of 56 bins", (int(_im.sum()), len(_ia)), (8, 56))
_ir = _an7.mult_amplitude(_cproj(_iw, 113, keep=_ik), 113)
_ie = _an7.mult_amplitude(_cproj(_iw, 113, remove=_ik), 113)
chk("7 P19: restricted keeps key amplitude, complement < 1e-13",
    (bool(np.allclose(_ir[_im], _ia[_im], rtol=1e-12)), bool(_ir[~_im].max() < 1e-13)), (True, True))
chk("7 P19: excluded zeroes the key bins, keeps the rest",
    (bool(_ie[_im].max() < 1e-13), bool(np.allclose(_ie[~_im], _ia[~_im], rtol=1e-12))), (True, True))
chk("7 P19: the 5x-median detector recovers the key set, 3 of 3 seeds",
    [[k + 1 for k in _k5(_an7.mult_amplitude(np.load(f"results/i1_internal/WE_engine_n113_s{s}.npz")["W_E"], 113))]
     == [k[0] for k in json.loads(str(np.load(f"results/i1_internal/I1_engine_n113_s{s}.npz", allow_pickle=True)["key_freqs"]))]
     for s in range(3)], [True] * 3)
chk("7 P19: ...and the paper prints 81.6%", "the key set holds $81.6\\%$ of embedding amplitude in $8$ of $56$"
    in " ".join(_secsC["07-results-causal.tex"].split()), True)
# P20 (same approval). C20's retracted rho(cells, variance) and its flat-variance noise
# control. The paper printed +0.85..+1.00 and +0.918 from ordinal ranks (jblocks at 477e067);
# midranks, the definition of Spearman under ties (the 2026-09-23 audit), read below.
from src.analysis import jblocks as _jb
_c20 = [_jb.naive_size_rho(np.load(f"results/k02_grid/WE_B_thesis_n{m}_s0.npz")["mlp_acts"], m)
        for m in (113, 119, 120, 121, 125, 165)]
chk("14 P20: C20 rho(cells, variance), k02 seed 0, six moduli", (f"{min(_c20):+.3f}", f"{max(_c20):+.3f}"), ("+0.856", "+0.949"))
chk("14 P20: the flat-variance noise control at n = 125",
    f"{_jb.naive_size_rho(np.random.default_rng(0).normal(0, 1.0, size=(125 * 125, 8)), 125):+.3f}", "+0.942")
# P21 (same day): "every absolute sparsity number in this paper is a float32 measurement" was
# false as unscoped -- the paper prints float64 cells (the 2x2, the engine arms). Scoped to PyTorch.
chk("11/15 P21: no unscoped 'every absolute sparsity number ... float32'",
    [f for f, t in _secsC.items() if re.search(r"every absolute sparsity number in this paper", " ".join(t.split()))], [])
chk("14 P20: ...and the paper prints both", "runs $+0.856$ to $+0.949$, and a flat-variance \\emph{noise} control scores $\\rho = +0.942$"
    in " ".join(_secsC["14-appendices.tex"].split()), True)
# L-F1 / L-F2 (researcher, 2026-09-30), found by the Lean formalisation (E4, A6). F2's caption
# printed "one stratum at a prime"; strata are J_d x J_e, so the count is the squared class count.
_s1 = " ".join(_secsC["01-setup.tex"].split())
chk("1 L-F1: strata at 113, 121, 125, 119, 120", [len(_alg.j_structure(m)) ** 2 for m in (113, 121, 125, 119, 120)], [4, 9, 16, 16, 256])
chk("1 L-F1: ...and F2's caption prints them", "There are $4$ strata at a prime, $9$ at $11^{2}$, $16$ at $5^{3}$ and at $7 \\cdot 17$, and $256$" in _s1, True)
# 0 is nilpotent in every ring; only NONZERO nilpotents track square-freeness.
from sympy import factorint as _fi
chk("1 L-F2: a nonzero nilpotent exists iff n is not square-free, n < 200",
    [m for m in range(2, 200) if any(x for j in _alg.j_structure(m) for x in j["nilpotents"]) != (max(_fi(m).values()) > 1)], [])
chk("1 L-F2: ...and the paper says x != 0", ("($x \\neq 0$ with $x^{k} = 0$" in _s1, "($x$ with $x^{k} = 0$" in _s1), (True, False))
# L-F3 (researcher, 2026-10-01): "these classes contain nilpotents" is false read universally.
# Every non-square-free n has a non-regular class holding a nonzero nilpotent (J_{n/p}), but not every one does.
_nr = lambda m: [j for j in _alg.j_structure(m) if not j["regular"]]
chk("0 L-F3: every non-square-free n < 200 has a non-regular class with a nonzero nilpotent",
    [m for m in range(2, 200) if max(_fi(m).values()) > 1 and not any(any(j["nilpotents"]) for j in _nr(m))], [])
chk("0 L-F3: ...but at n = 120 only 2 of its 8 non-regular classes do", (sum(any(j["nilpotents"]) for j in _nr(120)), len(_nr(120))), (2, 8))
_ab = " ".join(_secsC["00-abstract.tex"].split())
chk("0 L-F3: ...and the abstract says 'some', 'nonzero'",
    ("some of these classes contain nonzero nilpotents" in _ab, "and these classes contain nilpotents" in _ab), (True, False))
if pathlib.Path("for arxiv/abstract.txt").exists():   # the arXiv metadata abstract; not in the public export
    _aa = pathlib.Path("for arxiv/abstract.txt").read_text()
    chk("0 L-F3: ...and so does the arXiv metadata abstract",
        ("some of whose non-regular classes contain nonzero nilpotents" in _aa, "whose non-regular classes contain nilpotents" in _aa), (True, False))

print("\nNUMBER COVERAGE (P7) -- every number the paper prints is asserted above or ledgered")
# A number is covered when a PASSING check above compared a value that rounds to it at the
# precision the paper prints. minimal-solution: matching is by VALUE, so a coincidental equal value
# elsewhere can cover a number no check is about; a gap it reports is always real. Upgrade
# path: key each check to the (file, line) it scores.
# Anything uncovered must be in paper/number_ledger.json, keyed to its exact line so an edit
# re-opens it: "exempt" (not our result -- a literature value, an environment fact) with a
# reason, or "todo" with its source -- a pinning debt this prints every run. An uncovered
# number with no entry FAILS, so no new unasserted number can enter the paper.
_V = np.array(_ASSERTED)
_SCI = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(?:\\mathrm\{e\}\{([-+]?\d+)\}|\\times\s*10\^\{?([-+]?\d+)\}?)")
def _covered(tok, exp=0, V=None):
    V = _V if V is None else V
    t = tok.replace(",", "")
    dec = len(t.split(".")[1]) if "." in t else 0
    v, tol = float(t) * 10.0 ** exp, 0.5 * 10.0 ** (exp - dec) * (1 + 1e-9)
    return any(np.any(np.abs(V * sc - v) <= tol) for sc in (1, 100, 0.01))
def number_tokens():
    """(key, file, line, token, text) for every number printed in paper/sections."""
    import hashlib
    for f in sorted(pathlib.Path("paper/sections").glob("*.tex")):
        for i, l in enumerate(f.read_text().splitlines(), 1):
            if l.lstrip().startswith("%"):
                continue
            s = re.sub(r"\\(ref|eqref|label|cite[tp]?|includegraphics|url|href|input|texttt)(\[[^]]*\])?\{[^}]*\}", "", l)
            s = s.replace("{,}", ",")
            h = hashlib.sha1(" ".join(l.split()).encode()).hexdigest()[:10]
            for m in _SCI.finditer(s):
                yield f"{f.name}:{h}:{m.group(0)}", f.name, i, (m.group(1).rstrip(","), int(m.group(2) or m.group(3))), l.strip()
            s = re.sub(r"10\^\{?[-+]?\d+\}?", " ", _SCI.sub(" ", s))   # a bare power of ten is a magnitude
            for m in re.finditer(r"(?<![\w.\\^_{])(\d[\d,]*(?:\.\d+)?)(?![\w])", s):
                tok = m.group(1).rstrip(",")
                if not re.fullmatch(r"\d", tok):   # single digits are counts, section numbers, seeds
                    yield f"{f.name}:{h}:{tok}", f.name, i, (tok, 0), l.strip()
_ledger = json.loads(pathlib.Path("paper/number_ledger.json").read_text())
_unc = [(k, f, i, t, l) for k, f, i, t, l in number_tokens() if not _covered(*t)]
# A number whose only check was skipped for an input that does not ship is reported, not failed.
_skp = [x for x in _unc if x[0] not in _ledger and _covered(*x[3], V=np.array(_SKIPPED))]
_unc = [x for x in _unc if x not in _skp]
for k, f, i, t, l in _skp:
    print(f"  [SKIP] not re-derivable here (its input is not shipped): {f}:{i} {t[0]}")
_new = [(f, i, t[0]) for k, f, i, t, l in _unc if k not in _ledger]
if os.environ.get("NUMBER_LEDGER_DUMP"):
    json.dump([dict(key=k, file=f, line=i, tok=t[0], exp=t[1], text=l) for k, f, i, t, l in _unc],
              open(os.environ["NUMBER_LEDGER_DUMP"], "w"), indent=0)
_todo = [k for k, v in _ledger.items() if v.get("status") == "todo"]
_bad = [k for k, v in _ledger.items() if v.get("status") == "exempt" and not v.get("why")]
_keys = {k for k, *_ in number_tokens()}
_uk = {k for k, *_ in _unc}
_gone = sorted(set(_ledger) - _keys)
# A todo that a check now covers must be verified: delete it if a pin really asserts it, or mark
# it "coincidental" (with why) if the covering value belongs to a different claim. An exempt
# entry is not stale when a value happens to coincide -- it was never ours to assert.
_stale = _gone + sorted(k for k, v in _ledger.items() if k in _keys and k not in _uk
                        and v.get("status") == "todo" and not v.get("coincidental"))
if os.environ.get("NUMBER_LEDGER_STALE"):
    json.dump(_stale, open(os.environ["NUMBER_LEDGER_STALE"], "w"))
chk("coverage: every uncovered number has a ledger entry", _new, [])
chk("coverage: every exemption states its reason", _bad, [])
chk("coverage: no ledger entry for a number that is now asserted (or gone)", _stale, [])
print(f"  numbers printed: {sum(1 for _ in number_tokens())}   asserted: "
      f"{sum(1 for _ in number_tokens()) - len(_unc)}   exempt: {sum(v.get('status') == 'exempt' for v in _ledger.values())}"
      f"   TODO (pinning debt): {len(_todo)}")

print(f"\n{ok} passed, {fail} failed")
raise SystemExit(1 if fail else 0)
