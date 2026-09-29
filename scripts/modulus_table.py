"""The paper's modulus table (Setup, B1) -- carries C1 and C2.

  PYTHONPATH=. .venv/bin/python scripts/modulus_table.py            # markdown
  PYTHONPATH=. .venv/bin/python scripts/modulus_table.py --latex    # paper/sections body
  PYTHONPATH=. .venv/bin/python scripts/modulus_table.py --selfcheck

Every algebraic column comes from src/tasks/algebra.describe (C1, self-checking); the
grokked column is counted from the primary arm only -- B_thesis, train_frac 0.30, 40k
steps, 1L d_model 128 -- which is k03/k04/k09 and nothing else. k05 (train_frac sweep),
k06 (horizon), k07 (extra seeds) and k08 (init) vary a hyperparameter and would make the
column mean different things at different moduli.

Runs are classified by the median of the final 10 samples, never the last row (C27), and
test accuracy is read by column name from hist_cols, never by index (the engine and the
kernels put different quantities at index 3).
"""
import collections, glob, os, sys
import numpy as np
from sympy import factorint

from src.tasks import algebra

PRIMARY_ARMS = ["k03_grid_acts", "k04_extended", "k09_primes_zdd"]
GROK = 0.99


def grok_counts(dirs=PRIMARY_ARMS):
    per = collections.defaultdict(lambda: [0, 0])
    for d in dirs:
        for f in sorted(glob.glob(f"results/{d}/WE_B_thesis_n*_s*.npz")):
            n = int(os.path.basename(f).split("_n")[1].split("_")[0])
            z = np.load(f, allow_pickle=True)
            cols = [str(c) for c in z["hist_cols"]]
            acc = z["hist"][:, cols.index("test_acc")]
            per[n][1] += 1
            if float(np.median(acc[-10:])) > GROK:
                per[n][0] += 1
    return {n: tuple(v) for n, v in per.items()}


def fact_str(n, tex=False):
    f = sorted(factorint(n).items())
    parts = [str(p) if e == 1 else (f"{p}^{{{e}}}" if tex else f"{p}^{e}") for p, e in f]
    return r" \cdot ".join(parts) if tex else "·".join(parts)


def rows():
    g = grok_counts()
    out = []
    for n in sorted(g):
        d = algebra.describe(n)
        out.append(dict(n=n, fact=n, omega=d["omega"], sf=d["squarefree"],
                        cyclic=d["cyclic"], phi=d["phi"],
                        zdd=d["zero_divisors"] / n, nj=d["n_jclasses"],
                        nr=d["n_nonregular"], grok=g[n][0], seeds=g[n][1]))
    return out


def markdown(rs):
    print("| n | factorisation | ω | sf | cyc | φ(n) | zdd | 𝒥 | non-reg | grokked |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rs:
        print(f"| {r['n']} | {fact_str(r['n'])} | {r['omega']} | "
              f"{'y' if r['sf'] else 'n'} | {'y' if r['cyclic'] else 'n'} | {r['phi']} | "
              f"{r['zdd']:.3f} | {r['nj']} | {r['nr']} | {r['grok']}/{r['seeds']} |")


def latex(rs):
    for r in rs:
        print(f"    {r['n']} & ${fact_str(r['n'], tex=True)}$ & {r['omega']} & "
              f"{'\\yes' if r['sf'] else '\\no'} & {'\\yes' if r['cyclic'] else '\\no'} & "
              f"{r['phi']} & {r['zdd']:.3f} & {r['nj']} & {r['nr']} & "
              f"{r['grok']}/{r['seeds']} \\\\")


def _selfcheck():
    rs = rows()
    # MASTER_PLAN.md 0.1 -- the counts that go in the abstract.
    assert len(rs) == 23, len(rs)
    assert sum(1 for r in rs if not r["sf"]) == 14
    assert sum(1 for r in rs if r["omega"] == 1 and not r["sf"]) == 6   # prime powers
    assert sum(1 for r in rs if r["omega"] == 1 and r["sf"]) == 3       # primes
    # C2 verbatim: non-regular J-class counts for 113,119,121,125,120.
    nr = {r["n"]: r["nr"] for r in rs}
    assert [nr[n] for n in (113, 119, 121, 125, 120)] == [0, 0, 1, 2, 8]
    # Chen Thm D.17: square-free => every J-class regular. Holds on ALL of ours, not just
    # theirs -- and the converse holds too, which is the sentence the Setup section makes.
    assert all(r["nr"] == 0 for r in rs if r["sf"])
    assert all(r["nr"] >= 1 for r in rs if not r["sf"])
    # 128 is the only non-cyclic prime power -- the clause MASTER_PLAN 0.1 insists on.
    pp = [r["n"] for r in rs if r["omega"] == 1 and not r["sf"]]
    assert [r["n"] for r in rs if r["n"] in pp and not r["cyclic"]] == [128]
    print(f"selfcheck OK: {len(rs)} moduli, 14 non-square-free, 6 prime powers, C2 exact")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    elif "--latex" in sys.argv:
        latex(rows())
    else:
        markdown(rows())
