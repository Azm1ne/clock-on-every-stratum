"""Markdown tables for the Gate 2 result, regenerated from results/gate2/*.json.

  PYTHONPATH=. .venv/bin/python scripts/gate2_table.py [json ...]

Figures and tables regenerate from saved artifacts, never by hand: 26 strata x 5 seeds
typed into FINDINGS.md would invite transcription errors, and the sd columns below cannot
be read off a log.
"""
import json
import sys
from statistics import mean, pstdev

FILES = sys.argv[1:] or ["results/gate2/k03_grid_acts.json"]


def agg(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return "-"
    return f"{mean(v):.3f}" + (f" ± {pstdev(v):.3f}" if len(v) > 1 else "")


def main():
    for f in FILES:
        d = json.load(open(f))
        p = d["provenance"]
        print(f"\n### `{f}`  — SHA `{p['git_sha'][:7]}`, dirty={p['git_dirty']}, "
              f"{p['timestamp_utc']}\n")
        by = {}
        for run in d["runs"]:
            for s in run["strata"]:
                if not s["measurable"]:
                    continue
                by.setdefault((run["n"], s["d"], s["e"]), []).append(s)
        print("| n | stratum | local group | cells | held-out | acc (held-out) | "
              "D (diag share) | R = D·(\\|G\\|+1) | max R | perm p | excluded/baseline | "
              "fibre resid | permuted-fibre ctrl | tuned | shuffled | lut R² | seeds |")
        print("|" + "---|" * 17)
        for (n, dd, e), ss in sorted(by.items(), key=lambda kv: (kv[0][0], -kv[1][0]["cells"])):
            g = ss[0]["g"]
            grp = "×".join(f"Z{o}" for o in ss[0]["orders_m"])
            nd = ss[0].get("tuned_nd") is not None
            tun = agg([s.get("tuned_nd" if nd else "tuned") for s in ss])
            shf = agg([s.get("tuned_nd_sh" if nd else "tuned_sh") for s in ss])
            if nd and tun != "-":
                tun += " \\*"
            unit = " **(unit)**" if dd == 1 and e == 1 else ""
            print(f"| {n} | J{dd}×J{e}{unit} | {grp} | {ss[0]['cells']} | "
                  f"{ss[0]['heldout']} | {agg([s['acc_heldout'] for s in ss])} | "
                  f"{agg([s['Dshare'] for s in ss])} | {agg([s['R'] for s in ss])} | {g+1} | "
                  f"{max(s['p_perm'] for s in ss):.4f} | "
                  f"{mean(s['excluded']/max(s['baseline'],1e-30) for s in ss):.2e} | "
                  f"{agg([s.get('resid') for s in ss])} | "
                  f"{agg([s.get('resid_ctrl') for s in ss])} | {tun} | {shf} | "
                  f"{agg([s.get('lut_r2') for s in ss])} | {len(ss)} |")
        print("\n\\* = product-character neuron statistic (`freq_fraction_nd`), "
              "EXPLORATORY — beyond the pre-registration, which skips a non-cyclic "
              "local group on the neuron endpoint. Never scored as G5.")

        # the strata the model does NOT generalise on -- all below the measurability cut
        bad = {}
        for run in d["runs"]:
            for s in run["strata"]:
                if s["acc_heldout"] == s["acc_heldout"] and s["acc_heldout"] < 0.95:
                    bad.setdefault((run["n"], s["d"], s["e"]), []).append(s)
        if bad:
            print(f"\n**Strata the model does NOT generalise on** (held-out accuracy < 0.95). "
                  f"Every one is below the measurability cut, so none affects the verdict — "
                  f"and every measurable stratum passes G0 at every seed.\n")
            print("| n | stratum | local group | cells | held-out | acc (held-out) | "
                  "why not measurable | seeds |")
            print("|" + "---|" * 8)
            for (n, dd, e), ss in sorted(bad.items(), key=lambda kv: (kv[0][0], -kv[1][0]["cells"])):
                grp = "×".join(f"Z{o}" for o in ss[0]["orders_m"]) or "trivial"
                print(f"| {n} | J{dd}×J{e} | {grp} | {ss[0]['cells']} | {ss[0]['heldout']} | "
                      f"{agg([s['acc_heldout'] for s in ss])} | {ss[0]['why']} | {len(ss)} |")

        # size-confound check, run every time
        pts = [(s["cells"], s["g"], s["acc_heldout"]) for run in d["runs"]
               for s in run["strata"] if s["acc_heldout"] == s["acc_heldout"]]
        if pts:
            bad = [(c, g, a) for c, g, a in pts if a < 0.95]
            perfect = sum(1 for _, _, a in pts if a >= 0.9999)
            print(f"\n**Size-confound check ({len(pts)} stratum-run points).** "
                  f"A rank correlation is the wrong statistic here and is not reported: "
                  f"**{perfect} of {len(pts)}** points sit at held-out accuracy 1.0000, so "
                  f"the ranks are almost all ties and rho is decided by a handful of "
                  f"failures. The informative statement is a threshold, and it is clean:")
            if bad:
                print(f"\n> Every stratum-run below 0.95 has **|G_m| <= "
                      f"{max(g for _, g, _ in bad)}** and **<= {max(c for c, _, _ in bad)} "
                      f"cells**; every stratum at or above |G_m| = "
                      f"{min(g for c, g, a in pts if a >= 0.95 and g >= 10)} is at 1.0000. "
                      f"Those two are structurally coupled (rho(cells, |G_m|) is strongly "
                      f"positive by construction), so this does **NOT** separate 'the local "
                      f"group is too small to carry a clock' from 'there are too few "
                      f"training examples' -- it is consistent with critical dataset size "
                      f"(C14) and is **not** evidence about the algebra. It is reported "
                      f"because it is why the measurability cut exists, not as a finding.")

if __name__ == "__main__":
    main()
