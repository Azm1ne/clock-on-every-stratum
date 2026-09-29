#!/usr/bin/env python3
"""R1-4: a re-run at B = 10,000 must move only the draw-derived values.

The pre-registration asks for this as a literal diff wherever the artifact is
tracked, so this compares the committed json against the working tree one and
asserts every B-independent leaf is unmoved. Run it before reading a single p.

    PYTHONPATH=. .venv/bin/python scripts/r1_diff.py results/gate2/n7_engine.json
    PYTHONPATH=. .venv/bin/python scripts/r1_diff.py --selfcheck

Written for the engine arm after k03's was done by hand (Entry 57); it runs
four times in total -- k03, n7_engine, and the two --controls arms -- which is
why it is a script and not a heredoc.

Reconciled against that hand count, so the two numbers are not mistaken for a
discrepancy later. On k03 this prints 23,826 where Entry 57 reports 23,710:

    23,518  stratum leaves          <- Entry 57 counted these
       192  verdict flags (32 x 6)  <- and these          = 23,710
       116  run-level (29 x n, seed, split_recomputed, split_logged)

The 116 are an addition, not a disagreement: this asserts that the modulus, the
seed and both split-check accuracies are unmoved too. Nothing is dropped.
"""
import json, math, subprocess, sys

# Estimated FROM the permutation draws, so they are expected to move.
B_DEP = {"p_perm", "n_ge", "n_draws", "ctrl_median", "resid_ctrl"}
# analyze_gate2.py:557,560 -- G1 thresholds p_perm, G3 thresholds resid_ctrl.
# G0/G2/G4/G5 read acc_heldout/excluded/R/tuned and are B-independent.
B_DEP_VERDICT = {"G1", "G3"}
# A re-run legitimately restamps git_sha, git_dirty and the timestamp. Comparing
# them would fail R1-4 on noise, and the k03 dry run only passed because that file
# was untouched. Reported separately instead of asserted.
SKIP_SUBTREE = {"provenance"}
TOL = 1e-9


def _is_b_dep(path):
    key = path[-1]
    if key in B_DEP:
        return True
    return key in B_DEP_VERDICT and "verdict" in path


def _key_list(lst):
    """Natural identity for a list of runs or strata, or None if it has none.

    Runs are keyed by (n, seed) and strata by (d, e); both are unique within
    their parent. Falls back to positional comparison for anything else.
    """
    for keys in (("n", "seed"), ("d", "e")):
        if lst and all(isinstance(x, dict) and all(k in x for k in keys) for x in lst):
            ks = [tuple(x[k] for k in keys) for x in lst]
            if len(set(ks)) == len(ks):
                return ks
    return None


def walk(a, b, path=(), moved=None, n_indep=None, n_dep=None, struct=None):
    """Compare two parsed artifacts in parallel. Structure mismatch is fatal."""
    if type(a) is not type(b):
        struct.append(f"{'.'.join(map(str, path))}: type {type(a).__name__} -> {type(b).__name__}")
        return
    if path and path[-1] in SKIP_SUBTREE:
        return
    if isinstance(a, dict):
        if set(a) != set(b):
            only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
            struct.append(f"{'.'.join(map(str, path))}: keys -{only_a} +{only_b}")
            return
        for k in a:
            walk(a[k], b[k], path + (k,), moved, n_indep, n_dep, struct)
    elif isinstance(a, list):
        # Align by identity, not by index. A length mismatch used to abort the whole
        # subtree, so when the k04 control arm gained one run (6 -> 7) the six SHARED
        # runs were never compared and the report still printed a verdict. A check
        # whose failure mode looks like its success mode is not a check.
        ka, kb = _key_list(a), _key_list(b)
        if ka is not None and kb is not None:
            only_a, only_b = sorted(set(ka) - set(kb)), sorted(set(kb) - set(ka))
            for k in only_a:
                struct.append(f"{'.'.join(map(str, path))}: REMOVED {k}")
            for k in only_b:
                struct.append(f"{'.'.join(map(str, path))}: ADDED {k}")
            da, db = dict(zip(ka, a)), dict(zip(kb, b))
            for k in sorted(set(ka) & set(kb)):
                walk(da[k], db[k], path + (k,), moved, n_indep, n_dep, struct)
            return
        if len(a) != len(b):
            struct.append(f"{'.'.join(map(str, path))}: length {len(a)} -> {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, path + (i,), moved, n_indep, n_dep, struct)
    else:
        if _is_b_dep(path):
            n_dep.append(path)
            return
        n_indep.append(path)
        if isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, (int, float)):
            same = a == b
        elif math.isnan(a) and math.isnan(b):
            # NaN != NaN, so an untouched NaN reads as MOVED. Three acc_heldout
            # values in k03 did exactly that on the first dry run against an
            # artifact that had not changed at all.
            same = True
        else:
            same = abs(a - b) <= TOL * max(1.0, abs(a), abs(b))
        if not same:
            moved.append((".".join(map(str, path)), a, b))


def compare(old, new):
    moved, n_indep, n_dep, struct = [], [], [], []
    walk(old, new, (), moved, n_indep, n_dep, struct)
    return moved, n_indep, n_dep, struct


def measurable_set(art):
    return {(r["n"], r["seed"], s["d"], s["e"])
            for r in art.get("runs", []) for s in r.get("strata", [])
            if s.get("measurable")}


def main(path, ref):
    # Not in reproduce.sh on purpose: from a bare clone `git show HEAD:path` IS the
    # working tree, so this would pass vacuously. Its planted cases are the real
    # coverage, so run them on every invocation rather than only on --selfcheck.
    _selfcheck()
    old = json.loads(subprocess.run(["git", "show", f"{ref}:{path}"],
                                    capture_output=True, text=True, check=True).stdout)
    new = json.loads(open(path).read())
    moved, n_indep, n_dep, struct = compare(old, new)

    print(f"R1-4  {path}   {ref} -> working tree")
    print(f"  runs        {len(old.get('runs', []))} -> {len(new.get('runs', []))}")
    ns = lambda a: sum(len(r.get("strata", [])) for r in a.get("runs", []))
    print(f"  strata      {ns(old)} -> {ns(new)}")
    mo, mn = measurable_set(old), measurable_set(new)
    print(f"  measurable  {len(mo)} -> {len(mn)}   "
          f"{'IDENTICAL SET' if mo == mn else f'DIFFERS: -{len(mo-mn)} +{len(mn-mo)}'}")
    po, pn = old.get("provenance", {}), new.get("provenance", {})
    pmoved = {k: (po.get(k), pn.get(k)) for k in set(po) | set(pn) if po.get(k) != pn.get(k)}
    print(f"  provenance  (not asserted) {'unchanged' if not pmoved else ''}")
    for k, (a, b) in sorted(pmoved.items()):
        print(f"                {k}: {a!r} -> {b!r}")
    print(f"  B-dependent (expected to move, not asserted): {len(n_dep)}")
    print(f"  B-independent compared at {TOL:g}: {len(n_indep)}")

    fail = []
    if struct:
        fail.append(f"{len(struct)} STRUCTURAL mismatches")
        for s in struct[:10]:
            print(f"    STRUCT {s}")
    if mo != mn:
        fail.append("measurable set changed")
    # Two empty sets compare equal, so a walk that found nothing is not a pass.
    if not n_indep:
        fail.append("compared ZERO B-independent values -- the walk found nothing")
    if moved:
        fail.append(f"{len(moved)} B-independent values MOVED")
        for p, a, b in moved[:20]:
            print(f"    MOVED {p}: {a!r} -> {b!r}")

    print(f"\n  R1-4 {'FAIL: ' + '; '.join(fail) if fail else 'PASS -- 0 moved'}")
    return 1 if fail else 0


def _selfcheck():
    base = {"runs": [{"n": 113, "seed": 0, "strata": [
        {"d": 1, "e": 1, "measurable": True, "baseline": 1.0, "excluded": 2.0,
         "p_perm": 0.0, "n_ge": 0, "n_draws": 200, "ctrl_median": 3.0,
         "resid": 0.5, "resid_ctrl": 1.0, "acc_heldout": 1.0}]}],
        "verdict": {"113_J1xJ1": {"G0": True, "G1": True, "G2": True, "G3": True}}}
    import copy
    # 1. only the B-dependent leaves move -> PASS, and they are counted, not compared.
    b = copy.deepcopy(base)
    b["runs"][0]["strata"][0].update(p_perm=9.999e-05, n_ge=0, n_draws=10000, ctrl_median=3.5)
    b["runs"][0]["strata"][0]["resid_ctrl"] = 1.4
    b["verdict"]["113_J1xJ1"]["G3"] = False
    moved, ind, dep, st = compare(base, b)
    assert not moved and not st, (moved, st)
    # 7 = the 5 stratum leaves + BOTH verdict flags. dep counts every B-dependent
    # leaf ENCOUNTERED, not the ones that moved -- G1 is exempt whether or not it did.
    assert len(dep) == 7, dep
    assert ind, "walk found no B-independent leaves"
    # 2. a B-independent leaf moves -> caught.
    b2 = copy.deepcopy(b); b2["runs"][0]["strata"][0]["excluded"] = 2.0000001
    moved, _, _, _ = compare(base, b2)
    assert len(moved) == 1 and moved[0][0].endswith("excluded"), moved
    # 3. G2 is NOT exempt just because it is a G -- only G1/G3 are.
    b3 = copy.deepcopy(b); b3["verdict"]["113_J1xJ1"]["G2"] = False
    moved, _, _, _ = compare(base, b3)
    assert len(moved) == 1 and moved[0][0].endswith("G2"), moved
    # 4. a key named like a B-dependent one OUTSIDE verdict is still compared.
    b4 = copy.deepcopy(b); b4["runs"][0]["strata"][0]["acc_heldout"] = 0.9
    assert compare(base, b4)[0], "acc_heldout must be compared"
    # 5. structure changes are fatal, not silently skipped.
    b5 = copy.deepcopy(b); b5["runs"].append(b5["runs"][0])
    assert compare(base, b5)[3], "run count change must be structural"
    b6 = copy.deepcopy(b); b6["runs"][0]["strata"][0].pop("baseline")
    assert compare(base, b6)[3], "missing key must be structural"
    # 6. NaN -> NaN is NOT a move. Three k03 acc_heldout values failed on this.
    bn = copy.deepcopy(base); bn["runs"][0]["strata"][0]["acc_heldout"] = float("nan")
    bn2 = copy.deepcopy(bn)
    assert not compare(bn, bn2)[0], "nan -> nan must not count as moved"
    bn3 = copy.deepcopy(bn); bn3["runs"][0]["strata"][0]["acc_heldout"] = 1.0
    assert compare(bn, bn3)[0], "nan -> number must count as moved"
    # 7. provenance is skipped: a re-run restamps it and that is not a failure.
    bp = copy.deepcopy(base); bp["provenance"] = {"git_sha": "aaa"}
    base_p = copy.deepcopy(base); base_p["provenance"] = {"git_sha": "bbb"}
    assert not compare(base_p, bp)[0] and not compare(base_p, bp)[3], "provenance must be skipped"
    # 8. A list that gains an element must still compare the shared ones
    #    (k04's control arm went from 6 to 7 runs).
    grown = copy.deepcopy(base)
    extra = copy.deepcopy(base["runs"][0]); extra["seed"] = 2
    grown["runs"].append(extra)
    grown["runs"][0]["strata"][0]["excluded"] = 99.0      # a move inside a SHARED run
    moved, ind, _, st = compare(base, grown)
    assert any("ADDED" in x for x in st), st
    assert len(moved) == 1 and moved[0][0].endswith("excluded"), \
        f"a gained run must not hide a move in the shared ones: {moved}"
    assert ind, "shared runs must still be compared"
    # 9. a measurable stratum flipping is caught by the set check, not the walk.
    b7 = copy.deepcopy(b); b7["runs"][0]["strata"][0]["measurable"] = False
    assert measurable_set(base) != measurable_set(b7)
    print("r1_diff self-check PASS")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck(); sys.exit(0)
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    ref = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--ref=")), "HEAD")
    sys.exit(main(args[0], ref))
