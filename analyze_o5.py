"""O5 is NOT ANSWERABLE from the data on disk: primality has one run in it.

  PYTHONPATH=. .venv/bin/python analyze_o5.py
  PYTHONPATH=. .venv/bin/python analyze_o5.py --selfcheck

O5 asked: "Is Softmax Collapse composite-modulus-first?", raised from one scout observation
-- max logit 156 (n=121) vs 54 (n=113) at step 300.

The sighting reproduces. The inference does not, for a reason that is structural rather than
statistical: **113 is the only prime this project has ever trained** (18 moduli, k01-k08 and
every engine run), so any prime-vs-composite contrast has ONE RUN in one arm. Seed variance
on this box spans a factor of 3.8 on grokking time (O2), so a 1-run arm cannot carry a
group effect no matter which statistic is applied to it.

It is worse than under-powered, it is unidentifiable. In k01_scout -- the ONLY results
directory besides gate1 that saved `max_logit`, and the one O5 was raised from -- that single
prime is simultaneously:

  * the MINIMUM modulus of the set (113 < 119 < 120 < 121 < 125 < 165), and
  * the FASTEST to memorise (train_acc 1.0 at step 200; the others 300-2000),

so "prime", "smallest", and "furthest into training at a fixed step" name the same run and no
contrast among them can be separated. Same family as C7b / C19 / C20: the grouping variable is
structurally coupled to a size, and the statistic reads the size.

Reading it at the END of training reverses the sighting -- n=113 finishes with the HIGHEST max
logit of the six (204.4) -- which is what a one-run arm looks like.

RESOLUTION: not a null and not a confirmation. O5 needs a prime ABOVE the composite range
(127 or 131) at >= 3 seeds, with `max_logit` logged, before the question has any content.
"""
import glob, os, re, sys
import numpy as np

SCOUT = "results/k01_scout/scout_n*_seed0.npz"


def is_prime(n):
    return n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))


def scout_table(pattern=SCOUT):
    """Per-modulus early/late max_logit and the step memorisation completes."""
    rows = []
    for f in sorted(glob.glob(pattern)):
        n = int(re.search(r"_n(\d+)_", os.path.basename(f)).group(1))
        z = np.load(f)
        h, c = z["hist"], [str(x) for x in z["hist_cols"]]
        step, ml = h[:, c.index("step")], h[:, c.index("max_logit")]
        tr = h[:, c.index("train_acc")]
        mem = next((int(s) for s, a in zip(step, tr) if a >= 0.9999), None)
        rows.append(dict(n=n, prime=is_prime(n),
                         ml300=float(ml[int(np.argmin(abs(step - 300)))]),
                         mem_step=mem, ml_final=float(ml[-1])))
    return sorted(rows, key=lambda r: r["n"])


def verdict(moduli, mem_steps=None):
    """Can a prime-vs-composite contrast be identified over this modulus set?

    Returns (answerable, reasons). Unidentifiable when one arm has a single run, or when the
    primes do not straddle the composites (primality then co-varies perfectly with size).
    """
    primes = sorted(m for m in moduli if is_prime(m))
    comps = sorted(m for m in moduli if not is_prime(m))
    reasons = []
    if len(primes) < 2:
        reasons.append(f"prime arm has {len(primes)} modulus/moduli -- cannot carry a group effect")
    if primes and comps and (max(primes) < min(comps) or min(primes) > max(comps)):
        reasons.append(f"primes {primes} are separated from composites"
                       f" [{min(comps)}, {max(comps)}] by a size threshold"
                       " -- primality co-varies with modulus size")
    if mem_steps and len(primes) == 1:
        p = primes[0]
        if mem_steps.get(p) == min(mem_steps.values()):
            reasons.append(f"the one prime n={p} is also the fastest to memorise"
                           f" (step {mem_steps[p]}) -- phase and primality are the same axis")
    return (not reasons), reasons


def corpus_primes():
    seen = set()
    for f in glob.glob("results/*/*.npz"):
        m = re.search(r"n(\d+)_s(?:eed)?\d+", os.path.basename(f))
        if m:
            seen.add(int(m.group(1)))
    return sorted(seen)


def _selfcheck():
    # a set with one prime that is also the smallest -> unidentifiable, all three reasons
    ok, why = verdict([113, 119, 120, 121, 125, 165], {113: 200, 119: 400, 121: 300})
    assert not ok and len(why) == 3, why
    # two primes straddling the composites -> identifiable
    ok, why = verdict([109, 120, 121, 125, 131])
    assert ok, why
    # two primes, both below every composite -> still size-confounded
    ok, why = verdict([109, 113, 120, 121, 125])
    assert not ok and any("size threshold" in w for w in why), why
    assert is_prime(113) and not is_prime(121) and not is_prime(1)
    print("analyze_o5 selfcheck PASS  (1-run arm, straddle test, memorisation-phase trap)")


def main():
    rows = scout_table()
    if not rows:
        print("k01_scout not on disk -- skipped")
        return
    print("O5 -- Softmax Collapse proxy (max logit) over the scout, the only run set with max_logit\n")
    print(f"{'n':>5} {'prime':>6} {'mem_step':>9} {'maxlogit@300':>13} {'maxlogit@final':>15}")
    for r in rows:
        print(f"{r['n']:>5} {str(r['prime']):>6} {str(r['mem_step']):>9}"
              f" {r['ml300']:>13.2f} {r['ml_final']:>15.2f}")

    top300 = max(rows, key=lambda r: r["ml300"])
    topfin = max(rows, key=lambda r: r["ml_final"])
    print(f"\n  highest at step 300 : n={top300['n']} ({top300['ml300']:.1f})"
          f"   -- omega(125)=1, a PRIME POWER, not a many-factor composite")
    print(f"  highest at the end  : n={topfin['n']} ({topfin['ml_final']:.1f})"
          f"   -- the ordering REVERSES with when you look")

    mem = {r["n"]: r["mem_step"] for r in rows}
    ok, why = verdict([r["n"] for r in rows], mem)
    print(f"\nscout set: identifiable = {ok}")
    for w in why:
        print(f"  - {w}")

    allm = corpus_primes()
    pr = [m for m in allm if is_prime(m)]
    print(f"\ncorpus: {len(allm)} moduli ever trained, {len(pr)} prime {pr}")
    print(f"  {sorted(allm)}")
    print("\nVERDICT: O5 is NOT ANSWERABLE on existing data -- not a null, a design gap.")
    print("  Fix: a prime ABOVE the composite range (127 or 131), >= 3 seeds, max_logit logged.")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()
