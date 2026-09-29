"""Number theory for the modulus set. Pure CPU, no ML dependencies.

Everything the experimental design rests on is computed here, never looked up.
The monoid structure (Green's J-classes) follows Beyond-Groups (arXiv 2607.07066):
their mechanism needs each J-class to be *regular*, and non-regular classes are
their stated open problem -- which is this thesis.
"""
from math import gcd
from sympy import factorint, divisors, n_order, totient


def units(n):
    return [x for x in range(1, n) if gcd(x, n) == 1]


def primitive_root(n):
    """Smallest generator of (Z/nZ)*, or None if the group is not cyclic."""
    phi = totient(n)
    for g in units(n):
        if n_order(g, n) == phi:
            return g
    return None


def j_class(n, d):
    """Green's J-class J_d = {x in Z/nZ : gcd(x,n) = d}, for d | n."""
    return [x for x in range(n) if gcd(x, n) == d]


def j_structure(n):
    """Per-J-class structure. `regular` iff the class contains an idempotent --
    the condition under which local inverses, and hence local group characters,
    exist (2607.07066 Thm D.17: square-free n => every class regular)."""
    out = []
    for d in divisors(n):
        J = j_class(n, d)
        if not J:
            continue
        idem = [e for e in J if e * e % n == e]
        # x is nilpotent iff x^k == 0 for some k; k <= log2(n) suffices, but n is cheap
        nilp = [x for x in J if any(pow(x, k, n) == 0 for k in range(1, n.bit_length() + 1))]
        out.append({"d": d, "size": len(J), "idempotents": idem,
                    "regular": bool(idem), "nilpotents": nilp})
    return out


def describe(n):
    f = factorint(n)
    js = j_structure(n)
    g = primitive_root(n)
    return {
        "n": n,
        "factorization": dict(f),
        "omega": len(f),
        "squarefree": all(e == 1 for e in f.values()),
        "is_field": len(f) == 1 and next(iter(f.values())) == 1,
        "phi": int(totient(n)),
        "cyclic": g is not None,
        "primitive_root": g,
        "zero_divisors": n - len(units(n)),      # includes 0
        "n_jclasses": len(js),
        "n_nonregular": sum(1 for c in js if not c["regular"]),
        "n_nilpotent": sum(len(c["nilpotents"]) for c in js),
        "jclasses": js,
    }


def crt_decompose(n):
    """n -> its prime-power components. CRT applies iff there are >= 2 of them."""
    return [p ** e for p, e in sorted(factorint(n).items())]


PRIMARY = [113, 121, 125, 119, 120]
REPLICATION = [165, 143, 154]   # square-free, from 2607.07066 Table 1


def stratum_identity(n):
    """The Setup theorem, checked by exhaustion on Z/nZ rather than argued.

    For x in J_d, y in J_e write x = d*u, y = e*v; set m = gcd(de, n) and de = m*w.
    Then gcd(w, n/m) = 1 and

        x*y mod n  ==  m * ( w * (u*v) mod n/m )

    i.e. a stratum is a fixed unit relabelling of multiplication in the smaller ring
    Z/(n/m)Z, on the local group G_m = (Z/(n/m)Z)*. Returns the pair count checked.
    """
    for x in range(n):
        d, u = gcd(x, n), x // gcd(x, n)
        for y in range(n):
            e, v = gcd(y, n), y // gcd(y, n)
            m = gcd(d * e, n)
            w = (d * e) // m
            q = n // m
            # The four lemmas the proof rests on, each asserted rather than argued.
            assert m % d == 0 and m % e == 0, (n, x, y, "d|m or e|m fails")   # so q | n/d, q | n/e
            assert gcd(u, q) == 1 and gcd(v, q) == 1, (n, x, y, "u,v not units mod n/m")
            assert gcd(w, q) == 1, (n, x, y, "w not coprime to n/m")
            assert (m * (w * u * v)) % n == m * ((w * u * v) % q), (n, x, y, "m-scaling")
            assert (x * y) % n == (m * ((w * (u * v)) % q)) % n, (n, x, y)
    return n * n


def torsor(n):
    """Setup Proposition (prop:torsor), by exhaustion: for every d | n the map
    u -> d*u mod n is a bijection (Z/(n/d)Z)* -> J_d. Returns the divisor count checked."""
    for d in divisors(n):
        q = n // d
        local = [u for u in range(q) if gcd(u, q) == 1]      # q = 1 gives [0], phi(1) = 1
        image = [(d * u) % n for u in local]
        assert len(set(image)) == len(local) == totient(q), (n, d, "not injective / |J_d| != phi(n/d)")
        assert sorted(image) == j_class(n, d), (n, d, "not onto J_d")
    return len(divisors(n))


def _selfcheck():
    # PLAN.md section 1.3 table, regenerated -- must match by computation, not trust.
    expected = {  # n: (field, phi, cyclic, omega, non-regular J-classes)
        113: (True,  112, True,  1, 0),
        121: (False, 110, True,  1, 1),
        125: (False, 100, True,  1, 2),
        119: (False,  96, False, 2, 0),
        120: (False,  32, False, 3, 8),
    }
    for n, (field, phi, cyc, om, nreg) in expected.items():
        d = describe(n)
        assert d["is_field"] == field, (n, "field", d["is_field"])
        assert d["phi"] == phi, (n, "phi", d["phi"])
        assert d["cyclic"] == cyc, (n, "cyclic", d["cyclic"])
        assert d["omega"] == om, (n, "omega", d["omega"])
        assert d["n_nonregular"] == nreg, (n, "nonregular", d["n_nonregular"])

    # 2607.07066 Thm D.17: n square-free => every J-class regular. Their own moduli.
    for n in REPLICATION + [113, 119]:
        d = describe(n)
        assert d["squarefree"] and d["n_nonregular"] == 0, (n, "Thm D.17 violated", d)

    # A field has exactly two J-classes: the units, and {0}.
    assert describe(113)["n_jclasses"] == 2

    # n=121: the minimal non-regular case -- one class, all 10 elements nilpotent.
    j11 = [c for c in describe(121)["jclasses"] if c["d"] == 11][0]
    assert j11["size"] == 10 and len(j11["nilpotents"]) == 10 and not j11["regular"]

    # n=120 separates "non-regular" from "nilpotent": classes with neither inverse nor nilpotents.
    nonreg_no_nilp = [c["d"] for c in describe(120)["jclasses"]
                      if not c["regular"] and not c["nilpotents"]]
    assert nonreg_no_nilp == [2, 4, 6, 10, 12, 20], nonreg_no_nilp

    # The Setup proposition and theorem, by exhaustion over every n <= 200 -- the range the
    # paper's composition check (J_d . J_e = J_gcd(de,n), test_paper_numbers) already covers,
    # and a superset of all 23 trained moduli.
    for n in range(2, 201):
        torsor(n)
        assert stratum_identity(n) == n * n

    # Cyclic <=> a primitive root exists and generates the whole unit group.
    for n in PRIMARY:
        d = describe(n)
        if d["cyclic"]:
            assert n_order(d["primitive_root"], n) == d["phi"]

    print("algebra.py self-check: PASS")


if __name__ == "__main__":
    _selfcheck()
    hdr = f"{'n':>5} {'factorization':>14} {'field':>6} {'phi':>5} {'cyc':>5} {'w':>2} {'sqfree':>7} {'J':>3} {'non-reg':>8} {'nilp':>5}"
    print(hdr); print("-" * len(hdr))
    for n in PRIMARY + REPLICATION:
        d = describe(n)
        fac = "*".join(f"{p}^{e}" if e > 1 else str(p) for p, e in sorted(d["factorization"].items()))
        print(f"{n:>5} {fac:>14} {str(d['is_field']):>6} {d['phi']:>5} {str(d['cyclic']):>5} "
              f"{d['omega']:>2} {str(d['squarefree']):>7} {d['n_jclasses']:>3} {d['n_nonregular']:>8} {d['n_nilpotent']:>5}")


def j_multiplication_table(n):
    """How J-classes compose: J_d * J_e -> J_(gcd(de, n)) -- closed, by construction.

    2607.07066 Thm 3.4: a REGULAR class is a group, J_d = (Z/(n/d)Z)^x, so computation
    inside it is ordinary GCR. Non-regular classes have no idempotent and no local
    inverse, so that account fails -- their stated open problem. This table says what
    happens instead: products leave the class and descend toward 0.
    """
    from sympy import divisors
    ds = divisors(n)
    return {(d, e): gcd(d * e, n) for d in ds for e in ds}


def nilpotency_depth(n):
    """Least k with (rad n)^k = 0 in Z/nZ: how many multiplications collapse a
    zero-divisor to zero. 1 for a field, 2 for p^2, 3 for p^3."""
    from sympy import factorint
    rad = 1
    for p in factorint(n):
        rad *= p
    k, x = 1, rad
    while x % n != 0:
        x *= rad
        k += 1
    return k
