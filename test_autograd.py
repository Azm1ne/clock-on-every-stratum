"""Finite-difference tests ARE the spec for the autograd engine.

Every op's gradient is checked against a central difference. This is the one thing that
must not be trusted to inspection: a wrong broadcast-sum produces a gradient of the right
SHAPE and the wrong VALUE, which no shape assertion catches.
"""
import numpy as np
from src.autograd.engine import Tensor, _unbroadcast

RNG = np.random.default_rng(0)
EPS, TOL = 1e-6, 1e-6


def numeric_grad(f, arrays, i):
    """Central difference d f / d arrays[i], elementwise."""
    a = arrays[i]
    g = np.zeros_like(a)
    it = np.nditer(a, flags=["multi_index"])
    while not it.finished:
        k = it.multi_index
        orig = a[k]
        a[k] = orig + EPS; hi = f(*arrays)
        a[k] = orig - EPS; lo = f(*arrays)
        a[k] = orig
        g[k] = (hi - lo) / (2 * EPS)
        it.iternext()
    return g


def check(name, fn, *shapes):
    """fn maps Tensors -> scalar Tensor. Compare analytic vs numeric grad for each input."""
    arrays = [RNG.standard_normal(s) for s in shapes]
    ts = [Tensor(a.copy(), requires_grad=True) for a in arrays]
    out = fn(*ts)
    assert out.data.size == 1, f"{name}: not scalar"
    out.backward()
    for i, t in enumerate(ts):
        num = numeric_grad(lambda *a: float(fn(*[Tensor(x) for x in a]).data), arrays, i)
        err = np.abs(t.grad - num).max()
        rel = err / max(np.abs(num).max(), 1e-12)
        assert rel < 1e-4 and err < 1e-4, f"{name} arg{i}: max abs err {err:.2e}, rel {rel:.2e}"
    print(f"  {name:<34} OK   (args {[s for s in shapes]})")


def main():
    print("finite-difference gradient checks:")
    check("add",            lambda a, b: (a + b).sum(), (3, 4), (3, 4))
    check("add broadcast",  lambda a, b: (a + b).sum(), (3, 4), (4,))
    check("add bcast keepdim", lambda a, b: (a + b).sum(), (3, 4), (1, 4))
    check("mul",            lambda a, b: (a * b).sum(), (3, 4), (3, 4))
    check("mul broadcast",  lambda a, b: (a * b).sum(), (2, 3, 4), (4,))
    check("sub",            lambda a, b: (a - b).sum(), (3, 4), (3, 4))
    check("div",            lambda a, b: (a / (b * b + 3.0)).sum(), (3, 4), (3, 4))
    check("pow",            lambda a: (a ** 3).sum(), (3, 4))
    check("matmul",         lambda a, b: (a @ b).sum(), (3, 4), (4, 5))
    check("matmul batched", lambda a, b: (a @ b).sum(), (2, 3, 4), (2, 4, 5))
    check("matmul bcast",   lambda a, b: (a @ b).sum(), (2, 3, 4), (4, 5))
    check("relu",           lambda a: (a.relu()).sum(), (5, 6))
    check("exp",            lambda a: (a.exp()).sum(), (3, 4))
    check("log",            lambda a: ((a * a + 1.0).log()).sum(), (3, 4))
    check("sum axis",       lambda a: (a.sum(axis=1) ** 2).sum(), (3, 4))
    check("sum keepdims",   lambda a: (a.sum(axis=1, keepdims=True) ** 2).sum(), (3, 4))
    check("mean",           lambda a: a.mean(), (3, 4))
    check("mean axis",      lambda a: (a.mean(axis=0) ** 2).sum(), (3, 4))
    check("reshape",        lambda a: (a.reshape(6, 2) ** 2).sum(), (3, 4))
    check("transpose",      lambda a: (a.transpose(1, 0) ** 2).sum(), (3, 4))
    check("getitem",        lambda a: (a[np.array([0, 2, 2])] ** 2).sum(), (4, 3))
    check("chain",          lambda a, b: (((a @ b).relu() + a.sum()) ** 2).sum(), (3, 4), (4, 3))
    # max: skip finite-diff at ties (subgradient); random inputs make ties measure-zero
    check("max axis",       lambda a: (a.max(axis=1) ** 2).sum(), (4, 5))

    print("\n_unbroadcast adjoint identity  <g, x*1> == <unbroadcast(g), x>:")
    for gs, xs in [((3, 4), (4,)), ((2, 3, 4), (1, 4)), ((5,), (5,)), ((2, 3), (2, 1))]:
        g = RNG.standard_normal(gs); x = RNG.standard_normal(xs)
        lhs = (g * np.broadcast_to(x, gs)).sum()
        rhs = (_unbroadcast(g, xs) * x).sum()
        assert abs(lhs - rhs) < 1e-10, (gs, xs, lhs, rhs)
        print(f"  grad{gs} -> {xs}  OK")

    print("\naccumulation: a node used twice gets both gradients")
    a = Tensor(RNG.standard_normal((3, 3)), requires_grad=True)
    (a * a).sum().backward()
    assert np.abs(a.grad - 2 * a.data).max() < 1e-12, "diamond accumulation wrong"
    print("  a*a -> grad == 2a   OK")

    test_no_grad()
    test_dtype_flag()
    print("\ntest_autograd: PASS")


def test_dtype_flag():
    """ENGINE_DTYPE must reach EVERY array, not just the parameters.

    C30/O17 turns training precision into an experimental variable, so a float32 arm that
    is float32 in the weights and float64 in the gradients is not the arm we pre-registered
    -- and it is invisible from the loss curve. `max`'s backward divided by an int64 tie
    count and did exactly that to both softmaxes. Run in a subprocess: DTYPE is read at
    import, so it cannot be flipped in-process."""
    import subprocess, sys, os
    src = (
        "import numpy as np;"
        "from src.model.transformer import Transformer;"
        "from src.autograd.nn import cross_entropy, AdamW;"
        "m=Transformer(18,17,d_model=16,n_heads=2,d_head=8,d_mlp=32,seed=0);"
        "x=np.stack([np.arange(9)%17,np.arange(9)%17,np.full(9,17)],1);y=(np.arange(9)**2)%17;"
        "o=AdamW(m.parameters(),lr=1e-3,betas=(0.9,0.98),weight_decay=1.0);"
        "loss=cross_entropy(m(x),y);o.zero_grad();loss.backward();o.step();"
        "d={p.data.dtype for p in m.parameters()}|{p.grad.dtype for p in m.parameters()}"
        "|{a.dtype for a in o.m}|{a.dtype for a in o.v}|{loss.data.dtype};"
        "print(sorted(str(x) for x in d))"
    )
    for want in ("float64", "float32"):
        env = dict(os.environ, PYTHONPATH=".")
        env.pop("ENGINE_DTYPE", None)
        if want != "float64":
            env["ENGINE_DTYPE"] = want
        got = subprocess.run([sys.executable, "-c", src], env=env, capture_output=True,
                             text=True, check=True).stdout.strip()
        assert got == f"['{want}']", f"ENGINE_DTYPE={want}: mixed precision {got}"
        print(f"  ENGINE_DTYPE={want:>7}: weights, grads, Adam state and loss all {want}   OK")


def test_no_grad():
    """no_grad must build no graph and leave no gradients -- and must restore state."""
    from src.autograd.engine import no_grad
    import src.autograd.engine as E
    a = Tensor(RNG.standard_normal((4, 4)), requires_grad=True)
    with no_grad():
        out = (a @ a).sum()
        assert not out.requires_grad and out._prev == (), "no_grad still built a graph"
    assert E._GRAD_ENABLED, "no_grad did not restore state"
    b = (a * a).sum()
    assert b.requires_grad, "grad mode not restored"
    b.backward()
    assert np.abs(a.grad - 2 * a.data).max() < 1e-12
    # values must be identical with and without grad
    with no_grad():
        v = (a @ a).sum().data
    assert abs(float(v) - float((a @ a).sum().data)) < 1e-12
    print("  no_grad: no graph, state restored, values identical   OK")


if __name__ == "__main__":
    main()
