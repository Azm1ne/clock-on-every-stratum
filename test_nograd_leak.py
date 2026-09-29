"""Regression test: a no_grad() forward pass must not leak its tensors.

Why this exists: a long run's footprint grew ~1.76 MB/step. Every operator assigned
`out._backward = bw` after `_child()` had already decided the tensor needs no graph. `bw`
closes over `out`, so the tensor referenced itself and refcounting could not free it; only
the generational collector could, and each such tensor pinned a full-size activation
array. `_backward` is now a guarded property.

Training escapes this because backward() clears `_prev`/`_backward`. The eval path never
calls backward(), so the ~176 MB of every `with no_grad(): m(xte)` leaked, once per
LOG=100 steps.

test_memory.py cannot catch this: it runs 30 steps with no eval, where the leak is ~0 MB
against a 200 MB threshold. The horizon, not the threshold, is what makes it blind.
"""
import gc
import sys

sys.path.insert(0, '.')
import numpy as np

from src.autograd.engine import Tensor, no_grad, _noop
from src.model.transformer import Transformer


def test_nograd_backward_is_noop():
    """Under no_grad every op must leave _backward as _noop -- no closure, no cycle."""
    a = Tensor(np.ones((4, 4)))
    b = Tensor(np.ones((4, 4)))
    with no_grad():
        ops = {
            "mul": a * b, "add": a + b, "matmul": a @ b, "pow": a ** 2,
            "relu": a.relu(), "exp": a.exp(), "log": a.log(),
            "reshape": a.reshape(16), "transpose": a.transpose(1, 0),
        }
    bad = {k: v._backward for k, v in ops.items() if v._backward is not _noop}
    assert not bad, f"ops leaked a backward closure under no_grad: {sorted(bad)}"
    for k, v in ops.items():
        assert v._prev == (), f"{k}: _prev should be empty under no_grad, got {v._prev}"


def test_nograd_frees_by_refcount_alone():
    """With the cyclic collector OFF, no_grad tensors must still be freed.

    This is the decisive check: a self-referential tensor is invisible to refcounting, so
    if this grows, the cycle is back.
    """
    a = Tensor(np.ones((32, 32)))
    b = Tensor(np.ones((32, 32)))
    with no_grad():          # warm up any one-time allocations
        _ = a @ b
    gc.collect()
    gc.disable()
    try:
        before = len(gc.get_objects())
        for _ in range(200):
            with no_grad():
                t = (a @ b).relu() * b
        del t
        grown = len(gc.get_objects()) - before
    finally:
        gc.enable()
    assert grown < 50, (
        f"no_grad ops leaked {grown} tracked objects over 200 iterations with GC "
        f"disabled -- refcounting cannot free them, so they self-reference")


def test_eval_forward_does_not_grow_rss():
    """The real shape of the bug: repeated full-grid eval under no_grad, measuring
    CURRENT RSS (not ru_maxrss, which is a monotonic high-water mark and cannot fall).

    50 evals at this size leaked ~1.4 GB before the fix.
    """
    n = 113
    m = Transformer(n + 1, n, d_model=128, n_heads=4, d_head=32, d_mlp=512, seed=0)
    a, b = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    x_all = np.stack([a.ravel(), b.ravel(), np.full(n * n, n)], 1)[:9000]

    def rss_mb():
        for line in open('/proc/self/status'):
            if line.startswith('VmRSS'):
                return int(line.split()[1]) / 1024
        return 0.0

    with no_grad():                      # warm up: first pass allocates buffers
        m(x_all)
    gc.collect()
    base = rss_mb()
    for _ in range(50):
        with no_grad():
            m(x_all)
    grown = rss_mb() - base
    assert grown < 300, (
        f"50 no_grad evals grew RSS by {grown:.0f} MB -- the eval path is retaining "
        f"its forward pass (~176 MB/eval was the observed leak)")


if __name__ == "__main__":
    test_nograd_backward_is_noop()
    print("  no_grad leaves _backward as _noop: PASS")
    test_nograd_frees_by_refcount_alone()
    print("  no_grad frees under refcounting alone: PASS")
    test_eval_forward_does_not_grow_rss()
    print("  repeated eval does not grow RSS: PASS")
    print("nograd-leak: PASS")
