"""Reverse-mode autograd over numpy arrays. Written from scratch -- this is a thesis
deliverable, not a dependency.

Design: one Tensor type wrapping an ndarray, each op recording a local backward closure.
Broadcasting is where tensor autograd bugs hide, so every op routes its gradient through
`_unbroadcast`, and every op has a finite-difference test (test_autograd.py).
"""
import os
import numpy as np


_GRAD_ENABLED = True

# TRAINING PRECISION IS AN EXPERIMENTAL VARIABLE, NOT AN IMPLEMENTATION DETAIL (C30/O17).
# The engine trains in float64; the Kaggle/PyTorch kernels train in float32, and at
# byte-identical hyperparameters the engine reads Gini_mult 0.75-0.84 where torch reads
# 0.54-0.58 (N7's E5, 7/7 grokked pairs). `ENGINE_DTYPE=float32` runs the other arm of
# that A/B. **float64 stays the default** -- every engine number on record is float64, and
# flipping the default would silently re-base them.
DTYPE = np.dtype(os.environ.get("ENGINE_DTYPE", "float64"))


class no_grad:
    """Suppress graph construction. Evaluation and checkpointing do a forward pass over
    every input pair -- several times the training batch -- and the graph they build is
    never used. Building it anyway is what pushed this process to 10 GB RSS."""

    def __enter__(self):
        global _GRAD_ENABLED
        self._prev = _GRAD_ENABLED
        _GRAD_ENABLED = False
        return self

    def __exit__(self, *exc):
        global _GRAD_ENABLED
        _GRAD_ENABLED = self._prev
        return False


def _noop():
    return None


def _unbroadcast(grad, shape):
    """Sum a gradient back down to `shape`, undoing numpy broadcasting.

    Broadcasting prepends axes and stretches size-1 axes; the adjoint of each is a sum.
    Getting this wrong produces gradients that are silently the right shape but the
    wrong value, which is why it is factored out and tested directly.
    """
    while grad.ndim > len(shape):
        grad = grad.sum(0)
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(i, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    __slots__ = ("data", "grad", "_bw", "_prev", "requires_grad")

    def __init__(self, data, requires_grad=False, _prev=()):
        self.data = np.asarray(data, dtype=DTYPE)
        self.requires_grad = requires_grad
        self.grad = None
        self._bw = _noop
        self._prev = _prev

    # A closure assigned here captures `out`, so the tensor self-references and
    # REFCOUNTING CANNOT FREE IT -- only the generational collector can, and each such
    # tensor pins a full-size activation array. Training escapes that because backward()
    # clears the edges; the eval path never calls backward(), so every
    # `with no_grad(): m(xte)` leaked its whole forward pass (~176 MB at n=113, once per
    # 100 steps -> ~1.76 MB/step, which OOM-killed N7 at 1h12m on 2026-09-12).
    #
    # _child() already decides correctly whether a tensor needs a graph, but each operator
    # then assigned `out._backward = bw` unconditionally and overwrote that decision. The
    # guard lives HERE, at the one point every one of those assignments routes through,
    # rather than in twelve call sites where the thirteenth would forget it.
    @property
    def _backward(self):
        return self._bw

    @_backward.setter
    def _backward(self, fn):
        if self.requires_grad:
            self._bw = fn

    # -- construction -------------------------------------------------------
    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, requires_grad={self.requires_grad})"

    def _child(self, data, parents, backward):
        needs = _GRAD_ENABLED and any(p.requires_grad for p in parents)
        out = Tensor(data, requires_grad=needs, _prev=tuple(parents) if needs else ())
        if needs:
            out._backward = backward
        return out

    def _accum(self, g):
        if not self.requires_grad:
            return
        self.grad = g if self.grad is None else self.grad + g

    # -- ops ----------------------------------------------------------------
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = self._child(self.data + other.data, (self, other), lambda: None)

        def bw():
            self._accum(_unbroadcast(out.grad, self.data.shape))
            other._accum(_unbroadcast(out.grad, other.data.shape))
        out._backward = bw
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = self._child(self.data * other.data, (self, other), lambda: None)

        def bw():
            self._accum(_unbroadcast(out.grad * other.data, self.data.shape))
            other._accum(_unbroadcast(out.grad * self.data, other.data.shape))
        out._backward = bw
        return out

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "only scalar powers"
        out = self._child(self.data ** k, (self,), lambda: None)
        out._backward = lambda: self._accum(out.grad * k * self.data ** (k - 1))
        return out

    def __matmul__(self, other):
        out = self._child(self.data @ other.data, (self, other), lambda: None)

        def bw():
            g, a, b = out.grad, self.data, other.data
            # grad wrt a: (..., n, m) @ (m, k) -> (..., n, k)
            self._accum(_unbroadcast(g @ np.swapaxes(b, -1, -2), a.shape))
            # grad wrt b. The naive form swapaxes(a) @ g materialises one (k, m) gradient
            # PER batch element and only then sums -- for (3830, 3, 128) @ (128, 512) that
            # is a 2 GB temporary. When b is unbatched the sum over leading axes is just a
            # contraction, so fold the batch into the row axis and let BLAS do it in one gemm.
            if b.ndim == 2 and a.ndim > 2:
                gb = a.reshape(-1, a.shape[-1]).T @ g.reshape(-1, g.shape[-1])
                other._accum(gb.reshape(b.shape))
            else:
                other._accum(_unbroadcast(np.swapaxes(a, -1, -2) @ g, b.shape))
        out._backward = bw
        return out

    def sum(self, axis=None, keepdims=False):
        out = self._child(self.data.sum(axis=axis, keepdims=keepdims), (self,), lambda: None)

        def bw():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self._accum(np.broadcast_to(g, self.data.shape).copy())
        out._backward = bw
        return out

    def max(self, axis=None, keepdims=False):
        m = self.data.max(axis=axis, keepdims=True)
        out = self._child(m if keepdims else m.squeeze(axis), (self,), lambda: None)

        def bw():
            g = out.grad if keepdims or axis is None else np.expand_dims(out.grad, axis)
            mask = (self.data == m)
            # .astype(g.dtype): mask.sum() is int64, and float32 / int64 promotes back to
            # float64 (NEP 50), so under ENGINE_DTYPE=float32 this one line silently put a
            # float64 gradient into both softmaxes -- W_Q/W_K/W_in/W_out/W_U all came back
            # float64. Inert at the float64 default; load-bearing for the C30 A/B.
            self._accum(mask * g / mask.sum(axis=axis, keepdims=True).astype(g.dtype))
        out._backward = bw
        return out

    def relu(self):
        out = self._child(np.maximum(self.data, 0), (self,), lambda: None)
        out._backward = lambda: self._accum(out.grad * (self.data > 0))
        return out

    def exp(self):
        out = self._child(np.exp(self.data), (self,), lambda: None)
        out._backward = lambda: self._accum(out.grad * out.data)
        return out

    def log(self):
        out = self._child(np.log(self.data), (self,), lambda: None)
        out._backward = lambda: self._accum(out.grad / self.data)
        return out

    def reshape(self, *shape):
        out = self._child(self.data.reshape(*shape), (self,), lambda: None)
        out._backward = lambda: self._accum(out.grad.reshape(self.data.shape))
        return out

    def transpose(self, *axes):
        axes = axes or tuple(range(self.data.ndim))[::-1]
        out = self._child(self.data.transpose(axes), (self,), lambda: None)
        out._backward = lambda: self._accum(out.grad.transpose(np.argsort(axes)))
        return out

    def __getitem__(self, idx):
        out = self._child(self.data[idx], (self,), lambda: None)

        def bw():
            g = np.zeros_like(self.data)
            np.add.at(g, idx, out.grad)      # `at` so repeated indices accumulate
            self._accum(g)
        out._backward = bw
        return out

    # -- sugar --------------------------------------------------------------
    def __neg__(self):        return self * -1.0
    def __sub__(self, o):     return self + (-(o if isinstance(o, Tensor) else Tensor(o)))
    def __rsub__(self, o):    return (Tensor(o) if not isinstance(o, Tensor) else o) + (-self)
    def __truediv__(self, o): return self * ((o if isinstance(o, Tensor) else Tensor(o)) ** -1)
    def __rtruediv__(self, o): return (self ** -1) * o
    __radd__ = __add__
    __rmul__ = __mul__

    def mean(self, axis=None):
        k = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis) * (1.0 / k)

    # -- backprop -----------------------------------------------------------
    def backward(self):
        assert self.data.size == 1, "backward() only from a scalar"
        topo, seen = [], set()

        def build(t):                       # iterative: recursion blows up on deep graphs
            stack = [(t, False)]
            while stack:
                node, done = stack.pop()
                if done:
                    topo.append(node)
                    continue
                if id(node) in seen:
                    continue
                seen.add(id(node))
                stack.append((node, True))
                for p in node._prev:
                    if id(p) not in seen:
                        stack.append((p, False))
        build(self)
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            # A node reachable from the output but with no gradient of its own (a
            # constant, e.g. a piecewise mask) still carries a backward closure; running
            # it would dereference grad=None. Nothing to propagate, so skip.
            if node.grad is not None:
                node._backward()

        # Break the graph. Each node's _backward closure captures that same node, so the
        # graph is a mass of reference cycles: nothing is freed by refcounting and RSS
        # grows until the generational collector happens to run. With ~12 MB of float64
        # per intermediate that reached 12 GB in 1000 steps. Clearing the edges after the
        # pass makes every intermediate refcount-collectable immediately.
        # Consequence: backward() may be called once per graph (we never call it twice).
        for node in topo:
            node._prev = ()
            node._bw = _noop          # direct: this must clear unconditionally, and the
                                      # _backward setter ignores non-requires_grad nodes
