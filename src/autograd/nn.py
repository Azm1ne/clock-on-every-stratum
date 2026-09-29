"""Layers, losses and optimizer on the from-scratch engine.

Softmax is written max-subtracted because we are more exposed to Softmax Collapse than
people using PyTorch internals (H1.4): our scout already saw max logits of ~200 by 25k
steps. `stablemax` is here so H1.4 can be tested, not because it is the default.
"""
import numpy as np
from .engine import Tensor


def softmax(x, axis=-1):
    """Max-subtracted softmax. The shift is a constant w.r.t. the gradient, so it costs
    nothing and buys the whole dynamic range back."""
    z = x - x.max(axis=axis, keepdims=True)
    e = z.exp()
    return e / e.sum(axis=axis, keepdims=True)


def log_softmax(x, axis=-1):
    """log(softmax) computed without ever forming softmax -- avoids log(0) at large logits."""
    z = x - x.max(axis=axis, keepdims=True)
    return z - z.exp().sum(axis=axis, keepdims=True).log()


def cross_entropy(logits, targets):
    """Mean NLL. `targets` is an integer array of class indices."""
    ls = log_softmax(logits, axis=-1)
    n = len(targets)
    picked = ls[np.arange(n), np.asarray(targets)]
    return -(picked.sum() * (1.0 / n))


def stablemax(x, axis=-1):
    """Prieto et al. 2025's softmax replacement: s(x) = x+1 for x>=0, else 1/(1-x).
    A polynomial tail instead of an exponential one, so logit growth cannot saturate the
    normalizer. For H1.4 only -- softmax stays the default.

    Each branch is fed a masked input so its denominator stays >= 1. Without that, the
    inactive branch still evaluates 1/(1-x) at large positive x and pushes inf into the
    gradient even though the forward pass masks it out.
    """
    m = Tensor((x.data >= 0).astype(float))          # constant: piecewise, not differentiable
    pos = (x * m) + 1.0                              # >= 1 everywhere
    neg = (1.0 - (x * (1.0 - m))) ** -1              # denominator >= 1 everywhere
    num = pos * m + neg * (1.0 - m)
    return num / num.sum(axis=axis, keepdims=True)


class AdamW:
    """Decoupled weight decay (Loshchilov & Hutter). Decay is applied to the parameter,
    not folded into the gradient -- that distinction is what drives grokking here."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.98), eps=1e-8, weight_decay=1.0):
        self.params = list(params)
        self.lr, self.betas, self.eps, self.wd = lr, betas, eps, weight_decay
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        self.t += 1
        b1, b2 = self.betas
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * g * g
            mh = self.m[i] / (1 - b1 ** self.t)
            vh = self.v[i] / (1 - b2 ** self.t)
            p.data -= self.lr * (mh / (np.sqrt(vh) + self.eps) + self.wd * p.data)


def stablemax_cross_entropy(logits, targets):
    """Cross-entropy through stablemax instead of softmax (Prieto et al. 2025).

    Softmax's exponential tail drives p_correct to exactly 1.0 once logits grow, and the
    gradient vanishes with it -- the no-gradient regime we measured on this engine
    (|grad| 4e-1 -> 9e-7 while p_correct -> 1.000000). Stablemax's polynomial tail keeps
    a usable gradient at the same logit scale.
    """
    p = stablemax(logits, axis=-1)
    n = len(targets)
    return -(p[np.arange(n), np.asarray(targets)].log().sum() * (1.0 / n))
