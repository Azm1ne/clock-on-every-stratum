"""1-layer decoder-only transformer on the from-scratch engine.

Architecture is identical to kernels/k01_scout/run.py so the two are comparable; only the
autograd differs. No LayerNorm, no biases (Nanda's setup). Attention is expressed with
matmul/transpose/reshape rather than einsum because those are the ops the engine has and
the ops the finite-difference suite covers.
"""
import numpy as np
from ..autograd.engine import Tensor
from ..autograd.nn import softmax


class Transformer:
    def __init__(self, n_vocab, n_out, d_model=128, n_heads=4, d_head=32, d_mlp=512,
                 n_ctx=3, seed=0):
        g = np.random.default_rng(seed)
        s = 1 / np.sqrt(d_model)
        P = lambda *sh, sc=s: Tensor(g.standard_normal(sh) * sc, requires_grad=True)
        self.d_model, self.n_heads, self.d_head = d_model, n_heads, d_head
        self.W_E = P(n_vocab, d_model)
        self.W_pos = P(n_ctx, d_model)
        self.W_Q, self.W_K, self.W_V = P(d_model, n_heads * d_head), P(d_model, n_heads * d_head), P(d_model, n_heads * d_head)
        self.W_O = P(n_heads * d_head, d_model, sc=1 / np.sqrt(n_heads * d_head))
        self.W_in = P(d_model, d_mlp)
        self.W_out = P(d_mlp, d_model, sc=1 / np.sqrt(d_mlp))
        self.W_U = P(d_model, n_out)
        # additive causal mask, built once
        m = np.zeros((n_ctx, n_ctx))
        m[np.triu_indices(n_ctx, 1)] = -1e9
        self._mask = Tensor(m)

    def parameters(self):
        return [v for k, v in vars(self).items() if isinstance(v, Tensor) and v.requires_grad]

    def _heads(self, x, W, B, T):
        """(B,T,d) @ (d, H*dh) -> (B,H,T,dh)"""
        return (x @ W).reshape(B, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3)

    def __call__(self, toks, return_acts=False):
        toks = np.asarray(toks)
        B, T = toks.shape
        x = self.W_E[toks] + self.W_pos[:T]
        q, k, v = (self._heads(x, W, B, T) for W in (self.W_Q, self.W_K, self.W_V))
        att = softmax(q @ k.transpose(0, 1, 3, 2) * (1 / np.sqrt(self.d_head)) + self._mask)
        z = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, self.n_heads * self.d_head)
        x = x + z @ self.W_O
        # The MLP is position-wise and only position -1 reaches the logits, so the MLP at
        # positions 0..T-2 is computed and discarded -- its gradient is identically zero.
        # Evaluating it only at the read-out position is exactly equivalent and ~1.8x
        # cheaper overall (the MLP is two thirds of the matmul cost). test_model.py checks
        # this against a PyTorch reference that DOES compute all positions.
        last = x[:, -1]
        pre = last @ self.W_in
        last = last + pre.relu() @ self.W_out
        logits = last @ self.W_U
        return (logits, pre, att) if return_acts else logits
