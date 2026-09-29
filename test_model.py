"""Cross-check the from-scratch transformer against a PyTorch reference.

Same weights in both; the forward passes must agree to float precision, and the gradient
of the loss w.r.t. every parameter must agree too. This is the week-3 gate: if these match,
the engine is trustworthy enough to run the science on.
"""
import numpy as np, torch, torch.nn.functional as F
from src.model.transformer import Transformer
from src.autograd.nn import cross_entropy

N, D, H, DH, DM = 17, 32, 4, 8, 64          # small enough to be fast, big enough to be real
rng = np.random.default_rng(0)
m = Transformer(N + 1, N, d_model=D, n_heads=H, d_head=DH, d_mlp=DM, seed=0)

a, b = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
toks = np.stack([a.ravel(), b.ravel(), np.full(N * N, N)], 1)
y = (a.ravel() * b.ravel()) % N


def torch_forward(P, toks):
    """Reference implementation of the same architecture, in PyTorch."""
    t = {k: torch.tensor(v, dtype=torch.float64, requires_grad=True) for k, v in P.items()}
    tk = torch.as_tensor(toks)
    B, T = tk.shape
    x = t["W_E"][tk] + t["W_pos"][:T]
    sh = lambda W: (x @ W).reshape(B, T, H, DH).permute(0, 2, 1, 3)
    q, k, v = sh(t["W_Q"]), sh(t["W_K"]), sh(t["W_V"])
    mask = torch.zeros(T, T, dtype=torch.float64)
    mask[torch.triu_indices(T, T, 1)[0], torch.triu_indices(T, T, 1)[1]] = -1e9
    att = torch.softmax(q @ k.transpose(-1, -2) / np.sqrt(DH) + mask, dim=-1)
    z = (att @ v).permute(0, 2, 1, 3).reshape(B, T, H * DH)
    x = x + z @ t["W_O"]
    x = x + torch.relu(x @ t["W_in"]) @ t["W_out"]
    return t, x[:, -1] @ t["W_U"]


names = ["W_E", "W_pos", "W_Q", "W_K", "W_V", "W_O", "W_in", "W_out", "W_U"]
P = {k: getattr(m, k).data.copy() for k in names}

t, tl = torch_forward(P, toks)
ours = m(toks)
fwd_err = np.abs(ours.data - tl.detach().numpy()).max()
print(f"forward max |ours - torch| = {fwd_err:.3e}   {'OK' if fwd_err < 1e-9 else 'FAIL'}")
assert fwd_err < 1e-9

loss_t = F.cross_entropy(tl, torch.as_tensor(y))
loss_t.backward()
loss_o = cross_entropy(ours, y)
loss_o.backward()
print(f"loss  ours={float(loss_o.data):.10f}  torch={loss_t.item():.10f}   "
      f"{'OK' if abs(float(loss_o.data) - loss_t.item()) < 1e-10 else 'FAIL'}")

print("gradient agreement, per parameter:")
worst = 0
for k in names:
    go, gt = getattr(m, k).grad, t[k].grad.numpy()
    err = np.abs(go - gt).max() / max(np.abs(gt).max(), 1e-12)
    worst = max(worst, err)
    print(f"  {k:<7} shape {str(go.shape):<12} rel err {err:.3e}  {'OK' if err < 1e-8 else 'FAIL'}")
assert worst < 1e-8, worst
print(f"\ntest_model: PASS  (worst relative gradient error {worst:.2e})")
