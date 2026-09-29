"""Regression test: training must not grow RSS."""
import os, sys; sys.path.insert(0,'.')
import numpy as np, resource
from src.model.transformer import Transformer
from src.train.loop import modular_data
from src.autograd.nn import cross_entropy, AdamW
N=113
xtr,ytr,_,_=modular_data(N,"add",train_frac=0.3,seed=0)
m=Transformer(N+1,N,d_model=128,n_heads=4,d_head=32,d_mlp=512,seed=0)
opt=AdamW(m.parameters(),lr=1e-3,weight_decay=1.0)
rss=lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
for i in range(31):
    opt.zero_grad(); cross_entropy(m(xtr),ytr).backward(); opt.step()
    if i in (5,10,20,30): print(f"  step {i:3d}  peak RSS {rss():.0f} MB")
r0=rss()
for i in range(30):
    opt.zero_grad(); cross_entropy(m(xtr),ytr).backward(); opt.step()
r1=rss()
print(f"  growth over 30 more steps: {r1-r0:.0f} MB  {'OK' if r1-r0 < 200 else 'LEAK'}")
assert r1-r0 < 200, f"memory leak: +{r1-r0:.0f} MB"
print("memtest: PASS")
