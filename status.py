#!/usr/bin/env python
"""Live ground truth: git state, running jobs and results on disk. Facts only.

Run: .venv/bin/python status.py
"""
import json, subprocess, sys
from pathlib import Path

R = Path(__file__).parent
K = R / ".venv/bin/kaggle"


def sh(*a, timeout=90):
    try:
        r = subprocess.run(a, capture_output=True, text=True, timeout=timeout)
        return (r.stdout + r.stderr).strip()
    except Exception as e:
        return f"(unavailable: {e})"


print("=" * 72)
print("GROKKING THESIS — LIVE STATUS")
print("=" * 72)

print("\n[env]")
for mod in ("numpy", "sympy", "torch", "kaggle"):
    v = sh(str(R / ".venv/bin/python"), "-c", f"import {mod};print({mod}.__version__)")
    print(f"  {mod:<8} {v.splitlines()[-1] if v else '?'}")

print("\n[kaggle quota]")
q = sh(str(K), "kernels", "list", "-m", "--page-size", "1")
print("  auth:", "OK" if "ref" in q or "No kernels" in q else q[:120])
try:
    import urllib.request  # quota via CLI has no json flag; MCP is the reliable path
    print("  quota: ask via MCP get_accelerator_quota (30h/week, resets Fridays)")
except Exception:
    pass

print("\n[kernels]")
for kd in sorted((R / "kernels").glob("*/kernel-metadata.json")):
    m = json.loads(kd.read_text())
    st = sh(str(K), "kernels", "status", m["id"]).splitlines()
    st = st[0].split('"')[-2] if st and '"' in st[0] else (st[0] if st else "?")
    print(f"  {m['id']:<48} {m.get('machine_shape','(default!)'):<16} {st}")

print("\n[results on disk]")
for p in sorted((R / "results").rglob("*")):
    if p.is_file():
        print(f"  {p.relative_to(R)}  ({p.stat().st_size//1024} KB)")

print("\n[self-checks]")
for name, script in [("algebra", "src/tasks/algebra.py"), ("analysis", "test_analysis.py"),
                     ("autograd", "test_autograd.py"), ("model-vs-torch", "test_model.py"),
                     ("ablation", "test_ablation.py"), ("memory", "test_memory.py")]:
    cmd = [str(R / ".venv/bin/python"), str(R / script)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=R,
                       env={"PYTHONPATH": str(R), "PATH": "/usr/bin:/bin"})
    print(f"  {name:<15} {'PASS' if r.returncode == 0 else 'FAIL: ' + r.stderr.strip()[-200:]}")

print("\n[reference repos]")
for d in sorted((R / "reference").glob("*")):
    if d.is_dir():
        print(f"  {d.name}")

print("\n→ read STATE.md for what to do next")
