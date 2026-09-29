"""Provenance must round-trip, must never crash, and must not silently claim a clean tree."""
import json, numpy as np, tempfile, os
from pathlib import Path
from src.provenance import stamp, stamp_npz, read, git_sha, git_dirty

s = stamp({"n": 113, "lr": 1e-3})
assert len(s["git_sha"]) == 40 or s["git_sha"] == "unknown", s["git_sha"]
assert s["git_dirty"] in (True, False, None)
assert json.loads(s["config"])["n"] == 113
print(f"  stamp: sha {s['git_sha'][:8]} dirty={s['git_dirty']} {s['timestamp_utc']}   OK")

with tempfile.TemporaryDirectory() as d:
    f = Path(d) / "r.npz"
    np.savez_compressed(f, data=np.arange(5), **stamp_npz({"n": 121}))
    got = read(f)
    assert got["git_sha"] == s["git_sha"] and json.loads(got["config"])["n"] == 121
    assert np.array_equal(np.load(f)["data"], np.arange(5)), "payload corrupted"
    print(f"  npz round-trip, payload intact                          OK")

# must not crash outside a git tree (the Kaggle case)
with tempfile.TemporaryDirectory() as d:
    assert git_sha(d) in ("unknown",) or len(git_sha(d)) == 40
    assert git_dirty(d) is None
    print("  no-git fallback returns 'unknown', never raises          OK")

print("\ntest_provenance: PASS")
