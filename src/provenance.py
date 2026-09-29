"""Stamp every saved result with the code version that produced it.

Four months from now the question "which code made this number?" has to be answerable
from the file itself, not from memory or from a notebook entry that may have drifted.

Degrades gracefully: on Kaggle there is no git checkout, so `git_sha` records whatever the
kernel was told (via KERNEL_GIT_SHA) or "unknown" -- never a crash, and never a lie.
"""
import json, os, subprocess, sys, time
from pathlib import Path


def git_sha(repo=None):
    repo = Path(repo or Path(__file__).resolve().parents[1])
    if not (repo / ".git").exists():
        return os.environ.get("KERNEL_GIT_SHA", "unknown")
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def git_dirty(repo=None):
    """True if the tree had uncommitted changes -- the result is then NOT reproducible
    from the recorded SHA alone, and the stamp says so."""
    repo = Path(repo or Path(__file__).resolve().parents[1])
    if not (repo / ".git").exists():
        return None
    try:
        r = subprocess.run(["git", "status", "--porcelain"], cwd=repo,
                           capture_output=True, text=True, timeout=10)
        return bool(r.stdout.strip())
    except Exception:
        return None


def stamp(config=None, **extra):
    """Provenance record to embed alongside any saved result."""
    return dict(
        git_sha=git_sha(),
        git_dirty=git_dirty(),
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        python=sys.version.split()[0],
        argv=" ".join(sys.argv),
        config=json.dumps(config, sort_keys=True, default=str) if config else "",
        **extra,
    )


def stamp_npz(config=None, **extra):
    """np.savez keys must be array-like, so serialise the stamp to a single JSON string.
    Read back with: json.loads(str(np.load(f)['provenance']))"""
    import numpy as np
    return {"provenance": np.array(json.dumps(stamp(config, **extra)))}


def read(path):
    """Recover the provenance record from a saved .npz or .json."""
    import numpy as np
    p = Path(path)
    if p.suffix == ".npz":
        z = np.load(p, allow_pickle=False)
        return json.loads(str(z["provenance"])) if "provenance" in z else None
    d = json.loads(p.read_text())
    # A kernel's *_summary.json is a LIST of run records and carries no stamp. Return
    # None, as for any unstamped artifact, rather than raising AttributeError -- a
    # reader that crashes on 11 of the project's own files cannot be used to audit them.
    return d.get("provenance") if isinstance(d, dict) else None
