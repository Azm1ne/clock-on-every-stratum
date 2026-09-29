#!/usr/bin/env python
"""push -> poll -> pull for a Kaggle kernel directory.

Poll with backoff; on failure pull the log and print the traceback rather than
re-pushing blindly (§10.3). Usage: run_kernel.py kernels/k00_smoke [results/k00]
"""
import json, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path

SLOT_POLL = 600           # s between retries when both GPU sessions are busy
SLOT_WAIT_MAX = 9 * 3600  # give up waiting for a slot after this long
UNKNOWN_MAX = 20          # consecutive unrecognised statuses before bailing

REPO = Path(__file__).resolve().parents[1]
KAGGLE = str(REPO / ".venv/bin/kaggle")


def git_sha():
    """Exact commit the pushed code came from, '-dirty' if the tree is not clean.

    Kaggle has no git and no access to our env, so the SHA has to be BAKED INTO the
    script at push time -- `os.environ.get("KERNEL_GIT_SHA")` is always unset there and
    silently stamps "unknown". Reproducibility decree: a result whose code version is
    unknown is not usable.
    """
    r = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                       capture_output=True, text=True)
    sha = r.stdout.strip() or "unknown"
    d = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"],
                       capture_output=True, text=True).stdout.strip()
    return sha + ("-dirty" if d else "")


def kernel_id(meta, owner=None):
    """`<owner>/<slug>`, with the owner overridable by $KAGGLE_OWNER.

    A Kaggle kernel id is owner-scoped, so running the same kernel from a second account
    means rewriting the owner half of the id in five metadata files by hand -- which is
    how a push lands on the wrong account or gets rejected mid-launch. The SLUG never
    changes, which matters: a kernel's title must slugify to its slug or Kaggle answers
    409 Conflict.
    """
    slug = meta["id"].split("/", 1)[1]
    return f"{owner or meta['id'].split('/', 1)[0]}/{slug}"


def authenticated_user():
    """Whoever the CLI would push as right now, or None if that cannot be determined."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        return api.config_values.get("username")
    except Exception as e:
        print(f"  WARNING: could not determine the authenticated Kaggle user ({e})")
        return None


def staged_copy(kdir, sha, owner=None):
    """Copy the kernel dir to a temp dir with the placeholders and the id substituted.

    Pushing a temp copy keeps the tracked source clean -- the placeholder stays in git,
    the real SHA only ever exists in what Kaggle receives and in what it saves.
    """
    tmp = Path(tempfile.mkdtemp(prefix="kernel_push_"))
    dst = tmp / kdir.name
    shutil.copytree(kdir, dst)
    mpath = dst / "kernel-metadata.json"
    meta = json.loads(mpath.read_text())
    if owner:
        meta["id"] = kernel_id(meta, owner)
        mpath.write_text(json.dumps(meta, indent=2) + "\n")
    code = dst / meta["code_file"]
    src = code.read_text()
    if "__GIT_SHA__" not in src:
        print(f"  WARNING: {code.name} has no __GIT_SHA__ placeholder -- "
              f"results will not carry a code version")
    # __KAGGLE_OWNER__ is optional: a kernel that stamps it records WHICH ACCOUNT ran it,
    # which the reproducibility decree needs once results come from more than one.
    for ph, val in (("__GIT_SHA__", sha), ("__KAGGLE_OWNER__", meta["id"].split("/")[0])):
        src = src.replace(ph, val)
    code.write_text(src)
    return dst


def sh(*args):
    r = subprocess.run([KAGGLE, *args], capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()


def main(kdir, outdir=None, attach=False):
    kdir = Path(kdir)
    meta = json.loads((kdir / "kernel-metadata.json").read_text())
    owner = os.environ.get("KAGGLE_OWNER") or None
    slug = kernel_id(meta, owner)
    outdir = Path(outdir or f"results/{kdir.name}")
    outdir.mkdir(parents=True, exist_ok=True)

    if attach:
        print(f"attach {slug} (no push)")
    else:
        sha = git_sha()
        # Pushing as the wrong account is the failure this guard exists for: the id is
        # owner-scoped, so a mismatch is either a rejected push or a kernel written to an
        # account whose quota and results nobody is tracking.
        me = authenticated_user()
        if me and me != slug.split("/")[0]:
            sys.exit(f"refusing to push: kernel id is '{slug}' but the authenticated "
                     f"Kaggle user is '{me}'.\n"
                     f"  run as that account:  KAGGLE_OWNER={me} {' '.join(sys.argv)}\n"
                     f"  or switch credentials: KAGGLE_API_TOKEN=$HOME/.kaggle/<file>")
        print(f"push {slug}  @ {sha}  as {me or 'unknown account'}")
        if sha.endswith("-dirty"):
            print("  WARNING: tree is dirty; the stamped SHA does not fully identify "
                  "the code that ran. Commit before launching a real run.")
        # Kaggle allows only 2 concurrent batch GPU sessions. A push that hits the cap
        # returns rc 0 with the refusal in the BODY -- it silently does not start. Wait
        # for a slot instead of pretending we launched.
        waited_slot = 0
        while True:
            rc, o = sh("kernels", "push", "-p", str(staged_copy(kdir, sha, owner)))
            print(o)
            if rc != 0:
                sys.exit(f"push failed: {o}")
            if "session count" not in o.lower() and "maximum batch" not in o.lower():
                break
            if waited_slot >= SLOT_WAIT_MAX:
                sys.exit(f"no GPU slot after {waited_slot//3600}h -- giving up")
            print(f"  GPU slots full; retrying in {SLOT_POLL//60} min "
                  f"(waited {waited_slot//60} min). Not touching the running kernels.")
            time.sleep(SLOT_POLL)
            waited_slot += SLOT_POLL

    delay, waited, unknown = 15, 0, 0
    while True:
        time.sleep(delay)
        waited += delay
        rc, o = sh("kernels", "status", slug)
        status = o.lower()
        print(f"[{waited//60}m{waited%60:02d}s] {o.splitlines()[0] if o else '?'}")
        if "complete" in status:
            break
        # A status the poller does not recognise -- "cannot access", 404, a transient
        # SSL failure -- used to spin here forever at a 120 s cap. Tolerate a run of
        # them (Kaggle takes a moment to register a new kernel) then bail loudly.
        if not any(w in status for w in ("running", "queued", "complete",
                                         "error", "cancel")):
            unknown += 1
            if unknown > UNKNOWN_MAX:
                sys.exit(f"status unrecognised {unknown}x, last: {o}")
            continue
        unknown = 0
        if "error" in status or "cancel" in status:
            print("--- run failed; pulling log ---")
            sh("kernels", "output", slug, "-p", str(outdir))
            for log in outdir.glob("*.log"):
                print(log.read_text()[-4000:])
            sys.exit(1)
        delay = int(min(delay * 1.5, 120))

    pull(slug, outdir)


def pull(slug, outdir, tries=6):
    """Download kernel output, and refuse to report success on a broken download.

    The Kaggle CLI can print `Connection broken: IncompleteRead(...)`, return rc 0 and
    leave a 0-byte .npz on disk. This treats rc != 0, a connection error in the output, or
    any zero-byte artifact as a truncated pull, deletes the zero-byte files (the client
    skips files that already exist, so they would otherwise never be re-fetched) and
    retries. Kaggle's client does not resume, so each retry is whole-file.
    """
    outdir = Path(outdir)
    for attempt in range(1, tries + 1):
        rc, o = sh("kernels", "output", slug, "-p", str(outdir))
        print(o)
        # A zero-byte artifact is a truncated download, never a real result.
        empty = [f for f in outdir.glob("*.npz") if f.stat().st_size == 0]
        for f in empty:
            f.unlink()
        broken = ("connection broken" in o.lower() or "incompleteread" in o.lower()
                  or rc != 0 or bool(empty))
        good = [f for f in outdir.glob("*.npz") if f.stat().st_size > 1024]
        if not broken:
            print(f"pulled -> {outdir}  ({len(good)} non-empty .npz)")
            return good
        print(f"  TRUNCATED PULL (attempt {attempt}/{tries}): rc={rc}, "
              f"{len(empty)} zero-byte file(s) removed, {len(good)} complete so far. "
              f"Retrying.")
        time.sleep(min(15 * attempt, 90))
    sys.exit(f"pull failed after {tries} attempts: {slug} -> {outdir}. "
             f"{len(list(outdir.glob('*.npz')))} .npz on disk; DO NOT treat this as a "
             f"complete result set. The kernel itself is COMPLETE -- re-pull, do not "
             f"re-push.")


if __name__ == "__main__":
    a = sys.argv[1:]
    attach = "--attach" in a
    a = [x for x in a if x != "--attach"]
    main(*a, attach=attach)
