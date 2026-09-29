"""Runnable check for the push path: kernel ids, placeholder substitution, wrong-account
guard. No network -- everything here is string handling and one deliberate early exit.

The failure this guards against is expensive rather than loud: a kernel pushed under the
wrong account either gets rejected mid-launch or lands somewhere whose quota and results
nobody is tracking.
"""
import json, os, shutil, sys, tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "kernels")
import run_kernel as R

META = {"id": "alice/grokking-k09-demo", "title": "Grokking k09 Demo",
        "code_file": "run.py", "language": "python", "kernel_type": "script",
        "enable_gpu": True, "machine_shape": "NvidiaTeslaT4"}
CODE = 'GIT_SHA = "__GIT_SHA__"\nOWNER = "__KAGGLE_OWNER__"\nprint(GIT_SHA, OWNER)\n'


def fixture(tmp):
    k = Path(tmp) / "k09_demo"
    k.mkdir()
    (k / "kernel-metadata.json").write_text(json.dumps(META, indent=2))
    (k / "run.py").write_text(CODE)
    return k


def main():
    # 1. the slug is invariant; only the owner half moves. A changed slug would break the
    #    title->slug rule and Kaggle answers 409.
    assert R.kernel_id(META) == "alice/grokking-k09-demo"
    assert R.kernel_id(META, "bob") == "bob/grokking-k09-demo"
    assert R.kernel_id(META, None) == "alice/grokking-k09-demo"
    print("  kernel_id: owner overridable, slug invariant      OK")

    tmp = tempfile.mkdtemp()
    try:
        k = fixture(tmp)

        # 2. no owner given -> the id is untouched, placeholders still substituted
        d = R.staged_copy(k, "deadbeef")
        m = json.loads((d / "kernel-metadata.json").read_text())
        src = (d / "run.py").read_text()
        assert m["id"] == "alice/grokking-k09-demo", m["id"]
        assert 'GIT_SHA = "deadbeef"' in src, src
        assert 'OWNER = "alice"' in src, src
        print("  staged_copy without owner: id untouched          OK")

        # 3. owner given -> id rewritten, and the OWNER stamp follows it
        d = R.staged_copy(k, "deadbeef", owner="bob")
        m = json.loads((d / "kernel-metadata.json").read_text())
        src = (d / "run.py").read_text()
        assert m["id"] == "bob/grokking-k09-demo", m["id"]
        assert m["title"] == META["title"], "title must not move with the owner"
        assert 'OWNER = "bob"' in src, src
        print("  staged_copy with owner: id and stamp rewritten   OK")

        # 4. the TRACKED source is never modified -- the placeholder stays in git
        assert (k / "run.py").read_text() == CODE
        assert json.loads((k / "kernel-metadata.json").read_text())["id"] == META["id"]
        print("  tracked kernel dir left clean                    OK")

        # 5. the guard: authenticated as someone else -> refuse BEFORE any network call
        real = R.authenticated_user
        R.authenticated_user = lambda: "carol"
        R.sh = lambda *a: (_ for _ in ()).throw(AssertionError("network call after guard"))
        try:
            R.main(str(k), outdir=str(Path(tmp) / "out"))
        except SystemExit as e:
            assert "carol" in str(e) and "alice" in str(e), e
            print("  wrong-account push refused, no network           OK")
        else:
            raise AssertionError("guard did not fire on an account mismatch")
        finally:
            R.authenticated_user = real
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\ntest_run_kernel: PASS")


if __name__ == "__main__":
    main()
