"""The reproduction appendix's sweep table (tab:sweeps), derived from the artifacts.

Every cell is read from an artifact's provenance config, the summary JSON of an unstamped
sweep, a kernel's own constants, or a run log -- never typed. test_paper_numbers asserts the
paper's table body equals rows() exactly, so the table cannot drift from what ran.

  PYTHONPATH=. .venv/bin/python scripts/sweep_table.py        # print the LaTeX rows
"""
import glob, json, re
from src import provenance

# name, results dir, file filter, training script, analysis script
SWEEPS = [
    ("k01", "k01_scout", "scout_*", "kernels/k01_scout/run.py", "analyze_scout.py"),
    ("k02", "k02_grid", "WE_*", "kernels/k02_grid/run.py", "analyze_k02.py"),
    ("k03", "k03_grid_acts", "WE_*", "kernels/k03_grid_acts/run.py", "analyze_k03.py"),
    ("k04", "k04_extended", "WE_*", "kernels/k04_extended/run.py", "analyze_k04.py"),
    ("k05", "k05_lowdata", "WE_*", "kernels/k05_lowdata/run.py", "analyze_k05.py"),
    ("k06", "k06_horizon", "WE_*", "kernels/k06_horizon/run.py", "analyze_k06.py"),
    ("k07", "k07_arma_seeds", "WE_*", "kernels/k07_arma_seeds/run.py", "analyze_k07.py"),
    ("k08", "k08_init", "WE_*", "kernels/k08_init/run.py", "analyze_k08.py"),
    ("k09", "k09_primes_zdd", "WE_*", "kernels/k09_primes_zdd/run.py", "analyze_k09.py"),
    ("N7", "n7_engine", "WE_*", "run_n7.py", "analyze_n7.py"),
    ("N9", "n9_f32", "WE_*", "run_n7.py", "analyze_n9.py"),
    ("O18", "o18_cpu", "WE_*", "run_o18_cpu.py", "analyze_n9.py"),
    ("O18, f64", "o18_f64", "WE_*", "run_o18_cpu.py --f64", "analyze_n9.py"),
    ("O18-T", "o18_transplant", "WE_*", "run_o18_transplant.py", "run_o18_transplant.py"),
    ("I1", "i1_internal", "WE_*n113*", "run_n7.py", "run_intervention.py"),
    ("I1b", "i1_internal", "WE_*n121*", "run_n7.py", "run_intervention.py"),
    ("Gate 1", "gate1", "add113_final*", "run_gate1.py", "test_gate1.py"),
]
# Local run logs (gitignored, like the .npz): each engine run ends "done. ..., X.XX h".
LOGS = {"N7": "logs/n7_n*_s[0-9].log", "N9": "logs/n9_f32_n113_s[0-9].log",
        "I1": ["logs/i1_n113_s[0-9].log", "logs/i1_measure_s[0-9].log"],
        "I1b": ["logs/i1b_n121_s[0-9].log", "logs/i1b_measure_n121_s[0-9].log"],
        "O18": "logs/o18_cpu.log", "O18, f64": "logs/o18_f64.log",
        "O18-T": "logs/o18t_stage2.log", "Gate 1": "results/gate1/run.log"}


PY = "PYTHONPATH=. .venv/bin/python"


def _n7(grid, out, prefix, extra_env=""):
    """An engine sweep through scripts/run_n7_all.sh: (shell command, argv each run stamps)."""
    g = "\\n".join(f"{n} {s}" for n, s in grid)
    env = f"{extra_env}N7_OUT={out} N7_LOG_PREFIX={prefix} N7_GRID=$'{g}' "
    return env + "bash scripts/run_n7_all.sh", [f"run_n7.py {n} {s} 40000" for n, s in grid]


def _i1(n, prefix, measure_log):
    sh, argv = _n7([(n, s) for s in (0, 1, 2)], "results/i1_internal", prefix)
    meas = [f"run_intervention.py {n} {s} --draws 10000" for s in (0, 1, 2)]
    sh += "".join(f" && {PY} -u {m} > logs/{measure_log.format(s=s)} 2>&1"
                  for s, m in enumerate(meas))
    return sh, argv + meas


# The exact command that trained each sweep, run from the top directory AT THE SWEEP'S
# STAMPED SHA (commands() prints it). check_commands() asserts every stamped argv is one this
# command issues. A Kaggle sweep runs on the account in $KAGGLE_OWNER (run_kernel.py rewrites
# only the owner half of the kernel id, and needs that account's KAGGLE_API_TOKEN).
COMMANDS = {
    **{name: (f"KAGGLE_OWNER=<account> {PY} kernels/run_kernel.py kernels/{d}", [])
       for name, d, *_ in SWEEPS if d.startswith("k0")},
    "N7": _n7([(n, s) for s in (0, 1, 2) for n in (113, 121, 125, 119)], "results/n7_engine", "n7"),
    "N9": _n7([(113, s) for s in (0, 1, 2)], "results/n9_f32", "n9_f32", "ENGINE_DTYPE=float32 "),
    "O18": (f"{PY} -u run_o18_cpu.py > logs/o18_cpu.log 2>&1", []),
    "O18, f64": (f"{PY} -u run_o18_cpu.py --f64 > logs/o18_f64.log 2>&1", []),
    "O18-T": (f"{PY} -u run_o18_transplant.py --stage2 113 0 40000 > logs/o18t_stage2.log 2>&1",
              ["run_o18_transplant.py --stage2 113 0 40000"]),
    "I1": _i1(113, "i1", "i1_measure_s{s}.log"),
    "I1b": _i1(121, "i1b", "i1b_measure_n121_s{s}.log"),
    "Gate 1": (f"{PY} -u run_gate1.py > results/gate1/run.log 2>&1", ["run_gate1.py"]),
}


def configs(d, pat):
    fs = sorted(glob.glob(f"results/{d}/{pat}.npz"))
    assert fs, f"no artifacts for results/{d}/{pat}"
    out = []
    for f in fs:
        p = provenance.read(f)
        out.append(json.loads(p["config"]) if p and "config" in p else None)
    if all(c is not None for c in out):
        return out
    # k01/k02 predate provenance: n and seed from the summary, the rest from the kernel's
    # own constants (one commit of each run.py; see scripts/reconstruct_provenance.py).
    summ = json.load(open(glob.glob(f"results/{d}/*_summary.json")[0]))
    src = open(f"kernels/{d}/run.py").read()
    k = dict(steps=int(re.search(r"^STEPS\s*=\s*([\d_]+)", src, re.M).group(1).replace("_", "")),
             engine="pytorch-autograd")
    tf, lr, wd = re.search(r"^TRAIN_FRAC, LR, WD, BETAS = ([\d.]+), ([\de.-]+), ([\d.]+),",
                           src, re.M).groups()
    k.update(train_frac=float(tf), lr=float(lr), wd=float(wd))
    assert len(summ) == len(fs), f"{d}: {len(summ)} summary rows vs {len(fs)} artifacts"
    return [dict(k, n=r["n"], seed=r["seed"]) for r in summ]


def hours(name, d):
    if d.startswith("k0"):   # Kaggle session wall time: the log's last timestamp, in seconds
        return json.load(open(glob.glob(f"results/{d}/grokking-*.log")[0]))[-1]["time"] / 3600
    pats = LOGS[name] if isinstance(LOGS[name], list) else [LOGS[name]]
    fs = [f for p in pats for f in sorted(glob.glob(p))]
    assert fs, f"no logs for {name}"
    h = 0.0
    for f in fs:
        t = open(f).read()
        m = (re.findall(r"done\. grok_step=\d+, ([\d.]+) h", t) or [])
        mins = re.findall(r"\[([\d.]+) min\]$|done in ([\d.]+) min", t, re.M)
        h += sum(map(float, m)) + sum(float(a or b) for a, b in mins) / 60
    return h


def fmt_set(xs, ints=True):
    xs = sorted(set(xs))
    f = (lambda x: f"${x:,}$".replace(",", "{,}")) if ints else (lambda x: f"${x}$")
    if ints and len(xs) > 2 and xs == list(range(xs[0], xs[-1] + 1)):
        return f"{f(xs[0])}--{f(xs[-1])}"
    return ", ".join(map(f, xs))


def tt(x):
    return "\\texttt{" + x.replace("_", "\\_").replace("--", "-{}-") + "}"


def rows():
    """Two row sets for the two tables: configuration, then files and compute."""
    conf, files = [], []
    for name, d, pat, train, ana in SWEEPS:
        cs = configs(d, pat)
        impl = {"pytorch-autograd": "PyTorch", "from-scratch": "engine"}[cs[0]["engine"]]
        assert all(c["engine"] == cs[0]["engine"] for c in cs)
        # the engine's default is float64 (src/autograd/engine.py DTYPE), PyTorch's float32
        prec = sorted({c.get("dtype", "float64" if impl == "engine" else "float32") for c in cs})
        ns = sorted({c["n"] for c in cs})
        mod = fmt_set(ns) if len(ns) <= 4 else f"${len(ns)}$, ${ns[0]}$ to ${ns[-1]}$"
        conf.append([name, impl, ", ".join(prec), mod, fmt_set([c["seed"] for c in cs]),
                     fmt_set([c["steps"] for c in cs]),
                     fmt_set([c["train_frac"] for c in cs], ints=False),
                     "Kaggle T4" if d.startswith("k0") else "CPU"])
        files.append([name, tt(train), tt(f"results/{d}"), tt(ana), f"${hours(name, d):.1f}$"])
    return conf, files


# What a sweep's training (and, for I1/I1b, measurement) can execute. Re-running a command
# at HEAD reproduces the sweep only if every one of these is unchanged since the stamped SHA,
# or its change is shown inert below.
_ENGINE = ["run_n7.py", "scripts/run_n7_all.sh", "src/autograd", "src/model", "src/train", "src/tasks"]
_I1 = _ENGINE + ["run_intervention.py", "src/analysis/ablation.py", "src/analysis/intervention.py"]
TRAIN_PATH = {**{name: [f"kernels/{d}/run.py"] for name, d, *_ in SWEEPS if d.startswith("k0")},
              "N7": _ENGINE, "N9": _ENGINE, "I1": _I1, "I1b": _I1,
              "O18": ["run_o18_cpu.py", "kernels/k03_grid_acts/run.py"],
              "O18, f64": ["run_o18_cpu.py", "kernels/k03_grid_acts/run.py"],
              "O18-T": ["run_o18_transplant.py", "kernels/k03_grid_acts/run.py",
                        "src/autograd", "src/model", "src/train"],
              "Gate 1": ["run_gate1.py", "src/autograd", "src/model", "src/train", "src/tasks"]}
# file -> (blob at HEAD when it was reviewed, why its change since the stamped SHAs is inert).
# Keyed by blob, so editing the file again re-opens the question instead of passing silently.
INERT = {
    "src/tasks/algebra.py": ("8ccf6f38d10d", "no training script imports it (run_n7, run_gate1, "
                             "src/train/loop import only src.autograd/src.model)"),
    "run_n7.py": ("e15907b8c0b8", "cb3f941: snapshot() also saves W_Q..W_pos; no training line"),
    "scripts/run_n7_all.sh": ("508f3b646ecb", "N7_GRID/N7_LOG_PREFIX overrides; defaults unchanged"),
    "src/autograd/engine.py": ("cd307d8b1654", "6991152: NEP-50 fix, a no-op at float64 -- shown: "
                               "the I1 retrain at a9238dd reproduced N7 (cafbb36) n=113 with "
                               "max|dW_E| = max|dlogits| = 0 on 3/3 seeds"),
    "run_o18_cpu.py": ("7113c1a90ee5", "adds the --f64 branch and stamp fields; the float32 "
                       "patch list is unchanged"),
    "run_intervention.py": ("7c3a0f1f4379", "every change is inside the --summary block"),
}


def _git(*a):
    import subprocess
    return subprocess.run(["git", *a], capture_output=True, text=True, check=True).stdout.split()


def drift():
    """(sweep, sha, file) for every training-path change since a stamped SHA that INERT does
    not cover at its current blob. Empty means every command re-runs its sweep at HEAD."""
    bad = []
    for name, shas, _ in commands():
        for sha in shas:
            if sha == "unknown":
                continue   # k03: reconstructed by argument, scripts/reconstruct_provenance.py
            for f in _git("diff", "--name-only", sha, "HEAD", "--", *TRAIN_PATH[name]):
                blob, _why = INERT.get(f, (None, None))
                if blob is None or _git("hash-object", f)[0][:12] != blob:
                    bad.append((name, sha, f))
    return bad


def _stamps(name, d, pat):
    pats = [pat] + ([f"I1_*n{pat.split('n')[-1]}"] if name.startswith("I1") else [])  # + measurement
    return [provenance.read(f) for q in pats for f in sorted(glob.glob(f"results/{d}/{q}.npz"))]


def commands():
    """(sweep, stamped SHAs, exact command) for every sweep. An unstamped sweep has no SHA;
    scripts/reconstruct_provenance.py recovers its code version by argument."""
    out = []
    for name, d, pat, *_ in SWEEPS:
        shas = sorted({str(p["git_sha"])[:7] for p in _stamps(name, d, pat) if p and p.get("git_sha")})
        out.append((name, shas, COMMANDS[name][0]))
    return out


def check_commands():
    """Every sweep has a command; every stamped argv is one its command issues; the command's
    dtype is the one the artifacts record; a Kaggle command names a kernel that exists."""
    assert set(COMMANDS) == {s[0] for s in SWEEPS}, set(COMMANDS) ^ {s[0] for s in SWEEPS}
    n_argv = 0
    for name, d, pat, *_ in SWEEPS:
        sh, argv = COMMANDS[name]
        ps = _stamps(name, d, pat)
        for p in ps:
            if p and p.get("argv"):
                assert p["argv"] in argv, (name, p["argv"], "not issued by", sh)
                n_argv += 1
        if argv:   # the command issues nothing the artifacts do not show
            assert set(argv) == {p["argv"] for p in ps if p and p.get("argv")}, (name, argv)
        want = {json.loads(p["config"]).get("dtype") for p in ps if p and p.get("config")} - {None}
        assert want <= {"float32" if ("ENGINE_DTYPE=float32" in sh or d.startswith("k0")
                         or d == "o18_cpu") else "float64"}, (name, want, sh)
        if d.startswith("k0"):
            assert glob.glob(f"kernels/{d}/kernel-metadata.json"), (name, "no kernel dir")
    assert n_argv > 0
    return n_argv


def _selfcheck():
    n = check_commands()
    # planted: a command that issues the wrong seed must fail, both directions
    keep = COMMANDS["N9"]
    COMMANDS["N9"] = (keep[0], ["run_n7.py 113 0 40000", "run_n7.py 113 1 40000", "run_n7.py 113 3 40000"])
    try:
        check_commands(); raise SystemExit("PLANTED wrong-seed command was NOT caught")
    except AssertionError:
        pass
    COMMANDS["N9"] = (keep[0].replace("ENGINE_DTYPE=float32 ", ""), keep[1])   # dtype dropped
    try:
        check_commands(); raise SystemExit("PLANTED missing ENGINE_DTYPE was NOT caught")
    except AssertionError:
        pass
    COMMANDS["N9"] = keep
    assert drift() == [], drift()
    blob, why = INERT["run_o18_cpu.py"]
    INERT["run_o18_cpu.py"] = ("000000000000", why)   # planted: the file changed after review
    assert ("O18", "846dbb1", "run_o18_cpu.py") in drift(), "PLANTED stale INERT blob was NOT caught"
    INERT["run_o18_cpu.py"] = (blob, why)
    print(f"sweep_table selfcheck PASS: {len(COMMANDS)} sweeps, {n} stamped argv matched, no training-path drift, 3 plants caught")


if __name__ == "__main__":
    import sys
    if "--selfcheck" in sys.argv:
        _selfcheck(); sys.exit(0)
    if "--commands" in sys.argv:
        for name, shas, sh in commands():
            print(f"# {name}  (git SHA: {', '.join(shas) or 'unstamped -- scripts/reconstruct_provenance.py'})\n{sh}\n")
        sys.exit(0)
    for part in rows():
        for r in part:
            print("    " + " & ".join(r) + " \\\\")
        print()
