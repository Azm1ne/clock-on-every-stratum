"""Structural checks on the paper's LaTeX, and (--compile) a real pdflatex build.

The structural pass needs no TeX installation. It catches, locally and in a second, the three errors
that otherwise surface as a red Overleaf build minutes later: unbalanced
environments, a \\cite key that is not in refs.bib, and a \\ref with no \\label.

Usage:  .venv/bin/python paper/check_tex.py [paper_dir]
        .venv/bin/python paper/check_tex.py --compile [paper_dir]
Exit 1 on any failure, so it can gate a commit.

--compile runs a real pdflatex build (latexmk, -halt-on-error) and fails on any TeX error,
undefined reference or citation, or a box more than OVERFULL_PT into the margin. The
structural checks cannot see a TeX error that nonstop mode recovers from, such as a table
that declares fewer columns than its rows use.
"""
import collections
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

OVERFULL_PT = 10.0   # below this an overfull box is invisible in print

ENV = re.compile(r"\\(begin|end)\{([^}]+)\}")
CITE = re.compile(r"\\cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]+)\}")
REF = re.compile(r"\\(?:page|eq|auto|c)?ref\*?\{([^}]+)\}")
LABEL = re.compile(r"\\label\{([^}]+)\}")
SECTION = re.compile(r"\\(?:sub)*section\*?\{[^}]*\}")
# Verbatim-like environments may legitimately contain unbalanced-looking text.
SKIP_ENV = {"verbatim", "lstlisting", "minted"}
# scripts/make_overleaf_zip.sh's output, unzipped in place. A build artifact, never a source.
ARTIFACT_DIR = "overleaf_bundle"


def _strip_comments(text):
    """Drop % comments, honouring \\% escapes. LaTeX comments run to end of line."""
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in text.splitlines())


def _sources(d, pattern):
    """Every source file under d, with build artifacts excluded.

    An unzipped overleaf_bundle/ is a complete second copy of the sources, and rglob
    cannot tell a copy from an original: it would double every count and still pass,
    because a stale copy of a correct paper is also correct.
    """
    return [f for f in sorted(d.rglob(pattern)) if ARTIFACT_DIR not in f.parts]


def check(paper_dir="paper"):
    d = Path(paper_dir)
    tex = _sources(d, "*.tex")
    if not tex:
        print(f"no .tex under {d}/ yet -- nothing to check")
        return 0

    bib_keys = set()
    for b in _sources(d, "*.bib"):
        bib_keys |= set(re.findall(r"@\w+\{([^,]+),", b.read_text(errors="replace")))

    fails = []

    # Skipping the ONE artifact directory by name would leave every other stray copy
    # silently doubling the counts, so any repeated basename is a failure naming both paths.
    seen = {}
    for f in tex:
        seen.setdefault(f.name, []).append(f)
    for name, paths in seen.items():
        if len(paths) > 1:
            fails.append(f"{name}: {len(paths)} copies under {d}/ "
                         f"({', '.join(str(p) for p in paths)}) -- a copy of the sources "
                         f"doubles every count and still reports 0 failures")

    labels, refs, cites = set(), [], []

    for f in tex:
        src = _strip_comments(f.read_text(errors="replace"))
        stack = []
        for kind, name in ENV.findall(src):
            if name in SKIP_ENV:
                continue
            if kind == "begin":
                stack.append(name)
            elif not stack:
                fails.append(f"{f}: \\end{{{name}}} with no matching \\begin")
            elif stack[-1] != name:
                fails.append(f"{f}: \\end{{{name}}} closes \\begin{{{stack[-1]}}}")
                stack.pop()
            else:
                stack.pop()
        for name in stack:
            fails.append(f"{f}: \\begin{{{name}}} never closed")

        labels |= set(LABEL.findall(src))
        refs += [(f, r) for r in REF.findall(src)]
        # \cite{a,b} is two keys.
        cites += [(f, k.strip()) for grp in CITE.findall(src) for k in grp.split(",")]

    for f, r in refs:
        if r not in labels:
            fails.append(f"{f}: \\ref{{{r}}} has no \\label")

    # A \ref INSIDE the very sectioning unit it points at is never correct -- it is a
    # placeholder for a cross-reference that does not exist yet, and it passes the
    # has-a-label check above because the label really is there. That is how four of them
    # shipped in the first drafted section: a check whose failure mode is indistinguishable
    # from its success mode is not a check.
    for f in tex:
        src = _strip_comments(f.read_text(errors="replace"))
        marks = [m.start() for m in SECTION.finditer(src)] + [len(src)]
        for i in range(len(marks) - 1):
            # a block ends at the next heading of ANY level -- ending it at the next
            # \section makes a subsection swallow its siblings and false-positive on
            # a legitimate reference from one subsection to another.
            body = src[marks[i]:marks[i + 1]]
            # the block's OWN label follows its heading directly; a label further down
            # belongs to a float or an equation inside the block, and referring to that
            # from the same block is an ordinary cross-reference.
            head = SECTION.match(body)
            own = head and re.match(r"\s*\\label\{([^}]+)\}", body[head.end():])
            if not own:
                continue
            name = own.group(1)
            # the label's own \ref, anywhere later in the same section
            if any(r == name for r in REF.findall(body)):
                fails.append(f"{f}: \\ref{{{name}}} is inside the section it labels "
                             f"-- a self-reference, almost certainly a placeholder")
    # An empty .bib means the bibliography is not written yet; do not cry wolf.
    if bib_keys:
        for f, k in cites:
            if k not in bib_keys:
                fails.append(f"{f}: \\cite{{{k}}} not in refs.bib")
    elif cites:
        print(f"note: {len(cites)} \\cite keys but refs.bib is empty or absent -- not checked")

    for msg in fails:
        print("FAIL", msg)
    print(f"{len(tex)} .tex | {len(labels)} labels | {len(refs)} refs | "
          f"{len(cites)} cites | {len(bib_keys)} bib keys | {len(fails)} failures")
    return 1 if fails else 0


def compile_pdf(paper_dir="paper", main="main.tex", pdf_out=None):
    """A real pdflatex build into a temp dir; 1 on error, undefined ref/cite, or wide box.

    pdf_out: copy the PDF there ONLY if the build passed, so what ships is what was gated."""
    paper_dir = Path(paper_dir).resolve()
    with tempfile.TemporaryDirectory() as out:
        env = {**os.environ, "BIBINPUTS": f"{paper_dir}:", "BSTINPUTS": f"{paper_dir}:"}
        try:
            r = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode",
                                "-halt-on-error", f"-outdir={out}", main],
                               cwd=paper_dir, env=env, capture_output=True, text=True)
        except FileNotFoundError:
            print("FAIL latexmk not installed -- --compile cannot run, and a skip is not a pass")
            return 1
        log_path = Path(out) / (Path(main).stem + ".log")
        log = log_path.read_text(errors="replace") if log_path.exists() else ""
        fails = [l for l in log.splitlines() if l.startswith("!")][:3]
        if r.returncode and not fails:
            fails.append(f"latexmk exited {r.returncode}")
        fails += sorted({m.group(0) for m in re.finditer(
            r"(?:Reference|Citation) `[^']+' on page \d+ undefined", log)})
        fails += [m.group(0) for m in re.finditer(
            r"Overfull \\hbox \(([\d.]+)pt too wide\)[^\n]*", log)
            if float(m.group(1)) > OVERFULL_PT]
        pages = re.search(r"Output written on .*?\((\d+) pages?", log)
        if pdf_out and not fails:
            shutil.copy(Path(out) / (Path(main).stem + ".pdf"), pdf_out)
    for msg in fails:
        print("FAIL", msg)
    print(f"compile: {pages.group(1) if pages else '?'} pages | {len(fails)} failures")
    return 1 if fails else 0


def _compile_selfcheck(tmp):
    """The bug that motivated --compile must fail it; the same table fixed must pass."""
    doc = ("\\documentclass{article}\\begin{document}\n"
           "\\begin{tabular}{%s}a & b & c\\\\\\end{tabular}\n\\end{document}\n")
    (tmp / "t.tex").write_text(doc % "ll")
    assert compile_pdf(tmp, "t.tex") == 1, "3 cells under a 2-column spec must fail"
    (tmp / "t.tex").write_text(doc % "lll")
    assert compile_pdf(tmp, "t.tex") == 0, "the fixed table must compile"
    # An undefined \ref is a warning, not an error: latexmk exits 0 on it.
    (tmp / "t.tex").write_text("\\documentclass{article}\\begin{document}"
                               "See \\ref{nope}.\\end{document}\n")
    assert compile_pdf(tmp, "t.tex") == 1, "an undefined \\ref must fail"
    (tmp / "t.tex").unlink()


def _selfcheck(tmp):
    """Plant each error class and assert it is caught; assert a clean tree passes."""
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "refs.bib").write_text("@article{good2026, title={T}, year={2026}}\n")

    (tmp / "ok.tex").write_text(
        "\\begin{figure}\\label{fig:a}\\end{figure}\n"
        "See \\ref{fig:a} and \\cite{good2026}.\n"
        "% \\cite{commented_out} must be ignored\n")
    assert check(tmp) == 0, "clean tree must pass"

    (tmp / "bad.tex").write_text(
        "\\begin{table}\\end{figure}\n"      # mismatched
        "\\ref{nope}\n"                       # no label
        "\\cite{missing2026}\n")              # not in bib
    assert check(tmp) == 1, "planted errors must fail"

    (tmp / "bad.tex").unlink()
    (tmp / "unclosed.tex").write_text("\\begin{itemize}\n")
    assert check(tmp) == 1, "unclosed environment must fail"
    (tmp / "unclosed.tex").unlink()

    (tmp / "selfref.tex").write_text(
        "\\section{S}\\label{sec:s}\nAs shown in \\ref{sec:s} this is a placeholder.\n")
    assert check(tmp) == 1, "a \\ref inside the section it labels must fail"
    (tmp / "selfref.tex").unlink()
    assert check(tmp) == 0, "clean tree must still pass after the self-ref planting"
    # An UNLABELLED subsection holding a table: its label is the table's, not the
    # subsection's, and a later "see Table" is an ordinary cross-reference.
    (tmp / "tabref.tex").write_text(
        "\\subsection{T}\nIntro.\n\\begin{table}\\caption{C}\\label{tab:t}\\end{table}\n"
        "The numbers agree (Table~\\ref{tab:t}).\n")
    assert check(tmp) == 0, "a table referenced from its own unlabelled subsection must pass"
    (tmp / "tabref.tex").unlink()

    # The artifact copy: ignored, and the counts must not move.
    bundle = tmp / ARTIFACT_DIR / "sections"
    bundle.mkdir(parents=True)
    (bundle / "ok.tex").write_text((tmp / "ok.tex").read_text())
    assert check(tmp) == 0, f"{ARTIFACT_DIR}/ is a build artifact and must be skipped"

    # Any OTHER copy: a failure, not a silent doubling.
    stray = tmp / "sections"
    stray.mkdir(parents=True, exist_ok=True)
    (stray / "ok.tex").write_text((tmp / "ok.tex").read_text())
    assert check(tmp) == 1, "a duplicate source basename must fail"
    (stray / "ok.tex").unlink()
    assert check(tmp) == 0, "clean tree must still pass once the duplicate is gone"
    _compile_selfcheck(tmp)
    print("selfcheck OK")


# ------------------------------------------------------------------ the register pass (--style)
# Hard rules fail; soft rules are reported for a person to decide. Both lists are fixed here,
# before any section is edited, so they cannot be tuned per section. Text inside ``...''
# quotation marks is another author's words and is exempt from the word lists.
BANNED = [r"novel", r"striking", r"remarkabl\w*", r"surprising\w*", r"fascinating",
          r"crucial\w*", r"pivotal", r"we are the first", r"to the best of our knowledge",
          r"sheds? light on", r"paves? the way", r"opens? the door", r"a deep dive",
          r"rich structure", r"powerful", r"significantly", r"dramatically",
          r"substantially", r"considerably", r"a wide range of", r"numerous", r"various",
          r"several", r"moreover", r"additionally", r"furthermore", r"actually",
          r"notably", r"importantly", r"delv\w*", r"underscor\w*", r"showcas\w*",
          r"tapestry", r"testament", r"intricate\w*", r"realms?", r"harness\w*",
          r"leverag\w*", r"seamless\w*"]
PERFORMATIVE = [r"plainly", r"we report (?:that|it)", r"the reason is", r"we say so",
                r"this matters", r"honest\w*", r"in the open", r"named rather than rated"]
SOFT = {"contrast": r",\s+not\s|\bnot\b[^.]*\bbut\b|\brather than\b",
        "copula": r"\b(?:stands|serves as|exhibits)\b"}
# -ing words allowed in STE regions: technical nouns, and words that are not -ing forms.
STE_ING = {"training", "grokking", "logging", "embedding", "embeddings", "during",
           "string", "strings", "nothing", "something", "anything", "everything",
           "thing", "things", "ring", "rings", "ceiling", "setting", "settings"}
_PASSIVE = re.compile(r"\b(?:am|is|are|was|were|be|been|being)\s+(?:\w+ly\s+)?"
                      r"(?:\w+ed|built|chosen|done|drawn|found|given|held|kept|known|made|"
                      r"run|seen|set|shown|taken|written|read|put|left|lost|met|sent|split)\b",
                      re.I)
_PERFECT = re.compile(r"\b(?:has|have|had)\s+(?:\w+ly\s+)?(?:\w+ed|been|built|chosen|done|"
                      r"drawn|found|given|held|kept|known|made|run|seen|set|shown|taken|"
                      r"written|read|put|left|lost|met|sent|split)\b", re.I)
_ABBR = ["e.g.", "i.e.", "et al.", "cf.", "vs.", "etc.", "Fig.", "Eq.", "Sec.", "Prop.",
         "Thm.", "Def.", "resp.", "approx.", "No."]
_TABLE_ENVS = ("tabular", "tabular*")
_MATH_ENVS = ("equation", "equation*", "align", "align*", "gather", "gather*",
              "multline", "multline*")
_FLOAT_ENVS = ("figure", "figure*", "table", "table*")


def _blank(s, a, b, fill=" "):
    """s with s[a:b] replaced by `fill` padded to the same length; newlines kept."""
    mid = "".join("\n" if c == "\n" else " " for c in s[a:b])
    mid = fill + mid[len(fill):] if len(fill) <= len(mid) else mid
    return s[:a] + mid + s[b:]


def _brace_end(s, i):
    """Index just past the brace group that opens at s[i] == '{'."""
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{" and (j == 0 or s[j - 1] != "\\"):
            depth += 1
        elif s[j] == "}" and s[j - 1] != "\\":
            depth -= 1
            if depth == 0:
                return j + 1
    return len(s)


def _env_spans(s, names):
    """(start, end, body_start, body_end, name) of every \\begin{name}...\\end{name}."""
    out = []
    for name in names:
        b = re.escape(f"\\begin{{{name}}}")
        for m in re.finditer(b, s):
            e = s.find(f"\\end{{{name}}}", m.end())
            if e >= 0:
                out.append((m.start(), e + len(f"\\end{{{name}}}"), m.end(), e, name))
    return sorted(out)


def _ste_mode(orig):
    """[(start, end, mode)] from first-on-line `% STE procedural|descriptive` / `% STE end`."""
    regions, cur, pos = [], None, 0
    for line in orig.splitlines(keepends=True):
        m = re.match(r"\s*%\s*STE\s+(procedural|descriptive|end)\b", line)
        if m:
            if cur and m.group(1) == "end":
                regions.append((cur[0], pos, cur[1])); cur = None
            elif m.group(1) != "end":
                cur = (pos, m.group(1))
        pos += len(line)
    return regions


def plain(raw):
    """Prose words of a LaTeX fragment: inline math is one word, \\cite and \\ref none."""
    s = re.sub(r"\$[^$]*\$", " MATH ", raw)
    s = re.sub(r"\\cite\w*\*?(?:\[[^\]]*\])*\{[^}]*\}", " ", s)
    s = re.sub(r"\\(?:eq|page|auto|c|C)?ref\*?\{[^}]*\}", " ", s)
    for _ in range(4):
        s = re.sub(r"\\[a-zA-Z]+\*?\{([^{}]*)\}", r" \1 ", s)
    s = re.sub(r"\\[a-zA-Z]+\*?", " ", s)
    s = s.replace("~", " ").replace("---", " ").replace("--", " ")
    s = re.sub(r"\\.", " ", s)
    return re.sub(r"[{}]", "", s)


def words(raw):
    return [w for w in plain(raw).split() if re.search(r"[A-Za-z0-9]", w)]


def units(orig):
    """Split a section's source into units: sentences of prose, of captions and of list
    items, headings, table rows and display math. Each unit is a dict with kind, raw text,
    1-based line, paragraph index and STE mode (None outside a region)."""
    work = "\n".join(re.sub(r"(?<!\\)%.*$", lambda m: " " * len(m.group(0)), line)
                     for line in orig.split("\n"))
    ste = _ste_mode(orig)
    line_of = lambda pos: orig.count("\n", 0, pos) + 1
    mode_of = lambda pos: next((m for a, b, m in ste if a <= pos < b), None)
    out = []

    def add(kind, raw, pos, para=None, first=False):
        if raw.strip():
            pos += len(raw) - len(raw.lstrip())
            out.append({"kind": kind, "raw": raw.strip(), "line": line_of(pos),
                        "para": para, "ste": mode_of(pos), "first": first})

    # floats: keep the caption as prose, drop everything else in the environment
    caps = []
    for a, b, ba, be, _ in _env_spans(work, _FLOAT_ENVS):
        for m in re.finditer(r"\\caption(?:\[[^\]]*\])?\{", work[ba:be]):
            i = ba + m.end() - 1
            caps.append((i + 1, _brace_end(work, i) - 1))
    # tables: one unit per row; display math: one unit each, and MATH inside its sentence
    for a, b, ba, be, _ in _env_spans(work, _TABLE_ENVS):
        body = work[ba:be]
        body = body[_brace_end(body, body.find("{")):] if body.lstrip().startswith("{") else body
        off, pos = be - len(body), 0
        for m in re.finditer(r"\\\\|\Z", body):
            add("table-row", body[pos:m.start()], off + pos)
            pos = m.end()
        work = _blank(work, a, b)
    for a, b, ba, be, _ in _env_spans(work, _MATH_ENVS):
        add("math", work[ba:be], ba)
        work = _blank(work, a, b, " MATH ")
    for m in re.finditer(r"\\\[(.*?)\\\]", work, re.S):
        add("math", m.group(1), m.start())
        work = _blank(work, m.start(), m.end(), " MATH ")
    for a, b, ba, be, _ in _env_spans(work, _FLOAT_ENVS):
        work = _blank(work, a, b, "\x01")
    for i, (a, b) in enumerate(caps):
        _sentences(orig, work_caption(orig, a, b), a, add, "caption", f"cap{i}")
    # headings and structural commands become paragraph breaks
    for m in reversed(list(re.finditer(
            r"\\(?:sub)*section\*?\{|\\paragraph\*?\{|\\begin\{\w+\*?\}(?:\[[^\]]*\])?|"
            r"\\end\{\w+\*?\}|\\item\b|\\maketitle|\\appendix|\\noindent|\\label\{[^}]*\}",
            work))):
        tok = m.group(0)
        if tok.endswith("{") and not tok.startswith("\\label"):
            e = _brace_end(work, m.end() - 1)
            add("heading", work[m.end():e - 1], m.start())
            work = _blank(work, m.start(), e, "\x01")
        elif tok.startswith("\\label"):
            work = _blank(work, m.start(), m.end())
        else:
            title = re.search(r"\[([^\]]*)\]$", tok)
            if title:
                add("heading", title.group(1), m.start())
            work = _blank(work, m.start(), m.end(), "\x01")
    _sentences(orig, work, 0, add, "sentence", "p")
    return sorted(out, key=lambda u: u["line"])


def work_caption(orig, a, b):
    """The caption text, comment-stripped, as a standalone fragment."""
    return "\n".join(re.sub(r"(?<!\\)%.*$", lambda m: " " * len(m.group(0)), line)
                     for line in orig[a:b].split("\n"))


def _sentences(orig, text, base, add, kind, tag):
    """Split `text` (offset `base` in the file) into paragraphs, then sentences."""
    mask = re.sub(r"\$[^$]*\$", lambda m: "$" + "M" * (len(m.group(0)) - 2) + "$", text)
    for ab in _ABBR:
        mask = mask.replace(ab, ab.replace(".", "\x00"))
    mask = re.sub(r"\\\s", "\x00\x00", mask)            # "et al.\ " is not a full stop
    # \par is a paragraph break exactly as a blank line is (a caption cannot hold a blank
    # line, so an STE caption over six sentences breaks with \par). Same length, so every
    # offset into `text` stays valid.
    mask = re.sub(r"\\par(?![A-Za-z])", "\x01" * 4, mask)   # \x01 never joins a paragraph
    for pi, para in enumerate(re.finditer(r"(?:(?!\n\s*\n)[^\x01])+", mask)):
        start = para.start()
        cuts = [m.end() for m in re.finditer(r"[.!?][')\}]*(?=\s+[A-Z$\\(`\[])", para.group(0))]
        for k, (a, b) in enumerate(zip([0] + cuts, cuts + [len(para.group(0))])):
            add(kind, text[start + a:start + b], base + start + a, f"{tag}{pi}", k == 0)


def _findings_for(f, orig):
    """(severity, rule, file, line, message) for one file."""
    found = []
    per_para = {}
    for u in units(orig):
        raw, where = u["raw"], (str(f), u["line"])
        if u["kind"] in ("table-row", "math"):
            continue
        prose = re.sub(r"``.*?''", " QUOTE ", raw, flags=re.S)   # another author's words
        low = " ".join(plain(prose).lower().split())   # a phrase can wrap a line
        if "---" in raw or "\u2014" in raw:
            found.append(("hard", "em-dash", *where, raw[:70]))
        for pat in BANNED:
            if re.search(rf"\b{pat}\b", low):
                found.append(("hard", "banned", *where, pat))
        for pat in PERFORMATIVE:
            if re.search(rf"\b{pat}\b", low):
                found.append(("hard", "performative", *where, pat))
        if u["kind"] == "heading":
            continue
        bolds = [m.start() for m in re.finditer(r"\\textbf\{", raw)]
        lead_ok = (u["kind"] == "caption" and u["first"] and raw.startswith("\\textbf{")
                   and len(bolds) == 1)
        if bolds and not lead_ok:
            rule = "bold-lead" if raw.startswith("\\textbf{") else "bold"
            found.append(("hard", rule, *where, raw[:70]))
        n = len(words(raw))
        if n > 40:
            found.append(("hard", "long", *where, f"{n} words"))
        elif n > 30:
            found.append(("soft", "over-30", *where, f"{n} words"))
        for rule, pat in SOFT.items():
            if re.search(pat, plain(prose), re.I):
                found.append(("soft", rule, *where, raw[:60]))
        if u["ste"]:
            limit = 20 if u["ste"] == "procedural" else 25
            if n > limit:
                found.append(("hard", f"ste-{u['ste']}-length", *where, f"{n} > {limit}"))
            per_para.setdefault(u["para"], []).append(u)
            if ";" in raw:
                found.append(("hard", "ste-semicolon", *where, raw[:60]))
            if _PASSIVE.search(low):
                sev = "hard" if u["ste"] == "procedural" else "soft"
                found.append((sev, "ste-passive", *where, _PASSIVE.search(low).group(0)))
            if _PERFECT.search(low):
                found.append(("hard", "ste-perfect", *where, _PERFECT.search(low).group(0)))
            # "learning rate" is a technical noun; "learning" alone is a verb form
            for w in re.findall(r"\b[a-z]+ing\b", re.sub(r"\blearning rate\b", "rate", low)):
                if w not in STE_ING:
                    found.append(("hard", "ste-ing", *where, w))
        elif ";" in raw:
            found.append(("soft", "semicolon", *where, raw[:60]))
    for para, us in per_para.items():
        if us[0]["ste"] == "descriptive" and len(us) > 6:
            found.append(("hard", "ste-paragraph", str(f), us[0]["line"], f"{len(us)} sentences"))
    return found


def style_findings(paper_dir="paper"):
    d = Path(paper_dir)
    return [x for f in _sources(d / "sections", "*.tex")
            for x in _findings_for(f, f.read_text(errors="replace"))]


def style_report(paper_dir="paper", files=None):
    """The baseline table: prose metrics per file, from the same extractor the lint uses."""
    rows, tot = [], collections.Counter()
    for f in files or _sources(Path(paper_dir) / "sections", "*.tex"):
        orig = f.read_text(errors="replace") if isinstance(f, Path) else f[1]
        name = f.name if isinstance(f, Path) else f[0]
        us = units(orig)
        prose = [u for u in us if u["kind"] == "sentence"]
        caps = [u for u in us if u["kind"] == "caption"]
        wl = [len(words(u["raw"])) for u in prose]
        cw = sum(len(words(u["raw"])) for u in caps)
        body = "\n".join(u["raw"] for u in prose)
        r = dict(file=name, words=sum(wl), sent=len(wl),
                 mean=round(sum(wl) / len(wl), 1) if wl else 0,
                 over40=round(100 * sum(w > 40 for w in wl) / len(wl)) if wl else 0,
                 max=max(wl, default=0), dash=body.count("---") + body.count("\u2014"),
                 semi=body.count(";"),
                 contrast=sum(bool(re.search(SOFT["contrast"], plain(u["raw"]), re.I))
                              for u in prose),
                 bold=body.count("\\textbf{"), cap_words=cw,
                 cap_dash=sum(u["raw"].count("---") for u in caps))
        rows.append(r)
        tot.update({k: v for k, v in r.items() if isinstance(v, int) and k not in ("max",)})
    return rows, tot


def _style_selfcheck(tmp):
    """One planted violation per hard rule must fail for that rule; clean text must pass.

    Every plant is scored on its own file, so a finding from one cannot satisfy another."""
    sec = tmp / "sections"
    sec.mkdir(parents=True, exist_ok=True)
    long41 = " ".join(["word"] * 40) + " end."
    plants = {
        "em-dash": "The clock --- a circuit --- runs on units.\n",
        "em-dash-unicode": "The clock \u2014 a circuit \u2014 runs on units.\n",
        "banned": "This is a novel result about the clock.\n",
        "banned-plan": "Moreover, the clock runs on the units.\n",
        "performative": "The clock plainly runs on the units.\n",
        "performative-wrapped": "Both are reported here in the\nopen, as results.\n",
        "bold": "The clock \\textbf{runs} on the units.\n",
        "bold-lead": "\\textbf{The clock runs.} It runs on the units.\n",
        "long": long41 + "\n",
        "ste-procedural-length": "% STE procedural\n"
            + " ".join(["Run"] + ["word"] * 20) + ".\n% STE end\n",
        "ste-descriptive-length": "% STE descriptive\n"
            + " ".join(["The"] + ["word"] * 25) + ".\n% STE end\n",
        "ste-paragraph": "% STE descriptive\n"
            + " ".join(f"The run {i} holds." for i in range(7)) + "\n% STE end\n",
        "ste-passive": "% STE procedural\nThe seed is fixed before the run.\n% STE end\n",
        "ste-perfect": "% STE descriptive\nThe run has finished at step 400.\n% STE end\n",
        "ste-ing": "% STE procedural\nRun the script, checking the output.\n% STE end\n",
        "ste-semicolon": "% STE procedural\nInstall the packages; run the script.\n% STE end\n",
        # a caption is one paragraph unless it breaks with \par (a caption holds no blank line)
        "ste-paragraph-caption": "% STE descriptive\n\\begin{figure}\\caption{"
            + " ".join(f"The run {i} holds." for i in range(7)) + "}\\end{figure}\n% STE end\n",
        "ste-ing-verb": "% STE descriptive\nThe model is learning the table.\n% STE end\n",
        # a caption with a short title is still a caption, and must not vanish from the lint
        "em-dash-shortcaption": "\\begin{figure}\\caption[Short.]{\\textbf{Lead.} The clock"
            " --- a circuit --- runs.}\\end{figure}\n",
    }
    clean = (
        "\\section{Results}\n\\label{sec:r}\n"
        "\\paragraph{A run-in header is allowed.} The clock runs on the units.\n\n"
        "Ranges are en dashes, as in $1$--$2$, and Nanda et al.\\ use one.\n\n"
        "% moreover, novel --- a comment is not prose\n"
        "The paper quotes ``leverages all frequencies'' from another paper.\n\n"
        + " ".join(["word"] * 39) + " end.\n\n"
        "\\begin{figure}\\caption{\\textbf{A caption lead is allowed.} The panel shows"
        " the grid.}\\end{figure}\n"
        "\\begin{center}\\begin{tabular}{l}\\textbf{3/3} \\\\ --- \\\\ \\end{tabular}"
        "\\end{center}\n\n"
        "% STE procedural\nInstall the pinned packages.\n\nRun the script.\n% STE end\n\n"
        "% STE descriptive\nThe training split holds 30 percent of the pairs. The test set"
        " holds the rest.\n% STE end\n\n"
        # \par is a paragraph break, as a blank line is, and ends the sentence before it
        "% STE descriptive\n\\begin{figure}\\caption{"
        + " ".join(f"The run {i} holds." for i in range(4)) + "\\par "
        + " ".join(f"The run {i} holds." for i in range(4, 8)) + "}\\end{figure}\n"
        "The learning rate is $10^{-3}$.\n% STE end\n")
    # the sentence after a \par must not keep the tail of the control word ("ar ...")
    _after = [u["raw"] for u in units("\\begin{figure}\\caption{One two.\\par Three four.}"
                                      "\\end{figure}\n") if u["kind"] == "caption"]
    assert _after == ["One two.", "Three four."], f"\\par must split cleanly, got {_after}"
    for f in sec.glob("*.tex"):
        f.unlink()
    (sec / "clean.tex").write_text(clean)
    hard = [f for f in style_findings(tmp) if f[0] == "hard"]
    assert not hard, f"clean text must pass the lint, got {hard}"
    for rule, body in plants.items():
        (sec / "plant.tex").write_text(body)
        got = {f[1] for f in style_findings(tmp) if f[0] == "hard"}
        want = (rule.split("-unicode")[0].split("-wrapped")[0].split("-caption")[0]
                .split("-shortcaption")[0]
                .replace("banned-plan", "banned").replace("ste-ing-verb", "ste-ing"))
        assert want in got, f"planted {rule!r} must fail rule {want!r}, got {sorted(got)}"
        (sec / "plant.tex").unlink()
    print("style selfcheck OK")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:2] == ["--style", "--selfcheck"]:
        with tempfile.TemporaryDirectory() as t:
            _style_selfcheck(Path(t) / "paper")
    elif args[:2] == ["--style", "--report"]:
        # --rev <sha>: measure the sections as committed at <sha> (the main version)
        rev = args[args.index("--rev") + 1] if "--rev" in args else None
        files = None
        if rev:
            names = subprocess.run(["git", "ls-tree", "--name-only", f"{rev}:paper/sections"],
                                   capture_output=True, text=True, check=True).stdout.split()
            files = [(n, subprocess.run(["git", "show", f"{rev}:paper/sections/{n}"],
                                        capture_output=True, text=True, check=True).stdout)
                     for n in names if n.endswith(".tex")]
        rows, tot = style_report("paper", files)
        cols = ["file", "words", "sent", "mean", "over40", "max", "dash", "semi", "contrast",
                "bold", "cap_words", "cap_dash"]
        print("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols))
        for r in rows:
            print("| " + " | ".join(str(r[c]) for c in cols) + " |")
        print(f"| total | {tot['words']} | {tot['sent']} | "
              f"{round(tot['words'] / max(tot['sent'], 1), 1)} | | | {tot['dash']} | "
              f"{tot['semi']} | {tot['contrast']} | {tot['bold']} | {tot['cap_words']} | "
              f"{tot['cap_dash']} |")
    elif args[:1] == ["--style"]:
        rest = [a for a in args[1:] if not a.startswith("--")]
        found = style_findings(rest[0] if rest else "paper")
        hard = [f for f in found if f[0] == "hard"]
        for sev, rule, f, line, msg in found:
            if sev == "hard" or "--soft" in args:
                print(f"{sev.upper():4} {rule:24} {f}:{line}  {' '.join(str(msg).split())}")
        print(f"style: {len(hard)} hard, {len(found) - len(hard)} soft")
        sys.exit(1 if hard else 0)
    elif args[:1] == ["--selfcheck"]:
        with tempfile.TemporaryDirectory() as t:
            _selfcheck(Path(t) / "paper")
    elif args[:1] == ["--compile"]:
        pdf = args[args.index("--pdf") + 1] if "--pdf" in args else None
        rest = [a for a in args[1:] if a not in ("--pdf", pdf)]
        sys.exit(compile_pdf(rest[0] if rest else "paper", pdf_out=pdf))
    else:
        sys.exit(check(args[0] if args else "paper"))
