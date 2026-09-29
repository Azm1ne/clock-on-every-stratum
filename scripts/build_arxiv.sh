#!/usr/bin/env bash
# Build everything the arXiv submission needs into "for arxiv/".
#
#   bash scripts/build_arxiv.sh           # refuses while \addr [affiliation] is a placeholder
#   bash scripts/build_arxiv.sh --draft   # builds anyway, for checking the package
#
# for arxiv/arxiv_source.tar.gz  the source arXiv compiles: main.tex (tmlr [preprint], comment
#                                lines stripped), sections/, figures/ (paths rewritten from
#                                ../figures), tmlr.sty, fancyhdr.sty and main.bbl (arXiv runs no
#                                BibTeX; the .bbl must carry the main file's name)
# for arxiv/main.pdf             compiled FROM THAT TARBALL in a fresh directory, no BibTeX
# for arxiv/abstract.txt         the metadata abstract (tracked; <= 1,920 chars, arXiv's limit)
# for arxiv/SUBMIT.md            the form, field by field (tracked)
#
# Refuses a dirty tree: a submitted source must map to exactly one commit.
set -euo pipefail
cd "$(dirname "$0")/.."
DRAFT="${1:-}"
[ -z "$(git status --porcelain)" ] || { echo "refusing: working tree dirty -- commit first" >&2; exit 1; }
OUT="for arxiv"
S="$(mktemp -d)"; V="$(mktemp -d)"; trap 'rm -rf "$S" "$V"' EXIT

.venv/bin/python paper/check_tex.py
if grep -q '\[affiliation\]' paper/main.tex && [ "$DRAFT" != "--draft" ]; then
  echo "refusing: paper/main.tex still has the [affiliation] placeholder (use --draft to check the package)" >&2; exit 1
fi

# Stage. Figures are gitignored and regenerated; refuse any older than the code that draws them.
mkdir -p "$S/sections" "$S/figures"
cp paper/main.tex paper/tmlr.sty paper/fancyhdr.sty paper/tmlr.bst paper/refs.bib "$S/"
cp paper/sections/*.tex "$S/sections/"
T_CODE=$(git log -1 --format=%ct -- src/viz/plots.py scripts/render_all.py results)
for f in $(grep -ho 'includegraphics\[[^]]*\]{\.\./figures/[^}]*}' paper/sections/*.tex | sed 's/.*{\.\.\///; s/}$//' | sort -u); do
  [ -f "$f" ] || { echo "missing $f -- run scripts/render_all.py" >&2; exit 1; }
  [ "$(stat -c %Y "$f")" -ge "$T_CODE" ] || { echo "stale $f -- older than the last plots/results commit" >&2; exit 1; }
  cp "$f" "$S/figures/"
done

.venv/bin/python - "$S" <<'PY'
import pathlib, re, sys
S = pathlib.Path(sys.argv[1])
for p in [S / "main.tex", *sorted((S / "sections").glob("*.tex"))]:
    out = []
    for i, line in enumerate(p.read_text().splitlines(keepends=True), 1):
        if line.lstrip().startswith("%"):
            continue            # a whole-line comment: TeX drops it with its newline, so deleting it is exact
        # A trailing comment would ship an internal note. check_tex bans them; assert it here too.
        # A bare "%" ending a line (glue) is harmless and carries no text.
        m = re.search(r"(?<!\\)%(.*)$", line.rstrip("\n"))
        assert not (m and m.group(1).strip()), f"{p.name}:{i}: trailing comment would ship: {line.strip()}"
        out.append(line)
    t = "".join(out).replace("{../figures/", "{figures/")
    if p.name == "main.tex":
        assert t.count("\\usepackage{tmlr}") == 1
        t = t.replace("\\usepackage{tmlr}", "\\usepackage[preprint]{tmlr}")
    assert "../" not in t, f"{p.name}: a ../ path survives"
    p.write_text(t)
print("  staged: comments stripped, [preprint], figure paths rewritten")
PY

( cd "$S" && pdflatex -interaction=nonstopmode -halt-on-error main >/dev/null && bibtex main >/dev/null \
  && pdflatex -interaction=nonstopmode -halt-on-error main >/dev/null )
mkdir -p "$OUT"
tar -C "$S" -czf "$OUT/arxiv_source.tar.gz" main.tex main.bbl tmlr.sty fancyhdr.sty sections figures

# Verify: compile the TARBALL, as arXiv will, with no BibTeX and no repo around it.
tar -C "$V" -xzf "$OUT/arxiv_source.tar.gz"
( cd "$V" && for _ in 1 2 3; do pdflatex -interaction=nonstopmode -halt-on-error main >/dev/null; done )
! grep -E "undefined|Citation .* undefined|There were undefined|Rerun to get" "$V/main.log" \
  || { echo "FAIL: the tarball does not compile clean" >&2; exit 1; }
! grep -rlE "^\s*%" "$V/main.tex" "$V/sections" >/dev/null || { echo "FAIL: a comment line shipped" >&2; exit 1; }
cp "$V/main.pdf" "$OUT/main.pdf"

.venv/bin/python - <<'PY'
import re, pathlib
a = " ".join(pathlib.Path("for arxiv/abstract.txt").read_text().split())
assert len(a) <= 1920, f"arXiv abstract is {len(a)} chars (limit 1,920)"
assert not re.search(r"~|\\,|\\ |\\(?:emph|textbf|textit|em|it)\b", a), "a TeX command arXiv rejects"
# It abridges the paper's abstract and says nothing new: every number it prints is printed there.
p = pathlib.Path("paper/sections/00-abstract.tex").read_text().replace("{,}", ",")
num = lambda s: set(re.findall(r"\d[\d,]*(?:\.\d+)?", s))
extra = num(a) - num(p)
assert not extra, f"numbers in the arXiv abstract that the paper's abstract does not print: {extra}"
print(f"  for arxiv/abstract.txt: {len(a)} chars, every number from the paper's abstract")
PY

echo "for arxiv/ built from $(git rev-parse --short HEAD): $(pdfinfo "$OUT/main.pdf" | awk '/^Pages/{print $2}') pp," \
     "source $(du -h "$OUT/arxiv_source.tar.gz" | cut -f1)$([ "$DRAFT" = "--draft" ] && echo ' (DRAFT: affiliation is a placeholder)')"
