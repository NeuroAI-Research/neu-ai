#!/bin/sh
# Build every diagram docs/**/imgs/<name>.tex into <name>.svg next to it:
#   xelatex -no-pdf -> .xdv -> dvisvgm -> .svg (text stays real, selectable <text>, fonts embedded as woff2).
# --zoom=1.2: TeX draws text at 10pt (13px); scale the figure so it shows at 12pt = 16px, the page body size.
# Needs a TeX distribution with xelatex + dvisvgm (TinyTeX: pgf standalone xetex fontspec xecjk dvisvgm amsmath).
# Usage (from docs/):  sh tex/tex2svg.sh [file.tex ...]     (no arguments: all diagrams)
cd "$(dirname "$0")/.." || exit 1
TEXBIN="$HOME/Library/TinyTeX/bin/universal-darwin"
command -v xelatex >/dev/null || PATH="$TEXBIN:$PATH"
export TEXINPUTS="$PWD/tex//:"   # finds neuroai-fig.sty
BUILD="${TMPDIR:-/tmp}/neuroai-tex"
mkdir -p "$BUILD"
[ $# -gt 0 ] || set -- $(find docs -path '*/imgs/*.tex')
status=0
tex() {  # pgf must use its dvisvgm driver (set before \documentclass), or arrows and boxes are lost
  xelatex -no-pdf -interaction=nonstopmode -halt-on-error -output-directory="$BUILD" -jobname="$(basename "$1" .tex)" \
    "\\def\\pgfsysdriver{pgfsys-dvisvgm.def}\\input{$1}" >/dev/null
}
for f in "$@"; do
  n=$(basename "$f" .tex)
  # run twice: the first run measures the content width \W (written to .aux), the second uses it
  if tex "$f" && tex "$f"; then
    dvisvgm --zoom=1.2 --font-format=woff2 --exact-bbox -o "${f%.tex}.svg" "$BUILD/$n.xdv" 2>&1 | grep -iE 'warning|error'
    # TeX spaces are gaps, not characters: dvisvgm starts each word as a <tspan x=...>.
    # Put a space between such words so copied text reads "a b", not "ab"; positions are unchanged.
    perl -pi -e "s/<\\/tspan><tspan([^>]*? x=')/<\\/tspan> <tspan\$1/g" "${f%.tex}.svg"
    echo "built ${f%.tex}.svg"
  else
    echo "FAILED $f:"; grep -A3 '^!' "$BUILD/$n.log"; status=1
  fi
done
exit $status
