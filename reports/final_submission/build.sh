#!/usr/bin/env bash
# Builds the final report: numbers/tables/figures from results/final/, then the PDF.
# No model inference or download. Needs pdflatex + bibtex (or TECTONIC_BIN).
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python3 build_final.py
mkdir -p build validation
export TEXINPUTS="vendor//:${TEXINPUTS:-}"
export BSTINPUTS="vendor//:${BSTINPUTS:-}"
export SOURCE_DATE_EPOCH=1793404800
stem=Trimax-Final
if [[ -n "${TECTONIC_BIN:-}" ]]; then
  "$TECTONIC_BIN" -Z search-path=vendor --keep-logs --outdir build "$stem.tex" > "validation/build-$stem.txt" 2>&1
else
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build "$stem.tex" > "validation/build-$stem.txt" 2>&1
  bibtex "build/$stem" >> "validation/build-$stem.txt" 2>&1
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build "$stem.tex" >> "validation/build-$stem.txt" 2>&1
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build "$stem.tex" >> "validation/build-$stem.txt" 2>&1
fi
cp -- "build/$stem.pdf" "$stem.pdf"
python3 check_final.py
