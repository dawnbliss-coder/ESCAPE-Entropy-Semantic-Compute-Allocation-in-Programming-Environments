#!/usr/bin/env bash
# Builds only the report. No model inference, corpus download, or package install.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
mkdir -p build validation
export TEXINPUTS="vendor//:${TEXINPUTS:-}"
export BSTINPUTS="vendor//:${BSTINPUTS:-}"
# Fixed source date for reproducible metadata where supported by the TeX engine.
export SOURCE_DATE_EPOCH=1790553600
if [[ -n "${TECTONIC_BIN:-}" ]]; then
  for stem in Trimax-Mid-Submission Trimax-Mid-Submission-review; do
    "$TECTONIC_BIN" -Z search-path=vendor --keep-logs --keep-intermediates \
      --outdir build "$stem.tex" > "validation/build-$stem.txt" 2>&1
    cp -- "build/$stem.pdf" "$stem.pdf"
  done
else
  for command in pdflatex bibtex; do
    command -v "$command" >/dev/null || { echo "Missing $command; use a complete TeX installation or set TECTONIC_BIN." >&2; exit 1; }
  done
  for stem in Trimax-Mid-Submission Trimax-Mid-Submission-review; do
    pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build "$stem.tex" > "validation/build-$stem.txt" 2>&1
    bibtex "build/$stem" >> "validation/build-$stem.txt" 2>&1
    pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build "$stem.tex" >> "validation/build-$stem.txt" 2>&1
    pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build "$stem.tex" >> "validation/build-$stem.txt" 2>&1
    cp -- "build/$stem.pdf" "$stem.pdf"
  done
fi
printf '%s\n' 'Built named and anonymous review PDFs. Run check_report.py next.'
