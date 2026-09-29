#!/usr/bin/env bash
# Builds the final report from generated/results.tex (run build_final_results.py first).
# No model inference, downloads or package installs. Styles come from ../mid_submission/vendor.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
[[ -f generated/results.tex ]] || { echo "Run: python3 reports/final/build_final_results.py" >&2; exit 1; }
mkdir -p build validation
VENDOR=../mid_submission/vendor
export TEXINPUTS="$VENDOR//:${TEXINPUTS:-}" BSTINPUTS="$VENDOR//:${BSTINPUTS:-}"
export SOURCE_DATE_EPOCH=1790553600
for stem in Trimax-Final-Report Trimax-Final-Report-review; do
  if [[ -n "${TECTONIC_BIN:-}" ]]; then
    "$TECTONIC_BIN" -Z search-path="$VENDOR" --keep-logs --keep-intermediates \
      --outdir build "$stem.tex" > "validation/build-$stem.txt" 2>&1
  else
    for c in pdflatex bibtex; do command -v $c >/dev/null || { echo "Missing $c; set TECTONIC_BIN." >&2; exit 1; }; done
    pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build "$stem.tex" > "validation/build-$stem.txt" 2>&1
    bibtex "build/$stem" >> "validation/build-$stem.txt" 2>&1
    pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build "$stem.tex" >> "validation/build-$stem.txt" 2>&1
    pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build "$stem.tex" >> "validation/build-$stem.txt" 2>&1
  fi
  cp -- "build/$stem.pdf" "$stem.pdf"
done
# Overleaf/source package: flat, with the official style files at the root.
rm -rf build/src && mkdir -p build/src/generated
cp Trimax-Final-Report.tex Trimax-Final-Report-review.tex paper-preamble.tex final-body.tex references.bib \
   "$VENDOR/acl.sty" "$VENDOR/acl_natbib.bst" build/src/
cp generated/results.tex generated/depth_margin.pdf build/src/generated/
rm -f Trimax-Final-Report-source.zip
(cd build/src && zip -qr ../../Trimax-Final-Report-source.zip .)
printf '%s\n' 'Built named + review PDFs and source zip. Run check_report.py next.'
