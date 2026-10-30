#!/usr/bin/env bash
# Builds the viva deck from slides/Trimax-Final-Viva.tex; numbers come from generated/
# (run build.sh or python3 build_final.py first). Needs pdflatex + beamer.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/slides"
mkdir -p build
for _ in 1 2; do
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build Trimax-Final-Viva.tex > build/log.txt 2>&1
done
cp build/Trimax-Final-Viva.pdf .
if pdftotext Trimax-Final-Viva.pdf - | grep -q MISSING; then echo "slides contain MISSING results" >&2; exit 1; fi
echo "built slides/Trimax-Final-Viva.pdf"
