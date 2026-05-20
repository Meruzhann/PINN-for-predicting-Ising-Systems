#!/usr/bin/env bash
# Build main.pdf with all the passes LaTeX needs for refs + floats to settle.
set -euo pipefail
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main || true
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
echo "Done: main.pdf"
