#!/usr/bin/env bash
# Build the AIO2026 Module 4 lecture note.
# Run from this directory (vipythonhighlight.sty / tvietlistings.sty are local).
# resolves against the CWD, not against main.tex's location.
set -e
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
