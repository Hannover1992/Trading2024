#!/bin/bash

# Stoppe das Skript, wenn ein Befehl fehlschlägt
set -e

# LaTeX-Kompilierung
pdflatex 00thesis.tex 
biber 00thesis
pdflatex 00thesis.tex
pdflatex 00thesis.tex

# Aufräumen
# Vorsicht: Dieser Befehl löscht alle Dateien mit nicht erwünschten Endungen!
find . -type f \
! \( -name "*.png" -o -name "*.bib" -o -name "*.tex" -o -name "*.pdf" -o -name "*.sh" \) \
-exec rm {} +
