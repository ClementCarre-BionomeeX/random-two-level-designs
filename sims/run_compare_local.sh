#!/bin/bash
# Runs all cells of sims/cells_compare.txt with $1 parallel workers; one JSON line per cell in results/compare_designs.jsonl
cd "$(dirname "$0")/.."
W=${1:-4}; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cat sims/cells_compare.txt | xargs -P $W -I{} bash -c 'set -f; a=({}); out=$(nice -n 10 python3 sims/compare_designs.py {} 2>>logs/compare.err); [ -n "$out" ] && echo "$out" >> results/compare_designs.jsonl || echo "{}" >> logs/compare_failed.txt'
