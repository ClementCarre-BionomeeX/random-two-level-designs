#!/bin/sh
# Run one grid cell and append its JSON line to ${OUT:-results/regime.jsonl} (flock keeps lines whole).
# Usage (from repo root): [SCRIPT=regime_big.py OUT=results/regime_1000.jsonl] sims/run_cell_big.sh p scen s theta budget reps
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
tag=$(echo "$@" | tr ' ' '_')
out=$(python3 sims/${SCRIPT:-regime.py} "$@" 2>>"logs/$tag.err") || { echo "FAILED $*" >>logs/failed.txt; exit 0; }
flock results/.lock sh -c 'printf "%s\n" "$1" >>${OUT:-results/regime.jsonl}' _ "$out"
[ -s "logs/$tag.err" ] || rm -f "logs/$tag.err"
