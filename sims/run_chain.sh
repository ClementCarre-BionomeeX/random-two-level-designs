#!/bin/sh
# Run one sweep chain and append each JSON line to ${OUT:-results/sweep_runs.jsonl} as it comes (flock keeps lines whole).
# Usage (from repo root): [SCRIPT=sweep_runs2.py OUT=results/sweep_runs2.jsonl] sims/run_chain.sh p s theta [sizes] reps
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
tag=$(echo "$@" | tr ' ' '_')
python3 sims/${SCRIPT:-sweep_runs.py} "$@" 2>>"logs/sweep${TAG}_$tag.err" | while IFS= read -r line; do
  flock results/.lock sh -c 'printf "%s\n" "$1" >>${OUT:-results/sweep_runs.jsonl}' _ "$line"
done
[ -s "logs/sweep${TAG}_$tag.err" ] && echo "FAILED $*" >>logs/sweep${TAG}_failed.txt || rm -f "logs/sweep${TAG}_$tag.err"
