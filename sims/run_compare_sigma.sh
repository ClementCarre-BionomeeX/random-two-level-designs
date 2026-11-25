#!/bin/bash
# All structured designs vs random (s = 11 and 22, same 144 cells as compare_designs), with the lasso penalty set from the true sigma (SIGMA_KNOWN=1), as in the run-count sweeps.
# Checks whether the s = 11 and 22 comparisons depends on the scaled-lasso noise estimate. Output: results/compare_designs_sigma.jsonl
cd "$(dirname "$0")/.."; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 SIGMA_KNOWN=1
cat sims/cells_compare_sigma.txt | xargs -P ${1:-4} -I{} bash -c 'set -- {}; out=$(RFRAC=$1 nice -n 10 python3 sims/compare_designs.py ${@:2} 2>>logs/compare_sigma.err) && [ -n "$out" ] && flock results/.lock sh -c "printf \"%s\n\" \"\$1\" >>results/compare_designs_sigma.jsonl" _ "$out" || echo "FAILED {}" >>logs/compare_sigma_failed.txt'
date -u +%FT%TZ >logs/end_compare_sigma.txt
