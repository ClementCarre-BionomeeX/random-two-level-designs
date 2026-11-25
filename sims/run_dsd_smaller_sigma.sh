#!/bin/bash
# DSD at its full size against random fractions of half and a quarter of that size, with the lasso penalty set from the
# true sigma (SIGMA_KNOWN=1), the reference estimator of the paper. Same 72 cells as run_dsd_smaller.sh.
# Output: results/compare_dsd_smaller_sigma.jsonl
cd "$(dirname "$0")/.."; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 SIGMA_KNOWN=1
cat sims/cells_dsd_smaller.txt | xargs -P ${1:-4} -I{} bash -c 'set -- {}; out=$(RFRAC=$1 nice -n 10 python3 sims/compare_designs.py ${@:2} 2>>logs/dsd_smaller_sigma.err) && [ -n "$out" ] && flock results/.lock sh -c "printf \"%s\n\" \"\$1\" >>results/compare_dsd_smaller_sigma.jsonl" _ "$out" || echo "FAILED {}" >>logs/dsd_smaller_sigma_failed.txt'
date -u +%FT%TZ >logs/end_dsd_smaller_sigma.txt
