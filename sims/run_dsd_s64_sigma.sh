#!/bin/bash
# DSD vs random at p = 1000, s = 64, with the lasso penalty set from the true sigma (SIGMA_KNOWN=1), as in the run-count sweeps.
# Checks whether the s = 64 comparison depends on the scaled-lasso noise estimate. Output: results/compare_dsd_s64_sigma.jsonl
cd "$(dirname "$0")/.."; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 SIGMA_KNOWN=1
cat sims/cells_dsd_s64_sigma.txt | xargs -P ${1:-4} -I{} bash -c 'set -- {}; out=$(RFRAC=$1 nice -n 10 python3 sims/compare_designs.py ${@:2} 2>>logs/dsd_s64_sigma.err) && [ -n "$out" ] && flock results/.lock sh -c "printf \"%s\n\" \"\$1\" >>results/compare_dsd_s64_sigma.jsonl" _ "$out" || echo "FAILED {}" >>logs/dsd_s64_sigma_failed.txt'
date -u +%FT%TZ >logs/end_dsd_s64_sigma.txt
