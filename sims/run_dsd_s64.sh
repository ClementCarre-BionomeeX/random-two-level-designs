#!/bin/bash
# DSD with s = 64 (32 mains + 32 interactions) against random fractions of the same size (rfrac 1) and of half and a quarter of it.
# Same script and output format as run_dsd_smaller.sh; one JSON line per cell in results/compare_dsd_s64.jsonl.
cd "$(dirname "$0")/.."; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
date -u +%FT%TZ >logs/start_dsd_s64.txt
cat sims/cells_dsd_s64.txt | xargs -P ${1:-4} -I{} bash -c 'set -- {}; out=$(RFRAC=$1 nice -n 10 python3 sims/compare_designs.py ${@:2} 2>>logs/dsd_s64.err) && [ -n "$out" ] && flock results/.lock sh -c "printf \"%s\n\" \"\$1\" >>results/compare_dsd_s64.jsonl" _ "$out" || echo "FAILED {}" >>logs/dsd_s64_failed.txt'
date -u +%FT%TZ >logs/end_dsd_s64.txt
