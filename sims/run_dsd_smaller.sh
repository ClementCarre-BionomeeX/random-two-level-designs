#!/bin/bash
# DSD at its full size (2(q+1)+1 runs) against random fractions of half and a quarter of that size.
cd "$(dirname "$0")/.."; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cat sims/cells_dsd_smaller.txt | xargs -P ${1:-4} -I{} bash -c 'set -- {}; RFRAC=$1 nice -n 10 python3 sims/compare_designs.py ${@:2}' >> results/compare_dsd_smaller.jsonl 2>>logs/dsd_smaller.err
