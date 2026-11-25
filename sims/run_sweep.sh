#!/bin/sh
# Run every chain of ${CHAINS:-sims/chains_sweep.txt}, 4 at a time in nice 10.
# Usage (from repo root): [SCRIPT=sweep_runs2.py OUT=results/sweep_runs2.jsonl CHAINS=sims/chains_sweep2.txt TAG=2] nohup sims/run_sweep.sh [jobs] &
J=${1:-4}
mkdir -p results logs
date -u +%FT%TZ >logs/start_sweep${TAG}.txt
xargs -P "$J" -L 1 nice -n 10 sims/run_chain.sh <"${CHAINS:-sims/chains_sweep.txt}"
date -u +%FT%TZ >logs/end_sweep${TAG}.txt
