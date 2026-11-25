#!/bin/sh
# Run every cell of $CELLS in parallel. Usage (from repo root): SCRIPT=regime_big.py OUT=results/regime_1000.jsonl CELLS=sims/cells_1000.txt TAG=_1000 nohup sims/run_grid_big.sh [jobs] &
J=${1:-24}
mkdir -p results logs
date -u +%FT%TZ >logs/start${TAG}.txt
xargs -P "$J" -L 1 nice -n 19 sims/run_cell_big.sh <"${CELLS:-sims/cells.txt}"
date -u +%FT%TZ >logs/end${TAG}.txt
