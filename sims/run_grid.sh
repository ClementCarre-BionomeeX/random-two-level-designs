#!/bin/sh
# Run every cell of sims/cells.txt in parallel. Usage (from repo root): nohup sims/run_grid.sh [jobs] &
J=${1:-24}
mkdir -p results logs
date -u +%FT%TZ >logs/start.txt
xargs -P "$J" -L 1 nice -n 19 sims/run_cell.sh <sims/cells.txt
date -u +%FT%TZ >logs/end.txt
