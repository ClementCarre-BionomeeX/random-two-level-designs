#!/bin/bash
# Reviewer controls, run after the design comparison. Outputs in results/controls_*.
cd "$(dirname "$0")/.."; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# 1) classical ranking ablation on every regime cell (pooled |t| instead of stage-2 effects first); classical only,
#    same seeds as the original grid, so the comparison with results/regime*.jsonl is paired.
cat sims/cells.txt sims/cells_rest.txt 2>/dev/null | sort -u | grep -v '^$' | xargs -P 2 -I{} bash -c 'RANK=pooled SKIP_RANDOM=1 nice -n 10 python3 sims/regime.py {}' >> results/controls_rank_pooled.jsonl
sed 's/ 20000$//' sims/cells_1000.txt | xargs -P 2 -I{} bash -c 'RANK=pooled SKIP_RANDOM=1 nice -n 10 python3 sims/regime_fast.py {}' >> results/controls_rank_pooled.jsonl
# 2) screening size K at p=1000: K=20000 vs K=100000, same seeds, 50 replicates
for c in "mixed mid 0.25 nofold" "mixed high 0.25 nofold" "mixed mid 0.25 fold"; do
  for K in 20000 100000; do echo "$c 50 $K"; done
done | xargs -P 2 -I{} bash -c 'nice -n 10 python3 sims/regime_fast.py 1000 {}' >> results/controls_K.jsonl
# 3) family-wise error of Section 9, 1000 replicates per m, active truth and global null
cd sims && nice -n 10 python3 sim_fwer.py 1000 4 >> ../results/controls_fwer.jsonl
