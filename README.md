# pam-doe

Simulations comparing two ways of screening for sparse two-factor interactions among `p` two-level factors, at equal run budget. Simulated data only.

## Strategies

- **Random fraction + debiased lasso.** `M` runs drawn uniformly from {-1, +1}^p. Lasso on all main effects and pairwise interactions (λ = √(2 log N / M)), followed by a one-step debiasing correction; effects are ranked by |debiased estimate|.
- **Classical sequential strategy.** Stage 1: a regular two-level fraction (columns of a Hadamard matrix of size `nreg` = next power of 2 above `p`), optionally folded over (`fold`, doubling the runs). Main effects are estimated by OLS. Stage 2: a 2^5 full factorial on the 5 factors with the largest stage-1 |t|, the other factors held at a random fixed level. Ranking: stage-2 effects (5 mains + 10 interactions) by |t|, then the remaining main effects by stage-1 |t|.

Both strategies use the same total budget `M = nreg × (2 if fold else 1) + 32`.

## Grid

`sims/cells.txt`, 216 cells, 200 replicates each:

| parameter | values |
|---|---|
| `p` (factors) | 30, 100, 300 |
| scenario | `both`, `one`, `neither`, `mixed` |
| sparsity `s` | low (3 mains, 3 interactions), mid (5, 6), high (10, 12) |
| effect size θ (noise σ = 1) | 0.25, 0.5, 1.0 |
| budget | `nofold`, `fold` |

The scenario sets which factors the true interactions involve: both factors carry a main effect (`both`), exactly one does (`one`), neither does (`neither`), or a cycle of the three (`mixed`). All true effects have magnitude θ and a random sign.

## Metric

Recall of the true effects within the top `k` of each strategy's ranking, where `k` is the number of true effects. It is reported separately for main effects and for each interaction type (`both`, `one`, `neither`).

## Reproducibility

Each replicate is seeded with `crc32("p|scen|s|theta|budget|rep")`, so every cell is deterministic. Reference check:

```
python3 sims/regime.py 30 mixed low 0.5 fold 2
{"p": 30, "scen": "mixed", "s": "low", "theta": 0.5, "budget": "fold", "runs": 96, "reps": 2, "sec": 0.0, "classical": {"main": 0.833, "both": 1.0, "one": 0.0, "neither": 0.0}, "random": {"main": 0.667, "both": 0.5, "one": 0.5, "neither": 0.5}}
```

Obtained with numpy 2.4.6, scipy 1.17.1, scikit-learn 1.9.0.

## Running

```
nohup sims/run_grid.sh 24 > logs/grid.out 2>&1 &
```

Each cell appends one JSON line to `results/regime.jsonl`; lines are in completion order, not in `cells.txt` order. Failures go to `logs/failed.txt`, stderr to `logs/<cell>.err`.

## Large p (p = 1000)

`sims/regime_fast.py` runs the same truth, seeds, classical strategy and metric for p = 1000 (N = 500 500 candidate effects) without building the pair matrix. The random strategy keeps all main effects plus the K = 20 000 pairs with the largest marginal |Φⱼᵀy|, fits the lasso on those columns with the same penalty, then computes the debiased statistic for all N effects. The inner products of all pair columns with a vector v are read off `Xᵀ diag(v) X − (XᵀX/M)·Σv` (two p × p products). `sims/regime_big.py` is the blockwise version it was checked against. On p = 300 mixed mid 0.25 nofold, 30 replicates, K = 2000, both give the same numbers as `regime.py` (`results/validation_big.txt`).

Grid: `sims/cells_1000.txt` (72 cells, 200 replicates), results in `results/regime_1000.jsonl`.

```
SCRIPT=regime_fast.py OUT=results/regime_1000.jsonl CELLS=sims/cells_1000.txt TAG=_1000 nohup sims/run_grid_big.sh 4 &
```

## Run-count sweep

`sims/sweep_runs.py` measures how many runs the random fraction needs as a function of the number of active effects `s`, at fixed `p`. There is no classical strategy here: the budget `M` is the swept variable. The truth has `s // 2` main effects and `s − s // 2` interactions, cycling through the three types (both, one or neither of their factors with a main effect), all with absolute size θ (σ = 1) and random signs. For each `M` in 32, 48, …, 4096, the script reports the recall in the top `s`, overall and by type, plus the fraction of replicates where all `s` effects are found. A chain (p, s, θ) stops after two consecutive `M` with recall ≥ 0.99.

Grid: p ∈ {100, 300, 1000} × s ∈ {4, 8, 16, 32, 64} × θ ∈ {0.25, 0.5, 1}, 100 replicates per point, 45 chains (`sims/chains_sweep.txt`, the slowest first). Output: `results/sweep_runs.jsonl`, one line per (p, s, θ, M).

```sh
nohup sims/run_sweep.sh 4 > logs/sweep.out 2>&1 &
nohup sims/push_loop_sweep.sh > logs/push_sweep.out 2>&1 &
```

### Second sweep

`sims/sweep_runs2.py` repeats the sweep with the same strategy and truth, and removes four weaknesses of the first one:

1. Detection without knowing `s`: besides the top `s`, effects are called by Benjamini–Hochberg at level Q = 0.05 over all N effects, on the debiased z = d / se with se = σ̂ √M / (M − df) and σ̂² = ‖r‖² / (M − df). Recall and realised false discovery proportion are reported (`bh_recall`, `bh_all_found`, `bh_fdr`, `bh_ndisc`, `bh_by_type`).
2. "All found": fraction of replicates where every active effect is found, in the top `s` (`all_found`) and by Benjamini–Hochberg (`bh_all_found`).
3. Effect sizes: `equal` (|effect| = θ) or `spread` (|effect| = θ·exp(U(−ln 2, ln 2)), from θ/2 to 2θ).
4. Finer grid: M = round(32·2^(i/4)), ratio 1.19. A point is run with 20 replicates, then with the full number of replicates once `bh_all_found` reaches 0.3. A chain stops after two consecutive M meeting the four targets: recall ≥ 0.99, all_found ≥ 0.95, bh_recall ≥ 0.99, bh_all_found ≥ 0.95.

Usage: `python3 sims/sweep_runs2.py p s theta sizes reps [K]`, one JSON line per M on stdout.

Grid: p ∈ {100, 300, 1000} × sizes ∈ {equal, spread} × s ∈ {4, 8, 16, 32, 64} × θ ∈ {0.25, 0.5, 1} with 200 replicates near the threshold, plus 11 chains at p = 3000 (equal sizes, 100 replicates) kept out of the fit: 101 chains (`sims/chains_sweep2.txt`). Output: `results/sweep_runs2.jsonl`.

The launchers take the script, output, chain list and log tag from the environment (`SCRIPT`, `OUT`, `CHAINS`, `TAG`; defaults are those of the first sweep). The push loop runs `git pull --rebase` before each push, so commits pushed to `main` during a run do not block it.

```sh
SCRIPT=sweep_runs2.py OUT=results/sweep_runs2.jsonl CHAINS=sims/chains_sweep2.txt TAG=2 nohup sims/run_sweep.sh 4 > logs/sweep2.out 2>&1 &
OUT=results/sweep_runs2.jsonl TAG=2 nohup sims/push_loop_sweep.sh > logs/push_sweep2.out 2>&1 &
```

Fit, figures and table: `python3 sims/fig_sweep2.py results/sweep_runs2.jsonl figures` writes `figures/sweep2_fit.json`, `figures/sweep2_equal.pdf`, `figures/sweep2_spread.pdf` and `figures/sweep2_table.tex`.

### DSD at s = 64

`sims/run_dsd_s64.sh` runs `sims/compare_designs.py` on the cells of `sims/cells_dsd_s64.txt`: DSD with 32 main effects and 32 interactions (truths `equal` and `heredity`), p ∈ {100, 300, 1000}, θ ∈ {0.25, 0.5, 1}, 100 replicates, against random fractions of the same size, half and a quarter of it (`RFRAC` = 1, 0.5, 0.25). Output: `results/compare_dsd_s64.jsonl`, same format as `results/compare_dsd_smaller.jsonl`.

```sh
nohup sims/run_dsd_s64.sh 4 > logs/dsd_s64.out 2>&1 &
```

## License

Copyright (C) 2026 Clément Carré, BionomeeX SAS.

This project is released under the GNU Lesser General Public License, version 3 (LGPL-3.0-only); see `COPYING.LESSER`, which supplements the GNU General Public License v3 in `COPYING`.
