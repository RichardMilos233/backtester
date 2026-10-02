# Benchmark report: `docs/experiments/benchmark_core`

200 successful runs, 0 failed.

## Leaderboard

| framework_type   |   tasks |   runs |   percentile_val |   regret_test |   regret_vs_val_oracle |   test_sharpe |   budget_utilization |   pv_ci_lo |   pv_ci_hi |   parse_fail |   rejections |   tokens_per_run |   sec_per_run |   win_vs_random |   n_pairs |
|:-----------------|--------:|-------:|-----------------:|--------------:|-----------------------:|--------------:|---------------------:|-----------:|-----------:|-------------:|-------------:|-----------------:|--------------:|----------------:|----------:|
| R-random         |       8 |     40 |            0.891 |         0.903 |                 -0.016 |         0.195 |                1     |      0.874 |      0.912 |            0 |            0 |             0    |         0.299 |         nan     |       nan |
| R-grid           |       8 |     40 |            0.86  |         1.067 |                  0.148 |         0.031 |                1     |      0.745 |      0.948 |            0 |            0 |             0    |         0.276 |           0.425 |        40 |
| C-corr           |       8 |     40 |            0.802 |         1.002 |                  0.083 |         0.096 |                0.8   |      0.644 |      0.936 |            0 |           54 |          5028.43 |        18.95  |           0.45  |        40 |
| C-sota           |       8 |     40 |            0.76  |         1.028 |                  0.108 |         0.07  |                0.775 |      0.572 |      0.923 |            0 |           15 |          4251.02 |        15.024 |           0.45  |        40 |
| C                |       8 |     40 |            0.745 |         0.946 |                  0.026 |         0.152 |                0.766 |      0.56  |      0.903 |            0 |           11 |          3784.05 |        14.183 |           0.425 |        40 |

## Per-task percentile_val

| framework_type   |   GLD_w1 |   IWM_w2 |   QQQ_w2 |   SPY_w1 |   SPY_w2 |   SPY_w3 |   SYNTH_RW_full |   TLT_w3 |
|:-----------------|---------:|---------:|---------:|---------:|---------:|---------:|----------------:|---------:|
| C                |    0.917 |    0.852 |    0.973 |    0.432 |    1     |    0.258 |           0.779 |    0.75  |
| C-corr           |    0.955 |    0.943 |    0.974 |    0.57  |    1     |    0.355 |           0.762 |    0.862 |
| C-sota           |    0.944 |    0.952 |    0.98  |    0.466 |    1     |    0.233 |           0.717 |    0.785 |
| R-grid           |    0.495 |    0.818 |    0.991 |    0.869 |    0.977 |    0.793 |           0.969 |    0.965 |
| R-random         |    0.901 |    0.891 |    0.894 |    0.886 |    0.883 |    0.865 |           0.955 |    0.857 |

## Notes (2026-10-02, deepseek-chat, budget 8, 5 seeds, core suite)

Two confounds in this first run; both favour the baselines and must be fixed before the ranking is read as a result:

1. **Early stopping drains the LLM budget.** `patience=3, min_improvement=0.01` stops most LLM runs at 4–7 backtests (`budget_utilization` 0.77–0.80) while R-random / R-grid always use all 8. Fix: disable early stop in benchmark mode so every method spends the same budget.
2. **The agents never see vol scaling.** The Researcher prompt only exposes `momentum` and `macd`; 0 % of LLM picks use `use_vol_scaling`, whereas 82 % of R-random winners do, and the vol-scaled sublattice has a higher Val-Sharpe median (0.69 vs 0.50 on SPY_full). Fix: either expose vol_target/max_leverage in the prompt (new prompt version) or restrict baselines to the no-vol lattice.

What is already robust: zero failed runs, zero parse failures across 120 LLM runs; the three C variants are statistically indistinguishable (percentile_val CIs overlap heavily); on `SYNTH_RW_full` every method lands on negative Val Sharpe, so nothing is manufacturing alpha from noise.
