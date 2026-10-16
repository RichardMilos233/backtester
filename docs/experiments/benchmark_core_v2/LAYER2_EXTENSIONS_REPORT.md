# Layer 2 Extended Benchmark Report (`benchmark_core_v2`)

Evaluated **200** trajectories across **8** tasks (`GLD_w1, IWM_w2, QQQ_w2, SPY_w1, SPY_w2, SPY_w3, SYNTH_RW_full, TLT_w3`).

## 1. Sample Efficiency & Hypothesis Quality Leaderboard

- **`auc_percentile_val`**: Normalized Area Under the `best_percentile_val_so_far(t)` Curve across steps `t = 1..8`.
- **`step1_percentile_val` -> `final_best_percentile_val`**: Trajectory progression from step 1 to step 8 (`trajectory_val_gain`).
- **`step_improvement_rate`**: Fraction of steps `t >= 2` that improved Search Sharpe over step `t - 1`.
- **`median_improvement_rate`**: Fraction of steps `t >= 2` that beat the running median Search Sharpe of prior steps.
- **`val_generalization_corr`**: Within-trajectory Spearman rank correlation between Search Sharpe and Oracle Val Percentile (`percentile_val`).
- **`hypothesis_alignment`**: Fraction of steps where the agent's natural-language hypothesis aligns with its structured signal family, volatility scaling, and regime filter.

| framework_type   |   tasks |   runs |   selected_percentile_val |   auc_percentile_val |   step1_percentile_val |   final_best_percentile_val |   trajectory_val_gain |   step_improvement_rate |   median_improvement_rate |   val_generalization_corr |   hypothesis_alignment |   auc_ci_lo |   auc_ci_hi |
|:-----------------|--------:|-------:|--------------------------:|---------------------:|-----------------------:|----------------------------:|----------------------:|------------------------:|--------------------------:|--------------------------:|-----------------------:|------------:|------------:|
| R-random         |       8 |     40 |                     0.851 |                0.768 |                  0.515 |                       0.851 |                 0.337 |                   0.482 |                     0.489 |                     0.105 |                  1     |       0.695 |       0.83  |
| R-grid           |       8 |     40 |                     0.861 |                0.732 |                  0.518 |                       0.861 |                 0.343 |                   0.482 |                     0.571 |                    -0.051 |                  1     |       0.625 |       0.832 |
| C-sota           |       8 |     40 |                     0.783 |                0.709 |                  0.479 |                       0.783 |                 0.304 |                   0.489 |                     0.571 |                    -0.036 |                  0.995 |       0.598 |       0.821 |
| C                |       8 |     40 |                     0.811 |                0.708 |                  0.426 |                       0.811 |                 0.385 |                   0.504 |                     0.614 |                    -0.055 |                  0.984 |       0.579 |       0.834 |
| C-corr           |       8 |     40 |                     0.76  |                0.692 |                  0.492 |                       0.76  |                 0.268 |                   0.514 |                     0.586 |                    -0.051 |                  0.968 |       0.542 |       0.837 |

## 2. Best-So-Far Val Percentile Curve Across Budget Steps (`t = 1..8`)

| framework_type   |   best_pv_step_1 |   best_pv_step_2 |   best_pv_step_3 |   best_pv_step_4 |   best_pv_step_5 |   best_pv_step_6 |   best_pv_step_7 |   best_pv_step_8 |   auc_percentile_val |
|:-----------------|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|---------------------:|
| R-random         |            0.515 |            0.671 |            0.789 |            0.811 |            0.83  |            0.832 |            0.842 |            0.851 |                0.768 |
| R-grid           |            0.518 |            0.62  |            0.732 |            0.751 |            0.762 |            0.778 |            0.831 |            0.861 |                0.732 |
| C-sota           |            0.479 |            0.612 |            0.719 |            0.752 |            0.768 |            0.778 |            0.782 |            0.783 |                0.709 |
| C                |            0.426 |            0.574 |            0.7   |            0.757 |            0.791 |            0.799 |            0.807 |            0.811 |                0.708 |
| C-corr           |            0.492 |            0.633 |            0.692 |            0.718 |            0.741 |            0.747 |            0.756 |            0.76  |                0.692 |

## 3. Per-Task Sample Efficiency (`auc_percentile_val`)

| framework_type   |   GLD_w1 |   IWM_w2 |   QQQ_w2 |   SPY_w1 |   SPY_w2 |   SPY_w3 |   SYNTH_RW_full |   TLT_w3 |
|:-----------------|---------:|---------:|---------:|---------:|---------:|---------:|----------------:|---------:|
| C                |    0.509 |    0.911 |    0.528 |    0.427 |    0.925 |    0.64  |           0.823 |    0.901 |
| C-corr           |    0.468 |    0.832 |    0.652 |    0.365 |    0.952 |    0.459 |           0.904 |    0.908 |
| C-sota           |    0.499 |    0.815 |    0.525 |    0.577 |    0.955 |    0.603 |           0.79  |    0.907 |
| R-grid           |    0.551 |    0.678 |    0.75  |    0.948 |    0.488 |    0.878 |           0.828 |    0.733 |
| R-random         |    0.638 |    0.855 |    0.76  |    0.844 |    0.831 |    0.869 |           0.759 |    0.585 |

## 4. Per-Task Search-to-Val Generalization Rank Correlation (`val_generalization_corr`)

| framework_type   |   GLD_w1 |   IWM_w2 |   QQQ_w2 |   SPY_w1 |   SPY_w2 |   SPY_w3 |   SYNTH_RW_full |   TLT_w3 |
|:-----------------|---------:|---------:|---------:|---------:|---------:|---------:|----------------:|---------:|
| C                |    0.13  |   -0.371 |    0.056 |   -0.358 |   -0.281 |   -0.076 |           0.538 |   -0.081 |
| C-corr           |    0.125 |   -0.384 |   -0.174 |    0.015 |   -0.184 |    0.202 |           0.377 |   -0.385 |
| C-sota           |   -0.147 |   -0.483 |   -0.043 |   -0.387 |   -0.093 |    0.408 |           0.532 |   -0.071 |
| R-grid           |    0.143 |   -0.69  |    0.333 |   -0.429 |   -0.333 |   -0.119 |           0.19  |    0.5   |
| R-random         |    0.281 |   -0.162 |    0.048 |    0.195 |   -0.052 |    0.381 |          -0.129 |    0.277 |
