# Agent Infra Benchmark 协议

把"换模型 / 换架构"的实验改成可重复、可横向比较的 benchmark。
本文写死评分规则；跑 Test 之前不得更改。

## 一次运行

`(task_id, model, framework_type, seed)` 在固定预算 `max_backtests` 下搜索一次，
产出一行 `summary.csv` 和一条 `runs.jsonl`。

- **task**：`agent/tasks.py`。一个标的 + 一个日期范围，内部仍按 60/20/20 切 Train/Val/Test。
- **预算**：比的是送进回测的 spec 数，不是 LLM 调用数。被 Reviewer 打回不计。
- **选优**：只看 Val 夏普。Test 只对最终胜出的 spec 评一次。

## 任务集

| 套件 | 任务 | 用途 |
| :-- | :-- | :-- |
| `smoke` | `SPY_full` | 回归、调试；等于 Phase 1/2 的原始设置 |
| `core` | SPY w1/w2/w3, QQQ_w2, IWM_w2, TLT_w3, GLD_w1, SYNTH_RW_full | 日常比较，8 个任务 |
| `full` | 6 个 ETF × 3 窗口 + SYNTH_RW | 定稿 |

滚动窗口让 Test 落在不同行情：w1 ≈ 2020-07..2021-11（疫后单边涨），
w2 ≈ 2022-02..2023-06（熊市 + 震荡），w3 ≈ 2024-05..最新。
`SYNTH_RW` 是零漂移随机游走，任何策略都不该有稳定 alpha，用来测 agent 会不会追噪声。

数据：`scripts/fetch_ohlcv.py <TICKER>` 拉 Yahoo 复权日线；`scripts/make_synthetic.py` 生成合成序列。

## Oracle 与归一化分数

每个 task 先用 `scripts/build_oracle.py --task <id>` 穷举 `agent/space.py` 里的全部
12,020 个 spec（259 个 momentum 窗口 + 2,145 个 MACD 格点，各 × 5 档波动率缩放）在
Val 和 Test 上的指标。有了全空间分布，agent 的选择就能换算成与行情无关的分数：

| 列 | 含义 |
| :-- | :-- |
| `percentile_val` | agent 选中的 spec 在全空间 Val 夏普中的分位数。**主指标**：衡量搜索质量 |
| `regret_test` | 全空间 Test 最优夏普 − 选中 spec 的 Test 夏普 |
| `regret_vs_val_oracle` | "全知 Val 选择器"的 Test 夏普 − 选中 spec 的 Test 夏普。负值表示运气好过天花板 |
| `percentile_test` | 选中 spec 在全空间 Test 夏普中的分位数 |
| `on_lattice` | spec 是否落在格点上；不在则直接回测取值 |

**为什么主指标是 Val 分位而不是 Test 夏普。** 在 `SPY_full` 上，全空间 Val 夏普与
Test 夏普的 Spearman 相关是 −0.13。Val 上最好的 spec 在 Test 上只排中游。
这意味着 Test 夏普高低基本不反映搜索能力。agent 能控制的只有"在给定反馈下找到 Val 上多好的 spec"。

## 必须一起跑的非 LLM 基线

| 代号 | 做法 |
| :-- | :-- |
| `R-random` | 同预算，从全格点均匀抽样 |
| `R-grid` | 同预算，格点上等距取点，与种子无关 |

两者 `model="none"`，不发任何请求。LLM 架构要进榜，先要在 `percentile_val` 上
稳定赢过 `R-random`。

## 过程指标

`budget_utilization`（= actual_backtests / max_backtests，低于 1 说明架构在浪费预算或有 bug）、
`parse_failures`、`reviewer_rejections`、`llm_calls`、`llm_total_tokens`、`llm_retries`、
`llm_latency_seconds`、`elapsed_seconds`。`runs.jsonl` 里每条记录带 `trajectory`：
每步回测的 spec 和 best-so-far，用来画样本效率曲线。

## 汇总规则

1. 同一 `(task, model, infra)` 先对种子取均值。
2. 再跨 task 取均值。每个 task 权重相同。
3. 排名用 `percentile_val`；并列报 `regret_test`、对 `R-random` 的胜率、`budget_utilization`、`parse_failures`、token 成本。
4. 5 个种子 × 8 个 task 只能看方向；差距小于 1 个种子标准差的不写成结论。
5. 任何 prompt、温度、反馈字段、选优指标的改动都是新版本，不与旧版本同表。

## 命令

```bash
.venv/bin/python scripts/build_oracle.py --suite core          # 一次性，约 5 分钟/任务
.venv/bin/python scripts/run_experiment_grid.py --mode benchmark --suite core \
    --models deepseek-chat --infras C C-sota C-corr --budget 8 --with-baselines
.venv/bin/python scripts/score_against_oracle.py docs/experiments/phase2_infras --group framework_type
```

## 已知结论（截至 2026-10-02，仅 SPY_full，预算 4，3–5 种子）

| 架构 / 模型 | percentile_val | test_sharpe | budget_util |
| :-- | --: | --: | --: |
| R-random (5 seeds) | 0.82 | 0.49 | 1.00 |
| R-grid (5 seeds) | 0.46 | 0.93 | 1.00 |
| C-hier 修复后 (deepseek, 3 seeds) | 0.27 | 0.98 | 1.00 |
| claude-sonnet-4-6 / C | 0.33 | 1.03 | 1.00 |
| deepseek-chat / C | 0.18 | 1.27 | 1.00 |
| C-hier 修复前 | 0.15 | 1.16 | 0.25 |

读法：LLM agent 几乎全部选了 momentum 120 附近的"教科书参数"，Val 分位只有 0.15–0.33，
搜索质量输给随机抽样。它们的 Test 夏普高是因为 2024–2026 的 SPY 行情恰好奖励长周期动量。
这正是只用单任务 + 裸 Test 夏普会得出错误结论的例子。多任务结果待 core 套件 oracle 建完后补。
