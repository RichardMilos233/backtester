# TIA 三层评测基准运行协议 (Benchmark Protocol)

本文件说明三层评测框架（第一层：代码翻译与挑错、第二层：空间内策略搜索、第三层：合成 Planted Alpha 验证）的评分规则、文件结构与运行命令。

---

## 核心报告与数据导航

| 文件路径 | 用途 |
| :--- | :--- |
| `docs/LAYER1_BENCHMARK_SHOWCASE.html` | **第一层主报告（精简版）**：8 模型总榜 + 挑错题/写代码题正反案例对比 |
| `docs/LAYER1_FULL_EXPERIMENT_TABLE.html` | **第一层全量实验大表（网页交互版）**：40 题 × 8 模型（320 条）完整提问、标准答案、模型输出与对错批注 |
| `docs/experiments/layer1_canonical/LAYER1_FULL_RESULTS.xlsx` | **第一层全量实验大表（Excel 版）**：含 4 个 Sheet（对错矩阵、挑错题全表、写代码题全表、320 行长表） |
| `docs/BENCHMARK_DESIGN_DISCUSSION.html` | **三层框架总体设计讨论稿**：三层定位、12 项子任务状态与各层实测结论 |
| `docs/FINAL_RESEARCH_REPORT.md` | **底层回测引擎与策略基线实证报告**：SPY 12 年日频回测、摩擦冲击、四象限 Regime 与 ML 评估 |

---

## 第一层：策略代码翻译与挑错基准 (`agent/layer1/`)

测大模型两件事：**1. 给一段策略和回测代码，能不能找出里面的隐蔽 Bug（挑错题 20 道）**；**2. 给一段交易员大白话规则，能不能写出和标准答案持仓一致的代码（实现题 20 道）**。

### 1. 评分标准
- **第一部分：挑错题（20 题，`Identify Bad Alpha`）**
  - 包含 10 道跨模块耦合 / `pandas` 默认参数陷阱（如 `where()` 默认填 `NaN` 导致回测引擎锁仓、`resample('3D')` 默认 `label='left'`、`ewm` 默认 `min_periods=0`）、6 道经典/复合缺陷、4 道无缺陷干净对照题。
  - **通过标准（`strict_pass`）**：必须同时选对缺陷分类标签 + 在解释中准确指出底层触发机制（`mechanism_hit`）；干净对照题不误报才算通过。
- **第二部分：实现题（20 题，`Systematic Implementation`）**
  - 包含 6 道内部因子库与摩擦成本集成题（Track A）+ 14 道零代码提示的复杂状态机与重灾区题（Track B）。
  - **通过标准（`passed`）**：黑盒运行通过、截断+未来价格扰动测试无前视偏差（`no_lookahead`）、无残留 `NaN`、杠杆与摩擦参数匹配，且 **非空仓活跃 K 线持仓一致率 `ActiveActionMatch >= 95%`**、**交易回合匹配 `TradeF1 >= 90%`**。

### 2. 第一层 8 模型总榜（40 题正式版）

| 排名 | 模型 | 总分 (40 题) | 挑错题通过率 (20 题) | 写代码通过率 (20 题) | 活跃K线一致率 `ActiveMatch` |
| :---: | :--- | :---: | :---: | :---: | :---: |
| — | `golden`（标准参考） | **1.000** | 1.000 | 1.000 | 1.000 |
| 1 | `gpt-6-astra` | **0.975** | 0.950 (19/20) | **1.000** (20/20) | **1.000** |
| 2 | `claude-opus-5.5` | **0.900** | 0.950 (19/20) | 0.850 (17/20) | 0.920 |
| 3 | `claude-sonnet-4-6` | **0.850** | 0.850 (17/20) | 0.850 (17/20) | 0.892 |
| 4 | `gpt-5.4-mini` | **0.775** | 0.750 (15/20) | 0.800 (16/20) | 0.887 |
| 4 | `deepseek-v4-pro` | **0.775** | 0.800 (16/20) | 0.750 (15/20) | 0.831 |
| 6 | `deepseek-v3.2` | **0.650** | 0.750 (15/20) | 0.550 (11/20) | 0.646 |
| 7 | `claude-haiku-4-5` | **0.625** | 0.750 (15/20) | 0.500 (10/20) | 0.596 |
| 8 | `qwen-plus` | **0.500** | 0.600 (12/20) | 0.400 (8/20) | 0.449 |

### 3. 第一层运行与导出命令

```bash
# 运行 40 题正式版第一层评测（自动复用已有缓存）
.venv/bin/python scripts/run_layer1_canonical.py --out docs/experiments/layer1_canonical

# 重新导出全量网页表 (LAYER1_FULL_EXPERIMENT_TABLE.html)、Excel (.xlsx) 与宽/长 CSV
.venv/bin/python scripts/export_layer1_full_table.py
```

---

## 第二层：策略空间内搜索基准 (`agent/orchestrator.py` & `agent/oracle.py`)

固定策略参数空间与 8 次回测预算，衡量 Agent 选出的策略在全空间 Oracle 验证集夏普分布中的分位数（`percentile_val`），以及整条搜索轨迹的样本效率（`auc_percentile_val`）。

### 1. 空间与任务集
- **任务套件 `core`（8 个任务）**：`SPY_w1`, `SPY_w2`, `SPY_w3`, `QQQ_w2`, `IWM_w2`, `TLT_w3`, `GLD_w1`, `SYNTH_RW_full`（内部按 60/20/20 切 Search / Val / Test）。
- **v1 空间**（`docs/experiments/oracle/`）：momentum + macd × 波动率缩放，共 12,020 个格点。
- **v2 空间**（`docs/experiments/oracle_v2/`）：新增 mean_reversion / breakout / rsi / combo + 四象限 Regime 过滤，全空间 153,090 个格点（采样 20,000 个构建 Oracle）。

### 2. 第二层 v2 空间实测结果（8 任务 × 5 种子 × 预算 8，早停关闭）

详见 `docs/experiments/benchmark_core_v2/LAYER2_EXTENSIONS_REPORT.md`：

| 架构 / 方法 | 终点分位 `percentile_val` | 轨迹面积 `auc_percentile_val` | Step 1 $\to$ Step 8 分位提升 | 跑赢历史中位数比例 `median_imp_rate` |
| :--- | :---: | :---: | :---: | :---: |
| `R-grid`（等距网格基线） | **0.861** | 0.732 | $0.518 \to 0.861$ (+0.343) | 57.1% |
| `R-random`（均匀随机基线） | 0.851 | **0.768** | $0.515 \to 0.851$ (+0.337) | 48.9% |
| `C`（Researcher + Reviewer） | 0.811 | 0.708 | $0.426 \to 0.811$ (**+0.385**) | **61.4%** |
| `C-sota`（锚定历史最优） | 0.783 | 0.709 | $0.479 \to 0.783$ (+0.304) | 57.1% |
| `C-corr`（信号相关性去重） | 0.760 | 0.692 | $0.492 \to 0.760$ (+0.268) | 58.6% |

### 3. 第二层运行命令

```bash
# 构建 v2 空间 Oracle 分布表
.venv/bin/python scripts/build_oracle.py --suite core --regimes --sample 20000

# 运行第二层 core 套件基准
.venv/bin/python scripts/run_experiment_grid.py --mode benchmark --suite core \
    --models deepseek-chat --infras C C-sota C-corr --budget 8 --with-baselines

# 计算轨迹样本效率 (AUC) 与假设推理质量报告
.venv/bin/python scripts/score_layer2_extensions.py docs/experiments/benchmark_core_v2
```

---

## 第三层：合成 Planted Alpha 验证 (`agent/planted_alpha.py`)

切断大模型对历史行情的记忆，往合成 OHLCV 序列中注入精确已知的数学 Alpha（`SYNTH_MOM` 动量漂移、`SYNTH_MREV` O-U 均值回归、`SYNTH_REGIME_Q1` 象限条件漂移、`SYNTH_NULL` 零漂移纯噪声对照），检验 Agent 能否命中真实信号族并在纯噪声上拒绝过拟合：

```bash
.venv/bin/python scripts/make_planted_alpha.py
```
