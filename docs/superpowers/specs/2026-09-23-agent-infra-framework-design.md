# Agent Infra 模块化架构设计规范 (Design Spec)

## 1. 背景与目标 (Context & Objectives)

本规范定义 CTA 回测系统上层的 **Agent Infra（智能体基础设施）** 架构。
作为量化回测引擎的驱动与探索层，Agent Infra 承担以下核心职责：
1. **策略假说与参数探索**：通过 Researcher 智能体生成符合金融逻辑的候选策略规范（`StrategySpec`）。
2. **中心化审查与纠偏**：通过 Reviewer 智能体在回测前进行“硬规则 + 逻辑合理性”质检，阻断无效搜索与幻觉。
3. **严格时序因果与防数据泄漏**：实现严格的**三段式物理数据隔离**（搜索段 60%、选择段 20%、测试段 20%），杜绝前视偏差（Lookahead Bias）与后向填补泄漏。
4. **灵活消融与对比实验（A/B/C/D Frameworks）**：通过强类型配置“旋钮”（Knobs），支持便捷对比：
   - **Setup A**：手写经典基线策略（无 Agent）。
   - **Setup B**：单 Researcher 循环探索（无 Reviewer）。
   - **Setup C**：Researcher + Reviewer 质检拦截。
   - **Setup D**：双 Researcher 并行探索，由选择段挑优。
5. **支持 5 组随机种子重复实验**：定量评估并对比各框架在“选择段表现 vs 真实测试段表现”之间的过拟合衰减差（Overfitting Gap）。

---

## 2. 系统架构与模块划分 (Architecture & Modules)

所有 Agent Infra 代码均收敛在根目录的 `agent/` 包中，与底层的回测内核 `src/` 保持清晰的单向依赖（Agent 调用 `src/`，`src/` 不依赖 `agent/`）：

```text
agent/
├── __init__.py
├── spec.py          # StrategySpec 数据结构与内建合法性自检
├── config.py        # 强类型 Dataclass 配置类 (实验、旋钮、超参)
├── llm.py           # LLM 抽象基类、MockLLM 实现、OpenAI 兼容客户端
├── evaluator.py     # 三段式数据切分、信号驱动沙盒与分段回测
├── researcher.py    # 策略探索 Agent (Prompt 组装、上下文滚动记忆)
├── reviewer.py      # 前置质检 Agent (硬规则检验、反思打回机制)
└── orchestrator.py  # 迭代主循环、停止条件判定、跨框架 A/B/C/D 调度
```

---

## 3. 详细设计与数据契约 (Detailed Specifications)

### 3.1 策略契约：`StrategySpec` (`agent/spec.py`)

统一数据交换对象，具备内建的静态合法性校验方法：

```python
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class StrategySpec:
    spec_id: str                      # 唯一策略编号 (例如 "iter_1_cand_0")
    hypothesis: str                   # 策略核心量化假说阐述
    signal_type: str                  # 当前支持: "momentum" (SMA) 或 "macd"
    params: Dict[str, Any]            # 参数字典 (如 {'window': 50} 或 {'fast': 12, 'slow': 26, 'signal_span': 9})
    
    # 预留给后续 Session 6-8 扩展的接口字段（向后兼容）：
    allowed_regimes: Optional[list] = None    # 允许交易的市场状态 (Session 6-7)
    use_vol_scaling: bool = False             # 是否开启波动率缩放 (Session 8)
    vol_target: Optional[float] = None        # 目标波动率
    max_leverage: float = 1.0                 # 最大杠杆上限

    def validate(self) -> tuple[bool, str]:
        """内建合法性强校验"""
        if self.signal_type == "momentum":
            window = self.params.get("window")
            if not isinstance(window, int) or window <= 1 or window > 250:
                return False, f"Invalid momentum window: {window}"
        elif self.signal_type == "macd":
            fast = self.params.get("fast", 0)
            slow = self.params.get("slow", 0)
            span = self.params.get("signal_span", 0)
            if not (1 < fast < slow) or span <= 1:
                return False, f"Invalid MACD parameters: fast={fast}, slow={slow}, span={span}"
        else:
            return False, f"Unsupported signal_type: {self.signal_type}"
        return True, ""
```

### 3.2 实验与旋钮配置：`Config` (`agent/config.py`)

所有控制变量均在 Dataclass 中显式声明：

```python
from dataclasses import dataclass, field
from typing import List

@dataclass
class ExperimentConfig:
    framework_type: str = "B"           # "A": 基线, "B": 单Agent, "C": +Reviewer, "D": 双Agent并行
    max_iterations: int = 10            # 最大探索轮数 (停止条件)
    memory_length: int = 5              # Researcher 可见的最近历史记忆条数 (旋钮)
    feedback_metrics: List[str] = field(default_factory=lambda: [
        "sharpe_ratio", "max_drawdown", "turnover"
    ])                                  # 反馈给 Researcher 的指标维度 (旋钮)
    selection_metric: str = "sharpe_ratio" # 选择段裁决最优的标准 (旋钮)
    min_improvement: float = 0.01       # 早停提升容忍下限 (停止条件)
    patience: int = 3                   # 连续无提升早停轮数 (停止条件)
    seed: int = 42                      # 随机种子 (支持跑 5 个种子评估鲁棒性)
```

### 3.3 严格因果数据切分与沙盒：`StrategyEvaluator` (`agent/evaluator.py`)

#### 数据物理隔离机制
* **时间轴顺序切分**：按 60% : 20% : 20% 划分时间区间：
  1. **搜索段 (Search Split)**：仅将该段指标返回给 Researcher。
  2. **选择段 (Selection Split)**：严格对 Researcher 隐匿，仅 Orchestrator 在搜索结束后做策略裁决时调用。
  3. **测试段 (Holdout Test Split)**：最终胜出策略定稿后，全局仅执行一次。
* **因果指标计算与预热保护（Anti-Leakage Warmup）**：
  - 严禁任何反向填补算子（如 `bfill`）。
  - 全局信号计算前向滚动生成，进入选择段/测试段时，仅使用过去真实发生的历史数据作为前向预热缓冲（Lookback Buffer），各段回测独立初始化现金（如 100,000）与零持仓，杜绝跨段资金和仓位污染。

### 3.4 LLM 统一抽象与 Mock 驱动：`agent/llm.py`

* **`BaseLLMClient`**：抽象基类，定义 `generate(prompt: str, system_prompt: str = "") -> str`。
* **`MockLLMClient`**：
  - 依赖传入的随机种子 `seed` 确定性产出合法的 JSON 格式 `StrategySpec`。
  - 模拟策略探索行为（如微调参数、根据历史反馈调整方向），无需真实 API 即可极速完成全功能回归测试与 5 种子复现实验。
* **`OpenAICompatibleClient`**：
  - 标准化对接 OpenAI / DeepSeek / Claude 代理等实际大模型服务，包含重试、超时处理与 JSON 提取逻辑。

### 3.5 智能体实现 (`agent/researcher.py` & `agent/reviewer.py`)

* **`Researcher`**：
  - 管理滚动记忆（`history` 长度限制为 `memory_length`）。
  - 根据上轮 Reviewer 的打回意见（如果有）注入 Prompt，生成包含 `hypothesis`、`signal_type`、`params` 的新 Spec。
* **`Reviewer`**：
  - **第一道防线（硬规则）**：调用 `spec.validate()`，并比对历史策略参数库，杜绝无效或重复参数。
  - **第二道防线（合理性审查）**：对假设合理性进行质检。打回时生成具体拒绝原因（`rejection_reason`），反馈给下一轮 Researcher。

### 3.6 流程协调与框架对比：`StrategyOrchestrator` (`agent/orchestrator.py`)

* 支持统一执行入口 `run_experiment(config: ExperimentConfig)`。
* 内部根据 `config.framework_type` 自动路由至对应工作流：
  - **Framework A**：运行固定的 SMA50 与 MACD(12,26,9) 基线，输出评估。
  - **Framework B**：单 Researcher 无 Reviewer 循环迭代。
  - **Framework C**：Researcher + Reviewer 质检拦截循环（打回不占有效回测轮数，计入尝试日志）。
  - **Framework D**：两个独立配置的 Researcher 并行产生候选池，由选择段公平打分选拔。
* **终局测试与过拟合度量**：
  $$\text{Overfitting Gap} = \text{Metric}_{\text{selection}} - \text{Metric}_{\text{test}}$$

---

## 4. 验证与测试方案 (Verification Plan)

1. **单测覆盖 (Unit Tests)**：
   - `test_spec.py`：测试 `StrategySpec` 参数有效性检查与非法参数阻断。
   - `test_evaluator.py`：验证数据切分无交集、无重叠，验证预热不发生前视泄漏。
   - `test_llm_mock.py`：验证 `MockLLMClient` 在同一种子下输出一致，不同种子下可控扰动。
2. **端到端集成测试 (End-to-End Test)**：
   - 使用 `MockLLMClient` 完整运行 Framework A / B / C / D，确保能在 5 秒内完成 5 个种子运行并正确生成对比报表（含 Selection Sharpe, Test Sharpe, Overfitting Gap）。
