# Agent Infra Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a modular, extensible, and zero-lookahead Agent Infrastructure (`agent/`) on top of the native CTA backtester to explore quantitative strategies, review specs, and compare 4 agent architectures (A/B/C/D) across multiple seeds.

**Architecture:** A clean component pipeline where `Researcher` suggests strategy specifications (`StrategySpec`), `Reviewer` performs centralized gatekeeping, `StrategyEvaluator` executes strict 3-way time-split backtests (Search 60% / Select 20% / Holdout Test 20%), and `StrategyOrchestrator` controls iteration loops, selection gates, and ablation comparisons (Frameworks A, B, C, D).

**Tech Stack:** Python 3.9+, dataclasses, pandas, numpy, pytest.

## Global Constraints

- Root package for agent infrastructure is `agent/`.
- Must strictly isolate time splits: Search (60%), Selection (20%), Test (20%). No lookahead, no backfilling (`bfill` strictly forbidden).
- Evaluator must reset capital and position independently across splits so capital does not compound across stages.
- Virtual environment with pytest is located at `.venv/bin/pytest`.
- Follow strict typing and dataclasses for configuration and specs.
- Coaches/step-by-step methodology: atomic commits and verified tests for each task.

---

### Task 1: StrategySpec Data Contract and Validation

**Files:**
- Create: `agent/__init__.py`
- Create: `agent/spec.py`
- Test: `tests/test_spec.py`

**Interfaces:**
- Produces: `StrategySpec(spec_id: str, hypothesis: str, signal_type: str, params: dict, allowed_regimes=None, use_vol_scaling=False, vol_target=None, max_leverage=1.0)` with methods `.validate() -> tuple[bool, str]`, `.to_dict() -> dict`, `.from_dict(d: dict) -> StrategySpec`.

- [ ] **Step 1: Write the failing tests for StrategySpec**

```python
# tests/test_spec.py
import pytest
from agent.spec import StrategySpec

def test_valid_momentum_spec():
    spec = StrategySpec(
        spec_id="s1",
        hypothesis="Trend following with 50-day SMA",
        signal_type="momentum",
        params={"window": 50}
    )
    is_valid, reason = spec.validate()
    assert is_valid is True
    assert reason == ""

def test_invalid_momentum_spec():
    spec = StrategySpec(
        spec_id="s2",
        hypothesis="Invalid SMA window",
        signal_type="momentum",
        params={"window": 0}
    )
    is_valid, reason = spec.validate()
    assert is_valid is False
    assert "Invalid momentum window" in reason

def test_valid_macd_spec():
    spec = StrategySpec(
        spec_id="s3",
        hypothesis="Standard MACD",
        signal_type="macd",
        params={"fast": 12, "slow": 26, "signal_span": 9}
    )
    is_valid, reason = spec.validate()
    assert is_valid is True
    assert reason == ""

def test_invalid_macd_spec():
    spec = StrategySpec(
        spec_id="s4",
        hypothesis="Inverted MACD",
        signal_type="macd",
        params={"fast": 30, "slow": 10, "signal_span": 9}
    )
    is_valid, reason = spec.validate()
    assert is_valid is False
    assert "Invalid MACD parameters" in reason

def test_serialization():
    spec = StrategySpec(
        spec_id="s5",
        hypothesis="Test serialization",
        signal_type="momentum",
        params={"window": 20}
    )
    d = spec.to_dict()
    restored = StrategySpec.from_dict(d)
    assert restored.spec_id == spec.spec_id
    assert restored.params == spec.params
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_spec.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent'`

- [ ] **Step 3: Implement `agent/__init__.py` and `agent/spec.py`**

```python
# agent/__init__.py
"""Agent Infrastructure for CTA Backtesting."""
```

```python
# agent/spec.py
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional, List, Tuple

@dataclass
class StrategySpec:
    spec_id: str
    hypothesis: str
    signal_type: str
    params: Dict[str, Any]
    allowed_regimes: Optional[List[str]] = None
    use_vol_scaling: bool = False
    vol_target: Optional[float] = None
    max_leverage: float = 1.0

    def validate(self) -> Tuple[bool, str]:
        if self.signal_type == "momentum":
            window = self.params.get("window")
            if not isinstance(window, int) or window <= 1 or window > 250:
                return False, f"Invalid momentum window: {window}. Must be int in [2, 250]."
        elif self.signal_type == "macd":
            fast = self.params.get("fast")
            slow = self.params.get("slow")
            span = self.params.get("signal_span")
            if not (isinstance(fast, int) and isinstance(slow, int) and isinstance(span, int)):
                return False, f"MACD parameters must be integers: fast={fast}, slow={slow}, signal_span={span}"
            if not (1 < fast < slow) or span <= 1:
                return False, f"Invalid MACD parameters: fast={fast}, slow={slow}, span={span}. Must satisfy 1 < fast < slow and span > 1."
        else:
            return False, f"Unsupported signal_type: '{self.signal_type}'. Supported: ['momentum', 'macd']."
        return True, ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategySpec":
        return cls(
            spec_id=data["spec_id"],
            hypothesis=data.get("hypothesis", ""),
            signal_type=data["signal_type"],
            params=dict(data.get("params", {})),
            allowed_regimes=data.get("allowed_regimes"),
            use_vol_scaling=data.get("use_vol_scaling", False),
            vol_target=data.get("vol_target"),
            max_leverage=float(data.get("max_leverage", 1.0)),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_spec.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agent/__init__.py agent/spec.py tests/test_spec.py
git commit -m "feat(agent): implement StrategySpec and validation logic"
```

---

### Task 2: Config Definitions and Knobs

**Files:**
- Create: `agent/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `DataSplitConfig(search_ratio: float = 0.6, select_ratio: float = 0.2, test_ratio: float = 0.2)`
- Produces: `ExperimentConfig(framework_type: str = "B", max_iterations: int = 10, memory_length: int = 5, feedback_metrics: list = ..., selection_metric: str = "sharpe_ratio", min_improvement: float = 0.01, patience: int = 3, seed: int = 42)`

- [ ] **Step 1: Write failing tests for configuration classes**

```python
# tests/test_config.py
import pytest
from agent.config import DataSplitConfig, ExperimentConfig

def test_data_split_config_defaults():
    split = DataSplitConfig()
    assert split.search_ratio == 0.6
    assert split.select_ratio == 0.2
    assert split.test_ratio == 0.2
    assert split.validate() is True

def test_data_split_config_invalid():
    split = DataSplitConfig(search_ratio=0.5, select_ratio=0.5, test_ratio=0.5)
    with pytest.raises(ValueError, match="Sum of ratios must be 1.0"):
        split.validate()

def test_experiment_config_options():
    cfg = ExperimentConfig(
        framework_type="C",
        max_iterations=15,
        memory_length=3,
        seed=100
    )
    assert cfg.framework_type == "C"
    assert cfg.patience == 3
    assert "sharpe_ratio" in cfg.feedback_metrics
    assert cfg.selection_metric == "sharpe_ratio"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent.config'`

- [ ] **Step 3: Implement `agent/config.py`**

```python
# agent/config.py
from dataclasses import dataclass, field
from typing import List

@dataclass
class DataSplitConfig:
    search_ratio: float = 0.6
    select_ratio: float = 0.2
    test_ratio: float = 0.2

    def validate(self) -> bool:
        total = round(self.search_ratio + self.select_ratio + self.test_ratio, 6)
        if total != 1.0:
            raise ValueError(f"Sum of ratios must be 1.0, got {total}")
        if any(r <= 0 for r in (self.search_ratio, self.select_ratio, self.test_ratio)):
            raise ValueError("All split ratios must be positive numbers")
        return True

@dataclass
class ExperimentConfig:
    framework_type: str = "B"           # "A": Baseline, "B": Researcher-only, "C": Researcher+Reviewer, "D": Dual-Researcher
    max_iterations: int = 10            # Stop condition: max iterations
    memory_length: int = 5              # Knob: past iterations visible to Researcher
    feedback_metrics: List[str] = field(default_factory=lambda: [
        "sharpe_ratio", "max_drawdown", "turnover", "total_return"
    ])                                  # Knob: metrics fed back in prompt
    selection_metric: str = "sharpe_ratio" # Knob: metric used by Orchestrator on selection set
    min_improvement: float = 0.01       # Stop condition: early stopping threshold
    patience: int = 3                   # Stop condition: patience rounds
    seed: int = 42                      # Reproducibility seed
    initial_capital: float = 100000.0   # Backtester capital
    data_split: DataSplitConfig = field(default_factory=DataSplitConfig)

    def __post_init__(self):
        if self.framework_type not in ("A", "B", "C", "D"):
            raise ValueError(f"Invalid framework_type: '{self.framework_type}'. Must be one of ['A', 'B', 'C', 'D'].")
        self.data_split.validate()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agent/config.py tests/test_config.py
git commit -m "feat(agent): implement DataSplitConfig and ExperimentConfig"
```

---

### Task 3: Zero-Lookahead StrategyEvaluator and 3-Way Split Sandbox

**Files:**
- Create: `agent/evaluator.py`
- Test: `tests/test_evaluator.py`

**Interfaces:**
- Consumes: `StrategySpec` (`agent/spec.py`), `DataSplitConfig` (`agent/config.py`), `compute_moving_average_signal` & `compute_macd_signal` (`src/signals.py`), `Backtest` (`src/backtest.py`), `compute_strategy_metrics` (`src/metrics.py`).
- Produces: `StrategyEvaluator(df: pd.DataFrame, split_config: DataSplitConfig, initial_capital: float = 100000.0)` with methods:
  - `split_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`
  - `evaluate_spec(spec: StrategySpec, split_name: str) -> dict`
  - `evaluate_search(spec: StrategySpec, feedback_metrics: list[str]) -> dict`
  - `evaluate_selection(spec: StrategySpec) -> dict`
  - `evaluate_test(spec: StrategySpec) -> dict`

- [ ] **Step 1: Write failing tests for StrategyEvaluator**

```python
# tests/test_evaluator.py
import pytest
import pandas as pd
import numpy as np
from agent.config import DataSplitConfig
from agent.spec import StrategySpec
from agent.evaluator import StrategyEvaluator

@pytest.fixture
def dummy_market_data():
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(100))
    df = pd.DataFrame({
        "open": prices,
        "high": prices + 1,
        "low": prices - 1,
        "close": prices,
        "volume": 1000000
    }, index=dates)
    return df

def test_split_sizes_and_indices(dummy_market_data):
    split_cfg = DataSplitConfig(search_ratio=0.6, select_ratio=0.2, test_ratio=0.2)
    evaluator = StrategyEvaluator(dummy_market_data, split_config=split_cfg)
    
    search_df, select_df, test_df = evaluator.split_data()
    assert len(search_df) == 60
    assert len(select_df) == 20
    assert len(test_df) == 20
    
    # Assert monotonic time order with zero overlap
    assert search_df.index[-1] < select_df.index[0]
    assert select_df.index[-1] < test_df.index[0]

def test_search_evaluation_filters_feedback_metrics(dummy_market_data):
    evaluator = StrategyEvaluator(dummy_market_data)
    spec = StrategySpec("m1", "test", "momentum", {"window": 10})
    result = evaluator.evaluate_search(spec, feedback_metrics=["sharpe_ratio", "max_drawdown"])
    
    assert "sharpe_ratio" in result
    assert "max_drawdown" in result
    assert "calmar_ratio" not in result # Filtered out to avoid overwhelming the prompt

def test_independent_capital_per_split(dummy_market_data):
    evaluator = StrategyEvaluator(dummy_market_data, initial_capital=50000.0)
    spec = StrategySpec("m1", "test", "momentum", {"window": 10})
    
    res_search = evaluator.evaluate_spec(spec, "search")
    res_select = evaluator.evaluate_spec(spec, "select")
    
    # Both evaluations should independently start from initial_capital
    assert res_search["initial_capital"] == 50000.0
    assert res_select["initial_capital"] == 50000.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_evaluator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent.evaluator'`

- [ ] **Step 3: Implement `agent/evaluator.py`**

```python
# agent/evaluator.py
from typing import Tuple, Dict, Any, List
import pandas as pd
import numpy as np

from agent.spec import StrategySpec
from agent.config import DataSplitConfig
from src.signals import compute_moving_average_signal, compute_macd_signal
from src.backtest import Backtest
from src.metrics import compute_strategy_metrics

class StrategyEvaluator:
    def __init__(
        self,
        df: pd.DataFrame,
        split_config: DataSplitConfig = None,
        initial_capital: float = 100000.0
    ):
        self.df = df.sort_index()
        self.split_config = split_config or DataSplitConfig()
        self.split_config.validate()
        self.initial_capital = initial_capital
        self._search_df, self._select_df, self._test_df = self._create_splits()

    def _create_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        n = len(self.df)
        n_search = int(n * self.split_config.search_ratio)
        n_select = int(n * self.split_config.select_ratio)
        
        search_df = self.df.iloc[:n_search]
        select_df = self.df.iloc[n_search:n_search + n_select]
        test_df = self.df.iloc[n_search + n_select:]
        return search_df, select_df, test_df

    def split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return self._search_df, self._select_df, self._test_df

    def _generate_signals_for_split(self, spec: StrategySpec, split_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Calculates signals strictly forward in time.
        Uses historical lookback buffer from full df to warm up indicators without lookahead leakage.
        """
        # Determine warm-up window needed
        if spec.signal_type == "momentum":
            window = spec.params["window"]
        elif spec.signal_type == "macd":
            window = spec.params.get("slow", 26) + spec.params.get("signal_span", 9)
        else:
            raise ValueError(f"Unsupported signal_type: {spec.signal_type}")

        start_loc = self.df.index.get_loc(split_df.index[0])
        buffer_start = max(0, start_loc - window * 2)
        warmup_df = self.df.iloc[buffer_start: start_loc + len(split_df)]

        if spec.signal_type == "momentum":
            full_signal = compute_moving_average_signal(warmup_df, window=spec.params["window"])
        elif spec.signal_type == "macd":
            full_signal = compute_macd_signal(
                warmup_df,
                fast=spec.params["fast"],
                slow=spec.params["slow"],
                signal_span=spec.params["signal_span"]
            )
        
        # Slice signal strictly for the target split
        split_signal = full_signal.loc[split_df.index].fillna(0).astype(int)
        return split_df, split_signal

    def evaluate_spec(self, spec: StrategySpec, split_name: str) -> Dict[str, Any]:
        if split_name == "search":
            split_df = self._search_df
        elif split_name == "select":
            split_df = self._select_df
        elif split_name == "test":
            split_df = self._test_df
        else:
            raise ValueError(f"Unknown split_name: {split_name}")

        is_valid, reason = spec.validate()
        if not is_valid:
            raise ValueError(f"Invalid spec {spec.spec_id}: {reason}")

        data_slice, signals = self._generate_signals_for_split(spec, split_df)
        bt = Backtest(initial_capital=self.initial_capital)
        bt_result = bt.run(data_slice, signals)
        metrics_series = compute_strategy_metrics(bt_result)

        metrics_dict = metrics_series.to_dict()
        metrics_dict["initial_capital"] = self.initial_capital
        metrics_dict["spec_id"] = spec.spec_id
        metrics_dict["split_name"] = split_name
        return metrics_dict

    def evaluate_search(self, spec: StrategySpec, feedback_metrics: List[str] = None) -> Dict[str, Any]:
        full_res = self.evaluate_spec(spec, "search")
        if not feedback_metrics:
            return full_res
        filtered = {k: full_res[k] for k in feedback_metrics if k in full_res}
        filtered["spec_id"] = spec.spec_id
        return filtered

    def evaluate_selection(self, spec: StrategySpec) -> Dict[str, Any]:
        return self.evaluate_spec(spec, "select")

    def evaluate_test(self, spec: StrategySpec) -> Dict[str, Any]:
        return self.evaluate_spec(spec, "test")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_evaluator.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agent/evaluator.py tests/test_evaluator.py
git commit -m "feat(agent): implement zero-lookahead StrategyEvaluator with 3-way split"
```

---

### Task 4: Unified LLM Client Interface and Deterministic Mock

**Files:**
- Create: `agent/llm.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- Produces: `BaseLLMClient` (abstract class with `generate(prompt: str, system_prompt: str = "") -> str`)
- Produces: `MockLLMClient(seed: int = 42)` (deterministic mock that generates valid JSON `StrategySpec` responses)
- Produces: `OpenAICompatibleClient(api_key: str = "", base_url: str = "", model: str = "")`

- [ ] **Step 1: Write failing tests for LLM clients**

```python
# tests/test_llm.py
import json
import pytest
from agent.llm import MockLLMClient, BaseLLMClient

def test_mock_llm_client_returns_valid_json():
    client = MockLLMClient(seed=42)
    response = client.generate("Generate strategy with MA window")
    parsed = json.loads(response)
    
    assert "signal_type" in parsed
    assert "params" in parsed
    assert parsed["signal_type"] in ("momentum", "macd")

def test_mock_llm_deterministic_across_runs():
    client1 = MockLLMClient(seed=99)
    client2 = MockLLMClient(seed=99)
    
    res1 = client1.generate("Prompt 1")
    res2 = client2.generate("Prompt 1")
    assert res1 == res2

def test_mock_llm_reacts_to_rejection():
    client = MockLLMClient(seed=42)
    prompt_with_rejection = "Previous proposal rejected: Invalid momentum window: 0. Fix it."
    response = client.generate(prompt_with_rejection)
    parsed = json.loads(response)
    # Generated window should be valid
    if parsed["signal_type"] == "momentum":
        assert parsed["params"]["window"] > 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_llm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent.llm'`

- [ ] **Step 3: Implement `agent/llm.py`**

```python
# agent/llm.py
import abc
import json
import random
from typing import Optional, Dict, Any

class BaseLLMClient(abc.ABC):
    @abc.abstractmethod
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generates response text from the LLM given prompt."""
        pass

class MockLLMClient(BaseLLMClient):
    """
    Deterministic Mock LLM client for rapid testing, ablation studies,
    and seed reproducibility without API costs or network latency.
    """
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        self.call_count = 0

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        self.call_count += 1
        # Flip between momentum and macd based on call count and seed
        pick_momentum = (self.call_count % 2 == 1)
        
        if pick_momentum:
            # Pick a safe window between 10 and 200
            window = self.rng.choice([10, 20, 30, 50, 60, 100, 120, 150, 200])
            spec_dict = {
                "spec_id": f"mock_spec_{self.call_count}",
                "hypothesis": f"Trend momentum hypothesis #{self.call_count} with window {window}",
                "signal_type": "momentum",
                "params": {"window": window}
            }
        else:
            fast = self.rng.choice([8, 10, 12, 15])
            slow = self.rng.choice([20, 24, 26, 30])
            span = self.rng.choice([7, 9, 10])
            spec_dict = {
                "spec_id": f"mock_spec_{self.call_count}",
                "hypothesis": f"MACD crossover hypothesis #{self.call_count} (fast={fast}, slow={slow}, span={span})",
                "signal_type": "macd",
                "params": {"fast": fast, "slow": slow, "signal_span": span}
            }
        return json.dumps(spec_dict)

class OpenAICompatibleClient(BaseLLMClient):
    """
    Client for OpenAI-compatible REST APIs (DeepSeek, Qwen, Ollama, OpenAI).
    """
    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o",
        temperature: float = 0.7
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.temperature = temperature

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        # Placeholder for external HTTP request library (e.g. urllib or httpx/requests)
        # Raises clear error if used without API key
        if not self.api_key:
            raise ValueError("OpenAICompatibleClient requires a non-empty api_key.")
        raise NotImplementedError("Live API calls to be plugged in when external key is provided.")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_llm.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agent/llm.py tests/test_llm.py
git commit -m "feat(agent): implement BaseLLMClient and MockLLMClient"
```

---

### Task 5: Reviewer Agent with Dual-Gating

**Files:**
- Create: `agent/reviewer.py`
- Test: `tests/test_reviewer.py`

**Interfaces:**
- Consumes: `StrategySpec` (`agent/spec.py`)
- Produces: `ReviewResult(passed: bool, reason: str)`
- Produces: `Reviewer(allow_duplicates: bool = False)` with method `review(spec: StrategySpec, history: list[StrategySpec]) -> ReviewResult`

- [ ] **Step 1: Write failing tests for Reviewer**

```python
# tests/test_reviewer.py
import pytest
from agent.spec import StrategySpec
from agent.reviewer import Reviewer, ReviewResult

def test_reviewer_blocks_invalid_spec():
    reviewer = Reviewer()
    invalid_spec = StrategySpec("r1", "bad window", "momentum", {"window": -5})
    result = reviewer.review(invalid_spec, history=[])
    assert result.passed is False
    assert "Invalid momentum window" in result.reason

def test_reviewer_blocks_duplicate_parameters():
    reviewer = Reviewer(allow_duplicates=False)
    spec1 = StrategySpec("r1", "first", "momentum", {"window": 50})
    spec2 = StrategySpec("r2", "second with same params", "momentum", {"window": 50})
    
    result = reviewer.review(spec2, history=[spec1])
    assert result.passed is False
    assert "Duplicate strategy parameters" in result.reason

def test_reviewer_passes_novel_valid_spec():
    reviewer = Reviewer()
    spec1 = StrategySpec("r1", "first", "momentum", {"window": 50})
    spec2 = StrategySpec("r2", "novel", "momentum", {"window": 60})
    
    result = reviewer.review(spec2, history=[spec1])
    assert result.passed is True
    assert result.reason == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_reviewer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent.reviewer'`

- [ ] **Step 3: Implement `agent/reviewer.py`**

```python
# agent/reviewer.py
from dataclasses import dataclass
from typing import List, Tuple
from agent.spec import StrategySpec

@dataclass
class ReviewResult:
    passed: bool
    reason: str = ""

class Reviewer:
    """
    Centralized verification agent that screens strategy specs prior to backtesting.
    Filters invalid parameters, prevents duplicate re-evaluation, and enforces consistency.
    """
    def __init__(self, allow_duplicates: bool = False):
        self.allow_duplicates = allow_duplicates

    def review(self, spec: StrategySpec, history: List[StrategySpec]) -> ReviewResult:
        # Gate 1: Spec validation
        is_valid, reason = spec.validate()
        if not is_valid:
            return ReviewResult(passed=False, reason=f"Spec Validation Failed: {reason}")

        # Gate 2: Duplicate detection
        if not self.allow_duplicates:
            for past_spec in history:
                if (
                    past_spec.signal_type == spec.signal_type and
                    past_spec.params == spec.params
                ):
                    return ReviewResult(
                        passed=False,
                        reason=f"Duplicate strategy parameters already evaluated: {spec.signal_type} {spec.params}"
                    )

        # Gate 3: Hypothesis sanity check
        if not spec.hypothesis or len(spec.hypothesis.strip()) < 5:
            return ReviewResult(passed=False, reason="Insufficient hypothesis justification.")

        return ReviewResult(passed=True, reason="")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_reviewer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agent/reviewer.py tests/test_reviewer.py
git commit -m "feat(agent): implement Reviewer with rule and duplicate verification"
```

---

### Task 6: Researcher Agent with Rolling Memory and Feedback Loop

**Files:**
- Create: `agent/researcher.py`
- Test: `tests/test_researcher.py`

**Interfaces:**
- Consumes: `StrategySpec` (`agent/spec.py`), `BaseLLMClient` (`agent/llm.py`)
- Produces: `Researcher(llm_client: BaseLLMClient, memory_length: int = 5, researcher_id: str = "researcher_0")` with methods:
  - `propose(rejection_feedback: Optional[str] = None) -> StrategySpec`
  - `update_memory(spec: StrategySpec, metrics: dict)`

- [ ] **Step 1: Write failing tests for Researcher**

```python
# tests/test_researcher.py
import pytest
from agent.llm import MockLLMClient
from agent.researcher import Researcher
from agent.spec import StrategySpec

def test_researcher_proposes_spec():
    client = MockLLMClient(seed=42)
    researcher = Researcher(llm_client=client, memory_length=3)
    
    spec = researcher.propose()
    assert isinstance(spec, StrategySpec)
    assert spec.signal_type in ("momentum", "macd")

def test_researcher_rolling_memory():
    client = MockLLMClient(seed=42)
    researcher = Researcher(llm_client=client, memory_length=2)
    
    for i in range(4):
        s = StrategySpec(f"s_{i}", f"hypo_{i}", "momentum", {"window": 20 + i * 10})
        researcher.update_memory(s, {"sharpe_ratio": 0.5 + i * 0.1})
        
    assert len(researcher.memory) == 2
    assert researcher.memory[-1]["spec"].spec_id == "s_3"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_researcher.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent.researcher'`

- [ ] **Step 3: Implement `agent/researcher.py`**

```python
# agent/researcher.py
import json
from typing import Optional, List, Dict, Any
from agent.spec import StrategySpec
from agent.llm import BaseLLMClient

SYSTEM_PROMPT = """You are a quantitative CTA research agent.
Your objective is to propose profitable, sound CTA strategies avoiding overfitting.
You must output ONLY valid JSON matching the following schema:
{
  "spec_id": "<string>",
  "hypothesis": "<string explaining economic/statistical rationale>",
  "signal_type": "<'momentum' or 'macd'>",
  "params": {"<param_name>": <value>}
}
Rules:
- momentum params: {"window": integer between 2 and 250}
- macd params: {"fast": int, "slow": int, "signal_span": int} with 1 < fast < slow and signal_span > 1
"""

class Researcher:
    def __init__(
        self,
        llm_client: BaseLLMClient,
        memory_length: int = 5,
        researcher_id: str = "researcher_0"
    ):
        self.llm_client = llm_client
        self.memory_length = memory_length
        self.researcher_id = researcher_id
        self.memory: List[Dict[str, Any]] = []
        self.proposal_count = 0

    def update_memory(self, spec: StrategySpec, metrics: Dict[str, Any]):
        self.memory.append({"spec": spec, "metrics": metrics})
        if len(self.memory) > self.memory_length:
            self.memory.pop(0)

    def _build_prompt(self, rejection_feedback: Optional[str] = None) -> str:
        prompt_lines = [
            f"Researcher ID: {self.researcher_id}",
            f"Proposal Attempt: #{self.proposal_count + 1}",
            "\n--- Historical Memory (Most Recent Strategies & Performance) ---"
        ]
        if not self.memory:
            prompt_lines.append("No prior history. Propose initial baseline candidate.")
        else:
            for item in self.memory:
                spec = item["spec"]
                metrics = item["metrics"]
                prompt_lines.append(
                    f"- Spec {spec.spec_id} ({spec.signal_type}, {spec.params}): "
                    f"Metrics: {metrics}"
                )

        if rejection_feedback:
            prompt_lines.append(
                f"\n[ALERT - PRIOR PROPOSAL REJECTED BY REVIEWER]:\n"
                f"Reason: {rejection_feedback}\n"
                f"Please fix the issue and propose a revised specification."
            )
        else:
            prompt_lines.append("\nPlease propose a refined quantitative strategy.")

        return "\n".join(prompt_lines)

    def propose(self, rejection_feedback: Optional[str] = None) -> StrategySpec:
        self.proposal_count += 1
        prompt = self._build_prompt(rejection_feedback)
        response_text = self.llm_client.generate(prompt, system_prompt=SYSTEM_PROMPT)
        
        # Parse JSON
        try:
            data = json.loads(response_text)
            spec = StrategySpec.from_dict(data)
            spec.spec_id = f"{self.researcher_id}_iter_{self.proposal_count}"
            return spec
        except Exception as e:
            # Fallback safe spec if LLM fails formatting
            return StrategySpec(
                spec_id=f"{self.researcher_id}_iter_{self.proposal_count}",
                hypothesis=f"Fallback baseline due to parsing exception: {str(e)}",
                signal_type="momentum",
                params={"window": 50}
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_researcher.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agent/researcher.py tests/test_researcher.py
git commit -m "feat(agent): implement Researcher with rolling memory and prompt assembly"
```

---

### Task 7: StrategyOrchestrator and Framework Comparison (A/B/C/D)

**Files:**
- Create: `agent/orchestrator.py`
- Create: `scripts/run_agent_experiment.py`
- Test: `tests/test_orchestrator.py`

**Interfaces:**
- Consumes: `ExperimentConfig`, `StrategyEvaluator`, `Researcher`, `Reviewer`, `StrategySpec`.
- Produces: `StrategyOrchestrator(config: ExperimentConfig, evaluator: StrategyEvaluator)` with method `run() -> dict`.
- Computes:
  - `best_spec`: Winning strategy selected strictly on selection set.
  - `search_metrics`, `selection_metrics`, `test_metrics`.
  - `overfitting_gap`: `selection_metrics[metric] - test_metrics[metric]`.

- [ ] **Step 1: Write failing tests for StrategyOrchestrator**

```python
# tests/test_orchestrator.py
import pytest
import pandas as pd
import numpy as np
from agent.config import ExperimentConfig
from agent.evaluator import StrategyEvaluator
from agent.orchestrator import StrategyOrchestrator

@pytest.fixture
def synthetic_data():
    dates = pd.date_range("2020-01-01", periods=120, freq="B")
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(120))
    df = pd.DataFrame({
        "open": prices,
        "high": prices + 1,
        "low": prices - 1,
        "close": prices,
        "volume": 1000000
    }, index=dates)
    return df

@pytest.mark.parametrize("framework", ["A", "B", "C", "D"])
def test_all_frameworks_run_successfully(synthetic_data, framework):
    cfg = ExperimentConfig(
        framework_type=framework,
        max_iterations=4,
        memory_length=2,
        seed=42
    )
    evaluator = StrategyEvaluator(synthetic_data)
    orchestrator = StrategyOrchestrator(cfg, evaluator)
    result = orchestrator.run()
    
    assert "best_spec" in result
    assert "selection_metrics" in result
    assert "test_metrics" in result
    assert "overfitting_gap" in result
    assert result["framework_type"] == framework
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_orchestrator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent.orchestrator'`

- [ ] **Step 3: Implement `agent/orchestrator.py`**

```python
# agent/orchestrator.py
from typing import Dict, Any, List, Optional
import pandas as pd

from agent.config import ExperimentConfig
from agent.evaluator import StrategyEvaluator
from agent.spec import StrategySpec
from agent.llm import MockLLMClient
from agent.researcher import Researcher
from agent.reviewer import Reviewer

class StrategyOrchestrator:
    def __init__(self, config: ExperimentConfig, evaluator: StrategyEvaluator):
        self.config = config
        self.evaluator = evaluator
        self.evaluated_specs: List[StrategySpec] = []
        self.search_results: List[Dict[str, Any]] = []

    def run(self) -> Dict[str, Any]:
        if self.config.framework_type == "A":
            return self._run_framework_a()
        elif self.config.framework_type == "B":
            return self._run_framework_b()
        elif self.config.framework_type == "C":
            return self._run_framework_c()
        elif self.config.framework_type == "D":
            return self._run_framework_d()
        else:
            raise ValueError(f"Unknown framework_type {self.config.framework_type}")

    def _run_framework_a(self) -> Dict[str, Any]:
        """Framework A: Handwritten Baseline (no agent search)."""
        baselines = [
            StrategySpec("baseline_sma50", "Classic 50-day SMA", "momentum", {"window": 50}),
            StrategySpec("baseline_macd", "Classic MACD", "macd", {"fast": 12, "slow": 26, "signal_span": 9}),
        ]
        for spec in baselines:
            res = self.evaluator.evaluate_search(spec, self.config.feedback_metrics)
            self.evaluated_specs.append(spec)
            self.search_results.append(res)
        return self._finalize_selection_and_test()

    def _run_framework_b(self) -> Dict[str, Any]:
        """Framework B: Single Researcher, No Reviewer."""
        client = MockLLMClient(seed=self.config.seed)
        researcher = Researcher(client, memory_length=self.config.memory_length, researcher_id="res_b")
        
        for iteration in range(self.config.max_iterations):
            spec = researcher.propose()
            is_valid, _ = spec.validate()
            if not is_valid:
                continue
            metrics = self.evaluator.evaluate_search(spec, self.config.feedback_metrics)
            researcher.update_memory(spec, metrics)
            self.evaluated_specs.append(spec)
            self.search_results.append(metrics)
            
            if self._check_early_stop():
                break

        return self._finalize_selection_and_test()

    def _run_framework_c(self) -> Dict[str, Any]:
        """Framework C: Researcher + Reviewer Verification Gate."""
        client = MockLLMClient(seed=self.config.seed)
        researcher = Researcher(client, memory_length=self.config.memory_length, researcher_id="res_c")
        reviewer = Reviewer(allow_duplicates=False)
        
        rejection_reason = None
        for iteration in range(self.config.max_iterations):
            spec = researcher.propose(rejection_feedback=rejection_reason)
            review_result = reviewer.review(spec, self.evaluated_specs)
            
            if not review_result.passed:
                rejection_reason = review_result.reason
                continue
            
            rejection_reason = None
            metrics = self.evaluator.evaluate_search(spec, self.config.feedback_metrics)
            researcher.update_memory(spec, metrics)
            self.evaluated_specs.append(spec)
            self.search_results.append(metrics)

            if self._check_early_stop():
                break

        return self._finalize_selection_and_test()

    def _run_framework_d(self) -> Dict[str, Any]:
        """Framework D: Dual-Researcher Competition."""
        client1 = MockLLMClient(seed=self.config.seed)
        client2 = MockLLMClient(seed=self.config.seed + 1000)
        res1 = Researcher(client1, memory_length=self.config.memory_length, researcher_id="res_d1")
        res2 = Researcher(client2, memory_length=self.config.memory_length, researcher_id="res_d2")
        
        for iteration in range(self.config.max_iterations // 2):
            for res_agent in (res1, res2):
                spec = res_agent.propose()
                is_valid, _ = spec.validate()
                if not is_valid:
                    continue
                metrics = self.evaluator.evaluate_search(spec, self.config.feedback_metrics)
                res_agent.update_memory(spec, metrics)
                self.evaluated_specs.append(spec)
                self.search_results.append(metrics)

        return self._finalize_selection_and_test()

    def _check_early_stop(self) -> bool:
        if len(self.search_results) < self.config.patience + 1:
            return False
        metric = self.config.selection_metric
        recent = [r.get(metric, 0.0) for r in self.search_results[-self.config.patience:]]
        best_prior = max(r.get(metric, 0.0) for r in self.search_results[:-self.config.patience])
        if all((r - best_prior) < self.config.min_improvement for r in recent):
            return True
        return False

    def _finalize_selection_and_test(self) -> Dict[str, Any]:
        """
        Selection Gate: Evaluate all search candidates strictly on the selection set.
        Pick single best candidate.
        Holdout Test: Evaluate ONLY the winning candidate on the test set.
        """
        if not self.evaluated_specs:
            # Fallback baseline if all rejected
            fallback = StrategySpec("fallback", "Default SMA 50", "momentum", {"window": 50})
            self.evaluated_specs.append(fallback)

        selection_scores = []
        for spec in self.evaluated_specs:
            sel_metrics = self.evaluator.evaluate_selection(spec)
            selection_scores.append((sel_metrics.get(self.config.selection_metric, -999.0), spec, sel_metrics))

        # Sort descending
        selection_scores.sort(key=lambda x: x[0], reverse=True)
        best_score, best_spec, best_sel_metrics = selection_scores[0]

        # Holdout test on winner only
        test_metrics = self.evaluator.evaluate_test(best_spec)

        metric_name = self.config.selection_metric
        overfitting_gap = best_sel_metrics.get(metric_name, 0.0) - test_metrics.get(metric_name, 0.0)

        return {
            "framework_type": self.config.framework_type,
            "seed": self.config.seed,
            "total_candidates": len(self.evaluated_specs),
            "best_spec": best_spec.to_dict(),
            "selection_metrics": best_sel_metrics,
            "test_metrics": test_metrics,
            "overfitting_gap": overfitting_gap,
            "search_candidates_count": len(self.evaluated_specs)
        }
```

- [ ] **Step 4: Create runner script `scripts/run_agent_experiment.py`**

```python
# scripts/run_agent_experiment.py
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import load_market_data
from agent.config import ExperimentConfig
from agent.evaluator import StrategyEvaluator
from agent.orchestrator import StrategyOrchestrator

def run_multi_seed_experiment(data_path: str = "data/SPY.csv", seeds=(42, 43, 44, 45, 46)):
    print(f"Loading data from {data_path}...")
    df = load_market_data(data_path)
    evaluator = StrategyEvaluator(df)

    frameworks = ["A", "B", "C", "D"]
    all_results = []

    print("\nStarting Cross-Framework Ablation Experiment across 5 seeds...")
    for fw in frameworks:
        print(f"\n>>> Running Framework {fw}...")
        for seed in (seeds if fw != "A" else [42]):
            cfg = ExperimentConfig(framework_type=fw, seed=seed, max_iterations=8)
            orch = StrategyOrchestrator(cfg, evaluator)
            res = orch.run()
            all_results.append({
                "Framework": fw,
                "Seed": seed,
                "Best Spec": res["best_spec"]["signal_type"] + str(res["best_spec"]["params"]),
                "Select Sharpe": round(res["selection_metrics"]["sharpe_ratio"], 4),
                "Test Sharpe": round(res["test_metrics"]["sharpe_ratio"], 4),
                "Overfitting Gap": round(res["overfitting_gap"], 4),
                "Test MDD (%)": round(res["test_metrics"]["max_drawdown"] * 100, 2),
            })

    summary_df = pd.DataFrame(all_results)
    print("\n" + "=" * 80)
    print("                    AGENT INFRA EXPERIMENT SUMMARY")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80)

    # Average metrics per framework
    grouped = summary_df.groupby("Framework")[["Select Sharpe", "Test Sharpe", "Overfitting Gap"]].mean()
    print("\nMean Performance by Framework:")
    print(grouped.round(4).to_string())

if __name__ == "__main__":
    run_multi_seed_experiment()
```

- [ ] **Step 5: Run tests and runner script to verify end-to-end integration**

Run: `.venv/bin/pytest tests/test_orchestrator.py -v`
Expected: PASS
Run: `.venv/bin/python3 scripts/run_agent_experiment.py`
Expected: Print summary table showing metrics for Frameworks A, B, C, D across seeds.

- [ ] **Step 6: Commit**

```bash
git add agent/orchestrator.py scripts/run_agent_experiment.py tests/test_orchestrator.py
git commit -m "feat(agent): implement StrategyOrchestrator with A/B/C/D frameworks and runner script"
```
