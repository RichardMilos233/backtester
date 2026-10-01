# Agent Strategy Search Experiment Platform Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a robust, reproducible, resumable agent strategy search experiment platform that supports orthogonal evaluation of foundation models (Phase 1) and agent architectures (Phase 2) under identical backtest and budget constraints.

**Architecture:** 
- Extend `ExperimentConfig` and `StrategySpec` with model identification, fair backtest-capped budgeting, and parse-failure tagging.
- Expand `Reviewer`, `Researcher`, and `StrategyOrchestrator` to support modern agent topologies (`C-sota`, `C-corr`, `C-reflect`, `C-hier`) alongside baseline frameworks (`A`, `B`, `C`, `D`).
- Create `agent.grid` runner and `scripts/run_experiment_grid.py` with idempotent resumption, JSONL run trace persistence, and CSV benchmark summaries.

**Tech Stack:** Python 3.9+, dataclasses, pandas, numpy, urllib (standard library REST client), pytest.

## Global Constraints

- Never use lookahead data: all signals shift(1), execution on next-day open with 5bps slippage + 5bps commission.
- Train (60%) / Validation (20%) / Test (20%) strict causal isolation. Test set evaluated ONLY once on the single winner chosen by Validation.
- Equalized budgets: compare architectures and models by actual backtests performed on the search set (`max_backtests`), not unverified proposal attempts.
- Fallback specs generated on JSON parsing errors must be tracked explicitly (`is_fallback=True`) and never masquerade as legitimate discoveries.
- Resumability: if `(model, framework, seed)` exists in `summary.csv`, skip re-execution to conserve tokens and handle network hiccups gracefully.
- All code and docstrings in English; user explanations in Chinese.

---

### Task 1: Spec and Config Extensions for Fair Budgeting and Models

**Files:**
- Modify: `agent/spec.py`
- Modify: `agent/config.py`
- Test: `tests/test_config.py`
- Test: `tests/test_spec.py`

**Interfaces:**
- `StrategySpec`: add `is_fallback: bool = False` to dataclass with `to_dict()` and `from_dict()` compatibility.
- `ExperimentConfig`:
  - `model: str = "mock"`
  - `max_backtests: int = 8`
  - `max_proposals_multiplier: int = 3` (safety cap on total LLM proposals = `max_backtests * 3`)
  - `framework_type: str = "C"`: allow `["A", "B", "C", "D", "C-sota", "C-corr", "C-reflect", "C-hier"]`
  - `correlation_threshold: float = 0.95` (for correlation filtering)
  - `vol_scaling_enabled: bool = False`

- [ ] **Step 1: Write failing tests for StrategySpec.is_fallback and ExperimentConfig extensions**

In `tests/test_spec.py`:
```python
def test_spec_fallback_flag():
    spec = StrategySpec(
        spec_id="test_fallback",
        hypothesis="Fallback test",
        signal_type="momentum",
        params={"window": 50},
        is_fallback=True
    )
    assert spec.is_fallback is True
    d = spec.to_dict()
    assert d["is_fallback"] is True
    recovered = StrategySpec.from_dict(d)
    assert recovered.is_fallback is True
```

In `tests/test_config.py`:
```python
def test_config_expanded_frameworks_and_budget():
    for fw in ["A", "B", "C", "D", "C-sota", "C-corr", "C-reflect", "C-hier"]:
        cfg = ExperimentConfig(framework_type=fw, model="deepseek-chat", max_backtests=8)
        assert cfg.framework_type == fw
        assert cfg.model == "deepseek-chat"
        assert cfg.max_backtests == 8

def test_config_invalid_framework_raises():
    import pytest
    with pytest.raises(ValueError, match="Invalid framework_type"):
        ExperimentConfig(framework_type="UNKNOWN_INFRA")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_spec.py tests/test_config.py -v`
Expected: FAIL on `is_fallback` unexpected keyword or `Invalid framework_type`.

- [ ] **Step 3: Implement minimal code changes**

Update `agent/spec.py`:
- Add `is_fallback: bool = False` to `StrategySpec`.
- Ensure `to_dict` and `from_dict` serialize/deserialize `is_fallback`.

Update `agent/config.py`:
- Add fields: `model: str = "mock"`, `max_backtests: int = 8`, `max_proposals_multiplier: int = 3`, `correlation_threshold: float = 0.95`.
- In `__post_init__`, validate `framework_type in ("A", "B", "C", "D", "C-sota", "C-corr", "C-reflect", "C-hier")`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_spec.py tests/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agent/spec.py agent/config.py tests/test_spec.py tests/test_config.py
git commit -m "feat(agent): add is_fallback flag and support extended frameworks in config"
```

---

### Task 2: Robust Parse Tracking in Researcher and Correlation Gate in Reviewer

**Files:**
- Modify: `agent/researcher.py`
- Modify: `agent/reviewer.py`
- Test: `tests/test_researcher.py`
- Test: `tests/test_reviewer.py`

**Interfaces:**
- `Researcher.propose()` returns `StrategySpec` with `is_fallback=True` if JSON parsing fails.
- `Researcher.last_parse_failed: bool` property indicating whether last call was a fallback.
- `Reviewer`:
  - `def __init__(self, allow_duplicates: bool = False, check_correlation: bool = False, correlation_threshold: float = 0.95, evaluator: Optional[StrategyEvaluator] = None)`
  - If `check_correlation=True`, generate search signal for candidate spec, calculate Pearson correlation against signals of all `evaluated_specs`. If any `|corr| >= correlation_threshold`, reject with `Reason: High signal correlation ({corr:.2f} >= {threshold}) with past spec {id}`.

- [ ] **Step 1: Write failing tests for Researcher fallback tracking and Reviewer correlation gate**

In `tests/test_researcher.py`:
```python
def test_researcher_fallback_flag_on_invalid_json():
    class BadLLM(BaseLLMClient):
        def generate(self, prompt: str, system_prompt: str = "") -> str:
            return "This is not JSON at all."

    researcher = Researcher(BadLLM(), researcher_id="bad_json_res")
    spec = researcher.propose()
    assert spec.is_fallback is True
    assert researcher.last_parse_failed is True
    assert "Fallback baseline due to parsing exception" in spec.hypothesis
```

In `tests/test_reviewer.py`:
```python
def test_reviewer_correlation_gate(sample_ohlcv_data):
    from agent.evaluator import StrategyEvaluator
    evaluator = StrategyEvaluator(sample_ohlcv_data)
    reviewer = Reviewer(check_correlation=True, correlation_threshold=0.90, evaluator=evaluator)
    
    spec1 = StrategySpec("s1", "SMA 50", "momentum", {"window": 50})
    # spec2 is virtually identical (SMA 51), correlation ~ 0.99
    spec2 = StrategySpec("s2", "SMA 51", "momentum", {"window": 51})
    
    res = reviewer.review(spec2, history=[spec1])
    assert res.passed is False
    assert "High signal correlation" in res.reason
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_researcher.py tests/test_reviewer.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement fallback tracking and correlation gate**

In `agent/researcher.py`:
- Add `self.last_parse_failed = False` in `__init__`.
- In `propose()`, if JSON parsing succeeds set `self.last_parse_failed = False`, if it fails set `self.last_parse_failed = True` and set `is_fallback=True` on the returned fallback spec.

In `agent/reviewer.py`:
- Add `evaluator: Optional[Any] = None`, `check_correlation: bool = False`, `correlation_threshold: float = 0.95` to `Reviewer.__init__`.
- In `review()`, after Gate 1 & 2 & 3, if `self.check_correlation and self.evaluator and history`:
  - Compute candidate signal using `self.evaluator._generate_signal(spec, self.evaluator.data_search)`.
  - For each `past_spec in history`:
    - Compute past signal.
    - Check Pearson correlation on non-NaN rows.
    - If `abs(corr) >= self.correlation_threshold`: return `ReviewResult(passed=False, reason=f"High signal correlation ({corr:.4f} >= {self.correlation_threshold}) with {past_spec.spec_id}")`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_researcher.py tests/test_reviewer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agent/researcher.py agent/reviewer.py tests/test_researcher.py tests/test_reviewer.py
git commit -m "feat(agent): add parse-failure tracking in researcher and correlation gate in reviewer"
```

---

### Task 3: Support Advanced Agent Topologies in StrategyOrchestrator

**Files:**
- Modify: `agent/orchestrator.py`
- Test: `tests/test_orchestrator.py`

**Topologies to implement:**
- `C-sota`: Memory always retains the SOTA spec (highest search selection metric) in slot 0, with remaining slots for FIFO recent memory.
- `C-corr`: Uses `Reviewer` with `check_correlation=True`.
- `C-reflect`: After each backtest, Researcher builds a reflection prompt asking the LLM to reflect on why the strategy had high/low return/drawdown, and includes that reflection in memory.
- `C-hier`: Alternating levels: Even iterations optimize core indicator params (SMA window or MACD params), odd iterations freeze core indicator and optimize volatility scaling parameters (`vol_scaling_window`, `vol_target`) via `spec.params["vol_target"]`.
- Fair budget: All frameworks loop until `len(self.evaluated_specs) >= self.config.max_backtests` or `proposals >= max_proposals`.
- Process tracking metrics returned in orchestrator output:
  - `total_proposals: int`
  - `parse_failures: int`
  - `reviewer_rejections: int`
  - `actual_backtests: int`

- [ ] **Step 1: Write failing tests for new topologies and fair budget loop**

In `tests/test_orchestrator.py`:
```python
def test_orchestrator_c_sota(sample_ohlcv_data):
    cfg = ExperimentConfig(framework_type="C-sota", max_backtests=4)
    evaluator = StrategyEvaluator(sample_ohlcv_data)
    orch = StrategyOrchestrator(cfg, evaluator)
    res = orch.run()
    assert res["framework_type"] == "C-sota"
    assert res["actual_backtests"] == 4
    assert "parse_failures" in res
    assert "reviewer_rejections" in res

def test_orchestrator_c_corr(sample_ohlcv_data):
    cfg = ExperimentConfig(framework_type="C-corr", max_backtests=3)
    evaluator = StrategyEvaluator(sample_ohlcv_data)
    orch = StrategyOrchestrator(cfg, evaluator)
    res = orch.run()
    assert res["framework_type"] == "C-corr"
    assert res["actual_backtests"] <= 3

def test_orchestrator_c_hier(sample_ohlcv_data):
    cfg = ExperimentConfig(framework_type="C-hier", max_backtests=4)
    evaluator = StrategyEvaluator(sample_ohlcv_data)
    orch = StrategyOrchestrator(cfg, evaluator)
    res = orch.run()
    assert res["framework_type"] == "C-hier"
    assert res["actual_backtests"] == 4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_orchestrator.py -v`
Expected: FAIL on `Unknown framework_type`.

- [ ] **Step 3: Implement new frameworks in StrategyOrchestrator**

In `agent/orchestrator.py`:
- Refactor search loop into a clean runner that respects `max_backtests` and tracks `total_proposals`, `parse_failures`, `reviewer_rejections`.
- Implement `_run_framework_c_sota()`: updates memory keeping best spec at index 0.
- Implement `_run_framework_c_corr()`: sets `reviewer = Reviewer(check_correlation=True, correlation_threshold=self.config.correlation_threshold, evaluator=self.evaluator)`.
- Implement `_run_framework_c_reflect()`: appends critique to memory.
- Implement `_run_framework_c_hier()`: toggles hierarchical search between indicator tuning and volatility scaling.
- Disallow fallback specs from being selected in `_finalize_selection_and_test()` if any valid specs exist.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_orchestrator.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agent/orchestrator.py tests/test_orchestrator.py
git commit -m "feat(agent): implement C-sota, C-corr, C-reflect, and C-hier topologies in orchestrator"
```

---

### Task 4: Experiment Grid Runner and Resumability Layer

**Files:**
- Create: `agent/grid.py`
- Test: `tests/test_grid.py`

**Interfaces:**
- `ExperimentRecord`: dataclass storing complete run information (model, infra, seed, max_backtests, proposals, parse_fails, rejections, search/val/test metrics, overfitting_gap, elapsed_time).
- `ExperimentGridRunner`:
  - `__init__(self, output_dir: Path, llm_factory: Optional[Callable[[str, int], BaseLLMClient]] = None)`
  - `run_experiment(self, config: ExperimentConfig) -> ExperimentRecord`
  - `run_grid(self, configs: List[ExperimentConfig], resume: bool = True) -> pd.DataFrame`
  - Appends every completed run into `runs.jsonl` (detailed traces) and `summary.csv` (aggregated metrics).
  - Skips execution if `(config.model, config.framework_type, config.seed)` already exists in `summary.csv`.

- [ ] **Step 1: Write failing tests for ExperimentGridRunner**

In `tests/test_grid.py`:
```python
def test_grid_runner_resumption(tmp_path, sample_ohlcv_data):
    from agent.grid import ExperimentGridRunner
    from agent.config import ExperimentConfig
    from agent.evaluator import StrategyEvaluator
    
    evaluator = StrategyEvaluator(sample_ohlcv_data)
    runner = ExperimentGridRunner(output_dir=tmp_path, evaluator=evaluator)
    
    cfg = ExperimentConfig(framework_type="C", model="mock", seed=42, max_backtests=2)
    # First run
    df1 = runner.run_grid([cfg])
    assert len(df1) == 1
    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "runs.jsonl").exists()
    
    # Second run with resume=True should skip and not re-evaluate
    df2 = runner.run_grid([cfg], resume=True)
    assert len(df2) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_grid.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent.grid'`.

- [ ] **Step 3: Implement ExperimentGridRunner**

In `agent/grid.py`:
- Create `ExperimentRecord` dataclass.
- Create `ExperimentGridRunner` with JSONL and CSV write helpers.
- Check existing `summary.csv` for `(model, framework, seed)` tuples when `resume=True`.
- Wrap execution in timing block and record `elapsed_seconds`.
- Catch and log exceptions gracefully so a transient failure does not abort the entire grid.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_grid.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agent/grid.py tests/test_grid.py
git commit -m "feat(agent): create ExperimentGridRunner with resumability and trace logging"
```

---

### Task 5: Dynamic LLM Client Factory Supporting Frontier Models

**Files:**
- Modify: `agent/llm.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- `get_llm_client(model: str, seed: int = 42, temperature: float = 0.7) -> BaseLLMClient`:
  - If `model == "mock"`: returns `MockLLMClient(seed=seed)`
  - Otherwise: loads `OPENAI_API_KEY` and `OPENAI_BASE_URL` from env (or `.env`), instantiates `OpenAICompatibleClient` with `model=model`, `temperature=temperature`.
  - Supports standard model identifiers:
    - Claude: `claude-sonnet-4-6`, `claude-opus-4-6`, `claude-fable-5-1`
    - OpenAI: `gpt-5.6-terra`
    - Domestic: `glm-5`, `qwen3-coder`
    - Baseline: `deepseek-chat`

- [ ] **Step 1: Write failing tests for get_llm_client**

In `tests/test_llm.py`:
```python
def test_get_llm_client_mock():
    from agent.llm import get_llm_client, MockLLMClient
    client = get_llm_client("mock", seed=123)
    assert isinstance(client, MockLLMClient)
    assert client.seed == 123

def test_get_llm_client_real_env(monkeypatch):
    from agent.llm import get_llm_client, OpenAICompatibleClient
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.example.com/v1")
    client = get_llm_client("claude-sonnet-4-6")
    assert isinstance(client, OpenAICompatibleClient)
    assert client.model == "claude-sonnet-4-6"
    assert client.api_key == "test-key"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_llm.py -v`
Expected: FAIL with `ImportError: cannot import name 'get_llm_client'`.

- [ ] **Step 3: Implement get_llm_client**

In `agent/llm.py`:
- Add `get_llm_client(model: str, seed: int = 42, temperature: float = 0.7) -> BaseLLMClient`.
- Read from `os.environ` and safely load `.env` if not loaded.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_llm.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agent/llm.py tests/test_llm.py
git commit -m "feat(agent): add unified get_llm_client factory for mock and frontier models"
```

---

### Task 6: CLI Runner for Phase 1 & Phase 2 Experiments

**Files:**
- Create: `scripts/run_experiment_grid.py`
- Test: manual smoke execution via shell (`--smoke`)

**Interfaces:**
- `python scripts/run_experiment_grid.py --mode [smoke|phase1|phase2] [--models ...] [--infras ...] [--seeds ...] [--budget 8]`
  - `--mode smoke`: 1 seed, MockLLM, tests grid write and resume.
  - `--mode phase1`: Evaluates candidate foundation models on framework `C` across seeds `42..46`.
  - `--mode phase2`: Evaluates agent topologies (`C`, `C-sota`, `C-corr`, `C-reflect`, `C-hier`) on the selected winner model across seeds `42..46`.
- Outputs formatted Markdown table of summary results upon completion.

- [ ] **Step 1: Write CLI script `scripts/run_experiment_grid.py`**

Implement argparse, dataset loading (`load_market_data('SPY')`), configuration builder, and invocation of `ExperimentGridRunner`.

- [ ] **Step 2: Smoke test CLI with mock models**

Run: `.venv/bin/python3 scripts/run_experiment_grid.py --mode smoke`
Expected: Exits 0, produces `docs/experiments/grid_smoke/summary.csv` with 1 record, prints clean table.

- [ ] **Step 3: Verify existing test suite is 100% green**

Run: `.venv/bin/pytest`
Expected: 30+ tests passing.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_experiment_grid.py
git commit -m "feat(scripts): add run_experiment_grid CLI for Phase 1 and Phase 2 benchmarks"
```

---

### Task 7: Execute Phase 1 Live Model Bake-Off (Smoke + Benchmark Run)

**Files:**
- Target Output: `docs/experiments/phase1_models/summary.csv`
- Target Output: `docs/experiments/phase1_models/runs.jsonl`

**Scope:**
- Compare:
  1. `deepseek-chat`
  2. `claude-sonnet-4-6`
  3. `gpt-5.6-terra`
  4. `glm-5`
- Fixed framework: `C` (Researcher + Reviewer)
- Seeds: 42 (smoke check first), then 42-44.
- Budget: `max_backtests = 4` (to manage cost/time responsibly on initial run).

- [ ] **Step 1: Execute single-seed probe across candidate models to verify zero 404/500 formatting issues**
- [ ] **Step 2: Execute Phase 1 grid**
- [ ] **Step 3: Print and verify the resulting comparison table**
- [ ] **Step 4: Commit benchmark results**

```bash
git add docs/experiments/phase1_models/
git commit -m "docs(experiments): record Phase 1 foundation model benchmark results"
```
