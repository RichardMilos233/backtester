import pytest
import pandas as pd
import numpy as np
from agent.config import ExperimentConfig
from agent.evaluator import StrategyEvaluator
from agent.orchestrator import StrategyOrchestrator
from agent.spec import StrategySpec
from agent.space import spec_key

@pytest.fixture
def synthetic_data():
    # 300 bars → ~180-bar search split (60%), enough for 50-day SMA without NaN/warnings
    n = 300
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(n))
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
    assert "search_metrics" in result
    assert "selection_metrics" in result
    assert "test_metrics" in result
    assert "overfitting_gap" in result
    assert result["framework_type"] == framework


@pytest.mark.parametrize("framework", ["C-sota", "C-corr", "C-reflect", "C-hier"])
def test_extended_frameworks_run_successfully(synthetic_data, framework):
    cfg = ExperimentConfig(
        framework_type=framework,
        max_backtests=4,
        memory_length=2,
        seed=42
    )
    evaluator = StrategyEvaluator(synthetic_data)
    orchestrator = StrategyOrchestrator(cfg, evaluator)
    result = orchestrator.run()

    assert "best_spec" in result
    assert "search_metrics" in result
    assert "selection_metrics" in result
    assert "test_metrics" in result
    assert "overfitting_gap" in result
    assert "total_proposals" in result
    assert "parse_failures" in result
    assert "reviewer_rejections" in result
    assert "actual_backtests" in result
    assert result["actual_backtests"] <= 4
    assert result["framework_type"] == framework


def test_orchestrator_disallows_fallback_selection(synthetic_data):
    cfg = ExperimentConfig(framework_type="B", max_backtests=2)
    evaluator = StrategyEvaluator(synthetic_data)
    orchestrator = StrategyOrchestrator(cfg, evaluator)

    valid_spec = StrategySpec("valid_1", "Valid test", "momentum", {"window": 30}, is_fallback=False)
    fallback_spec = StrategySpec("fallback_1", "Fallback test", "momentum", {"window": 50}, is_fallback=True)

    orchestrator.evaluated_specs = [valid_spec, fallback_spec]
    orchestrator.search_results = [
        evaluator.evaluate_search(valid_spec),
        evaluator.evaluate_search(fallback_spec)
    ]
    res = orchestrator._finalize_selection_and_test()
    assert res["best_spec"]["spec_id"] == "valid_1"




def test_c_hier_uses_full_budget_and_vol_overlay_is_not_flagged_duplicate(synthetic_data):
    """Regression: vol-scaled overlay of the best core spec used to be rejected as a duplicate
    (reviewer compared only signal_type/params), so C-hier burned every proposal and ran 1 backtest."""
    cfg = ExperimentConfig(framework_type="C-hier", max_backtests=4, memory_length=2, seed=42)
    orchestrator = StrategyOrchestrator(cfg, StrategyEvaluator(synthetic_data))
    result = orchestrator.run()
    assert result["actual_backtests"] == 4
    assert result["reviewer_rejections"] == 0
    assert any(s.use_vol_scaling for s in orchestrator.evaluated_specs)


def _no_llm_factory(seed):
    raise AssertionError("non-LLM baseline must not request an LLM client")


@pytest.mark.parametrize("framework", ["R-random", "R-grid"])
def test_non_llm_baselines_use_exact_budget(synthetic_data, framework):
    cfg = ExperimentConfig(framework_type=framework, max_backtests=4, seed=42, model="none")
    orchestrator = StrategyOrchestrator(cfg, StrategyEvaluator(synthetic_data), llm_client_factory=_no_llm_factory)
    result = orchestrator.run()
    assert result["actual_backtests"] == 4
    assert result["total_proposals"] == 4
    assert result["parse_failures"] == 0
    assert result["reviewer_rejections"] == 0
    assert not any(s.is_fallback for s in orchestrator.evaluated_specs)
    assert not result["best_spec"]["is_fallback"]
    prefix = "rand_" if framework == "R-random" else "grid_"
    assert all(s.spec_id.startswith(prefix) for s in orchestrator.evaluated_specs)


def test_r_grid_is_seed_independent(synthetic_data):
    evaluator = StrategyEvaluator(synthetic_data)
    runs = []
    for seed in (42, 43):
        cfg = ExperimentConfig(framework_type="R-grid", max_backtests=4, seed=seed, model="none")
        orch = StrategyOrchestrator(cfg, evaluator, llm_client_factory=_no_llm_factory)
        res = orch.run()
        runs.append((res["best_spec"]["spec_id"], [spec_key(s) for s in orch.evaluated_specs]))
    assert runs[0] == runs[1]


def test_r_random_samples_distinct_specs(synthetic_data):
    cfg = ExperimentConfig(framework_type="R-random", max_backtests=4, seed=42, model="none")
    orch = StrategyOrchestrator(cfg, StrategyEvaluator(synthetic_data), llm_client_factory=_no_llm_factory)
    orch.run()
    keys = [spec_key(s) for s in orch.evaluated_specs]
    assert len(keys) == len(set(keys)) == 4


def test_selection_never_picks_nan_score_spec(synthetic_data):
    """A flat-signal spec (window longer than available history) has NaN Val Sharpe; it must not win."""
    cfg = ExperimentConfig(framework_type="R-random", max_backtests=2, model="none")
    evaluator = StrategyEvaluator(synthetic_data)
    orch = StrategyOrchestrator(cfg, evaluator, llm_client_factory=_no_llm_factory)
    flat = StrategySpec("flat", "too long for 300 bars", "momentum", {"window": 260})
    normal = StrategySpec("normal", "normal", "momentum", {"window": 20})
    assert np.isnan(evaluator.evaluate_selection(flat)["sharpe_ratio"])
    orch.evaluated_specs = [flat, normal]
    orch.search_results = [evaluator.evaluate_search(flat), evaluator.evaluate_search(normal)]
    assert orch._finalize_selection_and_test()["best_spec"]["spec_id"] == "normal"
