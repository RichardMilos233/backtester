import pytest
import pandas as pd
import numpy as np
from agent.config import ExperimentConfig
from agent.evaluator import StrategyEvaluator
from agent.orchestrator import StrategyOrchestrator
from agent.spec import StrategySpec

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


