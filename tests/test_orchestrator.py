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
