import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from agent.config import ExperimentConfig
from agent.evaluator import StrategyEvaluator


@pytest.fixture
def synthetic_data():
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


def test_grid_runner_resumption_and_logging(tmp_path, synthetic_data):
    from agent.grid import ExperimentGridRunner

    evaluator = StrategyEvaluator(synthetic_data)
    runner = ExperimentGridRunner(output_dir=tmp_path, evaluator=evaluator)

    cfg1 = ExperimentConfig(framework_type="C", model="mock", seed=42, max_backtests=2)
    cfg2 = ExperimentConfig(framework_type="C", model="mock", seed=43, max_backtests=2)

    # First run: execute both
    df1 = runner.run_grid([cfg1, cfg2], resume=True)
    assert len(df1) == 2
    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "runs.jsonl").exists()

    # Second run with same configs: resume=True should detect completed and not re-run
    # Track runs_jsonl line count
    with open(tmp_path / "runs.jsonl") as f:
        lines_before = len(f.readlines())
    assert lines_before == 2

    df2 = runner.run_grid([cfg1, cfg2], resume=True)
    assert len(df2) == 2

    with open(tmp_path / "runs.jsonl") as f:
        lines_after = len(f.readlines())
    assert lines_after == 2  # No new lines written because of resumption
