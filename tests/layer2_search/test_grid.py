import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from benchmark.common.config import ExperimentConfig
from benchmark.common.evaluator import StrategyEvaluator


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
    from benchmark.layer2_search.grid import ExperimentGridRunner

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


def test_grid_runner_scores_against_oracle_and_records_usage(tmp_path, synthetic_data):
    """Rows carry task_id, llm usage and oracle-normalised scores when an oracle table exists."""
    import benchmark.layer2_search.space as space
    from benchmark.layer2_search.oracle import build_oracle_table, save_oracle
    from benchmark.common.config import ExperimentConfig
    from benchmark.common.evaluator import StrategyEvaluator
    from benchmark.layer2_search.grid import ExperimentGridRunner

    ev = StrategyEvaluator(synthetic_data)
    orig = (space.MOMENTUM_WINDOWS, space.MACD_FAST, space.MACD_SLOW, space.MACD_SPAN, space.VOL_TARGETS)
    space.MOMENTUM_WINDOWS, space.MACD_FAST, space.MACD_SLOW, space.MACD_SPAN, space.VOL_TARGETS = \
        [10, 20, 30, 50], [8], [20], [9], [None]
    try:
        oracle_dir = tmp_path / "oracle"
        save_oracle(build_oracle_table(ev), oracle_dir / "SPY_full.csv.gz")
        runner = ExperimentGridRunner(tmp_path / "out", evaluator=ev, oracle_dir=oracle_dir)
        rows = runner.run_grid([
            ExperimentConfig(task_id="SPY_full", model="mock", framework_type="C", seed=42, max_backtests=2),
            ExperimentConfig(task_id="SPY_full", model="none", framework_type="R-grid", seed=42, max_backtests=2),
        ])
    finally:
        space.MOMENTUM_WINDOWS, space.MACD_FAST, space.MACD_SLOW, space.MACD_SPAN, space.VOL_TARGETS = orig

    assert list(rows["status"]) == ["success", "success"]
    assert set(rows["task_id"]) == {"SPY_full"}
    c_row = rows[rows.framework_type == "C"].iloc[0]
    assert c_row["llm_calls"] >= 2 and c_row["llm_total_tokens"] > 0
    assert 0 <= c_row["percentile_val"] <= 1
    assert c_row["budget_utilization"] == 1.0
    g_row = rows[rows.framework_type == "R-grid"].iloc[0]
    assert g_row["llm_calls"] == 0
    assert g_row["on_lattice"] in (True, "True")
    # resume key includes task_id: nothing re-runs
    again = runner.run_grid([ExperimentConfig(task_id="SPY_full", model="mock", framework_type="C", seed=42, max_backtests=2)])
    assert len(again) == 2
