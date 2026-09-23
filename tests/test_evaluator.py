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


def test_causal_isolation_no_lookahead(dummy_market_data):
    """Signals on a split slice must match signals computed only up to that point (no future bars)."""
    from src.signals import compute_moving_average_signal

    evaluator = StrategyEvaluator(dummy_market_data)
    spec = StrategySpec("m1", "causal check", "momentum", {"window": 10})
    search_df, select_df, test_df = evaluator.split_data()

    # Signals as produced by evaluator for the select split
    _, split_signals = evaluator._generate_signals_for_split(spec, select_df)

    # Recompute using only data up through the end of select (no test-period future)
    end_loc = dummy_market_data.index.get_loc(select_df.index[-1])
    prefix_df = dummy_market_data.iloc[: end_loc + 1]
    prefix_signals = compute_moving_average_signal(prefix_df, window=spec.params["window"])
    expected = prefix_signals.loc[select_df.index].fillna(0).astype(int)

    assert split_signals.equals(expected)
