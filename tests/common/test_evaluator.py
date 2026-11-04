import pytest
import pandas as pd
import numpy as np
from benchmark.common.config import DataSplitConfig
from benchmark.common.spec import StrategySpec
from benchmark.common.evaluator import StrategyEvaluator

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


# --- Layer 1: regime mask tests ---

@pytest.fixture
def regime_market_data():
    """Longer frame so regime classifier has enough warm-up (~1.25 years = ~315 bars)."""
    n = 800
    dates = pd.date_range("2018-01-01", periods=n, freq="B")
    rng = np.random.default_rng(7)
    prices = 100 + np.cumsum(rng.standard_normal(n))
    volume = np.abs(rng.standard_normal(n)) * 1e6 + 1e6
    return pd.DataFrame({
        "open": prices, "high": prices + 1, "low": prices - 1,
        "close": prices, "volume": volume,
    }, index=dates)


def test_regime_mask_zeroes_signal_outside_allowed(regime_market_data):
    """Signal must be 0 on days whose regime is NOT in allowed_regimes."""
    evaluator = StrategyEvaluator(regime_market_data)
    spec_all = StrategySpec("a", "all", "momentum", {"window": 20})
    spec_q1 = StrategySpec("b", "q1 only", "momentum", {"window": 20},
                           allowed_regimes=["Q1"])

    _, sig_all = evaluator._generate_signals_for_split(spec_all, evaluator._select_df)
    _, sig_q1 = evaluator._generate_signals_for_split(spec_q1, evaluator._select_df)

    regimes = evaluator._get_regimes().reindex(evaluator._select_df.index)
    # On days that are NOT Q1 and have a valid regime, signal must be 0
    non_q1_valid = regimes.notna() & (regimes != "Q1")
    assert (sig_q1[non_q1_valid] == 0).all(), "Signal should be 0 outside allowed regimes"


def test_regime_nan_days_are_allowed(regime_market_data):
    """NaN-regime warm-up days must NOT be zeroed out."""
    evaluator = StrategyEvaluator(regime_market_data)
    spec = StrategySpec("c", "q1 only", "momentum", {"window": 20},
                        allowed_regimes=["Q1"])

    _, sig = evaluator._generate_signals_for_split(spec, evaluator._search_df)
    regimes = evaluator._get_regimes().reindex(evaluator._search_df.index)

    nan_days = regimes.isna()
    if nan_days.any():
        # On NaN days the underlying signal (0 or 1) is preserved; confirm it isn't
        # forced to 0 by the regime mask
        spec_base = StrategySpec("d", "no regime", "momentum", {"window": 20})
        _, sig_base = evaluator._generate_signals_for_split(spec_base, evaluator._search_df)
        assert sig[nan_days].equals(sig_base[nan_days]), (
            "NaN-regime warm-up days should not be zeroed by regime mask"
        )


def test_all_four_regimes_same_as_no_filter(regime_market_data):
    """allowed_regimes=['Q1','Q2','Q3','Q4'] must give identical metrics to None."""
    evaluator = StrategyEvaluator(regime_market_data)
    spec_none = StrategySpec("e", "no regime filter", "momentum", {"window": 20})
    spec_all = StrategySpec("f", "all regimes", "momentum", {"window": 20},
                            allowed_regimes=["Q1", "Q2", "Q3", "Q4"])

    m_none = evaluator.evaluate_spec(spec_none, "select")
    m_all = evaluator.evaluate_spec(spec_all, "select")

    assert abs(m_none["sharpe_ratio"] - m_all["sharpe_ratio"]) < 1e-9
    assert abs(m_none["total_return"] - m_all["total_return"]) < 1e-9


def test_regime_mask_applied_before_vol_scaling(regime_market_data):
    """Regime-filtered vol-scaled spec must differ from unfiltered vol-scaled spec."""
    evaluator = StrategyEvaluator(regime_market_data)
    spec_vol = StrategySpec("g", "vol no regime", "momentum", {"window": 20},
                            use_vol_scaling=True, vol_target=0.15, max_leverage=1.5)
    spec_vol_regime = StrategySpec("h", "vol with regime", "momentum", {"window": 20},
                                   use_vol_scaling=True, vol_target=0.15, max_leverage=1.5,
                                   allowed_regimes=["Q1", "Q4"])

    regimes = evaluator._get_regimes().reindex(evaluator._select_df.index)
    has_filtered_days = (regimes.notna() & ~regimes.isin(["Q1", "Q4"])).any()
    if not has_filtered_days:
        pytest.skip("No filtered regime days in select split for this fixture")

    _, sig_vol = evaluator._generate_signals_for_split(spec_vol, evaluator._select_df)
    _, sig_vol_r = evaluator._generate_signals_for_split(spec_vol_regime, evaluator._select_df)
    assert not sig_vol.equals(sig_vol_r), "Vol-scaled signal must differ when regime filter active"

