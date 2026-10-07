"""Tests for new signal generators in src/signals.py."""
import numpy as np
import pandas as pd
import pytest

from src.signals import (
    compute_breakout_signal,
    compute_mean_reversion_signal,
    compute_rsi_signal,
    combine_signals,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(close, high=None, low=None):
    """Build a minimal OHLCV DataFrame from a close array."""
    n = len(close)
    close = np.asarray(close, dtype=float)
    if high is None:
        high = close * 1.005
    if low is None:
        low = close * 0.995
    return pd.DataFrame(
        {
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.ones(n) * 1_000,
        }
    )


def _flat_then_drop(flat_val=100.0, drop_val=85.0, flat_n=50, drop_n=10):
    """Flat series followed by a sharp drop – triggers mean reversion."""
    close = [flat_val] * flat_n + [drop_val] * drop_n
    return _make_df(close)


def _steady_rise(start=100.0, step=1.0, n=120):
    """Steadily rising price series."""
    close = [start + i * step for i in range(n)]
    return _make_df(close)


def _sustained_decline(start=100.0, step=0.5, n=100):
    """Steadily falling price – RSI reversion should eventually fire."""
    close = [start - i * step for i in range(n)]
    close = [max(c, 1.0) for c in close]
    return _make_df(close)


# ---------------------------------------------------------------------------
# compute_mean_reversion_signal
# ---------------------------------------------------------------------------

class TestMeanReversionSignal:

    def test_output_dtype(self):
        df = _flat_then_drop()
        sig = compute_mean_reversion_signal(df)
        assert sig.dtype == int

    def test_values_binary(self):
        df = _flat_then_drop()
        sig = compute_mean_reversion_signal(df)
        assert set(sig.unique()).issubset({0, 1})

    def test_index_aligned(self):
        df = _flat_then_drop()
        sig = compute_mean_reversion_signal(df)
        assert sig.index.equals(df.index)

    def test_warmup_rows_are_zero(self):
        window = 20
        df = _flat_then_drop()
        sig = compute_mean_reversion_signal(df, window=window)
        assert (sig.iloc[:window] == 0).all()

    def test_no_lookahead(self):
        df = _flat_then_drop(flat_n=50, drop_n=10)
        sig_full = compute_mean_reversion_signal(df, window=20)
        for k in [40, 50, 55]:
            sig_prefix = compute_mean_reversion_signal(df.iloc[:k], window=20)
            pd.testing.assert_series_equal(
                sig_full.iloc[:k].reset_index(drop=True),
                sig_prefix.reset_index(drop=True),
            )

    def test_goes_long_after_sharp_drop(self):
        # 50 flat bars at 100, then 10 bars at 85
        # z-score should dip well below -1 once std is established
        df = _flat_then_drop(flat_val=100.0, drop_val=85.0, flat_n=50, drop_n=10)
        sig = compute_mean_reversion_signal(df, window=20, entry_z=-1.0, exit_z=0.0)
        # Should be long in some of the drop rows
        assert sig.iloc[50:].sum() > 0

    def test_invalid_params_raise(self):
        df = _flat_then_drop()
        with pytest.raises(ValueError):
            compute_mean_reversion_signal(df, entry_z=0.0, exit_z=-1.0)
        with pytest.raises(ValueError):
            compute_mean_reversion_signal(df, entry_z=0.0, exit_z=0.0)


# ---------------------------------------------------------------------------
# compute_breakout_signal
# ---------------------------------------------------------------------------

class TestBreakoutSignal:

    def test_output_dtype(self):
        df = _steady_rise()
        sig = compute_breakout_signal(df)
        assert sig.dtype == int

    def test_values_binary(self):
        df = _steady_rise()
        sig = compute_breakout_signal(df)
        assert set(sig.unique()).issubset({0, 1})

    def test_index_aligned(self):
        df = _steady_rise()
        sig = compute_breakout_signal(df)
        assert sig.index.equals(df.index)

    def test_warmup_rows_are_zero(self):
        entry_window = 55
        df = _steady_rise(n=200)
        sig = compute_breakout_signal(df, entry_window=entry_window)
        # Before enough history for the shifted rolling max, must be 0
        assert (sig.iloc[:entry_window] == 0).all()

    def test_no_lookahead(self):
        df = _steady_rise(n=200)
        sig_full = compute_breakout_signal(df, entry_window=55, exit_window=20)
        for k in [60, 100, 150]:
            sig_prefix = compute_breakout_signal(
                df.iloc[:k], entry_window=55, exit_window=20
            )
            pd.testing.assert_series_equal(
                sig_full.iloc[:k].reset_index(drop=True),
                sig_prefix.reset_index(drop=True),
            )

    def test_goes_long_after_new_high(self):
        # 60 flat bars then a strong new high
        flat = [100.0] * 60
        spike = [120.0] * 20
        close = flat + spike
        df = _make_df(close)
        sig = compute_breakout_signal(df, entry_window=55, exit_window=20)
        assert sig.iloc[60:].sum() > 0

    def test_invalid_params_raise(self):
        df = _steady_rise()
        with pytest.raises(ValueError):
            compute_breakout_signal(df, entry_window=0)
        with pytest.raises(ValueError):
            compute_breakout_signal(df, exit_window=-5)


# ---------------------------------------------------------------------------
# compute_rsi_signal
# ---------------------------------------------------------------------------

class TestRsiSignal:

    def test_output_dtype_reversion(self):
        df = _sustained_decline()
        sig = compute_rsi_signal(df, mode="reversion")
        assert sig.dtype == int

    def test_output_dtype_trend(self):
        df = _steady_rise()
        sig = compute_rsi_signal(df, mode="trend")
        assert sig.dtype == int

    def test_values_binary_reversion(self):
        df = _sustained_decline()
        sig = compute_rsi_signal(df, mode="reversion")
        assert set(sig.unique()).issubset({0, 1})

    def test_values_binary_trend(self):
        df = _steady_rise()
        sig = compute_rsi_signal(df, mode="trend")
        assert set(sig.unique()).issubset({0, 1})

    def test_index_aligned(self):
        df = _sustained_decline()
        sig = compute_rsi_signal(df)
        assert sig.index.equals(df.index)

    def test_warmup_rows_are_zero(self):
        window = 14
        df = _sustained_decline(n=100)
        sig = compute_rsi_signal(df, window=window)
        assert (sig.iloc[:window] == 0).all()

    def test_no_lookahead_reversion(self):
        df = _sustained_decline(n=100)
        sig_full = compute_rsi_signal(df, mode="reversion")
        for k in [30, 60, 80]:
            sig_prefix = compute_rsi_signal(df.iloc[:k], mode="reversion")
            pd.testing.assert_series_equal(
                sig_full.iloc[:k].reset_index(drop=True),
                sig_prefix.reset_index(drop=True),
            )

    def test_no_lookahead_trend(self):
        df = _steady_rise(n=100)
        sig_full = compute_rsi_signal(df, mode="trend")
        for k in [30, 60, 80]:
            sig_prefix = compute_rsi_signal(df.iloc[:k], mode="trend")
            pd.testing.assert_series_equal(
                sig_full.iloc[:k].reset_index(drop=True),
                sig_prefix.reset_index(drop=True),
            )

    def test_reversion_goes_long_after_sustained_decline(self):
        df = _sustained_decline(start=100.0, step=0.8, n=80)
        sig = compute_rsi_signal(df, window=14, lower=30.0, upper=70.0, mode="reversion")
        assert sig.sum() > 0

    def test_trend_long_in_rising_series(self):
        df = _steady_rise(n=80)
        sig = compute_rsi_signal(df, window=14, mode="trend")
        # After warm-up most bars should be 1 in a rising series
        assert sig.iloc[14:].sum() > 0

    def test_invalid_mode_raises(self):
        df = _steady_rise()
        with pytest.raises(ValueError):
            compute_rsi_signal(df, mode="unknown")

    def test_invalid_bounds_raise(self):
        df = _steady_rise()
        with pytest.raises(ValueError):
            compute_rsi_signal(df, lower=70.0, upper=30.0)


# ---------------------------------------------------------------------------
# combine_signals
# ---------------------------------------------------------------------------

class TestCombineSignals:

    def _sigs(self):
        idx = pd.RangeIndex(5)
        a = pd.Series([1, 1, 0, 0, 1], index=idx)
        b = pd.Series([1, 0, 1, 0, 1], index=idx)
        c = pd.Series([0, 0, 1, 0, 1], index=idx)
        return a, b, c

    def test_all_mode(self):
        a, b, c = self._sigs()
        result = combine_signals([a, b, c], mode="all")
        expected = pd.Series([0, 0, 0, 0, 1])
        pd.testing.assert_series_equal(result.reset_index(drop=True), expected)

    def test_any_mode(self):
        a, b, c = self._sigs()
        result = combine_signals([a, b, c], mode="any")
        expected = pd.Series([1, 1, 1, 0, 1])
        pd.testing.assert_series_equal(result.reset_index(drop=True), expected)

    def test_majority_mode(self):
        a, b, c = self._sigs()
        result = combine_signals([a, b, c], mode="majority")
        # majority (>1.5): rows 0=[1,1,0]->2>1.5->1, 1=[1,0,0]->1->0,
        # 2=[0,1,1]->2->1, 3=[0,0,0]->0->0, 4=[1,1,1]->3->1
        expected = pd.Series([1, 0, 1, 0, 1])
        pd.testing.assert_series_equal(result.reset_index(drop=True), expected)

    def test_output_dtype(self):
        a, b, _ = self._sigs()
        result = combine_signals([a, b], mode="all")
        assert result.dtype == int

    def test_values_binary(self):
        a, b, c = self._sigs()
        for mode in ("all", "any", "majority"):
            result = combine_signals([a, b, c], mode=mode)
            assert set(result.unique()).issubset({0, 1})

    def test_union_index_filled_with_zero(self):
        idx1 = pd.RangeIndex(3)
        idx2 = pd.RangeIndex(2, 5)
        a = pd.Series([1, 1, 1], index=idx1)
        b = pd.Series([1, 1, 1], index=idx2)
        result = combine_signals([a, b], mode="all")
        # Union index 0..4; positions 0,1 only in a; 3,4 only in b -> filled 0
        assert result.loc[0] == 0  # b missing -> filled 0 -> all=0
        assert result.loc[2] == 1  # both present
        assert result.loc[4] == 0  # a missing -> all=0

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            combine_signals([], mode="all")

    def test_invalid_mode_raises(self):
        a, _, _ = self._sigs()
        with pytest.raises(ValueError):
            combine_signals([a], mode="wrong")
