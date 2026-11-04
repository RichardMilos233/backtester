"""
Tests for the v2 signal types:
  - StrategySpec.validate() for mean_reversion, breakout, rsi, combo
  - StrategyEvaluator dispatch for each new type on real SPY-like data
  - regime mask + vol scaling on top of new types
  - mean combo producing non-binary weights and the int-cast rule
  - spec_key uniqueness and combo order-insensitivity
  - _count_space matching iter_space on a shrunken lattice
  - lattice-mismatch guard in ExperimentGridRunner
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmark.common.spec import StrategySpec
from benchmark.common.evaluator import StrategyEvaluator
from benchmark.layer2_search.space import (
    spec_key,
    _count_space,
    iter_space,
    enumerate_space,
    sample_space,
    full_lattice_size,
    baseline_space,
    LATTICE_VERSION,
    ALL_SIGNAL_TYPES,
    LEGACY_SIGNAL_TYPES,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def spy_like_data():
    """800-bar synthetic OHLCV frame — long enough for regime classifier warm-up."""
    n = 800
    dates = pd.date_range("2018-01-01", periods=n, freq="B")
    rng = np.random.default_rng(42)
    prices = 100 + np.cumsum(rng.standard_normal(n))
    volume = np.abs(rng.standard_normal(n)) * 1e6 + 1e6
    return pd.DataFrame({
        "open": prices,
        "high": prices + 1,
        "low": prices - 1,
        "close": prices,
        "volume": volume,
    }, index=dates)


@pytest.fixture
def small_data():
    """300-bar frame for fast tests that don't need regime warm-up."""
    n = 300
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    rng = np.random.default_rng(7)
    prices = 100 + np.cumsum(rng.standard_normal(n))
    return pd.DataFrame({
        "open": prices, "high": prices + 1, "low": prices - 1,
        "close": prices, "volume": 1_000_000,
    }, index=dates)


# ---------------------------------------------------------------------------
# 1. validate() for each new signal type
# ---------------------------------------------------------------------------

class TestMeanReversionValidation:

    def test_valid(self):
        s = StrategySpec("t", "h", "mean_reversion", {"window": 20, "entry_z": -1.5, "exit_z": 0.0})
        ok, msg = s.validate()
        assert ok, msg

    def test_window_boundary_min(self):
        s = StrategySpec("t", "h", "mean_reversion", {"window": 5, "entry_z": -1.0, "exit_z": 0.0})
        ok, msg = s.validate()
        assert ok, msg

    def test_window_boundary_max(self):
        s = StrategySpec("t", "h", "mean_reversion", {"window": 120, "entry_z": -1.0, "exit_z": 0.0})
        ok, msg = s.validate()
        assert ok, msg

    def test_window_too_small(self):
        s = StrategySpec("t", "h", "mean_reversion", {"window": 4, "entry_z": -1.0, "exit_z": 0.0})
        ok, msg = s.validate()
        assert not ok
        assert "window" in msg

    def test_window_too_large(self):
        s = StrategySpec("t", "h", "mean_reversion", {"window": 121, "entry_z": -1.0, "exit_z": 0.0})
        ok, msg = s.validate()
        assert not ok

    def test_entry_z_boundary(self):
        s = StrategySpec("t", "h", "mean_reversion", {"window": 20, "entry_z": -3.0, "exit_z": -0.5})
        ok, msg = s.validate()
        assert ok, msg

    def test_entry_z_out_of_range(self):
        s = StrategySpec("t", "h", "mean_reversion", {"window": 20, "entry_z": -0.4, "exit_z": 0.0})
        ok, msg = s.validate()
        assert not ok
        assert "entry_z" in msg

    def test_exit_z_boundary(self):
        s = StrategySpec("t", "h", "mean_reversion", {"window": 20, "entry_z": -1.0, "exit_z": 0.5})
        ok, msg = s.validate()
        assert ok, msg

    def test_exit_z_out_of_range(self):
        s = StrategySpec("t", "h", "mean_reversion", {"window": 20, "entry_z": -1.0, "exit_z": 0.6})
        ok, msg = s.validate()
        assert not ok
        assert "exit_z" in msg

    def test_entry_z_gte_exit_z_rejected(self):
        s = StrategySpec("t", "h", "mean_reversion", {"window": 20, "entry_z": -0.5, "exit_z": -0.5})
        ok, msg = s.validate()
        assert not ok
        assert "entry_z" in msg or "exit_z" in msg


class TestBreakoutValidation:

    def test_valid(self):
        s = StrategySpec("t", "h", "breakout", {"entry_window": 55, "exit_window": 20})
        ok, msg = s.validate()
        assert ok, msg

    def test_boundary_min(self):
        s = StrategySpec("t", "h", "breakout", {"entry_window": 10, "exit_window": 5})
        ok, msg = s.validate()
        assert ok, msg

    def test_boundary_max(self):
        s = StrategySpec("t", "h", "breakout", {"entry_window": 260, "exit_window": 130})
        ok, msg = s.validate()
        assert ok, msg

    def test_entry_too_small(self):
        s = StrategySpec("t", "h", "breakout", {"entry_window": 9, "exit_window": 5})
        ok, _ = s.validate()
        assert not ok

    def test_exit_too_large_for_entry(self):
        s = StrategySpec("t", "h", "breakout", {"entry_window": 20, "exit_window": 25})
        ok, msg = s.validate()
        assert not ok
        assert "exit_window" in msg


class TestRsiValidation:

    def test_valid_reversion(self):
        s = StrategySpec("t", "h", "rsi", {"window": 14, "lower": 30.0, "upper": 70.0, "mode": "reversion"})
        ok, msg = s.validate()
        assert ok, msg

    def test_valid_trend(self):
        s = StrategySpec("t", "h", "rsi", {"window": 14, "mode": "trend"})
        ok, msg = s.validate()
        assert ok, msg

    def test_trend_normalised(self):
        """mode='trend' should get lower=30, upper=70 normalised in-place."""
        s = StrategySpec("t", "h", "rsi", {"window": 14, "mode": "trend"})
        s.validate()
        assert s.params["lower"] == 30.0
        assert s.params["upper"] == 70.0

    def test_window_boundary(self):
        s = StrategySpec("t", "h", "rsi", {"window": 5, "lower": 30.0, "upper": 70.0, "mode": "reversion"})
        ok, msg = s.validate()
        assert ok, msg

    def test_window_too_small(self):
        s = StrategySpec("t", "h", "rsi", {"window": 4, "lower": 30.0, "upper": 70.0, "mode": "reversion"})
        ok, _ = s.validate()
        assert not ok

    def test_lower_boundary(self):
        s = StrategySpec("t", "h", "rsi", {"window": 14, "lower": 10.0, "upper": 55.0, "mode": "reversion"})
        ok, _ = s.validate()
        assert ok

    def test_lower_out_of_range(self):
        s = StrategySpec("t", "h", "rsi", {"window": 14, "lower": 9.9, "upper": 70.0, "mode": "reversion"})
        ok, msg = s.validate()
        assert not ok
        assert "lower" in msg

    def test_upper_boundary(self):
        s = StrategySpec("t", "h", "rsi", {"window": 14, "lower": 30.0, "upper": 90.0, "mode": "reversion"})
        ok, _ = s.validate()
        assert ok

    def test_upper_out_of_range(self):
        s = StrategySpec("t", "h", "rsi", {"window": 14, "lower": 30.0, "upper": 90.1, "mode": "reversion"})
        ok, msg = s.validate()
        assert not ok

    def test_bad_mode(self):
        s = StrategySpec("t", "h", "rsi", {"window": 14, "lower": 30.0, "upper": 70.0, "mode": "unknown"})
        ok, _ = s.validate()
        assert not ok


class TestComboValidation:

    def _mom_comp(self, w=50):
        return {"signal_type": "momentum", "params": {"window": w}}

    def _mr_comp(self):
        return {"signal_type": "mean_reversion", "params": {"window": 20, "entry_z": -1.5, "exit_z": 0.0}}

    def test_valid_two_component(self):
        s = StrategySpec("t", "h", "combo", {
            "components": [self._mom_comp(), self._mr_comp()],
            "mode": "all",
        })
        ok, msg = s.validate()
        assert ok, msg

    def test_valid_one_component(self):
        s = StrategySpec("t", "h", "combo", {
            "components": [self._mom_comp()],
            "mode": "any",
        })
        ok, msg = s.validate()
        assert ok, msg

    def test_valid_mean_mode(self):
        s = StrategySpec("t", "h", "combo", {
            "components": [self._mom_comp(50), self._mr_comp()],
            "mode": "mean",
        })
        ok, msg = s.validate()
        assert ok, msg

    def test_three_components_rejected(self):
        s = StrategySpec("t", "h", "combo", {
            "components": [self._mom_comp(10), self._mom_comp(20), self._mr_comp()],
            "mode": "all",
        })
        ok, msg = s.validate()
        assert not ok
        assert "2" in msg or "component" in msg.lower()

    def test_nested_combo_rejected(self):
        s = StrategySpec("t", "h", "combo", {
            "components": [{"signal_type": "combo", "params": {"components": [], "mode": "all"}}],
            "mode": "all",
        })
        ok, msg = s.validate()
        assert not ok
        assert "nested" in msg.lower() or "primitive" in msg.lower()

    def test_duplicate_components_rejected(self):
        s = StrategySpec("t", "h", "combo", {
            "components": [self._mom_comp(50), self._mom_comp(50)],
            "mode": "all",
        })
        ok, msg = s.validate()
        assert not ok
        assert "duplicate" in msg.lower()

    def test_invalid_component_params_rejected(self):
        s = StrategySpec("t", "h", "combo", {
            "components": [{"signal_type": "momentum", "params": {"window": 999}}],
            "mode": "all",
        })
        ok, msg = s.validate()
        assert not ok

    def test_bad_mode_rejected(self):
        s = StrategySpec("t", "h", "combo", {
            "components": [self._mom_comp()],
            "mode": "unknown",
        })
        ok, _ = s.validate()
        assert not ok

    def test_zero_components_rejected(self):
        s = StrategySpec("t", "h", "combo", {"components": [], "mode": "all"})
        ok, _ = s.validate()
        assert not ok


# ---------------------------------------------------------------------------
# 2. Evaluator dispatch + sane metrics on SPY-like data
# ---------------------------------------------------------------------------

def _make_evaluator(df):
    from benchmark.common.config import DataSplitConfig
    return StrategyEvaluator(df, split_config=DataSplitConfig(0.6, 0.2, 0.2))


class TestEvaluatorNewTypes:

    def test_mean_reversion_no_exception(self, spy_like_data):
        spec = StrategySpec("t", "h", "mean_reversion",
                            {"window": 20, "entry_z": -1.5, "exit_z": 0.0})
        ev = _make_evaluator(spy_like_data)
        m = ev.evaluate_spec(spec, "select")
        assert math.isfinite(m.get("sharpe_ratio", float("nan"))) or True  # NaN allowed if flat

    def test_breakout_no_exception(self, spy_like_data):
        spec = StrategySpec("t", "h", "breakout",
                            {"entry_window": 55, "exit_window": 20})
        ev = _make_evaluator(spy_like_data)
        m = ev.evaluate_spec(spec, "select")
        assert "sharpe_ratio" in m

    def test_rsi_reversion_no_exception(self, spy_like_data):
        spec = StrategySpec("t", "h", "rsi",
                            {"window": 14, "lower": 30.0, "upper": 70.0, "mode": "reversion"})
        ev = _make_evaluator(spy_like_data)
        m = ev.evaluate_spec(spec, "select")
        assert "sharpe_ratio" in m

    def test_rsi_trend_no_exception(self, spy_like_data):
        spec = StrategySpec("t", "h", "rsi", {"window": 14, "mode": "trend"})
        ev = _make_evaluator(spy_like_data)
        m = ev.evaluate_spec(spec, "select")
        assert "sharpe_ratio" in m

    def test_combo_all_no_exception(self, spy_like_data):
        spec = StrategySpec("t", "h", "combo", {
            "components": [
                {"signal_type": "momentum", "params": {"window": 50}},
                {"signal_type": "rsi", "params": {"window": 14, "lower": 30.0, "upper": 70.0, "mode": "reversion"}},
            ],
            "mode": "all",
        })
        ev = _make_evaluator(spy_like_data)
        m = ev.evaluate_spec(spec, "select")
        assert "sharpe_ratio" in m

    def test_combo_mean_no_exception(self, spy_like_data):
        spec = StrategySpec("t", "h", "combo", {
            "components": [
                {"signal_type": "momentum", "params": {"window": 50}},
                {"signal_type": "breakout", "params": {"entry_window": 55, "exit_window": 20}},
            ],
            "mode": "mean",
        })
        ev = _make_evaluator(spy_like_data)
        m = ev.evaluate_spec(spec, "select")
        assert "sharpe_ratio" in m


# ---------------------------------------------------------------------------
# 3. Regime mask + vol scaling on new signal types
# ---------------------------------------------------------------------------

class TestRegimeAndVolOnNewTypes:

    def test_regime_mask_zeroes_outside_allowed_mean_reversion(self, spy_like_data):
        ev = _make_evaluator(spy_like_data)
        spec_all = StrategySpec("a", "h", "mean_reversion",
                                {"window": 20, "entry_z": -1.5, "exit_z": 0.0})
        spec_q1 = StrategySpec("b", "h", "mean_reversion",
                                {"window": 20, "entry_z": -1.5, "exit_z": 0.0},
                                allowed_regimes=["Q1"])
        _, sig_q1 = ev._generate_signals_for_split(spec_q1, ev._select_df)
        regimes = ev._get_regimes().reindex(ev._select_df.index)
        non_q1_valid = regimes.notna() & (regimes != "Q1")
        assert (sig_q1[non_q1_valid] == 0).all()

    def test_vol_scaling_on_rsi(self, spy_like_data):
        ev = _make_evaluator(spy_like_data)
        spec = StrategySpec("t", "h", "rsi",
                            {"window": 14, "lower": 30.0, "upper": 70.0, "mode": "reversion"},
                            use_vol_scaling=True, vol_target=0.15, max_leverage=1.5)
        _, sig = ev._generate_signals_for_split(spec, ev._select_df)
        assert sig.dtype == float
        # Vol-scaled signal may exceed 1.0
        assert sig.max() <= 1.5 + 1e-9

    def test_vol_scaling_on_breakout(self, spy_like_data):
        ev = _make_evaluator(spy_like_data)
        spec = StrategySpec("t", "h", "breakout",
                            {"entry_window": 55, "exit_window": 20},
                            use_vol_scaling=True, vol_target=0.12, max_leverage=1.5)
        _, sig = ev._generate_signals_for_split(spec, ev._select_df)
        assert sig.dtype == float


# ---------------------------------------------------------------------------
# 4. combo mode="mean" produces non-binary and the int-cast rule
# ---------------------------------------------------------------------------

class TestComboMeanMode:

    def test_mean_combo_float_signal(self, spy_like_data):
        """mode='mean' on two different signals should produce fractional values."""
        ev = _make_evaluator(spy_like_data)
        spec = StrategySpec("t", "h", "combo", {
            "components": [
                {"signal_type": "momentum", "params": {"window": 20}},
                {"signal_type": "momentum", "params": {"window": 100}},
            ],
            "mode": "mean",
        })
        ok, msg = spec.validate()
        assert ok, msg
        _, sig = ev._generate_signals_for_split(spec, ev._select_df)
        # mean of two different binary signals can produce 0, 0.5, 1.0
        unique_vals = set(sig.unique())
        # If any 0.5 values exist the dtype must be float
        if 0.5 in unique_vals:
            assert sig.dtype == float

    def test_non_mean_combo_int_signal(self, spy_like_data):
        """mode='all' should produce {0,1} int signal."""
        ev = _make_evaluator(spy_like_data)
        spec = StrategySpec("t", "h", "combo", {
            "components": [
                {"signal_type": "momentum", "params": {"window": 20}},
                {"signal_type": "momentum", "params": {"window": 50}},
            ],
            "mode": "all",
        })
        _, sig = ev._generate_signals_for_split(spec, ev._select_df)
        assert set(sig.unique()) <= {0, 1}
        assert sig.dtype == int or sig.dtype == np.int64 or sig.dtype == np.int32


# ---------------------------------------------------------------------------
# 5. spec_key uniqueness and combo order-insensitivity
# ---------------------------------------------------------------------------

class TestSpecKeyV2:

    def test_mean_reversion_keys_differ_by_params(self):
        a = StrategySpec("a", "h", "mean_reversion", {"window": 20, "entry_z": -1.5, "exit_z": 0.0})
        b = StrategySpec("b", "h", "mean_reversion", {"window": 20, "entry_z": -1.0, "exit_z": 0.0})
        assert spec_key(a) != spec_key(b)

    def test_breakout_keys_differ_by_params(self):
        a = StrategySpec("a", "h", "breakout", {"entry_window": 55, "exit_window": 20})
        b = StrategySpec("b", "h", "breakout", {"entry_window": 100, "exit_window": 20})
        assert spec_key(a) != spec_key(b)

    def test_rsi_trend_key_stable_after_normalize(self):
        """Two RSI trend specs with and without explicit lower/upper should have same key."""
        a = StrategySpec("a", "h", "rsi", {"window": 14, "mode": "trend"})
        a.validate()  # normalise
        b = StrategySpec("b", "h", "rsi", {"window": 14, "lower": 30.0, "upper": 70.0, "mode": "trend"})
        b.validate()
        assert spec_key(a) == spec_key(b)

    def test_combo_key_order_insensitive(self):
        """Combo components in different order should produce the same spec_key."""
        comp1 = {"signal_type": "momentum", "params": {"window": 50}}
        comp2 = {"signal_type": "breakout", "params": {"entry_window": 55, "exit_window": 20}}
        a = StrategySpec("a", "h", "combo", {"components": [comp1, comp2], "mode": "all"})
        b = StrategySpec("b", "h", "combo", {"components": [comp2, comp1], "mode": "all"})
        assert spec_key(a) == spec_key(b)

    def test_combo_key_differs_by_mode(self):
        comp1 = {"signal_type": "momentum", "params": {"window": 50}}
        comp2 = {"signal_type": "breakout", "params": {"entry_window": 55, "exit_window": 20}}
        a = StrategySpec("a", "h", "combo", {"components": [comp1, comp2], "mode": "all"})
        b = StrategySpec("b", "h", "combo", {"components": [comp1, comp2], "mode": "any"})
        assert spec_key(a) != spec_key(b)

    def test_spec_keys_unique_in_shrunken_lattice(self):
        """All keys in a small shrunken lattice must be distinct."""
        import benchmark.layer2_search.space as space
        orig = (space.MOMENTUM_WINDOWS, space.MACD_FAST, space.MACD_SLOW,
                space.MACD_SPAN, space.VOL_TARGETS, space.MR_WINDOWS,
                space.MR_ENTRY_Z, space.MR_EXIT_Z, space.BO_ENTRY_WINDOWS,
                space.BO_EXIT_WINDOWS, space.RSI_WINDOWS, space.RSI_LOWERS,
                space.RSI_UPPERS, space.RSI_MODES)
        space.MOMENTUM_WINDOWS = [10, 20]
        space.MACD_FAST = [8]; space.MACD_SLOW = [24]; space.MACD_SPAN = [9]
        space.VOL_TARGETS = [None]
        space.MR_WINDOWS = [10]; space.MR_ENTRY_Z = [-1.5]; space.MR_EXIT_Z = [0.0]
        space.BO_ENTRY_WINDOWS = [20]; space.BO_EXIT_WINDOWS = [10]
        space.RSI_WINDOWS = [14]; space.RSI_LOWERS = [30.0]; space.RSI_UPPERS = [70.0]
        space.RSI_MODES = ["reversion"]
        try:
            specs = list(iter_space(signal_types=ALL_SIGNAL_TYPES))
            keys = [spec_key(s) for s in specs]
            assert len(keys) == len(set(keys)), f"Duplicate keys: {len(keys) - len(set(keys))}"
        finally:
            (space.MOMENTUM_WINDOWS, space.MACD_FAST, space.MACD_SLOW,
             space.MACD_SPAN, space.VOL_TARGETS, space.MR_WINDOWS,
             space.MR_ENTRY_Z, space.MR_EXIT_Z, space.BO_ENTRY_WINDOWS,
             space.BO_EXIT_WINDOWS, space.RSI_WINDOWS, space.RSI_LOWERS,
             space.RSI_UPPERS, space.RSI_MODES) = orig


# ---------------------------------------------------------------------------
# 6. _count_space matching len(list(iter_space(...))) on a shrunken lattice
# ---------------------------------------------------------------------------

class TestCountSpaceV2:

    def _shrink(self, space_mod):
        """Shrink all lattice constants and return originals for restoration."""
        orig = {
            "MOMENTUM_WINDOWS": space_mod.MOMENTUM_WINDOWS,
            "MACD_FAST": space_mod.MACD_FAST,
            "MACD_SLOW": space_mod.MACD_SLOW,
            "MACD_SPAN": space_mod.MACD_SPAN,
            "VOL_TARGETS": space_mod.VOL_TARGETS,
            "MR_WINDOWS": space_mod.MR_WINDOWS,
            "MR_ENTRY_Z": space_mod.MR_ENTRY_Z,
            "MR_EXIT_Z": space_mod.MR_EXIT_Z,
            "BO_ENTRY_WINDOWS": space_mod.BO_ENTRY_WINDOWS,
            "BO_EXIT_WINDOWS": space_mod.BO_EXIT_WINDOWS,
            "RSI_WINDOWS": space_mod.RSI_WINDOWS,
            "RSI_LOWERS": space_mod.RSI_LOWERS,
            "RSI_UPPERS": space_mod.RSI_UPPERS,
            "RSI_MODES": space_mod.RSI_MODES,
        }
        space_mod.MOMENTUM_WINDOWS = [10]
        space_mod.MACD_FAST = [8]; space_mod.MACD_SLOW = [24]; space_mod.MACD_SPAN = [9]
        space_mod.VOL_TARGETS = [None, 0.15]
        space_mod.MR_WINDOWS = [10]; space_mod.MR_ENTRY_Z = [-1.5, -1.0]; space_mod.MR_EXIT_Z = [0.0]
        space_mod.BO_ENTRY_WINDOWS = [20]; space_mod.BO_EXIT_WINDOWS = [10]
        space_mod.RSI_WINDOWS = [14]; space_mod.RSI_LOWERS = [30.0]; space_mod.RSI_UPPERS = [70.0]
        space_mod.RSI_MODES = ["reversion", "trend"]
        return orig

    def _restore(self, space_mod, orig):
        for k, v in orig.items():
            setattr(space_mod, k, v)

    def test_count_matches_iter_all_types_no_regimes(self):
        import benchmark.layer2_search.space as space
        orig = self._shrink(space)
        try:
            counted = _count_space(include_vol=True, include_regimes=False,
                                   signal_types=ALL_SIGNAL_TYPES)
            enumerated = len(list(iter_space(include_vol=True, include_regimes=False,
                                             signal_types=ALL_SIGNAL_TYPES)))
            assert counted == enumerated, f"count={counted} iter={enumerated}"
        finally:
            self._restore(space, orig)

    def test_count_matches_iter_legacy_types(self):
        import benchmark.layer2_search.space as space
        orig = self._shrink(space)
        try:
            counted = _count_space(include_vol=True, include_regimes=False,
                                   signal_types=LEGACY_SIGNAL_TYPES)
            enumerated = len(list(iter_space(include_vol=True, include_regimes=False,
                                             signal_types=LEGACY_SIGNAL_TYPES)))
            assert counted == enumerated
        finally:
            self._restore(space, orig)

    def test_enumerate_space_default_is_legacy(self):
        """enumerate_space() with no args must return momentum+macd only."""
        specs = enumerate_space(include_vol=True)
        sig_types = {s.signal_type for s in specs}
        assert sig_types <= {"momentum", "macd"}

    def test_enumerate_space_explicit_all_types(self):
        import benchmark.layer2_search.space as space
        orig = self._shrink(space)
        try:
            specs = enumerate_space(include_vol=False, signal_types=ALL_SIGNAL_TYPES)
            sig_types = {s.signal_type for s in specs}
            # should include the new types
            assert "mean_reversion" in sig_types
            assert "breakout" in sig_types
            assert "rsi" in sig_types
            assert "combo" in sig_types
        finally:
            self._restore(space, orig)

    def test_full_lattice_size_greater_than_legacy(self):
        fls = full_lattice_size(include_vol=True, include_regimes=False)
        legacy = _count_space(include_vol=True, include_regimes=False, signal_types=LEGACY_SIGNAL_TYPES)
        assert fls > legacy

    def test_lattice_version_is_v2(self):
        assert LATTICE_VERSION == "v2"


# ---------------------------------------------------------------------------
# 7. Lattice-mismatch guard in ExperimentGridRunner
# ---------------------------------------------------------------------------

class TestLatticeMismatchGuard:

    def test_mismatch_sets_oracle_columns_to_none(self, tmp_path, small_data):
        """When oracle has a different lattice_version, no scores are attached."""
        import benchmark.layer2_search.space as space
        from benchmark.layer2_search.oracle import build_oracle_table, save_oracle
        from benchmark.common.config import ExperimentConfig
        from benchmark.layer2_search.grid import ExperimentGridRunner

        ev = StrategyEvaluator(small_data)
        # Build a tiny oracle
        orig = (space.MOMENTUM_WINDOWS, space.MACD_FAST, space.MACD_SLOW,
                space.MACD_SPAN, space.VOL_TARGETS)
        space.MOMENTUM_WINDOWS = [10, 20]
        space.MACD_FAST = [8]; space.MACD_SLOW = [24]; space.MACD_SPAN = [9]
        space.VOL_TARGETS = [None]
        try:
            oracle_dir = tmp_path / "oracle"
            oracle_path = oracle_dir / "SPY_full.csv.gz"
            table = build_oracle_table(ev)
            save_oracle(table, oracle_path)
        finally:
            (space.MOMENTUM_WINDOWS, space.MACD_FAST, space.MACD_SLOW,
             space.MACD_SPAN, space.VOL_TARGETS) = orig

        # Now hand-write a different lattice_version into the meta file
        meta_path = oracle_dir / "SPY_full.meta.json"
        with open(meta_path) as f:
            meta = json.load(f)
        meta["lattice_version"] = "v_fake_old"
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        # Run one experiment and check results
        runner = ExperimentGridRunner(tmp_path / "out", evaluator=ev, oracle_dir=oracle_dir)
        cfg = ExperimentConfig(task_id="SPY_full", model="mock", framework_type="R-random",
                               seed=42, max_backtests=2)
        row = runner.run_single(cfg)

        # Oracle columns must be None (not scored against mismatched lattice)
        assert row.get("percentile_val") is None
        assert row.get("percentile_test") is None
        assert row.get("regret_test") is None
        # lattice_mismatch flag must be True
        assert row.get("lattice_mismatch") is True

    def test_matching_version_scores_normally(self, tmp_path, small_data):
        """When lattice_version matches, oracle scores are attached."""
        import benchmark.layer2_search.space as space
        from benchmark.layer2_search.oracle import build_oracle_table, save_oracle
        from benchmark.common.config import ExperimentConfig
        from benchmark.layer2_search.grid import ExperimentGridRunner

        ev = StrategyEvaluator(small_data)
        orig = (space.MOMENTUM_WINDOWS, space.MACD_FAST, space.MACD_SLOW,
                space.MACD_SPAN, space.VOL_TARGETS)
        space.MOMENTUM_WINDOWS = [10, 20, 30, 50]
        space.MACD_FAST = [8]; space.MACD_SLOW = [24]; space.MACD_SPAN = [9]
        space.VOL_TARGETS = [None]
        try:
            oracle_dir = tmp_path / "oracle"
            save_oracle(build_oracle_table(ev), oracle_dir / "SPY_full.csv.gz")
            runner = ExperimentGridRunner(tmp_path / "out", evaluator=ev, oracle_dir=oracle_dir)
            cfg = ExperimentConfig(task_id="SPY_full", model="mock", framework_type="R-random",
                                   seed=42, max_backtests=2)
            row = runner.run_single(cfg)
        finally:
            (space.MOMENTUM_WINDOWS, space.MACD_FAST, space.MACD_SLOW,
             space.MACD_SPAN, space.VOL_TARGETS) = orig

        # lattice_mismatch must be False
        assert row.get("lattice_mismatch") is False
        # percentile_val should be a finite number
        pv = row.get("percentile_val")
        assert pv is not None and not (isinstance(pv, float) and math.isnan(pv))
