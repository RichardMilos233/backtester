import numpy as np
import pandas as pd
import pytest

from benchmark.layer2_search.oracle import OracleScorer, build_oracle_table
from benchmark.layer2_search.space import enumerate_space, spec_key, MOMENTUM_WINDOWS, sample_space, _count_space, REGIME_OPTIONS
from benchmark.common.spec import StrategySpec


def test_space_enumeration_is_unique_and_valid():
    specs = enumerate_space(include_vol=True)
    keys = [spec_key(s) for s in specs]
    assert len(keys) == len(set(keys))
    assert all(s.validate()[0] for s in specs)
    assert len(enumerate_space(include_vol=False)) == len(specs) // 5


def test_spec_key_ignores_ids_but_not_vol_overlay():
    a = StrategySpec("x", "h1", "momentum", {"window": 120})
    b = StrategySpec("y", "h2", "momentum", {"window": 120})
    c = StrategySpec("z", "h3", "momentum", {"window": 120}, use_vol_scaling=True, vol_target=0.12, max_leverage=1.5)
    assert spec_key(a) == spec_key(b) != spec_key(c)


def test_scorer_percentile_and_regret():
    df = pd.DataFrame(
        {"val_sharpe_ratio": [0.1, 0.5, 0.9, 0.3], "test_sharpe_ratio": [1.0, 0.2, 0.6, 0.8]},
        index=pd.Index(["a", "b", "c", "d"], name="spec_key"),
    )
    sc = OracleScorer(df)
    s = sc.score(0.5, 0.6)
    assert s["percentile_val"] == pytest.approx(0.75)
    assert s["regret_test"] == pytest.approx(0.4)
    # Omniscient Val-selector picks 'c' (val 0.9) whose test is 0.6 -> zero regret vs that ceiling
    assert s["regret_vs_val_oracle"] == pytest.approx(0.0)
    assert np.isnan(sc.percentile(float("nan"), "val"))


def test_build_oracle_table_small(synthetic_data_module):
    from benchmark.common.evaluator import StrategyEvaluator
    import benchmark.layer2_search.space as space
    ev = StrategyEvaluator(synthetic_data_module)
    # shrink the lattice for speed
    orig = (space.MOMENTUM_WINDOWS, space.MACD_FAST, space.MACD_SLOW, space.MACD_SPAN, space.VOL_TARGETS)
    space.MOMENTUM_WINDOWS, space.MACD_FAST, space.MACD_SLOW, space.MACD_SPAN, space.VOL_TARGETS = \
        [10, 20], [8], [20], [9], [None, 0.15]
    try:
        table = build_oracle_table(ev, signal_types=space.LEGACY_SIGNAL_TYPES)
    finally:
        space.MOMENTUM_WINDOWS, space.MACD_FAST, space.MACD_SLOW, space.MACD_SPAN, space.VOL_TARGETS = orig
    assert len(table) == 6
    assert {"val_sharpe_ratio", "test_sharpe_ratio"} <= set(table.columns)
    sc = OracleScorer(table)
    pick = StrategySpec("p", "h", "momentum", {"window": 10})
    out = sc.score_spec(pick)
    assert out["on_lattice"] and 0 < out["percentile_val"] <= 1


@pytest.fixture
def synthetic_data_module():
    n = 400
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    rng = np.random.default_rng(0)
    prices = 100 + np.cumsum(rng.standard_normal(n))
    return pd.DataFrame({"open": prices, "high": prices + 1, "low": prices - 1, "close": prices, "volume": 1e6}, index=dates)


# --- Layer 1: spec_key includes regimes ---

def test_spec_key_differs_by_regime():
    a = StrategySpec("a", "h", "momentum", {"window": 50})
    b = StrategySpec("b", "h", "momentum", {"window": 50}, allowed_regimes=["Q1", "Q2"])
    c = StrategySpec("c", "h", "momentum", {"window": 50}, allowed_regimes=["Q3", "Q4"])
    assert spec_key(a) != spec_key(b)
    assert spec_key(b) != spec_key(c)
    assert spec_key(a) != spec_key(c)


def test_spec_key_regime_is_sorted():
    a = StrategySpec("a", "h", "momentum", {"window": 50}, allowed_regimes=["Q2", "Q1"])
    b = StrategySpec("b", "h", "momentum", {"window": 50}, allowed_regimes=["Q1", "Q2"])
    # validate() normalises allowed_regimes to sorted list; validate first
    a.validate()
    b.validate()
    assert spec_key(a) == spec_key(b)


# --- Layer 1: sample_space ---

def test_sample_space_returns_n_distinct_specs():
    specs = sample_space(n=50, seed=42, include_vol=True, include_regimes=True)
    assert len(specs) == 50
    keys = [spec_key(s) for s in specs]
    assert len(set(keys)) == 50, "sample_space must return distinct specs"


def test_sample_space_deterministic_by_seed():
    specs1 = sample_space(n=30, seed=7, include_vol=True, include_regimes=True)
    specs2 = sample_space(n=30, seed=7, include_vol=True, include_regimes=True)
    assert [spec_key(s) for s in specs1] == [spec_key(s) for s in specs2]


def test_sample_space_different_seeds_differ():
    keys1 = {spec_key(s) for s in sample_space(n=20, seed=1)}
    keys2 = {spec_key(s) for s in sample_space(n=20, seed=2)}
    assert keys1 != keys2


def test_sample_space_all_specs_valid():
    specs = sample_space(n=100, seed=0, include_vol=True, include_regimes=True)
    for s in specs:
        ok, msg = s.validate()
        assert ok, f"Invalid spec from sample_space: {msg}"


def test_count_space_with_regimes():
    base = _count_space(include_vol=True, include_regimes=False)
    with_regimes = _count_space(include_vol=True, include_regimes=True)
    assert with_regimes == base * len(REGIME_OPTIONS)


def test_build_oracle_table_sampled(synthetic_data_module):
    """A sampled oracle table should have <= sample size rows and build quickly."""
    from benchmark.common.evaluator import StrategyEvaluator
    import benchmark.layer2_search.space as space

    ev = StrategyEvaluator(synthetic_data_module)
    orig = (space.MOMENTUM_WINDOWS, space.MACD_FAST, space.MACD_SLOW, space.MACD_SPAN, space.VOL_TARGETS)
    space.MOMENTUM_WINDOWS = [10, 20, 30]
    space.MACD_FAST = [8]
    space.MACD_SLOW = [20]
    space.MACD_SPAN = [9]
    space.VOL_TARGETS = [None, 0.15]
    try:
        table = build_oracle_table(ev, include_vol=True, include_regimes=False, sample=4, seed=99)
    finally:
        (space.MOMENTUM_WINDOWS, space.MACD_FAST,
         space.MACD_SLOW, space.MACD_SPAN, space.VOL_TARGETS) = orig

    assert len(table) <= 4
    assert table.attrs.get("sampled") is True
    assert {"val_sharpe_ratio", "test_sharpe_ratio"} <= set(table.columns)



def test_build_oracle_table_default_covers_all_signal_types(synthetic_data_module):
    """Regression: the first v2 oracle build silently sampled only momentum/macd."""
    from benchmark.common.evaluator import StrategyEvaluator
    import benchmark.layer2_search.space as space
    ev = StrategyEvaluator(synthetic_data_module)
    table = build_oracle_table(ev, include_regimes=True, sample=300, seed=7)
    assert set(table["signal_type"]) == set(space.ALL_SIGNAL_TYPES)
    assert table.attrs["n_total_lattice"] == space._count_space(True, True, space.ALL_SIGNAL_TYPES)
