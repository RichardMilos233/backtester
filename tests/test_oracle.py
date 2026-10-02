import numpy as np
import pandas as pd
import pytest

from agent.oracle import OracleScorer, build_oracle_table
from agent.space import enumerate_space, spec_key, MOMENTUM_WINDOWS
from agent.spec import StrategySpec


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
    from agent.evaluator import StrategyEvaluator
    import agent.space as space
    ev = StrategyEvaluator(synthetic_data_module)
    # shrink the lattice for speed
    orig = (space.MOMENTUM_WINDOWS, space.MACD_FAST, space.MACD_SLOW, space.MACD_SPAN, space.VOL_TARGETS)
    space.MOMENTUM_WINDOWS, space.MACD_FAST, space.MACD_SLOW, space.MACD_SPAN, space.VOL_TARGETS = \
        [10, 20], [8], [20], [9], [None, 0.15]
    try:
        table = build_oracle_table(ev)
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
