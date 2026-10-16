from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.regimes import classify_market_regimes
from agent.layer1.cases.systematic_cases import SYSTEMATIC_CASES
from agent.layer1.systematic import (
    evaluate_systematic_case,
    evaluate_systematic_suite,
    summarize_systematic_results,
    GoldenSystematicLLMClient,
)


def _make_benchmark_df(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(20261015)
    dates = pd.bdate_range("2021-01-04", periods=n)
    # Create realistic multi-regime price path (trend + oscillation + vol shifts)
    steps = rng.normal(0.0004, 0.012, n)
    steps[100:180] += np.sin(np.linspace(0, 6 * np.pi, 80)) * 0.015
    steps[220:300] *= 2.0
    close = 100.0 * np.exp(np.cumsum(steps))
    open_ = np.r_[100.0, close[:-1]] * (1.0 + rng.normal(0.0, 0.002, n))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.006, n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.006, n)))
    volume = rng.integers(2_000_000, 25_000_000, n)

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )
    df["return"] = df["close"].pct_change()
    df["rolling_vol"] = df["return"].rolling(20).std() * np.sqrt(252)
    df = classify_market_regimes(df)
    return df


def test_systematic_cases_count_and_categories():
    assert len(SYSTEMATIC_CASES) == 18
    ids = [c.case_id for c in SYSTEMATIC_CASES]
    assert len(ids) == len(set(ids)), "Case IDs must be unique"

    cats = [c.category for c in SYSTEMATIC_CASES]
    assert cats.count("primitive") == 6
    assert cats.count("overlay_combo") == 6
    assert cats.count("friction_stateful") == 6


def test_all_golden_references_are_causal_and_pass_self_check():
    df = _make_benchmark_df(400)
    client = GoldenSystematicLLMClient()
    results = evaluate_systematic_suite(client, df, SYSTEMATIC_CASES)
    for r in results:
        assert r.exec_pass is True, f"{r.case_id} failed exec: {r.error_message}"
        assert r.no_lookahead is True, f"{r.case_id} failed lookahead: {r.lookahead_reason}"
        assert r.signal_agreement == pytest.approx(1.0), f"{r.case_id} signal_agreement={r.signal_agreement}"
        assert r.position_agreement == pytest.approx(1.0), f"{r.case_id} position_agreement={r.position_agreement}"
        assert r.friction_match == 1.0, f"{r.case_id} friction_match={r.friction_match}"
        assert r.sharpe_abs_error == pytest.approx(0.0, abs=1e-9)
        assert r.passed is True

    summary = summarize_systematic_results(results)
    assert summary["total_cases"] == 18
    assert summary["pass_rate"] == 1.0
    assert summary["no_lookahead_rate"] == 1.0
    assert summary["mean_position_agreement"] == pytest.approx(1.0)
    assert summary["friction_match_rate"] == 1.0


def test_evaluate_systematic_case_rejects_lookahead_and_wrong_friction():
    df = _make_benchmark_df(200)
    case = SYSTEMATIC_CASES[0]  # sys_mom_01 (60-day SMA)

    # 1) Lookahead code fails `passed` even if it runs
    leaky_code = """
def build_strategy(df):
    sma = df["close"].rolling(60).mean()
    sig = (df["close"].shift(-1) > sma).astype(int)
    return {"signals": sig, "slippage_rate": 0.0005, "commission_rate": 0.0005, "rebalance_tolerance": 0.05}
"""
    res_leaky = evaluate_systematic_case(case, leaky_code, df)
    assert res_leaky.exec_pass is True
    assert res_leaky.no_lookahead is False
    assert res_leaky.passed is False

    # 2) Wrong friction parameter sets friction_match = 0.0 and passed = False
    wrong_friction_code = """
from src.signals import compute_moving_average_signal

def build_strategy(df):
    sig = compute_moving_average_signal(df, window=60)
    return {"signals": sig, "slippage_rate": 0.0099, "commission_rate": 0.0005, "rebalance_tolerance": 0.05}
"""
    res_fric = evaluate_systematic_case(case, wrong_friction_code, df)
    assert res_fric.exec_pass is True
    assert res_fric.no_lookahead is True
    assert res_fric.signal_agreement == pytest.approx(1.0)
    assert res_fric.friction_match == 0.0
    assert res_fric.passed is False
