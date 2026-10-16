from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest

from src.regimes import classify_market_regimes
from agent.layer1.v2_hard import (
    SUBTLE_AUDIT_CASES,
    HIGH_FREEDOM_IMPL_CASES,
    score_subtle_audit_case,
    evaluate_high_freedom_case,
    compute_active_action_match,
    compute_trade_f1,
    GoldenSubtleAuditClient,
    GoldenHighFreedomClient,
)


def _make_test_df(n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(20261015)
    dates = pd.bdate_range("2021-01-04", periods=n)
    steps = rng.normal(0.0004, 0.013, n)
    steps[80:160] += np.sin(np.linspace(0, 6 * np.pi, 80)) * 0.016
    steps[220:300] -= 0.002
    steps[250] = -0.042  # single-day shock < -3%
    steps[320:420] += 0.0025
    close = 100.0 * np.exp(np.cumsum(steps))
    open_ = np.r_[100.0, close[:-1]] * (1.0 + rng.normal(0.0, 0.002, n))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.005, n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.005, n)))
    volume = rng.integers(2_000_000, 30_000_000, n)

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )
    df["return"] = df["close"].pct_change()
    df["rolling_vol"] = df["return"].rolling(20).std() * np.sqrt(252)
    df = classify_market_regimes(df)
    return df


def test_subtle_audit_cases_structure_and_no_obvious_hints():
    assert len(SUBTLE_AUDIT_CASES) == 12
    ids = [c.case_id for c in SUBTLE_AUDIT_CASES]
    assert len(ids) == len(set(ids))

    flawed = [c for c in SUBTLE_AUDIT_CASES if c.has_flaw]
    clean = [c for c in SUBTLE_AUDIT_CASES if not c.has_flaw]
    assert len(flawed) == 10
    assert len(clean) == 2

    # Ensure none of the subtle audit cases contain obvious v1 giveaways like shift(-1) or center=True
    for c in SUBTLE_AUDIT_CASES:
        assert "shift(-1)" not in c.code
        assert "center=True" not in c.code
        assert "BUG" not in c.code
        assert "leak" not in c.code.lower()
        assert len(c.required_mechanism_keywords) >= 1 if c.has_flaw else True


def test_score_subtle_audit_requires_mechanism_hit():
    case = SUBTLE_AUDIT_CASES[0]  # sa_nan_lock_01 (where default NaN + Backtest NaN hold)

    # 1. Guessed category right, but completely missed the mechanism (e.g. generic comment)
    shallow_resp = json.dumps({
        "has_flaw": True,
        "flaw_types": list(case.flaw_types),
        "flawed_location": "build_strategy",
        "explanation": "The window size of 40 days might overfit the market regime.",
    })
    res_shallow = score_subtle_audit_case(case, shallow_resp)
    assert res_shallow.category_f1 == 1.0
    assert res_shallow.mechanism_hit is False
    assert res_shallow.strict_pass is False

    # 2. Identified both category and the exact cross-module NaN-hold mechanism
    deep_resp = json.dumps({
        "has_flaw": True,
        "flaw_types": list(case.flaw_types),
        "flawed_location": "base_sig.where(...) -> Backtest.run pd.isna(signal)",
        "explanation": (
            "Series.where() without a second argument fills False entries with NaN instead of 0. "
            "In Backtest.run, pd.isna(signal) sets trade_shares = 0 (holding current_position) "
            "rather than liquidating like signal == 0."
        ),
    })
    res_deep = score_subtle_audit_case(case, deep_resp)
    assert res_deep.category_f1 == 1.0
    assert res_deep.mechanism_hit is True
    assert res_deep.strict_pass is True


def test_active_action_match_and_trade_f1_metrics():
    # 90% flat series: a dummy all-zero prediction gets 0.90 raw agreement,
    # but 0.0 on active_action_match and 0.0 on trade_f1!
    ref = pd.Series([0.0] * 45 + [1.0] * 5 + [0.0] * 50)
    all_zero = pd.Series([0.0] * 100)
    assert compute_active_action_match(ref, all_zero) == 0.0
    assert compute_trade_f1(ref, all_zero, tol_bars=1) == 0.0

    # Shifted by 1 bar at entry due to min_periods difference: trade_f1 with tol_bars=1 still matches!
    sl_one = pd.Series([0.0] * 46 + [1.0] * 4 + [0.0] * 50)
    assert compute_active_action_match(ref, sl_one) == pytest.approx(0.8)
    assert compute_trade_f1(ref, sl_one, tol_bars=1) == pytest.approx(1.0)


def test_all_high_freedom_golden_implementations_pass():
    assert len(HIGH_FREEDOM_IMPL_CASES) == 10
    df = _make_test_df(500)
    client = GoldenHighFreedomClient()

    for case in HIGH_FREEDOM_IMPL_CASES:
        raw_code = client.generate(f"Case ID: {case.case_id}")
        res = evaluate_high_freedom_case(case, raw_code, df)
        assert res.exec_pass is True, f"{case.case_id} exec failed: {res.error_message}"
        assert res.no_lookahead is True, f"{case.case_id} lookahead: {res.lookahead_reason}"
        assert res.nan_hygiene is True, f"{case.case_id} failed nan_hygiene"
        assert res.bounds_ok is True, f"{case.case_id} failed bounds_ok"
        assert res.active_action_match == pytest.approx(1.0), f"{case.case_id} active_match={res.active_action_match}"
        assert res.trade_f1 == pytest.approx(1.0), f"{case.case_id} trade_f1={res.trade_f1}"
        assert res.passed is True
