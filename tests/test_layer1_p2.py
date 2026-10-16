from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.regimes import classify_market_regimes
from agent.layer1.schema import (
    VALID_AMBIGUITY_SLOTS,
    AmbiguityCase,
    AmbiguityEvalResult,
    FactorUsageEvalResult,
)
from agent.layer1.cases.systematic_cases import SYSTEMATIC_CASES
from agent.layer1.cases.ambiguity_cases import AMBIGUITY_CASES
from agent.layer1.systematic import GoldenSystematicLLMClient
from agent.layer1.factor_usage import (
    evaluate_factor_usage_case,
    evaluate_factor_usage_suite,
    summarize_factor_usage_results,
    score_runs_jsonl_factor_usage,
)
from agent.layer1.ambiguity import (
    build_ambiguity_prompt,
    score_ambiguity_prediction,
    evaluate_ambiguity_suite,
    summarize_ambiguity_results,
    GoldenAmbiguityLLMClient,
)


def _make_sample_df(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(20261015)
    dates = pd.bdate_range("2022-01-03", periods=n)
    steps = rng.normal(0.0003, 0.011, n)
    close = 100.0 * np.exp(np.cumsum(steps))
    open_ = np.r_[100.0, close[:-1]] * (1.0 + rng.normal(0.0, 0.002, n))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.005, n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.005, n)))
    volume = rng.integers(3_000_000, 20_000_000, n)
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )
    df["return"] = df["close"].pct_change()
    df["rolling_vol"] = df["return"].rolling(20).std() * np.sqrt(252)
    return classify_market_regimes(df)


# =============================================================================
# Part 1: Factor Library Usage Tests
# =============================================================================

def test_golden_systematic_client_achieves_perfect_factor_usage():
    df = _make_sample_df(300)
    client = GoldenSystematicLLMClient()
    results = evaluate_factor_usage_suite(client, df, SYSTEMATIC_CASES)
    assert len(results) == 18

    for r in results:
        assert r.factor_call_recall == pytest.approx(1.0), f"{r.case_id} recall={r.factor_call_recall}"
        assert r.no_reinvented_wheel is True, f"{r.case_id} reinvented wheel: {r.notes}"
        assert r.signature_valid is True, f"{r.case_id} signature invalid: {r.signature_errors}"

    summary = summarize_factor_usage_results(results)
    assert summary["total_cases"] == 18
    assert summary["factor_cases"] == 15
    assert summary["mean_factor_call_recall"] == pytest.approx(1.0)
    assert summary["no_reinvented_wheel_rate"] == pytest.approx(1.0)
    assert summary["signature_valid_rate"] == pytest.approx(1.0)


def test_factor_usage_detects_reinvented_wheel_and_partial_recall():
    df = _make_sample_df(200)
    case_mom = SYSTEMATIC_CASES[0]  # sys_mom_01: expected=("compute_moving_average_signal",)

    # Re-implementing SMA by hand instead of calling compute_moving_average_signal
    hand_rolled_code = """
def build_strategy(df):
    sma = df["close"].rolling(60).mean()
    sig = (df["close"] > sma).astype(int)
    return {"signals": sig, "slippage_rate": 0.0005, "commission_rate": 0.0005, "rebalance_tolerance": 0.05}
"""
    res_hand = evaluate_factor_usage_case(case_mom, hand_rolled_code, df)
    assert res_hand.factor_call_recall == pytest.approx(0.0)
    assert res_hand.no_reinvented_wheel is False
    assert res_hand.signature_valid is False

    # Partial recall on sys_combo_all_07: expected=("compute_moving_average_signal", "compute_rsi_signal", "combine_signals")
    case_combo = SYSTEMATIC_CASES[6]
    partial_combo_code = """
def build_strategy(df):
    s1 = compute_moving_average_signal(df, window=50)
    s2 = compute_rsi_signal(df, window=14, lower=30.0, upper=70.0, mode="reversion")
    sig = (s1 & s2).astype(int)
    return {"signals": sig, "slippage_rate": 0.0005, "commission_rate": 0.0005, "rebalance_tolerance": 0.05}
"""
    res_partial = evaluate_factor_usage_case(case_combo, partial_combo_code, df)
    assert res_partial.factor_call_recall == pytest.approx(2.0 / 3.0)
    assert res_partial.no_reinvented_wheel is False
    assert res_partial.signature_valid is True


def test_factor_usage_detects_invalid_signatures():
    df = _make_sample_df(200)
    case_mom = SYSTEMATIC_CASES[0]  # sys_mom_01
    case_macd = SYSTEMATIC_CASES[1]  # sys_macd_02

    # 1) Passing Series df['close'] instead of DataFrame df as first argument
    bad_first_arg_code = """
def build_strategy(df):
    sig = compute_moving_average_signal(df["close"], window=60)
    return {"signals": sig, "slippage_rate": 0.0005, "commission_rate": 0.0005, "rebalance_tolerance": 0.05}
"""
    res_arg = evaluate_factor_usage_case(case_mom, bad_first_arg_code, df)
    assert res_arg.factor_call_recall == pytest.approx(1.0)
    assert res_arg.signature_valid is False
    assert len(res_arg.signature_errors) > 0

    # 2) Passing hallucinated keyword argument `signal=9` instead of `signal_span=9`
    bad_kwarg_code = """
def build_strategy(df):
    sig = compute_macd_signal(df, fast=12, slow=26, signal=9)
    return {"signals": sig, "slippage_rate": 0.0005, "commission_rate": 0.0005, "rebalance_tolerance": 0.05}
"""
    res_kw = evaluate_factor_usage_case(case_macd, bad_kwarg_code, df)
    assert res_kw.factor_call_recall == pytest.approx(1.0)
    assert res_kw.signature_valid is False
    assert any("signal" in err for err in res_kw.signature_errors)


def test_score_existing_layer1_canonical_runs_jsonl():
    runs_path = Path(__file__).resolve().parents[1] / "docs/experiments/layer1_canonical/runs.jsonl"
    assert runs_path.exists()
    df = _make_sample_df(300)
    scored_df, lb_df = score_runs_jsonl_factor_usage(runs_path, df=df)

    assert not scored_df.empty
    assert "golden" in lb_df.index
    assert "deepseek-v3.2" in lb_df.index

    assert lb_df.loc["golden", "factor_call_recall"] == pytest.approx(1.0)
    assert lb_df.loc["golden", "no_reinvented_wheel"] == pytest.approx(1.0)
    assert lb_df.loc["golden", "signature_valid"] == pytest.approx(1.0)

    # On the 6 Track A factor cases in canonical, deepseek-v3.2 recalls 100% of factor names
    # and has valid signatures on 3/6 cases (mom_01, rsi_trd_06, regime_vol_11)
    assert lb_df.loc["deepseek-v3.2", "factor_call_recall"] == pytest.approx(1.0)
    assert lb_df.loc["deepseek-v3.2", "signature_valid"] == pytest.approx(3.0 / 6.0)


# =============================================================================
# Part 2: Ambiguity Detection Tests
# =============================================================================

def test_ambiguity_cases_count_and_structure():
    assert len(AMBIGUITY_CASES) == 12
    ids = [c.case_id for c in AMBIGUITY_CASES]
    assert len(ids) == len(set(ids)), "AmbiguityCase IDs must be unique"

    for c in AMBIGUITY_CASES:
        c.validate()
        assert len(c.prompt.strip()) > 15

    ambiguous = [c for c in AMBIGUITY_CASES if c.is_ambiguous]
    controls = [c for c in AMBIGUITY_CASES if not c.is_ambiguous]
    assert len(ambiguous) == 8
    assert len(controls) == 4

    # All 4 core ambiguity slot types must be covered
    covered_slots = {s for c in ambiguous for s in c.required_slots}
    for req_slot in ("lookback_window", "entry_exit_threshold", "volatility_definition", "execution_timing"):
        assert req_slot in covered_slots


def test_score_ambiguity_prediction_cases():
    amb_case = AmbiguityCase(
        case_id="test_amb_multi",
        title="Ambiguous momentum + vol strategy",
        prompt="做一个均线突破加波动率目标缩放策略。",
        is_ambiguous=True,
        required_slots=("lookback_window", "volatility_definition"),
        slot_keywords={
            "lookback_window": ("均线周期", "window", "多少天", "lookback"),
            "volatility_definition": ("目标波动率", "target_vol", "最大杠杆"),
        },
        reference_questions=("请问均线窗口周期是多少天？", "请问目标波动率 target_vol 和最大杠杆是多少？"),
    )

    # 1) Full hit via missing_slots
    raw_full = json.dumps({
        "action": "clarify",
        "questions": ["What is the SMA window and target volatility?"],
        "missing_slots": ["lookback_window", "volatility_definition"],
    })
    res_full = score_ambiguity_prediction(amb_case, raw_full)
    assert res_full.parse_ok is True
    assert res_full.clarify_correct == 1.0
    assert res_full.slot_recall == pytest.approx(1.0)
    assert res_full.false_clarify == 0.0

    # 2) Partial hit via natural-language question keywords only
    raw_partial = json.dumps({
        "action": "clarify",
        "questions": ["请问均线周期是多少天？"],
        "missing_slots": [],
    })
    res_part = score_ambiguity_prediction(amb_case, raw_partial)
    assert res_part.clarify_correct == 1.0
    assert res_part.slot_recall == pytest.approx(0.5)

    # 3) Premature implement on ambiguous prompt
    raw_miss = json.dumps({
        "action": "implement",
        "questions": [],
        "missing_slots": [],
    })
    res_miss = score_ambiguity_prediction(amb_case, raw_miss)
    assert res_miss.clarify_correct == 0.0
    assert res_miss.slot_recall == pytest.approx(0.0)

    # 4) Unambiguous control case: implement vs false clarify
    ctrl_case = [c for c in AMBIGUITY_CASES if not c.is_ambiguous][0]
    res_ctrl_ok = score_ambiguity_prediction(
        ctrl_case,
        json.dumps({"action": "implement", "questions": [], "missing_slots": []}),
    )
    assert res_ctrl_ok.clarify_correct == 1.0
    assert res_ctrl_ok.slot_recall is None
    assert res_ctrl_ok.false_clarify == 0.0

    res_ctrl_fp = score_ambiguity_prediction(
        ctrl_case,
        json.dumps({"action": "clarify", "questions": ["Are you sure?"], "missing_slots": ["lookback_window"]}),
    )
    assert res_ctrl_fp.clarify_correct == 0.0
    assert res_ctrl_fp.false_clarify == 1.0


def test_golden_ambiguity_client_achieves_perfect_score():
    client = GoldenAmbiguityLLMClient()
    results = evaluate_ambiguity_suite(client, AMBIGUITY_CASES)
    summary = summarize_ambiguity_results(results)

    assert summary["total_cases"] == 12
    assert summary["ambiguous_cases"] == 8
    assert summary["control_cases"] == 4
    assert summary["clarify_accuracy"] == pytest.approx(1.0)
    assert summary["slot_recall"] == pytest.approx(1.0)
    assert summary["false_clarify_rate"] == pytest.approx(0.0)
