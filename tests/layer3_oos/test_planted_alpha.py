from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmark.common.spec import StrategySpec
from benchmark.layer3_oos.planted_alpha import (
    PLANTED_ALPHA_SPECS,
    PlantedAlphaPickEval,
    generate_planted_alpha_suite,
    compute_planted_series_sharpe,
    evaluate_planted_alpha_pick,
    summarize_planted_alpha_picks,
)


@pytest.fixture(scope="module")
def planted_suite():
    return generate_planted_alpha_suite(n_bars=1500, seed=20261015)


def test_generate_planted_alpha_suite_structure(planted_suite):
    assert set(planted_suite.keys()) == {
        "SYNTH_MOM",
        "SYNTH_MREV",
        "SYNTH_REGIME_Q1",
        "SYNTH_NULL",
    }
    for task_id, df in planted_suite.items():
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1500
        for col in ("open", "high", "low", "close", "volume", "return", "rolling_vol", "regime"):
            assert col in df.columns, f"{task_id} missing column {col}"
        assert (df["close"] > 0).all()
        assert (df["high"] >= df["low"]).all()


def test_synth_mom_sharpe_separation(planted_suite):
    df = planted_suite["SYNTH_MOM"]

    for w in (20, 50, 100):
        mom_spec = StrategySpec(f"mom_{w}", "Momentum trend", "momentum", {"window": w})
        mom_sharpe = compute_planted_series_sharpe(df, mom_spec)
        assert mom_sharpe > 1.2, f"SYNTH_MOM momentum(window={w}) Sharpe={mom_sharpe:.3f} <= 1.2"

    mrev_spec = StrategySpec("mrev_20", "Mean reversion", "mean_reversion", {"window": 20, "entry_z": -1.5, "exit_z": 0.0})
    mrev_ret, mrev_sharpe = compute_planted_series_sharpe(df, mrev_spec, return_total=True)
    assert mrev_ret < 0.0 and mrev_sharpe < 0.0, (
        f"SYNTH_MOM mean_reversion should lose money, got total_return={mrev_ret:.3f}, sharpe={mrev_sharpe:.3f}"
    )


def test_synth_mrev_sharpe_separation(planted_suite):
    df = planted_suite["SYNTH_MREV"]

    mrev_spec = StrategySpec("mrev_20", "OU mean reversion", "mean_reversion", {"window": 20, "entry_z": -1.5, "exit_z": 0.0})
    mrev_sharpe = compute_planted_series_sharpe(df, mrev_spec)
    assert mrev_sharpe > 1.2, f"SYNTH_MREV mean_reversion Sharpe={mrev_sharpe:.3f} <= 1.2"

    rsi_spec = StrategySpec(
        "rsi_rev",
        "RSI reversion",
        "rsi",
        {"window": 14, "lower": 30.0, "upper": 70.0, "mode": "reversion"},
    )
    rsi_sharpe = compute_planted_series_sharpe(df, rsi_spec)
    assert rsi_sharpe > 1.2, f"SYNTH_MREV rsi(reversion) Sharpe={rsi_sharpe:.3f} <= 1.2"

    mom_spec = StrategySpec("mom_50", "Momentum 50", "momentum", {"window": 50})
    mom_ret, mom_sharpe = compute_planted_series_sharpe(df, mom_spec, return_total=True)
    assert mom_ret < 0.0 and mom_sharpe < 0.0, (
        f"SYNTH_MREV momentum should lose money, got total_return={mom_ret:.3f}, sharpe={mom_sharpe:.3f}"
    )


def test_synth_regime_q1_sharpe_separation(planted_suite):
    df = planted_suite["SYNTH_REGIME_Q1"]

    q1_spec = StrategySpec(
        "mom_q1",
        "Momentum filtered to Q1 high-vol/high-volume regime",
        "momentum",
        {"window": 20},
        allowed_regimes=["Q1"],
    )
    unfiltered_spec = StrategySpec(
        "mom_all",
        "Unfiltered momentum across all regimes",
        "momentum",
        {"window": 20},
    )
    wrong_regime_spec = StrategySpec(
        "mom_q3",
        "Momentum in low-vol/low-volume Q3",
        "momentum",
        {"window": 20},
        allowed_regimes=["Q3"],
    )

    q1_sharpe = compute_planted_series_sharpe(df, q1_spec)
    unfiltered_sharpe = compute_planted_series_sharpe(df, unfiltered_spec)
    q3_sharpe = compute_planted_series_sharpe(df, wrong_regime_spec)

    assert q1_sharpe > 1.2, f"SYNTH_REGIME_Q1 Q1-filtered Sharpe={q1_sharpe:.3f} <= 1.2"
    assert unfiltered_sharpe < 0.2, f"SYNTH_REGIME_Q1 unfiltered Sharpe={unfiltered_sharpe:.3f} should be much lower than Q1"
    assert q3_sharpe <= 0.0, f"SYNTH_REGIME_Q1 Q3-only Sharpe={q3_sharpe:.3f} should be <= 0"


def test_evaluate_planted_alpha_pick_and_null_false_positive(planted_suite):
    # 1) Correct planted hits
    hit_mom = evaluate_planted_alpha_pick(
        "SYNTH_MOM",
        StrategySpec("s1", "Trend momentum", "momentum", {"window": 60}),
        df=planted_suite["SYNTH_MOM"],
    )
    assert hit_mom.planted_hit is True
    assert hit_mom.false_positive is False
    assert hit_mom.true_sharpe > 1.2

    hit_mrev = evaluate_planted_alpha_pick(
        "SYNTH_MREV",
        StrategySpec("s2", "Z-score reversion", "mean_reversion", {"window": 20, "entry_z": -1.5, "exit_z": 0.0}),
        df=planted_suite["SYNTH_MREV"],
    )
    assert hit_mrev.planted_hit is True
    assert hit_mrev.true_sharpe > 1.2

    hit_q1 = evaluate_planted_alpha_pick(
        "SYNTH_REGIME_Q1",
        StrategySpec("s3", "Q1 breakout", "momentum", {"window": 20}, allowed_regimes=["Q1"]),
        df=planted_suite["SYNTH_REGIME_Q1"],
    )
    assert hit_q1.planted_hit is True
    assert hit_q1.true_sharpe > 1.2

    # 2) Misses on planted tasks
    miss_mom = evaluate_planted_alpha_pick(
        "SYNTH_MOM",
        StrategySpec("s4", "Wrong family on mom", "mean_reversion", {"window": 20, "entry_z": -1.5, "exit_z": 0.0}),
        df=planted_suite["SYNTH_MOM"],
    )
    assert miss_mom.planted_hit is False

    miss_q1 = evaluate_planted_alpha_pick(
        "SYNTH_REGIME_Q1",
        StrategySpec("s5", "Missing Q1 filter", "momentum", {"window": 20}, allowed_regimes=["Q3", "Q4"]),
        df=planted_suite["SYNTH_REGIME_Q1"],
    )
    assert miss_q1.planted_hit is False

    # 3) SYNTH_NULL false positive check
    null_fp = evaluate_planted_alpha_pick(
        "SYNTH_NULL",
        StrategySpec("s6", "Chasing noise on random walk", "momentum", {"window": 35}),
        df=planted_suite["SYNTH_NULL"],
    )
    assert null_fp.is_null_control is True
    assert null_fp.planted_hit is False
    assert null_fp.false_positive is True

    summary = summarize_planted_alpha_picks([hit_mom, hit_mrev, hit_q1, null_fp])
    assert summary["planted_tasks"] == 3
    assert summary["null_tasks"] == 1
    assert summary["planted_hit_rate"] == pytest.approx(1.0)
    assert summary["null_false_positive_rate"] == pytest.approx(1.0)
