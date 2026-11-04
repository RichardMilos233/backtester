from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from benchmark.layer2_search.oracle import OracleScorer
from benchmark.common.spec import StrategySpec
from benchmark.layer2_search.metrics import (
    TrajectoryEvalResult,
    score_trajectory,
    compute_hypothesis_alignment,
    aggregate_layer2_extensions,
)


def _make_mock_oracle() -> OracleScorer:
    rows = []
    # Create 100 synthetic oracle rows with val_sharpe uniformly spaced in [0.0, 0.99]
    for i in range(100):
        w = i + 2
        rows.append({
            "spec_key": f"mom_w{w}",
            "signal_type": "momentum",
            "params": f"{{'window': {w}}}",
            "use_vol_scaling": False,
            "vol_target": None,
            "allowed_regimes": None,
            "val_sharpe_ratio": i / 100.0,
            "test_sharpe_ratio": i / 100.0,
        })
    df = pd.DataFrame(rows).set_index("spec_key")
    return OracleScorer(df, metric="sharpe_ratio")


def test_score_trajectory_sample_efficiency_and_auc():
    scorer = _make_mock_oracle()

    # 4-step trajectory with val percentiles:
    # step 1: mom_w21 (val_sharpe=0.19 -> percentile=0.20) -> best_so_far=0.20
    # step 2: mom_w11 (val_sharpe=0.09 -> percentile=0.10) -> best_so_far=0.20
    # step 3: mom_w61 (val_sharpe=0.59 -> percentile=0.60) -> best_so_far=0.60
    # step 4: mom_w81 (val_sharpe=0.79 -> percentile=0.80) -> best_so_far=0.80
    # Expected AUC = mean([0.20, 0.20, 0.60, 0.80]) = 0.45
    steps = [
        (1, 21, 0.10, "Momentum trend following with 21-day window"),
        (2, 11, 0.05, "Faster momentum with 11-day window"),
        (3, 61, 0.50, "Medium momentum trend with 61-day window"),
        (4, 81, 0.90, "Longer momentum trend with 81-day window"),
    ]
    trajectory = []
    best = -float("inf")
    for step_idx, w, s_sharpe, hyp in steps:
        best = max(best, s_sharpe)
        spec = StrategySpec(f"s_{step_idx}", hyp, "momentum", {"window": w})
        trajectory.append({
            "step": step_idx,
            "spec_id": spec.spec_id,
            "spec_key": f"mom_w{w}",
            "spec": spec.to_dict(),
            "search_metrics": {"sharpe_ratio": s_sharpe},
            "best_so_far": best,
        })

    res = score_trajectory(trajectory, scorer)
    assert isinstance(res, TrajectoryEvalResult)
    assert len(res.step_percentile_val) == 4
    assert res.step_percentile_val == pytest.approx((0.20, 0.10, 0.60, 0.80))
    assert res.best_percentile_val_so_far == pytest.approx((0.20, 0.20, 0.60, 0.80))
    assert res.auc_percentile_val == pytest.approx(0.45)
    assert res.step1_percentile_val == pytest.approx(0.20)
    assert res.final_best_percentile_val == pytest.approx(0.80)

    # Step improvement on search_sharpe:
    # step 2: 0.05 < 0.10 (False)
    # step 3: 0.50 > 0.05 (True)
    # step 4: 0.90 > 0.50 (True)
    # -> 2 / 3
    assert res.step_improvement_rate == pytest.approx(2.0 / 3.0)

    # Search sharpe [0.10, 0.05, 0.50, 0.90] has exact same rank order as val percentile [0.20, 0.10, 0.60, 0.80]
    assert res.val_generalization_corr == pytest.approx(1.0)
    assert res.hypothesis_alignment == pytest.approx(1.0)


def test_score_trajectory_inverted_rank_correlation():
    scorer = _make_mock_oracle()
    # Search sharpe increases while Oracle Val percentile strictly decreases -> Spearman corr = -1.0
    steps = [
        (1, 81, 0.10, "Momentum window 81"),
        (2, 61, 0.20, "Momentum window 61"),
        (3, 41, 0.30, "Momentum window 41"),
        (4, 21, 0.40, "Momentum window 21"),
    ]
    trajectory = [
        {
            "step": idx,
            "spec_id": f"s_{idx}",
            "spec_key": f"mom_w{w}",
            "spec": StrategySpec(f"s_{idx}", hyp, "momentum", {"window": w}).to_dict(),
            "search_metrics": {"sharpe_ratio": ss},
            "best_so_far": ss,
        }
        for idx, w, ss, hyp in steps
    ]

    res = score_trajectory(trajectory, scorer)
    assert res.step_improvement_rate == pytest.approx(1.0)
    assert res.val_generalization_corr == pytest.approx(-1.0)
    # Since step 1 was already 0.80 and subsequent steps are lower, best_so_far stays at 0.80
    assert res.best_percentile_val_so_far == pytest.approx((0.80, 0.80, 0.80, 0.80))
    assert res.auc_percentile_val == pytest.approx(0.80)


def test_hypothesis_alignment_checks_signal_and_overlays():
    spec_aligned = StrategySpec(
        "s1",
        "Mean reversion using 20-day Z-score with volatility scaling in Q1/Q4 regimes",
        "mean_reversion",
        {"window": 20, "entry_z": -1.5, "exit_z": 0.0},
        allowed_regimes=["Q1", "Q4"],
        use_vol_scaling=True,
        vol_target=0.12,
        max_leverage=1.5,
    )
    assert compute_hypothesis_alignment(spec_aligned) == pytest.approx(1.0)

    spec_contradictory = StrategySpec(
        "s2",
        "Pure MACD trend following with no overlays",
        "rsi",
        {"window": 14, "lower": 30.0, "upper": 70.0, "mode": "reversion"},
        allowed_regimes=["Q3"],
        use_vol_scaling=True,
        vol_target=0.12,
        max_leverage=1.5,
    )
    assert compute_hypothesis_alignment(spec_contradictory) < 0.5


def test_benchmark_core_v2_extensions_artifacts():
    repo_root = Path(__file__).resolve().parents[2]
    csv_path = repo_root / "results/layer2_search/core_v2/trajectory_metrics.csv"
    md_path = repo_root / "results/layer2_search/core_v2/LAYER2_EXTENSIONS_REPORT.md"

    assert csv_path.exists()
    assert md_path.exists()

    df = pd.read_csv(csv_path)
    assert len(df) == 200
    for col in (
        "auc_percentile_val",
        "step_improvement_rate",
        "median_improvement_rate",
        "val_generalization_corr",
        "hypothesis_alignment",
    ):
        assert col in df.columns
        assert df[col].notna().all()

    lb = aggregate_layer2_extensions(df, group="framework_type")
    assert set(lb.index) == {"R-random", "R-grid", "C", "C-sota", "C-corr"}
    assert (lb["auc_percentile_val"] > 0.60).all()
    assert (lb["auc_percentile_val"] < 0.90).all()

