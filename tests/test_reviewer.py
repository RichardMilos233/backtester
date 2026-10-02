import pytest
from agent.spec import StrategySpec
from agent.reviewer import Reviewer, ReviewResult


def test_reviewer_blocks_invalid_spec():
    reviewer = Reviewer()
    invalid_spec = StrategySpec("r1", "bad window", "momentum", {"window": -5})
    result = reviewer.review(invalid_spec, history=[])
    assert result.passed is False
    assert "Invalid momentum window" in result.reason


def test_reviewer_blocks_duplicate_parameters():
    reviewer = Reviewer(allow_duplicates=False)
    spec1 = StrategySpec("r1", "first", "momentum", {"window": 50})
    spec2 = StrategySpec("r2", "second with same params", "momentum", {"window": 50})

    result = reviewer.review(spec2, history=[spec1])
    assert result.passed is False
    assert "Duplicate strategy parameters" in result.reason


def test_reviewer_passes_novel_valid_spec():
    reviewer = Reviewer()
    spec1 = StrategySpec("r1", "first", "momentum", {"window": 50})
    spec2 = StrategySpec("r2", "novel", "momentum", {"window": 60})

    result = reviewer.review(spec2, history=[spec1])
    assert result.passed is True
    assert result.reason == ""


def test_reviewer_blocks_short_hypothesis():
    reviewer = Reviewer()
    short_hyp = StrategySpec("r1", "ab", "momentum", {"window": 50})
    result = reviewer.review(short_hyp, history=[])
    assert result.passed is False
    assert "Insufficient hypothesis" in result.reason


def test_reviewer_allows_duplicates_when_flag_true():
    reviewer = Reviewer(allow_duplicates=True)
    spec1 = StrategySpec("r1", "first hypothesis", "momentum", {"window": 50})
    spec2 = StrategySpec("r2", "same params again", "momentum", {"window": 50})

    result = reviewer.review(spec2, history=[spec1])
    assert result.passed is True
    assert result.reason == ""


def test_reviewer_correlation_gate():
    import numpy as np
    import pandas as pd
    from agent.evaluator import StrategyEvaluator

    dates = pd.date_range("2020-01-01", periods=200, freq="B")
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(200))
    df = pd.DataFrame({
        "open": prices,
        "high": prices + 1,
        "low": prices - 1,
        "close": prices,
        "volume": 1000000
    }, index=dates)
    evaluator = StrategyEvaluator(df)
    reviewer = Reviewer(check_correlation=True, correlation_threshold=0.90, evaluator=evaluator)

    spec1 = StrategySpec("s1", "SMA 50 baseline", "momentum", {"window": 50})
    spec2 = StrategySpec("s2", "SMA 51 nearly identical", "momentum", {"window": 51})

    res = reviewer.review(spec2, history=[spec1])
    assert res.passed is False
    assert "High signal correlation" in res.reason



def test_reviewer_distinguishes_vol_scaled_overlay_from_core_spec():
    from agent.reviewer import Reviewer
    from agent.spec import StrategySpec
    core = StrategySpec("core", "hypothesis text", "momentum", {"window": 120})
    overlay = StrategySpec("ovl", "hypothesis text", "momentum", {"window": 120},
                           use_vol_scaling=True, vol_target=0.12, max_leverage=1.5)
    same_overlay = StrategySpec("ovl2", "hypothesis text", "momentum", {"window": 120},
                                use_vol_scaling=True, vol_target=0.12, max_leverage=1.5)
    r = Reviewer(allow_duplicates=False)
    assert r.review(overlay, [core]).passed
    assert not r.review(same_overlay, [core, overlay]).passed
