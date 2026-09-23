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
