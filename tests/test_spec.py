import pytest
from agent.spec import StrategySpec

def test_valid_momentum_spec():
    spec = StrategySpec(
        spec_id="s1",
        hypothesis="Trend following with 50-day SMA",
        signal_type="momentum",
        params={"window": 50}
    )
    is_valid, reason = spec.validate()
    assert is_valid is True
    assert reason == ""

def test_invalid_momentum_spec():
    spec = StrategySpec(
        spec_id="s2",
        hypothesis="Invalid SMA window",
        signal_type="momentum",
        params={"window": 0}
    )
    is_valid, reason = spec.validate()
    assert is_valid is False
    assert "Invalid momentum window" in reason

def test_valid_macd_spec():
    spec = StrategySpec(
        spec_id="s3",
        hypothesis="Standard MACD",
        signal_type="macd",
        params={"fast": 12, "slow": 26, "signal_span": 9}
    )
    is_valid, reason = spec.validate()
    assert is_valid is True
    assert reason == ""

def test_invalid_macd_spec():
    spec = StrategySpec(
        spec_id="s4",
        hypothesis="Inverted MACD",
        signal_type="macd",
        params={"fast": 30, "slow": 10, "signal_span": 9}
    )
    is_valid, reason = spec.validate()
    assert is_valid is False
    assert "Invalid MACD parameters" in reason

def test_serialization():
    spec = StrategySpec(
        spec_id="s5",
        hypothesis="Test serialization",
        signal_type="momentum",
        params={"window": 20}
    )
    d = spec.to_dict()
    restored = StrategySpec.from_dict(d)
    assert restored.spec_id == spec.spec_id
    assert restored.params == spec.params
    assert restored.is_fallback is False

def test_spec_fallback_flag():
    spec = StrategySpec(
        spec_id="test_fallback",
        hypothesis="Fallback test",
        signal_type="momentum",
        params={"window": 50},
        is_fallback=True
    )
    assert spec.is_fallback is True
    d = spec.to_dict()
    assert d["is_fallback"] is True
    recovered = StrategySpec.from_dict(d)
    assert recovered.is_fallback is True


# --- Layer 1: allowed_regimes validation ---

def test_validate_allowed_regimes_valid_subset():
    spec = StrategySpec("r1", "regime test", "momentum", {"window": 50},
                        allowed_regimes=["Q1", "Q3"])
    ok, msg = spec.validate()
    assert ok
    # validate normalises: dedupe + sort
    assert spec.allowed_regimes == ["Q1", "Q3"]


def test_validate_allowed_regimes_deduped_and_sorted():
    spec = StrategySpec("r2", "regime test", "momentum", {"window": 50},
                        allowed_regimes=["Q4", "Q1", "Q1"])
    ok, _ = spec.validate()
    assert ok
    assert spec.allowed_regimes == ["Q1", "Q4"]


def test_validate_allowed_regimes_empty_list_rejected():
    spec = StrategySpec("r3", "regime test", "momentum", {"window": 50},
                        allowed_regimes=[])
    ok, msg = spec.validate()
    assert not ok
    assert "non-empty" in msg.lower() or "allowed_regimes" in msg


def test_validate_allowed_regimes_unknown_rejected():
    spec = StrategySpec("r4", "regime test", "momentum", {"window": 50},
                        allowed_regimes=["Q1", "Q5"])
    ok, msg = spec.validate()
    assert not ok
    assert "Q5" in msg or "Unknown" in msg


def test_validate_all_four_regimes_is_valid():
    spec = StrategySpec("r5", "all regimes", "momentum", {"window": 50},
                        allowed_regimes=["Q1", "Q2", "Q3", "Q4"])
    ok, _ = spec.validate()
    assert ok


def test_validate_no_regimes_is_valid():
    spec = StrategySpec("r6", "no regimes", "momentum", {"window": 50},
                        allowed_regimes=None)
    ok, _ = spec.validate()
    assert ok


# --- Layer 1: vol-scaling validation ---

def test_validate_vol_target_out_of_range():
    spec = StrategySpec("v1", "extreme vol", "momentum", {"window": 50},
                        use_vol_scaling=True, vol_target=0.99, max_leverage=1.5)
    ok, msg = spec.validate()
    assert not ok
    assert "vol_target" in msg


def test_validate_max_leverage_out_of_range():
    spec = StrategySpec("v2", "high lev", "momentum", {"window": 50},
                        use_vol_scaling=True, vol_target=0.15, max_leverage=5.0)
    ok, msg = spec.validate()
    assert not ok
    assert "max_leverage" in msg


def test_validate_max_leverage_nonunit_without_vol_scaling():
    spec = StrategySpec("v3", "no vol", "momentum", {"window": 50},
                        use_vol_scaling=False, max_leverage=1.5)
    ok, msg = spec.validate()
    assert not ok
    assert "max_leverage" in msg


def test_validate_vol_scaling_valid_range():
    spec = StrategySpec("v4", "good vol", "momentum", {"window": 50},
                        use_vol_scaling=True, vol_target=0.12, max_leverage=1.5)
    ok, _ = spec.validate()
    assert ok


def test_from_dict_missing_fields_use_defaults():
    d = {"spec_id": "x", "signal_type": "momentum", "params": {"window": 20}}
    spec = StrategySpec.from_dict(d)
    assert spec.allowed_regimes is None
    assert spec.use_vol_scaling is False
    assert spec.max_leverage == 1.0

