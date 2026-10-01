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

