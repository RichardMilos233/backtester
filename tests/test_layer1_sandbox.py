import numpy as np
import pandas as pd
import pytest

from agent.layer1.schema import (
    VALID_FLAW_TYPES,
    BadAlphaCase,
    BadAlphaEvalResult,
    SystematicCase,
    SystematicEvalResult,
)
from agent.layer1.sandbox import (
    extract_python_code,
    compile_build_strategy,
    check_no_lookahead,
)


def _make_sample_df(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2022-01-03", periods=n)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, n)))
    open_ = close * (1.0 + rng.normal(0.0, 0.002, n))
    high = np.maximum(open_, close) * 1.005
    low = np.minimum(open_, close) * 0.995
    vol = rng.integers(1_000_000, 5_000_000, n)
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=dates,
    )
    df["return"] = df["close"].pct_change()
    df["rolling_vol"] = df["return"].rolling(20).std() * np.sqrt(252)
    df["regime"] = "Q1"
    return df


def test_valid_flaw_types_taxonomy():
    assert VALID_FLAW_TYPES == {
        "lookahead_bias",
        "friction_erosion",
        "data_snooping",
        "logical_inconsistency",
    }


def test_extract_and_compile_causal_strategy():
    raw = """Here is the strategy implementation:
```python
from src.signals import compute_moving_average_signal

def build_strategy(df):
    sig = compute_moving_average_signal(df, window=10)
    return {
        "signals": sig,
        "slippage_rate": 0.0005,
        "commission_rate": 0.0005,
        "rebalance_tolerance": 0.05,
    }
```
"""
    code = extract_python_code(raw)
    assert "def build_strategy" in code
    assert "```" not in code

    fn = compile_build_strategy(code)
    df = _make_sample_df(120)
    out = fn(df.copy())
    assert "signals" in out
    assert len(out["signals"]) == len(df)
    ok, reason = check_no_lookahead(fn, df)
    assert ok is True, f"Expected causal strategy to pass, got: {reason}"


def test_check_no_lookahead_catches_full_sample_normalization():
    leaky_code = """
import pandas as pd

def build_strategy(df):
    # Full-sample mean and std leaks future prices into earlier bars
    z = (df["close"] - df["close"].mean()) / df["close"].std()
    sig = (z < -0.2).astype(float)
    return {"signals": sig}
"""
    fn = compile_build_strategy(leaky_code)
    df = _make_sample_df(120)
    ok, reason = check_no_lookahead(fn, df)
    assert ok is False
    assert "lookahead" in reason.lower() or "mismatch" in reason.lower()


def test_check_no_lookahead_catches_negative_shift():
    leaky_code = """
def build_strategy(df):
    sig = (df["close"].shift(-1) > df["close"]).astype(int)
    return {"signals": sig}
"""
    fn = compile_build_strategy(leaky_code)
    df = _make_sample_df(120)
    ok, _ = check_no_lookahead(fn, df)
    assert ok is False


def test_compile_build_strategy_rejects_missing_function():
    with pytest.raises(ValueError, match="build_strategy"):
        compile_build_strategy("x = 1 + 1")
