"""
Market data frames used for scoring Layer 1 submissions.

Both loaders return daily OHLCV with the derived columns every task prompt promises:
`return`, `rolling_vol` (20-day annualised) and `regime` (four-quadrant label).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.regimes import classify_market_regimes

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "ohlcv_daily"


def _add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df["return"] = df["close"].pct_change()
    df["rolling_vol"] = df["return"].rolling(20).std() * np.sqrt(252)
    return classify_market_regimes(df)


def load_market_df(ticker: str, n_bars: int) -> pd.DataFrame:
    """Last `n_bars` daily bars of `data/ohlcv_daily/<ticker>.csv`. Raises if the file is missing."""
    path = DATA_DIR / f"{ticker}.csv"
    df = pd.read_csv(path, parse_dates=["date"], index_col="date").sort_index()
    if len(df) > n_bars:
        df = df.iloc[-n_bars:].copy()
    return _add_derived_columns(df)


def load_layer1_market_df(n_bars: int = 600) -> pd.DataFrame:
    """SPY slice if available on disk, otherwise a deterministic multi-regime synthetic frame (tests / CI)."""
    spy_path = DATA_DIR / "SPY.csv"
    if spy_path.exists():
        df = pd.read_csv(spy_path, parse_dates=["date"], index_col="date").sort_index()
        if len(df) > n_bars:
            df = df.iloc[-n_bars:].copy()
    else:
        rng = np.random.default_rng(20261015)
        dates = pd.bdate_range("2021-01-04", periods=n_bars)
        steps = rng.normal(0.0004, 0.012, n_bars)
        steps[100:220] += np.sin(np.linspace(0, 8 * np.pi, 120)) * 0.015
        steps[300:420] *= 2.0
        close = 100.0 * np.exp(np.cumsum(steps))
        open_ = np.r_[100.0, close[:-1]] * (1.0 + rng.normal(0.0, 0.002, n_bars))
        high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.006, n_bars)))
        low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.006, n_bars)))
        volume = rng.integers(2_000_000, 25_000_000, n_bars)
        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=dates,
        )
    return _add_derived_columns(df)
