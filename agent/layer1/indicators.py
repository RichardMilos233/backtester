"""
Shared indicator helpers for golden implementations.

Every golden in the task bank uses these so that "RSI(14)", "ATR(14)", "EMA(n)" mean one
thing. The same conventions are spelled out in each task prompt.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def sma(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=n).mean()


def ema(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(span=n, adjust=False).mean()


def wilder_rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi = rsi.where(avg_loss > 0, 100.0)
    rsi.iloc[0] = np.nan
    return rsi


def wilder_atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    prev_close = df["close"].astype(float).shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    tr.iloc[0] = float(high.iloc[0] - low.iloc[0])
    return tr.ewm(alpha=1.0 / n, adjust=False).mean()


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mid = close.rolling(n, min_periods=n).mean()
    sd = close.rolling(n, min_periods=n).std()  # pandas default ddof=1
    return mid - k * sd, mid, mid + k * sd


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


def default_output(signals: np.ndarray, index: pd.Index) -> dict:
    return {
        "signals": pd.Series(signals, index=index, dtype=float),
        "slippage_rate": 0.0005,
        "commission_rate": 0.0005,
        "rebalance_tolerance": 0.05,
    }
