import numpy as np
import pandas as pd


def compute_moving_average_signal(df: pd.DataFrame, window: int = 50) -> pd.Series:
    sma = df['close'].rolling(window).mean()
    return (df['close'] > sma).astype(int)


def compute_macd_signal(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal_span: int = 9,
) -> pd.Series:
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal_span, adjust=False).mean()
    return (dif > dea).astype(int)


def compute_volatility_scaled_signal(
    df: pd.DataFrame,
    base_signal: pd.Series,
    target_vol: float = 0.12,
    max_leverage: float = 1.0,
) -> pd.Series:
    vol = df['rolling_vol'].replace(0, np.nan)
    scale = (target_vol / vol).clip(lower=0.0, upper=max_leverage).fillna(1.0)
    return scale * base_signal
