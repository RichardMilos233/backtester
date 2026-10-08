import numpy as np
import pandas as pd


def compute_moving_average_signal(closes: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    sma = closes.rolling(window).mean()
    return (closes > sma).astype(int)


def compute_macd_signal(
    closes: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal_span: int = 9,
) -> pd.DataFrame:
    ema_fast = closes.ewm(span=fast, adjust=False).mean()
    ema_slow = closes.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal_span, adjust=False).mean()
    return (dif > dea).astype(int)


def compute_volatility_scaled_signal(
    rolling_vol: pd.DataFrame,
    base_weights: pd.DataFrame,
    target_vol: float = 0.12,
    max_leverage: float = 1.0,
) -> pd.DataFrame:
    vol = rolling_vol.replace(0, np.nan)
    scale = (target_vol / vol).clip(lower=0.0, upper=max_leverage).fillna(1.0)
    return scale * base_weights
