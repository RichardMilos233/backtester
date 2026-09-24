import pandas as pd
import numpy as np

def compute_moving_average_signal(
    df: pd.DataFrame, 
    window: int = 50, 
    price_col: str = 'close'
) -> pd.Series:
    sma = df[price_col].rolling(window).mean()
    signal = (df[price_col] > sma).astype(int)
    return signal

def compute_macd_signal(
    df: pd.DataFrame, 
    fast=12, 
    slow=26, 
    signal_span=9,
    price_col = 'close'
) -> pd.Series:
    ema_fast = df[price_col].ewm(span=fast, adjust=False).mean()
    ema_slow = df[price_col].ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal_span, adjust=False).mean()
    signal = (dif > dea).astype(int)
    return signal

def compute_regime_filtered_signal(
    df: pd.DataFrame, 
    base_signal: pd.Series, 
    filter_regimes: list = ['Q2']
) -> pd.Series:
    # does not work because always stop loss when stock falls really hard
    filtered_signal = base_signal.copy()
    mask = df['regime'].isin(filter_regimes)
    filtered_signal[mask] = 0
    return filtered_signal

def compute_volatility_scaled_signal(
    df: pd.DataFrame, 
    base_signal: pd.Series, 
    target_vol: float = 0.12, 
    max_leverage: float = 1.0,
    vol_col: str = 'rolling_vol',
) -> pd.Series:
    if vol_col in df.columns:
        vol = df[vol_col]
    else:
        vol = df['return'].rolling(20).std() * np.sqrt(252)
    vol_safe = vol.replace(0, np.nan)

    scale = (target_vol / vol_safe).clip(lower=0.0, upper=max_leverage).fillna(1.0)
    return scale * base_signal