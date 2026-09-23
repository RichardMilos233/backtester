import pandas as pd

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
