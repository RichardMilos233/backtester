import pandas as pd
import numpy as np

def classify_market_regimes(
    df: pd.DataFrame, 
    vol_window: int = 20, 
    volume_window: int = 20
) -> pd.DataFrame:
    out = df.copy()
    T = 252
    rolling_vol = out['return'].rolling(window=vol_window).std() * np.sqrt(T)
    rolling_volume = out['volume'].rolling(window=volume_window).mean()

    vol_median = rolling_vol.rolling(window=T).median()
    volume_median = rolling_volume.rolling(window=T).median()

    valid_mask = vol_median.notna() & volume_median.notna()
    is_high_vol = rolling_vol > vol_median
    is_high_volume = rolling_volume > volume_median

    # Cartesian quadrants. X is volume, Y is volatility, origin is the rolling median.
    # Q1 (top-right): high volume, high volatility
    # Q2 (top-left): low volume, high volatility
    # Q3 (bottom-left): low volume, low volatility
    # Q4 (bottom-right): high volume, low volatility
    regime = pd.Series(index=out.index, dtype='object')
    regime[valid_mask & is_high_volume & is_high_vol] = 'Q1'
    regime[valid_mask & (~is_high_volume) & is_high_vol] = 'Q2'
    regime[valid_mask & (~is_high_volume) & (~is_high_vol)] = 'Q3'
    regime[valid_mask & is_high_volume & (~is_high_vol)] = 'Q4'

    out['rolling_vol'] = rolling_vol
    out['rolling_volume'] = rolling_volume
    out['regime'] = regime
    return out