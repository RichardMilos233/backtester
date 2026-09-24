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

    # 4 象限严格对应笛卡尔数学坐标系：
    # 横轴 X: 成交量 (Volume), 纵轴 Y: 波动率 (Volatility)
    # 原点 (0, 0): 历史滚动中位数 (Median)
    # Q1 (右上，+, +): 高量、高波 (High Volume, High Volatility)
    # Q2 (左上，-, +): 低量、高波 (Low Volume, High Volatility)
    # Q3 (左下，-, -): 低量、低波 (Low Volume, Low Volatility)
    # Q4 (右下，+, -): 高量、低波 (High Volume, Low Volatility)
    regime = pd.Series(index=out.index, dtype='object')
    regime[valid_mask & is_high_volume & is_high_vol] = 'Q1'
    regime[valid_mask & (~is_high_volume) & is_high_vol] = 'Q2'
    regime[valid_mask & (~is_high_volume) & (~is_high_vol)] = 'Q3'
    regime[valid_mask & is_high_volume & (~is_high_vol)] = 'Q4'

    out['rolling_vol'] = rolling_vol
    out['rolling_volume'] = rolling_volume
    out['regime'] = regime
    return out