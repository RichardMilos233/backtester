import numpy as np
import pandas as pd


def classify_market_regimes(
    returns: pd.DataFrame,
    volume: pd.DataFrame,
    vol_window: int = 20,
    volume_window: int = 20,
) -> dict[str, pd.DataFrame]:
    """Label each date and ticker with a volume-volatility quadrant.

    Returns rolling volatility, rolling volume, and regime labels. Each table
    has the same dates and tickers as `returns`.
    """
    volume = volume.reindex(index=returns.index, columns=returns.columns)
    periods_per_year = 252
    rolling_vol = returns.rolling(window=vol_window).std() * np.sqrt(periods_per_year)
    rolling_volume = volume.rolling(window=volume_window).mean()
    vol_median = rolling_vol.rolling(window=periods_per_year).median()
    volume_median = rolling_volume.rolling(window=periods_per_year).median()

    valid = vol_median.notna() & volume_median.notna()
    high_vol = rolling_vol > vol_median
    high_volume = rolling_volume > volume_median

    regime = pd.DataFrame(pd.NA, index=returns.index, columns=returns.columns, dtype='object')
    regime = regime.mask(valid & high_volume & high_vol, 'Q1')
    regime = regime.mask(valid & ~high_volume & high_vol, 'Q2')
    regime = regime.mask(valid & ~high_volume & ~high_vol, 'Q3')
    regime = regime.mask(valid & high_volume & ~high_vol, 'Q4')

    return {
        'rolling_vol': rolling_vol,
        'rolling_volume': rolling_volume,
        'regime': regime,
    }
