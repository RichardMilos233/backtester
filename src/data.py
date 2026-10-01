from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / 'data' / 'ohlcv_daily'
COLUMNS = ['open', 'high', 'low', 'close', 'volume']


def load_market_data(ticker: str = 'SPY') -> pd.DataFrame:
    """Load adjusted OHLCV and add the current and next-period returns."""
    path = DATA_DIR / f'{ticker}.csv'
    if not path.exists():
        raise FileNotFoundError(f'Market data file not found: {path}')

    df = pd.read_csv(path, index_col='date', parse_dates=True)
    df = df[COLUMNS].copy()
    df['return'] = df['close'].pct_change()
    df['forward_return'] = df['close'].shift(-1) / df['close'] - 1
    return df


def compute_metrics(returns: pd.Series) -> pd.Series:
    T = 252
    metrics = {
        'mean_daily_return': returns.mean(),
        'annualized_return': returns.mean() * T,
        'annualized_volatility': returns.std() * np.sqrt(T),
        'skewness': returns.skew(),
        'excess_kurtosis': returns.kurt(),
        'max_daily_gain': returns.max(),
        'max_daily_loss': returns.min(),
    }
    return pd.Series(metrics)
