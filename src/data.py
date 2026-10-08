from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / 'data' / 'ohlcv_daily'
FIELDS = ['open', 'high', 'low', 'close', 'volume']


def load_market_data(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Load adjusted OHLCV for one or more tickers.

    Each value is a date-by-ticker table. A single ticker is a one-column table.
    """
    if isinstance(tickers, str):
        raise TypeError("tickers must be a list. One ticker is a one-item list.")

    frames = []
    for ticker in tickers:
        path = DATA_DIR / f'{ticker}.csv'
        if not path.exists():
            raise FileNotFoundError(f'Market data file not found: {path}')
        df = pd.read_csv(path, index_col='date', parse_dates=True)
        df = df[FIELDS].copy()
        df.columns = pd.MultiIndex.from_product([df.columns, [ticker]])
        frames.append(df)

    panel = pd.concat(frames, axis=1).dropna().sort_index()
    prices = {field: panel[field] for field in FIELDS}
    prices['return'] = prices['close'].pct_change()
    prices['forward_return'] = prices['close'].shift(-1) / prices['close'] - 1
    return prices


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
