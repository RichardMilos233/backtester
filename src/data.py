from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / 'data' / 'ohlcv_daily'
COLUMNS = ['open', 'high', 'low', 'close', 'volume']


def load_market_data(ticker: str = 'SPY') -> pd.DataFrame:
    """Load adjusted OHLCV and add the current and next-period returns."""
    candidate_paths = [
        DATA_DIR / f'{ticker}.csv',
        Path(ticker),
        Path(__file__).resolve().parents[1] / 'data' / f'{ticker}.csv',
        Path(__file__).resolve().parents[1] / 'data' / 'SPY.csv',
    ]

    path = None
    for p in candidate_paths:
        if p.exists() and p.is_file():
            path = p
            break

    if path is None:
        raise FileNotFoundError(f'Market data file not found for ticker/path: {ticker}')

    df = pd.read_csv(path, index_col='date', parse_dates=True)

    # If raw unadjusted data with 'adj_close' is loaded, convert to adjusted OHLCV
    if 'adj_close' in df.columns:
        factor = df['adj_close'] / df['close']
        adj_df = pd.DataFrame(index=df.index)
        adj_df['open'] = df['open'] * factor
        adj_df['high'] = df['high'] * factor
        adj_df['low'] = df['low'] * factor
        adj_df['close'] = df['adj_close']
        adj_df['volume'] = df['volume']
        df = adj_df

        # Save to local ohlcv_daily for caching and future one-click use
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        cached_path = DATA_DIR / 'SPY.csv'
        if not cached_path.exists():
            df.to_csv(cached_path)
    else:
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
