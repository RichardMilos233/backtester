import pandas as pd
import numpy as np
def load_market_data(filepath: str = './data/SPY.csv') -> pd.DataFrame:
    df = pd.read_csv(filepath, index_col='date', parse_dates=True)
    # return = (p_t - p_t-1) / p_t-1
    if 'return' not in df.columns:
        # df['return'] = df['adj_close'] / df['adj_close'].shift(1) - 1
        df['return'] = df['adj_close'].pct_change()
    # forward return = (p_t+1 - p_t) / p_t
    if 'forward_return' not in df.columns:
        df['forward_return'] = df['adj_close'].shift(-1) / df['adj_close'] - 1
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