import sys
import os

# Make src importable when this file is run as a script.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso

from src.data import load_market_data
from src.regimes import classify_market_regimes
from src.backtest import Backtest
from src.metrics import compute_strategy_metrics
from src.signals import compute_macd_signal, compute_volatility_scaled_signal
from src.ml import prepare_ml_features, train_test_split_by_date, train_model_and_get_signals


# ==============================================================================
# Strategy library: technical rules, machine learning, and one hybrid
# ==============================================================================

# --- [1] SMA crossover ---
def strategy_sma_cross(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.Series:
    """Long when the fast SMA is above the slow SMA."""
    sma_fast = df['close'].rolling(fast).mean()
    sma_slow = df['close'].rolling(slow).mean()
    return (sma_fast > sma_slow).astype(float)


# --- [2] MACD crossover ---
def strategy_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal_span: int = 9) -> pd.Series:
    """Long when DIF is above DEA."""
    return compute_macd_signal(df, fast=fast, slow=slow, signal_span=signal_span).astype(float)


# --- [3] RSI mean reversion ---
def strategy_rsi(df: pd.DataFrame, window: int = 14, oversold: float = 35.0, overbought: float = 65.0) -> pd.Series:
    """Long on oversold RSI, flat on overbought RSI."""
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    signals = pd.Series(np.nan, index=df.index)
    signals[rsi < oversold] = 1.0
    signals[rsi > overbought] = 0.0
    # Hold the previous state between signals. Start flat.
    return signals.ffill().fillna(0.0)


# --- [4] Bollinger breakout ---
def strategy_bollinger(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.Series:
    """Long on a close above the upper band, flat on a close below the middle band."""
    mid_band = df['close'].rolling(window).mean()
    std = df['close'].rolling(window).std()
    upper_band = mid_band + num_std * std
    
    signals = pd.Series(np.nan, index=df.index)
    signals[df['close'] > upper_band] = 1.0
    signals[df['close'] < mid_band] = 0.0
    return signals.ffill().fillna(0.0)


# --- [5] Random forest ---
def strategy_ml_random_forest(df: pd.DataFrame, split_date: str = '2022-01-01', n_estimators: int = 100, max_depth: int = 3) -> pd.Series:
    """Train on data before split_date. Long when the predicted next return is positive."""
    X, y = prepare_ml_features(df)
    X_train, X_test, y_train, y_test = train_test_split_by_date(X, y, split_date=split_date)
    
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf_sig = train_model_and_get_signals(rf, X_train, y_train, X_test)
    
    full_signals = pd.Series(0.0, index=df.index)
    full_signals.loc[rf_sig.index] = rf_sig
    return full_signals


# --- [6] Ridge regression ---
def strategy_ml_ridge(df: pd.DataFrame, split_date: str = '2022-01-01', alpha: float = 10.0) -> pd.Series:
    """Train on data before split_date. Long when the predicted next return is positive."""
    X, y = prepare_ml_features(df)
    X_train, X_test, y_train, y_test = train_test_split_by_date(X, y, split_date=split_date)
    
    ridge = Ridge(alpha=alpha)
    ridge_sig = train_model_and_get_signals(ridge, X_train, y_train, X_test)
    
    full_signals = pd.Series(0.0, index=df.index)
    full_signals.loc[ridge_sig.index] = ridge_sig
    return full_signals


# --- [7] Hybrid: 200-day trend, random forest, volatility scaling ---
def strategy_hybrid_trend_ml_vol(df: pd.DataFrame, split_date: str = '2022-01-01', target_vol: float = 0.12) -> pd.Series:
    """
    Long only when price is above its 200-day average and the forest predicts a positive return.
    Scale that position by target volatility.
    """
    bull_market_gate = df['close'] > df['close'].rolling(200).mean()
    ml_sig = strategy_ml_random_forest(df, split_date=split_date)
    combined = (bull_market_gate & (ml_sig > 0)).astype(float)
    vol_scaled = compute_volatility_scaled_signal(df, combined, target_vol=target_vol)
    return vol_scaled


# ==============================================================================
# Switch strategies by commenting one return and uncommenting another.
# ==============================================================================
def my_strategy(df: pd.DataFrame) -> pd.Series:
    """Return the target-weight series for the strategy under test."""
    # [1] SMA 20 / SMA 50 crossover
    # return strategy_sma_cross(df, fast=20, slow=50)

    # [2] MACD
    # return strategy_macd(df)

    # [3] RSI mean reversion
    # return strategy_rsi(df, window=14, oversold=35, overbought=65)

    # [4] Bollinger breakout
    # return strategy_bollinger(df, window=20, num_std=2.0)

    # [5] Random forest
    return strategy_ml_random_forest(df, n_estimators=100, max_depth=4, split_date='2022-01-01')

    # [6] Ridge
    # return strategy_ml_ridge(df, split_date='2022-01-01')

    # [7] Hybrid: 200-day trend + random forest + volatility scaling
    # return strategy_hybrid_trend_ml_vol(df, split_date='2022-01-01', target_vol=0.12)


def main():
    print("Loading data and market regimes...")
    df = load_market_data('SPY')
    df = classify_market_regimes(df)

    signals = my_strategy(df)

    # 5 bps slippage, 5 bps commission, 5% rebalance band.
    bt = Backtest(
        initial_capital=100000.0,
        slippage_rate=0.0005,
        commission_rate=0.0005,
        rebalance_tolerance=0.05
    )

    print("Running the backtest...")
    res = bt.run(df, signals)

    metrics = compute_strategy_metrics(res)

    print("\n" + "=" * 45)
    print("              STRATEGY REPORT")
    print("=" * 45)
    print(metrics.round(4).to_string())
    print("=" * 45)
    print(f"Initial capital: ${bt.initial_capital:,.2f}")
    print(f"Final equity:    ${res['total_value'].iloc[-1]:,.2f}")
    print(f"Total return:    {(res['total_value'].iloc[-1] / bt.initial_capital - 1) * 100:.2f}%")
    print(f"Commission:      ${res['commission'].sum():,.2f}")
    print("=" * 45 + "\n")


if __name__ == '__main__':
    main()
