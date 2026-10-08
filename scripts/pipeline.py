import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd

import src.strategies as strategy
from src.backtest import Backtest
from src.data import load_market_data
from src.metrics import compute_strategy_metrics
from src.regimes import classify_market_regimes


def run(ticker: str = 'SPY') -> tuple[pd.DataFrame, pd.Series]:
    """Load data, build target weights from the strategy, backtest them, and print the metrics."""
    df = classify_market_regimes(load_market_data(ticker))
    weights = strategy.random_forest(df, n_estimators=100, max_depth=4, split_date='2022-01-01')

    bt = Backtest(
        initial_capital=100000.0,
        slippage_rate=0.0005,
        commission_rate=0.0005,
        rebalance_tolerance=0.05,
    )
    result = bt.run(df, weights)
    metrics = compute_strategy_metrics(result)

    print(f"\nTicker: {ticker}    Strategy: {strategy.random_forest.__name__}")
    print("=" * 45)
    print(metrics.round(4).to_string())
    print("=" * 45)
    print(f"Initial capital: ${bt.initial_capital:,.2f}")
    print(f"Final equity:    ${result['total_value'].iloc[-1]:,.2f}")
    print(f"Total return:    {(result['total_value'].iloc[-1] / bt.initial_capital - 1) * 100:.2f}%")
    print(f"Commission:      ${result['commission'].sum():,.2f}")
    print("=" * 45 + "\n")
    return result, metrics


if __name__ == '__main__':
    run()
