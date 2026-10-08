import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd

import src.strategies as strategy
from src.backtest import Backtest
from src.data import load_market_data
from src.metrics import compute_strategy_metrics


def run(tickers: list[str] = ['SPY']) -> tuple[pd.DataFrame, pd.Series]:
    """Load data, build target weights from the strategy, backtest them, and print the metrics."""
    prices = load_market_data(tickers)
    weights = strategy.compute_macd_signal(prices['close'])

    bt = Backtest(
        initial_capital=100000.0,
        slippage_rate=0.0005,
        commission_rate=0.0005,
        rebalance_tolerance=0.05,
    )
    result = bt.run(prices['open'], prices['close'], weights)
    metrics = compute_strategy_metrics(result)

    print(f"\nTickers: {', '.join(tickers)}    Strategy: {strategy.compute_macd_signal.__name__}")
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
