import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from src.data import load_market_data
from src.signals import compute_moving_average_signal, compute_macd_signal
from src.backtest import Backtest
from src.metrics import compute_strategy_metrics


def run_agent_strategy_evaluation(
    data_path: str = "data/SPY.csv",
    output_img: str = "docs/agent_strategy_performance.png"
):
    print(f"Loading market data from {data_path}...")
    df = load_market_data(data_path)
    dividend_events = pd.read_csv('data/SPY_dividends.csv', parse_dates=['ex_date', 'pay_date']) if os.path.exists('data/SPY_dividends.csv') else None
    initial_capital = 100000.0
    bt = Backtest(initial_capital=initial_capital)

    # 1. Benchmarks & Baselines
    print("Running 1. Buy & Hold benchmark (SPY)...")
    sig_bh = (df['close'] > 0).astype(int)
    res_bh = bt.run(df, sig_bh, dividend_events)
    m_bh = compute_strategy_metrics(res_bh)

    print("Running 2. Baseline SMA 50...")
    sig_ma50 = compute_moving_average_signal(df, window=50)
    res_ma50 = bt.run(df, sig_ma50, dividend_events)
    m_ma50 = compute_strategy_metrics(res_ma50)

    print("Running 3. Baseline MACD (12, 26, 9)...")
    sig_macd = compute_macd_signal(df, fast=12, slow=26, signal_span=9)
    res_macd = bt.run(df, sig_macd, dividend_events)
    m_macd = compute_strategy_metrics(res_macd)

    # 2. Agent Proposals
    print("Running 4. Agent Winning Strategy: SMA 120 (DeepSeek-Chat)...")
    sig_agent_120 = compute_moving_average_signal(df, window=120)
    res_agent_120 = bt.run(df, sig_agent_120, dividend_events)
    m_agent_120 = compute_strategy_metrics(res_agent_120)

    print("Running 5. Agent Long-term Strategy: SMA 252 (Claude Sonnet 4.6)...")
    sig_agent_252 = compute_moving_average_signal(df, window=252)
    res_agent_252 = bt.run(df, sig_agent_252, dividend_events)
    m_agent_252 = compute_strategy_metrics(res_agent_252)

    # Summary Table
    summary = pd.DataFrame({
        "Buy & Hold": m_bh,
        "Baseline SMA 50": m_ma50,
        "Baseline MACD": m_macd,
        "Agent SMA 120 (DeepSeek)": m_agent_120,
        "Agent SMA 252 (Claude)": m_agent_252,
    })

    print("\n" + "=" * 90)
    print("             FULL SAMPLE PERFORMANCE (2015 - 2026) IN YOUR BACKTESTER")
    print("=" * 90)
    print(summary.round(4).to_string())
    print("=" * 90 + "\n")

    # Generate Equity Curve Comparison Plot
    print(f"Generating performance plot -> {output_img}...")
    os.makedirs(os.path.dirname(output_img), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 9), sharex=True, gridspec_kw={'height_ratios': [2.5, 1]}
    )

    strategies = [
        (res_bh, "Buy & Hold (SPY)", "#7f7f7f", 1.5, "--"),
        (res_ma50, "Baseline SMA 50", "#1f77b4", 1.4, "-"),
        (res_macd, "Baseline MACD (12,26,9)", "#2ca02c", 1.4, "-"),
        (res_agent_120, "Agent SMA 120 (DeepSeek)", "#d62728", 2.2, "-"),
        (res_agent_252, "Agent SMA 252 (Claude)", "#9467bd", 1.8, "-"),
    ]

    # Subplot 1: Equity Curves
    for res, label, color, lw, ls in strategies:
        ax1.plot(res.index, res['total_value'] / initial_capital, label=label, color=color, lw=lw, linestyle=ls)

    ax1.set_title("CTA Backtest: Agent Generated Strategies vs Baselines (SPY 2015-2026)", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Portfolio Value (Multiple of Initial $100k)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", frameon=True, fontsize=10)

    # Subplot 2: Drawdown Curves
    for res, label, color, lw, ls in strategies:
        peak = res['total_value'].cummax()
        dd = (res['total_value'] - peak) / peak
        ax2.plot(res.index, dd * 100, label=label, color=color, lw=lw, linestyle=ls)

    ax2.set_title("Drawdown Comparison (%)", fontsize=12)
    ax2.set_ylabel("Drawdown (%)", fontsize=11)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.axhline(0, color="black", lw=0.8, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_img, dpi=300)
    plt.close()
    print("Plot saved successfully!\n")
    return summary


if __name__ == "__main__":
    run_agent_strategy_evaluation()
