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


def run_baseline_analysis(data_path: str = 'data/SPY.csv', output_img: str = 'docs/baseline_performance.png'):
    print(f"Loading market data from {data_path}...")
    df = load_market_data(data_path)
    
    initial_capital = 100000.0
    bt = Backtest(initial_capital=initial_capital)
    
    # 1. Buy & Hold Benchmark (Always Long)
    print("Running Buy & Hold benchmark...")
    sig_bh = (df['close'] > 0).astype(int)
    res_bh = bt.run(df, sig_bh)
    m_bh = compute_strategy_metrics(res_bh)
    
    # 2. SMA 50 Momentum Strategy
    print("Running SMA 50 strategy...")
    sig_ma = compute_moving_average_signal(df, window=50)
    res_ma = bt.run(df, sig_ma)
    m_ma = compute_strategy_metrics(res_ma)
    
    # 3. MACD (12, 26, 9) Strategy
    print("Running MACD (12, 26, 9) strategy...")
    sig_macd = compute_macd_signal(df, fast=12, slow=26, signal_span=9)
    res_macd = bt.run(df, sig_macd)
    m_macd = compute_strategy_metrics(res_macd)
    
    # 4. Print Summary Table
    summary = pd.DataFrame({
        'Buy & Hold': m_bh,
        'SMA 50': m_ma,
        'MACD (12,26,9)': m_macd
    })
    print("\n" + "=" * 55)
    print("              BASELINE PERFORMANCE SUMMARY")
    print("=" * 55)
    print(summary.round(4).to_string())
    print("=" * 55 + "\n")
    
    # 5. Plotting Normalized Equity Curves & Drawdown
    print(f"Generating performance plot -> {output_img}...")
    os.makedirs(os.path.dirname(output_img), exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2.5, 1]})
    
    # Subplot 1: Normalized Equity Curve
    ax1.plot(res_bh.index, res_bh['total_value'] / initial_capital, label='Buy & Hold (SPY)', color='#2b5c8f', lw=1.8)
    ax1.plot(res_ma.index, res_ma['total_value'] / initial_capital, label='SMA 50 Trend', color='#d95f02', lw=1.5)
    ax1.plot(res_macd.index, res_macd['total_value'] / initial_capital, label='MACD (12,26,9)', color='#2ca02c', lw=1.5)
    ax1.set_title('SPY CTA Baseline Strategies - Equity Curves (Normalized)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Portfolio Value (Multiple of Initial)', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper left', frameon=True)
    
    # Subplot 2: Drawdown Comparison
    for res, label, color in [
        (res_bh, 'Buy & Hold', '#2b5c8f'),
        (res_ma, 'SMA 50', '#d95f02'),
        (res_macd, 'MACD', '#2ca02c')
    ]:
        peak = res['total_value'].cummax()
        dd = (res['total_value'] - peak) / peak
        ax2.plot(res.index, dd * 100, label=label, color=color, lw=1.2)
        
    ax2.set_title('Drawdown Comparison (%)', fontsize=12)
    ax2.set_ylabel('Drawdown (%)', fontsize=11)
    ax2.set_xlabel('Date', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.axhline(0, color='black', lw=0.8, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_img, dpi=300)
    plt.close()
    print("Plot successfully saved!\n")
    return summary


if __name__ == '__main__':
    run_baseline_analysis()
