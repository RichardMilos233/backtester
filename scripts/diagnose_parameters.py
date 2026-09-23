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


def run_parameter_sweep(data_path: str = 'data/SPY.csv', output_img: str = 'docs/parameter_sweep.png'):
    print(f"Loading data from {data_path}...")
    df = load_market_data(data_path)
    bt = Backtest(initial_capital=100000.0)
    
    results = {}
    
    # 0. Benchmark: Buy & Hold
    sig_bh = (df['close'] > 0).astype(int)
    results['Buy & Hold'] = compute_strategy_metrics(bt.run(df, sig_bh))
    
    # 1. SMA Parameter Sweep
    sma_windows = [10, 20, 50, 100, 200]
    for w in sma_windows:
        sig = compute_moving_average_signal(df, window=w)
        res = bt.run(df, sig)
        results[f'SMA {w}'] = compute_strategy_metrics(res)
        
    # 2. MACD Parameter Sweep
    macd_configs = [
        (8, 17, 9, 'MACD Fast (8,17,9)'),
        (12, 26, 9, 'MACD Standard (12,26,9)'),
        (16, 35, 9, 'MACD Medium (16,35,9)'),
        (24, 52, 18, 'MACD Slow (24,52,18)')
    ]
    for fast, slow, span, name in macd_configs:
        sig = compute_macd_signal(df, fast=fast, slow=slow, signal_span=span)
        res = bt.run(df, sig)
        results[name] = compute_strategy_metrics(res)
        
    summary_df = pd.DataFrame(results).T
    
    # Format columns for display
    print("\n" + "=" * 90)
    print("                     SESSION 4: PARAMETER SENSITIVITY SWEEP")
    print("=" * 90)
    
    display_df = summary_df.copy()
    display_df['total_return'] = (display_df['total_return'] * 100).map('{:.2f}%'.format)
    display_df['annualized_return'] = (display_df['annualized_return'] * 100).map('{:.2f}%'.format)
    display_df['annualized_volatility'] = (display_df['annualized_volatility'] * 100).map('{:.2f}%'.format)
    display_df['max_drawdown'] = (display_df['max_drawdown'] * 100).map('{:.2f}%'.format)
    display_df['sharpe_ratio'] = display_df['sharpe_ratio'].map('{:.3f}'.format)
    display_df['calmar_ratio'] = display_df['calmar_ratio'].map('{:.3f}'.format)
    display_df['trade_count'] = display_df['trade_count'].astype(int)
    display_df['turnover'] = display_df['turnover'].map('{:.1f}x'.format)
    display_df['daily_win_rate'] = (display_df['daily_win_rate'] * 100).map('{:.2f}%'.format)
    
    print(display_df.to_string())
    print("=" * 90 + "\n")
    
    # 3. Visualization: SMA Parameter Sensitivity (Sharpe, Drawdown, Trade Count vs Window)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    sma_keys = [f'SMA {w}' for w in sma_windows]
    sma_metrics = summary_df.loc[sma_keys]
    
    # Plot 1: Return & Risk vs SMA Window
    ax1.plot(sma_windows, sma_metrics['sharpe_ratio'], marker='o', color='#2b5c8f', lw=2, label='Sharpe Ratio')
    ax1.set_xlabel('SMA Window (Days)', fontsize=11)
    ax1.set_ylabel('Sharpe Ratio', color='#2b5c8f', fontsize=11)
    ax1.tick_params(axis='y', labelcolor='#2b5c8f')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(sma_windows, sma_metrics['max_drawdown'] * 100, marker='s', color='#d95f02', lw=2, linestyle='--', label='Max Drawdown (%)')
    ax1_twin.set_ylabel('Max Drawdown (%)', color='#d95f02', fontsize=11)
    ax1_twin.tick_params(axis='y', labelcolor='#d95f02')
    ax1.set_title('SMA Trend: Risk-Adjusted Return vs Window Size', fontsize=12, fontweight='bold')
    
    # Plot 2: Turnover & Trade Count vs SMA Window
    ax2.plot(sma_windows, sma_metrics['trade_count'], marker='^', color='#7570b3', lw=2, label='Trade Count')
    ax2.set_xlabel('SMA Window (Days)', fontsize=11)
    ax2.set_ylabel('Trade Count (Transactions)', color='#7570b3', fontsize=11)
    ax2.tick_params(axis='y', labelcolor='#7570b3')
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    ax2_twin = ax2.twinx()
    ax2_twin.plot(sma_windows, sma_metrics['turnover'], marker='d', color='#e7298a', lw=2, linestyle='--', label='Turnover (Multiple)')
    ax2_twin.set_ylabel('Total Turnover (x)', color='#e7298a', fontsize=11)
    ax2_twin.tick_params(axis='y', labelcolor='#e7298a')
    ax2.set_title('SMA Trend: Overtrading & Turnover vs Window Size', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_img), exist_ok=True)
    plt.savefig(output_img, dpi=300)
    plt.close()
    print(f"Sensitivity plot saved to {output_img}")
    
    return summary_df


if __name__ == '__main__':
    run_parameter_sweep()
