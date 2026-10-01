import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from src.data import load_market_data
from src.regimes import classify_market_regimes
from src.signals import compute_moving_average_signal, compute_volatility_scaled_signal
from src.backtest import Backtest
from src.metrics import compute_strategy_metrics
from src.ml import prepare_ml_features, train_test_split_by_date, train_model_and_get_signals


def run_final_evolution_comparison(
    ticker: str = 'SPY',
    output_img: str = 'docs/final_strategy_comparison.png'
):
    print("=================================================================")
    print("    SESSION 10: THE 4 GENERATIONS OF CTA STRATEGY EVOLUTION      ")
    print("=================================================================")

    print("1. Loading data and market regimes...")
    df = load_market_data(ticker)
    df_reg = classify_market_regimes(df)

    print("2. Training the model on data before 2022-01-01...")
    X, y = prepare_ml_features(df_reg)
    split_date = '2022-01-01'
    X_train, X_test, y_train, y_test = train_test_split_by_date(X, y, split_date=split_date)
    
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
    rf_test_signal = train_model_and_get_signals(rf_model, X_train, y_train, X_test)
    
    print("3. Building strategy signals on the 2022-2026 test window...")
    df_test = df_reg.loc[X_test.index]
    
    sig_bh = pd.Series(1.0, index=X_test.index)
    sig_sma50_raw = compute_moving_average_signal(df_reg, 50).loc[X_test.index]
    sig_vol_scaled = compute_volatility_scaled_signal(df_reg, compute_moving_average_signal(df_reg, 50), target_vol=0.12).loc[X_test.index]
    sig_rf = rf_test_signal

    print("4. Running the backtests...")
    initial_cap = 100000.0
    
    # Gen 1: no slippage, no commission.
    bt_naive = Backtest(initial_capital=initial_cap, slippage_rate=0.0, commission_rate=0.0, rebalance_tolerance=0.0)
    res_gen1 = bt_naive.run(df_test, sig_sma50_raw)
    
    # Gen 2: 5 bps slippage and 5 bps commission.
    bt_realistic = Backtest(initial_capital=initial_cap, slippage_rate=0.0005, commission_rate=0.0005, rebalance_tolerance=0.0)
    res_gen2 = bt_realistic.run(df_test, sig_sma50_raw)
    
    # Gen 3: same costs, plus a 5% rebalance band.
    bt_vscaled = Backtest(initial_capital=initial_cap, slippage_rate=0.0005, commission_rate=0.0005, rebalance_tolerance=0.05)
    res_gen3 = bt_vscaled.run(df_test, sig_vol_scaled)
    
    # Gen 4: random forest with the same costs and rebalance band.
    res_gen4 = bt_vscaled.run(df_test, sig_rf)
    
    # Buy and hold, with the same costs as Gen 2.
    res_bh = bt_realistic.run(df_test, sig_bh)

    strategies = {
        'Buy & Hold': res_bh,
        'Gen 1: SMA50, no costs': res_gen1,
        'Gen 2: SMA50, with costs': res_gen2,
        'Gen 3: volatility scaled': res_gen3,
        'Gen 4: random forest': res_gen4,
    }

    # Metrics
    metrics = {name: compute_strategy_metrics(res) for name, res in strategies.items()}
    summary_df = pd.DataFrame(metrics)

    print("\n" + "=" * 95)
    print("         SESSION 10: STRATEGY COMPARISON (out of sample, 2022-2026)")
    print("=" * 95)
    
    display_df = summary_df.astype(object).copy()
    display_df.loc['total_return'] = (summary_df.loc['total_return'] * 100).map('{:.2f}%'.format)
    display_df.loc['annualized_return'] = (summary_df.loc['annualized_return'] * 100).map('{:.2f}%'.format)
    display_df.loc['annualized_volatility'] = (summary_df.loc['annualized_volatility'] * 100).map('{:.2f}%'.format)
    display_df.loc['max_drawdown'] = (summary_df.loc['max_drawdown'] * 100).map('{:.2f}%'.format)
    display_df.loc['sharpe_ratio'] = summary_df.loc['sharpe_ratio'].map('{:.3f}'.format)
    display_df.loc['calmar_ratio'] = summary_df.loc['calmar_ratio'].map('{:.3f}'.format)
    display_df.loc['trade_count'] = summary_df.loc['trade_count'].astype(int)
    display_df.loc['turnover'] = summary_df.loc['turnover'].map('{:.1f}x'.format)
    display_df.loc['daily_win_rate'] = (summary_df.loc['daily_win_rate'] * 100).map('{:.2f}%'.format)

    print(display_df.to_string())
    print("=" * 95 + "\n")

    print(f"5. Saving the comparison plot -> {output_img}...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9), sharex=True, gridspec_kw={'height_ratios': [2.2, 1.0]})

    color_palette = {
        'Buy & Hold': ('#2b5c8f', '-', 2.0),
        'Gen 1: SMA50, no costs': ('#969696', ':', 1.5),
        'Gen 2: SMA50, with costs': ('#e6550d', '--', 1.6),
        'Gen 3: volatility scaled': ('#756bb1', '-', 1.8),
        'Gen 4: random forest': ('#2ca02c', '-', 2.0),
    }

    # Equity curves
    for name, res in strategies.items():
        col, ls, lw = color_palette[name]
        ax1.plot(res.index, res['total_value'] / initial_cap, label=name, color=col, linestyle=ls, lw=lw)

    ax1.set_title('Strategy comparison, out of sample 2022-2026', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Portfolio value (multiple of initial)', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper left', frameon=True, fontsize=10)

    # Drawdowns
    for name, res in strategies.items():
        col, ls, lw = color_palette[name]
        peak = res['total_value'].cummax()
        dd = (res['total_value'] - peak) / peak
        ax2.plot(res.index, dd * 100, label=name, color=col, linestyle=ls, lw=lw)

    ax2.set_title('Drawdown (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Drawdown (%)', fontsize=11)
    ax2.set_xlabel('Date', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.axhline(0, color='black', lw=0.8, linestyle='--')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_img), exist_ok=True)
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    plt.close()
    print("Plot saved.\n")

    return summary_df


if __name__ == '__main__':
    run_final_evolution_comparison()
