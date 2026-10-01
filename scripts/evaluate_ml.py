import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor

from src.data import load_market_data
from src.regimes import classify_market_regimes
from src.signals import compute_moving_average_signal
from src.backtest import Backtest
from src.metrics import compute_strategy_metrics
from src.ml import (
    prepare_ml_features, 
    train_test_split_by_date, 
    train_model_and_get_signals
)


def run_ml_evaluation(
    ticker: str = 'SPY',
    split_date: str = '2022-01-01',
    output_img: str = 'docs/ml_evaluation.png'
):
    print("=================================================================")
    print("     SESSION 9: MACHINE LEARNING INCREMENTAL EVALUATION (OOS)    ")
    print("=================================================================")

    print("1. Loading data and market regimes...")
    df = load_market_data(ticker)
    df_reg = classify_market_regimes(df)

    print("2. Building features and the next-day return target...")
    X, y = prepare_ml_features(df_reg)

    X_train, X_test, y_train, y_test = train_test_split_by_date(X, y, split_date=split_date)
    print(f"   Train: {X_train.index[0].strftime('%Y-%m-%d')} to {X_train.index[-1].strftime('%Y-%m-%d')} ({len(X_train)} days)")
    print(f"   Test:  {X_test.index[0].strftime('%Y-%m-%d')} to {X_test.index[-1].strftime('%Y-%m-%d')} ({len(X_test)} days)")

    # Models
    models = {
        'Ridge': Ridge(alpha=10.0),
        'Lasso': Lasso(alpha=0.00005, random_state=42),
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42),
    }

    print("\n3. Training models and building out-of-sample signals...")
    signals = {}
    for name, model in models.items():
        signals[name] = train_model_and_get_signals(model, X_train, y_train, X_test)

    # Benchmarks
    signals['Buy & Hold'] = pd.Series(1.0, index=X_test.index)
    sma50_full = compute_moving_average_signal(df_reg, window=50)
    signals['Rule SMA50'] = sma50_full.loc[X_test.index]

    print("4. Running the out-of-sample backtest with trading costs...")
    df_test = df_reg.loc[X_test.index]
    bt = Backtest(initial_capital=100000.0)

    backtest_results = {}
    metrics_results = {}
    for name, sig in signals.items():
        res = bt.run(df_test, sig)
        backtest_results[name] = res
        metrics_results[name] = compute_strategy_metrics(res)

    # Out-of-sample comparison
    summary = pd.DataFrame(metrics_results)[['Buy & Hold', 'Rule SMA50', 'Ridge', 'Lasso', 'RandomForest']]
    print("\n" + "=" * 85)
    print("             OUT-OF-SAMPLE COMPARISON (2022-2026)")
    print("=" * 85)
    
    display_df = summary.astype(object).copy()
    display_df.loc['total_return'] = (summary.loc['total_return'] * 100).map('{:.2f}%'.format)
    display_df.loc['annualized_return'] = (summary.loc['annualized_return'] * 100).map('{:.2f}%'.format)
    display_df.loc['annualized_volatility'] = (summary.loc['annualized_volatility'] * 100).map('{:.2f}%'.format)
    display_df.loc['max_drawdown'] = (summary.loc['max_drawdown'] * 100).map('{:.2f}%'.format)
    display_df.loc['sharpe_ratio'] = summary.loc['sharpe_ratio'].map('{:.3f}'.format)
    display_df.loc['calmar_ratio'] = summary.loc['calmar_ratio'].map('{:.3f}'.format)
    display_df.loc['trade_count'] = summary.loc['trade_count'].astype(int)
    display_df.loc['turnover'] = summary.loc['turnover'].map('{:.1f}x'.format)
    display_df.loc['daily_win_rate'] = (summary.loc['daily_win_rate'] * 100).map('{:.2f}%'.format)

    print(display_df.to_string())
    print("=" * 85)

    # Coefficients and feature importance
    coef_df = pd.DataFrame({
        'Ridge_coef': models['Ridge'].coef_,
        'Lasso_coef': models['Lasso'].coef_,
        'RF_importance': models['RandomForest'].feature_importances_
    }, index=X.columns)

    print(f"\n5. Saving the comparison plot -> {output_img}...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Equity curves
    init_cap = 100000.0
    for name, color, ls, lw in [
        ('Buy & Hold', '#2b5c8f', '-', 1.8),
        ('Rule SMA50', '#d95f02', '--', 1.5),
        ('Ridge', '#7570b3', '-', 1.8),
        ('RandomForest', '#2ca02c', '-', 1.8),
    ]:
        res = backtest_results[name]
        ax1.plot(res.index, res['total_value'] / init_cap, label=name, color=color, linestyle=ls, lw=lw)
    ax1.set_title('Out-of-sample equity curves (2022-2026)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Portfolio value (multiple of initial)', fontsize=11)
    ax1.set_xlabel('Date', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper left', frameon=True)

    # Random forest feature importance
    rf_imp = coef_df['RF_importance'].sort_values(ascending=True)
    ax2.barh(rf_imp.index, rf_imp.values, color='#31a354', alpha=0.85)
    ax2.set_title('Random forest feature importance', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Importance', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_img), exist_ok=True)
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    plt.close()
    print("Plot saved.\n")

    return summary, coef_df


if __name__ == '__main__':
    run_ml_evaluation()
