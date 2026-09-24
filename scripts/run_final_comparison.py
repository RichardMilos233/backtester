import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Configure Chinese font support for macOS
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'Heiti SC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from src.data import load_market_data
from src.regimes import classify_market_regimes
from src.signals import compute_moving_average_signal, compute_volatility_scaled_signal
from src.backtest import Backtest
from src.metrics import compute_strategy_metrics
from src.ml import prepare_ml_features, train_test_split_by_date, train_model_and_get_signals


def run_final_evolution_comparison(
    data_path: str = 'data/SPY.csv', 
    output_img: str = 'docs/final_strategy_comparison.png'
):
    print("=================================================================")
    print("    SESSION 10: THE 4 GENERATIONS OF CTA STRATEGY EVOLUTION      ")
    print("=================================================================")

    # 1. 加载数据与市场状态
    print("1. 加载数据与市场状态...")
    df = load_market_data(data_path)
    dividend_events = pd.read_csv(
        'data/SPY_dividends.csv',
        parse_dates=['ex_date', 'pay_date'],
    )
    df_reg = classify_market_regimes(df)

    # 2. 训练机器学习模型 (严格在 2022-01-01 之前训练)
    print("2. 准备机器学习模型与时序预测...")
    X, y = prepare_ml_features(df_reg, target_type='continuous')
    split_date = '2022-01-01'
    X_train, X_test, y_train, y_test = train_test_split_by_date(X, y, split_date=split_date)
    
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
    rf_test_signal = train_model_and_get_signals(rf_model, X_train, y_train, X_test)
    
    # 3. 准备 4 代核心策略与基准信号 (在样本外测试集 2022-2026 上统一对齐检验)
    print("3. 生成 4 代策略目标信号...")
    df_test = df_reg.loc[X_test.index]
    
    sig_bh = pd.Series(1.0, index=X_test.index)
    sig_sma50_raw = compute_moving_average_signal(df_reg, 50).loc[X_test.index]
    sig_vol_scaled = compute_volatility_scaled_signal(df_reg, compute_moving_average_signal(df_reg, 50), target_vol=0.12).loc[X_test.index]
    sig_rf = rf_test_signal

    # 4. 运行回测引擎
    print("4. 执行真实时序撮合与账本结算...")
    initial_cap = 100000.0
    
    # Gen 1: 朴素无摩擦基准 (0 手续费、0 滑点)
    bt_naive = Backtest(initial_capital=initial_cap, slippage_rate=0.0, commission_rate=0.0, rebalance_tolerance=0.0)
    res_gen1 = bt_naive.run(df_test, sig_sma50_raw, dividend_events)
    
    # Gen 2: 真实摩擦基准 (万 5 手续费 + 万 5 滑点)
    bt_realistic = Backtest(initial_capital=initial_cap, slippage_rate=0.0005, commission_rate=0.0005, rebalance_tolerance=0.0)
    res_gen2 = bt_realistic.run(df_test, sig_sma50_raw, dividend_events)
    
    # Gen 3: 波动率自适应进阶版 (真实摩擦 + 5% 容忍带抗磨损)
    bt_vscaled = Backtest(initial_capital=initial_cap, slippage_rate=0.0005, commission_rate=0.0005, rebalance_tolerance=0.05)
    res_gen3 = bt_vscaled.run(df_test, sig_vol_scaled, dividend_events)
    
    # Gen 4: 机器学习增强版 (Random Forest + 真实摩擦 + 容忍带)
    res_gen4 = bt_vscaled.run(df_test, sig_rf, dividend_events)
    
    # Benchmark: Buy & Hold (真实摩擦)
    res_bh = bt_realistic.run(df_test, sig_bh, dividend_events)

    strategies = {
        'Buy & Hold (标普大盘)': res_bh,
        'Gen 1: 朴素基线 (无摩擦)': res_gen1,
        'Gen 2: 现实基线 (真实摩擦)': res_gen2,
        'Gen 3: 波动率缩放进阶版': res_gen3,
        'Gen 4: 机器学习增强版': res_gen4,
    }

    # 5. 计算全套统计指标
    metrics = {name: compute_strategy_metrics(res) for name, res in strategies.items()}
    summary_df = pd.DataFrame(metrics)

    print("\n" + "=" * 95)
    print("         SESSION 10: 4 代 CTA 策略进化史全景对比表 (样本外测试集 2022-2026)")
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

    # 6. 生成高分辨率演进对比大图
    print(f"5. 正在生成演进对比全景图 -> {output_img}...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9), sharex=True, gridspec_kw={'height_ratios': [2.2, 1.0]})

    color_palette = {
        'Buy & Hold (标普大盘)': ('#2b5c8f', '-', 2.0),
        'Gen 1: 朴素基线 (无摩擦)': ('#969696', ':', 1.5),
        'Gen 2: 现实基线 (真实摩擦)': ('#e6550d', '--', 1.6),
        'Gen 3: 波动率缩放进阶版': ('#756bb1', '-', 1.8),
        'Gen 4: 机器学习增强版': ('#2ca02c', '-', 2.0),
    }

    # 上半图: 净值曲线
    for name, res in strategies.items():
        col, ls, lw = color_palette[name]
        ax1.plot(res.index, res['total_value'] / initial_cap, label=name, color=col, linestyle=ls, lw=lw)

    ax1.set_title('CTA 策略演进史：从朴素基线到机器学习增强版 (净值走势 2022-2026 样本外)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('净值倍数 (Multiple of Initial)', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper left', frameon=True, fontsize=10)

    # 下半图: 最大回撤曲线
    for name, res in strategies.items():
        col, ls, lw = color_palette[name]
        peak = res['total_value'].cummax()
        dd = (res['total_value'] - peak) / peak
        ax2.plot(res.index, dd * 100, label=name, color=col, linestyle=ls, lw=lw)

    ax2.set_title('动态回撤走势对比 (Drawdown %)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('回撤百分比 (%)', fontsize=11)
    ax2.set_xlabel('日期', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.axhline(0, color='black', lw=0.8, linestyle='--')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_img), exist_ok=True)
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    plt.close()
    print("全景对比图保存成功！\n")

    return summary_df


if __name__ == '__main__':
    run_final_evolution_comparison()
