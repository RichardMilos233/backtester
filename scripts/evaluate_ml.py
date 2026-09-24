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
    data_path: str = 'data/SPY.csv', 
    split_date: str = '2022-01-01',
    output_img: str = 'docs/ml_evaluation.png'
):
    print("=================================================================")
    print("     SESSION 9: MACHINE LEARNING INCREMENTAL EVALUATION (OOS)    ")
    print("=================================================================")

    # 1. 加载数据并生成状态特征
    print("1. 加载数据并生成状态特征...")
    df = load_market_data(data_path)
    dividend_events = pd.read_csv(
        'data/SPY_dividends.csv',
        parse_dates=['ex_date', 'pay_date'],
    )
    df_reg = classify_market_regimes(df)

    # 2. 构建特征矩阵 X 与目标 y
    print("2. 构建纯特征矩阵 X 与目标 y (预测明日收益率)...")
    X, y = prepare_ml_features(df_reg, target_type='continuous')

    # 3. 严格按时间切分：样本内训练集 (In-Sample) vs 样本外测试集 (Out-of-Sample)
    X_train, X_test, y_train, y_test = train_test_split_by_date(X, y, split_date=split_date)
    print(f"   样本内训练期 (Train): {X_train.index[0].strftime('%Y-%m-%d')} ~ {X_train.index[-1].strftime('%Y-%m-%d')} ({len(X_train)} 天)")
    print(f"   样本外测试期 (Test) : {X_test.index[0].strftime('%Y-%m-%d')} ~ {X_test.index[-1].strftime('%Y-%m-%d')} ({len(X_test)} 天)")

    # 4. 模型实例化
    models = {
        'Ridge': Ridge(alpha=10.0),
        'Lasso': Lasso(alpha=0.00005, random_state=42),
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42),
    }

    # 5. 逐个训练模型并生成样本外测试集信号
    print("\n3. 正在训练模型并生成样本外信号...")
    signals = {}
    for name, model in models.items():
        signals[name] = train_model_and_get_signals(model, X_train, y_train, X_test)

    # 6. 对照基准信号
    signals['Buy & Hold'] = pd.Series(1.0, index=X_test.index)
    sma50_full = compute_moving_average_signal(df_reg, window=50)
    signals['Rule SMA50'] = sma50_full.loc[X_test.index]

    # 7. 运行真实样本外摩擦回测
    print("4. 运行样本外回测 (注入真实交易摩擦)...")
    df_test = df_reg.loc[X_test.index]
    bt = Backtest(initial_capital=100000.0)

    backtest_results = {}
    metrics_results = {}
    for name, sig in signals.items():
        res = bt.run(df_test, sig, dividend_events)
        backtest_results[name] = res
        metrics_results[name] = compute_strategy_metrics(res)

    # 8. 输出样本外全景对比报表
    summary = pd.DataFrame(metrics_results)[['Buy & Hold', 'Rule SMA50', 'Ridge', 'Lasso', 'RandomForest']]
    print("\n" + "=" * 85)
    print("             样本外测试期 (2022-2026) 全策略横向大比拼")
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

    # 9. 模型可解释性诊断
    coef_df = pd.DataFrame({
        'Ridge_coef': models['Ridge'].coef_,
        'Lasso_coef': models['Lasso'].coef_,
        'RF_importance': models['RandomForest'].feature_importances_
    }, index=X.columns)

    # 10. 绘图：样本外净值走势与特征重要性
    print(f"\n5. 正在生成样本外策略对比图 -> {output_img}...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 子图 1: 样本外净值走势对比
    init_cap = 100000.0
    for name, color, ls, lw in [
        ('Buy & Hold', '#2b5c8f', '-', 1.8),
        ('Rule SMA50', '#d95f02', '--', 1.5),
        ('Ridge', '#7570b3', '-', 1.8),
        ('RandomForest', '#2ca02c', '-', 1.8),
    ]:
        res = backtest_results[name]
        ax1.plot(res.index, res['total_value'] / init_cap, label=name, color=color, linestyle=ls, lw=lw)
    ax1.set_title('样本外测试集 (2022-2026) 策略净值走势对比', fontsize=12, fontweight='bold')
    ax1.set_ylabel('账户净值倍数 (Multiple)', fontsize=11)
    ax1.set_xlabel('日期', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper left', frameon=True)

    # 子图 2: 随机森林特征重要性排行
    rf_imp = coef_df['RF_importance'].sort_values(ascending=True)
    ax2.barh(rf_imp.index, rf_imp.values, color='#31a354', alpha=0.85)
    ax2.set_title('Random Forest 特征重要性排序 (Feature Importance)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('相对重要性权重 (Importance)', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_img), exist_ok=True)
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    plt.close()
    print("图表已成功保存！\n")

    return summary, coef_df


if __name__ == '__main__':
    run_ml_evaluation()
