import sys
import os

# 确保能正确导入 src 模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso

from src.data import load_market_data
from src.regimes import classify_market_regimes
from src.backtest import Backtest
from src.metrics import compute_strategy_metrics
from src.signals import compute_moving_average_signal, compute_macd_signal, compute_volatility_scaled_signal
from src.ml import prepare_ml_features, train_test_split_by_date, train_model_and_get_signals


# ==============================================================================
# 📦 策略库预定义 (经典技术指标 + 机器学习 + 混合双引擎)
# ==============================================================================

# --- [策略 1] 经典双均线交叉 (Momentum / Trend) ---
def strategy_sma_cross(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.Series:
    """双均线动量策略：快线上穿慢线做多，下穿平仓"""
    sma_fast = df['close'].rolling(fast).mean()
    sma_slow = df['close'].rolling(slow).mean()
    return (sma_fast > sma_slow).astype(float)


# --- [策略 2] 经典 MACD 动能交叉 ---
def strategy_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal_span: int = 9) -> pd.Series:
    """MACD 策略：金叉 (DIF > DEA) 做多，死叉平仓"""
    return compute_macd_signal(df, fast=fast, slow=slow, signal_span=signal_span).astype(float)


# --- [策略 3] RSI 均值回归抄底策略 (Mean-Reversion) ---
def strategy_rsi(df: pd.DataFrame, window: int = 14, oversold: float = 35.0, overbought: float = 65.0) -> pd.Series:
    """RSI 策略：超卖 (<35) 逢低抄底做多，超买 (>65) 止盈离场"""
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    signals = pd.Series(np.nan, index=df.index)
    signals[rsi < oversold] = 1.0     # 触发超卖：买入
    signals[rsi > overbought] = 0.0   # 触发超买：平仓
    # 在买入和卖出之间保持前一个状态，初始默认空仓
    return signals.ffill().fillna(0.0)


# --- [策略 4] 布林带通道突破策略 (Bollinger Bands Breakout) ---
def strategy_bollinger(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.Series:
    """布林带策略：价格突破上轨放量做多，跌破中轨 (SMA20) 止损平仓"""
    mid_band = df['close'].rolling(window).mean()
    std = df['close'].rolling(window).std()
    upper_band = mid_band + num_std * std
    
    signals = pd.Series(np.nan, index=df.index)
    signals[df['close'] > upper_band] = 1.0  # 突破上轨开仓
    signals[df['close'] < mid_band] = 0.0    # 跌回均线止损
    return signals.ffill().fillna(0.0)


# --- [策略 5] 纯机器学习：Random Forest 随机森林预测 ---
def strategy_ml_random_forest(df: pd.DataFrame, split_date: str = '2022-01-01', n_estimators: int = 100, max_depth: int = 3) -> pd.Series:
    """随机森林：用 split_date 之前的数据训练多棵树，样本外预测明日收益率，预测 > 0 做多"""
    X, y = prepare_ml_features(df, target_type='continuous')
    X_train, X_test, y_train, y_test = train_test_split_by_date(X, y, split_date=split_date)
    
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf_sig = train_model_and_get_signals(rf, X_train, y_train, X_test)
    
    full_signals = pd.Series(0.0, index=df.index)
    full_signals.loc[rf_sig.index] = rf_sig
    return full_signals


# --- [策略 6] 纯机器学习：Ridge 岭回归预测 ---
def strategy_ml_ridge(df: pd.DataFrame, split_date: str = '2022-01-01', alpha: float = 10.0) -> pd.Series:
    """Ridge L2 正则化回归：抗共线性，样本外预测明日收益率，预测 > 0 做多"""
    X, y = prepare_ml_features(df, target_type='continuous')
    X_train, X_test, y_train, y_test = train_test_split_by_date(X, y, split_date=split_date)
    
    ridge = Ridge(alpha=alpha)
    ridge_sig = train_model_and_get_signals(ridge, X_train, y_train, X_test)
    
    full_signals = pd.Series(0.0, index=df.index)
    full_signals.loc[ridge_sig.index] = ridge_sig
    return full_signals


# --- [策略 7] 终极混合双引擎：年线宏观大趋势 + ML 进攻先锋 + 波动率风控 ---
def strategy_hybrid_trend_ml_vol(df: pd.DataFrame, split_date: str = '2022-01-01', target_vol: float = 0.12) -> pd.Series:
    """
    对冲基金级混合策略：
    1. 宏观防守门控：收盘价必须高于 200 日牛熊线 (大熊市绝不加仓)；
    2. 微观进场信号：Random Forest 预测明日正向收益；
    3. 动态仓位风控：根据 rolling_vol 目标波动率自适应平滑缩放。
    """
    # 门控：牛市门控
    bull_market_gate = df['close'] > df['close'].rolling(200).mean()
    
    # ML 信号
    ml_sig = strategy_ml_random_forest(df, split_date=split_date)
    
    # 双重交集
    combined = (bull_market_gate & (ml_sig > 0)).astype(float)
    
    # 波动率自适应缩放 (目标波动率反比)
    vol_scaled = compute_volatility_scaled_signal(df, combined, target_vol=target_vol)
    return vol_scaled


# ==============================================================================
# 🎯 策略切换开关 (取消注释想要测试的策略即可！)
# ==============================================================================
def my_strategy(df: pd.DataFrame) -> pd.Series:
    """
    直接注释 / 取消注释对应的一行 return 即可切换策略测试：
    """
    # [选项 1] 经典双均线动量 (SMA 20 > SMA 50)
    # return strategy_sma_cross(df, fast=20, slow=50)

    # [选项 2] 经典 MACD 动能策略
    # return strategy_macd(df)

    # [选项 3] RSI 均值回归抄底策略
    # return strategy_rsi(df, window=14, oversold=35, overbought=65)

    # [选项 4] 布林带通道突破策略
    # return strategy_bollinger(df, window=20, num_std=2.0)

    # [选项 5] 纯机器学习：Random Forest 随机森林
    return strategy_ml_random_forest(df, n_estimators=100, max_depth=4, split_date='2022-01-01')

    # [选项 6] 纯机器学习：Ridge 岭回归
    # return strategy_ml_ridge(df, split_date='2022-01-01')

    # [选项 7] 终极混合双引擎 (200日年线 + 随机森林 + 波动率风控)
    # return strategy_hybrid_trend_ml_vol(df, split_date='2022-01-01', target_vol=0.12)


# ==============================================================================
# 🚀 最小回测闭环运行器
# ==============================================================================
def main():
    # 1. 加载数据 (自动计算收益率与市场状态)
    print("正在加载数据与市场状态...")
    df = load_market_data('data/SPY.csv')
    df = classify_market_regimes(df)

    # 2. 生成当前选中的策略目标信号
    signals = my_strategy(df)

    # 3. 初始化回测引擎 (带万5滑点、万5手续费、5%再平衡容忍带抗频繁摩擦)
    bt = Backtest(
        initial_capital=100000.0,
        slippage_rate=0.0005,
        commission_rate=0.0005,
        rebalance_tolerance=0.05
    )

    # 4. 执行时序撮合与结算
    print("正在运行真实摩擦回测...")
    res = bt.run(df, signals)

    # 5. 打印策略体检报告
    metrics = compute_strategy_metrics(res)
    
    print("\n" + "=" * 45)
    print("           我的策略回测体检报告")
    print("=" * 45)
    print(metrics.round(4).to_string())
    print("=" * 45)
    print(f"期初本金:     ${bt.initial_capital:,.2f}")
    print(f"期末总资产:   ${res['total_value'].iloc[-1]:,.2f}")
    print(f"累计收益率:   {(res['total_value'].iloc[-1] / bt.initial_capital - 1) * 100:.2f}%")
    print(f"累计手续费:   ${res['commission'].sum():,.2f}")
    print("=" * 45 + "\n")


if __name__ == '__main__':
    main()
