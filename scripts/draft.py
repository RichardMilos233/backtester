import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.data import load_market_data
from src.regimes import classify_market_regimes
from src.signals import compute_moving_average_signal
from src.backtest import Backtest
from src.metrics import compute_conditional_metrics

# 1. 加载数据并划分象限
df = load_market_data('data/SPY.csv')
df_reg = classify_market_regimes(df)

# 2. 生成基线策略信号（例如 SMA 50）
sig = compute_moving_average_signal(df_reg, window=50)

# 3. 运行真实回测（产出带 regime 的账本 res）
bt = Backtest(initial_capital=100000.0)
res = bt.run(df_reg, sig)

# 4. 测试条件分析函数
cond_metrics = compute_conditional_metrics(res)
print("\n=== 四象限条件表现诊断报告 ===")
print(cond_metrics.round(4))