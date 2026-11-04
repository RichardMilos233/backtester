from __future__ import annotations

from typing import List
from benchmark.layer1_code.schema import BadAlphaCase

BAD_ALPHA_CASES: List[BadAlphaCase] = [
    # =========================================================================
    # 1. Single-flaw: lookahead_bias (3 cases)
    # =========================================================================
    BadAlphaCase(
        case_id="ba_lookahead_01",
        title="Full-sample Z-score normalization in mean reversion",
        description=(
            "日频均值回归策略：计算收盘价的 Z-score，当 Z-score 低于 -1.5 时次日开盘做多，"
            "高于 0.0 时平仓。单边滑点 5bps，佣金 5bps。"
        ),
        code="""import pandas as pd

def run_strategy(df: pd.DataFrame) -> pd.Series:
    # Normalize close across the dataset before thresholding
    mean_price = df["close"].mean()
    std_price = df["close"].std()
    z = (df["close"] - mean_price) / std_price
    sig = (z < -1.5).astype(int)
    return sig.shift(1).fillna(0)
""",
        has_flaw=True,
        flaw_types=("lookahead_bias",),
        reference_notes="Uses full-sample df['close'].mean() and .std() across the entire time series instead of a causal rolling window.",
    ),
    BadAlphaCase(
        case_id="ba_lookahead_02",
        title="Same-day close signal executed at same-day open",
        description=(
            "动量突破策略：当今日收盘价突破 50 日均线时，以今日开盘价买入建仓；"
            "跌破 50 日均线时以今日开盘价卖出平仓。"
        ),
        code="""import pandas as pd

def backtest_same_day(df: pd.DataFrame) -> pd.Series:
    sma50 = df["close"].rolling(50).mean()
    signal = (df["close"] > sma50).astype(float)
    # Execute at today's open and earn open-to-close return on the same bar
    intraday_ret = (df["close"] - df["open"]) / df["open"]
    strat_ret = signal * intraday_ret
    return strat_ret
""",
        has_flaw=True,
        flaw_types=("lookahead_bias",),
        reference_notes="Signal depends on today's close (df['close'] > sma50) but executes at today's open, earning today's open-to-close return without shifting by 1 bar.",
    ),
    BadAlphaCase(
        case_id="ba_lookahead_03",
        title="Centered rolling window filter for volatility regime",
        description=(
            "波动率过滤趋势策略：用 20 日平滑波动率识别低波环境，在低波且收盘价站上 40 日均线时持仓。"
        ),
        code="""import numpy as np
import pandas as pd

def generate_signal(df: pd.DataFrame) -> pd.Series:
    ret = df["close"].pct_change()
    # Smooth volatility using centered window for cleaner regime boundaries
    smooth_vol = ret.rolling(20, center=True).std() * np.sqrt(252)
    sma40 = df["close"].rolling(40).mean()
    return ((df["close"] > sma40) & (smooth_vol < 0.18)).astype(int)
""",
        has_flaw=True,
        flaw_types=("lookahead_bias",),
        reference_notes="rolling(20, center=True) uses 10 future bars to compute today's volatility regime.",
    ),

    # =========================================================================
    # 2. Single-flaw: friction_erosion (3 cases)
    # =========================================================================
    BadAlphaCase(
        case_id="ba_friction_01",
        title="Ultra-short 2-day SMA flip under 15bps one-way cost",
        description=(
            "日频 ETF 趋势策略：当收盘价高于 2 日均线时满仓做多，低于 2 日均线时清仓。"
            "目标市场单边滑点 10bps + 单边佣金 5bps（合计单边 15bps，双边往返 30bps），"
            "预期捕获日间微弱自相关（日均毛 Alpha 约 2bps）。"
        ),
        code="""import pandas as pd
from src.backtest import Backtest

def run_2day_sma(df: pd.DataFrame):
    sma2 = df["close"].rolling(2).mean()
    signals = (df["close"] > sma2).astype(int)
    bt = Backtest(slippage_rate=0.0010, commission_rate=0.0005, rebalance_tolerance=0.0)
    return bt.run(df, signals)
""",
        has_flaw=True,
        flaw_types=("friction_erosion",),
        reference_notes="A 2-day SMA flips position every 2-3 days (~100 round-trips/year). At 30bps round-trip cost (~30%/yr drag), friction completely destroys a 2bps gross edge.",
    ),
    BadAlphaCase(
        case_id="ba_friction_02",
        title="Zero rebalance tolerance with noisy daily volatility scaling",
        description=(
            "5 日滚动波动率缩放策略：每日根据过去 5 日实现波动率动态调整杠杆权重，"
            "设置 rebalance_tolerance=0.0 要求每天精确调仓至目标权重，单边滑点 8bps，单边佣金 7bps。"
        ),
        code="""import numpy as np
import pandas as pd
from src.backtest import Backtest

def run_noisy_vol_target(df: pd.DataFrame):
    vol5 = df["close"].pct_change().rolling(5).std() * np.sqrt(252)
    weights = (0.15 / vol5).clip(0.0, 2.0).fillna(1.0)
    bt = Backtest(slippage_rate=0.0008, commission_rate=0.0007, rebalance_tolerance=0.0)
    return bt.run(df, weights)
""",
        has_flaw=True,
        flaw_types=("friction_erosion",),
        reference_notes="5-day realised vol fluctuates every single day and rebalance_tolerance=0.0 forces daily trading under 15bps one-way friction.",
    ),
    BadAlphaCase(
        case_id="ba_friction_03",
        title="Daily 1-day return sign reversal in high-cost illiquid ETF",
        description=(
            "日度涨跌反转策略：只要昨日收阴线今天就满仓买入，昨日收阳线今天就清仓卖出。"
            "在单边滑点 12bps、单边佣金 8bps（双边 40bps）的品种上运行。"
        ),
        code="""import pandas as pd
from src.backtest import Backtest

def run_daily_flip(df: pd.DataFrame):
    ret1 = df["close"].pct_change()
    signals = (ret1 < 0).astype(int)
    bt = Backtest(slippage_rate=0.0012, commission_rate=0.0008, rebalance_tolerance=0.01)
    return bt.run(df, signals)
""",
        has_flaw=True,
        flaw_types=("friction_erosion",),
        reference_notes="Flips between 0% and 100% roughly every other day (~125 round-trips/yr) paying 40bps per round-trip (~50% annual friction loss).",
    ),

    # =========================================================================
    # 3. Single-flaw: data_snooping (3 cases)
    # =========================================================================
    BadAlphaCase(
        case_id="ba_snooping_01",
        title="Five conjunctive magic thresholds tuned to historical quirks",
        description=(
            "复合增强策略：当 17 日均线高于 43 日均线、且 11 日 RSI 介于 53.7 到 68.2 之间、"
            "且星期几不是周二、且成交量大于过去 9 日均量的 1.137 倍时才做多。"
        ),
        code="""import pandas as pd
from src.signals import compute_rsi_signal

def build_magic_strategy(df: pd.DataFrame) -> pd.Series:
    sma17 = df["close"].rolling(17).mean()
    sma43 = df["close"].rolling(43).mean()
    delta = df["close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/11, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/11, adjust=False).mean()
    rsi11 = 100 - 100 / (1 + gain / loss)
    vol_ratio = df["volume"] / df["volume"].rolling(9).mean()
    not_tuesday = df.index.dayofweek != 1
    sig = (sma17 > sma43) & (rsi11 > 53.7) & (rsi11 < 68.2) & not_tuesday & (vol_ratio > 1.137)
    return sig.astype(int)
""",
        has_flaw=True,
        flaw_types=("data_snooping",),
        reference_notes="Stacks 5 arbitrary over-specific parameters (17/43, RSI 53.7..68.2, not Tuesday, vol_ratio > 1.137) with no economic rationale.",
    ),
    BadAlphaCase(
        case_id="ba_snooping_02",
        title="Hardcoded calendar date exclusions for historical market crashes",
        description=(
            "稳健趋势跟踪策略：使用 50 日均线做多，但自带宏观风控模块，在 2020 年 2 月 19 日至 3 月 23 日、"
            "以及 2022 年 1 月至 10 月期间强制空仓以规避回撤。"
        ),
        code="""import pandas as pd

def build_crash_free_momentum(df: pd.DataFrame) -> pd.Series:
    sma50 = df["close"].rolling(50).mean()
    sig = (df["close"] > sma50).astype(int)
    idx = df.index
    covid_crash = (idx >= "2020-02-19") & (idx <= "2020-03-23")
    bear_2022 = (idx >= "2022-01-03") & (idx <= "2022-10-15")
    sig = sig.where(~(covid_crash | bear_2022), 0)
    return sig
""",
        has_flaw=True,
        flaw_types=("data_snooping",),
        reference_notes="Hardcodes exact historical crash dates (COVID crash and 2022 bear market) into the strategy rule, which is pure hindsight data snooping.",
    ),
    BadAlphaCase(
        case_id="ba_snooping_03",
        title="Selecting best hyperparameter directly on hold-out test split",
        description=(
            "自适应动量参数优选流程：在搜索集、验证集和测试集划分完成后，遍历 2 到 260 的所有窗口，"
            "挑选在 test_df 上夏普比率最高的窗口作为最终上线参数。"
        ),
        code="""import pandas as pd
from src.signals import compute_moving_average_signal
from src.backtest import Backtest
from src.metrics import compute_strategy_metrics

def select_best_window(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> int:
    best_w, best_sharpe = 50, -999.0
    bt = Backtest()
    for w in range(2, 261):
        sig = compute_moving_average_signal(test_df, window=w)
        metrics = compute_strategy_metrics(bt.run(test_df, sig))
        if metrics["sharpe_ratio"] > best_sharpe:
            best_sharpe = metrics["sharpe_ratio"]
            best_w = w
    return best_w
""",
        has_flaw=True,
        flaw_types=("data_snooping",),
        reference_notes="Optimizes window directly over `test_df` (hold-out test set) across 259 candidates.",
    ),

    # =========================================================================
    # 4. Single-flaw: logical_inconsistency (3 cases)
    # =========================================================================
    BadAlphaCase(
        case_id="ba_logic_01",
        title="Mutually exclusive entry conditions that can never be true simultaneously",
        description=(
            "高胜率突破回调策略：当今日收盘价创过去 20 日最高收盘价、且今日收盘价低于过去 20 日简单移动平均线时买入。"
        ),
        code="""import pandas as pd

def build_dead_signal(df: pd.DataFrame) -> pd.Series:
    max20 = df["close"].rolling(20).max()
    sma20 = df["close"].rolling(20).mean()
    # Enter when price is at 20-day maximum AND strictly below 20-day average
    sig = ((df["close"] >= max20) & (df["close"] < sma20)).astype(int)
    return sig
""",
        has_flaw=True,
        flaw_types=("logical_inconsistency",),
        reference_notes="Mathematical impossibility: the maximum of 20 numbers can never be strictly less than their arithmetic mean, so the signal is permanently 0.",
    ),
    BadAlphaCase(
        case_id="ba_logic_02",
        title="Inverted volatility scaling formula increases leverage in high volatility",
        description=(
            "风险平价波动率目标策略：为了在市场波动加剧时降低仓位保护本金、在低波动时放大仓位，"
            "将基础信号乘以 (rolling_vol / target_vol) 进行缩放。"
        ),
        code="""import pandas as pd
from src.signals import compute_moving_average_signal

def build_inverted_vol_scaling(df: pd.DataFrame, target_vol: float = 0.15) -> pd.Series:
    base = compute_moving_average_signal(df, window=50)
    # Scale position inversely to risk
    scale = (df["rolling_vol"] / target_vol).clip(0.0, 2.0).fillna(1.0)
    return base * scale
""",
        has_flaw=True,
        flaw_types=("logical_inconsistency",),
        reference_notes="Formula is inverted: multiplying by `rolling_vol / target_vol` scales UP leverage when volatility is high and scales DOWN when volatility is low, contradicting the stated risk-parity goal.",
    ),
    BadAlphaCase(
        case_id="ba_logic_03",
        title="Mean reversion state machine with inverted entry/exit thresholds",
        description=(
            "Z-score 均值回归策略：当价格跌破均值 2 个标准差（z <= -2.0）时抄底做多，"
            "反弹回到均值附近（z >= 0.0）时止盈平仓。但代码中把入场写成了 z <= 0.0、出场写成 z >= -2.0。"
        ),
        code="""import numpy as np
import pandas as pd

def build_broken_state_machine(df: pd.DataFrame) -> pd.Series:
    m = df["close"].rolling(20).mean()
    s = df["close"].rolling(20).std()
    z = ((df["close"] - m) / s).to_numpy()
    out = np.zeros(len(df), dtype=int)
    state = 0
    for i, zi in enumerate(z):
        if np.isnan(zi):
            continue
        if state == 0 and zi <= 0.0:
            state = 1
        elif state == 1 and zi >= -2.0:
            state = 0
        out[i] = state
    return pd.Series(out, index=df.index)
""",
        has_flaw=True,
        flaw_types=("logical_inconsistency",),
        reference_notes="entry_z=0.0 is greater than exit_z=-2.0: as soon as state enters at z <= 0.0 (e.g. z = -0.5), the next bar immediately satisfies z >= -2.0 and exits.",
    ),

    # =========================================================================
    # 5. Compound-flaw cases (3 cases)
    # =========================================================================
    BadAlphaCase(
        case_id="ba_compound_01",
        title="Next-day close leak combined with daily high-friction flipping",
        description=(
            "超短线日内动量策略：比较明日收盘价与今日收盘价，并在单边滑点 10bps、佣金 10bps、"
            "零容忍带下每天调仓。"
        ),
        code="""import pandas as pd
from src.backtest import Backtest

def run_leaky_churn(df: pd.DataFrame):
    # Use next bar's close to decide today's signal and flip every day under 20bps one-way cost
    sig = (df["close"].shift(-1) > df["close"]).astype(int)
    bt = Backtest(slippage_rate=0.0010, commission_rate=0.0010, rebalance_tolerance=0.0)
    return bt.run(df, sig)
""",
        has_flaw=True,
        flaw_types=("lookahead_bias", "friction_erosion"),
        reference_notes="Contains both direct lookahead bias (`shift(-1)`) and daily turnover under 40bps round-trip friction with `rebalance_tolerance=0.0`.",
    ),
    BadAlphaCase(
        case_id="ba_compound_02",
        title="Full-sample quantile normalization plus hardcoded date & magic filters",
        description=(
            "全样本分位数择时策略：用整段历史的成交量分位数与收盘价均值归一化，"
            "同时剔除 2020 年 3 月并限制 RSI 精确落在 41.3 到 58.9 之间。"
        ),
        code="""import pandas as pd

def build_leaky_overfit(df: pd.DataFrame) -> pd.Series:
    # Full-history quantile and mean
    vol_q83 = df["volume"].quantile(0.83)
    z_full = (df["close"] - df["close"].mean()) / df["close"].std()
    not_march_2020 = ~((df.index >= "2020-03-01") & (df.index <= "2020-03-31"))
    not_thursday = df.index.dayofweek != 3
    sig = (df["volume"] > vol_q83) & (z_full > -0.314) & (z_full < 1.271) & not_march_2020 & not_thursday
    return sig.astype(int)
""",
        has_flaw=True,
        flaw_types=("lookahead_bias", "data_snooping"),
        reference_notes="Uses full-sample `.quantile(0.83)` and `.mean()/.std()` (lookahead_bias) AND hardcodes March 2020 exclusion + arbitrary decimal cutoffs (data_snooping).",
    ),
    BadAlphaCase(
        case_id="ba_compound_03",
        title="Contradictory RSI condition combined with overfitted calendar/window rules",
        description=(
            "多因子精调反转策略：要求 13 日 RSI 同时小于 24.6 且大于 71.8，"
            "并且仅在每月第 7、13、19 个交易日且周四时入场。"
        ),
        code="""import pandas as pd

def build_impossible_snooped(df: pd.DataFrame) -> pd.Series:
    delta = df["close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/13, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/13, adjust=False).mean()
    rsi13 = 100 - 100 / (1 + gain / loss)
    lucky_days = df.index.day.isin([7, 13, 19]) & (df.index.dayofweek == 3)
    # Require RSI < 24.6 AND RSI > 71.8 simultaneously
    sig = (rsi13 < 24.6) & (rsi13 > 71.8) & lucky_days
    return sig.astype(int)
""",
        has_flaw=True,
        flaw_types=("logical_inconsistency", "data_snooping"),
        reference_notes="`(rsi13 < 24.6) & (rsi13 > 71.8)` is logically impossible (always False), combined with arbitrary calendar day snooping.",
    ),

    # =========================================================================
    # 6. Clean negative control cases (3 cases, has_flaw=False)
    # =========================================================================
    BadAlphaCase(
        case_id="ba_clean_01",
        title="Clean 60-day SMA trend following with causal vol scaling",
        description=(
            "中长线趋势跟踪策略：当收盘价高于 60 日简单均线时做多，叠加 20 日历史实现波动率"
            "进行 15% 目标波动率缩放（上限 1.5 倍杠杆），次日开盘成交，容忍带 5%。"
        ),
        code="""import pandas as pd
from src.signals import compute_moving_average_signal, compute_volatility_scaled_signal
from src.backtest import Backtest

def run_clean_trend(df: pd.DataFrame):
    base = compute_moving_average_signal(df, window=60)
    scaled = compute_volatility_scaled_signal(df, base, target_vol=0.15, max_leverage=1.5)
    bt = Backtest(slippage_rate=0.0005, commission_rate=0.0005, rebalance_tolerance=0.05)
    return bt.run(df, scaled)
""",
        has_flaw=False,
        flaw_types=(),
        reference_notes="Clean causal implementation: slow 60-day trend, causal rolling vol scaling, 5% rebalance tolerance, and Backtest shifts signals by 1 day internally.",
    ),
    BadAlphaCase(
        case_id="ba_clean_02",
        title="Clean Wilder RSI(14) mean reversion state machine",
        description=(
            "标准 14 日 Wilder RSI 均值回归策略：预热期 14 日空仓，RSI < 30 进场做多，"
            "持仓直至 RSI > 70 平仓，次日开盘执行。"
        ),
        code="""import pandas as pd
from src.signals import compute_rsi_signal
from src.backtest import Backtest

def run_clean_rsi(df: pd.DataFrame):
    sig = compute_rsi_signal(df, window=14, lower=30.0, upper=70.0, mode="reversion")
    bt = Backtest(slippage_rate=0.0005, commission_rate=0.0005, rebalance_tolerance=0.05)
    return bt.run(df, sig)
""",
        has_flaw=False,
        flaw_types=(),
        reference_notes="Clean stateful Wilder RSI reversion using causal EWM, warm-up guard, and low turnover.",
    ),
    BadAlphaCase(
        case_id="ba_clean_03",
        title="Clean Donchian(55, 20) breakout filtered by rolling liquidity regimes",
        description=(
            "55 日唐奇安通道突破入场、20 日通道跌破出场（均使用 shift(1) 排除当日极值），"
            "仅在滚动一年高成交量象限 Q1 与 Q4 持仓。"
        ),
        code="""import pandas as pd
from src.signals import compute_breakout_signal
from src.regimes import classify_market_regimes
from src.backtest import Backtest

def run_clean_breakout(df: pd.DataFrame):
    classified = classify_market_regimes(df.copy())
    sig = compute_breakout_signal(classified, entry_window=55, exit_window=20)
    mask = classified["regime"].isna() | classified["regime"].isin(["Q1", "Q4"])
    filtered_sig = sig.where(mask, 0)
    bt = Backtest(slippage_rate=0.0005, commission_rate=0.0005, rebalance_tolerance=0.05)
    return bt.run(classified, filtered_sig)
""",
        has_flaw=False,
        flaw_types=(),
        reference_notes="Clean Donchian breakout (`shift(1)` on rolling high/low) combined with causal 252-day rolling median regime classification.",
    ),
]
