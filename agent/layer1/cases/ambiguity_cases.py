from __future__ import annotations

from typing import List
from agent.layer1.schema import AmbiguityCase

AMBIGUITY_CASES: List[AmbiguityCase] = [
    # =========================================================================
    # 8 Ambiguous Strategy Prompts (expected action = "clarify")
    # =========================================================================
    AmbiguityCase(
        case_id="amb_window_01",
        title="Moving average trend missing lookback window",
        prompt=(
            "实现简单移动平均趋势跟踪策略：当每日收盘价站上简单均线（SMA）时满仓做多（目标权重 1.0），"
            "跌破简单均线时清仓（目标权重 0.0）。单边滑点 5bps，佣金 5bps，调仓容忍带 5%，T+1 开盘执行。"
        ),
        is_ambiguous=True,
        required_slots=("lookback_window",),
        slot_keywords={
            "lookback_window": ("均线周期", "窗口", "多少天", "多少日", "window", "period", "lookback", "sma"),
        },
        reference_questions=("请问简单移动平均线（SMA）的回看窗口周期（window）是多少个交易日？",),
    ),
    AmbiguityCase(
        case_id="amb_window_02",
        title="Donchian channel breakout missing entry/exit windows",
        prompt=(
            "实现唐奇安通道突破状态机策略：当收盘价突破近期最高价（不含当日）时入场满仓做多，"
            "当收盘价跌破近期最低价（不含当日）时平仓退出。摩擦参数使用默认 5bps 滑点、5bps 佣金、5% 容忍带，次日开盘成交。"
        ),
        is_ambiguous=True,
        required_slots=("lookback_window",),
        slot_keywords={
            "lookback_window": ("entry_window", "exit_window", "突破窗口", "周期", "多少天", "多少日", "近期", "window", "lookback"),
        },
        reference_questions=("请问突破入场的高点回看窗口（entry_window）和跌破出场的低点回看窗口（exit_window）分别是多少天？",),
    ),
    AmbiguityCase(
        case_id="amb_thresh_03",
        title="14-day Wilder RSI reversion missing oversold/overbought thresholds",
        prompt=(
            "使用 14 日 Wilder RSI 指标构建均值回归状态机策略：前 14 个交易日预热期空仓，"
            "之后当 RSI 进入超卖区时全仓买入（权重 1.0），持仓直至 RSI 回到超买区时平仓（权重 0.0）。"
            "次日开盘执行，默认摩擦成本（slippage=5bps, commission=5bps, tolerance=5%）。"
        ),
        is_ambiguous=True,
        required_slots=("entry_exit_threshold",),
        slot_keywords={
            "entry_exit_threshold": ("阈值", "超卖", "超买", "lower", "upper", "threshold", "30", "70", "入场", "出场"),
        },
        reference_questions=("请问 RSI 超卖入场阈值（lower）和超买平仓阈值（upper）具体设为多少（例如 30 和 70）？",),
    ),
    AmbiguityCase(
        case_id="amb_thresh_04",
        title="20-day Z-score mean reversion missing entry_z and exit_z",
        prompt=(
            "基于收盘价的 20 日滚动均值与标准差计算 Z-score，当价格大幅跌离均值时入场做多，"
            "待价格回归均值附近后平仓退出。T+1 开盘成交，滑点与佣金各 5bps，调仓容忍带 5%。"
        ),
        is_ambiguous=True,
        required_slots=("entry_exit_threshold",),
        slot_keywords={
            "entry_exit_threshold": ("entry_z", "exit_z", "阈值", "z-score", "标准差", "threshold", "入场", "平仓"),
        },
        reference_questions=("请问触发做多的入场 Z-score 阈值（entry_z）和触发平仓的出场 Z-score 阈值（exit_z）分别是多少？",),
    ),
    AmbiguityCase(
        case_id="amb_vol_05",
        title="60-day SMA with unspecified volatility scaling parameters",
        prompt=(
            "基础信号采用 60 日均线趋势（收盘价 > 60 日 SMA 为 1，否则为 0），"
            "请在此基础上叠加目标波动率缩放（Volatility Scaling）动态调整每日杠杆仓位。"
            "摩擦成本使用默认值（slippage=5bps, commission=5bps, rebalance_tolerance=5%），次日开盘成交。"
        ),
        is_ambiguous=True,
        required_slots=("volatility_definition",),
        slot_keywords={
            "volatility_definition": ("target_vol", "目标波动率", "max_leverage", "最大杠杆", "波动率窗口", "volatility"),
        },
        reference_questions=("请问目标年化波动率（target_vol）、波动率计算窗口以及最大杠杆上限（max_leverage）分别是多少？",),
    ),
    AmbiguityCase(
        case_id="amb_vol_06",
        title="50-day SMA with vague 'panic/high volatility' circuit breaker",
        prompt=(
            "实现 50 日简单均线趋势策略（收盘价 > SMA 50 做多），但当市场处于恐慌、波动率过高时触发熔断强制空仓。"
            "单边滑点 5bps，佣金 5bps，调仓容忍带 5%，次日开盘执行。"
        ),
        is_ambiguous=True,
        required_slots=("volatility_definition", "entry_exit_threshold"),
        slot_keywords={
            "volatility_definition": ("波动率", "恐慌", "rolling_vol", "volatility", "定义", "指标"),
            "entry_exit_threshold": ("阈值", "熔断", "上限", "threshold", "cutoff", "多少"),
        },
        reference_questions=(
            "请问如何量化定义市场恐慌/波动率指标（例如使用 20 日年化滚动波动率 rolling_vol 还是 Regime 象限）？",
            "请问触发强制空仓的波动率熔断阈值具体是多少？",
        ),
    ),
    AmbiguityCase(
        case_id="amb_exec_07",
        title="Price-volume surge trigger missing execution timing and exit/holding period",
        prompt=(
            "构建放量突破做多策略：当某日收益率高于 +2.5% 且当日成交量超过 20 日均量的 1.8 倍时触发做多信号（目标权重 1.0），"
            "买入后持有一段时间平仓。单边滑点 5bps，佣金 5bps。"
        ),
        is_ambiguous=True,
        required_slots=("execution_timing", "entry_exit_threshold"),
        slot_keywords={
            "execution_timing": ("成交时点", "次日开盘", "持仓", "多久", "何时平仓", "holding", "execution", "timing", "调仓"),
            "entry_exit_threshold": ("平仓条件", "出场", "止盈", "止损", "exit", "持有天数"),
        },
        reference_questions=(
            "请问信号触发后是在次日开盘成交吗？持仓多少个交易日或满足什么出场条件时平仓？",
        ),
    ),
    AmbiguityCase(
        case_id="amb_compound_08",
        title="Underspecified discretionary momentum + vol sizing prompt",
        prompt=(
            "帮我写一个动量趋势加波动率风险控制的策略：近期走势强的时候做多，并且根据市场波动情况动态调整仓位大小。"
        ),
        is_ambiguous=True,
        required_slots=("lookback_window", "entry_exit_threshold", "volatility_definition"),
        slot_keywords={
            "lookback_window": ("窗口", "周期", "近期", "window", "lookback", "多少天"),
            "entry_exit_threshold": ("走势强", "阈值", "入场", "出场", "threshold", "条件"),
            "volatility_definition": ("波动", "target_vol", "max_leverage", "volatility", "缩放"),
        },
        reference_questions=(
            "请问动量指标的具体计算方式与回看窗口周期（lookback_window）是多少？",
            "请问判断走势强弱的入场/出场阈值（entry_exit_threshold）是什么？",
            "请问波动率度量方式、目标波动率（target_vol）及最大杠杆（volatility_definition）如何设定？",
        ),
    ),

    # =========================================================================
    # 4 Unambiguous Control Prompts (expected action = "implement")
    # =========================================================================
    AmbiguityCase(
        case_id="amb_clear_09",
        title="Complete 60-day SMA trend specification",
        prompt=(
            "实现 60 日简单移动平均趋势策略：计算收盘价的 60 日简单算术均值 SMA(60)，"
            "当当日收盘价严格大于 SMA(60) 时目标权重为 1.0，否则为 0.0。"
            "信号在 T+1 日开盘成交（由 Backtest 内部自动 shift(1)），"
            "摩擦参数为 slippage_rate=0.0005, commission_rate=0.0005, rebalance_tolerance=0.05。"
        ),
        is_ambiguous=False,
        required_slots=(),
        reference_questions=(),
    ),
    AmbiguityCase(
        case_id="amb_clear_10",
        title="Complete 14-day Wilder RSI(30, 70) reversion specification",
        prompt=(
            "实现 14 日 Wilder RSI 状态机均值回归策略（调用 compute_rsi_signal(df, window=14, lower=30.0, upper=70.0, mode='reversion')）："
            "前 14 个交易日目标权重为 0；之后当 RSI < 30.0 时入场做多（权重 1.0），持仓直至 RSI > 70.0 时平仓（权重 0.0）。"
            "次日开盘成交，摩擦成本使用 slippage_rate=0.0005, commission_rate=0.0005, rebalance_tolerance=0.05。"
        ),
        is_ambiguous=False,
        required_slots=(),
        reference_questions=(),
    ),
    AmbiguityCase(
        case_id="amb_clear_11",
        title="Complete 20-day Z-score mean reversion with 12% vol scaling",
        prompt=(
            "实现带波动率缩放的 20 日 Z-score 均值回归策略："
            "1) 基础信号调用 compute_mean_reversion_signal(df, window=20, entry_z=-1.5, exit_z=0.0)；"
            "2) 使用 df['rolling_vol']（20 日年化波动率）调用 compute_volatility_scaled_signal(df, base, target_vol=0.12, max_leverage=1.5)；"
            "3) 次日开盘成交，摩擦参数为 slippage_rate=0.0005, commission_rate=0.0005, rebalance_tolerance=0.05。"
        ),
        is_ambiguous=False,
        required_slots=(),
        reference_questions=(),
    ),
    AmbiguityCase(
        case_id="amb_clear_12",
        title="Complete Donchian(55, 20) breakout filtered to Q1 and Q4 regimes",
        prompt=(
            "实现带 Regime 过滤的唐奇安通道突破策略："
            "1) 调用 compute_breakout_signal(df, entry_window=55, exit_window=20) 计算 55 日前高突破入场、20 日前低跌破出场信号；"
            "2) 仅在 df['regime'] 属于 ['Q1', 'Q4'] 或为 NaN（预热期）时保留信号，其余象限置 0；"
            "3) 次日开盘执行，摩擦参数设为 slippage_rate=0.0005, commission_rate=0.0005, rebalance_tolerance=0.05。"
        ),
        is_ambiguous=False,
        required_slots=(),
        reference_questions=(),
    ),
]
