from __future__ import annotations

from typing import Any, Dict, List
import numpy as np
import pandas as pd

from src.signals import (
    compute_moving_average_signal,
    compute_macd_signal,
    compute_volatility_scaled_signal,
    compute_mean_reversion_signal,
    compute_breakout_signal,
    compute_rsi_signal,
    combine_signals,
)
from benchmark.layer1_code.schema import SystematicCase


# =============================================================================
# Category 1: Primitive signal implementations (6 cases)
# =============================================================================

def _golden_sys_mom_01(df: pd.DataFrame) -> Dict[str, Any]:
    sig = compute_moving_average_signal(df, window=60)
    return {
        "signals": sig,
        "slippage_rate": 0.0005,
        "commission_rate": 0.0005,
        "rebalance_tolerance": 0.05,
    }


def _golden_sys_macd_02(df: pd.DataFrame) -> Dict[str, Any]:
    sig = compute_macd_signal(df, fast=12, slow=26, signal_span=9)
    return {
        "signals": sig,
        "slippage_rate": 0.0005,
        "commission_rate": 0.0005,
        "rebalance_tolerance": 0.05,
    }


def _golden_sys_mrev_03(df: pd.DataFrame) -> Dict[str, Any]:
    sig = compute_mean_reversion_signal(df, window=20, entry_z=-1.5, exit_z=0.0)
    return {
        "signals": sig,
        "slippage_rate": 0.0005,
        "commission_rate": 0.0005,
        "rebalance_tolerance": 0.05,
    }


def _golden_sys_brk_04(df: pd.DataFrame) -> Dict[str, Any]:
    sig = compute_breakout_signal(df, entry_window=55, exit_window=20)
    return {
        "signals": sig,
        "slippage_rate": 0.0005,
        "commission_rate": 0.0005,
        "rebalance_tolerance": 0.05,
    }


def _golden_sys_rsi_rev_05(df: pd.DataFrame) -> Dict[str, Any]:
    sig = compute_rsi_signal(df, window=14, lower=30.0, upper=70.0, mode="reversion")
    return {
        "signals": sig,
        "slippage_rate": 0.0005,
        "commission_rate": 0.0005,
        "rebalance_tolerance": 0.05,
    }


def _golden_sys_rsi_trd_06(df: pd.DataFrame) -> Dict[str, Any]:
    sig = compute_rsi_signal(df, window=21, lower=30.0, upper=70.0, mode="trend")
    return {
        "signals": sig,
        "slippage_rate": 0.0005,
        "commission_rate": 0.0005,
        "rebalance_tolerance": 0.05,
    }


# =============================================================================
# Category 2: Overlays, Regime Filters & Signal Combinations (6 cases)
# =============================================================================

def _golden_sys_combo_all_07(df: pd.DataFrame) -> Dict[str, Any]:
    s1 = compute_moving_average_signal(df, window=50)
    s2 = compute_rsi_signal(df, window=14, lower=30.0, upper=70.0, mode="reversion")
    sig = combine_signals([s1, s2], mode="all")
    return {
        "signals": sig,
        "slippage_rate": 0.0005,
        "commission_rate": 0.0005,
        "rebalance_tolerance": 0.05,
    }


def _golden_sys_combo_mean_08(df: pd.DataFrame) -> Dict[str, Any]:
    s1 = compute_moving_average_signal(df, window=20)
    s2 = compute_macd_signal(df, fast=12, slow=26, signal_span=9)
    sig = combine_signals([s1, s2], mode="mean")
    return {
        "signals": sig,
        "slippage_rate": 0.0005,
        "commission_rate": 0.0005,
        "rebalance_tolerance": 0.05,
    }


def _golden_sys_regime_09(df: pd.DataFrame) -> Dict[str, Any]:
    base = compute_moving_average_signal(df, window=40)
    mask = df["regime"].isna() | df["regime"].isin(["Q1", "Q4"])
    sig = base.where(mask, 0)
    return {
        "signals": sig,
        "slippage_rate": 0.0005,
        "commission_rate": 0.0005,
        "rebalance_tolerance": 0.05,
    }


def _golden_sys_vol_10(df: pd.DataFrame) -> Dict[str, Any]:
    base = compute_moving_average_signal(df, window=100)
    sig = compute_volatility_scaled_signal(df, base, target_vol=0.15, max_leverage=1.5)
    return {
        "signals": sig,
        "slippage_rate": 0.0005,
        "commission_rate": 0.0005,
        "rebalance_tolerance": 0.05,
    }


def _golden_sys_regime_vol_11(df: pd.DataFrame) -> Dict[str, Any]:
    base = compute_breakout_signal(df, entry_window=40, exit_window=15)
    mask = df["regime"].isna() | df["regime"].isin(["Q1", "Q2"])
    masked = base.where(mask, 0)
    sig = compute_volatility_scaled_signal(df, masked, target_vol=0.12, max_leverage=2.0)
    return {
        "signals": sig,
        "slippage_rate": 0.0005,
        "commission_rate": 0.0005,
        "rebalance_tolerance": 0.05,
    }


def _golden_sys_combo_regime_12(df: pd.DataFrame) -> Dict[str, Any]:
    s1 = compute_mean_reversion_signal(df, window=30, entry_z=-1.0, exit_z=0.0)
    s2 = compute_rsi_signal(df, window=14, lower=35.0, upper=65.0, mode="reversion")
    combined = combine_signals([s1, s2], mode="any")
    mask = df["regime"].isna() | df["regime"].isin(["Q3", "Q4"])
    sig = combined.where(mask, 0)
    return {
        "signals": sig,
        "slippage_rate": 0.0005,
        "commission_rate": 0.0005,
        "rebalance_tolerance": 0.05,
    }


# =============================================================================
# Category 3: Friction Accounting & Custom Rule Boundaries (6 cases)
# =============================================================================

def _golden_sys_fric_high_13(df: pd.DataFrame) -> Dict[str, Any]:
    sig = compute_moving_average_signal(df, window=30)
    return {
        "signals": sig,
        "slippage_rate": 0.0015,
        "commission_rate": 0.0010,
        "rebalance_tolerance": 0.08,
    }


def _golden_sys_fric_zero_14(df: pd.DataFrame) -> Dict[str, Any]:
    sig = compute_macd_signal(df, fast=8, slow=21, signal_span=5)
    return {
        "signals": sig,
        "slippage_rate": 0.0002,
        "commission_rate": 0.0,
        "rebalance_tolerance": 0.02,
    }


def _golden_sys_fric_vol_15(df: pd.DataFrame) -> Dict[str, Any]:
    base = compute_moving_average_signal(df, window=50)
    sig = compute_volatility_scaled_signal(df, base, target_vol=0.10, max_leverage=1.2)
    return {
        "signals": sig,
        "slippage_rate": 0.0005,
        "commission_rate": 0.0005,
        "rebalance_tolerance": 0.10,
    }


def _golden_sys_custom_band_16(df: pd.DataFrame) -> Dict[str, Any]:
    sma20 = df["close"].rolling(20).mean()
    sma60 = df["close"].rolling(60).mean()
    sig = (sma20 > sma60).astype(int)
    return {
        "signals": sig,
        "slippage_rate": 0.0008,
        "commission_rate": 0.0004,
        "rebalance_tolerance": 0.05,
    }


def _golden_sys_custom_volfilter_17(df: pd.DataFrame) -> Dict[str, Any]:
    sma50 = df["close"].rolling(50).mean()
    sig = ((df["close"] > sma50) & (df["rolling_vol"] < 0.25)).astype(int)
    return {
        "signals": sig,
        "slippage_rate": 0.0005,
        "commission_rate": 0.0005,
        "rebalance_tolerance": 0.05,
    }


def _golden_sys_custom_halfpos_18(df: pd.DataFrame) -> Dict[str, Any]:
    delta = df["close"].diff()
    gain = delta.clip(lower=0.0).fillna(0.0)
    loss = (-delta.clip(upper=0.0)).fillna(0.0)
    avg_gain = gain.ewm(alpha=1.0 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / 14, adjust=False).mean().replace(0.0, np.nan)
    rsi = (100.0 - 100.0 / (1.0 + avg_gain / avg_loss)).fillna(100.0)
    weights = np.where(rsi < 25.0, 1.0, np.where(rsi < 40.0, 0.5, 0.0))
    weights[:14] = 0.0
    sig = pd.Series(weights, index=df.index, dtype=float)
    return {
        "signals": sig,
        "slippage_rate": 0.0006,
        "commission_rate": 0.0003,
        "rebalance_tolerance": 0.04,
    }


SYSTEMATIC_CASES: List[SystematicCase] = [
    SystematicCase(
        case_id="sys_mom_01",
        title="60-day SMA trend following",
        category="primitive",
        prompt=(
            "实现 60 日简单移动平均趋势策略：当当日收盘价严格大于 60 日简单移动平均线（SMA 60）时目标权重为 1，否则为 0。"
            "可直接调用 compute_moving_average_signal(df, window=60)。"
            "摩擦成本使用默认值：slippage_rate=0.0005, commission_rate=0.0005, rebalance_tolerance=0.05。"
        ),
        golden_fn=_golden_sys_mom_01,
        expected_factor_calls=("compute_moving_average_signal",),
    ),
    SystematicCase(
        case_id="sys_macd_02",
        title="MACD(12, 26, 9) trend strategy",
        category="primitive",
        prompt=(
            "实现标准 MACD(fast=12, slow=26, signal_span=9) 趋势策略：当 DIF 高于 DEA 时目标权重为 1，否则为 0。"
            "可直接调用 compute_macd_signal(df, fast=12, slow=26, signal_span=9)。"
            "摩擦成本使用默认值：slippage_rate=0.0005, commission_rate=0.0005, rebalance_tolerance=0.05。"
        ),
        golden_fn=_golden_sys_macd_02,
        expected_factor_calls=("compute_macd_signal",),
    ),
    SystematicCase(
        case_id="sys_mrev_03",
        title="20-day Z-score stateful mean reversion (-1.5 / 0.0)",
        category="primitive",
        prompt=(
            "实现 20 日 Z-score 状态机均值回归策略：当收盘价相对 20 日滚动均值/标准差的 Z-score 跌至 -1.5 或以下时入场做多（权重 1），"
            "持仓直至 Z-score 回升至 0.0 或以上时平仓（权重 0）。可直接调用 compute_mean_reversion_signal(df, window=20, entry_z=-1.5, exit_z=0.0)。"
            "摩擦成本使用默认值：slippage_rate=0.0005, commission_rate=0.0005, rebalance_tolerance=0.05。"
        ),
        golden_fn=_golden_sys_mrev_03,
        expected_factor_calls=("compute_mean_reversion_signal",),
    ),
    SystematicCase(
        case_id="sys_brk_04",
        title="Donchian(55, 20) channel breakout",
        category="primitive",
        prompt=(
            "实现唐奇安通道突破状态机策略：当收盘价突破过去 55 日最高价（不含当日，即 shift(1)）时入场做多（权重 1），"
            "当收盘价跌破过去 20 日最低价（不含当日）时平仓（权重 0）。可直接调用 compute_breakout_signal(df, entry_window=55, exit_window=20)。"
            "摩擦成本使用默认值：slippage_rate=0.0005, commission_rate=0.0005, rebalance_tolerance=0.05。"
        ),
        golden_fn=_golden_sys_brk_04,
        expected_factor_calls=("compute_breakout_signal",),
    ),
    SystematicCase(
        case_id="sys_rsi_rev_05",
        title="14-day Wilder RSI(30, 70) reversion",
        category="primitive",
        prompt=(
            "实现 14 日 Wilder RSI 均值回归策略（mode='reversion'）：前 14 日预热期空仓（权重 0），"
            "当 RSI < 30.0 时入场做多（权重 1），持仓直至 RSI > 70.0 时平仓（权重 0）。"
            "可直接调用 compute_rsi_signal(df, window=14, lower=30.0, upper=70.0, mode='reversion')。"
            "摩擦成本使用默认值：slippage_rate=0.0005, commission_rate=0.0005, rebalance_tolerance=0.05。"
        ),
        golden_fn=_golden_sys_rsi_rev_05,
        expected_factor_calls=("compute_rsi_signal",),
    ),
    SystematicCase(
        case_id="sys_rsi_trd_06",
        title="21-day Wilder RSI trend mode",
        category="primitive",
        prompt=(
            "实现 21 日 Wilder RSI 趋势策略（mode='trend'）：前 21 日预热期空仓（权重 0），"
            "之后只要 RSI > 50.0 目标权重为 1，否则为 0。"
            "可直接调用 compute_rsi_signal(df, window=21, lower=30.0, upper=70.0, mode='trend')。"
            "摩擦成本使用默认值：slippage_rate=0.0005, commission_rate=0.0005, rebalance_tolerance=0.05。"
        ),
        golden_fn=_golden_sys_rsi_trd_06,
        expected_factor_calls=("compute_rsi_signal",),
    ),
    SystematicCase(
        case_id="sys_combo_all_07",
        title="Combo ALL: 50-day SMA and 14-day RSI(30, 70) reversion",
        category="overlay_combo",
        prompt=(
            "实现双信号共振策略（mode='all'）：只有当 50 日均线信号（window=50）与 14 日 Wilder RSI 反转信号（window=14, lower=30.0, upper=70.0, mode='reversion'）"
            "同时为 1 时才做多（权重 1），否则为 0。可使用 combine_signals([s1, s2], mode='all')。"
            "摩擦成本使用默认值：slippage_rate=0.0005, commission_rate=0.0005, rebalance_tolerance=0.05。"
        ),
        golden_fn=_golden_sys_combo_all_07,
        expected_factor_calls=("compute_moving_average_signal", "compute_rsi_signal", "combine_signals"),
    ),
    SystematicCase(
        case_id="sys_combo_mean_08",
        title="Combo MEAN: equal-weight 20-day SMA and MACD(12, 26, 9)",
        category="overlay_combo",
        prompt=(
            "实现双信号等权组合策略（mode='mean'）：将 20 日均线信号（window=20）与 MACD(12, 26, 9) 信号取算术平均（权重可取 0.0, 0.5, 1.0）。"
            "可使用 combine_signals([s1, s2], mode='mean')。"
            "摩擦成本使用默认值：slippage_rate=0.0005, commission_rate=0.0005, rebalance_tolerance=0.05。"
        ),
        golden_fn=_golden_sys_combo_mean_08,
        expected_factor_calls=("compute_moving_average_signal", "compute_macd_signal", "combine_signals"),
    ),
    SystematicCase(
        case_id="sys_regime_09",
        title="40-day SMA restricted to high-volume regimes Q1 and Q4",
        category="overlay_combo",
        prompt=(
            "实现带 Regime 过滤的 40 日均线趋势策略：基础信号为 compute_moving_average_signal(df, window=40)，"
            "仅在 df['regime'] 为 'Q1' 或 'Q4'（或者 df['regime'] 为 NaN 的预热期）时保留信号，其余象限强制空仓（权重置 0）。"
            "摩擦成本使用默认值：slippage_rate=0.0005, commission_rate=0.0005, rebalance_tolerance=0.05。"
        ),
        golden_fn=_golden_sys_regime_09,
        expected_factor_calls=("compute_moving_average_signal",),
    ),
    SystematicCase(
        case_id="sys_vol_10",
        title="100-day SMA with 15% volatility target and 1.5x max leverage",
        category="overlay_combo",
        prompt=(
            "实现带目标波动率缩放的 100 日均线策略：基础信号为 compute_moving_average_signal(df, window=100)，"
            "调用 compute_volatility_scaled_signal(df, base, target_vol=0.15, max_leverage=1.5) 对仓位进行波动率缩放。"
            "摩擦成本使用默认值：slippage_rate=0.0005, commission_rate=0.0005, rebalance_tolerance=0.05。"
        ),
        golden_fn=_golden_sys_vol_10,
        expected_factor_calls=("compute_moving_average_signal", "compute_volatility_scaled_signal"),
    ),
    SystematicCase(
        case_id="sys_regime_vol_11",
        title="Donchian(40, 15) in Q1/Q2 regimes + 12% vol scaling (max 2.0x)",
        category="overlay_combo",
        prompt=(
            "实现叠加 Regime 过滤与波动率缩放的唐奇安突破策略："
            "1) 先计算唐奇安突破信号 compute_breakout_signal(df, entry_window=40, exit_window=15)；"
            "2) 仅在 df['regime'] 属于 ['Q1', 'Q2'] 或为 NaN 时保留信号，其余象限置 0；"
            "3) 再对过滤后的信号调用 compute_volatility_scaled_signal(df, masked, target_vol=0.12, max_leverage=2.0)。"
            "摩擦成本使用默认值：slippage_rate=0.0005, commission_rate=0.0005, rebalance_tolerance=0.05。"
        ),
        golden_fn=_golden_sys_regime_vol_11,
        expected_factor_calls=("compute_breakout_signal", "compute_volatility_scaled_signal"),
    ),
    SystematicCase(
        case_id="sys_combo_regime_12",
        title="Combo ANY of MeanRev(30, -1.0, 0.0) and RSI(14, 35, 65) in Q3/Q4",
        category="overlay_combo",
        prompt=(
            "实现低波象限双反转并集策略："
            "1) 信号 1 为 compute_mean_reversion_signal(df, window=30, entry_z=-1.0, exit_z=0.0)；"
            "2) 信号 2 为 compute_rsi_signal(df, window=14, lower=35.0, upper=65.0, mode='reversion')；"
            "3) 使用 combine_signals([s1, s2], mode='any') 求并集；"
            "4) 仅在 df['regime'] 属于 ['Q3', 'Q4'] 或为 NaN 时保留信号，其余象限置 0。"
            "摩擦成本使用默认值：slippage_rate=0.0005, commission_rate=0.0005, rebalance_tolerance=0.05。"
        ),
        golden_fn=_golden_sys_combo_regime_12,
        expected_factor_calls=("compute_mean_reversion_signal", "compute_rsi_signal", "combine_signals"),
    ),
    SystematicCase(
        case_id="sys_fric_high_13",
        title="30-day SMA with high friction (15bps slippage, 10bps commission, 8% tolerance)",
        category="friction_stateful",
        prompt=(
            "实现 30 日均线趋势策略（compute_moving_average_signal(df, window=30)），"
            "并严格按以下高摩擦交易制度配置返回参数："
            "单边滑点率 slippage_rate=0.0015（15bps），单边佣金率 commission_rate=0.0010（10bps），"
            "调仓容忍带 rebalance_tolerance=0.08（8%）。"
        ),
        golden_fn=_golden_sys_fric_high_13,
        expected_factor_calls=("compute_moving_average_signal",),
    ),
    SystematicCase(
        case_id="sys_fric_zero_14",
        title="MACD(8, 21, 5) with zero commission, 2bps slippage, 2% tolerance",
        category="friction_stateful",
        prompt=(
            "实现快速 MACD(fast=8, slow=21, signal_span=5) 趋势策略，"
            "并配置零佣金与窄容忍带参数：slippage_rate=0.0002（2bps），commission_rate=0.0（零佣金），"
            "rebalance_tolerance=0.02（2%）。"
        ),
        golden_fn=_golden_sys_fric_zero_14,
        expected_factor_calls=("compute_macd_signal",),
    ),
    SystematicCase(
        case_id="sys_fric_vol_15",
        title="50-day SMA + 10% vol scaling (max 1.2x) with 10% rebalance tolerance",
        category="friction_stateful",
        prompt=(
            "实现 50 日均线趋势叠加波动率缩放（target_vol=0.10, max_leverage=1.2），"
            "为减少波动率缩放带来的频繁微调，将调仓容忍带设为 rebalance_tolerance=0.10（10%），"
            "滑点与佣金保持 slippage_rate=0.0005, commission_rate=0.0005。"
        ),
        golden_fn=_golden_sys_fric_vol_15,
        expected_factor_calls=("compute_moving_average_signal", "compute_volatility_scaled_signal"),
    ),
    SystematicCase(
        case_id="sys_custom_band_16",
        title="Dual SMA crossover (SMA 20 > SMA 60) with 8bps slippage and 4bps commission",
        category="friction_stateful",
        prompt=(
            "实现双均线金叉策略：计算收盘价的 20 日简单均线（SMA 20）与 60 日简单均线（SMA 60），"
            "当 SMA 20 严格大于 SMA 60 时目标权重为 1，否则为 0。"
            "摩擦成本参数设为：slippage_rate=0.0008, commission_rate=0.0004, rebalance_tolerance=0.05。"
        ),
        golden_fn=_golden_sys_custom_band_16,
    ),
    SystematicCase(
        case_id="sys_custom_volfilter_17",
        title="50-day SMA trend with rolling_vol < 0.25 circuit breaker",
        category="friction_stateful",
        prompt=(
            "实现带波动率熔断的 50 日均线策略：仅当收盘价严格大于 50 日简单均线（df['close'] > SMA 50）"
            "且当日年化滚动波动率严格小于 25%（df['rolling_vol'] < 0.25）时目标权重为 1，否则为 0。"
            "摩擦参数使用默认值：slippage_rate=0.0005, commission_rate=0.0005, rebalance_tolerance=0.05。"
        ),
        golden_fn=_golden_sys_custom_volfilter_17,
    ),
    SystematicCase(
        case_id="sys_custom_halfpos_18",
        title="Two-tier stepped Wilder RSI(14) sizing (100% < 25, 50% in [25, 40), 0% >= 40)",
        category="friction_stateful",
        prompt=(
            "实现阶梯仓位 14 日 Wilder RSI 策略（无状态，按 ewm(alpha=1/14, adjust=False) 计算 RSI）："
            "前 14 个交易日（index 0..13）目标权重强制为 0.0；从第 15 个交易日（index >= 14）起，"
            "若 RSI < 25.0 则目标权重为 1.0；若 25.0 <= RSI < 40.0 则目标权重为 0.5；若 RSI >= 40.0 则目标权重为 0.0。"
            "摩擦参数设为：slippage_rate=0.0006, commission_rate=0.0003, rebalance_tolerance=0.04。"
        ),
        golden_fn=_golden_sys_custom_halfpos_18,
    ),
]
