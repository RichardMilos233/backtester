"""
Layer 1B task bank, hand-written stateful cases ("hf_*", 10 cases).

Each case is a Chinese natural-language spec with a golden implementation. Used by the
v2 prototype suite; the canonical 20 and the QuantCode-Bench port are in
`v3_hard_cases.py` and `qcb_cases.py`.
"""
from __future__ import annotations

import inspect
from typing import Any, Dict, List
import numpy as np
import pandas as pd

from src.backtest import Backtest
from benchmark.common.llm import BaseLLMClient, LLMUsage
from benchmark.layer1_code.impl.tasks.schema import HighFreedomImplCase


# --- Golden Reference Implementations for the 10 High-Freedom Stateful Cases ---

def _golden_hf_01(df: pd.DataFrame) -> Dict[str, Any]:
    prior_hi20 = df["close"].rolling(20, min_periods=20).max().shift(1).to_numpy()
    close_arr = df["close"].to_numpy()
    out = np.zeros(len(df), dtype=float)
    state = 0
    peak = 0.0
    for i in range(len(df)):
        c = close_arr[i]
        if state == 0:
            if not np.isnan(prior_hi20[i]) and c > prior_hi20[i]:
                state = 1
                peak = c
        else:
            if c > peak:
                peak = c
            if c < 0.94 * peak:
                state = 0
                peak = 0.0
        out[i] = float(state)
    return {"signals": pd.Series(out, index=df.index), "slippage_rate": 0.0005, "commission_rate": 0.0005, "rebalance_tolerance": 0.05}


def _golden_hf_02(df: pd.DataFrame) -> Dict[str, Any]:
    delta = df["close"].diff()
    gain = delta.clip(lower=0.0).fillna(0.0).ewm(alpha=1.0 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0.0)).fillna(0.0).ewm(alpha=1.0 / 14, adjust=False).mean().replace(0.0, np.nan)
    rsi = (100.0 - 100.0 / (1.0 + gain / loss)).fillna(100.0).to_numpy()
    close_arr = df["close"].to_numpy()

    out = np.zeros(len(df), dtype=float)
    state = 0
    entry_close = 0.0
    cooldown = 0
    for i in range(len(df)):
        if i < 14:
            continue
        if state == 0:
            if cooldown > 0:
                cooldown -= 1
            elif rsi[i] < 30.0:
                state = 1
                entry_close = close_arr[i]
        else:
            if rsi[i] > 65.0 or close_arr[i] < 0.95 * entry_close:
                state = 0
                entry_close = 0.0
                cooldown = 5
        out[i] = float(state)
    return {"signals": pd.Series(out, index=df.index), "slippage_rate": 0.0005, "commission_rate": 0.0005, "rebalance_tolerance": 0.05}


def _golden_hf_03(df: pd.DataFrame) -> Dict[str, Any]:
    m = df["close"].rolling(20, min_periods=20).mean().to_numpy()
    s = df["close"].rolling(20, min_periods=20).std().to_numpy()
    close_arr = df["close"].to_numpy()
    out = np.zeros(len(df), dtype=float)
    state = 0
    bars_held = 0
    for i in range(len(df)):
        if np.isnan(m[i]) or np.isnan(s[i]):
            continue
        if state == 0:
            if close_arr[i] <= m[i] - 1.5 * s[i]:
                state = 1
                bars_held = 1
        else:
            if close_arr[i] >= m[i] or bars_held >= 10:
                state = 0
                bars_held = 0
            else:
                bars_held += 1
        out[i] = float(state)
    return {"signals": pd.Series(out, index=df.index), "slippage_rate": 0.0005, "commission_rate": 0.0005, "rebalance_tolerance": 0.05}


def _golden_hf_04(df: pd.DataFrame) -> Dict[str, Any]:
    sma50 = df["close"].rolling(50, min_periods=50).mean().to_numpy()
    close_arr = df["close"].to_numpy()
    out = np.zeros(len(df), dtype=float)
    state = 0
    entry_price = 0.0
    stop_price = 0.0
    for i in range(len(df)):
        if np.isnan(sma50[i]):
            continue
        c = close_arr[i]
        if state == 0:
            if c > sma50[i]:
                state = 1
                entry_price = c
                stop_price = 0.96 * c
        else:
            if c >= 1.05 * entry_price:
                stop_price = max(stop_price, entry_price)
            if c < stop_price or c < sma50[i]:
                state = 0
                entry_price = 0.0
                stop_price = 0.0
        out[i] = float(state)
    return {"signals": pd.Series(out, index=df.index), "slippage_rate": 0.0005, "commission_rate": 0.0005, "rebalance_tolerance": 0.05}


def _golden_hf_05(df: pd.DataFrame) -> Dict[str, Any]:
    w_close = df["close"].resample("W-FRI", label="right", closed="right").last()
    w_ma8 = w_close.rolling(8, min_periods=8).mean()
    w_bull = (w_close > w_ma8).astype(float).reindex(df.index, method="ffill").fillna(0.0).to_numpy()

    m10 = df["close"].rolling(10, min_periods=10).mean().to_numpy()
    s10 = df["close"].rolling(10, min_periods=10).std().to_numpy()
    c_arr = df["close"].to_numpy()
    out = np.zeros(len(df), dtype=float)
    state = 0
    for i in range(len(df)):
        if np.isnan(m10[i]) or np.isnan(s10[i]) or s10[i] == 0:
            continue
        z = (c_arr[i] - m10[i]) / s10[i]
        if state == 0:
            if w_bull[i] == 1.0 and z <= -1.2:
                state = 1
        else:
            if z >= 0.0 or w_bull[i] == 0.0:
                state = 0
        out[i] = float(state)
    return {"signals": pd.Series(out, index=df.index), "slippage_rate": 0.0005, "commission_rate": 0.0005, "rebalance_tolerance": 0.05}


def _golden_hf_06(df: pd.DataFrame) -> Dict[str, Any]:
    sma20 = df["close"].rolling(20, min_periods=20).mean()
    sma60 = df["close"].rolling(60, min_periods=60).mean()
    sma120 = df["close"].rolling(120, min_periods=120).mean()

    base = pd.Series(0.0, index=df.index)
    cond1 = (sma20 > sma60) & (sma60 <= sma120)
    cond2 = (sma20 > sma60) & (sma60 > sma120)
    base[cond1] = 0.5
    base[cond2] = 1.0

    mask = df["regime"].isna() | df["regime"].isin(["Q1", "Q4"])
    base = base.where(mask, 0.0)

    vol = df["rolling_vol"].replace(0.0, np.nan)
    scale = (0.15 / vol).fillna(1.0)
    final_sig = (base * scale).clip(lower=0.0, upper=1.2).fillna(0.0)
    return {"signals": final_sig, "slippage_rate": 0.0005, "commission_rate": 0.0005, "rebalance_tolerance": 0.05}


def _golden_hf_07(df: pd.DataFrame) -> Dict[str, Any]:
    hi30 = df["close"].rolling(30, min_periods=30).max().shift(1).to_numpy()
    lo15 = df["close"].rolling(15, min_periods=15).min().shift(1).to_numpy()
    c_arr = df["close"].to_numpy()
    out = np.zeros(len(df), dtype=float)
    w = 0.0
    entry_c = 0.0
    for i in range(len(df)):
        c = c_arr[i]
        if w == 0.0:
            if not np.isnan(hi30[i]) and c > hi30[i]:
                w = 0.5
                entry_c = c
        else:
            if not np.isnan(lo15[i]) and c < lo15[i]:
                w = 0.0
                entry_c = 0.0
            elif w == 0.5 and c >= 1.03 * entry_c:
                w = 1.0
        out[i] = w
    return {"signals": pd.Series(out, index=df.index), "slippage_rate": 0.0005, "commission_rate": 0.0005, "rebalance_tolerance": 0.05}


def _golden_hf_08(df: pd.DataFrame) -> Dict[str, Any]:
    sma40 = df["close"].rolling(40, min_periods=40).mean().to_numpy()
    ret = df["return"].to_numpy()
    c_arr = df["close"].to_numpy()
    out = np.zeros(len(df), dtype=float)
    freeze = 0
    for i in range(len(df)):
        if not np.isnan(ret[i]) and ret[i] < -0.03:
            freeze = 8
            out[i] = 0.0
            continue
        if freeze > 0:
            freeze -= 1
            out[i] = 0.0
            continue
        if not np.isnan(sma40[i]) and c_arr[i] > sma40[i]:
            out[i] = 1.0
    return {"signals": pd.Series(out, index=df.index), "slippage_rate": 0.0005, "commission_rate": 0.0005, "rebalance_tolerance": 0.05}


def _golden_hf_09(df: pd.DataFrame) -> Dict[str, Any]:
    ema30 = df["close"].ewm(span=30, adjust=False).mean().to_numpy()
    c_arr = df["close"].to_numpy()
    out = np.zeros(len(df), dtype=float)
    state = 0
    for i in range(len(df)):
        if i < 30:
            continue
        if state == 0:
            if c_arr[i] > 1.015 * ema30[i]:
                state = 1
        else:
            if c_arr[i] < 0.990 * ema30[i]:
                state = 0
        out[i] = float(state)
    return {"signals": pd.Series(out, index=df.index), "slippage_rate": 0.0005, "commission_rate": 0.0005, "rebalance_tolerance": 0.05}


def _golden_hf_10(df: pd.DataFrame) -> Dict[str, Any]:
    sma50 = df["close"].rolling(50, min_periods=50).mean().to_numpy()
    c_arr = df["close"].to_numpy()
    reg_arr = df["regime"].to_numpy()
    out = np.zeros(len(df), dtype=float)
    w = 0.0
    for i in range(len(df)):
        if np.isnan(sma50[i]) or c_arr[i] <= sma50[i]:
            w = 0.0
        else:
            in_good_reg = pd.isna(reg_arr[i]) or (reg_arr[i] in ("Q1", "Q4"))
            if in_good_reg:
                w = 1.0
            else:
                w = max(0.0, w - 0.25)
        out[i] = w
    return {"signals": pd.Series(out, index=df.index), "slippage_rate": 0.0005, "commission_rate": 0.0005, "rebalance_tolerance": 0.05}


HIGH_FREEDOM_IMPL_CASES: List[HighFreedomImplCase] = [
    HighFreedomImplCase(
        case_id="hf_trailing_stop_01",
        title="20-day high breakout with 6% post-entry high-water-mark trailing stop",
        state_complexity="long_state_ratchet",
        prompt=(
            "实现 20 日收盘价突破加 6% 高水位移动止损策略：当空仓且今日收盘价创过去 20 个完整交易日（不含今日）新高时满仓做多（权重 1.0）；"
            "持仓期间持续跟踪自该笔建仓以来的最高收盘价，一旦今日收盘价较该持仓最高收盘价回落超过 6%（即低于最高价的 94%）则当日平仓归零。"
            "未满 20 日预热期时保持空仓 0.0。使用默认摩擦参数。"
        ),
        golden_fn=_golden_hf_01,
    ),
    HighFreedomImplCase(
        case_id="hf_cooldown_02",
        title="Wilder RSI(14) reversion with 5% stop-loss and 5-bar cooldown clock",
        state_complexity="long_state_ratchet",
        prompt=(
            "实现带止损与 5 日冷却期的 14 日 Wilder RSI 均值回归策略：前 14 个交易日为预热期保持空仓；"
            "空仓且不处于冷却期时，若当日 Wilder RSI(14) 低于 30 则满仓做多；持仓期间若 RSI(14) 高于 65 或收盘价较入场当日收盘价下跌超过 5%（低于入场价的 95%），"
            "则立即平仓归零，并在随后的 5 个交易日内强制冷却禁止再次开仓。使用默认摩擦参数。"
        ),
        golden_fn=_golden_hf_02,
    ),
    HighFreedomImplCase(
        case_id="hf_time_stop_03",
        title="20-day Bollinger(-1.5 std) dip entry with 10-bar maximum holding time stop",
        state_complexity="medium_state",
        prompt=(
            "实现带最长持仓时间止损（Time Stop）的 20 日布林带均值回归策略：计算收盘价 20 日简单均线与 20 日标准差（预热期保持空仓）；"
            "空仓时若收盘价跌破或等于『20日均线 - 1.5倍标准差』则满仓入场（计为持仓第 1 天）；"
            "持仓期间若收盘价回升至 20 日均线及以上、或者已经连续持仓满 10 个交易日，则在当日收盘信号平仓归零。使用默认摩擦参数。"
        ),
        golden_fn=_golden_hf_03,
    ),
    HighFreedomImplCase(
        case_id="hf_breakeven_ratchet_04",
        title="50-day SMA trend with -4% initial stop ratcheting to breakeven at +5% gain",
        state_complexity="long_state_ratchet",
        prompt=(
            "实现带保本止损棘轮（Breakeven Ratchet Stop）的 50 日简单均线策略：空仓时若收盘价站上 50 日均线则满仓做多，"
            "初始止损线设为入场收盘价下方 4%（0.96 倍入场价）；持仓期间一旦某日收盘价较入场价浮盈达到或超过 +5%（1.05 倍入场价），"
            "止损线永久上移锁定至入场保本价（1.00 倍入场价）；持仓期间若收盘价跌破当前止损线或跌破 50 日均线则平仓归零。使用默认摩擦参数。"
        ),
        golden_fn=_golden_hf_04,
    ),
    HighFreedomImplCase(
        case_id="hf_mtf_weekly_05",
        title="Causal weekly 8-week SMA filter gating daily 10-day Z-score mean reversion",
        state_complexity="multi_tf_sizing",
        prompt=(
            "实现周线趋势过滤下的日线均值回归策略：以每周五收盘构建周线收盘价并计算 8 周简单均线，向后对齐到日频；"
            "日线上计算 10 日收盘价 Z-score。空仓时，仅当最新已完成的周线处于 8 周均线上方且日线 Z-score <= -1.2 时满仓做多；"
            "持仓时，若日线 Z-score >= 0.0 或周线跌破 8 周均线则平仓归零。使用默认摩擦参数。"
        ),
        golden_fn=_golden_hf_05,
    ),
    HighFreedomImplCase(
        case_id="hf_vol_cap_chain_06",
        title="Two-tier SMA(20/60/120) trend + Q1/Q4 filter + 15% vol scaling capped at 1.2x",
        state_complexity="multi_tf_sizing",
        max_leverage=1.2,
        prompt=(
            "实现分级均线趋势叠加象限过滤与波动率目标策略：当 20 日均线高于 60 日均线且 60 日均线高于 120 日均线时基础仓位为 1.0，"
            "当 20 日均线高于 60 日均线但 60 日均线不高于 120 日均线时基础仓位为 0.5，其余情况为 0.0；"
            "仅在高成交量市场象限（Q1、Q4 以及 regime 为 NaN 的预热期）保留基础仓位，在 Q2、Q3 象限必须清零；"
            "最后按 15% 目标年化波动率对仓位进行反比缩放，并确保最终输出的目标权重上限严禁超过 1.2 倍。使用默认摩擦参数。"
        ),
        golden_fn=_golden_hf_06,
    ),
    HighFreedomImplCase(
        case_id="hf_pyramid_scale_07",
        title="Donchian(30, 15) breakout with 2-tranche pyramiding (50% initial, 100% at +3%)",
        state_complexity="long_state_ratchet",
        prompt=(
            "实现分批加仓（Pyramiding）通道突破策略：当空仓且今日收盘价突破过去 30 个交易日（不含今日）最高收盘价时建立 50% 首仓（权重 0.5）；"
            "在持有首仓期间，若收盘价较首仓入场收盘价继续上涨达到或超过 +3%，则加满至 100% 仓位（权重 1.0）；"
            "无论处于半仓还是全仓，一旦今日收盘价跌破过去 15 个交易日（不含今日）最低收盘价则全部清仓归零。使用默认摩擦参数。"
        ),
        golden_fn=_golden_hf_07,
    ),
    HighFreedomImplCase(
        case_id="hf_shock_freeze_08",
        title="40-day SMA trend with -3% single-day crash liquidation and 8-bar freeze",
        state_complexity="medium_state",
        prompt=(
            "实现带单日暴跌熔断锁定期的 40 日均线趋势策略：正常情况下当收盘价高于 40 日简单均线时满仓（1.0），否则空仓（0.0）；"
            "但只要某日单日跌幅超过 3%（即当日 return < -0.03），当日立即强制清仓（0.0），并且在随后的连续 8 个交易日内处于熔断锁定期（无论均线如何均强制保持 0.0 仓位；若锁定期内再次发生单日跌幅超 3%，锁定期重新重置为 8 个交易日）。使用默认摩擦参数。"
        ),
        golden_fn=_golden_hf_08,
    ),
    HighFreedomImplCase(
        case_id="hf_asym_band_09",
        title="30-day EMA asymmetric hysteresis band (+1.5% upper entry, -1.0% lower exit)",
        state_complexity="medium_state",
        prompt=(
            "实现 30 日 EMA 非对称迟滞通道（Hysteresis Band）策略：计算收盘价的 30 日指数移动平均（ewm span=30, adjust=False），"
            "前 30 个交易日为预热期保持空仓；之后当空仓且收盘价突破 30 日 EMA 上方 1.5% 时满仓做多；"
            "持仓期间直到收盘价跌破 30 日 EMA 下方 1.0% 时才平仓归零，处于上下轨之间时保持原仓位状态不变。使用默认摩擦参数。"
        ),
        golden_fn=_golden_hf_09,
    ),
    HighFreedomImplCase(
        case_id="hf_regime_decay_10",
        title="50-day SMA with step-down 0.25/day position decay in low-volume Q2/Q3 regimes",
        state_complexity="multi_tf_sizing",
        prompt=(
            "实现带低量象限阶梯减仓的 50 日均线策略：当收盘价不高于 50 日简单均线（或处于前 49 日预热期）时立即清仓（权重 0.0）；"
            "当收盘价高于 50 日简单均线时，若当日处于高成交量象限（Q1、Q4 或预热期 NaN）则目标权重为 1.0；"
            "若收盘价仍高于 50 日均线但当日处于低成交量象限（Q2 或 Q3），则在上一交易日目标权重的基础上每天递减 0.25（最低减至 0.0）。使用默认摩擦参数。"
        ),
        golden_fn=_golden_hf_10,
    ),
]


class GoldenHighFreedomClient(BaseLLMClient):
    def __init__(self):
        self.usage = LLMUsage()
        self._by_id = {c.case_id: c for c in HIGH_FREEDOM_IMPL_CASES}

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        self.usage.calls += 1
        for cid, c in self._by_id.items():
            if cid in prompt:
                lines = inspect.getsource(c.golden_fn).splitlines()
                lines[0] = "def build_strategy(df):"
                return "```python\n" + "\n".join(lines) + "\n```"
        return "```python\ndef build_strategy(df):\n    return {'signals': df['close']*0}\n```"
