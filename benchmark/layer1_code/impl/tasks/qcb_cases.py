"""
Task bank: strategies ported from QuantCode-Bench (arXiv 2604.15151, MIT, LimexAILab/QuantCode-Bench).

Each case keeps the original loose wording in `original_prompt` and carries a pinned,
unambiguous specification in `prompt`. Both share one golden. The pinning decisions are
listed in results/layer1_impl/qcb_screen.md and restated inside each spec.

Conventions shared by every spec (also stated in CONVENTIONS, prepended to each prompt):
  * decisions are made on the close of bar t and signals[t] is the target weight at the
    end of bar t (the Backtest shifts by one bar);
  * when holding, exit rules are checked before any add/entry rule; a bar on which the
    position is closed is flat and re-entry is possible from the next bar at the earliest;
  * all price tests use `close` unless the spec says high/low;
  * SMA(n) = rolling mean with n bars; EMA(n) = ewm(span=n, adjust=False); RSI(n) and
    ATR(n) are Wilder (ewm(alpha=1/n, adjust=False)); Bollinger std uses ddof=1;
    MACD = (12, 26, 9).
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from benchmark.layer1_code.impl.tasks.indicators import (
    sma, ema, wilder_rsi, wilder_atr, bollinger, macd, default_output,
)
from benchmark.layer1_code.impl.tasks.schema import HighFreedomImplCase

CONVENTIONS = (
    "通用约定：所有判断在 bar t 收盘时做出，signals[t] 为 bar t 收盘后的目标仓位（Backtest 内部再 shift 1 根执行）。"
    "持仓时先检查全部出场规则，再检查加仓/入场规则；平仓当根 bar 仓位为 0，最早下一根 bar 才能重新入场。"
    "除特别说明外价格判断一律用 close。指标定义：SMA(n)=rolling(n).mean()；EMA(n)=ewm(span=n, adjust=False).mean()；"
    "RSI(n) 与 ATR(n) 用 Wilder 平滑 ewm(alpha=1/n, adjust=False)，ATR 的 TR 第一根取 high-low；"
    "布林带标准差用 pandas 默认 ddof=1；MACD 参数 (12, 26, 9)，柱 = DIF - DEA。"
    "预热期内（指标未定义或题目指定的根数）保持空仓。使用默认摩擦成本。"
)


def _spec(text: str) -> str:
    return CONVENTIONS + "\n\n" + text


# ---------------------------------------------------------------------------
# qcb_101: ATR trailing stop + MACD histogram exit
# ---------------------------------------------------------------------------
def _golden_qcb_101(df: pd.DataFrame) -> Dict[str, Any]:
    close = df["close"].astype(float)
    high = df["high"].astype(float).to_numpy()
    c = close.to_numpy()
    sma9 = sma(close, 9).to_numpy()
    _, _, hist = macd(close)
    h = hist.to_numpy()
    atr = wilder_atr(df, 14).to_numpy()
    n = len(df)
    out = np.zeros(n)
    holding = False
    stop = 0.0
    for t in range(35, n):
        if holding:
            cross_dn = h[t - 1] >= 0.0 and h[t] < 0.0
            if c[t] <= stop or cross_dn:
                holding = False
            else:
                stop = max(stop, high[t] - 2.0 * atr[t])
                out[t] = 1.0
        else:
            cross_up = h[t - 1] <= 0.0 and h[t] > 0.0
            if c[t] > sma9[t] and cross_up:
                holding = True
                stop = c[t] - 2.0 * atr[t]
                out[t] = 1.0
    return default_output(out, df.index)


# ---------------------------------------------------------------------------
# qcb_284: Bollinger breakout + 1% ratcheting trailing stop
# ---------------------------------------------------------------------------
def _golden_qcb_284(df: pd.DataFrame) -> Dict[str, Any]:
    close = df["close"].astype(float)
    high = df["high"].astype(float).to_numpy()
    c = close.to_numpy()
    _, _, upper = bollinger(close, 20, 2.0)
    up = upper.to_numpy()
    rsi = wilder_rsi(close, 14).to_numpy()
    n = len(df)
    out = np.zeros(n)
    holding = False
    highest = 0.0
    stop = 0.0
    for t in range(20, n):
        if holding:
            highest = max(highest, high[t])
            stop = max(stop, highest * 0.99)
            if c[t] <= stop:
                holding = False
            else:
                out[t] = 1.0
        else:
            if np.isnan(up[t - 1]):
                continue
            if c[t - 1] <= up[t - 1] and c[t] > up[t] and rsi[t] > 55.0:
                holding = True
                highest = high[t]
                stop = c[t] * 0.99
                out[t] = 1.0
    return default_output(out, df.index)


# ---------------------------------------------------------------------------
# qcb_226: SMA252 regime + ATR% gate + 2% stop / 5% take-profit
# ---------------------------------------------------------------------------
def _golden_qcb_226(df: pd.DataFrame) -> Dict[str, Any]:
    close = df["close"].astype(float)
    c = close.to_numpy()
    sma252 = sma(close, 252).to_numpy()
    atr_pct = (sma(wilder_atr(df, 14), 252) / close).to_numpy()
    n = len(df)
    out = np.zeros(n)
    holding = False
    entry = 0.0
    for t in range(251, n):
        if holding:
            if c[t] < sma252[t] or c[t] <= entry * 0.98 or c[t] >= entry * 1.05:
                holding = False
            else:
                out[t] = 1.0
        else:
            if c[t] > sma252[t] and atr_pct[t] <= 0.05:
                holding = True
                entry = c[t]
                out[t] = 1.0
    return default_output(out, df.index)


# ---------------------------------------------------------------------------
# qcb_064: EMA ribbon ordering + 2% stop / 4% take-profit
# ---------------------------------------------------------------------------
def _golden_qcb_064(df: pd.DataFrame) -> Dict[str, Any]:
    close = df["close"].astype(float)
    c = close.to_numpy()
    e = {p: ema(close, p).to_numpy() for p in (8, 13, 21, 34, 55)}
    ordered = (e[8] > e[13]) & (e[13] > e[21]) & (e[21] > e[34]) & (e[34] > e[55])
    fast_min = np.minimum(np.minimum(e[8], e[13]), e[21])
    slow_max = np.maximum(e[34], e[55])
    broken = fast_min <= slow_max
    n = len(df)
    out = np.zeros(n)
    holding = False
    entry = 0.0
    for t in range(55, n):
        if holding:
            if broken[t] or c[t] <= entry * 0.98 or c[t] >= entry * 1.04:
                holding = False
            else:
                out[t] = 1.0
        else:
            if ordered[t] and not ordered[t - 1]:
                holding = True
                entry = c[t]
                out[t] = 1.0
    return default_output(out, df.index)


# ---------------------------------------------------------------------------
# qcb_317: first-to-fifth trading day of month, SMA20 filter, 2% stop
# ---------------------------------------------------------------------------
def _golden_qcb_317(df: pd.DataFrame) -> Dict[str, Any]:
    close = df["close"].astype(float)
    c = close.to_numpy()
    sma20 = sma(close, 20).to_numpy()
    idx = pd.DatetimeIndex(df.index)
    month_key = idx.year * 12 + idx.month
    k = np.ones(len(df), dtype=int)
    for t in range(1, len(df)):
        k[t] = k[t - 1] + 1 if month_key[t] == month_key[t - 1] else 1
    n = len(df)
    out = np.zeros(n)
    holding = False
    entry = 0.0
    for t in range(19, n):
        if holding:
            if k[t] >= 5 or c[t] <= entry * 0.98:
                holding = False
            else:
                out[t] = 1.0
        else:
            if k[t] == 1 and c[t] > sma20[t]:
                holding = True
                entry = c[t]
                out[t] = 1.0
    return default_output(out, df.index)


# ---------------------------------------------------------------------------
# qcb_089: SMA20/50 cross entry, exit rule depends on open profit
# ---------------------------------------------------------------------------
def _golden_qcb_089(df: pd.DataFrame) -> Dict[str, Any]:
    close = df["close"].astype(float)
    c = close.to_numpy()
    s20 = sma(close, 20).to_numpy()
    s50 = sma(close, 50).to_numpy()
    n = len(df)
    out = np.zeros(n)
    holding = False
    entry = 0.0
    peak = 0.0
    for t in range(50, n):
        if holding:
            peak = max(peak, c[t])
            profit = c[t] / entry - 1.0
            if profit < 0.10:
                exit_now = s20[t] <= s50[t]
            else:
                exit_now = c[t] <= peak * 0.95
            if exit_now:
                holding = False
            else:
                out[t] = 1.0
        else:
            if s20[t - 1] <= s50[t - 1] and s20[t] > s50[t]:
                holding = True
                entry = c[t]
                peak = c[t]
                out[t] = 1.0
    return default_output(out, df.index)


# ---------------------------------------------------------------------------
# qcb_227: universal portfolio (wealth-weighted constant fractions)
# ---------------------------------------------------------------------------
def _golden_qcb_227(df: pd.DataFrame) -> Dict[str, Any]:
    close = df["close"].astype(float)
    r = (close / close.shift(1) - 1.0).to_numpy()
    n = len(df)
    out = np.zeros(n)
    ws = np.round(np.arange(0.0, 1.0001, 0.1), 1)
    for t in range(252, n):
        window = r[t - 251 : t + 1]
        wealth = np.array([np.prod(1.0 + w * window) for w in ws])
        out[t] = float(np.sum(ws * wealth) / np.sum(wealth))
    return default_output(out, df.index)


# ---------------------------------------------------------------------------
# qcb_013: pyramiding on RSI with partial exits
# ---------------------------------------------------------------------------
def _golden_qcb_013(df: pd.DataFrame) -> Dict[str, Any]:
    close = df["close"].astype(float)
    c = close.to_numpy()
    e50 = ema(close, 50).to_numpy()
    rsi = wilder_rsi(close, 14).to_numpy()
    n = len(df)
    out = np.zeros(n)
    p = 0.0
    avg_entry = 0.0
    n_entries = 0
    armed70 = False
    seen60 = False
    underwater = False
    be_done = False
    tp_done = False
    for t in range(50, n):
        if p > 0.0:
            if rsi[t] >= 60.0:
                seen60 = True
            if c[t] < e50[t] or (seen60 and rsi[t] < 40.0):
                p = 0.0
                continue
            if not armed70 and rsi[t] >= 70.0:
                armed70 = True
            elif armed70 and rsi[t] <= 60.0 and n_entries < 4:
                avg_entry = (avg_entry * p + c[t] * 0.25) / (p + 0.25)
                p += 0.25
                n_entries += 1
                armed70 = False
            profit = c[t] / avg_entry - 1.0
            if profit < 0.0:
                underwater = True
            if not be_done and underwater and profit >= 0.0:
                p *= 0.75
                be_done = True
            if not tp_done and profit >= 0.05:
                p *= 0.5
                tp_done = True
            out[t] = p
        else:
            if c[t] > e50[t] and rsi[t - 1] < 50.0 and rsi[t] >= 50.0:
                p = 0.25
                avg_entry = c[t]
                n_entries = 1
                armed70 = False
                seen60 = rsi[t] >= 60.0
                underwater = False
                be_done = False
                tp_done = False
                out[t] = p
    return default_output(out, df.index)


# ---------------------------------------------------------------------------
# qcb_044: ATR pullback scale-in (up to 4 units), TP 2.5 ATR, EMA50 exit
# ---------------------------------------------------------------------------
def _golden_qcb_044(df: pd.DataFrame) -> Dict[str, Any]:
    close = df["close"].astype(float)
    c = close.to_numpy()
    high = df["high"].astype(float).to_numpy()
    e50 = ema(close, 50).to_numpy()
    atr = wilder_atr(df, 14).to_numpy()
    n = len(df)
    out = np.zeros(n)
    recent_high = np.nan
    units = 0
    avg_entry = 0.0
    last_entry = 0.0
    atr_e = 0.0
    for t in range(50, n):
        # running high since the last upward cross of EMA50
        if c[t] > e50[t]:
            if c[t - 1] <= e50[t - 1] or np.isnan(recent_high):
                recent_high = high[t]
            else:
                recent_high = max(recent_high, high[t])
        if units > 0:
            if c[t] < e50[t] or c[t] >= avg_entry + 2.5 * atr_e or (units == 4 and c[t] <= last_entry - 1.5 * atr_e):
                units = 0
                continue
            if units < 4 and c[t] <= last_entry - 1.5 * atr_e:
                avg_entry = (avg_entry * units + c[t]) / (units + 1)
                last_entry = c[t]
                units += 1
            out[t] = 0.25 * units
        else:
            if c[t] > e50[t] and not np.isnan(recent_high) and c[t] <= recent_high - 1.5 * atr[t]:
                units = 1
                avg_entry = c[t]
                last_entry = c[t]
                atr_e = atr[t]
                out[t] = 0.25
    return default_output(out, df.index)


# ---------------------------------------------------------------------------
# qcb_212: 1.0 / 0.95 volatility rotation with anchor and 30-bar cooldown
# ---------------------------------------------------------------------------
def _golden_qcb_212(df: pd.DataFrame) -> Dict[str, Any]:
    close = df["close"].astype(float)
    c = close.to_numpy()
    high = df["high"].astype(float).to_numpy()
    _, _, upper = bollinger(close, 20, 2.0)
    up = upper.to_numpy()
    atr = wilder_atr(df, 14)
    atr_np = atr.to_numpy()
    atr_sma30 = sma(atr, 30).to_numpy()
    n = len(df)
    out = np.zeros(n)
    reduced = False
    cooldown_until = -1  # last bar index on which a new sell is still forbidden
    anchor_high = 0.0
    sell_price = 0.0
    for t in range(29, n):
        if reduced:
            if c[t] <= anchor_high * 0.95 or c[t] >= sell_price * 1.03:
                reduced = False
                cooldown_until = t + 30
                out[t] = 1.0
            else:
                out[t] = 0.95
        else:
            if t > cooldown_until and high[t] >= up[t] and atr_np[t] > 0.5 * atr_sma30[t]:
                reduced = True
                anchor_high = high[t]
                sell_price = c[t]
                out[t] = 0.95
            else:
                out[t] = 1.0
    return default_output(out, df.index)


# ---------------------------------------------------------------------------
# qcb_175: mean reversion with profit / stop / 5-bar time exit
# ---------------------------------------------------------------------------
def _golden_qcb_175(df: pd.DataFrame) -> Dict[str, Any]:
    close = df["close"].astype(float)
    c = close.to_numpy()
    e20 = ema(close, 20).to_numpy()
    rsi = wilder_rsi(close, 14).to_numpy()
    atr = wilder_atr(df, 14).to_numpy()
    n = len(df)
    out = np.zeros(n)
    holding = False
    entry = 0.0
    atr_e = 0.0
    held = 0
    for t in range(20, n):
        if holding:
            held += 1
            stop_lvl = entry - min(atr_e, 0.02 * entry)
            if c[t] >= entry + 0.5 * atr_e or c[t] >= e20[t] or c[t] <= stop_lvl or held >= 5:
                holding = False
            else:
                out[t] = 1.0
        else:
            day_ret = c[t] / c[t - 1] - 1.0
            if c[t] < e20[t] and rsi[t] < 40.0 and atr[t] / c[t] < 0.04 and day_ret >= -0.01:
                holding = True
                entry = c[t]
                atr_e = atr[t]
                held = 0
                out[t] = 1.0
    return default_output(out, df.index)


# ---------------------------------------------------------------------------
# qcb_027: entry gated by previous exit price
# ---------------------------------------------------------------------------
def _golden_qcb_027(df: pd.DataFrame) -> Dict[str, Any]:
    close = df["close"].astype(float)
    c = close.to_numpy()
    s20 = sma(close, 20).to_numpy()
    n = len(df)
    out = np.zeros(n)
    holding = False
    prev_exit = np.nan
    for t in range(20, n):
        longcond = c[t] > s20[t]
        if holding:
            if not longcond:
                holding = False
                prev_exit = c[t]
            else:
                out[t] = 1.0
        else:
            if longcond and (np.isnan(prev_exit) or c[t - 1] < prev_exit):
                holding = True
                out[t] = 1.0
    return default_output(out, df.index)


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------
QCB_CASES: List[HighFreedomImplCase] = [
    HighFreedomImplCase(
        case_id="qcb_101_atr_trail_macd_exit",
        title="QCB 101: ATR trailing stop with MACD-histogram exit",
        state_complexity="coupled_state",
        source="quantcode-bench:101",
        prompt=_spec(
            "实现 ATR 追踪止损 + MACD 柱退出的趋势策略。指标：SMA(9)、MACD(12,26,9) 柱、ATR(14)。"
            "前 35 根（index 0..34）空仓。"
            "入场（空仓时）：close > SMA9 且 MACD 柱上穿 0（柱[t-1] <= 0 且 柱[t] > 0），满仓 1.0；入场价 = 当根 close；"
            "初始止损 = 入场价 - 2*ATR[t]。"
            "持仓时每根 bar：先用上一根结束时的止损位判断，若 close <= 止损位，或 MACD 柱下穿 0（柱[t-1] >= 0 且 柱[t] < 0），则平仓；"
            "否则更新止损位 = max(旧止损位, high[t] - 2*ATR[t])，止损位只升不降。"
        ),
        original_prompt=(
            "Create a trading strategy with a trailing stop for holding positions (\"runners\"). Indicators: ATR (default 14) "
            "for the baseline stop; MACD (fast 12, slow 26, signal 9) for momentum; SMA (default 9) for trend. "
            "Long entry: bar close above the 9-day SMA and the MACD histogram crosses the zero line from below. "
            "Exit: (1) main trailing stop, initially 2x ATR below entry, afterwards moving up lagging the current bar's high by 2x ATR; "
            "(2) if the MACD histogram crosses the zero line from above to below, close the entire position."
        ),
        golden_fn=_golden_qcb_101,
    ),
    HighFreedomImplCase(
        case_id="qcb_284_bb_breakout_ratchet_trail",
        title="QCB 284: Bollinger breakout with 1% ratcheting trailing stop",
        state_complexity="coupled_state",
        source="quantcode-bench:284",
        prompt=_spec(
            "实现布林带突破 + 1% 棘轮追踪止损策略。指标：布林带(20, 2) 上轨、RSI(14)。前 20 根（index 0..19）空仓。"
            "入场（空仓时）：close 上穿上轨（close[t-1] <= 上轨[t-1] 且 close[t] > 上轨[t]）且 RSI[t] > 55，满仓 1.0；"
            "入场当根：highest = high[t]，止损位 = close[t] * 0.99。"
            "持仓时每根 bar：先更新 highest = max(highest, high[t])，止损位 = max(旧止损位, highest * 0.99)；"
            "然后若 close[t] <= 止损位 则平仓。止损位只升不降。"
        ),
        original_prompt=(
            "Momentum strategy with entry on a breakout above the upper Bollinger Band and a 1% trailing stop. "
            "1) Enter long when the closing price crosses above the upper Bollinger Band (period 20, deviation 2) and RSI(14) > 55. "
            "2) On the entry day set trailing_stop_loss = entry_price * (1 - 1%). 3) While open, each bar update "
            "highest_price = max(previous_highest_price, high) and trailing_stop_loss = max(previous, highest_price * (1 - 1%)); "
            "the stop may only rise. 4) Exit on the day when close <= trailing_stop_loss. 5) Position size: 1 share."
        ),
        golden_fn=_golden_qcb_284,
    ),
    HighFreedomImplCase(
        case_id="qcb_226_sma252_atrpct_gate",
        title="QCB 226: SMA252 regime with ATR% gate, 2% stop and 5% take-profit",
        state_complexity="coupled_state",
        source="quantcode-bench:226",
        prompt=_spec(
            "实现年线动量策略。指标：SMA(252)；ATR% = SMA(ATR(14), 252) / close。SMA(252) 未定义期间（index 0..250）空仓。"
            "入场（空仓时）：close > SMA252 且 ATR% <= 0.05，满仓 1.0，入场价 = 当根 close。"
            "持仓时：close < SMA252，或 close <= 入场价 * 0.98（止损），或 close >= 入场价 * 1.05（止盈），任一满足则平仓。"
            "ATR% > 0.05 期间不开新仓，但不影响已持有的仓位。"
        ),
        original_prompt=(
            "Impulse Strategy 12-1 on daily data: use SMA(252) of the close as a proxy for annual momentum. "
            "Indicators: SMA(252); ATR% = SMA(ATR(14), 252) / Close. Entry (long): Close > SMA(252) and ATR% <= 0.05. "
            "Exit: Close < SMA(252) or the stop-loss/take-profit is triggered. Position fixed at 10% of capital; "
            "stop-loss 2% from entry; take-profit 5% from entry. If ATR% > 0.05, do not open new positions."
        ),
        golden_fn=_golden_qcb_226,
    ),
    HighFreedomImplCase(
        case_id="qcb_064_ema_ribbon_order",
        title="QCB 064: EMA ribbon ordering with 2% stop and 4% take-profit",
        state_complexity="coupled_state",
        source="quantcode-bench:64",
        prompt=_spec(
            "实现 EMA 均线束策略。指标：EMA(8)、EMA(13)、EMA(21)、EMA(34)、EMA(55)。前 55 根（index 0..54）空仓。"
            "定义『完全有序』= EMA8 > EMA13 > EMA21 > EMA34 > EMA55（全部严格）。"
            "入场（空仓时）：当根完全有序且上一根不完全有序，满仓 1.0，入场价 = 当根 close。"
            "持仓时：min(EMA8, EMA13, EMA21) <= max(EMA34, EMA55)（快线组不再整体位于慢线组之上），"
            "或 close <= 入场价 * 0.98，或 close >= 入场价 * 1.04，任一满足则平仓。"
        ),
        original_prompt=(
            "EMA Ribbon strategy with five EMAs of periods 8, 13, 21, 34 and 55. Open a long position when all fast EMAs "
            "(8, 13, 21) cross the slow EMAs (34, 55) from below and form an ordered sequence (fast above slow). "
            "Close when the opposite crossover occurs and the EMAs stop being ordered. Fixed stop-loss 2% below entry "
            "and take-profit 4% above entry. Position size 2% of capital per trade."
        ),
        golden_fn=_golden_qcb_064,
    ),
    HighFreedomImplCase(
        case_id="qcb_317_month_day1_to_day5",
        title="QCB 317: long from first to fifth trading day of the month with SMA20 filter",
        state_complexity="coupled_state",
        source="quantcode-bench:317",
        prompt=_spec(
            "实现月初持有策略。指标：SMA(20)。index 0..18 空仓。df 的索引为交易日 DatetimeIndex。"
            "定义 k[t] 为 bar t 在其所属日历月内的交易日序号（该月第一根为 1，逐根加 1）。"
            "入场：空仓且 k[t] == 1 且 close > SMA20，满仓 1.0，入场价 = 当根 close。若该月第一根不满足条件，当月不再入场。"
            "持仓时：k[t] >= 5（当月第 5 个交易日收盘）或 close <= 入场价 * 0.98，任一满足则平仓。"
        ),
        original_prompt=(
            "Trading strategy that goes long with all capital on the first trading day of each month, only if the closing "
            "price is above the 20-day SMA. The position is held until the fifth trading day of the month, then closed. "
            "Includes a stop-loss set at 2% below entry price."
        ),
        golden_fn=_golden_qcb_317,
    ),
    HighFreedomImplCase(
        case_id="qcb_089_profit_dependent_exit",
        title="QCB 089: SMA cross entry with profit-dependent exit rule",
        state_complexity="coupled_state",
        source="quantcode-bench:89",
        prompt=_spec(
            "实现盈亏依赖退出策略。指标：SMA(20)、SMA(50)。前 50 根（index 0..49）空仓。"
            "入场（空仓时）：SMA20 上穿 SMA50（SMA20[t-1] <= SMA50[t-1] 且 SMA20[t] > SMA50[t]），满仓 1.0，入场价 = 当根 close，"
            "peak = 当根 close。"
            "持仓时每根 bar：先更新 peak = max(peak, close[t])，再计算浮盈 = close[t] / 入场价 - 1。"
            "若浮盈 < 10%：当 SMA20[t] <= SMA50[t] 时平仓；若浮盈 >= 10%：当 close[t] <= peak * 0.95 时平仓。"
            "两条规则按当根浮盈所在区间二选一，每根重新判定。"
        ),
        original_prompt=(
            "Strategy with dynamic exit rules based on the current profit of the trade. Entry: long when SMA(20) crosses "
            "SMA(50) from below. After entering, track the percentage return. If the current trade profit is below 10%, "
            "close using sell_below10(); if above 10%, close using sell_abv10(). Position 100% of capital; one position "
            "at a time; no fixed stop-loss or take-profit."
        ),
        golden_fn=_golden_qcb_089,
    ),
    HighFreedomImplCase(
        case_id="qcb_227_universal_portfolio",
        title="QCB 227: universal portfolio weight (wealth-weighted constant fractions)",
        state_complexity="stateless_continuous",
        source="quantcode-bench:227",
        prompt=_spec(
            "实现单资产通用组合（Universal Portfolio）仓位。日收益 r[t] = close[t] / close[t-1] - 1。"
            "index 0..251 空仓（不足 252 个日收益时不交易）。"
            "从 index 252 起，每根 bar 取最近 252 个日收益（r[t-251..t]，含当日），对 w ∈ {0.0, 0.1, ..., 1.0} 共 11 个常数比例，"
            "计算 wealth(w) = Π(1 + w * r)；目标仓位 signals[t] = Σ_w [w * wealth(w)] / Σ_w [wealth(w)]。仓位连续取值，每日重算。"
        ),
        original_prompt=(
            "Universal portfolio for a single asset as an allocation between the asset and cash. Daily rebalancing based on "
            "the wealth-weighted average across all constant fractions w in {0.0, 0.1, ..., 1.0} over a 252-day window. "
            "(1) daily returns r_t for the last 252 days; (2) for each w simulate wealth(w) = prod(1 + w*r_t); "
            "(3) w_next = sum_w[w*wealth(w)] / sum_w[wealth(w)]; (4) set the position to w_next, rebalance daily. "
            "If there is insufficient data (<252 days), do not trade."
        ),
        golden_fn=_golden_qcb_227,
    ),
    HighFreedomImplCase(
        case_id="qcb_013_rsi_pyramiding_partial_exits",
        title="QCB 013: RSI pyramiding with partial exits",
        state_complexity="multi_state",
        source="quantcode-bench:13",
        prompt=_spec(
            "实现趋势金字塔加仓策略。指标：EMA(50)、RSI(14)。前 50 根（index 0..49）空仓。每份仓位 = 0.25，最多 4 份。"
            "首次入场（空仓时）：close > EMA50 且 RSI 上穿 50（RSI[t-1] < 50 且 RSI[t] >= 50），仓位 0.25，平均入场价 = 当根 close，"
            "重置所有状态标记。"
            "持仓时每根 bar 按以下顺序处理："
            "(1) 若 RSI[t] >= 60 则记『曾达 60』；若 close < EMA50，或（曾达 60 且 RSI[t] < 40），全部平仓并结束本根。"
            "(2) 加仓：若未『武装』且 RSI[t] >= 70 则置为武装；否则若已武装且 RSI[t] <= 60 且累计入场份数 < 4，加仓 0.25"
            "（平均入场价按份数加权更新）并解除武装。同一根 bar 只能做其中一件。"
            "(3) 计算浮盈 = close / 平均入场价 - 1（用本根加仓后的平均价）。若浮盈 < 0 记『曾浮亏』。"
            "保本减仓：尚未做过、且曾浮亏、且浮盈 >= 0，则仓位乘 0.75，只做一次。"
            "止盈减仓：尚未做过、且浮盈 >= 5%，则仓位乘 0.5，只做一次。两者同一根可先后触发。"
            "signals[t] 为处理完毕后的仓位。"
        ),
        original_prompt=(
            "Pyramiding in Trend. Indicators: EMA(50) trend filter, RSI(14). Entry: close above EMA(50); first entry when RSI "
            "drops below 50 and rises back above 50. Pyramiding: each time RSI reaches 70 and then drops back to 60, open an "
            "additional position of the same size; maximum 4 positions in total. Partial exits: at break-even across all "
            "positions close 25% of the total volume; at +5% total profit close another 50% of the remaining volume. "
            "Full exit: close everything if RSI drops below 40 after having been above 60, or close falls below EMA(50)."
        ),
        golden_fn=_golden_qcb_013,
    ),
    HighFreedomImplCase(
        case_id="qcb_044_atr_pullback_scale_in",
        title="QCB 044: ATR pullback scale-in up to four units",
        state_complexity="multi_state",
        source="quantcode-bench:44",
        prompt=_spec(
            "实现 ATR 回调分批建仓策略。指标：EMA(50)、ATR(14)。前 50 根（index 0..49）空仓。每份仓位 0.25，最多 4 份。"
            "『近期高点』：当 close 上穿 EMA50（close[t-1] <= EMA50[t-1] 且 close[t] > EMA50[t]）时重置为 high[t]；"
            "之后只要 close > EMA50 就更新为 max(近期高点, high[t])；close <= EMA50 时不更新。平仓不重置近期高点。"
            "首次入场（空仓时）：close > EMA50 且 close <= 近期高点 - 1.5 * ATR[t]，建 1 份；记录平均入场价 = 最近入场价 = close，"
            "并冻结 ATR_e = ATR[t]，之后本轮所有阈值都用 ATR_e。"
            "持仓时每根 bar：(1) 出场：close < EMA50，或 close >= 平均入场价 + 2.5 * ATR_e（止盈），"
            "或（已满 4 份且 close <= 最近入场价 - 1.5 * ATR_e）（止损），任一满足全部平仓。"
            "(2) 否则若份数 < 4 且 close <= 最近入场价 - 1.5 * ATR_e，加 1 份，更新平均入场价（按份数加权）和最近入场价。"
            "signals[t] = 0.25 * 份数。"
        ),
        original_prompt=(
            "Trend-following ETF strategy. Indicators: ATR(14) and EMA(50). Buy when price is above EMA(50) and makes a pullback "
            "of 1.5 ATR from a recent high. If after entering the price keeps falling and reaches a new 1.5 ATR pullback level from "
            "the next high, open an additional position (scaling in); maximum 4 consecutive entries. Exit the whole position at a "
            "take-profit of 2.5 ATR from entry or if price closes below EMA(50). Each position 1% of capital; stop-loss 1.5 ATR "
            "below each entry."
        ),
        golden_fn=_golden_qcb_044,
    ),
    HighFreedomImplCase(
        case_id="qcb_212_vol_rotation_095",
        title="QCB 212: 1.0 / 0.95 volatility rotation with anchor high and cooldown",
        state_complexity="coupled_state",
        source="quantcode-bench:212",
        prompt=_spec(
            "实现单资产波动率轮换策略。基础仓位 1.0。指标：布林带(20, 2) 上轨、ATR(14)、SMA(ATR(14), 30)。"
            "index 0..28 空仓；从 index 29 起默认持有 1.0。"
            "减仓（当前为 1.0 且不在冷却期）：high[t] >= 上轨[t] 且 ATR[t] > 0.5 * SMA(ATR,30)[t]，仓位改为 0.95；"
            "记录锚定高点 = high[t]，卖出价 = close[t]。"
            "买回（当前为 0.95）：close[t] <= 锚定高点 * 0.95，或 close[t] >= 卖出价 * 1.03，任一满足则仓位回到 1.0，周期结束，"
            "随后 30 个交易日（t+1..t+30）为冷却期，不得再次减仓。"
            "仓位只取 0、0.95、1.0 三个值。"
        ),
        original_prompt=(
            "Single-Asset Volatility Rotation: maintain a base long position and rotate 5% of the portfolio at extremes. "
            "Indicators: Bollinger Bands(20, 2), ATR(14) and SMA(ATR, 30). Volatility filter: ATR(14) > 0.5 * SMA(ATR(14), 30). "
            "Tactical sell of 5% when the close touches/breaks the upper band and the filter is met, while the cooldown is observed; "
            "record anchor high = high of the entry bar and the sell price. Buy back 5% if price falls 5% from the anchor high, or "
            "rises 3% from the sell price before that. After completing the cycle, cooldown of 30 trading days."
        ),
        golden_fn=_golden_qcb_212,
    ),
    HighFreedomImplCase(
        case_id="qcb_175_mean_reversion_time_stop",
        title="QCB 175: mean reversion with profit, stop and 5-bar time exit",
        state_complexity="coupled_state",
        source="quantcode-bench:175",
        prompt=_spec(
            "实现均值回归做多策略。指标：EMA(20)、RSI(14)、ATR(14)。前 20 根（index 0..19）空仓。"
            "入场（空仓时）同时满足：close < EMA20；RSI < 40；ATR[t] / close[t] < 0.04；close[t] / close[t-1] - 1 >= -1%。"
            "满仓 1.0，入场价 = 当根 close，冻结 ATR_e = ATR[t]，持有根数 = 0。"
            "持仓时每根 bar：持有根数加 1；若 close >= 入场价 + 0.5 * ATR_e，或 close >= EMA20[t]，"
            "或 close <= 入场价 - min(ATR_e, 0.02 * 入场价)，或持有根数 >= 5（入场后第 5 根收盘），任一满足则平仓。"
        ),
        original_prompt=(
            "Mean-reversion long as a proxy for selling a cash-secured put. Indicators: EMA(20), RSI(14), ATR(14). Entry: "
            "(1) Close < EMA(20); (2) RSI(14) < 40; (3) ATR(14)/Close < 0.04; (4) the current close is not lower than the previous "
            "close by more than 1%. Enter at market on the next bar, one position at a time. Exit (profit): Close >= Entry + 0.5*ATR "
            "or Close >= EMA(20). Exit (stop): Close <= Entry - min(1*ATR, 0.02*Entry). Time stop: close after 5 trading days. "
            "Position size so that risk to the stop is at most 1% of capital."
        ),
        golden_fn=_golden_qcb_175,
    ),
    HighFreedomImplCase(
        case_id="qcb_027_prev_exit_price_gate",
        title="QCB 027: long entry gated by the previous exit price",
        state_complexity="coupled_state",
        source="quantcode-bench:27",
        eval_ticker="TLT",  # SPY's long bull run leaves only 4 trades; TLT gives 56
        prompt=_spec(
            "实现带退出价记忆的做多策略。主信号 longcondition[t] = close[t] > SMA(20)[t]。前 20 根（index 0..19）空仓。"
            "入场（空仓时）：longcondition[t] 为真，且（尚无历史退出价，或 close[t-1] < 上一次退出价），满仓 1.0。"
            "持仓时：longcondition[t] 为假则平仓，并记录退出价 = close[t]。"
        ),
        original_prompt=(
            "Long-only strategy where entry depends on the main signal and on a comparison of the current price with the price of "
            "the previous exit. Entry: the logical indicator `longcondition` is true, and close[1] (previous bar's close) is below "
            "previousExitPrice. Exit: when `longcondition` becomes false. Track previousExitPrice; on the first trade, when there is "
            "no previous exit, the comparison must not block entry. Position size 100% of capital."
        ),
        golden_fn=_golden_qcb_027,
    ),
]


# ===========================================================================
# Second batch: the "usable as-is" fillers from the screen (stateless / simple state).
# qcb_163 (EMA10 +/- 3*ATR10 channel) was dropped: 0-4 trades on every ticker over 1500 bars.
# ===========================================================================
def _golden_qcb_004(df: pd.DataFrame) -> Dict[str, Any]:
    close = df["close"].astype(float)
    mom = (close / close.shift(252) * 100.0).to_numpy()
    n = len(df)
    out = np.zeros(n)
    holding = False
    for t in range(252, n):
        if mom[t] > 100.0:
            holding = True
        elif mom[t] < 100.0:
            holding = False
        out[t] = 1.0 if holding else 0.0
    return default_output(out, df.index)


def _golden_qcb_039(df: pd.DataFrame) -> Dict[str, Any]:
    close = df["close"].astype(float)
    s10 = sma(close, 10).to_numpy()
    s30 = sma(close, 30).to_numpy()
    n = len(df)
    out = np.zeros(n)
    holding = False
    for t in range(30, n):
        if not holding and s10[t - 1] <= s30[t - 1] and s10[t] > s30[t]:
            holding = True
        elif holding and s10[t - 1] >= s30[t - 1] and s10[t] < s30[t]:
            holding = False
        out[t] = 1.0 if holding else 0.0
    return default_output(out, df.index)


def _golden_qcb_052(df: pd.DataFrame) -> Dict[str, Any]:
    close = df["close"].astype(float)
    line, sig, _ = macd(close)
    l, s = line.to_numpy(), sig.to_numpy()
    n = len(df)
    out = np.zeros(n)
    holding = False
    for t in range(35, n):
        if not holding and l[t - 1] <= s[t - 1] and l[t] > s[t]:
            holding = True
        elif holding and l[t - 1] >= s[t - 1] and l[t] < s[t]:
            holding = False
        out[t] = 1.0 if holding else 0.0
    return default_output(out, df.index)


def _golden_qcb_110(df: pd.DataFrame) -> Dict[str, Any]:
    close = df["close"].astype(float)
    rsi = wilder_rsi(close, 14).to_numpy()
    n = len(df)
    out = np.zeros(n)
    holding = False
    for t in range(15, n):
        if not holding and rsi[t - 1] <= 30.0 and rsi[t] > 30.0:
            holding = True
        elif holding and rsi[t - 1] >= 70.0 and rsi[t] < 70.0:
            holding = False
        out[t] = 1.0 if holding else 0.0
    return default_output(out, df.index)


def _dema(close: pd.Series, n: int) -> pd.Series:
    e1 = ema(close, n)
    return 2.0 * e1 - ema(e1, n)


def _supertrend_dir(df: pd.DataFrame, n: int = 12, mult: float = 3.0) -> np.ndarray:
    """Returns +1 / -1 per bar. Standard ratcheting bands; direction flips when close breaks
    the previous bar's final band."""
    high = df["high"].astype(float).to_numpy()
    low = df["low"].astype(float).to_numpy()
    close = df["close"].astype(float).to_numpy()
    atr = wilder_atr(df, n).to_numpy()
    hl2 = (high + low) / 2.0
    basic_up = hl2 + mult * atr
    basic_lo = hl2 - mult * atr
    N = len(df)
    fin_up = basic_up.copy()
    fin_lo = basic_lo.copy()
    direction = np.ones(N, dtype=int)
    for t in range(1, N):
        fin_up[t] = basic_up[t] if (basic_up[t] < fin_up[t - 1] or close[t - 1] > fin_up[t - 1]) else fin_up[t - 1]
        fin_lo[t] = basic_lo[t] if (basic_lo[t] > fin_lo[t - 1] or close[t - 1] < fin_lo[t - 1]) else fin_lo[t - 1]
        if direction[t - 1] == -1 and close[t] > fin_up[t]:
            direction[t] = 1
        elif direction[t - 1] == 1 and close[t] < fin_lo[t]:
            direction[t] = -1
        else:
            direction[t] = direction[t - 1]
    return direction


def _golden_qcb_126(df: pd.DataFrame) -> Dict[str, Any]:
    close = df["close"].astype(float)
    c = close.to_numpy()
    dema = _dema(close, 200).to_numpy()
    d = _supertrend_dir(df, 12, 3.0)
    n = len(df)
    out = np.zeros(n)
    holding = False
    for t in range(400, n):
        if holding:
            if d[t - 1] == 1 and d[t] == -1:
                holding = False
        else:
            if d[t - 1] == -1 and d[t] == 1 and c[t] > dema[t]:
                holding = True
        out[t] = 1.0 if holding else 0.0
    return default_output(out, df.index)


def _golden_qcb_163(df: pd.DataFrame) -> Dict[str, Any]:
    close = df["close"].astype(float)
    c = close.to_numpy()
    dema = _dema(close, 200).to_numpy()
    e10 = ema(close, 10)
    atr10 = wilder_atr(df, 10)
    up = (e10 + 3.0 * atr10).to_numpy()
    lo = (e10 - 3.0 * atr10).to_numpy()
    n = len(df)
    out = np.zeros(n)
    holding = False
    for t in range(400, n):
        if holding:
            if (c[t - 1] >= lo[t - 1] and c[t] < lo[t]) or c[t] < dema[t]:
                holding = False
        else:
            if c[t] > dema[t] and c[t - 1] <= up[t - 1] and c[t] > up[t]:
                holding = True
        out[t] = 1.0 if holding else 0.0
    return default_output(out, df.index)


def _golden_qcb_207(df: pd.DataFrame) -> Dict[str, Any]:
    close = df["close"].astype(float)
    c = close.to_numpy()
    _, _, up = bollinger(close, 10, 0.674)
    u = up.to_numpy()
    n = len(df)
    out = np.zeros(n)
    holding = False
    for t in range(9, n):
        if c[t] > u[t]:
            holding = True
        elif c[t] < u[t]:
            holding = False
        out[t] = 1.0 if holding else 0.0
    return default_output(out, df.index)


def _golden_qcb_353(df: pd.DataFrame) -> Dict[str, Any]:
    close = df["close"].astype(float)
    c = close.to_numpy()
    tp = (df["high"].astype(float) + df["low"].astype(float) + close) / 3.0
    vol = df["volume"].astype(float)
    vwap = ((tp * vol).rolling(20, min_periods=20).sum() / vol.rolling(20, min_periods=20).sum()).to_numpy()
    line, sig, _ = macd(close)
    l, s = line.to_numpy(), sig.to_numpy()
    rsi = wilder_rsi(close, 14).to_numpy()
    n = len(df)
    out = np.zeros(n)
    holding = False
    for t in range(35, n):
        cross_up = l[t - 1] <= s[t - 1] and l[t] > s[t]
        cross_dn = l[t - 1] >= s[t - 1] and l[t] < s[t]
        if holding:
            if c[t] < vwap[t] and cross_dn and rsi[t] > 30.0:
                holding = False
        else:
            if c[t] > vwap[t] and cross_up and rsi[t] < 70.0:
                holding = True
        out[t] = 1.0 if holding else 0.0
    return default_output(out, df.index)


def _golden_qcb_092(df: pd.DataFrame) -> Dict[str, Any]:
    close = df["close"].astype(float)
    c = close.to_numpy()
    _, _, up = bollinger(close, 20, 2.0)
    u = up.to_numpy()
    n = len(df)
    out = np.zeros(n)
    holding = False
    entry = 0.0
    for t in range(19, n):
        if holding:
            if c[t] >= entry * 1.06 or c[t] <= entry * 0.97:
                holding = False
            else:
                out[t] = 1.0
        else:
            if c[t] > u[t]:
                holding = True
                entry = c[t]
                out[t] = 1.0
    return default_output(out, df.index)


def _golden_qcb_243(df: pd.DataFrame) -> Dict[str, Any]:
    close = df["close"].astype(float)
    c = close.to_numpy()
    _, mid, up = bollinger(close, 20, 2.0)
    u = up.to_numpy()
    m = mid.to_numpy()
    n = len(df)
    out = np.zeros(n)
    holding = False
    entry = 0.0
    for t in range(22, n):
        if holding:
            if c[t] < u[t] or c[t] >= entry * 1.02:
                holding = False
            else:
                out[t] = 1.0
        else:
            prior_below = all(c[t - k] <= u[t - k] for k in (1, 2, 3))
            if c[t] > u[t] and prior_below and m[t] > m[t - 1]:
                holding = True
                entry = c[t]
                out[t] = 1.0
    return default_output(out, df.index)


def _golden_qcb_256(df: pd.DataFrame) -> Dict[str, Any]:
    close = df["close"].astype(float)
    c = close.to_numpy()
    s50 = sma(close, 50).to_numpy()
    s200 = sma(close, 200).to_numpy()
    atr = wilder_atr(df, 14).to_numpy()
    n = len(df)
    out = np.zeros(n)
    holding = False
    tp_level = 0.0
    for t in range(199, n):
        if holding:
            if c[t] >= tp_level or c[t] < s50[t]:
                holding = False
            else:
                out[t] = 1.0
        else:
            if c[t] > s50[t] and c[t] > s200[t] and atr[t] / c[t] <= 0.015:
                holding = True
                tp_level = min(c[t] * 1.05, c[t] + 2.5 * atr[t])
                out[t] = 1.0
    return default_output(out, df.index)


QCB_CASES_BATCH2: List[HighFreedomImplCase] = [
    HighFreedomImplCase(
        case_id="qcb_004_abs_momentum_252",
        title="QCB 004: absolute momentum oscillator (252) with hysteresis at 100",
        state_complexity="simple_state",
        source="quantcode-bench:4",
        eval_ticker="IWM",  # 15 trades on SPY, 45 on IWM
        prompt=_spec(
            "实现绝对动量策略。动量 = close[t] / close[t-252] * 100。index 0..251 空仓。"
            "动量 > 100 时持有满仓 1.0；动量 < 100 时空仓；动量恰好等于 100 时保持上一根的仓位。"
        ),
        original_prompt=(
            "Absolute Momentum Strategy. Indicator: Momentum Oscillator on closing prices, period 252 days, threshold 100. "
            "Buy (fully invested) when the oscillator becomes strictly greater than 100. Sell (move to cash) when it becomes "
            "strictly less than 100. Position size 95% of capital."
        ),
        golden_fn=_golden_qcb_004,
    ),
    HighFreedomImplCase(
        case_id="qcb_039_sma10_30_cross",
        title="QCB 039: SMA(10)/SMA(30) crossover",
        state_complexity="stateless",
        source="quantcode-bench:39",
        prompt=_spec(
            "实现双均线交叉策略。指标 SMA(10)、SMA(30)。index 0..29 空仓。"
            "空仓时若 SMA10 上穿 SMA30（SMA10[t-1] <= SMA30[t-1] 且 SMA10[t] > SMA30[t]）则满仓 1.0；"
            "持仓时若 SMA10 下穿 SMA30（SMA10[t-1] >= SMA30[t-1] 且 SMA10[t] < SMA30[t]）则空仓。其他情况保持。"
        ),
        original_prompt=(
            "Trading strategy based on the crossover of two simple moving averages. Fast SMA period 10, slow SMA period 30. "
            "Open a long position when the fast SMA crosses the slow SMA from below to above. Close the position when the fast "
            "SMA crosses the slow SMA from above to below. Allow the periods to be adjusted via strategy parameters."
        ),
        golden_fn=_golden_qcb_039,
    ),
    HighFreedomImplCase(
        case_id="qcb_052_macd_line_signal_cross",
        title="QCB 052: MACD line / signal crossover",
        state_complexity="stateless",
        source="quantcode-bench:52",
        prompt=_spec(
            "实现 MACD 金叉死叉策略。MACD(12,26,9)：DIF = EMA12 - EMA26，DEA = EMA(DIF, 9)。index 0..34 空仓。"
            "空仓时若 DIF 上穿 DEA（DIF[t-1] <= DEA[t-1] 且 DIF[t] > DEA[t]）则满仓 1.0；"
            "持仓时若 DIF 下穿 DEA（DIF[t-1] >= DEA[t-1] 且 DIF[t] < DEA[t]）则空仓。其他情况保持。"
        ),
        original_prompt=(
            "Trading strategy based on the crossover of the MACD indicator lines, standard parameters: fast EMA 12, slow EMA 26, "
            "signal 9. Open a long position when the MACD line crosses the signal line from below upward. Close the position when "
            "the MACD line crosses the signal line from above downward."
        ),
        golden_fn=_golden_qcb_052,
    ),
    HighFreedomImplCase(
        case_id="qcb_110_rsi_30_up_70_down",
        title="QCB 110: RSI(14) enter on cross above 30, exit on cross below 70",
        state_complexity="simple_state",
        source="quantcode-bench:110",
        prompt=_spec(
            "实现 RSI 反转策略。指标 RSI(14)。index 0..14 空仓。"
            "空仓时若 RSI 上穿 30（RSI[t-1] <= 30 且 RSI[t] > 30）则满仓 1.0；"
            "持仓时若 RSI 下穿 70（RSI[t-1] >= 70 且 RSI[t] < 70）则空仓；RSI 一直未达到 70 则持续持有。"
        ),
        original_prompt=(
            "Trading strategy based on RSI with period 14, trading reversals from extreme values. Open a long position when RSI "
            "crosses the 30 level from below upward. Close the position when RSI crosses the 70 level from above downward."
        ),
        golden_fn=_golden_qcb_110,
    ),
    HighFreedomImplCase(
        case_id="qcb_126_supertrend_dema200",
        title="QCB 126: SuperTrend(12, 3) flips with DEMA(200) filter",
        state_complexity="simple_state",
        source="quantcode-bench:126",
        prompt=_spec(
            "实现 SuperTrend + DEMA 策略。DEMA(200) = 2*EMA(200) - EMA(EMA(200), 200)。"
            "SuperTrend(12, 3)：hl2 = (high+low)/2；基础上轨 = hl2 + 3*ATR(12)，基础下轨 = hl2 - 3*ATR(12)；"
            "最终上轨[t] = 基础上轨[t] 若（基础上轨[t] < 最终上轨[t-1] 或 close[t-1] > 最终上轨[t-1]）否则 = 最终上轨[t-1]；"
            "最终下轨[t] = 基础下轨[t] 若（基础下轨[t] > 最终下轨[t-1] 或 close[t-1] < 最终下轨[t-1]）否则 = 最终下轨[t-1]；"
            "方向初始为上(+1)；若上一根为下且 close[t] > 最终上轨[t] 则转上；若上一根为上且 close[t] < 最终下轨[t] 则转下；否则沿用。"
            "index 0..399 空仓。空仓时若方向由下转上且 close > DEMA200 则满仓 1.0；持仓时若方向由上转下则空仓。"
        ),
        original_prompt=(
            "Strategy combining DEMA and SuperTrend. DEMA period 200 on closes. SuperTrend with ATR period 12 and multiplier 3.0. "
            "Open a long position when the current close is above DEMA(200) and SuperTrend generates a buy signal (direction changes "
            "from downward to upward). Close the long when SuperTrend generates a sell signal (direction changes from upward to downward)."
        ),
        golden_fn=_golden_qcb_126,
    ),
    HighFreedomImplCase(
        case_id="qcb_207_bb10_0674_breakout",
        title="QCB 207: Bollinger(10, 0.674) upper-band breakout",
        state_complexity="stateless",
        source="quantcode-bench:207",
        prompt=_spec(
            "实现布林上轨突破策略。布林带(10, 0.674) 上轨。index 0..8 空仓。"
            "close > 上轨 时持有满仓 1.0；close < 上轨 时空仓；close 恰好等于上轨时保持上一根仓位。"
        ),
        original_prompt=(
            "Breakout strategy using the upper Bollinger Band as an approximation of the 75th percentile: Bollinger Bands(period=10, "
            "devfactor=0.674) on closes; open a long position when the close crosses above the upper band; exit when the close falls "
            "below the upper band. Position size 100% of capital, no stops."
        ),
        golden_fn=_golden_qcb_207,
    ),
    HighFreedomImplCase(
        case_id="qcb_353_vwap20_macd_rsi",
        title="QCB 353: rolling VWAP(20) + MACD cross + RSI gates",
        state_complexity="simple_state",
        source="quantcode-bench:353",
        prompt=_spec(
            "实现 VWAP + MACD + RSI 策略。VWAP(20) = Σ(典型价*volume, 20) / Σ(volume, 20)，典型价 = (high+low+close)/3。"
            "MACD(12,26,9) 的 DIF 与 DEA；RSI(14)。index 0..34 空仓。"
            "空仓时若 close > VWAP 且 DIF 上穿 DEA（DIF[t-1] <= DEA[t-1] 且 DIF[t] > DEA[t]）且 RSI < 70，满仓 1.0。"
            "持仓时若 close < VWAP 且 DIF 下穿 DEA（DIF[t-1] >= DEA[t-1] 且 DIF[t] < DEA[t]）且 RSI > 30，三者同时满足才空仓。"
        ),
        original_prompt=(
            "PulseWave strategy. Indicators: MACD(12, 26, 9) and RSI(14). Rolling VWAP over N=20 bars: VWAP = Sum(typical_price * "
            "volume, N) / Sum(volume, N), typical_price = (high+low+close)/3. Entry (long): close > VWAP AND MACD line crosses above "
            "signal AND RSI < 70. Exit: close < VWAP AND MACD line crosses below signal AND RSI > 30. No stop loss or take profit; "
            "single position at a time."
        ),
        golden_fn=_golden_qcb_353,
    ),
    HighFreedomImplCase(
        case_id="qcb_092_bb_breakout_tp6_sl3",
        title="QCB 092: close above upper Bollinger, +6% take-profit / -3% stop",
        state_complexity="coupled_state",
        source="quantcode-bench:92",
        prompt=_spec(
            "实现布林突破 + 固定止盈止损策略。布林带(20, 2) 上轨。index 0..18 空仓。"
            "空仓时若 close > 上轨 则满仓 1.0，入场价 = 当根 close。"
            "持仓时若 close >= 入场价 * 1.06 或 close <= 入场价 * 0.97 则空仓。平仓后下一根若仍 close > 上轨 可立即再入场。"
        ),
        original_prompt=(
            "Bollinger Bands Breakout for Bitcoin: open a long position when the current candle's close is above the upper Bollinger "
            "Band (20, 2), assuming trend continuation. Close the position on +6% profit from entry (take-profit) or -3% loss from entry "
            "(stop-loss). Position size is not fixed."
        ),
        golden_fn=_golden_qcb_092,
    ),
    HighFreedomImplCase(
        case_id="qcb_243_bb_breakout_after_3_below",
        title="QCB 243: Bollinger breakout after three closes at/below the band, rising SMA20",
        state_complexity="coupled_state",
        source="quantcode-bench:243",
        prompt=_spec(
            "实现确认式布林突破策略。布林带(20, 2) 上轨与中轨 SMA(20)。index 0..21 空仓。"
            "空仓时若 close[t] > 上轨[t]，且 close[t-1]、close[t-2]、close[t-3] 各自 <= 对应 bar 的上轨，且 SMA20[t] > SMA20[t-1]，满仓 1.0，"
            "入场价 = 当根 close。"
            "持仓时若 close < 上轨（假突破）或 close >= 入场价 * 1.02（止盈）则空仓。"
        ),
        original_prompt=(
            "Breakout strategy on D1 with a proxy trend line via Bollinger Bands: buy on a breakout and close above the upper Bollinger "
            "Band (20, 2), if the last 3 candles closed on/below the upper band and SMA(20) is rising (SMA(20) > SMA(20)[1]). Stop-loss: "
            "close the position on a close below the upper band (false breakout). Take-profit: +2% from entry. Position size so that "
            "risk to the stop is at most 1% of capital."
        ),
        golden_fn=_golden_qcb_243,
    ),
    HighFreedomImplCase(
        case_id="qcb_256_sma50_200_atr_target",
        title="QCB 256: SMA50/200 trend with ATR% filter and capped ATR take-profit",
        state_complexity="coupled_state",
        source="quantcode-bench:256",
        prompt=_spec(
            "实现趋势 + 低波动入场、ATR 目标止盈策略。指标 SMA(50)、SMA(200)、ATR(14)。index 0..198 空仓。"
            "空仓时若 close > SMA50 且 close > SMA200 且 ATR[t] / close[t] <= 0.015，满仓 1.0；"
            "入场时记录目标价 TP = min(close * 1.05, close + 2.5 * ATR[t])（ATR 取入场当根，之后不变）。"
            "持仓时若 close >= TP 或 close < SMA50 则空仓。忽略原文的组合回撤熔断。"
        ),
        original_prompt=(
            "Managed Income with Capital Protection (ATR targets instead of covered calls). Indicators: SMA(50), SMA(200), ATR(14). "
            "Entry: no position, close above SMA(50) and SMA(200), ATR(14)/Close <= 0.015. Position size at most 5% of equity. "
            "Take-profit TP = min(Entry*1.05, Entry + 2.5*ATR(14) at entry). Stop: close the position if Close falls below SMA(50). "
            "If equity drawdown >= 10% from the starting level, close all and stop trading. Re-entries allowed when conditions are met again."
        ),
        golden_fn=_golden_qcb_256,
    ),
]

QCB_CASES_ALL: List[HighFreedomImplCase] = QCB_CASES + QCB_CASES_BATCH2
