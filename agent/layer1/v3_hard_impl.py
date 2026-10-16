from __future__ import annotations

from typing import Any, Dict, List
import numpy as np
import pandas as pd

from agent.layer1.v2_hard import (
    HighFreedomImplCase,
    HighFreedomEvalResult,
    evaluate_high_freedom_case,
)

evaluate_v3_hard_impl_case = evaluate_high_freedom_case



# =============================================================================
# Zone 1: Silent Formula Substitution Traps (Wilder ATR / Zero-Mean Realized Vol)
# =============================================================================

def _golden_v3_01_wilder_atr_chandelier(df: pd.DataFrame) -> Dict[str, Any]:
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    prev_close = df["close"].shift(1).to_numpy(dtype=float)

    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)),
    )
    tr[0] = high[0] - low[0]
    atr14 = pd.Series(tr, index=df.index).ewm(alpha=1.0 / 14.0, adjust=False).mean().to_numpy()
    prior_hi20 = df["high"].rolling(20, min_periods=20).max().shift(1).to_numpy()

    out = np.zeros(len(df), dtype=float)
    state = 0
    peak_high = 0.0
    for i in range(len(df)):
        if i < 20 or np.isnan(prior_hi20[i]):
            continue
        if state == 0:
            if close[i] > prior_hi20[i]:
                state = 1
                peak_high = high[i]
        else:
            peak_high = max(peak_high, high[i])
            if close[i] < peak_high - 3.0 * atr14[i]:
                state = 0
                peak_high = 0.0
        out[i] = float(state)
    return {"signals": pd.Series(out, index=df.index), "slippage_rate": 0.0005, "commission_rate": 0.0005, "rebalance_tolerance": 0.05}


def _golden_v3_02_compounded_drawdown_gate(df: pd.DataFrame) -> Dict[str, Any]:
    sma40 = df["close"].rolling(40, min_periods=40).mean()
    comp_ret10 = df["close"] / df["close"].shift(10) - 1.0
    # Zero-mean realized volatility: sqrt(252 * mean(r^2 over 20 bars))
    r2_mean20 = (df["return"] ** 2).rolling(20, min_periods=20).mean()
    zero_mean_vol = np.sqrt(252.0 * r2_mean20)

    cond = (df["close"] > sma40) & (comp_ret10 >= -0.04)
    base = cond.astype(float)
    base.iloc[:40] = 0.0
    scale = (0.15 / zero_mean_vol.replace(0.0, np.nan)).fillna(1.0)
    sig = (base * scale).clip(lower=0.0, upper=1.5).fillna(0.0)
    return {"signals": sig, "slippage_rate": 0.0005, "commission_rate": 0.0005, "rebalance_tolerance": 0.05}


# =============================================================================
# Zone 2: Calendar vs Trading-Bar Traps (Month-End Last Trading Day / Weekly Holiday)
# =============================================================================

def _golden_v3_03_month_end_rebalance(df: pd.DataFrame) -> Dict[str, Any]:
    sma60 = df["close"].rolling(60, min_periods=60).mean().to_numpy()
    ret20 = (df["close"] / df["close"].shift(20) - 1.0).to_numpy()
    close = df["close"].to_numpy()

    # Causal detection of the last actual trading day of each calendar month in df:
    # Note: In live trading the calendar schedule of exchange trading days is known a priori;
    # in a daily DataFrame, a date is the last trading day of its month iff it is the last row
    # of that (year, month) group (or thelast business day of that calendar month).
    # Using pd.tseries.offsets.BMonthEnd().rollforward(date) == date is 100% causal and
    # independent of future rows in df!
    bme_dates = pd.to_datetime(df.index).map(lambda d: pd.tseries.offsets.BMonthEnd().rollforward(d) == d).to_numpy()

    out = np.zeros(len(df), dtype=float)
    current_weight = 0.0
    for i in range(len(df)):
        if i < 60:
            out[i] = 0.0
            continue
        if bme_dates[i]:
            if close[i] > sma60[i] and ret20[i] > 0.0:
                current_weight = 1.0
            else:
                current_weight = 0.0
        out[i] = current_weight
    return {"signals": pd.Series(out, index=df.index), "slippage_rate": 0.0005, "commission_rate": 0.0005, "rebalance_tolerance": 0.05}


def _golden_v3_04_weekly_macd_dip(df: pd.DataFrame) -> Dict[str, Any]:
    # Causal weekly close sampled at Friday close
    w_close = df["close"].resample("W-FRI", label="right", closed="right").last().dropna()
    ema12 = w_close.ewm(span=12, adjust=False).mean()
    ema26 = w_close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = dif - dea
    w_bull = (hist > 0.0).astype(float)
    w_bull.iloc[:26] = 0.0
    daily_w_bull = w_bull.reindex(df.index, method="ffill").fillna(0.0).to_numpy()

    sma20 = df["close"].rolling(20, min_periods=20).mean().to_numpy()
    close = df["close"].to_numpy()

    out = np.zeros(len(df), dtype=float)
    state = 0
    for i in range(len(df)):
        if i < 20 or np.isnan(sma20[i]):
            continue
        if state == 0:
            if daily_w_bull[i] == 1.0 and close[i] < sma20[i]:
                state = 1
        else:
            if close[i] >= sma20[i] or daily_w_bull[i] == 0.0:
                state = 0
        out[i] = float(state)
    return {"signals": pd.Series(out, index=df.index), "slippage_rate": 0.0005, "commission_rate": 0.0005, "rebalance_tolerance": 0.05}


# =============================================================================
# Zone 3: Coupled Multi-State & Exit-Reason Dependent Cooldown
# =============================================================================

def _golden_v3_05_exit_reason_cooldown(df: pd.DataFrame) -> Dict[str, Any]:
    sma20 = df["close"].rolling(20, min_periods=20).mean().to_numpy()
    close = df["close"].to_numpy()
    out = np.zeros(len(df), dtype=float)
    state = 0
    entry_p = 0.0
    cooldown = 0
    for i in range(len(df)):
        if i < 20 or np.isnan(sma20[i]):
            continue
        c = close[i]
        if state == 0:
            if cooldown > 0:
                cooldown -= 1
            elif c < 0.97 * sma20[i]:
                state = 1
                entry_p = c
        else:
            if c >= sma20[i]:
                # Take-profit exit: NO cooldown
                state = 0
                entry_p = 0.0
            elif c < 0.96 * entry_p:
                # Stop-loss exit: 6-bar cooldown starting next bar
                state = 0
                entry_p = 0.0
                cooldown = 6
        out[i] = float(state)
    return {"signals": pd.Series(out, index=df.index), "slippage_rate": 0.0005, "commission_rate": 0.0005, "rebalance_tolerance": 0.05}


def _golden_v3_06_profit_armed_trailing(df: pd.DataFrame) -> Dict[str, Any]:
    hi30 = df["close"].rolling(30, min_periods=30).max().shift(1).to_numpy()
    sma30 = df["close"].rolling(30, min_periods=30).mean().to_numpy()
    close = df["close"].to_numpy()
    out = np.zeros(len(df), dtype=float)
    state = 0
    entry_p = 0.0
    peak_p = 0.0
    armed = False
    for i in range(len(df)):
        if i < 30 or np.isnan(hi30[i]):
            continue
        c = close[i]
        if state == 0:
            if c > hi30[i]:
                state = 1
                entry_p = c
                peak_p = c
                armed = False
        else:
            peak_p = max(peak_p, c)
            if peak_p >= 1.04 * entry_p:
                armed = True
            active_stop = (0.975 * peak_p) if armed else (0.965 * entry_p)
            if c < active_stop or c < sma30[i]:
                state = 0
                entry_p = 0.0
                peak_p = 0.0
                armed = False
        out[i] = float(state)
    return {"signals": pd.Series(out, index=df.index), "slippage_rate": 0.0005, "commission_rate": 0.0005, "rebalance_tolerance": 0.05}


# =============================================================================
# Zone 4: 3-State / 4-State Finite Automata (Breakout-then-Retest & Scale-In/Out)
# =============================================================================

def _golden_v3_07_breakout_retest_3state(df: pd.DataFrame) -> Dict[str, Any]:
    hi20 = df["close"].rolling(20, min_periods=20).max().shift(1).to_numpy()
    sma20 = df["close"].rolling(20, min_periods=20).mean().to_numpy()
    close = df["close"].to_numpy()
    out = np.zeros(len(df), dtype=float)

    # mode: 0 = flat, 1 = armed_watch (pos=0), 2 = long (pos=1)
    mode = 0
    watch_left = 0
    entry_p = 0.0

    for i in range(len(df)):
        if i < 20 or np.isnan(hi20[i]):
            continue
        c = close[i]
        ma = sma20[i]
        if mode == 0:
            if c > hi20[i]:
                mode = 1
                watch_left = 5
            out[i] = 0.0
        elif mode == 1:
            if c < ma:
                mode = 0
                watch_left = 0
                out[i] = 0.0
            elif c <= 1.015 * ma:
                mode = 2
                entry_p = c
                watch_left = 0
                out[i] = 1.0
            else:
                watch_left -= 1
                if watch_left <= 0:
                    mode = 0
                out[i] = 0.0
        else:
            if c < ma or c >= 1.06 * entry_p:
                mode = 0
                entry_p = 0.0
                out[i] = 0.0
            else:
                out[i] = 1.0
    return {"signals": pd.Series(out, index=df.index), "slippage_rate": 0.0005, "commission_rate": 0.0005, "rebalance_tolerance": 0.05}


def _golden_v3_08_scale_in_scale_out_4state(df: pd.DataFrame) -> Dict[str, Any]:
    m = df["close"].rolling(20, min_periods=20).mean().to_numpy()
    s = df["close"].rolling(20, min_periods=20).std().to_numpy()
    close = df["close"].to_numpy()
    out = np.zeros(len(df), dtype=float)

    # state: "flat", "half_in", "full", "half_out"
    st = "flat"
    for i in range(len(df)):
        if i < 20 or np.isnan(m[i]) or np.isnan(s[i]):
            continue
        c, mi, si = close[i], m[i], s[i]
        if st == "flat":
            if c <= mi - 2.0 * si:
                st = "full"
            elif c < mi - 1.2 * si:
                st = "half_in"
        elif st == "half_in":
            if c >= mi:
                st = "flat"
            elif c <= mi - 2.0 * si:
                st = "full"
        elif st == "full":
            if c >= mi + 1.0 * si:
                st = "flat"
            elif c >= mi:
                st = "half_out"
        elif st == "half_out":
            if c >= mi + 1.0 * si or c < mi - 2.5 * si:
                st = "flat"

        if st == "flat":
            out[i] = 0.0
        elif st in ("half_in", "half_out"):
            out[i] = 0.5
        else:
            out[i] = 1.0
    return {"signals": pd.Series(out, index=df.index), "slippage_rate": 0.0005, "commission_rate": 0.0005, "rebalance_tolerance": 0.05}


# =============================================================================
# Zone 5: Pulse Crossover Trigger vs Level Holding & Asynchronous Event Window
# =============================================================================

def _golden_v3_09_crossover_pulse_hold(df: pd.DataFrame) -> Dict[str, Any]:
    sma10 = df["close"].rolling(10, min_periods=10).mean().to_numpy()
    sma30 = df["close"].rolling(30, min_periods=30).mean().to_numpy()
    vol20 = df["volume"].rolling(20, min_periods=20).mean().to_numpy()
    vol = df["volume"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)

    out = np.zeros(len(df), dtype=float)
    state = 0
    entry_p = 0.0
    for i in range(len(df)):
        if i < 30 or np.isnan(sma30[i]) or np.isnan(sma30[i - 1]):
            continue
        if state == 0:
            golden_cross_today = (sma10[i] > sma30[i]) and (sma10[i - 1] <= sma30[i - 1])
            vol_surge_today = vol[i] > 1.2 * vol20[i]
            if golden_cross_today and vol_surge_today:
                state = 1
                entry_p = close[i]
        else:
            if sma10[i] <= sma30[i] or close[i] < 0.95 * entry_p:
                state = 0
                entry_p = 0.0
        out[i] = float(state)
    return {"signals": pd.Series(out, index=df.index), "slippage_rate": 0.0005, "commission_rate": 0.0005, "rebalance_tolerance": 0.05}


def _golden_v3_10_recent_cross_window(df: pd.DataFrame) -> Dict[str, Any]:
    delta = df["close"].diff()
    g = delta.clip(lower=0.0).fillna(0.0).ewm(alpha=1.0 / 14.0, adjust=False).mean()
    l = (-delta.clip(upper=0.0)).fillna(0.0).ewm(alpha=1.0 / 14.0, adjust=False).mean().replace(0.0, np.nan)
    rsi = (100.0 - 100.0 / (1.0 + g / l)).fillna(100.0).to_numpy()
    sma20 = df["close"].rolling(20, min_periods=20).mean().to_numpy()
    close = df["close"].to_numpy()

    cross_up45 = np.zeros(len(df), dtype=bool)
    for i in range(1, len(df)):
        if rsi[i] >= 45.0 and rsi[i - 1] < 45.0:
            cross_up45[i] = True

    out = np.zeros(len(df), dtype=float)
    state = 0
    for i in range(len(df)):
        if i < 20 or np.isnan(sma20[i]):
            continue
        recent_cross = bool(np.any(cross_up45[max(0, i - 4) : i + 1]))
        if state == 0:
            if close[i] > sma20[i] and recent_cross:
                state = 1
        else:
            if close[i] < sma20[i] or rsi[i] > 70.0:
                state = 0
        out[i] = float(state)
    return {"signals": pd.Series(out, index=df.index), "slippage_rate": 0.0005, "commission_rate": 0.0005, "rebalance_tolerance": 0.05}


# =============================================================================
# Zone 6: Intrabar High/Low vs Close Execution Reference Traps
# =============================================================================

def _golden_v3_11_intrabar_high_low_trailing(df: pd.DataFrame) -> Dict[str, Any]:
    prior_hi25 = df["high"].rolling(25, min_periods=25).max().shift(1).to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()

    out = np.zeros(len(df), dtype=float)
    state = 0
    peak_high = 0.0
    for i in range(len(df)):
        if i < 25 or np.isnan(prior_hi25[i]):
            continue
        if state == 0:
            if close[i] > prior_hi25[i]:
                state = 1
                peak_high = high[i]
        else:
            if low[i] < 0.94 * peak_high:
                state = 0
                peak_high = 0.0
            else:
                peak_high = max(peak_high, high[i])
        out[i] = float(state)
    return {"signals": pd.Series(out, index=df.index), "slippage_rate": 0.0005, "commission_rate": 0.0005, "rebalance_tolerance": 0.05}


def _golden_v3_12_keltner_intrabar_band(df: pd.DataFrame) -> Dict[str, Any]:
    mid = df["close"].ewm(span=20, adjust=False).mean().to_numpy()
    rng_ma = (df["high"] - df["low"]).rolling(20, min_periods=20).mean().to_numpy()
    open_ = df["open"].to_numpy()
    close = df["close"].to_numpy()
    low = df["low"].to_numpy()

    out = np.zeros(len(df), dtype=float)
    state = 0
    for i in range(len(df)):
        if i < 20 or np.isnan(rng_ma[i]):
            continue
        upper = mid[i] + 1.5 * rng_ma[i]
        if state == 0:
            if close[i] > upper and close[i] > open_[i]:
                state = 1
        else:
            if low[i] < mid[i]:
                state = 0
        out[i] = float(state)
    return {"signals": pd.Series(out, index=df.index), "slippage_rate": 0.0005, "commission_rate": 0.0005, "rebalance_tolerance": 0.05}


# =============================================================================
# Zone 7: Dynamic Active-Substrategy Renormalization & Stateful Hold Buffer
# =============================================================================

def _golden_v3_13_active_invvol_renorm(df: pd.DataFrame) -> Dict[str, Any]:
    sma20 = df["close"].rolling(20, min_periods=20).mean().to_numpy()
    sma60 = df["close"].rolling(60, min_periods=60).mean().to_numpy()
    delta = df["close"].diff()
    g = delta.clip(lower=0.0).fillna(0.0).ewm(alpha=1.0 / 14.0, adjust=False).mean()
    l = (-delta.clip(upper=0.0)).fillna(0.0).ewm(alpha=1.0 / 14.0, adjust=False).mean().replace(0.0, np.nan)
    rsi = (100.0 - 100.0 / (1.0 + g / l)).fillna(100.0).to_numpy()

    vol10 = df["return"].rolling(10, min_periods=10).std().to_numpy()
    vol20 = df["return"].rolling(20, min_periods=20).std().to_numpy()
    vol30 = df["return"].rolling(30, min_periods=30).std().to_numpy()
    close = df["close"].to_numpy()

    out = np.zeros(len(df), dtype=float)
    for i in range(len(df)):
        if i < 60 or np.isnan(sma60[i]) or np.isnan(vol10[i]) or np.isnan(vol20[i]) or np.isnan(vol30[i]):
            continue
        active_inv_vols = []
        active_levs = []
        if close[i] > sma20[i] and vol10[i] > 0:
            active_inv_vols.append(1.0 / vol10[i])
            active_levs.append(1.2)
        if close[i] > sma60[i] and vol30[i] > 0:
            active_inv_vols.append(1.0 / vol30[i])
            active_levs.append(0.6)
        if rsi[i] < 35.0 and vol20[i] > 0:
            active_inv_vols.append(1.0 / vol20[i])
            active_levs.append(0.8)

        if not active_inv_vols:
            out[i] = 0.0
        else:
            inv_sum = sum(active_inv_vols)
            out[i] = sum((iv / inv_sum) * lev for iv, lev in zip(active_inv_vols, active_levs))
    return {"signals": pd.Series(out, index=df.index), "slippage_rate": 0.0005, "commission_rate": 0.0005, "rebalance_tolerance": 0.05}


def _golden_v3_14_turnover_hold_buffer(df: pd.DataFrame) -> Dict[str, Any]:
    sma50 = df["close"].rolling(50, min_periods=50).mean().to_numpy()
    rvol = df["rolling_vol"].to_numpy()
    close = df["close"].to_numpy()

    out = np.zeros(len(df), dtype=float)
    prev_sig = 0.0
    for i in range(len(df)):
        if i < 50 or np.isnan(sma50[i]):
            prev_sig = 0.0
            out[i] = 0.0
            continue
        if close[i] <= sma50[i]:
            raw_target = 0.0
        else:
            v = rvol[i]
            scale = (0.15 / v) if (not np.isnan(v) and v > 0) else 1.0
            raw_target = float(np.clip(scale, 0.0, 1.5))

        if raw_target == 0.0:
            sig = 0.0
        elif prev_sig == 0.0:
            sig = raw_target
        elif abs(raw_target - prev_sig) >= 0.15:
            sig = raw_target
        else:
            sig = prev_sig

        out[i] = sig
        prev_sig = sig
    return {"signals": pd.Series(out, index=df.index), "slippage_rate": 0.0005, "commission_rate": 0.0005, "rebalance_tolerance": 0.05}


V3_HARD_IMPL_CASES: List[HighFreedomImplCase] = [
    HighFreedomImplCase(
        case_id="v3_impl_01_wilder_atr_chandelier",
        title="Zone 1A: 14-day Wilder ATR Chandelier Exit on 20-day High Breakout",
        state_complexity="zone1_formula_substitution",
        prompt=(
            "实现 20 日高点突破加 14 日 Wilder ATR 吊灯止损（Chandelier Exit）策略："
            "前 20 个交易日空仓。空仓时，若收盘价严格突破过去 20 日盘中最高价则满仓做多；"
            "持仓期间追踪自建仓以来的最高盘中价，当收盘价严格跌破『持仓最高盘中价减去 3 倍 14 日 Wilder ATR』时平仓。使用默认摩擦成本。"
        ),
        golden_fn=_golden_v3_01_wilder_atr_chandelier,
    ),
    HighFreedomImplCase(
        case_id="v3_impl_02_compounded_drawdown_gate",
        title="Zone 1B: 10-day Compounded Return Filter + Zero-Mean Realized Volatility Target",
        state_complexity="zone1_formula_substitution",
        max_leverage=1.5,
        prompt=(
            "实现带 10 日动量过滤与零均值已实现波动率目标缩放的 40 日均线策略："
            "前 40 个交易日空仓。当收盘价高于 40 日均线且过去 10 个交易日累计收益率不低于 -4% 时做多；"
            "按 15% 目标年化波动率除以『20日零均值年化已实现波动率（基于日收益率平方均值计算）』进行仓位缩放，最大杠杆 1.5 倍。使用默认摩擦成本。"
        ),
        golden_fn=_golden_v3_02_compounded_drawdown_gate,
    ),
    HighFreedomImplCase(
        case_id="v3_impl_03_month_end_rebalance",
        title="Zone 2A: Monthly Rebalance on Last Actual Trading Day of Each Calendar Month",
        state_complexity="zone2_calendar_boundary",
        prompt=(
            "实现月末调仓的趋势动量策略：前 60 个交易日空仓；"
            "之后仅在每个月月末收盘时重新评估一次信号（当月其余交易日维持上月末仓位不变）："
            "若月末收盘价严格高于 60 日均线且过去 20 个交易日累计涨跌幅严格大于 0，则目标权重设为 1.0，否则设为 0.0。使用默认摩擦成本。"
        ),
        golden_fn=_golden_v3_03_month_end_rebalance,
    ),
    HighFreedomImplCase(
        case_id="v3_impl_04_weekly_macd_dip",
        title="Zone 2B: Causal Weekly MACD Histogram Filter Gating Daily Pullback Below SMA(20)",
        state_complexity="zone2_calendar_boundary",
        prompt=(
            "实现周线 MACD 过滤下的日线回调买入策略："
            "在周线收盘价上计算标准 MACD(12, 26, 9)，前 26 周及日线前 20 个交易日空仓；"
            "当最新已完成周线的 MACD 柱状线大于 0 且日线收盘价回调至日线 20 日均线下方时满仓做多；"
            "持仓期间一旦日线收盘价回到 20 日均线及以上或最新已完成周线 MACD 柱状线转负则平仓。使用默认摩擦成本。"
        ),
        golden_fn=_golden_v3_04_weekly_macd_dip,
    ),
    HighFreedomImplCase(
        case_id="v3_impl_05_exit_reason_cooldown",
        title="Zone 3A: Exit-Reason Dependent Cooldown (6-bar Cooldown on Stop-Loss Only, 0 on Take-Profit)",
        state_complexity="zone3_coupled_state",
        prompt=(
            "实现带止损冷却期的 20 日均线偏离均值回归策略：前 20 个交易日空仓。"
            "空仓且不处于冷却期时，若收盘价低于 20 日均线 3% 以上则满仓做多；"
            "持仓时若收盘价回到 20 日均线及以上则止盈平仓，若收盘价较建仓价下跌超 4% 则止损平仓。"
            "触发止损平仓后需冷却 6 个交易日不交易，正常止盈平仓不冷却。使用默认摩擦成本。"
        ),
        golden_fn=_golden_v3_05_exit_reason_cooldown,
    ),
    HighFreedomImplCase(
        case_id="v3_impl_06_profit_armed_trailing",
        title="Zone 3B: Profit-Armed Two-Stage Ratchet Stop (-3.5% Fixed -> +4% Armed -> -2.5% Trailing)",
        state_complexity="zone3_coupled_state",
        prompt=(
            "实现浮盈激活型双阶段移动止损突破策略：前 30 个交易日空仓。"
            "空仓时若收盘价突破过去 30 日最高收盘价则满仓做多。持仓期间止损分两阶段："
            "在持仓最高收盘价未达到建仓价 +4% 之前，使用建仓价下方 3.5% 的固定止损；"
            "一旦持仓最高收盘价达到或超过建仓价 +4%，永久切换为从持仓最高收盘价回撤 2.5% 的追踪止损。"
            "持仓期间跌破当前生效止损线或跌破 30 日均线则平仓。使用默认摩擦成本。"
        ),
        golden_fn=_golden_v3_06_profit_armed_trailing,
    ),
    HighFreedomImplCase(
        case_id="v3_impl_07_breakout_retest_3state",
        title="Zone 4A: 3-State Breakout-then-5-Bar-Pullback-Retest Confirmation Automaton",
        state_complexity="zone4_multi_state_automaton",
        prompt=(
            "实现突破回踩确认策略：前 20 个交易日空仓。收盘价创过去 20 日收盘新高时不直接买入，进入 5 个交易日观察期；"
            "观察期内若收盘价回踩至 20 日均线上方 1.5% 以内且未跌破 20 日均线则满仓建仓（若观察期内跌破 20 日均线或满 5 个交易日仍未回踩则取消观察）；"
            "建仓后跌破 20 日均线或较建仓价浮盈达 6% 时平仓。使用默认摩擦成本。"
        ),
        golden_fn=_golden_v3_07_breakout_retest_3state,
    ),
    HighFreedomImplCase(
        case_id="v3_impl_08_scale_in_scale_out_4state",
        title="Zone 4B: 4-State Asymmetric Scale-In & Scale-Out Bollinger Strategy",
        state_complexity="zone4_multi_state_automaton",
        prompt=(
            "实现分批建仓与分批止盈的 20 日布林带均值回归策略（记 20 日均线为 m，20 日标准差为 s，前 20 日空仓）："
            "空仓时若收盘价跌破 m - 1.2s 但高于 m - 2.0s 则建半仓（0.5），若跌破或等于 m - 2.0s 则直接建满仓（1.0）；"
            "持有首笔半仓时若继续跌破或等于 m - 2.0s 则加至满仓（1.0），若回到 m 及以上则清仓（0.0）；"
            "持有满仓时若回到 m 及以上但低于 m + 1.0s 则减半仓至 0.5 锁定利润（减仓后的剩余半仓禁止再次加仓，直到触及 m + 1.0s 止盈或跌破 m - 2.5s 止损时全部清仓），"
            "若直接达到 m + 1.0s 及以上则全部清仓。使用默认摩擦成本。"
        ),
        golden_fn=_golden_v3_08_scale_in_scale_out_4state,
    ),
    HighFreedomImplCase(
        case_id="v3_impl_09_crossover_pulse_hold",
        title="Zone 5A: Exact-Bar Golden-Cross + Volume-Surge Pulse Entry with Persistent Hold",
        state_complexity="zone5_pulse_vs_level",
        prompt=(
            "实现均线金叉放量确认入场策略：前 30 个交易日空仓。"
            "空仓时，仅在 10 日均线上穿 30 日均线的金叉发生当日、且当日成交量高于 20 日均量 1.2 倍时满仓建仓（若金叉当日未放量则本轮不入场）；"
            "建仓后持续持有，直到 10 日均线跌破或等于 30 日均线、或收盘价较建仓价下跌超 5% 时平仓。使用默认摩擦成本。"
        ),
        golden_fn=_golden_v3_09_crossover_pulse_hold,
    ),
    HighFreedomImplCase(
        case_id="v3_impl_10_recent_cross_window",
        title="Zone 5B: Asynchronous 5-Bar Lookback Window for RSI(14) Cross-Above-45 + SMA(20) Filter",
        state_complexity="zone5_pulse_vs_level",
        prompt=(
            "实现异步事件窗口共振策略：前 20 个交易日空仓。"
            "当空仓时，若今日收盘价高于 20 日均线，且在最近 5 个交易日内（含今日）14 日 Wilder RSI 曾发生过上穿 45.0 的事件，则满仓做多；"
            "持仓时若收盘价跌破 20 日均线或 RSI 高于 70.0 则平仓。使用默认摩擦成本。"
        ),
        golden_fn=_golden_v3_10_recent_cross_window,
    ),
    HighFreedomImplCase(
        case_id="v3_impl_11_intrabar_high_low_trailing",
        title="Zone 6A: Intrabar High Watermark Update with Intrabar Low Stop-Loss Check Ordering",
        state_complexity="zone6_intrabar_high_low",
        prompt=(
            "实现盘中极值追踪止损策略：前 25 个交易日空仓。当收盘价突破过去 25 日盘中最高价时满仓做多；"
            "持仓期间按每日盘中最高价更新最高水位，一旦盘中最低价从截至昨日的最高水位回撤超过 6% 则当日止损平仓。使用默认摩擦成本。"
        ),
        golden_fn=_golden_v3_11_intrabar_high_low_trailing,
    ),
    HighFreedomImplCase(
        case_id="v3_impl_12_keltner_intrabar_band",
        title="Zone 6B: Keltner Channel Bullish-Candle Breakout with Intrabar Low Midline Exit",
        state_complexity="zone6_intrabar_high_low",
        prompt=(
            "实现肯特纳通道阳线突破策略：前 20 个交易日空仓。中轨为收盘价 20 日指数移动平均，通道宽度为日内高低价差的 20 日简单均线，上轨为中轨加 1.5 倍通道宽度；"
            "空仓时若收盘价突破上轨且当日收阳线则满仓做多；持仓时若盘中最低价跌破中轨则平仓。使用默认摩擦成本。"
        ),
        golden_fn=_golden_v3_12_keltner_intrabar_band,
    ),
    HighFreedomImplCase(
        case_id="v3_impl_13_active_invvol_renorm",
        title="Zone 7A: 3-Substrategy Active-Only Inverse-Vol Dynamic Renormalization",
        state_complexity="zone7_portfolio_buffer",
        max_leverage=1.2,
        prompt=(
            "实现三子策略动态风险平价组合：前 60 个交易日空仓。包含三个多头子策略："
            "A（收盘 > 20日均线，目标杠杆 1.2，风险为 10日收益标准差）、"
            "B（收盘 > 60日均线，目标杠杆 0.6，风险为 30日收益标准差）、"
            "C（14日Wilder RSI < 35，目标杠杆 0.8，风险为 20日收益标准差）。"
            "每日对当前触发做多的子策略按风险倒数归一化配权，输出加权后的总目标仓位。使用默认摩擦成本。"
        ),
        golden_fn=_golden_v3_13_active_invvol_renorm,
    ),
    HighFreedomImplCase(
        case_id="v3_impl_14_turnover_hold_buffer",
        title="Zone 7B: Continuous Vol-Targeted Trend with Stateful 15% Rebalance Hold Buffer",
        state_complexity="zone7_portfolio_buffer",
        max_leverage=1.5,
        prompt=(
            "实现带 15% 换手抑制缓冲带的 50 日均线波动率目标策略：前 50 个交易日空仓。"
            "当收盘价高于 50 日均线时，理想仓位为 15% 除以 rolling_vol（上限 1.5 倍），否则理想仓位为 0。"
            "若理想仓位为 0 则立即清仓；若从空仓新建仓则直接采用理想仓位；"
            "若已在持仓中，仅当理想仓位与上一日实际发出仓位的绝对差值达到或超过 0.15 时才更新仓位，否则维持上一日仓位不变。使用默认摩擦成本。"
        ),
        golden_fn=_golden_v3_14_turnover_hold_buffer,
    ),
]
