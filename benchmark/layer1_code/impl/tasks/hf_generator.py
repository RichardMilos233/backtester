"""
Layer 1B scaled suite: deterministic generator for 60 composable stateful
implementation cases (entry family x exit family x sizing), each with a golden.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple
import numpy as np
import pandas as pd

from benchmark.layer1_code.impl.tasks.schema import HighFreedomImplCase


# =============================================================================
# Part 2: 60 Composable High-Freedom Stateful Implementation Cases
# =============================================================================

def _make_composed_golden_fn(
    entry_family: str,
    entry_params: Dict[str, Any],
    exit_mod: str,
    exit_params: Dict[str, Any],
    overlay_mode: str,
    overlay_params: Dict[str, Any],
) -> Callable[[pd.DataFrame], Dict[str, Any]]:
    """
    Build a deterministic, 100% causal golden_fn for the (entry_family, exit_mod, overlay_mode)
    combination.
    """
    def _golden(df: pd.DataFrame) -> Dict[str, Any]:
        n = len(df)
        c_arr = df["close"].to_numpy(dtype=float)
        ret_arr = df["return"].to_numpy(dtype=float)

        # 1. Precompute causal entry & base-exit triggers
        warmup = 20
        if entry_family == "breakout":
            ew, xw = entry_params["entry_w"], entry_params["exit_w"]
            warmup = ew
            hi = df["close"].rolling(ew, min_periods=ew).max().shift(1).to_numpy()
            lo = df["close"].rolling(xw, min_periods=xw).min().shift(1).to_numpy()
            enter_cond = lambda i: (not np.isnan(hi[i])) and (c_arr[i] > hi[i])
            base_exit_cond = lambda i: (not np.isnan(lo[i])) and (c_arr[i] < lo[i])
        elif entry_family == "wilder_rsi":
            w, lo_th, up_th = entry_params["w"], entry_params["lower"], entry_params["upper"]
            warmup = w
            delta = df["close"].diff()
            g = delta.clip(lower=0.0).fillna(0.0).ewm(alpha=1.0 / w, adjust=False).mean()
            l = (-delta.clip(upper=0.0)).fillna(0.0).ewm(alpha=1.0 / w, adjust=False).mean().replace(0.0, np.nan)
            rsi = (100.0 - 100.0 / (1.0 + g / l)).fillna(100.0).to_numpy()
            enter_cond = lambda i: (i >= w) and (rsi[i] < lo_th)
            base_exit_cond = lambda i: (i >= w) and (rsi[i] > up_th)
        elif entry_family == "zscore_dip":
            w, ez, xz = entry_params["w"], entry_params["entry_z"], entry_params["exit_z"]
            warmup = w
            m = df["close"].rolling(w, min_periods=w).mean().to_numpy()
            s = df["close"].rolling(w, min_periods=w).std().to_numpy()
            enter_cond = lambda i: (not np.isnan(m[i])) and (s[i] > 0) and ((c_arr[i] - m[i]) / s[i] <= ez)
            base_exit_cond = lambda i: (not np.isnan(m[i])) and (s[i] > 0) and ((c_arr[i] - m[i]) / s[i] >= xz)
        elif entry_family == "ema_hysteresis":
            w, up_b, dn_b = entry_params["w"], entry_params["up_pct"], entry_params["dn_pct"]
            warmup = w
            ema = df["close"].ewm(span=w, adjust=False).mean().to_numpy()
            enter_cond = lambda i: (i >= w) and (c_arr[i] > (1.0 + up_b) * ema[i])
            base_exit_cond = lambda i: (i >= w) and (c_arr[i] < (1.0 - dn_b) * ema[i])
        elif entry_family == "dual_sma":
            fw, sw = entry_params["fast_w"], entry_params["slow_w"]
            warmup = sw
            f_ma = df["close"].rolling(fw, min_periods=fw).mean().to_numpy()
            s_ma = df["close"].rolling(sw, min_periods=sw).mean().to_numpy()
            enter_cond = lambda i: (not np.isnan(s_ma[i])) and (f_ma[i] > s_ma[i])
            base_exit_cond = lambda i: (not np.isnan(s_ma[i])) and (f_ma[i] <= s_ma[i])
        elif entry_family == "sma_trend":
            w = entry_params["w"]
            warmup = w
            sma = df["close"].rolling(w, min_periods=w).mean().to_numpy()
            enter_cond = lambda i: (not np.isnan(sma[i])) and (c_arr[i] > sma[i])
            base_exit_cond = lambda i: (not np.isnan(sma[i])) and (c_arr[i] <= sma[i])
        else:
            raise ValueError(f"Unknown entry_family: {entry_family}")

        # 2. Execute stateful loop with the specified exit_mod
        out = np.zeros(n, dtype=float)
        state = 0
        entry_price = 0.0
        peak_price = 0.0
        stop_price = 0.0
        bars_held = 0
        cooldown = 0
        freeze = 0

        for i in range(n):
            if i < warmup:
                out[i] = 0.0
                continue

            c = c_arr[i]

            if exit_mod == "crash_freeze":
                crash_th = exit_params["crash_ret"]
                f_bars = exit_params["freeze_bars"]
                if not np.isnan(ret_arr[i]) and ret_arr[i] < crash_th:
                    state = 0
                    freeze = f_bars
                    out[i] = 0.0
                    continue
                if freeze > 0:
                    freeze -= 1
                    state = 0
                    out[i] = 0.0
                    continue

            if state == 0:
                if exit_mod == "stop_cooldown" and cooldown > 0:
                    cooldown -= 1
                    out[i] = 0.0
                    continue
                if enter_cond(i):
                    state = 1
                    entry_price = c
                    peak_price = c
                    bars_held = 1
                    if exit_mod == "breakeven_ratchet":
                        stop_price = (1.0 - exit_params["init_stop"]) * c
                out[i] = float(state)
            else:
                peak_price = max(peak_price, c)
                should_exit = base_exit_cond(i)

                if exit_mod == "trailing_hwm":
                    if c < (1.0 - exit_params["trail_dd"]) * peak_price:
                        should_exit = True
                elif exit_mod == "stop_cooldown":
                    if c < (1.0 - exit_params["stop_dd"]) * entry_price:
                        should_exit = True
                elif exit_mod == "time_stop":
                    if bars_held >= exit_params["max_bars"]:
                        should_exit = True
                elif exit_mod == "breakeven_ratchet":
                    if c >= (1.0 + exit_params["trigger_gain"]) * entry_price:
                        stop_price = max(stop_price, entry_price)
                    if c < stop_price:
                        should_exit = True

                if should_exit:
                    state = 0
                    entry_price = 0.0
                    peak_price = 0.0
                    stop_price = 0.0
                    bars_held = 0
                    if exit_mod == "stop_cooldown":
                        cooldown = exit_params["cooldown_bars"]
                else:
                    bars_held += 1
                out[i] = float(state)

        sig = pd.Series(out, index=df.index, dtype=float)

        # 3. Apply overlay_mode
        if overlay_mode == "q1q4_vol_cap":
            mask = df["regime"].isna() | df["regime"].isin(["Q1", "Q4"])
            sig = sig.where(mask, 0.0)
            vol = df["rolling_vol"].replace(0.0, np.nan)
            scale = (overlay_params["target_vol"] / vol).fillna(1.0)
            sig = (sig * scale).clip(lower=0.0, upper=overlay_params["max_leverage"]).fillna(0.0)

        return {
            "signals": sig,
            "slippage_rate": float(overlay_params.get("slippage_rate", 0.0005)),
            "commission_rate": float(overlay_params.get("commission_rate", 0.0005)),
            "rebalance_tolerance": float(overlay_params.get("rebalance_tolerance", 0.05)),
        }

    return _golden


def generate_high_freedom_impl_suite() -> List[HighFreedomImplCase]:
    """
    Deterministically generate 60 Composable High-Freedom Stateful Implementation cases:
      6 Entry Signal Families x 5 Stateful Exit/Risk Modifiers x 2 Overlay Modes = 60 cases.
    """
    entry_families: List[Tuple[str, Dict[str, Any], str]] = [
        (
            "breakout",
            {"entry_w": 25, "exit_w": 12},
            "通道突破入场：前 25 个交易日（index 0..24）为预热期保持空仓；空仓时若今日收盘价严格大于过去 25 个交易日（不含今日）最高收盘价则触发入场；基础出场条件为今日收盘价严格小于过去 12 个交易日（不含今日）最低收盘价",
        ),
        (
            "wilder_rsi",
            {"w": 14, "lower": 32.0, "upper": 68.0},
            "Wilder RSI(14) 反转入场：使用 ewm(alpha=1/14, adjust=False) 计算 RSI，前 14 个交易日（index 0..13）预热期保持空仓；空仓时若今日 RSI < 32.0 则触发入场；基础出场条件为今日 RSI > 68.0",
        ),
        (
            "zscore_dip",
            {"w": 20, "entry_z": -1.4, "exit_z": 0.0},
            "20 日 Z-score 超跌入场：计算 20 日收盘价均值与标准差，前 20 个交易日（index 0..19）预热期保持空仓；空仓时若今日 Z-score <= -1.4 则触发入场；基础出场条件为今日 Z-score >= 0.0",
        ),
        (
            "ema_hysteresis",
            {"w": 30, "up_pct": 0.012, "dn_pct": 0.008},
            "30 日 EMA 迟滞通道入场：计算 ewm(span=30, adjust=False).mean()，前 30 个交易日（index 0..29）预热期保持空仓；空仓时若今日收盘价 > 1.012 * EMA30 则触发入场；基础出场条件为今日收盘价 < 0.992 * EMA30",
        ),
        (
            "dual_sma",
            {"fast_w": 20, "slow_w": 50},
            "双均线趋势入场：计算 20 日与 50 日简单均线，前 50 个交易日（index 0..49）预热期保持空仓；空仓时若今日 SMA20 > SMA50 则触发入场；基础出场条件为今日 SMA20 <= SMA50",
        ),
        (
            "sma_trend",
            {"w": 40},
            "40 日均线趋势入场：计算 40 日简单均线，前 40 个交易日（index 0..39）预热期保持空仓；空仓时若今日收盘价 > SMA40 则触发入场；基础出场条件为今日收盘价 <= SMA40",
        ),
    ]

    exit_modifiers: List[Tuple[str, Dict[str, Any], str, str]] = [
        (
            "trailing_hwm",
            {"trail_dd": 0.06},
            "long_state_ratchet",
            "叠加高水位 6% 移动止损：持仓期间每个交易日持续更新自本次入场以来的最高收盘价 peak_price = max(peak_price, 今日收盘价)；若触发基础出场条件或今日收盘价 < 0.94 * peak_price，则当日平仓归零",
        ),
        (
            "stop_cooldown",
            {"stop_dd": 0.05, "cooldown_bars": 5},
            "long_state_ratchet",
            "叠加 5% 固定止损与 5 个交易日冷却期：持仓期间若触发基础出场条件或今日收盘价 < 0.95 * 入场收盘价，则当日平仓归零并将冷却计数器设为 cooldown = 5（在随后的 5 个交易日内每过一天减 1，期间禁止再次入场）",
        ),
        (
            "time_stop",
            {"max_bars": 10},
            "medium_state",
            "叠加 10 个交易日最长持仓时间止损：入场当日计为已持仓第 1 天（bars_held = 1）；持仓期间每个交易日检查若触发基础出场条件或 bars_held >= 10，则当日平仓归零（否则继续持仓并将 bars_held 加 1）",
        ),
        (
            "breakeven_ratchet",
            {"init_stop": 0.04, "trigger_gain": 0.05},
            "long_state_ratchet",
            "叠加保本止损棘轮（Breakeven Ratchet）：入场时初始止损价 stop_price = 0.96 * 入场收盘价；持仓期间若今日收盘价 >= 1.05 * 入场收盘价，则将 stop_price 上移锁定至入场收盘价；若触发基础出场条件或今日收盘价 < stop_price 则当日平仓归零",
        ),
        (
            "crash_freeze",
            {"crash_ret": -0.03, "freeze_bars": 8},
            "medium_state",
            "叠加单日暴跌熔断锁定期：无论处于空仓还是持仓，只要今日收益率 df['return'] < -0.03，当日立即强制清仓（0.0）并将熔断计数器设为 freeze = 8（在随后的 8 个交易日内强制保持 0.0 仓位且每过一天 freeze 减 1）",
        ),
    ]

    overlays: List[Tuple[str, Dict[str, Any], str]] = [
        (
            "standard",
            {"max_leverage": 1.0, "slippage_rate": 0.0005, "commission_rate": 0.0005, "rebalance_tolerance": 0.05},
            "仓位与摩擦设置：持仓状态目标权重为 1.0，空仓为 0.0（严禁输出 NaN）；摩擦参数使用默认值 slippage_rate=0.0005, commission_rate=0.0005, rebalance_tolerance=0.05。",
        ),
        (
            "q1q4_vol_cap",
            {"target_vol": 0.15, "max_leverage": 1.2, "slippage_rate": 0.0008, "commission_rate": 0.0004, "rebalance_tolerance": 0.06},
            "象限过滤与波动率缩放设置：仅在 df['regime'] 为 'Q1'、'Q4' 或 NaN 时保留状态机信号，在 'Q2'、'Q3' 象限强制将权重置为 0.0（严禁留下 NaN）；随后乘以 (0.15 / df['rolling_vol'])（若 rolling_vol 为 NaN 或 0 则乘子取 1.0），并对最终权重执行 .clip(0.0, 1.2) 确保上限不超过 1.2；摩擦参数设为 slippage_rate=0.0008, commission_rate=0.0004, rebalance_tolerance=0.06。",
        ),
    ]

    cases: List[HighFreedomImplCase] = []
    idx = 0
    for ef_name, ef_params, ef_desc in entry_families:
        for ex_name, ex_params, complexity, ex_desc in exit_modifiers:
            for ov_name, ov_params, ov_desc in overlays:
                idx += 1
                cid = f"hf_gen_{idx:02d}_{ef_name}_{ex_name}_{ov_name}"
                title = f"[{idx:02d}] {ef_name} + {ex_name} ({ov_name})"
                prompt = (
                    f"实现以下复合状态机量化策略：\n"
                    f"1) {ef_desc}；\n"
                    f"2) {ex_desc}；\n"
                    f"3) {ov_desc}"
                )
                g_fn = _make_composed_golden_fn(ef_name, ef_params, ex_name, ex_params, ov_name, ov_params)
                cases.append(
                    HighFreedomImplCase(
                        case_id=cid,
                        title=title,
                        state_complexity=complexity if ov_name == "standard" else "multi_tf_sizing",
                        prompt=prompt,
                        golden_fn=g_fn,
                        max_leverage=float(ov_params["max_leverage"]),
                    )
                )

    return cases
