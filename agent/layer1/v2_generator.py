from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple
import numpy as np
import pandas as pd

from agent.layer1.v2_hard import (
    SubtleAuditCase,
    HighFreedomImplCase,
)


# =============================================================================
# Part 1: 60 Subtle Cross-Module / Default-Parameter Audit Generator
# =============================================================================

def generate_subtle_audit_suite() -> List[SubtleAuditCase]:
    """
    Deterministically generate 60 high-craft multi-module / default-parameter audit cases:
      - 48 flawed cases across 8 subtle bug families (6 variations per family)
      - 12 complex multi-function clean controls
    Zero surface giveaways (`shift(-1)`, `center=True`, or `# BUG` comments are never used).
    """
    cases: List[SubtleAuditCase] = []

    # -------------------------------------------------------------------------
    # Family 1 (6 cases): Series.where / Series.mask default other=NaN + Backtest NaN hold
    # -------------------------------------------------------------------------
    f1_specs = [
        ("01", "40日均线趋势", 'sma = df["close"].rolling(40).mean()\n    raw_sig = (df["close"] > sma).astype(float)', '["Q3", "Q4"]', "where(allowed)"),
        ("02", "20日均线趋势", 'sma = df["close"].rolling(20).mean()\n    raw_sig = (df["close"] > sma).astype(float)', '["Q1", "Q4"]', "where(allowed)"),
        ("03", "MACD(12,26,9)趋势", 'fast = df["close"].ewm(span=12, adjust=False).mean()\n    slow = df["close"].ewm(span=26, adjust=False).mean()\n    dea = (fast - slow).ewm(span=9, adjust=False).mean()\n    raw_sig = ((fast - slow) > dea).astype(float)', '["Q1", "Q4"]', "where(allowed)"),
        ("04", "50日均线低波过滤", 'sma = df["close"].rolling(50).mean()\n    raw_sig = (df["close"] > sma).astype(float)', '["Q3", "Q4"]', "mask(~allowed)"),
        ("05", "30日唐奇安突破", 'hi = df["high"].rolling(30).max().shift(1)\n    raw_sig = (df["close"] > hi).astype(float)', '["Q1", "Q2"]', "where(allowed)"),
        ("06", "14日RSI动量过滤", 'delta = df["close"].diff()\n    g = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()\n    l = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()\n    rsi = 100 - 100 / (1 + g / l)\n    raw_sig = (rsi > 55).astype(float)', '["Q1", "Q4"]', "mask(~allowed)"),
    ]
    for idx, name, sig_code, reg_list, op_call in f1_specs:
        kw_first = "mask" if "mask" in op_call else "where"
        cases.append(
            SubtleAuditCase(
                case_id=f"sa_gen_nan_lock_{idx}",
                title=f"Regime-filtered {name} interacting with Backtest NaN handling",
                description=(
                    f"{name}加象限过滤策略：仅在指定象限 {reg_list} 允许持仓做多；"
                    f"一旦市场切换至其他象限，必须立即在次日开盘清仓至 0 股。"
                ),
                code=f"""import pandas as pd

def build_filtered_signal(df: pd.DataFrame) -> pd.Series:
    {sig_code}
    allowed = df["regime"].isin({reg_list})
    return raw_sig.{op_call}

def execute_backtest_bar(open_price: float, cash: float, position: int, signal: float, tol: float = 0.05) -> int:
    v_open = cash + position * open_price
    if pd.isna(signal):
        trade_shares = 0
    elif signal == 0:
        trade_shares = -position
    else:
        delta = v_open * signal - position * open_price
        trade_shares = int(delta // open_price) if abs(delta / v_open) >= tol else 0
    return position + trade_shares
""",
                has_flaw=True,
                flaw_types=("logical_inconsistency",),
                required_mechanism_keywords=(
                    (kw_first, "nan", "isna"),
                    ("hold", "keep", "trade_shares = 0", "not liquidate", "清仓", "保持", "不平仓"),
                ),
                reference_notes=f"`raw_sig.{op_call}` defaults `other=np.nan`, and `execute_backtest_bar` treats `NaN` as `trade_shares = 0` (hold position) instead of liquidating.",
            )
        )

    # -------------------------------------------------------------------------
    # Family 2 (6 cases): pandas resample('2D'/'3D'/'4D'/'5D') default label='left', closed='left'
    # -------------------------------------------------------------------------
    f2_specs = [
        ("07", "2D", 10, "2日K线10周期均线"),
        ("08", "3D", 8, "3日K线8周期均线"),
        ("09", "4D", 6, "4日K线6周期均线"),
        ("10", "5D", 12, "5日K线12周期均线"),
        ("11", "3D", 15, "3日K线15周期突破"),
        ("12", "2D", 20, "2日K线20周期Z-score"),
    ]
    for idx, freq, win, label_desc in f2_specs:
        cases.append(
            SubtleAuditCase(
                case_id=f"sa_gen_resample_left_{idx}",
                title=f"Multi-timeframe {freq} resampled filter aligned to daily bars",
                description=(
                    f"跨周期过滤策略：将日频收盘价按 `{freq}` 重采样构建高周期收盘价并计算 {label_desc}，"
                    f"前向填充对齐回日频索引后，与日线 14 日 RSI 反转信号取交集。"
                ),
                code=f"""import pandas as pd
from src.signals import compute_rsi_signal

def build_mtf_filtered_signal(df: pd.DataFrame) -> pd.Series:
    htf_close = df["close"].resample("{freq}").last()
    htf_ma = htf_close.rolling({win}).mean()
    htf_bull = (htf_close > htf_ma).astype(int)

    daily_htf = htf_bull.reindex(df.index, method="ffill").fillna(0)
    daily_rsi = compute_rsi_signal(df, window=14, lower=30.0, upper=70.0, mode="reversion")
    return (daily_htf & (daily_rsi == 1)).astype(int)
""",
                has_flaw=True,
                flaw_types=("lookahead_bias",),
                required_mechanism_keywords=(
                    ("resample", freq.lower()),
                    ("label='left'", 'label="left"', "closed='left'", 'closed="left"', "defaults to left", "default is left", "默认左", "左标签"),
                ),
                reference_notes=f"`resample('{freq}')` defaults to `label='left', closed='left'`, stamping the multi-day close on the first day of the bucket and leaking future closes via `ffill`.",
            )
        )

    # -------------------------------------------------------------------------
    # Family 3 (6 cases): Cross-module double shift(1) delaying short-horizon signal
    # -------------------------------------------------------------------------
    f3_specs = [
        ("13", "-0.015", "单日跌幅超 1.5% 次日开盘抄底"),
        ("14", "-0.020", "单日跌幅超 2.0% 次日开盘反弹"),
        ("15", "-0.012", "单日跌幅超 1.2% 隔日反转"),
        ("16", "-0.018", "单日急跌 1.8% 次日开盘做多"),
        ("17", "-0.025", "单日暴跌 2.5% 次日开盘博反弹"),
        ("18", "-0.010", "单日回落 1.0% 次日开盘短多"),
    ]
    for idx, thresh, desc in f3_specs:
        cases.append(
            SubtleAuditCase(
                case_id=f"sa_gen_double_shift_{idx}",
                title=f"Short-horizon 1-bar reversal ({thresh}) with Backtest execution",
                description=f"日频短线反转策略：{desc}，要求在触发日 T 的下一个交易日（T+1）开盘买入并仅持有 1 天。",
                code=f"""import pandas as pd
from src.backtest import Backtest

def compute_reversal_weights(df: pd.DataFrame) -> pd.Series:
    ret1 = df["close"].pct_change()
    raw = (ret1 < {thresh}).astype(float)
    return raw.shift(1).fillna(0.0)

def run_reversal_backtest(df: pd.DataFrame) -> pd.DataFrame:
    weights = compute_reversal_weights(df)
    bt = Backtest(slippage_rate=0.0005, commission_rate=0.0005, rebalance_tolerance=0.01)
    return bt.run(df, weights)
""",
                has_flaw=True,
                flaw_types=("logical_inconsistency",),
                required_mechanism_keywords=(
                    ("shift", "double", "twice", "two", "t+2", "两次", "双重", "延迟"),
                ),
                reference_notes="`compute_reversal_weights` shifts by 1 bar and `Backtest.run` shifts by 1 bar again, delaying T+1 execution to T+2.",
            )
        )

    # -------------------------------------------------------------------------
    # Family 4 (6 cases): ewm(span=W) default min_periods=0 premature trading during warm-up
    # -------------------------------------------------------------------------
    f4_specs = [
        ("19", 30, 0.12, 1.5),
        ("20", 45, 0.15, 1.8),
        ("21", 60, 0.10, 2.0),
        ("22", 75, 0.12, 1.5),
        ("23", 90, 0.15, 2.0),
        ("24", 120, 0.10, 1.5),
    ]
    for idx, span, tv, mlev in f4_specs:
        cases.append(
            SubtleAuditCase(
                case_id=f"sa_gen_ewm_warmup_{idx}",
                title=f"{span}-day EWM trend & vol scaling with mandatory {span}-day warm-up",
                description=(
                    f"{span} 日指数加权趋势策略：风控规则明确要求前 {span} 个交易日为指标预热期，严禁产生任何非零持仓；"
                    f"预热期结束后，当收盘价高于 {span} 日 EMA 时按 {int(tv*100)}% 目标波动率缩放做多（上限 {mlev} 倍）。"
                ),
                code=f"""import numpy as np
import pandas as pd

def build_ewm_strategy(df: pd.DataFrame) -> pd.Series:
    ema = df["close"].ewm(span={span}, adjust=False).mean()
    ewm_vol = df["close"].pct_change().ewm(span={span}, adjust=False).std() * np.sqrt(252)

    trend = (df["close"] > ema).astype(float)
    scale = ({tv} / ewm_vol.replace(0, np.nan)).clip(lower=0.0, upper={mlev})
    return (trend * scale).fillna(0.0)
""",
                has_flaw=True,
                flaw_types=("logical_inconsistency",),
                required_mechanism_keywords=(
                    ("ewm",),
                    ("min_periods", "not nan", "non-nan", "bar 1", "bar 2", "from the start", "immediately", "不会产生nan", "默认从第"),
                ),
                reference_notes=f"`ewm(span={span})` defaults to `min_periods=0`, producing non-NaN signals from bar 1/2 instead of staying flat during the {span}-bar warm-up.",
            )
        )

    # -------------------------------------------------------------------------
    # Family 5 (6 cases): Stateful trailing stop re-anchored to rolling window
    # -------------------------------------------------------------------------
    f5_specs = [
        ("25", 20, 0.95, "5%"),
        ("26", 25, 0.94, "6%"),
        ("27", 30, 0.93, "7%"),
        ("28", 40, 0.92, "8%"),
        ("29", 50, 0.95, "5%"),
        ("30", 60, 0.90, "10%"),
    ]
    for idx, win, mult, dd_str in f5_specs:
        cases.append(
            SubtleAuditCase(
                case_id=f"sa_gen_reanchor_{idx}",
                title=f"{win}-day breakout with {dd_str} post-entry high-water-mark trailing stop",
                description=(
                    f"突破加移动止损策略：当收盘价突破过去 {win} 日最高价（shift(1)）时入场做多；"
                    f"持仓期间跟踪自该笔建仓入场以来的最高收盘价，一旦回撤超过 {dd_str}（跌破最高价的 {mult} 倍）则平仓。"
                ),
                code=f"""import numpy as np
import pandas as pd

def build_breakout_trailing(df: pd.DataFrame) -> pd.Series:
    entry_hi = df["close"].rolling({win}).max().shift(1).to_numpy()
    peak_series = df["close"].rolling({win}).max().to_numpy()
    close_arr = df["close"].to_numpy()

    out = np.zeros(len(df), dtype=float)
    state = 0
    for i in range(len(df)):
        if state == 0:
            if not np.isnan(entry_hi[i]) and close_arr[i] > entry_hi[i]:
                state = 1
        else:
            if close_arr[i] < {mult} * peak_series[i]:
                state = 0
        out[i] = state
    return pd.Series(out, index=df.index)
""",
                has_flaw=True,
                flaw_types=("logical_inconsistency",),
                required_mechanism_keywords=(
                    ("rolling", "peak", "anchor", "entry", "window", "最高", "滚动", "重置"),
                ),
                reference_notes=f"`peak_series` uses a {win}-bar rolling max instead of tracking the highest close since entry, re-anchoring downward on long holds.",
            )
        )

    # -------------------------------------------------------------------------
    # Family 6 (6 cases): Multi-step sizing chain breaching max_leverage ceiling
    # -------------------------------------------------------------------------
    f6_specs = [
        ("31", 40, 0.15, 1.5, 1.30),
        ("32", 50, 0.12, 1.2, 1.25),
        ("33", 60, 0.15, 1.5, 1.40),
        ("34", 30, 0.10, 1.0, 1.50),
        ("35", 80, 0.15, 1.8, 1.25),
        ("36", 100, 0.12, 1.5, 1.35),
    ]
    for idx, win, tv, mlev, boost in f6_specs:
        cases.append(
            SubtleAuditCase(
                case_id=f"sa_gen_uncapped_chain_{idx}",
                title=f"Multi-step vol-targeted SMA({win}) with {boost}x conviction boost",
                description=(
                    f"多级定仓策略：基础信号为 {win} 日均线，经 {int(tv*100)}% 目标波动率缩放后，"
                    f"在放量日乘以 {boost} 倍加成；风控硬约束要求最终输出的目标权重在任何时刻严禁超过 {mlev} 倍。"
                ),
                code=f"""import pandas as pd
from src.signals import compute_moving_average_signal, compute_volatility_scaled_signal

def step1_vol_target(df: pd.DataFrame, sig: pd.Series) -> pd.Series:
    return compute_volatility_scaled_signal(df, sig, target_vol={tv}, max_leverage={mlev})

def step2_liquidity_boost(df: pd.DataFrame, weights: pd.Series) -> pd.Series:
    high_liq = (df["volume"] > df["volume"].rolling(40).mean()).map({{True: {boost}, False: 1.0}})
    return weights * high_liq

def build_strategy(df: pd.DataFrame) -> dict:
    base = compute_moving_average_signal(df, window={win})
    w1 = step1_vol_target(df, base)
    w2 = step2_liquidity_boost(df, w1)
    return {{"signals": w2, "slippage_rate": 0.0005, "commission_rate": 0.0005, "rebalance_tolerance": 0.05}}
""",
                has_flaw=True,
                flaw_types=("logical_inconsistency",),
                required_mechanism_keywords=(
                    (str(boost), str(mlev), "clip", "cap", "exceed", "上限", "超过"),
                ),
                reference_notes=f"`step2_liquidity_boost` multiplies already-capped weights by {boost} without re-applying `.clip(upper={mlev})`.",
            )
        )

    # -------------------------------------------------------------------------
    # Family 7 (6 cases): groupby(...).transform(...) full-sample leakage
    # -------------------------------------------------------------------------
    f7_specs = [
        ("37", '"regime"', '"rolling_vol"', '"median"', "象限内波动率中位数"),
        ("38", '"regime"', '"volume"', '"mean"', "象限内成交量均值"),
        ("39", 'df.index.month', '"rolling_vol"', '"median"', "同月份历史波动率中位数"),
        ("40", '"regime"', '"return"', '"std"', "象限内收益率标准差"),
        ("41", 'df.index.quarter', '"volume"', '"median"', "同季度成交量中位数"),
        ("42", '"regime"', '"rolling_vol"', '"mean"', "象限内波动率均值"),
    ]
    for idx, grp_expr, col_expr, agg_fn, desc in f7_specs:
        cases.append(
            SubtleAuditCase(
                case_id=f"sa_gen_groupby_leak_{idx}",
                title=f"Groupby {grp_expr} {agg_fn} normalization on {col_expr}",
                description=f"条件归一化趋势策略：按 {desc} 对当日特征进行相对高低判断，低风险时满仓持有 50 日均线信号，否则半仓。",
                code=f"""import pandas as pd
from src.signals import compute_moving_average_signal

def build_group_normalized_strategy(df: pd.DataFrame) -> pd.Series:
    base = compute_moving_average_signal(df, window=50).astype(float)
    grp_benchmark = df.groupby({grp_expr})[{col_expr}].transform({agg_fn})
    mult = (df[{col_expr}] <= grp_benchmark).map({{True: 1.0, False: 0.5}})
    return (base * mult).fillna(0.0)
""",
                has_flaw=True,
                flaw_types=("lookahead_bias",),
                required_mechanism_keywords=(
                    ("groupby", "transform"),
                    ("future", "full", "entire", "lookahead", "全样本", "未来"),
                ),
                reference_notes=f"`groupby({grp_expr})[{col_expr}].transform({agg_fn})` aggregates over all past and future rows in `df`.",
            )
        )

    # -------------------------------------------------------------------------
    # Family 8 (6 cases): rebalance_tolerance blocking gradual steps (3) + silent fillna(0.0) (3)
    # -------------------------------------------------------------------------
    for idx, step_sz, tol_sz in [("43", 0.02, 0.05), ("44", 0.03, 0.06), ("45", 0.04, 0.08)]:
        cases.append(
            SubtleAuditCase(
                case_id=f"sa_gen_tol_block_{idx}",
                title=f"Daily {int(step_sz*100)}% linear scale-out under rebalance_tolerance={tol_sz}",
                description=(
                    f"线性减仓策略：当 20 日均线跌破 60 日均线后，要求每天固定减仓 {int(step_sz*100)}%（即每日权重递减 {step_sz}），"
                    f"在默认 Backtest(rebalance_tolerance={tol_sz}) 下逐日平滑卖出。"
                ),
                code=f"""import numpy as np
import pandas as pd
from src.backtest import Backtest

def build_linear_scaleout(df: pd.DataFrame) -> pd.Series:
    bull = (df["close"].rolling(20).mean() > df["close"].rolling(60).mean()).to_numpy()
    out = np.zeros(len(df), dtype=float)
    w = 0.0
    for i in range(len(df)):
        w = 1.0 if bull[i] else max(0.0, round(w - {step_sz}, 4))
        out[i] = w
    return pd.Series(out, index=df.index)

def run_scaleout(df: pd.DataFrame) -> pd.DataFrame:
    bt = Backtest(rebalance_tolerance={tol_sz})
    return bt.run(df, build_linear_scaleout(df))
""",
                has_flaw=True,
                flaw_types=("logical_inconsistency", "friction_erosion"),
                required_mechanism_keywords=(
                    ("rebalance_tolerance", str(tol_sz), str(step_sz), "tolerance", "容忍带"),
                ),
                reference_notes=f"Daily weight step {step_sz} is smaller than `rebalance_tolerance={tol_sz}`, so `Backtest.run` skips individual daily reductions.",
            )
        )

    for idx, col_name, thresh in [("46", "vix_z", 0.5), ("47", "credit_spread_z", 0.8), ("48", "macro_risk_score", 1.0)]:
        cases.append(
            SubtleAuditCase(
                case_id=f"sa_gen_silent_zero_{idx}",
                title=f"External {col_name} filter with fillna(0.0) neutral fallback",
                description=(
                    f"外部风险过滤策略：将外部风险指标 `{col_name}` 对齐到日频，仅当 `{col_name} <= {thresh}` 时允许做多 50 日均线；"
                    f"若某交易日外部风险数据缺失，风控要求严禁开仓（必须保持 0 仓位）。"
                ),
                code=f"""import pandas as pd
from src.signals import compute_moving_average_signal

def build_risk_gated_trend(df: pd.DataFrame, ext_df: pd.DataFrame) -> pd.Series:
    base = compute_moving_average_signal(df, window=50)
    risk_series = ext_df["{col_name}"].reindex(df.index).fillna(0.0)
    allow_long = (risk_series <= {thresh}).astype(int)
    return (base & allow_long).astype(int)
""",
                has_flaw=True,
                flaw_types=("logical_inconsistency",),
                required_mechanism_keywords=(
                    ("fillna(0", "0.0", "missing", "reindex", "缺失"),
                    (str(thresh), "allow", "safe", "做多", "开仓"),
                ),
                reference_notes=f"`.fillna(0.0)` turns missing `{col_name}` dates into `0.0 <= {thresh}` (`True`), allowing long entries when external data is missing.",
            )
        )

    # -------------------------------------------------------------------------
    # Family 9 (12 cases): Complex Multi-Function Clean Controls
    # -------------------------------------------------------------------------
    clean_windows = [20, 30, 40, 50, 60, 80, 25, 35, 45, 55, 70, 90]
    for k, w in enumerate(clean_windows, start=49):
        cases.append(
            SubtleAuditCase(
                case_id=f"sa_gen_clean_{k:02d}",
                title=f"Causal {w}-day stateful breakout with explicit 0.0 regime mask and vol cap",
                description=(
                    f"合规多模块突破策略：突破过去 {w} 日最高价（shift(1)）入场，跟踪入场后最高收盘价回撤 6% 止损；"
                    f"仅在 Q1/Q4 象限持仓（非允许象限显式置 0.0），按 15% 目标波动率缩放并在最终输出前再次执行 1.5 倍上限截断。"
                ),
                code=f"""import numpy as np
import pandas as pd

def compute_causal_hwm_breakout(df: pd.DataFrame, window: int = {w}) -> pd.Series:
    hi = df["close"].rolling(window, min_periods=window).max().shift(1).to_numpy()
    c_arr = df["close"].to_numpy()
    out = np.zeros(len(df), dtype=float)
    state, peak = 0, 0.0
    for i in range(len(df)):
        if state == 0:
            if not np.isnan(hi[i]) and c_arr[i] > hi[i]:
                state, peak = 1, c_arr[i]
        else:
            peak = max(peak, c_arr[i])
            if c_arr[i] < 0.94 * peak:
                state, peak = 0, 0.0
        out[i] = float(state)
    return pd.Series(out, index=df.index)

def build_clean_pipeline(df: pd.DataFrame) -> dict:
    base = compute_causal_hwm_breakout(df, window={w})
    mask = df["regime"].isna() | df["regime"].isin(["Q1", "Q4"])
    gated = base.where(mask, 0.0)
    vol = df["rolling_vol"].replace(0.0, np.nan)
    scale = (0.15 / vol).fillna(1.0)
    final_sig = (gated * scale).clip(lower=0.0, upper=1.5).fillna(0.0)
    return {{"signals": final_sig, "slippage_rate": 0.0005, "commission_rate": 0.0005, "rebalance_tolerance": 0.05}}
""",
                has_flaw=False,
                flaw_types=(),
                required_mechanism_keywords=(),
                reference_notes="Clean control: causal shift(1), true post-entry peak tracking, explicit `.where(mask, 0.0)`, and final `.clip(0.0, 1.5)`.",
            )
        )

    return cases


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
