from __future__ import annotations

from dataclasses import dataclass, asdict
import inspect
import json
import math
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.backtest import Backtest
from src.metrics import compute_strategy_metrics
from agent.llm import BaseLLMClient, LLMUsage
from agent.layer1.schema import VALID_FLAW_TYPES
from agent.layer1.sandbox import (
    extract_python_code,
    compile_build_strategy,
    check_no_lookahead,
)


# =============================================================================
# Part 1: Subtle Cross-Module / Implicit-Default Audit Cases (12 cases)
# =============================================================================

@dataclass(frozen=True)
class SubtleAuditCase:
    case_id: str
    title: str
    description: str
    code: str
    has_flaw: bool
    flaw_types: Tuple[str, ...]
    # Groups of keyword alternatives: at least one keyword from EACH group must appear
    # in the agent's explanation/flawed_location to prove it identified the true mechanism.
    required_mechanism_keywords: Tuple[Tuple[str, ...], ...]
    reference_notes: str


@dataclass
class SubtleAuditEvalResult:
    case_id: str
    has_flaw_gt: bool
    flaw_types_gt: Tuple[str, ...]
    has_flaw_pred: bool
    flaw_types_pred: Tuple[str, ...]
    binary_correct: float
    category_f1: float
    mechanism_hit: bool
    strict_pass: bool
    flawed_location: str = ""
    explanation: str = ""
    raw_response: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["flaw_types_gt"] = list(self.flaw_types_gt)
        d["flaw_types_pred"] = list(self.flaw_types_pred)
        return d


SUBTLE_AUDIT_SYSTEM_PROMPT = """You are a principal quantitative researcher auditing multi-module strategy and backtesting code.
Carefully inspect the interaction between the strategy code, pandas/numpy default parameters, and the backtest engine / data pipeline shown in the snippet.

Valid flaw categories (choose ONLY from this exact list):
- "lookahead_bias": Future information leaks into historical decisions (e.g., via pandas resampling defaults, cross-sectional/group transforms, unaligned timestamps, or execution timing mismatches).
- "friction_erosion": Rebalance tolerance, turnover, or fee/slippage mechanics degrade or distort execution.
- "data_snooping": Overfitting via hindsight selection or sample leakage.
- "logical_inconsistency": Cross-module contract mismatches, NaN vs 0 semantics, default parameter traps, state-machine re-anchoring bugs, or sizing caps that fail to bind.

If the code handles all edge cases, defaults, and cross-module contracts properly, set "has_flaw" to false and "flaw_types" to [].

Output ONLY valid JSON matching this schema:
{
  "has_flaw": <bool>,
  "flaw_types": [<zero or more strings from the valid flaw categories>],
  "flawed_location": "<specific function call or cross-module interaction where the bug occurs, or 'none'>",
  "explanation": "<precise technical explanation of the implicit default or cross-module interaction causing the bug>"
}
"""


SUBTLE_AUDIT_CASES: List[SubtleAuditCase] = [
    # 1. Series.where(cond) default other=NaN + Backtest.run pd.isna(signal) holding position
    SubtleAuditCase(
        case_id="sa_nan_lock_01",
        title="Regime-filtered trend strategy with Backtest engine",
        description=(
            "趋势加象限过滤策略：当收盘价高于 40 日均线时做多，且仅在低波动象限（Q3、Q4）允许持仓；"
            "一旦市场进入高波动象限（Q1、Q2），策略必须立即清仓（0 仓位）以规避暴跌风险。"
        ),
        code="""import pandas as pd

# --- Module 1: Strategy signal generator ---
def build_regime_filtered_signal(df: pd.DataFrame) -> pd.Series:
    sma40 = df["close"].rolling(40).mean()
    raw_sig = (df["close"] > sma40).astype(float)
    allowed = df["regime"].isin(["Q3", "Q4"])
    return raw_sig.where(allowed)

# --- Module 2: Backtest execution loop (from src/backtest.py) ---
def run_backtest_step(open_price, current_cash, current_position, signal, tol=0.05):
    v_open = current_cash + current_position * open_price
    if pd.isna(signal):
        trade_shares = 0
    elif signal == 0:
        trade_shares = -current_position
    else:
        target_val = v_open * signal
        delta = target_val - current_position * open_price
        trade_shares = int(delta // open_price) if abs(delta / v_open) >= tol else 0
    return current_position + trade_shares
""",
        has_flaw=True,
        flaw_types=("logical_inconsistency",),
        required_mechanism_keywords=(
            ("where", "nan", "isna"),
            ("hold", "keep", "trade_shares = 0", "not liquidate", "清仓", "保持", "不平仓", "0"),
        ),
        reference_notes=(
            "`raw_sig.where(allowed)` omits `other=0.0`, so `pandas.Series.where` defaults to filling False "
            "entries with `NaN`. In `run_backtest_step`, `if pd.isna(signal): trade_shares = 0` keeps "
            "`current_position` unchanged instead of liquidating (`signal == 0`), locking a long position through Q1/Q2 crashes."
        ),
    ),

    # 2. pandas resample('3D') default label='left', closed='left' + reindex(method='ffill')
    SubtleAuditCase(
        case_id="sa_resample_left_02",
        title="Multi-timeframe 3-day bar momentum filter on daily data",
        description=(
            "多周期共振策略：将日频收盘价聚合为 3 日 K 线，计算 3 日 K 线的 10 周期均线；"
            "将其对齐回日频索引后，当高周期收盘价高于高周期均线且日线 RSI(14) < 35 时做多。"
        ),
        code="""import pandas as pd
from src.signals import compute_rsi_signal

def build_mtf_strategy(df: pd.DataFrame) -> pd.Series:
    # Aggregate daily close to 3-day bars and compute 10-bar trend
    htf_close = df["close"].resample("3D").last()
    htf_ma = htf_close.rolling(10).mean()
    htf_bull = (htf_close > htf_ma).astype(int)

    # Align higher-timeframe filter back to daily trading dates
    daily_htf_bull = htf_bull.reindex(df.index, method="ffill").fillna(0)
    daily_rsi = compute_rsi_signal(df, window=14, lower=35.0, upper=65.0, mode="reversion")
    return (daily_htf_bull & (daily_rsi == 1)).astype(int)
""",
        has_flaw=True,
        flaw_types=("lookahead_bias",),
        required_mechanism_keywords=(
            ("resample", "3d"),
            ("label='left'", "label=\"left\"", "closed='left'", "closed=\"left\"", "defaults to left", "default is left", "默认左", "左标签"),
        ),
        reference_notes=(
            "In pandas, `resample('3D')` defaults to `label='left', closed='left'`. `.last()` takes the close "
            "at the end of the 3-day window but stamps it with the first day's date. Forward-filling onto `df.index` "
            "exposes day t+2's close on day t."
        ),
    ),

    # 3. Double shift(1) across strategy module and Backtest.run
    SubtleAuditCase(
        case_id="sa_double_shift_03",
        title="Short-horizon 1-day return reversal with Backtest runner",
        description=(
            "日度超跌反弹策略：当 T 日收盘单日跌幅超过 -1.5% 时产生做多信号，要求在 T+1 日开盘买入，"
            "并在 T+2 日开盘平仓（仅持有 1 个交易日以捕获次日隔夜+日内反弹）。"
        ),
        code="""import pandas as pd
from src.backtest import Backtest

def build_one_day_bounce_signal(df: pd.DataFrame) -> pd.Series:
    ret_1d = df["close"].pct_change()
    raw_trigger = (ret_1d < -0.015).astype(float)
    # Shift by 1 bar so day T's close trigger is ready for day T+1 execution
    exec_signal = raw_trigger.shift(1).fillna(0.0)
    return exec_signal

def evaluate_bounce(df: pd.DataFrame) -> pd.DataFrame:
    signals = build_one_day_bounce_signal(df)
    # Backtest.run internally starts with: target_signals = signals.shift(1)
    bt = Backtest(slippage_rate=0.0005, commission_rate=0.0005, rebalance_tolerance=0.01)
    return bt.run(df, signals)
""",
        has_flaw=True,
        flaw_types=("logical_inconsistency",),
        required_mechanism_keywords=(
            ("shift", "double", "twice", "two", "t+2", "两次", "双重", "延迟"),
        ),
        reference_notes=(
            "`build_one_day_bounce_signal` calls `.shift(1)` and `Backtest.run` also calls `signals.shift(1)`. "
            "The signal is shifted twice, so a drop on day T is executed at day T+2 open instead of T+1 open."
        ),
    ),

    # 4. ewm(span=60).std() default min_periods=0 causing un-warmed max leverage on bar 2
    SubtleAuditCase(
        case_id="sa_ewm_warmup_04",
        title="60-day EWM volatility-targeted trend strategy",
        description=(
            "60 日指数加权趋势与波动率目标策略：要求在满 60 个交易日的预热期后再开始交易；"
            "当收盘价高于 60 日 EWM 均线时做多，并按 60 日 EWM 年化波动率将仓位缩放到 12% 目标波动率（上限 2.0 倍）。"
        ),
        code="""import numpy as np
import pandas as pd

def build_ewm_vol_target(df: pd.DataFrame) -> pd.Series:
    ema60 = df["close"].ewm(span=60, adjust=False).mean()
    ewm_vol60 = df["close"].pct_change().ewm(span=60, adjust=False).std() * np.sqrt(252)

    raw_trend = (df["close"] > ema60).astype(float)
    scale = (0.12 / ewm_vol60.replace(0, np.nan)).clip(lower=0.0, upper=2.0)
    # Rely on NaN during the 60-day warm-up before filling with 0
    weights = (raw_trend * scale).fillna(0.0)
    return weights
""",
        has_flaw=True,
        flaw_types=("logical_inconsistency",),
        required_mechanism_keywords=(
            ("ewm",),
            ("min_periods", "not nan", "non-nan", "bar 1", "bar 2", "from the start", "immediately", "不会产生nan", "默认从第"),
        ),
        reference_notes=(
            "Unlike `.rolling(60)`, `pandas.Series.ewm(span=60)` defaults to `min_periods=0` (producing values from bar 0/1). "
            "`raw_trend * scale` is non-NaN starting on bar 2, violating the 60-day warm-up requirement and trading on noisy 2-bar vol."
        ),
    ),

    # 5. Trailing stop re-anchored to rolling window instead of entry-since peak (MintEval silent bug)
    SubtleAuditCase(
        case_id="sa_trailing_reanchor_05",
        title="Breakout strategy with 6% high-water-mark trailing stop",
        description=(
            "趋势突破加移动止损策略：当收盘价突破过去 30 日最高收盘价（不含当日）时建仓做多；"
            "入场后跟踪持仓期间达到的最高收盘价（High-Water Mark since entry），一旦收盘价从该持仓最高点回落超过 6% 则平仓出场。"
        ),
        code="""import numpy as np
import pandas as pd

def build_trailing_stop_breakout(df: pd.DataFrame) -> pd.Series:
    prior_high30 = df["close"].rolling(30).max().shift(1)
    running_peak = df["close"].rolling(30).max()

    close_arr = df["close"].to_numpy()
    hi30_arr = prior_high30.to_numpy()
    peak_arr = running_peak.to_numpy()

    out = np.zeros(len(df), dtype=float)
    state = 0
    for i in range(len(df)):
        if state == 0:
            if not np.isnan(hi30_arr[i]) and close_arr[i] > hi30_arr[i]:
                state = 1
        else:
            if close_arr[i] < 0.94 * peak_arr[i]:
                state = 0
        out[i] = state
    return pd.Series(out, index=df.index)
""",
        has_flaw=True,
        flaw_types=("logical_inconsistency",),
        required_mechanism_keywords=(
            ("rolling(30)", "peak", "anchor", "entry", "window", "最高", "滚动", "重置"),
        ),
        reference_notes=(
            "`running_peak = df['close'].rolling(30).max()` computes a 30-bar rolling max rather than the highest close "
            "since entry. If a position is held >30 bars and drifts down slowly (<6% per 30 bars), `peak_arr[i]` ratchets "
            "downward and forgets the true post-entry high-water mark."
        ),
    ),

    # 6. Multi-step sizing chain where conviction boost bypasses max_leverage cap (Vidovich 2026 Defect 3)
    SubtleAuditCase(
        case_id="sa_sizing_chain_uncapped_06",
        title="Volatility-scaled momentum with volume-conviction multiplier",
        description=(
            "带成交量确信度加成的波动率目标策略：基础信号为 50 日均线，经 15% 目标波动率缩放后，"
            "在当日成交量高于 60 日均量时将仓位上调 1.35 倍；风控硬约束要求传入回测的最终目标杠杆率在任何交易日严禁超过 1.5 倍。"
        ),
        code="""import pandas as pd
from src.signals import compute_moving_average_signal, compute_volatility_scaled_signal

def apply_risk_budget(df: pd.DataFrame, base_sig: pd.Series, max_lev: float = 1.5) -> pd.Series:
    # Step 1: Volatility scaling enforces max_leverage = 1.5
    return compute_volatility_scaled_signal(df, base_sig, target_vol=0.15, max_leverage=max_lev)

def apply_volume_conviction(df: pd.DataFrame, sized_sig: pd.Series) -> pd.Series:
    # Step 2: Boost weight by 1.35x on high-volume days
    vol_ma60 = df["volume"].rolling(60).mean()
    boost = df["volume"].gt(vol_ma60).map({True: 1.35, False: 1.0})
    return sized_sig * boost

def build_strategy(df: pd.DataFrame) -> dict:
    base = compute_moving_average_signal(df, window=50)
    sized = apply_risk_budget(df, base, max_lev=1.5)
    final_signals = apply_volume_conviction(df, sized)
    return {"signals": final_signals, "slippage_rate": 0.0005, "commission_rate": 0.0005, "rebalance_tolerance": 0.05}
""",
        has_flaw=True,
        flaw_types=("logical_inconsistency",),
        required_mechanism_keywords=(
            ("1.35", "1.5", "clip", "cap", "exceed", "2.025", "上限", "超过"),
        ),
        reference_notes=(
            "`apply_risk_budget` caps leverage at 1.5, but `apply_volume_conviction` multiplies by 1.35 afterwards "
            "without re-applying `.clip(upper=1.5)`, pushing target leverage up to 1.5 * 1.35 = 2.025x."
        ),
    ),

    # 7. groupby('regime').transform('median') leaking future regime distribution
    SubtleAuditCase(
        case_id="sa_groupby_transform_leak_07",
        title="Regime-conditional relative volatility sizing",
        description=(
            "象限内相对波动率定仓策略：使用因果的 252 日滚动象限标签 `regime` 和 20 日滚动波动率 `rolling_vol`，"
            "当当日波动率低于所属象限的中位数水平时满仓做多 50 日均线信号，高于所属象限中位数时减半仓。"
        ),
        code="""import pandas as pd
from src.signals import compute_moving_average_signal

def build_regime_relative_vol_strategy(df: pd.DataFrame) -> pd.Series:
    base = compute_moving_average_signal(df, window=50).astype(float)
    # Compute median volatility within each market regime quadrant (Q1..Q4)
    regime_median_vol = df.groupby("regime")["rolling_vol"].transform("median")
    scale = (df["rolling_vol"] <= regime_median_vol).map({True: 1.0, False: 0.5})
    return (base * scale).fillna(0.0)
""",
        has_flaw=True,
        flaw_types=("lookahead_bias",),
        required_mechanism_keywords=(
            ("groupby", "transform"),
            ("future", "full", "entire", "lookahead", "全样本", "未来"),
        ),
        reference_notes=(
            "Even though `df['regime']` and `df['rolling_vol']` are causal rolling columns, "
            "`df.groupby('regime')['rolling_vol'].transform('median')` computes the median across all past AND future "
            "bars belonging to each regime in `df`."
        ),
    ),

    # 8. Gradual scale-out blocked by Backtest rebalance_tolerance=0.05
    SubtleAuditCase(
        case_id="sa_rebalance_tol_drift_08",
        title="Daily 3% linear scale-out execution under default Backtest tolerance",
        description=(
            "平滑减仓策略：当 20 日均线跌破 50 日均线后，要求从 100% 仓位以每天固定减仓 3%（0.97, 0.94, 0.91...）"
            "的节奏在每个交易日开盘逐步卖出平仓，避免单日冲击成本。使用默认 Backtest(rebalance_tolerance=0.05) 运行。"
        ),
        code="""import numpy as np
import pandas as pd
from src.backtest import Backtest

def build_gradual_scaleout(df: pd.DataFrame) -> pd.Series:
    bull = (df["close"].rolling(20).mean() > df["close"].rolling(50).mean()).to_numpy()
    weights = np.zeros(len(df), dtype=float)
    w = 0.0
    for i in range(len(df)):
        if bull[i]:
            w = 1.0
        else:
            w = max(0.0, round(w - 0.03, 4))
        weights[i] = w
    return pd.Series(weights, index=df.index)

def run_smooth_exit(df: pd.DataFrame) -> pd.DataFrame:
    sig = build_gradual_scaleout(df)
    bt = Backtest(rebalance_tolerance=0.05)
    return bt.run(df, sig)
""",
        has_flaw=True,
        flaw_types=("logical_inconsistency", "friction_erosion"),
        required_mechanism_keywords=(
            ("rebalance_tolerance", "0.05", "0.03", "tolerance", "容忍带"),
        ),
        reference_notes=(
            "The strategy decrements target weight by 0.03 (3%) per day, but `Backtest(rebalance_tolerance=0.05)` "
            "suppresses any non-zero rebalance where `abs(drift_ratio) < 0.05`. Thus daily 3% reductions are skipped "
            "on bars where drift stays under 5%, breaking the daily linear execution requirement."
        ),
    ),

    # 9. Silent zero fallback on missing external filter series (Vidovich 2026 Defect 5)
    SubtleAuditCase(
        case_id="sa_silent_zero_merge_09",
        title="Macro stress filter alignment with fillna(0.0) fallback",
        description=(
            "宏观压力过滤趋势策略：对齐外部信用利差压力指数 `stress_z`（正常值域 -2.0 到 +4.0，大于 1.0 代表市场处于高风险压力状态需空仓）；"
            "仅当 `stress_z <= 0.5` 且收盘价高于 50 日均线时才允许做多；若某日外部压力数据缺失，出于风控原则严禁盲目开仓。"
        ),
        code="""import pandas as pd
from src.signals import compute_moving_average_signal

def build_macro_filtered_trend(df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.Series:
    trend = compute_moving_average_signal(df, window=50)
    # Align macro stress series to equity trading calendar
    stress_z = macro_df["stress_z"].reindex(df.index).fillna(0.0)
    safe_regime = (stress_z <= 0.5).astype(int)
    return (trend & safe_regime).astype(int)
""",
        has_flaw=True,
        flaw_types=("logical_inconsistency",),
        required_mechanism_keywords=(
            ("fillna(0", "0.0", "missing", "reindex", "缺失"),
            ("0.5", "safe", "allow", "做多", "开仓"),
        ),
        reference_notes=(
            "`reindex(df.index).fillna(0.0)` maps missing macro dates to `stress_z = 0.0`, which satisfies "
            "`stress_z <= 0.5` (`safe_regime = 1`), silently treating missing risk data as a safe market and allowing long entries."
        ),
    ),

    # 10. Asymmetric buy vs sell sizing in Backtest cash accounting with high slippage
    SubtleAuditCase(
        case_id="sa_sell_overshoot_10",
        title="Target weight down-scaling with integer share truncation in Backtest",
        description=(
            "杠杆再平衡模块审阅：检查下面组合再平衡函数中，当目标仓位从 1.0 下调至 0.5 时，"
            "实际成交后的持仓市值比例是否与含滑点/佣金的买入公式对称，以及在 signal < 0（做空信号）时该引擎会如何处理。"
        ),
        code="""import pandas as pd

def execute_rebalance(open_price: float, cash: float, pos: int, signal: float, slip: float = 0.001, comm: float = 0.001):
    cost_per_share = open_price * (1 + slip) * (1 + comm)
    v_open = cash + pos * open_price
    if pd.isna(signal):
        trade_shares = 0
    elif signal == 0:
        trade_shares = -pos
    else:
        target_value = v_open * signal
        current_value = pos * open_price
        delta_value = target_value - current_value
        if delta_value > 0:
            trade_shares = min(delta_value // cost_per_share, cash // cost_per_share)
        elif delta_value < 0:
            # Sell shares to reduce position toward target_value
            desired_sell = abs(delta_value) // open_price
            trade_shares = -min(pos, desired_sell)
        else:
            trade_shares = 0
    return trade_shares
""",
        has_flaw=True,
        flaw_types=("logical_inconsistency",),
        required_mechanism_keywords=(
            ("short", "negative", "signal < 0", "-min(pos", "做空", "负", "sell", "cost_per_share", "open_price"),
        ),
        reference_notes=(
            "When `signal < 0` (short target) and `pos == 0`, `delta_value < 0`, so `trade_shares = -min(0, desired_sell) = 0` "
            "(silently fails to open any short position), and if `pos > 0` it clips sell shares at `pos` (only flattening to 0). "
            "Additionally, selling divides by `open_price` while buying divides by `cost_per_share`."
        ),
    ),

    # 11. Clean Control 1: Tricky multi-timeframe weekly resample with proper right-labeling
    SubtleAuditCase(
        case_id="sa_clean_subtle_11",
        title="Causal weekly-to-daily multi-timeframe trend filter with explicit fill",
        description=(
            "周线-日线双周期趋势策略：以每周五收盘构建周线 K 线（label='right', closed='right'），"
            "计算 12 周简单均线并向后对齐到日线索引；仅当已完成的周线收盘高于 12 周均线且日线收盘高于 20 日均线时持仓，其余时刻空仓（0.0）。"
        ),
        code="""import pandas as pd

def build_clean_mtf_strategy(df: pd.DataFrame) -> dict:
    weekly_close = df["close"].resample("W-FRI", label="right", closed="right").last()
    weekly_ma12 = weekly_close.rolling(12, min_periods=12).mean()
    weekly_bull = (weekly_close > weekly_ma12).astype(float)

    # Forward-fill completed Friday signal onto subsequent daily bars; fill warm-up NaNs with 0.0
    daily_weekly_filter = weekly_bull.reindex(df.index, method="ffill").fillna(0.0)
    daily_ma20 = df["close"].rolling(20, min_periods=20).mean()
    daily_bull = (df["close"] > daily_ma20).astype(float).fillna(0.0)

    signals = (daily_weekly_filter * daily_bull).fillna(0.0)
    return {
        "signals": signals,
        "slippage_rate": 0.0005,
        "commission_rate": 0.0005,
        "rebalance_tolerance": 0.05,
    }
""",
        has_flaw=False,
        flaw_types=(),
        required_mechanism_keywords=(),
        reference_notes="Clean control: uses `W-FRI` with `label='right', closed='right'`, `min_periods`, and explicit `.fillna(0.0)`.",
    ),

    # 12. Clean Control 2: Tricky stateful high-water-mark trailing stop loop
    SubtleAuditCase(
        case_id="sa_clean_subtle_12",
        title="Stateful Donchian entry with post-entry high-water-mark trailing stop",
        description=(
            "突破入场加持仓最高点移动止损策略：当收盘价突破过去 40 日最高收盘价（shift(1)）时入场做多；"
            "入场后持续更新该笔交易自入场以来的最高收盘价 `entry_peak`，当收盘价跌破 `0.93 * entry_peak` 时平仓归零。"
        ),
        code="""import numpy as np
import pandas as pd

def build_clean_hwm_stop(df: pd.DataFrame) -> dict:
    prior_max40 = df["close"].rolling(40, min_periods=40).max().shift(1).to_numpy()
    close_arr = df["close"].to_numpy()
    out = np.zeros(len(df), dtype=float)

    state = 0
    entry_peak = 0.0
    for i in range(len(df)):
        c = close_arr[i]
        if state == 0:
            if not np.isnan(prior_max40[i]) and c > prior_max40[i]:
                state = 1
                entry_peak = c
        else:
            if c > entry_peak:
                entry_peak = c
            if c < 0.93 * entry_peak:
                state = 0
                entry_peak = 0.0
        out[i] = float(state)

    return {
        "signals": pd.Series(out, index=df.index),
        "slippage_rate": 0.0005,
        "commission_rate": 0.0005,
        "rebalance_tolerance": 0.05,
    }
""",
        has_flaw=False,
        flaw_types=(),
        required_mechanism_keywords=(),
        reference_notes="Clean control: properly tracks `entry_peak` per trade, resets on exit, uses `shift(1)` on entry channel, and returns 0.0 during warm-up.",
    ),
]


def _parse_audit_json(raw_text: str) -> Dict[str, Any]:
    text = raw_text.strip()
    if "```" in text:
        lines = [ln for ln in text.splitlines() if not ln.strip().startswith("```")]
        text = "\n".join(lines).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


def score_subtle_audit_case(case: SubtleAuditCase, raw_response: str) -> SubtleAuditEvalResult:
    gt_set = set(case.flaw_types)
    try:
        data = _parse_audit_json(raw_response)
        raw_types = data.get("flaw_types", [])
        pred_set = {str(t).strip() for t in (raw_types if isinstance(raw_types, list) else []) if str(t).strip() in VALID_FLAW_TYPES}
        has_flaw_pred = bool(data.get("has_flaw", len(pred_set) > 0))
        if not has_flaw_pred:
            pred_set = set()
        flawed_location = str(data.get("flawed_location", ""))
        explanation = str(data.get("explanation", ""))
    except Exception as e:
        pred_set = set()
        has_flaw_pred = False
        flawed_location = ""
        explanation = f"Parse error: {e}"

    binary_correct = 1.0 if (has_flaw_pred == case.has_flaw) else 0.0
    tp = len(pred_set & gt_set)
    if len(pred_set) == 0 and len(gt_set) == 0:
        f1 = 1.0 if not has_flaw_pred else 0.0
    elif len(pred_set) == 0 or len(gt_set) == 0:
        f1 = 0.0
    else:
        p = tp / len(pred_set)
        r = tp / len(gt_set)
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    combined_text = f"{flawed_location} {explanation}".lower()
    if not case.has_flaw:
        mechanism_hit = (not has_flaw_pred) and (len(pred_set) == 0)
    else:
        mechanism_hit = bool(
            has_flaw_pred
            and all(
                any(kw.lower() in combined_text for kw in group)
                for group in case.required_mechanism_keywords
            )
        )

    strict_pass = bool((binary_correct == 1.0) and (f1 > 0.0) and mechanism_hit)

    return SubtleAuditEvalResult(
        case_id=case.case_id,
        has_flaw_gt=case.has_flaw,
        flaw_types_gt=tuple(sorted(gt_set)),
        has_flaw_pred=has_flaw_pred,
        flaw_types_pred=tuple(sorted(pred_set)),
        binary_correct=binary_correct,
        category_f1=f1,
        mechanism_hit=mechanism_hit,
        strict_pass=strict_pass,
        flawed_location=flawed_location,
        explanation=explanation,
        raw_response=raw_response,
    )


# =============================================================================
# Part 2: High-Freedom Stateful Implementation Cases & Behavioral Scorers
# =============================================================================

@dataclass(frozen=True)
class HighFreedomImplCase:
    case_id: str
    title: str
    state_complexity: str  # "medium_state" | "long_state_ratchet" | "multi_tf_sizing"
    prompt: str
    golden_fn: Callable[[pd.DataFrame], Dict[str, Any]]
    max_leverage: float = 1.0


@dataclass
class HighFreedomEvalResult:
    case_id: str
    state_complexity: str
    exec_pass: bool
    no_lookahead: bool
    nan_hygiene: bool
    bounds_ok: bool
    active_action_match: float
    trade_f1: float
    friction_match: float
    sharpe_ref: float
    sharpe_agent: float
    passed: bool
    error_message: str = ""
    lookahead_reason: str = ""
    extracted_code: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_active_action_match(
    sig_ref: pd.Series,
    sig_agent: pd.Series,
    atol: float = 0.02,
) -> float:
    """
    MintEval Active-Bar ActionMatch:
    Fraction of active bars (where either reference or agent targets a non-zero position)
    on which `|sig_agent - sig_ref| <= atol`.
    Prevents flat periods from inflating agreement scores.
    """
    r = sig_ref.fillna(0.0).to_numpy(dtype=float)
    a = sig_agent.reindex(sig_ref.index).fillna(0.0).to_numpy(dtype=float)
    active = (np.abs(r) > 1e-6) | (np.abs(a) > 1e-6)
    if not np.any(active):
        return 1.0
    matched = np.abs(r[active] - a[active]) <= atol
    return float(np.mean(matched))


def _extract_round_trips(sig: pd.Series) -> List[Tuple[int, int]]:
    """Extract (entry_bar_idx, exit_bar_idx) intervals where signal > 1e-6."""
    arr = (np.abs(sig.fillna(0.0).to_numpy(dtype=float)) > 1e-6).astype(int)
    trips: List[Tuple[int, int]] = []
    in_trade = False
    start = 0
    for i, v in enumerate(arr):
        if v == 1 and not in_trade:
            in_trade = True
            start = i
        elif v == 0 and in_trade:
            in_trade = False
            trips.append((start, i - 1))
    if in_trade:
        trips.append((start, len(arr) - 1))
    return trips


def compute_trade_f1(
    sig_ref: pd.Series,
    sig_agent: pd.Series,
    tol_bars: int = 1,
) -> float:
    """
    Round-trip Trade F1 allowing `+/- tol_bars` boundary tolerance on entry and exit bars
    (so minor `min_periods = W` vs `W-1` conventions are not penalized as total failures).
    """
    ref_trips = _extract_round_trips(sig_ref)
    agent_trips = _extract_round_trips(sig_agent.reindex(sig_ref.index))
    if not ref_trips and not agent_trips:
        return 1.0
    if not ref_trips or not agent_trips:
        return 0.0

    used_agent = set()
    matched = 0
    for r_en, r_ex in ref_trips:
        for j, (a_en, a_ex) in enumerate(agent_trips):
            if j in used_agent:
                continue
            if abs(a_en - r_en) <= tol_bars and abs(a_ex - r_ex) <= tol_bars:
                used_agent.add(j)
                matched += 1
                break

    precision = matched / len(agent_trips)
    recall = matched / len(ref_trips)
    return float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0


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


def _safe_sharpe(ledger: pd.DataFrame) -> float:
    m = compute_strategy_metrics(ledger)
    v = float(m.get("sharpe_ratio", 0.0))
    return 0.0 if (math.isnan(v) or math.isinf(v)) else v


def evaluate_high_freedom_case(
    case: HighFreedomImplCase,
    raw_response: str,
    df: pd.DataFrame,
) -> HighFreedomEvalResult:
    ref_raw = case.golden_fn(df.copy())
    sig_ref = ref_raw["signals"].reindex(df.index).fillna(0.0).astype(float)
    bt_ref = Backtest(
        slippage_rate=float(ref_raw.get("slippage_rate", 0.0005)),
        commission_rate=float(ref_raw.get("commission_rate", 0.0005)),
        rebalance_tolerance=float(ref_raw.get("rebalance_tolerance", 0.05)),
    )
    sharpe_ref = _safe_sharpe(bt_ref.run(df, sig_ref))

    extracted = extract_python_code(raw_response)
    try:
        agent_fn = compile_build_strategy(extracted)
        raw_out = agent_fn(df.copy())
        if isinstance(raw_out, pd.Series):
            raw_out = {"signals": raw_out}
        if not isinstance(raw_out, dict) or "signals" not in raw_out:
            raise ValueError("build_strategy must return a dict containing 'signals'")
        raw_sig = raw_out["signals"]
        if not isinstance(raw_sig, pd.Series):
            raw_sig = pd.Series(raw_sig, index=df.index)
        raw_sig = raw_sig.reindex(df.index)
    except Exception as e:
        return HighFreedomEvalResult(
            case_id=case.case_id,
            state_complexity=case.state_complexity,
            exec_pass=False,
            no_lookahead=False,
            nan_hygiene=False,
            bounds_ok=False,
            active_action_match=0.0,
            trade_f1=0.0,
            friction_match=0.0,
            sharpe_ref=sharpe_ref,
            sharpe_agent=0.0,
            passed=False,
            error_message=str(e),
            extracted_code=extracted,
        )

    # Property 1: NaN hygiene (does the raw returned Series contain NaNs after warm-up that would lock Backtest?)
    nan_hygiene = bool(not raw_sig.iloc[120:].isna().any())
    sig_agent = raw_sig.fillna(0.0).astype(float)

    # Property 2: Bounds check
    arr_agent = sig_agent.to_numpy(dtype=float)
    bounds_ok = bool(
        not np.isinf(arr_agent).any()
        and (arr_agent >= -1e-6).all()
        and (arr_agent <= case.max_leverage + 1e-6).all()
    )

    # Property 3: Causality check
    try:
        no_lookahead, lookahead_reason = check_no_lookahead(agent_fn, df)
    except Exception as e:
        no_lookahead = False
        lookahead_reason = str(e)

    # Behavioral equivalence metrics
    active_match = compute_active_action_match(sig_ref, sig_agent, atol=0.02)
    t_f1 = compute_trade_f1(sig_ref, sig_agent, tol_bars=1)

    slip_a = float(raw_out.get("slippage_rate", 0.0005))
    comm_a = float(raw_out.get("commission_rate", 0.0005))
    tol_a = float(raw_out.get("rebalance_tolerance", 0.05))
    friction_match = (
        1.0
        if (
            abs(slip_a - float(ref_raw.get("slippage_rate", 0.0005))) <= 1e-7
            and abs(comm_a - float(ref_raw.get("commission_rate", 0.0005))) <= 1e-7
            and abs(tol_a - float(ref_raw.get("rebalance_tolerance", 0.05))) <= 1e-7
        )
        else 0.0
    )

    bt_agent = Backtest(slippage_rate=slip_a, commission_rate=comm_a, rebalance_tolerance=tol_a)
    sharpe_agent = _safe_sharpe(bt_agent.run(df, sig_agent))

    passed = bool(
        no_lookahead
        and nan_hygiene
        and bounds_ok
        and (active_match >= 0.95)
        and (t_f1 >= 0.90)
        and (friction_match == 1.0)
    )

    return HighFreedomEvalResult(
        case_id=case.case_id,
        state_complexity=case.state_complexity,
        exec_pass=True,
        no_lookahead=no_lookahead,
        nan_hygiene=nan_hygiene,
        bounds_ok=bounds_ok,
        active_action_match=active_match,
        trade_f1=t_f1,
        friction_match=friction_match,
        sharpe_ref=sharpe_ref,
        sharpe_agent=sharpe_agent,
        passed=passed,
        lookahead_reason=lookahead_reason,
        extracted_code=extracted,
    )


class GoldenSubtleAuditClient(BaseLLMClient):
    def __init__(self):
        self.usage = LLMUsage()
        self._by_id = {c.case_id: c for c in SUBTLE_AUDIT_CASES}

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        self.usage.calls += 1
        for cid, c in self._by_id.items():
            if cid in prompt:
                kw_str = " ".join(g[0] for g in c.required_mechanism_keywords)
                return json.dumps({
                    "has_flaw": c.has_flaw,
                    "flaw_types": list(c.flaw_types),
                    "flawed_location": kw_str,
                    "explanation": f"{c.reference_notes} ({kw_str})",
                })
        return json.dumps({"has_flaw": False, "flaw_types": [], "flawed_location": "none", "explanation": ""})


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
