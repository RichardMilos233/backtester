"""
Layer 1A (identify bad alpha), subtle-case track.

Cross-module / implicit-default audit cases plus the mechanism-keyword scorer
(`score_subtle_audit_case`). Classic flaw cases live in `bad_alpha_cases.py`;
the 20-case canonical audit suite that mixes both is assembled in
`benchmark/layer1_code/canonical.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import inspect
import json
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

from src.backtest import Backtest
from benchmark.common.llm import BaseLLMClient, LLMUsage
from benchmark.layer1_code.schema import VALID_FLAW_TYPES


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

