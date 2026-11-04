"""
Layer 1B scorer: bar-by-bar position match against the golden implementation.

`evaluate_high_freedom_case` runs the model's `build_strategy(df)` in the sandbox and
checks, in order: it executes, no look-ahead (truncation + future-price perturbation),
no NaN after warm-up, weights inside [0, max_leverage], friction parameters match,
ActiveActionMatch >= 0.95 on bars where either side holds a position, and round-trip
TradeF1 >= 0.90. All must hold for `passed`.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

from src.backtest import Backtest
from src.metrics import compute_strategy_metrics
from benchmark.layer1_code.impl.tasks.schema import HighFreedomImplCase
from benchmark.layer1_code.impl.scoring.sandbox import (
    extract_python_code,
    compile_build_strategy,
    check_no_lookahead,
)


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

