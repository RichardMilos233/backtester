from __future__ import annotations

import inspect
import math
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.backtest import Backtest
from src.metrics import compute_strategy_metrics
from agent.llm import BaseLLMClient, LLMUsage
from agent.layer1.schema import SystematicCase, SystematicEvalResult
from agent.layer1.sandbox import (
    extract_python_code,
    compile_build_strategy,
    normalize_strategy_output,
    check_no_lookahead,
)
from agent.layer1.cases.systematic_cases import SYSTEMATIC_CASES

SYSTEMATIC_SYSTEM_PROMPT = """You are a quantitative strategy developer.
Translate the user's natural-language strategy specification into a causal Python function named `build_strategy(df)`.

Environment & Available Imports (already imported or importable from `src.signals`):
- `pandas as pd`, `numpy as np`
- `from src.signals import compute_moving_average_signal, compute_macd_signal, compute_volatility_scaled_signal, compute_mean_reversion_signal, compute_breakout_signal, compute_rsi_signal, combine_signals`

Input `df` columns:
- `'open'`, `'high'`, `'low'`, `'close'`, `'volume'`, `'return'`, `'rolling_vol'`, `'regime'`
- Indexed by `pd.DatetimeIndex` sorted ascending.

Contract for `build_strategy(df)`:
- Must return a dict:
  {
      "signals": pd.Series,          # target weight indexed by df.index (causal at end of day t; Backtest shifts by 1 bar internally)
      "slippage_rate": float,        # default 0.0005 unless specified
      "commission_rate": float,      # default 0.0005 unless specified
      "rebalance_tolerance": float,  # default 0.05 unless specified
  }
- IMPORTANT: Do NOT call `.shift(1)` on the final `signals` Series because `Backtest.run()` already shifts `signals` by 1 bar to execute at next day's open.
- Output ONLY a single ```python ... ``` code block containing `def build_strategy(df):`.
"""


def build_systematic_prompt(case: SystematicCase) -> str:
    return (
        f"Case ID: {case.case_id}\n"
        f"Title: {case.title}\n"
        f"Category: {case.category}\n\n"
        f"Specification:\n{case.prompt}\n\n"
        f"Write the Python function `build_strategy(df)` now."
    )


def _safe_sharpe(ledger: pd.DataFrame) -> float:
    metrics = compute_strategy_metrics(ledger)
    val = float(metrics.get("sharpe_ratio", 0.0))
    if math.isnan(val) or math.isinf(val):
        return 0.0
    return val


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.allclose(a, b, atol=1e-6):
        return 1.0
    std_a, std_b = float(np.std(a)), float(np.std(b))
    if std_a < 1e-12 or std_b < 1e-12:
        return 0.0
    c = float(np.corrcoef(a, b)[0, 1])
    return 0.0 if math.isnan(c) else c


def evaluate_systematic_case(
    case: SystematicCase,
    raw_response: str,
    df: pd.DataFrame,
) -> SystematicEvalResult:
    # 1. Run golden reference
    ref_out = normalize_strategy_output(case.golden_fn(df.copy()), df)
    bt_ref = Backtest(
        slippage_rate=ref_out["slippage_rate"],
        commission_rate=ref_out["commission_rate"],
        rebalance_tolerance=ref_out["rebalance_tolerance"],
    )
    ledger_ref = bt_ref.run(df, ref_out["signals"])
    sharpe_ref = _safe_sharpe(ledger_ref)

    extracted = extract_python_code(raw_response)

    # 2. Compile & execute agent strategy
    try:
        agent_fn = compile_build_strategy(extracted)
        agent_out = normalize_strategy_output(agent_fn(df.copy()), df)
    except Exception as e:
        return SystematicEvalResult(
            case_id=case.case_id,
            category=case.category,
            exec_pass=False,
            no_lookahead=False,
            signal_agreement=0.0,
            position_agreement=0.0,
            position_corr=0.0,
            friction_match=0.0,
            sharpe_ref=sharpe_ref,
            sharpe_agent=0.0,
            sharpe_abs_error=abs(sharpe_ref),
            passed=False,
            error_message=str(e),
            extracted_code=extracted,
            raw_response=raw_response,
        )

    # 3. Truncation causality test (no lookahead)
    try:
        no_lookahead, lookahead_reason = check_no_lookahead(agent_fn, df)
    except Exception as e:
        no_lookahead = False
        lookahead_reason = f"Causality check error: {e}"

    # 4. Run agent backtest and compare signals, positions, friction, and Sharpe
    try:
        bt_agent = Backtest(
            slippage_rate=agent_out["slippage_rate"],
            commission_rate=agent_out["commission_rate"],
            rebalance_tolerance=agent_out["rebalance_tolerance"],
        )
        ledger_agent = bt_agent.run(df, agent_out["signals"])
        sharpe_agent = _safe_sharpe(ledger_agent)
    except Exception as e:
        return SystematicEvalResult(
            case_id=case.case_id,
            category=case.category,
            exec_pass=False,
            no_lookahead=no_lookahead,
            signal_agreement=0.0,
            position_agreement=0.0,
            position_corr=0.0,
            friction_match=0.0,
            sharpe_ref=sharpe_ref,
            sharpe_agent=0.0,
            sharpe_abs_error=abs(sharpe_ref),
            passed=False,
            lookahead_reason=lookahead_reason,
            error_message=f"Backtest error: {e}",
            extracted_code=extracted,
            raw_response=raw_response,
        )

    sig_ref_arr = ref_out["signals"].to_numpy(dtype=float)
    sig_agent_arr = agent_out["signals"].to_numpy(dtype=float)
    signal_agreement = float(np.mean(np.abs(sig_agent_arr - sig_ref_arr) <= 1e-4))

    pos_ref_arr = ledger_ref["position"].to_numpy(dtype=float)
    pos_agent_arr = ledger_agent["position"].to_numpy(dtype=float)
    position_agreement = float(np.mean(np.abs(pos_agent_arr - pos_ref_arr) <= 1.0))
    position_corr = _safe_corr(pos_agent_arr, pos_ref_arr)

    friction_ok = (
        abs(agent_out["slippage_rate"] - ref_out["slippage_rate"]) <= 1e-7
        and abs(agent_out["commission_rate"] - ref_out["commission_rate"]) <= 1e-7
        and abs(agent_out["rebalance_tolerance"] - ref_out["rebalance_tolerance"]) <= 1e-7
    )
    friction_match = 1.0 if friction_ok else 0.0
    sharpe_abs_error = abs(sharpe_agent - sharpe_ref)

    passed = bool(
        no_lookahead
        and (position_agreement >= 0.98)
        and (friction_match == 1.0)
    )

    return SystematicEvalResult(
        case_id=case.case_id,
        category=case.category,
        exec_pass=True,
        no_lookahead=no_lookahead,
        signal_agreement=signal_agreement,
        position_agreement=position_agreement,
        position_corr=position_corr,
        friction_match=friction_match,
        sharpe_ref=sharpe_ref,
        sharpe_agent=sharpe_agent,
        sharpe_abs_error=sharpe_abs_error,
        passed=passed,
        lookahead_reason=lookahead_reason,
        error_message="",
        extracted_code=extracted,
        raw_response=raw_response,
    )


def evaluate_systematic_suite(
    llm_client: BaseLLMClient,
    df: pd.DataFrame,
    cases: Optional[List[SystematicCase]] = None,
) -> List[SystematicEvalResult]:
    suite = cases if cases is not None else SYSTEMATIC_CASES
    results: List[SystematicEvalResult] = []
    for case in suite:
        prompt = build_systematic_prompt(case)
        raw = llm_client.generate(prompt, system_prompt=SYSTEMATIC_SYSTEM_PROMPT)
        results.append(evaluate_systematic_case(case, raw, df))
    return results


def summarize_systematic_results(results: List[SystematicEvalResult]) -> Dict[str, Any]:
    if not results:
        return {}
    return {
        "total_cases": len(results),
        "exec_pass_rate": float(np.mean([1.0 if r.exec_pass else 0.0 for r in results])),
        "no_lookahead_rate": float(np.mean([1.0 if r.no_lookahead else 0.0 for r in results])),
        "mean_signal_agreement": float(np.mean([r.signal_agreement for r in results])),
        "mean_position_agreement": float(np.mean([r.position_agreement for r in results])),
        "mean_position_corr": float(np.mean([r.position_corr for r in results])),
        "friction_match_rate": float(np.mean([r.friction_match for r in results])),
        "mean_sharpe_abs_error": float(np.mean([r.sharpe_abs_error for r in results])),
        "pass_rate": float(np.mean([1.0 if r.passed else 0.0 for r in results])),
    }


class GoldenSystematicLLMClient(BaseLLMClient):
    """Oracle client that emits the exact golden reference code for each SystematicCase."""

    def __init__(self, cases: Optional[List[SystematicCase]] = None):
        self.usage = LLMUsage()
        self._by_id = {c.case_id: c for c in (cases or SYSTEMATIC_CASES)}

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        self.usage.calls += 1
        for cid, case in self._by_id.items():
            if f"Case ID: {cid}" in prompt:
                src_lines = inspect.getsource(case.golden_fn).splitlines()
                # Rename `def _golden_sys_...(df: ...)` to `def build_strategy(df):`
                src_lines[0] = "def build_strategy(df):"
                body = "\n".join(src_lines)
                return f"```python\n{body}\n```"
        return "```python\ndef build_strategy(df):\n    return {'signals': df['close'] * 0}\n```"
