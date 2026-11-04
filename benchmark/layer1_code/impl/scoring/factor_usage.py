from __future__ import annotations

import ast
import inspect
import json
import math
import types
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

import src.signals as signals_mod
import src.regimes as regimes_mod
from src.signals import (
    compute_moving_average_signal,
    compute_macd_signal,
    compute_volatility_scaled_signal,
    compute_mean_reversion_signal,
    compute_breakout_signal,
    compute_rsi_signal,
    combine_signals,
)
from src.regimes import classify_market_regimes
from benchmark.common.llm import BaseLLMClient
from benchmark.layer1_code.schema import SystematicCase, FactorUsageEvalResult
from benchmark.layer1_code.impl.scoring.sandbox import extract_python_code
from benchmark.layer1_code.impl.tasks.systematic_cases import SYSTEMATIC_CASES
from benchmark.layer1_code.impl.scoring.systematic import SYSTEMATIC_SYSTEM_PROMPT, build_systematic_prompt

KNOWN_FACTOR_FUNCTIONS: Dict[str, Callable[..., Any]] = {
    "compute_moving_average_signal": compute_moving_average_signal,
    "compute_macd_signal": compute_macd_signal,
    "compute_volatility_scaled_signal": compute_volatility_scaled_signal,
    "compute_mean_reversion_signal": compute_mean_reversion_signal,
    "compute_breakout_signal": compute_breakout_signal,
    "compute_rsi_signal": compute_rsi_signal,
    "combine_signals": combine_signals,
}

MANUAL_WHEEL_METHODS = {"rolling", "ewm"}


def _inspect_ast_factor_usage(code: str) -> Tuple[Tuple[str, ...], Tuple[str, ...], List[str]]:
    """
    Parse Python source code to extract:
      1. Factor functions from `src.signals` referenced in `ast.Call` nodes
      2. Manual `.rolling(...)` / `.ewm(...)` calls
      3. Static signature errors if the code cannot execute at runtime
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return (), (), [f"SyntaxError: {e}"]

    ast_factor_calls: List[str] = []
    manual_ops: List[str] = []
    static_sig_errors: List[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        fn_name: Optional[str] = None
        if isinstance(node.func, ast.Name) and node.func.id in KNOWN_FACTOR_FUNCTIONS:
            fn_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr in KNOWN_FACTOR_FUNCTIONS:
                fn_name = node.func.attr
            elif node.func.attr in MANUAL_WHEEL_METHODS:
                manual_ops.append(node.func.attr)

        if fn_name is not None:
            ast_factor_calls.append(fn_name)
            real_fn = KNOWN_FACTOR_FUNCTIONS[fn_name]
            sig = inspect.signature(real_fn)
            valid_params = set(sig.parameters.keys())
            for kw in node.keywords:
                if kw.arg is not None and kw.arg not in valid_params:
                    static_sig_errors.append(
                        f"{fn_name}: unexpected keyword argument '{kw.arg}' (valid: {sorted(valid_params)})"
                    )

    return tuple(dict.fromkeys(ast_factor_calls)), tuple(dict.fromkeys(manual_ops)), static_sig_errors


def _run_with_factor_spies(
    code: str,
    df: pd.DataFrame,
) -> Tuple[bool, Tuple[str, ...], List[str]]:
    """
    Execute `build_strategy(df)` with runtime spy wrappers around all `src.signals` functions.
    Each spy records the call, validates `inspect.signature` binding and runtime argument types
    (`DataFrame` as first arg for `compute_*`, `list`/`tuple` for `combine_signals`), and returns
    a safe zero `pd.Series` fallback if a call has invalid arguments so subsequent calls in
    `build_strategy` are also reached and audited.
    """
    clean_code = extract_python_code(code)
    runtime_calls: List[str] = []
    sig_errors: List[str] = []

    def _make_spy(fn_name: str, real_fn: Callable[..., Any]) -> Callable[..., Any]:
        sig = inspect.signature(real_fn)

        def _spy(*args: Any, **kwargs: Any) -> Any:
            runtime_calls.append(fn_name)
            try:
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
            except TypeError as e:
                sig_errors.append(f"{fn_name}: {e}")
                return pd.Series(0, index=df.index, dtype=int)

            if fn_name == "combine_signals":
                sigs_arg = bound.arguments.get("signals")
                if not isinstance(sigs_arg, (list, tuple)) or len(sigs_arg) == 0:
                    sig_errors.append(
                        f"{fn_name}: first arg 'signals' must be a non-empty list/tuple, got {type(sigs_arg).__name__}"
                    )
                    return pd.Series(0, index=df.index, dtype=int)
            else:
                df_arg = bound.arguments.get("df")
                if not isinstance(df_arg, pd.DataFrame):
                    sig_errors.append(
                        f"{fn_name}: first arg 'df' must be a pandas DataFrame, got {type(df_arg).__name__}"
                    )
                    return pd.Series(0, index=df.index, dtype=int)

                if fn_name == "compute_volatility_scaled_signal":
                    base_arg = bound.arguments.get("base_signal")
                    if not isinstance(base_arg, pd.Series):
                        sig_errors.append(
                            f"{fn_name}: second arg 'base_signal' must be a pandas Series, got {type(base_arg).__name__}"
                        )
                        return pd.Series(0.0, index=df.index, dtype=float)

            try:
                return real_fn(*args, **kwargs)
            except Exception as e:
                sig_errors.append(f"{fn_name} runtime error: {e}")
                return pd.Series(0, index=df.index, dtype=int)

        return _spy

    spy_funcs = {name: _make_spy(name, fn) for name, fn in KNOWN_FACTOR_FUNCTIONS.items()}
    spy_signals_mod = types.SimpleNamespace(**spy_funcs)

    env: Dict[str, Any] = {
        "__builtins__": __builtins__,
        "pd": pd,
        "np": np,
        "math": math,
        "signals": spy_signals_mod,
        "regimes": regimes_mod,
        "classify_market_regimes": classify_market_regimes,
        **spy_funcs,
    }

    try:
        exec(compile(clean_code, "<agent_factor_spy>", "exec"), env, env)
        # Re-inject spies in case the code ran `from src.signals import ...` which overwrote globals
        for name, spy_fn in spy_funcs.items():
            env[name] = spy_fn
        env["signals"] = spy_signals_mod

        fn = env.get("build_strategy")
        if not callable(fn):
            return False, (), ["Code did not define a callable `build_strategy(df)`"]
        fn(df.copy())
        return True, tuple(dict.fromkeys(runtime_calls)), sig_errors
    except Exception as e:
        return False, tuple(dict.fromkeys(runtime_calls)), sig_errors + [f"Execution error: {e}"]


def evaluate_factor_usage_case(
    case: SystematicCase,
    raw_response: str,
    df: pd.DataFrame,
) -> FactorUsageEvalResult:
    extracted = extract_python_code(raw_response)
    ast_calls, manual_ops, static_errors = _inspect_ast_factor_usage(extracted)
    compile_ok, runtime_calls, runtime_sig_errors = _run_with_factor_spies(extracted, df)

    expected = tuple(case.expected_factor_calls)
    expected_set = set(expected)

    observed_calls = runtime_calls if runtime_calls else ast_calls
    observed_set = set(observed_calls)

    if len(expected_set) == 0:
        recall = 1.0
    else:
        recall = float(len(observed_set & expected_set) / len(expected_set))

    if len(expected_set) == 0:
        no_reinvented = True
    else:
        no_reinvented = bool(recall == 1.0 and len(manual_ops) == 0)

    all_sig_errors = list(dict.fromkeys(runtime_sig_errors + static_errors))
    if len(expected_set) > 0:
        sig_valid = bool(compile_ok and len(observed_set) > 0 and len(all_sig_errors) == 0)
    else:
        sig_valid = bool(compile_ok and len(all_sig_errors) == 0)

    notes_parts = []
    if len(expected_set) > 0 and len(manual_ops) > 0:
        notes_parts.append(f"Manual rolling/ewm used ({list(manual_ops)}) when library factor was expected")
    if len(expected_set) > 0 and recall < 1.0:
        missing = sorted(expected_set - observed_set)
        notes_parts.append(f"Missing expected factor calls: {missing}")
    if all_sig_errors:
        notes_parts.append("; ".join(all_sig_errors))

    return FactorUsageEvalResult(
        case_id=case.case_id,
        category=case.category,
        expected_factor_calls=expected,
        ast_factor_calls=ast_calls,
        runtime_factor_calls=runtime_calls,
        factor_call_recall=recall,
        no_reinvented_wheel=no_reinvented,
        signature_valid=sig_valid,
        signature_errors=tuple(all_sig_errors),
        manual_rolling_ops=manual_ops,
        notes=" | ".join(notes_parts),
    )


def evaluate_factor_usage_suite(
    llm_client: BaseLLMClient,
    df: pd.DataFrame,
    cases: Optional[List[SystematicCase]] = None,
) -> List[FactorUsageEvalResult]:
    suite = cases if cases is not None else SYSTEMATIC_CASES
    results: List[FactorUsageEvalResult] = []
    for case in suite:
        prompt = build_systematic_prompt(case)
        raw = llm_client.generate(prompt, system_prompt=SYSTEMATIC_SYSTEM_PROMPT)
        results.append(evaluate_factor_usage_case(case, raw, df))
    return results


def summarize_factor_usage_results(results: List[FactorUsageEvalResult]) -> Dict[str, Any]:
    if not results:
        return {}
    factor_cases = [r for r in results if len(r.expected_factor_calls) > 0]
    return {
        "total_cases": len(results),
        "factor_cases": len(factor_cases),
        "mean_factor_call_recall": float(np.mean([r.factor_call_recall for r in results])),
        "factor_only_call_recall": (
            float(np.mean([r.factor_call_recall for r in factor_cases])) if factor_cases else 1.0
        ),
        "no_reinvented_wheel_rate": float(np.mean([1.0 if r.no_reinvented_wheel else 0.0 for r in results])),
        "signature_valid_rate": float(np.mean([1.0 if r.signature_valid else 0.0 for r in results])),
    }


def score_runs_jsonl_factor_usage(
    runs_jsonl_path: Path,
    df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Score all `task_type == "systematic"` entries in an existing `runs.jsonl` file
    on Factor Library Usage metrics (`factor_call_recall`, `no_reinvented_wheel`, `signature_valid`).
    """
    path = Path(runs_jsonl_path)
    if df is None:
        rng = np.random.default_rng(20261015)
        n_bars = 300
        dates = pd.bdate_range("2021-01-04", periods=n_bars)
        steps = rng.normal(0.0004, 0.012, n_bars)
        close = 100.0 * np.exp(np.cumsum(steps))
        open_ = np.r_[100.0, close[:-1]]
        high = np.maximum(open_, close) * 1.005
        low = np.minimum(open_, close) * 0.995
        volume = rng.integers(2_000_000, 25_000_000, n_bars)
        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=dates,
        )
        df["return"] = df["close"].pct_change()
        df["rolling_vol"] = df["return"].rolling(20).std() * np.sqrt(252)
        df = classify_market_regimes(df)

    cases_by_id = {c.case_id: c for c in SYSTEMATIC_CASES}
    rows: List[Dict[str, Any]] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("task_type") not in ("systematic", "implementation"):
            continue
        res_dict = rec.get("result", {})
        cid = str(res_dict.get("case_id", ""))
        if cid not in cases_by_id:
            continue
        case = cases_by_id[cid]
        raw_code = res_dict.get("extracted_code") or res_dict.get("raw_response") or ""
        fu = evaluate_factor_usage_case(case, raw_code, df)
        rows.append({
            "model": rec.get("model", "unknown"),
            "seed": int(rec.get("seed", 0)),
            "case_id": cid,
            "category": case.category,
            "has_expected_factors": len(case.expected_factor_calls) > 0,
            "factor_call_recall": fu.factor_call_recall,
            "no_reinvented_wheel": 1.0 if fu.no_reinvented_wheel else 0.0,
            "signature_valid": 1.0 if fu.signature_valid else 0.0,
            "ast_factor_calls": ",".join(fu.ast_factor_calls),
            "runtime_factor_calls": ",".join(fu.runtime_factor_calls),
            "signature_errors": "; ".join(fu.signature_errors),
            "notes": fu.notes,
        })

    scored_df = pd.DataFrame(rows)
    if scored_df.empty:
        return scored_df, pd.DataFrame()

    lb_df = (
        scored_df.groupby("model")
        .agg(
            cases=("case_id", "count"),
            factor_call_recall=("factor_call_recall", "mean"),
            no_reinvented_wheel=("no_reinvented_wheel", "mean"),
            signature_valid=("signature_valid", "mean"),
        )
        .sort_values(["signature_valid", "factor_call_recall"], ascending=False)
    )
    return scored_df, lb_df
