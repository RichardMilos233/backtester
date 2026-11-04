from __future__ import annotations

import math
from typing import Any, Callable, Dict, Tuple
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


def extract_python_code(text: str) -> str:
    """Extract Python source code from an LLM response (handles ```python ... ``` fences)."""
    stripped = text.strip()
    if "```" not in stripped:
        return stripped

    lines = stripped.splitlines()
    blocks = []
    current = []
    in_block = False
    for line in lines:
        fence = line.strip()
        if fence.startswith("```"):
            if in_block:
                blocks.append("\n".join(current).strip())
                current = []
                in_block = False
            else:
                in_block = True
            continue
        if in_block:
            current.append(line)

    if current and in_block:
        blocks.append("\n".join(current).strip())

    for b in blocks:
        if "def build_strategy" in b:
            return b
    return blocks[0] if blocks else stripped


def compile_build_strategy(code: str) -> Callable[[pd.DataFrame], Dict[str, Any]]:
    """Compile extracted Python code in a controlled namespace and return `build_strategy`."""
    clean_code = extract_python_code(code)
    env: Dict[str, Any] = {
        "__builtins__": __builtins__,
        "pd": pd,
        "np": np,
        "math": math,
        "signals": signals_mod,
        "regimes": regimes_mod,
        "compute_moving_average_signal": compute_moving_average_signal,
        "compute_macd_signal": compute_macd_signal,
        "compute_volatility_scaled_signal": compute_volatility_scaled_signal,
        "compute_mean_reversion_signal": compute_mean_reversion_signal,
        "compute_breakout_signal": compute_breakout_signal,
        "compute_rsi_signal": compute_rsi_signal,
        "combine_signals": combine_signals,
        "classify_market_regimes": classify_market_regimes,
    }
    exec(compile(clean_code, "<agent_strategy>", "exec"), env, env)
    fn = env.get("build_strategy")
    if not callable(fn):
        raise ValueError("Code did not define a callable `build_strategy(df)` function.")
    return fn


def normalize_strategy_output(raw_out: Any, df: pd.DataFrame) -> Dict[str, Any]:
    """Validate and normalize the dict returned by `build_strategy(df)`."""
    if isinstance(raw_out, pd.Series):
        raw_out = {"signals": raw_out}
    if not isinstance(raw_out, dict):
        raise ValueError(f"build_strategy must return a dict, got {type(raw_out).__name__}")
    if "signals" not in raw_out:
        raise ValueError("build_strategy output dict must contain key 'signals'")

    sig = raw_out["signals"]
    if not isinstance(sig, pd.Series):
        sig = pd.Series(sig, index=df.index)
    sig = sig.reindex(df.index).fillna(0.0).astype(float)
    if np.isinf(sig.to_numpy()).any():
        raise ValueError("signals contains +/-inf values")

    return {
        "signals": sig,
        "slippage_rate": float(raw_out.get("slippage_rate", 0.0005)),
        "commission_rate": float(raw_out.get("commission_rate", 0.0005)),
        "rebalance_tolerance": float(raw_out.get("rebalance_tolerance", 0.05)),
    }


def check_no_lookahead(
    fn: Callable[[pd.DataFrame], Dict[str, Any]],
    df: pd.DataFrame,
    atol: float = 1e-7,
) -> Tuple[bool, str]:
    """
    Causality test (truncation + future-perturbation invariance):
    1. Prefix truncation: signals on `df.iloc[:t_k]` must match `signals_full[:t_k]`.
    2. Future shock invariance: multiplying future prices/volumes in `df.iloc[t_k:]` by 10x
       and 0.1x must not alter any signal in `[:t_k]`. This catches binary `shift(-k)` leaks
       even when the unperturbed signal happened to be 0 at `t_k - 1`.
    """
    n = len(df)
    full_out = normalize_strategy_output(fn(df.copy()), df)
    sig_full = full_out["signals"].to_numpy(dtype=float)

    checkpoints = sorted({max(20, int(0.4 * n)), max(30, int(0.65 * n)), max(40, int(0.85 * n))})
    price_cols = [c for c in ("open", "high", "low", "close", "rolling_vol") if c in df.columns]

    for tk in checkpoints:
        if tk >= n:
            continue
        sig_ref = sig_full[:tk]

        # Test 1: Truncation
        sub_df = df.iloc[:tk].copy()
        sub_out = normalize_strategy_output(fn(sub_df), sub_df)
        sig_trunc = sub_out["signals"].to_numpy(dtype=float)
        if not np.allclose(sig_trunc, sig_ref, atol=atol, rtol=1e-6):
            diff = np.abs(sig_trunc - sig_ref)
            bad_idx = int(np.argmax(diff))
            return (
                False,
                f"Lookahead bias detected at truncation checkpoint {tk}/{n}: "
                f"bar {bad_idx} changed from {sig_ref[bad_idx]:.6f} (full) "
                f"to {sig_trunc[bad_idx]:.6f} (truncated), max_diff={diff[bad_idx]:.6e}",
            )

        # Test 2: Future perturbation (+10x and 0.1x on bars >= tk)
        for mult in (10.0, 0.1):
            pert_df = df.copy()
            for col in price_cols:
                pert_df.iloc[tk:, pert_df.columns.get_loc(col)] = pert_df[col].iloc[tk:] * mult
            pert_out = normalize_strategy_output(fn(pert_df), pert_df)
            sig_pert = pert_out["signals"].to_numpy(dtype=float)[:tk]
            if not np.allclose(sig_pert, sig_ref, atol=atol, rtol=1e-6):
                diff = np.abs(sig_pert - sig_ref)
                bad_idx = int(np.argmax(diff))
                return (
                    False,
                    f"Lookahead bias detected under future perturbation (x{mult} after {tk}/{n}): "
                    f"bar {bad_idx} changed from {sig_ref[bad_idx]:.6f} "
                    f"to {sig_pert[bad_idx]:.6f}, max_diff={diff[bad_idx]:.6e}",
                )
    return True, ""
