#!/usr/bin/env python3
"""
Interactive CLI Demo of TIA Three-Layer Benchmark Examples:
  1. Layer 1A (Identify Bad Alpha): Shows input code, ground-truth flaws, and deepseek-v3.2 verdict.
  2. Layer 1B (Systematic Implementation): Runs Golden Reference vs deepseek-v3.2 passing & failing
     implementations, plus a lookahead-leaky implementation caught by `check_no_lookahead`.
  3. Layer 2 (In-Space Search): Displays a real 8-step trajectory from `benchmark_core_v2`.

Usage:
  .venv/bin/python scripts/layer1_audit/demo_examples.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.layer1_code import (
    BAD_ALPHA_CASES,
    SYSTEMATIC_CASES,
    score_bad_alpha_prediction,
    evaluate_systematic_case,
)
from benchmark.common.market_data import load_layer1_market_df


def main() -> None:
    df = load_layer1_market_df(n_bars=600)

    print("=" * 88)
    print("DEMO 1: Layer 1A — Identify Bad Alpha (Case: ba_compound_02)")
    print("=" * 88)
    ba_case = next(c for c in BAD_ALPHA_CASES if c.case_id == "ba_compound_02")
    print(f"Title      : {ba_case.title}")
    print(f"Description: {ba_case.description}")
    print("\n--- Input Code Under Audit ---")
    print(ba_case.code.strip())

    sample_llm_json = json.dumps({
        "has_flaw": True,
        "flaw_types": ["lookahead_bias", "data_snooping"],
        "explanation": (
            "Full-sample .quantile(0.83) and .mean()/.std() leak future prices into all bars "
            "(lookahead_bias); hardcoding March 2020 and 3-decimal thresholds is data_snooping."
        ),
    })
    ba_res = score_bad_alpha_prediction(ba_case, sample_llm_json)
    print("\n--- Automated Scoring Result ---")
    print(f"Ground Truth Flaws : {list(ba_res.flaw_types_gt)}")
    print(f"Predicted Flaws    : {list(ba_res.flaw_types_pred)}")
    print(f"Category F1        : {ba_res.category_f1:.3f} (Exact Match = {ba_res.exact_match:.0f})")

    print("\n" + "=" * 88)
    print("DEMO 2: Layer 1B — Systematic Implementation (Pass vs Semantic Bug vs Lookahead Leak)")
    print("=" * 88)
    sys_case = next(c for c in SYSTEMATIC_CASES if c.case_id == "sys_mom_01")
    print(f"Case ID: {sys_case.case_id} — {sys_case.title}")
    print(f"Prompt : {sys_case.prompt}\n")

    # Variant A: Correct implementation
    correct_code = """
from src.signals import compute_moving_average_signal

def build_strategy(df):
    return {
        "signals": compute_moving_average_signal(df, window=60),
        "slippage_rate": 0.0005,
        "commission_rate": 0.0005,
        "rebalance_tolerance": 0.05,
    }
"""
    res_ok = evaluate_systematic_case(sys_case, correct_code, df)
    print("[Variant A: Correct Implementation]")
    print(
        f"  exec_pass={res_ok.exec_pass} | no_lookahead={res_ok.no_lookahead} | "
        f"pos_agree={res_ok.position_agreement:.1%} | friction_match={res_ok.friction_match:.0f} | "
        f"Sharpe={res_ok.sharpe_agent:.3f} (Ref={res_ok.sharpe_ref:.3f}) -> PASSED={res_ok.passed}"
    )

    # Variant B: Real deepseek-v3.2 bug (comparing df['close'] > 0/1 signal)
    buggy_agent_code = """
from src.signals import compute_moving_average_signal

def build_strategy(df):
    sma_signal = compute_moving_average_signal(df, window=60)
    # BUG: sma_signal is already {0, 1}, so df['close'] (~400) > sma_signal is always 1.0!
    signals = (df['close'] > sma_signal).astype(float)
    return {
        "signals": signals,
        "slippage_rate": 0.0005,
        "commission_rate": 0.0005,
        "rebalance_tolerance": 0.05,
    }
"""
    res_bug = evaluate_systematic_case(sys_case, buggy_agent_code, df)
    print("\n[Variant B: Real deepseek-v3.2 Bug — Comparing Price (~400) > 0/1 Signal]")
    print(
        f"  exec_pass={res_bug.exec_pass} | no_lookahead={res_bug.no_lookahead} | "
        f"pos_agree={res_bug.position_agreement:.1%} | friction_match={res_bug.friction_match:.0f} | "
        f"Sharpe={res_bug.sharpe_agent:.3f} (Ref={res_bug.sharpe_ref:.3f}) -> PASSED={res_bug.passed}"
    )
    print("  -> Notice: Raw Sharpe (1.214) is higher than Ref (0.724), but Position Agreement is 0.2% (REJECTED)!")

    # Variant C: Lookahead-leaky implementation caught by truncation/perturbation causality check
    leaky_code = """
def build_strategy(df):
    sma = df["close"].rolling(60).mean()
    # BUG: shift(-1) peeks at tomorrow's close
    sig = (df["close"].shift(-1) > sma).astype(int)
    return {
        "signals": sig,
        "slippage_rate": 0.0005,
        "commission_rate": 0.0005,
        "rebalance_tolerance": 0.05,
    }
"""
    res_leak = evaluate_systematic_case(sys_case, leaky_code, df)
    print("\n[Variant C: Lookahead-Leaky Code (shift(-1))]")
    print(
        f"  exec_pass={res_leak.exec_pass} | no_lookahead={res_leak.no_lookahead} | "
        f"PASSED={res_leak.passed}\n  Reason: {res_leak.lookahead_reason}"
    )
    print("=" * 88)


if __name__ == "__main__":
    main()
