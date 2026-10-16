#!/usr/bin/env python3
"""
Layer 1 (Translation Fidelity) P1 Benchmark Runner:
  - Task 1A: Identify bad alpha (18 curated flaw-detection cases)
  - Task 1B: Systematic implementation + Friction accounting (18 curated code-generation cases)

Usage:
  .venv/bin/python scripts/run_layer1_benchmark.py --models golden mock --out docs/experiments/layer1_smoke
  .venv/bin/python scripts/run_layer1_benchmark.py --models deepseek-v3.2 --seeds 0 --out docs/experiments/layer1_v1
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from src.regimes import classify_market_regimes
from agent.llm import BaseLLMClient, get_llm_client
from agent.layer1 import (
    BAD_ALPHA_CASES,
    SYSTEMATIC_CASES,
    AMBIGUITY_CASES,
    BAD_ALPHA_SYSTEM_PROMPT,
    SYSTEMATIC_SYSTEM_PROMPT,
    AMBIGUITY_SYSTEM_PROMPT,
    build_bad_alpha_prompt,
    score_bad_alpha_prediction,
    GoldenBadAlphaLLMClient,
    build_systematic_prompt,
    evaluate_systematic_case,
    GoldenSystematicLLMClient,
    evaluate_factor_usage_case,
    build_ambiguity_prompt,
    score_ambiguity_prediction,
    GoldenAmbiguityLLMClient,
)


class MockLayer1LLMClient(BaseLLMClient):
    """Deterministic mock baseline for Layer 1 tasks."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.calls = 0

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        self.calls += 1
        if "Audit this strategy and return the JSON verdict" in prompt:
            flaws = []
            if ".mean()" in prompt or "center=True" in prompt or "shift(-1)" in prompt:
                flaws.append("lookahead_bias")
            if "rebalance_tolerance=0.0" in prompt or "rolling(2)" in prompt:
                flaws.append("friction_erosion")
            if "2020-02-19" in prompt or "test_df" in prompt:
                flaws.append("data_snooping")
            if "< sma20" in prompt or "entry_z" in prompt:
                flaws.append("logical_inconsistency")
            out = json.dumps({
                "has_flaw": len(flaws) > 0,
                "flaw_types": flaws,
                "explanation": "Heuristic mock audit",
            })
        elif "Decide whether to clarify or implement and return the JSON verdict" in prompt:
            if "window=" in prompt or "SMA(60)" in prompt or "entry_window=" in prompt:
                out = json.dumps({"action": "implement", "questions": [], "missing_slots": []})
            else:
                out = json.dumps({
                    "action": "clarify",
                    "questions": ["Could you clarify the lookback window and entry/exit threshold?"],
                    "missing_slots": ["lookback_window", "entry_exit_threshold"],
                })
        else:
            out = (
                "```python\n"
                "from src.signals import compute_moving_average_signal\n\n"
                "def build_strategy(df):\n"
                "    return {\n"
                "        'signals': compute_moving_average_signal(df, window=60),\n"
                "        'slippage_rate': 0.0005,\n"
                "        'commission_rate': 0.0005,\n"
                "        'rebalance_tolerance': 0.05,\n"
                "    }\n"
                "```"
            )
        self.record(prompt_tokens=len(prompt) // 4, completion_tokens=len(out) // 4, latency=0.0)
        return out


def load_layer1_market_df(n_bars: int = 600) -> pd.DataFrame:
    """Load SPY slice if available on disk, otherwise build deterministic multi-regime OHLCV frame."""
    spy_path = REPO_ROOT / "data/ohlcv_daily/SPY.csv"
    if spy_path.exists():
        df = pd.read_csv(spy_path, parse_dates=["date"], index_col="date").sort_index()
        if len(df) > n_bars:
            df = df.iloc[-n_bars:].copy()
    else:
        rng = np.random.default_rng(20261015)
        dates = pd.bdate_range("2021-01-04", periods=n_bars)
        steps = rng.normal(0.0004, 0.012, n_bars)
        steps[100:220] += np.sin(np.linspace(0, 8 * np.pi, 120)) * 0.015
        steps[300:420] *= 2.0
        close = 100.0 * np.exp(np.cumsum(steps))
        open_ = np.r_[100.0, close[:-1]] * (1.0 + rng.normal(0.0, 0.002, n_bars))
        high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.006, n_bars)))
        low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.006, n_bars)))
        volume = rng.integers(2_000_000, 25_000_000, n_bars)
        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=dates,
        )

    df["return"] = df["close"].pct_change()
    df["rolling_vol"] = df["return"].rolling(20).std() * np.sqrt(252)
    df = classify_market_regimes(df)
    return df


def _make_client_for_task(model: str, task_type: str, seed: int, temperature: float) -> BaseLLMClient:
    if model == "golden":
        if task_type == "bad_alpha":
            return GoldenBadAlphaLLMClient()
        if task_type == "ambiguity":
            return GoldenAmbiguityLLMClient()
        return GoldenSystematicLLMClient()
    if model == "mock":
        return MockLayer1LLMClient(seed=seed)
    return get_llm_client(model=model, seed=seed, temperature=temperature)


def _flush_artifacts(
    out_dir: Optional[Path],
    rows: List[Dict[str, Any]],
    jsonl_records: List[Dict[str, Any]],
    run_ba: bool,
    run_sys: bool,
    run_amb: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_df = pd.DataFrame(rows)

    ba_lb = pd.DataFrame()
    if run_ba and not summary_df.empty and (summary_df["task_type"] == "bad_alpha").any():
        sub = summary_df[summary_df["task_type"] == "bad_alpha"]
        ba_lb = (
            sub.groupby("model")
            .agg(
                cases=("case_id", "count"),
                category_f1=("category_f1", "mean"),
                category_recall=("category_recall", "mean"),
                category_precision=("category_precision", "mean"),
                exact_match=("exact_match", "mean"),
                binary_acc=("binary_correct", "mean"),
                fp_labels=("false_positive_labels", "sum"),
                tokens_per_case=("tokens", "mean"),
            )
            .sort_values("category_f1", ascending=False)
        )

    sys_lb = pd.DataFrame()
    fu_lb = pd.DataFrame()
    if run_sys and not summary_df.empty and (summary_df["task_type"] == "systematic").any():
        sub = summary_df[summary_df["task_type"] == "systematic"]
        sys_lb = (
            sub.groupby("model")
            .agg(
                cases=("case_id", "count"),
                pass_rate=("passed", "mean"),
                position_agreement=("position_agreement", "mean"),
                signal_agreement=("signal_agreement", "mean"),
                no_lookahead_rate=("no_lookahead", "mean"),
                friction_match_rate=("friction_match", "mean"),
                exec_pass_rate=("exec_pass", "mean"),
                sharpe_abs_error=("sharpe_abs_error", "mean"),
                tokens_per_case=("tokens", "mean"),
            )
            .sort_values(["pass_rate", "position_agreement"], ascending=False)
        )
        if "factor_call_recall" in sub.columns and sub["factor_call_recall"].notna().any():
            fu_lb = (
                sub.groupby("model")
                .agg(
                    cases=("case_id", "count"),
                    factor_call_recall=("factor_call_recall", "mean"),
                    no_reinvented_wheel=("no_reinvented_wheel", "mean"),
                    signature_valid=("signature_valid", "mean"),
                )
                .sort_values(["signature_valid", "factor_call_recall"], ascending=False)
            )

    amb_lb = pd.DataFrame()
    if run_amb and not summary_df.empty and (summary_df["task_type"] == "ambiguity").any():
        sub = summary_df[summary_df["task_type"] == "ambiguity"]
        amb_lb = (
            sub.groupby("model")
            .agg(
                cases=("case_id", "count"),
                clarify_accuracy=("clarify_correct", "mean"),
                slot_recall=("slot_recall", "mean"),
                false_clarify_rate=("false_clarify", lambda s: s[sub.loc[s.index, "category"] == "control"].mean()),
                tokens_per_case=("tokens", "mean"),
            )
            .sort_values(["clarify_accuracy", "slot_recall"], ascending=False)
        )

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(out_dir / "summary.csv", index=False)
        with (out_dir / "runs.jsonl").open("w", encoding="utf-8") as f:
            for rec in jsonl_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        md_lines = ["# Layer 1 Translation Benchmark Report\n"]
        if not ba_lb.empty:
            md_lines.append("## Task 1A: Identify Bad Alpha Leaderboard\n")
            md_lines.append(ba_lb.round(3).to_markdown() + "\n")
        if not sys_lb.empty:
            md_lines.append("## Task 1B: Systematic Implementation Leaderboard\n")
            md_lines.append(sys_lb.round(3).to_markdown() + "\n")
        if not fu_lb.empty:
            md_lines.append("## Task 1C: Factor Library Usage Leaderboard\n")
            md_lines.append(fu_lb.round(3).to_markdown() + "\n")
        if not amb_lb.empty:
            md_lines.append("## Task 1D: Ambiguity Detection Leaderboard\n")
            md_lines.append(amb_lb.round(3).to_markdown() + "\n")
        (out_dir / "REPORT.md").write_text("\n".join(md_lines), encoding="utf-8")

    return summary_df, ba_lb, sys_lb, fu_lb, amb_lb


def run_layer1_benchmark(
    models: List[str],
    seeds: List[int],
    task: str = "all",
    out_dir: Optional[Path] = None,
    temperature: float = 0.0,
    resume: bool = True,
) -> Dict[str, Any]:
    df_market = load_layer1_market_df()
    rows: List[Dict[str, Any]] = []
    jsonl_records: List[Dict[str, Any]] = []
    completed_keys: Set[Tuple[str, str, str, int]] = set()

    if resume and out_dir is not None and (out_dir / "summary.csv").exists() and (out_dir / "runs.jsonl").exists():
        try:
            existing_df = pd.read_csv(out_dir / "summary.csv")
            rows = existing_df.to_dict(orient="records")
            for r in rows:
                completed_keys.add((str(r["task_type"]), str(r["case_id"]), str(r["model"]), int(r["seed"])))
            for line in (out_dir / "runs.jsonl").read_text(encoding="utf-8").splitlines():
                if line.strip():
                    jsonl_records.append(json.loads(line))
        except Exception:
            rows = []
            jsonl_records = []
            completed_keys = set()

    run_ba = task in ("all", "all_p2", "bad_alpha")
    run_sys = task in ("all", "all_p2", "systematic")
    run_amb = task in ("all_p2", "ambiguity")

    for model in models:
        for seed in seeds:
            if run_ba:
                client = _make_client_for_task(model, "bad_alpha", seed, temperature)
                for case in BAD_ALPHA_CASES:
                    key = ("bad_alpha", case.case_id, model, seed)
                    if key in completed_keys:
                        continue
                    t0 = time.perf_counter()
                    prompt = build_bad_alpha_prompt(case)
                    before_tokens = client.usage.total_tokens
                    try:
                        raw = client.generate(prompt, system_prompt=BAD_ALPHA_SYSTEM_PROMPT)
                    except Exception as e:
                        raw = json.dumps({"error": str(e)})
                    elapsed = time.perf_counter() - t0
                    tokens_used = client.usage.total_tokens - before_tokens

                    res = score_bad_alpha_prediction(case, raw)
                    row = {
                        "task_type": "bad_alpha",
                        "case_id": case.case_id,
                        "category": ",".join(case.flaw_types) if case.flaw_types else "clean",
                        "model": model,
                        "seed": seed,
                        "binary_correct": res.binary_correct,
                        "exact_match": res.exact_match,
                        "category_recall": res.category_recall,
                        "category_precision": res.category_precision,
                        "category_f1": res.category_f1,
                        "false_positive_labels": res.false_positive_labels,
                        "exec_pass": np.nan,
                        "no_lookahead": np.nan,
                        "signal_agreement": np.nan,
                        "position_agreement": np.nan,
                        "position_corr": np.nan,
                        "friction_match": np.nan,
                        "sharpe_abs_error": np.nan,
                        "factor_call_recall": np.nan,
                        "no_reinvented_wheel": np.nan,
                        "signature_valid": np.nan,
                        "passed": res.exact_match,
                        "tokens": tokens_used,
                        "elapsed_seconds": elapsed,
                    }
                    rows.append(row)
                    jsonl_records.append({
                        "task_type": "bad_alpha",
                        "model": model,
                        "seed": seed,
                        "result": res.to_dict(),
                        "tokens": tokens_used,
                        "elapsed_seconds": elapsed,
                    })
                    completed_keys.add(key)
                    print(
                        f"[bad_alpha] {model} | {case.case_id} | f1={res.category_f1:.2f} "
                        f"exact={res.exact_match:.0f} ({elapsed:.1f}s)",
                        flush=True,
                    )
                _flush_artifacts(out_dir, rows, jsonl_records, run_ba, run_sys, run_amb)

            if run_sys:
                client = _make_client_for_task(model, "systematic", seed, temperature)
                for case in SYSTEMATIC_CASES:
                    key = ("systematic", case.case_id, model, seed)
                    if key in completed_keys:
                        continue
                    t0 = time.perf_counter()
                    prompt = build_systematic_prompt(case)
                    before_tokens = client.usage.total_tokens
                    try:
                        raw = client.generate(prompt, system_prompt=SYSTEMATIC_SYSTEM_PROMPT)
                    except Exception as e:
                        raw = f"# LLM error: {e}"
                    elapsed = time.perf_counter() - t0
                    tokens_used = client.usage.total_tokens - before_tokens

                    res = evaluate_systematic_case(case, raw, df_market)
                    fu = evaluate_factor_usage_case(case, raw, df_market)
                    row = {
                        "task_type": "systematic",
                        "case_id": case.case_id,
                        "category": case.category,
                        "model": model,
                        "seed": seed,
                        "binary_correct": np.nan,
                        "exact_match": np.nan,
                        "category_recall": np.nan,
                        "category_precision": np.nan,
                        "category_f1": np.nan,
                        "false_positive_labels": np.nan,
                        "exec_pass": 1.0 if res.exec_pass else 0.0,
                        "no_lookahead": 1.0 if res.no_lookahead else 0.0,
                        "signal_agreement": res.signal_agreement,
                        "position_agreement": res.position_agreement,
                        "position_corr": res.position_corr,
                        "friction_match": res.friction_match,
                        "sharpe_abs_error": res.sharpe_abs_error,
                        "factor_call_recall": fu.factor_call_recall,
                        "no_reinvented_wheel": 1.0 if fu.no_reinvented_wheel else 0.0,
                        "signature_valid": 1.0 if fu.signature_valid else 0.0,
                        "passed": 1.0 if res.passed else 0.0,
                        "tokens": tokens_used,
                        "elapsed_seconds": elapsed,
                    }
                    rows.append(row)
                    jsonl_records.append({
                        "task_type": "systematic",
                        "model": model,
                        "seed": seed,
                        "result": res.to_dict(),
                        "factor_usage": fu.to_dict(),
                        "tokens": tokens_used,
                        "elapsed_seconds": elapsed,
                    })
                    completed_keys.add(key)
                    print(
                        f"[systematic] {model} | {case.case_id} | pass={int(res.passed)} "
                        f"pos_agree={res.position_agreement:.3f} lookahead_ok={int(res.no_lookahead)} ({elapsed:.1f}s)",
                        flush=True,
                    )
                _flush_artifacts(out_dir, rows, jsonl_records, run_ba, run_sys, run_amb)

            if run_amb:
                client = _make_client_for_task(model, "ambiguity", seed, temperature)
                for case in AMBIGUITY_CASES:
                    key = ("ambiguity", case.case_id, model, seed)
                    if key in completed_keys:
                        continue
                    t0 = time.perf_counter()
                    prompt = build_ambiguity_prompt(case)
                    before_tokens = client.usage.total_tokens
                    try:
                        raw = client.generate(prompt, system_prompt=AMBIGUITY_SYSTEM_PROMPT)
                    except Exception as e:
                        raw = json.dumps({"error": str(e)})
                    elapsed = time.perf_counter() - t0
                    tokens_used = client.usage.total_tokens - before_tokens

                    res = score_ambiguity_prediction(case, raw)
                    row = {
                        "task_type": "ambiguity",
                        "case_id": case.case_id,
                        "category": "ambiguous" if case.is_ambiguous else "control",
                        "model": model,
                        "seed": seed,
                        "clarify_correct": res.clarify_correct,
                        "slot_recall": res.slot_recall if res.slot_recall is not None else np.nan,
                        "false_clarify": res.false_clarify,
                        "passed": res.clarify_correct,
                        "tokens": tokens_used,
                        "elapsed_seconds": elapsed,
                    }
                    rows.append(row)
                    jsonl_records.append({
                        "task_type": "ambiguity",
                        "model": model,
                        "seed": seed,
                        "result": res.to_dict(),
                        "tokens": tokens_used,
                        "elapsed_seconds": elapsed,
                    })
                    completed_keys.add(key)
                    print(
                        f"[ambiguity] {model} | {case.case_id} | acc={res.clarify_correct:.0f} "
                        f"slot_recall={res.slot_recall if res.slot_recall is not None else 1.0:.2f} ({elapsed:.1f}s)",
                        flush=True,
                    )
                _flush_artifacts(out_dir, rows, jsonl_records, run_ba, run_sys, run_amb)

    summary_df, ba_lb, sys_lb, fu_lb, amb_lb = _flush_artifacts(
        out_dir, rows, jsonl_records, run_ba, run_sys, run_amb
    )
    return {
        "summary_df": summary_df,
        "bad_alpha_leaderboard": ba_lb,
        "systematic_leaderboard": sys_lb,
        "factor_usage_leaderboard": fu_lb,
        "ambiguity_leaderboard": amb_lb,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Layer 1 Translation Fidelity Benchmark (P1 & P2)")
    ap.add_argument("--models", nargs="+", default=["golden", "mock"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--task", choices=["all", "all_p2", "bad_alpha", "systematic", "ambiguity"], default="all")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--out", default=str(REPO_ROOT / "docs/experiments/layer1_smoke"))
    args = ap.parse_args()

    out_dir = Path(args.out)
    res = run_layer1_benchmark(
        models=args.models,
        seeds=args.seeds,
        task=args.task,
        out_dir=out_dir,
        temperature=args.temperature,
        resume=not args.no_resume,
    )

    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    if not res["bad_alpha_leaderboard"].empty:
        print("\n=== Task 1A: Identify Bad Alpha Leaderboard ===")
        print(res["bad_alpha_leaderboard"].to_string())
    if not res["systematic_leaderboard"].empty:
        print("\n=== Task 1B: Systematic Implementation Leaderboard ===")
        print(res["systematic_leaderboard"].to_string())
    if not res["factor_usage_leaderboard"].empty:
        print("\n=== Task 1C: Factor Library Usage Leaderboard ===")
        print(res["factor_usage_leaderboard"].to_string())
    if not res["ambiguity_leaderboard"].empty:
        print("\n=== Task 1D: Ambiguity Detection Leaderboard ===")
        print(res["ambiguity_leaderboard"].to_string())
    print(f"\nSaved Layer 1 benchmark artifacts to {out_dir}")


if __name__ == "__main__":
    main()
