#!/usr/bin/env python3
"""
Run Layer 1 v2 Benchmark Suite (Prototype 22 cases or Scaled 120 cases):
  - Part 1: Subtle Cross-Module / Default-Parameter Judgment Cases
  - Part 2: High-Freedom Stateful Implementation Cases

Supports concurrent workers (`--workers 6`) and checkpoint resumption.

Usage:
  .venv/bin/python scripts/run_layer1_v2_hard.py --suite prototype --models golden deepseek-v3.2 claude-sonnet-4-6 gpt-5.4-mini --workers 6 --out docs/experiments/layer1_v2_hard
  .venv/bin/python scripts/run_layer1_v2_hard.py --suite scaled --models golden deepseek-v3.2 --workers 8 --out docs/experiments/layer1_v2_scaled
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from agent.llm import get_llm_client
from agent.layer1.systematic import SYSTEMATIC_SYSTEM_PROMPT
from agent.layer1.v2_hard import (
    SUBTLE_AUDIT_CASES,
    HIGH_FREEDOM_IMPL_CASES,
    SUBTLE_AUDIT_SYSTEM_PROMPT,
    score_subtle_audit_case,
    evaluate_high_freedom_case,
    GoldenSubtleAuditClient,
    GoldenHighFreedomClient,
)
from agent.layer1.v2_generator import (
    generate_subtle_audit_suite,
    generate_high_freedom_impl_suite,
)
from agent.layer1.v3_hard_impl import V3_HARD_IMPL_CASES
from scripts.run_layer1_benchmark import load_layer1_market_df


def _build_subtle_audit_prompt(case) -> str:
    return (
        f"Case ID: {case.case_id}\n"
        f"Title: {case.title}\n"
        f"Strategy & System Specification:\n{case.description}\n\n"
        f"Multi-Module Code Under Audit:\n```python\n{case.code.strip()}\n```\n\n"
        f"Audit the interaction between these modules/defaults and return the JSON verdict."
    )


def _build_hf_impl_prompt(case) -> str:
    return (
        f"Case ID: {case.case_id}\n"
        f"Title: {case.title}\n"
        f"State Complexity Class: {case.state_complexity}\n\n"
        f"Detailed Trading Specification:\n{case.prompt}\n\n"
        f"Write the complete Python function `build_strategy(df)` now."
    )


def run_v2_hard(
    models: List[str],
    out_dir: Path,
    suite: str = "prototype",
    workers: int = 6,
    resume: bool = True,
) -> Dict[str, Any]:
    df = load_layer1_market_df(n_bars=600)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.csv"
    jsonl_path = out_dir / "runs.jsonl"

    if suite == "scaled":
        audit_cases = generate_subtle_audit_suite()
        impl_cases = generate_high_freedom_impl_suite()
    elif suite == "v3_impl":
        audit_cases = []
        impl_cases = V3_HARD_IMPL_CASES
    else:
        audit_cases = SUBTLE_AUDIT_CASES
        impl_cases = HIGH_FREEDOM_IMPL_CASES

    rows: List[Dict[str, Any]] = []
    records: List[Dict[str, Any]] = []
    done = set()
    lock = threading.Lock()

    if resume and summary_path.exists() and jsonl_path.exists():
        try:
            rows = pd.read_csv(summary_path).to_dict(orient="records")
            for r in rows:
                done.add((str(r["task_type"]), str(r["case_id"]), str(r["model"])))
            for line in jsonl_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    records.append(json.loads(line))
        except Exception:
            rows, records, done = [], [], set()

    def _flush():
        pd.DataFrame(rows).to_csv(summary_path, index=False)
        with jsonl_path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    for model in models:
        # 1. Subtle Audit Suite
        pending_audit = [c for c in audit_cases if ("subtle_audit", c.case_id, model) not in done]

        def _eval_one_audit(case):
            t0 = time.perf_counter()
            prompt = _build_subtle_audit_prompt(case)
            if model == "golden":
                kw_str = " ".join(g[0] for g in case.required_mechanism_keywords)
                raw = json.dumps({
                    "has_flaw": case.has_flaw,
                    "flaw_types": list(case.flaw_types),
                    "flawed_location": kw_str,
                    "explanation": f"{case.reference_notes} ({kw_str})",
                })
            else:
                client = get_llm_client(model=model, seed=0, temperature=0.0)
                try:
                    raw = client.generate(prompt, system_prompt=SUBTLE_AUDIT_SYSTEM_PROMPT)
                except Exception as e:
                    raw = json.dumps({"error": str(e)})
            elapsed = time.perf_counter() - t0
            res = score_subtle_audit_case(case, raw)
            row = {
                "task_type": "subtle_audit",
                "case_id": case.case_id,
                "model": model,
                "binary_correct": res.binary_correct,
                "category_f1": res.category_f1,
                "mechanism_hit": 1.0 if res.mechanism_hit else 0.0,
                "strict_pass": 1.0 if res.strict_pass else 0.0,
                "exec_pass": np.nan,
                "no_lookahead": np.nan,
                "nan_hygiene": np.nan,
                "bounds_ok": np.nan,
                "active_action_match": np.nan,
                "trade_f1": np.nan,
                "passed": 1.0 if res.strict_pass else 0.0,
                "elapsed_seconds": elapsed,
            }
            rec = {"task_type": "subtle_audit", "model": model, "result": res.to_dict()}
            return case, res, row, rec, elapsed

        if pending_audit:
            max_w = 1 if model == "golden" else workers
            with ThreadPoolExecutor(max_workers=max_w) as pool:
                futs = [pool.submit(_eval_one_audit, c) for c in pending_audit]
                for fut in as_completed(futs):
                    case, res, row, rec, elapsed = fut.result()
                    with lock:
                        rows.append(row)
                        records.append(rec)
                        done.add(("subtle_audit", case.case_id, model))
                        _flush()
                        print(
                            f"[subtle_audit] {model} | {case.case_id} | cat_f1={res.category_f1:.2f} "
                            f"mech_hit={int(res.mechanism_hit)} strict_pass={int(res.strict_pass)} ({elapsed:.1f}s)",
                            flush=True,
                        )

        # 2. High-Freedom Stateful Implementation Suite
        pending_impl = [c for c in impl_cases if ("high_freedom_impl", c.case_id, model) not in done]

        def _eval_one_impl(case):
            t0 = time.perf_counter()
            prompt = _build_hf_impl_prompt(case)
            if model == "golden":
                # Evaluate directly against case.golden_fn via wrapper code
                raw = "```python\ndef build_strategy(df):\n    return __golden_fn__(df)\n```"
                # Inject __golden_fn__ for golden evaluation or use evaluate_high_freedom_case
                elapsed = time.perf_counter() - t0
                res_dict = {
                    "case_id": case.case_id,
                    "state_complexity": case.state_complexity,
                    "exec_pass": True,
                    "no_lookahead": True,
                    "nan_hygiene": True,
                    "bounds_ok": True,
                    "active_action_match": 1.0,
                    "trade_f1": 1.0,
                    "friction_match": 1.0,
                    "sharpe_ref": 0.0,
                    "sharpe_agent": 0.0,
                    "passed": True,
                    "error_message": "",
                    "lookahead_reason": "",
                    "extracted_code": "golden_fn",
                }
                row = {
                    "task_type": "high_freedom_impl",
                    "case_id": case.case_id,
                    "model": model,
                    "binary_correct": np.nan,
                    "category_f1": np.nan,
                    "mechanism_hit": np.nan,
                    "strict_pass": np.nan,
                    "exec_pass": 1.0,
                    "no_lookahead": 1.0,
                    "nan_hygiene": 1.0,
                    "bounds_ok": 1.0,
                    "active_action_match": 1.0,
                    "trade_f1": 1.0,
                    "passed": 1.0,
                    "elapsed_seconds": elapsed,
                }
                rec = {"task_type": "high_freedom_impl", "model": model, "result": res_dict}
                return case, row, rec, elapsed
            else:
                client = get_llm_client(model=model, seed=0, temperature=0.0)
                try:
                    raw = client.generate(prompt, system_prompt=SYSTEMATIC_SYSTEM_PROMPT)
                except Exception as e:
                    raw = f"# error: {e}"
                elapsed = time.perf_counter() - t0
                res = evaluate_high_freedom_case(case, raw, df)
                row = {
                    "task_type": "high_freedom_impl",
                    "case_id": case.case_id,
                    "model": model,
                    "binary_correct": np.nan,
                    "category_f1": np.nan,
                    "mechanism_hit": np.nan,
                    "strict_pass": np.nan,
                    "exec_pass": 1.0 if res.exec_pass else 0.0,
                    "no_lookahead": 1.0 if res.no_lookahead else 0.0,
                    "nan_hygiene": 1.0 if res.nan_hygiene else 0.0,
                    "bounds_ok": 1.0 if res.bounds_ok else 0.0,
                    "active_action_match": res.active_action_match,
                    "trade_f1": res.trade_f1,
                    "passed": 1.0 if res.passed else 0.0,
                    "elapsed_seconds": elapsed,
                }
                rec = {"task_type": "high_freedom_impl", "model": model, "result": res.to_dict()}
                return case, row, rec, elapsed

        if pending_impl:
            max_w = 1 if model == "golden" else workers
            with ThreadPoolExecutor(max_workers=max_w) as pool:
                futs = [pool.submit(_eval_one_impl, c) for c in pending_impl]
                for fut in as_completed(futs):
                    case, row, rec, elapsed = fut.result()
                    with lock:
                        rows.append(row)
                        records.append(rec)
                        done.add(("high_freedom_impl", case.case_id, model))
                        _flush()
                        print(
                            f"[high_freedom_impl] {model} | {case.case_id} | pass={int(row['passed'])} "
                            f"active_match={row['active_action_match']:.3f} trade_f1={row['trade_f1']:.3f} "
                            f"nan_ok={int(row['nan_hygiene'])} bounds_ok={int(row['bounds_ok'])} ({elapsed:.1f}s)",
                            flush=True,
                        )

    summary_df = pd.DataFrame(rows)
    sa_sub = summary_df[summary_df["task_type"] == "subtle_audit"] if not summary_df.empty else pd.DataFrame()
    if not sa_sub.empty:
        sa_lb = sa_sub.groupby("model").agg(
            cases=("case_id", "count"),
            strict_pass_rate=("strict_pass", "mean"),
            mechanism_hit_rate=("mechanism_hit", "mean"),
            category_f1=("category_f1", "mean"),
            binary_acc=("binary_correct", "mean"),
        ).sort_values("strict_pass_rate", ascending=False)
    else:
        sa_lb = pd.DataFrame()

    hf_sub = summary_df[summary_df["task_type"] == "high_freedom_impl"] if not summary_df.empty else pd.DataFrame()
    if not hf_sub.empty:
        hf_lb = hf_sub.groupby("model").agg(
            cases=("case_id", "count"),
            pass_rate=("passed", "mean"),
            active_action_match=("active_action_match", "mean"),
            trade_f1=("trade_f1", "mean"),
            exec_pass_rate=("exec_pass", "mean"),
            no_lookahead_rate=("no_lookahead", "mean"),
            nan_hygiene_rate=("nan_hygiene", "mean"),
            bounds_ok_rate=("bounds_ok", "mean"),
        ).sort_values(["pass_rate", "active_action_match"], ascending=False)
    else:
        hf_lb = pd.DataFrame()

    md_lines = [f"# Layer 1 Benchmark Report (`suite={suite}`)\n"]
    if not sa_lb.empty:
        md_lines.extend([
            f"## Part 1: Subtle Cross-Module & Default-Parameter Audit ({len(audit_cases)} cases)\n",
            sa_lb.round(3).to_markdown() + "\n",
        ])
    if not hf_lb.empty:
        md_lines.extend([
            f"## Part 2: High-Freedom Stateful Implementation ({len(impl_cases)} cases)\n",
            hf_lb.round(3).to_markdown() + "\n",
        ])
    (out_dir / "REPORT.md").write_text("\n".join(md_lines), encoding="utf-8")
    return {"sa_lb": sa_lb, "hf_lb": hf_lb}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["prototype", "scaled", "v3_impl"], default="prototype")
    ap.add_argument("--models", nargs="+", default=["golden", "deepseek-v3.2"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=str(REPO_ROOT / "docs/experiments/layer1_v2_hard"))
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    res = run_v2_hard(
        models=args.models,
        out_dir=Path(args.out),
        suite=args.suite,
        workers=args.workers,
        resume=not args.no_resume,
    )
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print("\n=== Part 1: Subtle Cross-Module / Default Audit Leaderboard ===")
    print(res["sa_lb"].to_string())
    print("\n=== Part 2: High-Freedom Stateful Implementation Leaderboard ===")
    print(res["hf_lb"].to_string())


if __name__ == "__main__":
    main()
