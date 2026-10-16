#!/usr/bin/env python3
"""
Unified Canonical Layer 1 Benchmark Runner (40 High-Discriminative Cases):
  - Part 1: Identify Bad Alpha (20 cases = 10 Cross-Module/Default Traps + 6 Classic/Compound + 4 Clean Controls)
  - Part 2: Systematic Implementation (20 cases = 6 Factor API/Friction + 14 Unhinted Real-World Stateful Zones)

Automatically reuses cached LLM responses from prior experiment directories when available,
evaluates all models under one unified scoring protocol, and writes:
  - docs/experiments/layer1_canonical/summary.csv
  - docs/experiments/layer1_canonical/runs.jsonl
  - docs/experiments/layer1_canonical/REPORT.md

Usage:
  .venv/bin/python scripts/run_layer1_canonical.py --models golden claude-sonnet-4-6 gpt-5.4-mini glm-5.3 qwen3-coder deepseek-v3.2 mock
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from agent.llm import get_llm_client
from agent.layer1.systematic import SYSTEMATIC_SYSTEM_PROMPT
from agent.layer1.v2_hard import SUBTLE_AUDIT_SYSTEM_PROMPT
from agent.layer1.canonical import (
    CANONICAL_AUDIT_CASES,
    CANONICAL_IMPL_CASES,
    evaluate_canonical_audit_case,
    evaluate_canonical_impl_case,
    GoldenCanonicalAuditClient,
    GoldenCanonicalImplClient,
)
from scripts.run_layer1_benchmark import MockLayer1LLMClient, load_layer1_market_df


def _load_prior_response_cache() -> Dict[Tuple[str, str], str]:
    """Load (model, case_id) -> raw_response / extracted_code from prior runs to avoid redundant API calls."""
    cache: Dict[Tuple[str, str], str] = {}
    prior_files = [
        REPO_ROOT / "docs/experiments/layer1_canonical/runs.jsonl",
    ]
    for p in prior_files:
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                model = str(r["model"])
                res = r["result"]
                cid = str(res["case_id"])
                raw = res.get("raw_response") or res.get("extracted_code") or ""
                if raw and "error" not in raw.lower()[:30]:
                    cache[(model, cid)] = raw
            except Exception:
                continue
    return cache


def _audit_prompt(case) -> str:
    return (
        f"Case ID: {case.case_id}\n"
        f"Title: {case.title}\n"
        f"Strategy & System Specification:\n{case.description}\n\n"
        f"Code Under Audit:\n```python\n{case.code.strip()}\n```\n\n"
        f"Audit this code and return the JSON verdict."
    )


def _impl_prompt(case) -> str:
    return (
        f"Case ID: {case.case_id}\n"
        f"Title: {case.title}\n"
        f"Track: {case.state_complexity}\n\n"
        f"Strategy Specification:\n{case.prompt}\n\n"
        f"Write the complete Python function `build_strategy(df)` now."
    )


def run_canonical_benchmark(
    models: List[str],
    out_dir: Path,
    workers: int = 8,
) -> Dict[str, Any]:
    df = load_layer1_market_df(n_bars=600)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = _load_prior_response_cache()

    rows: List[Dict[str, Any]] = []
    records: List[Dict[str, Any]] = []
    lock = threading.Lock()

    golden_audit = GoldenCanonicalAuditClient()
    golden_impl = GoldenCanonicalImplClient()

    def _flush():
        pd.DataFrame(rows).to_csv(out_dir / "summary.csv", index=False)
        with (out_dir / "runs.jsonl").open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    for model in models:
        # ---------------------------------------------------------------------
        # Part 1: 20 Canonical Audit Cases
        # ---------------------------------------------------------------------
        def _eval_audit(case):
            t0 = time.perf_counter()
            if model == "golden":
                raw = golden_audit.generate(_audit_prompt(case))
            elif model == "mock":
                raw = MockLayer1LLMClient(seed=0).generate(_audit_prompt(case))
            elif (model, case.case_id) in cache:
                raw = cache[(model, case.case_id)]
            else:
                client = get_llm_client(model=model, seed=0, temperature=0.0)
                try:
                    raw = client.generate(_audit_prompt(case), system_prompt=SUBTLE_AUDIT_SYSTEM_PROMPT)
                except Exception as e:
                    raw = json.dumps({"error": str(e)})
            elapsed = time.perf_counter() - t0
            res = evaluate_canonical_audit_case(case, raw)
            sub_group = (
                "clean_control"
                if not case.has_flaw
                else ("subtle_cross_module_or_default" if case.case_id.startswith("sa_") else "classic_or_compound")
            )
            row = {
                "task_type": "audit",
                "sub_group": sub_group,
                "case_id": case.case_id,
                "model": model,
                "has_flaw_gt": 1.0 if case.has_flaw else 0.0,
                "binary_correct": res.binary_correct,
                "category_f1": res.category_f1,
                "mechanism_hit": 1.0 if res.mechanism_hit else 0.0,
                "strict_pass": 1.0 if res.strict_pass else 0.0,
                "clean_fp": (1.0 if res.has_flaw_pred else 0.0) if not case.has_flaw else np.nan,
                "exec_pass": np.nan,
                "no_lookahead": np.nan,
                "nan_hygiene": np.nan,
                "bounds_ok": np.nan,
                "active_action_match": np.nan,
                "trade_f1": np.nan,
                "friction_match": np.nan,
                "passed": 1.0 if res.strict_pass else 0.0,
                "elapsed_seconds": elapsed,
            }
            rec = {"task_type": "audit", "model": model, "result": res.to_dict()}
            return case, res, row, rec, elapsed

        max_w = 1 if model in ("golden", "mock") else workers
        with ThreadPoolExecutor(max_workers=max_w) as pool:
            futs = [pool.submit(_eval_audit, c) for c in CANONICAL_AUDIT_CASES]
            for fut in as_completed(futs):
                case, res, row, rec, elapsed = fut.result()
                with lock:
                    rows.append(row)
                    records.append(rec)
                    _flush()
                    print(
                        f"[audit] {model:18s} | {case.case_id:28s} | strict_pass={int(res.strict_pass)} "
                        f"mech={int(res.mechanism_hit)} f1={res.category_f1:.2f} ({elapsed:.1f}s)",
                        flush=True,
                    )

        # ---------------------------------------------------------------------
        # Part 2: 20 Canonical Implementation Cases
        # ---------------------------------------------------------------------
        def _eval_impl(case):
            t0 = time.perf_counter()
            if model == "golden":
                raw = golden_impl.generate(_impl_prompt(case))
            elif model == "mock":
                raw = MockLayer1LLMClient(seed=0).generate(_impl_prompt(case))
            elif (model, case.case_id) in cache:
                raw = cache[(model, case.case_id)]
            else:
                client = get_llm_client(model=model, seed=0, temperature=0.0)
                try:
                    raw = client.generate(_impl_prompt(case), system_prompt=SYSTEMATIC_SYSTEM_PROMPT)
                except Exception as e:
                    raw = f"# error: {e}"
            elapsed = time.perf_counter() - t0
            res = evaluate_canonical_impl_case(case, raw, df)
            sub_group = "track_a_factor_api" if case.case_id.startswith("sys_") else "track_b_unhinted_stateful"
            row = {
                "task_type": "implementation",
                "sub_group": sub_group,
                "case_id": case.case_id,
                "model": model,
                "has_flaw_gt": np.nan,
                "binary_correct": np.nan,
                "category_f1": np.nan,
                "mechanism_hit": np.nan,
                "strict_pass": np.nan,
                "clean_fp": np.nan,
                "exec_pass": 1.0 if res.exec_pass else 0.0,
                "no_lookahead": 1.0 if res.no_lookahead else 0.0,
                "nan_hygiene": 1.0 if res.nan_hygiene else 0.0,
                "bounds_ok": 1.0 if res.bounds_ok else 0.0,
                "active_action_match": res.active_action_match,
                "trade_f1": res.trade_f1,
                "friction_match": res.friction_match,
                "passed": 1.0 if res.passed else 0.0,
                "elapsed_seconds": elapsed,
            }
            rec = {"task_type": "implementation", "model": model, "result": res.to_dict()}
            return case, res, row, rec, elapsed

        with ThreadPoolExecutor(max_workers=max_w) as pool:
            futs = [pool.submit(_eval_impl, c) for c in CANONICAL_IMPL_CASES]
            for fut in as_completed(futs):
                case, res, row, rec, elapsed = fut.result()
                with lock:
                    rows.append(row)
                    records.append(rec)
                    _flush()
                    print(
                        f"[impl]  {model:18s} | {case.case_id:28s} | pass={int(res.passed)} "
                        f"active_match={res.active_action_match:.3f} trade_f1={res.trade_f1:.3f} ({elapsed:.1f}s)",
                        flush=True,
                    )

    summary_df = pd.DataFrame(rows)

    # 1. Audit Leaderboard (20 cases)
    aud = summary_df[summary_df["task_type"] == "audit"]
    aud_subtle = aud[aud["sub_group"] == "subtle_cross_module_or_default"].groupby("model")["strict_pass"].mean()
    aud_classic = aud[aud["sub_group"] == "classic_or_compound"].groupby("model")["strict_pass"].mean()
    aud_lb = aud.groupby("model").agg(
        cases=("case_id", "count"),
        audit_strict_pass=("strict_pass", "mean"),
        mechanism_hit=("mechanism_hit", "mean"),
        category_f1=("category_f1", "mean"),
        binary_acc=("binary_correct", "mean"),
        clean_fpr=("clean_fp", "mean"),
    )
    aud_lb["subtle_10_pass"] = aud_subtle
    aud_lb["classic_6_pass"] = aud_classic

    # 2. Implementation Leaderboard (20 cases)
    imp = summary_df[summary_df["task_type"] == "implementation"]
    imp_track_a = imp[imp["sub_group"] == "track_a_factor_api"].groupby("model")["passed"].mean()
    imp_track_b = imp[imp["sub_group"] == "track_b_unhinted_stateful"].groupby("model")["passed"].mean()
    imp_lb = imp.groupby("model").agg(
        cases=("case_id", "count"),
        impl_pass_rate=("passed", "mean"),
        active_action_match=("active_action_match", "mean"),
        trade_f1=("trade_f1", "mean"),
        exec_pass_rate=("exec_pass", "mean"),
        no_lookahead_rate=("no_lookahead", "mean"),
    )
    imp_lb["track_a_api_6_pass"] = imp_track_a
    imp_lb["track_b_stateful_14_pass"] = imp_track_b

    # 3. Combined Overall Layer 1 Leaderboard (40 cases)
    overall_lb = pd.DataFrame({
        "total_cases": aud_lb["cases"] + imp_lb["cases"],
        "layer1_overall_score": 0.5 * (aud_lb["audit_strict_pass"] + imp_lb["impl_pass_rate"]),
        "audit_strict_pass (20)": aud_lb["audit_strict_pass"],
        "audit_subtle_traps (10)": aud_lb["subtle_10_pass"],
        "audit_clean_fpr (4)": aud_lb["clean_fpr"],
        "impl_pass_rate (20)": imp_lb["impl_pass_rate"],
        "impl_factor_api (6)": imp_lb["track_a_api_6_pass"],
        "impl_unhinted_stateful (14)": imp_lb["track_b_stateful_14_pass"],
        "impl_active_match": imp_lb["active_action_match"],
        "impl_trade_f1": imp_lb["trade_f1"],
    }).sort_values("layer1_overall_score", ascending=False)

    aud_lb = aud_lb.loc[overall_lb.index]
    imp_lb = imp_lb.loc[overall_lb.index]

    md_lines = [
        "# Layer 1 Canonical Benchmark Report (40 High-Discriminative Cases)\n",
        "## 1. Overall Layer 1 Leaderboard (20 Audit + 20 Implementation = 40 Cases)\n",
        overall_lb.round(3).to_markdown() + "\n",
        "## 2. Task 1A: Strategy Code Audit (`Identify Bad Alpha`, 20 Cases)\n",
        aud_lb.round(3).to_markdown() + "\n",
        "## 3. Task 1B: Strategy Implementation (`Systematic Implementation`, 20 Cases)\n",
        imp_lb.round(3).to_markdown() + "\n",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(md_lines), encoding="utf-8")
    return {"overall_lb": overall_lb, "aud_lb": aud_lb, "imp_lb": imp_lb}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models",
        nargs="+",
        default=[
            "golden",
            "claude-opus-5.5",
            "gpt-6-astra",
            "deepseek-v4-pro",
            "claude-sonnet-4-6",
            "gpt-5.4-mini",
            "claude-haiku-4-5",
            "qwen-plus",
            "deepseek-v3.2",
            "mock",
        ],
    )
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(REPO_ROOT / "docs/experiments/layer1_canonical"))
    args = ap.parse_args()

    res = run_canonical_benchmark(args.models, Path(args.out), workers=args.workers)
    pd.set_option("display.width", 240)
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print("\n=== OVERALL LAYER 1 CANONICAL LEADERBOARD (40 CASES) ===")
    print(res["overall_lb"].to_string())
    print("\n=== PART 1: AUDIT LEADERBOARD (20 CASES) ===")
    print(res["aud_lb"].to_string())
    print("\n=== PART 2: IMPLEMENTATION LEADERBOARD (20 CASES) ===")
    print(res["imp_lb"].to_string())


if __name__ == "__main__":
    main()
