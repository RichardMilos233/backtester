#!/usr/bin/env python3
"""
Run the Layer 1 implementation suite under a given harness for a list of models.

One row per (harness, model, case). Resumable: existing (harness, model, case_id) rows
in <out>/runs.jsonl are skipped. Token usage and call counts are recorded per row.

Usage:
  set -a; source .env; set +a
  .venv/bin/python scripts/layer1_impl/run_harness_matrix.py --harness G0 \
      --models claude-haiku-5-5 gpt-5.4-mini deepseek-v4.1-flash --workers 6 \
      --out results/layer1_impl/harness_pilot_v1
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from benchmark.common.llm import get_llm_client
import numpy as np

from benchmark.layer1_code.canonical import evaluate_canonical_impl_case
from benchmark.layer1_code.impl.harness.harness import get_harness, InfraError
from benchmark.layer1_code.impl.tasks.task_bank import get_task_set
from benchmark.common.market_data import load_market_df


def _impl_prompt(case, variant: str = "pinned") -> str:
    """variant='pinned' uses the unambiguous spec; 'original' uses the loose real-world wording
    (falls back to the pinned spec when a case has no original_prompt)."""
    spec = case.prompt
    if variant == "original" and getattr(case, "original_prompt", None):
        spec = case.original_prompt
    return (
        f"Case ID: {case.case_id}\n"
        f"Title: {case.title}\n"
        f"Track: {case.state_complexity}\n\n"
        f"Strategy Specification:\n{spec}\n\n"
        f"Write the complete Python function `build_strategy(df)` now."
    )


def _load_done(jsonl_path: Path) -> Set[Tuple[str, str, str]]:
    done: Set[Tuple[str, str, str]] = set()
    if not jsonl_path.exists():
        return done
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("harness") in ("G4", "G4_v2"):
            expected = get_harness(r["harness"]).version
            if r.get("harness_version") != expected:
                raise ValueError(f"G4 harness version mismatch in {jsonl_path}; use a new output directory for {expected}")
        if r.get("infra_error"):
            continue
        done.add((r["harness"], r.get("variant", "pinned"), r["model"], r["case_id"]))
    return done


def run(harness_name: str, models: List[str], out_dir: Path, workers: int, n_bars: int, task_set: str, variant: str = "pinned") -> pd.DataFrame:
    cases = get_task_set(task_set)
    dfs: Dict[str, pd.DataFrame] = {}
    for c in cases:
        tk = getattr(c, "eval_ticker", "SPY")
        if tk not in dfs:
            dfs[tk] = load_market_df(tk, n_bars)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "runs.jsonl"
    done = _load_done(jsonl_path)
    lock = threading.Lock()

    def _one(model: str, case) -> Dict[str, Any]:
        harness = get_harness(harness_name)

        def make_client(temperature: float):
            return get_llm_client(model=model, seed=0, temperature=temperature)

        clients: List[Any] = []

        def make_client_tracked(temperature: float):
            c = make_client(temperature)
            clients.append(c)
            return c

        t0 = time.perf_counter()
        last_err = None
        for attempt in range(4):
            clients.clear()
            try:
                out = harness.run(_impl_prompt(case, variant), make_client_tracked,
                                  context={"pinned_spec": case.prompt, "case_id": case.case_id})
                break
            except InfraError as e:
                last_err = e
                time.sleep(15 * (attempt + 1))
        else:
            # never scored; summarize() drops these rows and reports the count
            return {"harness": harness_name, "task_set": task_set, "variant": variant, "model": model,
                    "harness_version": getattr(harness, "version", None),
                    "case_id": case.case_id, "infra_error": True, "error_message": str(last_err)[:500]}
        elapsed = time.perf_counter() - t0
        finish_reason = getattr(clients[-1], "last_finish_reason", None) if clients else None
        res = evaluate_canonical_impl_case(case, out.chosen, dfs[getattr(case, "eval_ticker", "SPY")])
        if case.case_id.startswith("sys_"):
            sub_group = "track_a_factor_api"
        elif case.case_id.startswith("qcb_"):
            sub_group = "qcb_ported"
        else:
            sub_group = "track_b_unhinted_stateful"
        return {
            "harness": harness_name,
            "harness_version": getattr(harness, "version", None),
            "task_set": task_set,
            "variant": variant,
            "n_bars": n_bars,
            "model": model,
            "case_id": case.case_id,
            "sub_group": sub_group,
            "state_complexity": case.state_complexity,
            "exec_pass": float(res.exec_pass),
            "no_lookahead": float(res.no_lookahead),
            "nan_hygiene": float(res.nan_hygiene),
            "bounds_ok": float(res.bounds_ok),
            "active_action_match": res.active_action_match,
            "trade_f1": res.trade_f1,
            "friction_match": res.friction_match,
            "passed": float(res.passed),
            "llm_calls": out.llm_calls,
            "prompt_tokens": out.prompt_tokens,
            "completion_tokens": out.completion_tokens,
            "n_candidates": len(out.candidates),
            "chosen_index": out.chosen_index,
            "elapsed_seconds": elapsed,
            "infra_error": False,
            "finish_reason": finish_reason,
            "error_message": res.error_message,
            "chosen_raw": out.chosen,
            "candidates": [c.raw for c in out.candidates],
            "trace": out.trace,
        }

    jobs = [(m, c) for m in models for c in cases if (harness_name, variant, m, c.case_id) not in done]
    print(f"{len(jobs)} jobs to run ({len(done)} already done)", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_one, m, c) for m, c in jobs]
        for fut in as_completed(futs):
            row = fut.result()
            if row.get("infra_error"):
                with lock:
                    with jsonl_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"[{row['harness']}] {row['model']:22s} | {row['case_id']:36s} | INFRA ERROR (not scored)", flush=True)
                continue
            with lock:
                with jsonl_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                f"[{row['harness']}] {row['model']:22s} | {row['case_id']:36s} | pass={int(row['passed'])} "
                f"am={row['active_action_match']:.3f} f1={row['trade_f1']:.3f} "
                f"calls={row['llm_calls']} tok={row['prompt_tokens']}+{row['completion_tokens']} ({row['elapsed_seconds']:.1f}s)",
                flush=True,
            )

    return summarize(out_dir)


def summarize(out_dir: Path) -> pd.DataFrame:
    jsonl_path = out_dir / "runs.jsonl"
    rows = [json.loads(l) for l in jsonl_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    df = pd.DataFrame(rows).drop(columns=["chosen_raw", "candidates", "trace"], errors="ignore")
    if "infra_error" in df.columns:
        n_infra = int(df["infra_error"].fillna(False).astype(bool).sum())
        if n_infra:
            print(f"[summarize] dropping {n_infra} infra-error rows (not scored)")
        df = df[~df["infra_error"].fillna(False).astype(bool)]
    df.to_csv(out_dir / "summary.csv", index=False)
    if "variant" not in df.columns:
        df["variant"] = "pinned"
    lb = df.groupby(["harness", "variant", "model"]).agg(
        cases=("case_id", "count"),
        pass_rate=("passed", "mean"),
        exec_pass=("exec_pass", "mean"),
        no_lookahead=("no_lookahead", "mean"),
        active_match=("active_action_match", "mean"),
        trade_f1=("trade_f1", "mean"),
        calls_per_case=("llm_calls", "mean"),
        prompt_tok=("prompt_tokens", "sum"),
        completion_tok=("completion_tokens", "sum"),
    ).round(3)
    track = df.groupby(["harness", "variant", "model", "sub_group"])["passed"].mean().unstack("sub_group").round(3)
    lb = lb.join(track)
    lb.to_csv(out_dir / "leaderboard.csv")
    print("\n" + lb.to_string())
    return lb


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness", default="G0")
    ap.add_argument("--models", nargs="+", default=["deepseek-v4.1-flash", "claude-fable-5.1", "gpt-6-astra"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--n-bars", type=int, default=1500)
    ap.add_argument("--task-set", default="all")
    ap.add_argument("--variant", default="pinned", choices=["pinned", "original"])
    ap.add_argument("--out", default=str(REPO_ROOT / "results/layer1_impl/harness_runs"))
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()
    if args.summarize_only:
        summarize(Path(args.out))
    else:
        run(args.harness, args.models, Path(args.out), args.workers, args.n_bars, args.task_set, args.variant)
