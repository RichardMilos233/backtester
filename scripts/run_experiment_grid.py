#!/usr/bin/env python3
"""
CLI entry point for the agent-infrastructure benchmark.

One run = (task_id, model, framework_type, seed) at a fixed backtest budget.
Each run appends a row to <output-dir>/summary.csv and a trace to runs.jsonl.
When docs/experiments/oracle/<task_id>.csv exists, the row also carries
regime-free scores (percentile_val, regret_test, ...). Build oracles first with
scripts/build_oracle.py.

Modes:
  --mode smoke      : Mock LLM, 1 run on SPY_full.
  --mode phase1     : Foundation model bake-off, framework C, task SPY_full.
  --mode phase2     : Architecture ablation on a fixed model, task SPY_full.
  --mode baselines  : Non-LLM baselines (R-random, R-grid), model="none".
  --mode benchmark  : models x infras x seeds over a task suite (--suite core|full|smoke or --tasks ...).
  --mode custom     : models x infras x seeds on --tasks (default SPY_full).
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from agent.config import ExperimentConfig
from agent.grid import ExperimentGridRunner
from agent.tasks import SUITE_CORE, SUITE_FULL, SUITE_SMOKE, get_task

ORACLE_DIR = REPO_ROOT / "docs/experiments/oracle"

DEFAULT_PHASE1_MODELS = [
    "gpt-5.6-sol",
    "claude-opus-4-8",
    "claude-sonnet-5",
    "glm-5.3",
    "gemini-3-flash",
    "deepseek-chat",
]

DEFAULT_PHASE2_INFRAS = ["C", "C-sota", "C-corr", "C-reflect", "C-hier"]
BASELINE_INFRAS = ["R-random", "R-grid"]
SUITES = {"smoke": SUITE_SMOKE, "core": SUITE_CORE, "full": SUITE_FULL}


def print_summary_table(df: pd.DataFrame, group_col: str):
    if df.empty or "status" not in df.columns:
        print("No results to display.")
        return
    ok = df[df["status"] == "success"].copy()
    failed = df[df["status"] != "success"]
    if ok.empty:
        print("All runs failed or no successful runs recorded.")
        return

    # Oracle-normalised columns first (regime-free), raw Test Sharpe second.
    agg_rules = {
        "percentile_val": ["mean"],
        "regret_test": ["mean"],
        "regret_vs_val_oracle": ["mean"],
        "test_sharpe": ["mean", "std"],
        "val_sharpe": ["mean"],
        "overfitting_gap": ["mean"],
        "budget_utilization": ["mean"],
        "parse_failures": ["sum"],
        "reviewer_rejections": ["sum"],
        "llm_calls": ["sum"],
        "llm_total_tokens": ["sum"],
        "elapsed_seconds": ["mean"],
    }
    for c in agg_rules:
        if c in ok.columns:
            ok[c] = pd.to_numeric(ok[c], errors="coerce")
    agg_rules = {k: v for k, v in agg_rules.items() if k in ok.columns and ok[k].notna().any()}

    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    grouped = ok.groupby(group_col).agg(agg_rules)
    grouped.insert(0, ("n", "runs"), ok.groupby(group_col).size())
    if "percentile_val" in ok.columns and ok["percentile_val"].notna().any():
        grouped = grouped.sort_values(("percentile_val", "mean"), ascending=False)
    print("\n" + "=" * 110)
    print(f"BENCHMARK SUMMARY BY {group_col.upper()}  ({len(ok)} ok, {len(failed)} failed)")
    print("=" * 110)
    print(grouped.to_string())
    if "task_id" in ok.columns and ok["task_id"].nunique() > 1 and "percentile_val" in ok.columns:
        print("\nper-task percentile_val:")
        print(ok.pivot_table(index=group_col, columns="task_id", values="percentile_val", aggfunc="mean").to_string())
    print("=" * 110 + "\n")


def build_configs(tasks, models, infras, seeds, budget, early_stop=False):
    return [
        ExperimentConfig(task_id=t, model=m, framework_type=f, seed=s, max_backtests=budget, early_stop=early_stop)
        for t in tasks for m in models for f in infras for s in seeds
    ]


def main():
    parser = argparse.ArgumentParser(description="Agent Strategy Search Benchmark Runner")
    parser.add_argument("--mode", choices=["smoke", "phase1", "phase2", "baselines", "benchmark", "custom"], default="smoke")
    parser.add_argument("--models", nargs="+", default=None, help="Model IDs (or 'mock', 'none')")
    parser.add_argument("--infras", nargs="+", default=None,
                        help="Framework types: A B C D C-sota C-corr C-reflect C-hier R-random R-grid")
    parser.add_argument("--base-model", default="deepseek-chat", help="Fixed model for phase2")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Default: 42 43 44 45 46")
    parser.add_argument("--budget", type=int, default=4, help="Max backtests on the search split per run")
    parser.add_argument("--tasks", nargs="+", default=None, help="Task ids (see agent/tasks.py)")
    parser.add_argument("--suite", choices=sorted(SUITES), default=None, help="Named task suite for --mode benchmark")
    parser.add_argument("--with-baselines", action="store_true", help="Also run R-random/R-grid (model=none) on every task")
    parser.add_argument("--early-stop", action="store_true",
                        help="Re-enable patience-based early stopping (off by default so every method spends the full budget)")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--oracle-dir", default=None,
                        help="Directory holding <task_id>.csv.gz oracle tables. "
                             "Defaults to docs/experiments/oracle (v1). Point this at "
                             "docs/experiments/oracle_v2 to score against the v2 lattice.")
    args = parser.parse_args()

    seeds = args.seeds or [42, 43, 44, 45, 46]
    out_dir = Path(args.output_dir) if args.output_dir else Path("docs/experiments") / args.mode
    group_col = "model"
    configs = []

    if args.mode == "smoke":
        out_dir = Path(args.output_dir or "docs/experiments/grid_smoke")
        configs = build_configs(["SPY_full"], ["mock"], ["C"], [42], 2, args.early_stop)

    elif args.mode == "phase1":
        configs = build_configs(["SPY_full"], args.models or DEFAULT_PHASE1_MODELS, ["C"], seeds, args.budget, args.early_stop)

    elif args.mode == "phase2":
        group_col = "framework_type"
        configs = build_configs(["SPY_full"], [args.base_model], args.infras or DEFAULT_PHASE2_INFRAS, seeds, args.budget, args.early_stop)

    elif args.mode == "baselines":
        group_col = "framework_type"
        tasks = args.tasks or SUITES.get(args.suite or "smoke")
        configs = build_configs(tasks, ["none"], BASELINE_INFRAS, seeds, args.budget, args.early_stop)

    elif args.mode in ("benchmark", "custom"):
        if args.mode == "benchmark":
            tasks = args.tasks or SUITES[args.suite or "core"]
        else:
            tasks = args.tasks or ["SPY_full"]
        models = args.models or [args.base_model]
        infras = args.infras or ["C"]
        group_col = "model" if len(models) > len(infras) else "framework_type"
        configs = build_configs(tasks, models, infras, seeds, args.budget, args.early_stop)
        if args.with_baselines:
            configs += build_configs(tasks, ["none"], BASELINE_INFRAS, seeds, args.budget, args.early_stop)

    for t in sorted({c.task_id for c in configs}):
        get_task(t)  # fail fast on unknown ids
        oracle_dir_to_check = Path(args.oracle_dir) if args.oracle_dir else ORACLE_DIR
        if not (oracle_dir_to_check / f"{t}.csv.gz").exists():
            print(f"WARNING: no oracle for task {t}; run scripts/build_oracle.py --task {t} for normalised scores")

    print(f"Total configurations: {len(configs)}  ->  {out_dir}")
    effective_oracle_dir = Path(args.oracle_dir) if args.oracle_dir else ORACLE_DIR
    runner = ExperimentGridRunner(output_dir=out_dir, oracle_dir=effective_oracle_dir)
    summary_df = runner.run_grid(configs, resume=not args.no_resume)
    print_summary_table(summary_df, group_col=group_col)


if __name__ == "__main__":
    main()
