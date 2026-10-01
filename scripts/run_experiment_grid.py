#!/usr/bin/env python3
"""
CLI entry point for running orthogonal Agent strategy search experiments.

Modes:
  --mode smoke   : Quick verification with Mock LLM (1 run).
  --mode phase1  : Foundation model bake-off on fixed Framework C across seeds.
  --mode phase2  : Agent architecture ablation on fixed model across seeds.
  --mode custom  : Custom matrix specified via --models, --infras, --seeds.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.data import load_market_data
from agent.config import ExperimentConfig
from agent.evaluator import StrategyEvaluator
from agent.grid import ExperimentGridRunner


DEFAULT_PHASE1_MODELS = [
    "deepseek-chat",
    "claude-sonnet-4-6",
    "gpt-5.6-terra",
    "glm-5",
]

DEFAULT_PHASE2_INFRAS = [
    "C",
    "C-sota",
    "C-corr",
    "C-reflect",
    "C-hier",
]


def print_summary_table(df: pd.DataFrame, group_col: str):
    if df.empty or "status" not in df.columns:
        print("No results to display.")
        return

    success_df = df[df["status"] == "success"].copy()
    if success_df.empty:
        print("All runs failed or no successful runs recorded.")
        return

    # Ensure numeric columns
    numeric_cols = [
        "test_sharpe", "val_sharpe", "overfitting_gap",
        "test_total_return", "test_max_drawdown", "test_turnover",
        "parse_failures", "reviewer_rejections", "elapsed_seconds"
    ]
    for c in numeric_cols:
        if c in success_df.columns:
            success_df[c] = pd.to_numeric(success_df[c], errors="coerce")

    agg_rules = {
        "test_sharpe": ["mean", "std"],
        "val_sharpe": ["mean"],
        "overfitting_gap": ["mean"],
        "test_total_return": ["mean"],
        "test_max_drawdown": ["mean"],
        "test_turnover": ["mean"],
        "parse_failures": ["sum"],
        "reviewer_rejections": ["sum"],
        "elapsed_seconds": ["mean"],
    }
    # Keep only rules where column exists
    agg_rules = {k: v for k, v in agg_rules.items() if k in success_df.columns}

    grouped = success_df.groupby(group_col).agg(agg_rules)
    print("\n" + "=" * 100)
    print(f"BENCHMARK SUMMARY BY {group_col.upper()}")
    print("=" * 100)
    print(grouped.to_string())
    print("=" * 100 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Agent Strategy Search Grid Runner")
    parser.add_argument("--mode", choices=["smoke", "phase1", "phase2", "custom"], default="smoke",
                        help="Experiment mode")
    parser.add_argument("--models", nargs="+", default=None,
                        help="List of model IDs to test")
    parser.add_argument("--infras", nargs="+", default=None,
                        help="List of framework types (e.g. C C-sota C-corr C-reflect C-hier)")
    parser.add_argument("--base-model", default="deepseek-chat",
                        help="Fixed model to use during phase2 infra comparisons")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="List of random seeds (default: 42 43 44 45 46)")
    parser.add_argument("--budget", type=int, default=4,
                        help="Max backtests allowed on search set per seed (default: 4)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for runs.jsonl and summary.csv")
    parser.add_argument("--no-resume", action="store_true",
                        help="Do not resume; re-run all configurations")

    args = parser.parse_args()

    # Load market data
    print("Loading market data (SPY)...")
    market_df = load_market_data("SPY")
    evaluator = StrategyEvaluator(market_df)

    # Determine default paths & configurations
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path("docs/experiments") / args.mode

    seeds = args.seeds or [42, 43, 44, 45, 46]
    configs = []
    group_col = "model"

    if args.mode == "smoke":
        out_dir = Path("docs/experiments/grid_smoke")
        configs = [
            ExperimentConfig(model="mock", framework_type="C", seed=42, max_backtests=2)
        ]
        group_col = "model"

    elif args.mode == "phase1":
        models = args.models or DEFAULT_PHASE1_MODELS
        group_col = "model"
        for m in models:
            for s in seeds:
                configs.append(
                    ExperimentConfig(
                        model=m,
                        framework_type="C",
                        seed=s,
                        max_backtests=args.budget
                    )
                )

    elif args.mode == "phase2":
        infras = args.infras or DEFAULT_PHASE2_INFRAS
        group_col = "framework_type"
        for inf in infras:
            for s in seeds:
                configs.append(
                    ExperimentConfig(
                        model=args.base_model,
                        framework_type=inf,
                        seed=s,
                        max_backtests=args.budget
                    )
                )

    elif args.mode == "custom":
        models = args.models or [args.base_model]
        infras = args.infras or ["C"]
        group_col = "model" if len(models) > len(infras) else "framework_type"
        for m in models:
            for inf in infras:
                for s in seeds:
                    configs.append(
                        ExperimentConfig(
                            model=m,
                            framework_type=inf,
                            seed=s,
                            max_backtests=args.budget
                        )
                    )

    print(f"Total configurations to evaluate: {len(configs)}")
    print(f"Output directory: {out_dir}")

    runner = ExperimentGridRunner(output_dir=out_dir, evaluator=evaluator)
    summary_df = runner.run_grid(configs, resume=not args.no_resume)

    print_summary_table(summary_df, group_col=group_col)


if __name__ == "__main__":
    main()
