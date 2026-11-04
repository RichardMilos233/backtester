#!/usr/bin/env python3
"""
Score Layer 2 Extended Tasks (Sample Efficiency & Hypothesis Quality) on an experiment directory.

Usage:
  .venv/bin/python scripts/layer2_search/score_layer2_extensions.py
  .venv/bin/python scripts/layer2_search/score_layer2_extensions.py --exp-dir results/layer2_search/core_v2 --oracle-dir results/layer2_search/oracle_v2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from benchmark.layer2_search.metrics import (
    score_experiment_trajectories,
    aggregate_layer2_extensions,
)


def build_step_curve_table(df: pd.DataFrame, group: str = "framework_type") -> pd.DataFrame:
    step_cols = [c for c in df.columns if c.startswith("best_pv_step_")]
    if not step_cols:
        return pd.DataFrame()
    per_task = df.groupby(["task_id", group])[step_cols].mean().reset_index()
    curve = per_task.groupby(group)[step_cols].mean()
    curve["auc_percentile_val"] = curve[step_cols].mean(axis=1)
    return curve.sort_values("auc_percentile_val", ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Score Layer 2 Sample Efficiency & Hypothesis Quality")
    ap.add_argument(
        "--exp-dir",
        default=str(REPO_ROOT / "results/layer2_search/core_v2"),
        help="Experiment directory containing runs.jsonl and summary.csv",
    )
    ap.add_argument(
        "--oracle-dir",
        default=str(REPO_ROOT / "results/layer2_search/oracle_v2"),
        help="Oracle directory containing <task_id>.csv.gz tables",
    )
    ap.add_argument(
        "--group",
        default="framework_type",
        choices=["framework_type", "model"],
    )
    ap.add_argument(
        "--out-md",
        default=None,
        help="Output markdown report path (defaults to <exp-dir>/LAYER2_EXTENSIONS_REPORT.md)",
    )
    args = ap.parse_args()

    exp_dir = Path(args.exp_dir)
    oracle_dir = Path(args.oracle_dir)
    out_md = Path(args.out_md) if args.out_md else (exp_dir / "LAYER2_EXTENSIONS_REPORT.md")

    scored_df = score_experiment_trajectories(exp_dir, oracle_dir)
    out_csv = exp_dir / "trajectory_metrics.csv"
    scored_df.to_csv(out_csv, index=False)

    lb = aggregate_layer2_extensions(scored_df, group=args.group)
    curve_df = build_step_curve_table(scored_df, group=args.group)
    per_task_auc = scored_df.pivot_table(
        index=args.group, columns="task_id", values="auc_percentile_val", aggfunc="mean"
    )
    per_task_corr = scored_df.pivot_table(
        index=args.group, columns="task_id", values="val_generalization_corr", aggfunc="mean"
    )

    pd.set_option("display.width", 240)
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print("\n=== Layer 2 Extended Tasks Leaderboard (Sample Efficiency & Hypothesis Quality) ===")
    print(lb.to_string())
    print("\n=== Best-So-Far Oracle Val Percentile Curve (t = 1..8) ===")
    print(curve_df.to_string())

    md_lines = [
        f"# Layer 2 Extended Benchmark Report (`{exp_dir.name}`)\n",
        f"Evaluated **{len(scored_df)}** trajectories across **{scored_df['task_id'].nunique()}** tasks "
        f"(`{', '.join(sorted(scored_df['task_id'].unique()))}`).\n",
        "## 1. Sample Efficiency & Hypothesis Quality Leaderboard\n",
        "- **`auc_percentile_val`**: Normalized Area Under the `best_percentile_val_so_far(t)` Curve across steps `t = 1..8`.",
        "- **`step1_percentile_val` -> `final_best_percentile_val`**: Trajectory progression from step 1 to step 8 (`trajectory_val_gain`).",
        "- **`step_improvement_rate`**: Fraction of steps `t >= 2` that improved Search Sharpe over step `t - 1`.",
        "- **`median_improvement_rate`**: Fraction of steps `t >= 2` that beat the running median Search Sharpe of prior steps.",
        "- **`val_generalization_corr`**: Within-trajectory Spearman rank correlation between Search Sharpe and Oracle Val Percentile (`percentile_val`).",
        "- **`hypothesis_alignment`**: Fraction of steps where the agent's natural-language hypothesis aligns with its structured signal family, volatility scaling, and regime filter.\n",
        lb.round(3).to_markdown() + "\n",
        "## 2. Best-So-Far Val Percentile Curve Across Budget Steps (`t = 1..8`)\n",
        curve_df.round(3).to_markdown() + "\n",
        "## 3. Per-Task Sample Efficiency (`auc_percentile_val`)\n",
        per_task_auc.round(3).to_markdown() + "\n",
        "## 4. Per-Task Search-to-Val Generalization Rank Correlation (`val_generalization_corr`)\n",
        per_task_corr.round(3).to_markdown() + "\n",
    ]
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\nSaved trajectory metrics to {out_csv} and report to {out_md}")


if __name__ == "__main__":
    main()
