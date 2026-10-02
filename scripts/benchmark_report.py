#!/usr/bin/env python3
"""
Aggregate a benchmark summary.csv into a leaderboard following docs/BENCHMARK.md:
  1. mean over seeds within (task, model, framework)
  2. mean over tasks (equal weight)
  3. bootstrap 95% CI over tasks x seeds
  4. win rate vs R-random on percentile_val, paired by (task, seed)

  .venv/bin/python scripts/benchmark_report.py docs/experiments/benchmark_core [--group framework_type] [--md out.md]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["percentile_val", "regret_test", "regret_vs_val_oracle", "test_sharpe", "budget_utilization"]


def bootstrap_ci(task_means: pd.Series, n: int = 2000, seed: int = 0):
    """CI of the cross-task mean, resampling tasks (the unit we generalise over)."""
    x = task_means.dropna().to_numpy()
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boots = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return (float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)))


def leaderboard(df: pd.DataFrame, group: str) -> pd.DataFrame:
    ok = df[df["status"] == "success"].copy()
    for c in METRICS + ["parse_failures", "reviewer_rejections", "llm_calls", "llm_total_tokens", "elapsed_seconds"]:
        if c in ok.columns:
            ok[c] = pd.to_numeric(ok[c], errors="coerce")

    # step 1: seed means per (task, group)
    per_task = ok.groupby(["task_id", group])[METRICS].mean().reset_index()

    rows = []
    for g, sub in per_task.groupby(group):
        row = {group: g, "tasks": sub["task_id"].nunique(), "runs": int((ok[group] == g).sum())}
        for m in METRICS:
            row[m] = sub[m].mean()
        lo, hi = bootstrap_ci(sub.set_index("task_id")["percentile_val"])
        row["pv_ci_lo"], row["pv_ci_hi"] = lo, hi
        raw = ok[ok[group] == g]
        row["parse_fail"] = raw["parse_failures"].sum()
        row["rejections"] = raw["reviewer_rejections"].sum()
        row["tokens_per_run"] = raw["llm_total_tokens"].mean() if "llm_total_tokens" in raw else np.nan
        row["sec_per_run"] = raw["elapsed_seconds"].mean()
        rows.append(row)
    lb = pd.DataFrame(rows).set_index(group)

    # step 4: paired win rate vs R-random on percentile_val
    if "R-random" in ok[group].values:
        base = ok[ok[group] == "R-random"].set_index(["task_id", "seed"])["percentile_val"]
        for g in lb.index:
            if g == "R-random":
                lb.loc[g, "win_vs_random"] = np.nan
                continue
            cand = ok[ok[group] == g].set_index(["task_id", "seed"])["percentile_val"]
            joined = pd.concat([cand.rename("c"), base.rename("b")], axis=1, join="inner").dropna()
            lb.loc[g, "win_vs_random"] = (joined["c"] > joined["b"]).mean() if len(joined) else np.nan
            lb.loc[g, "n_pairs"] = len(joined)
    return lb.sort_values("percentile_val", ascending=False)


def per_task_table(df: pd.DataFrame, group: str, metric: str = "percentile_val") -> pd.DataFrame:
    ok = df[df["status"] == "success"]
    return ok.pivot_table(index=group, columns="task_id", values=metric, aggfunc="mean")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_dir")
    ap.add_argument("--group", default="framework_type", choices=["framework_type", "model"])
    ap.add_argument("--md", default=None, help="Also write a markdown report here")
    args = ap.parse_args()

    df = pd.read_csv(Path(args.exp_dir) / "summary.csv")
    failed = df[df["status"] != "success"]
    lb = leaderboard(df, args.group)
    pt = per_task_table(df, args.group)

    pd.set_option("display.width", 240)
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print(f"\n{len(df) - len(failed)} ok, {len(failed)} failed runs in {args.exp_dir}\n")
    print("LEADERBOARD (mean over seeds, then over tasks; CI = bootstrap over tasks)")
    print(lb.to_string())
    print("\nPER-TASK percentile_val")
    print(pt.to_string())
    if len(failed):
        print("\nFAILED RUNS")
        print(failed[["task_id", args.group, "seed", "error_message"]].to_string(index=False))

    if args.md:
        out = Path(args.md)
        with out.open("w") as f:
            f.write(f"# Benchmark report: `{args.exp_dir}`\n\n")
            f.write(f"{len(df) - len(failed)} successful runs, {len(failed)} failed.\n\n")
            f.write("## Leaderboard\n\n")
            f.write(lb.round(3).to_markdown() + "\n\n")
            f.write("## Per-task percentile_val\n\n")
            f.write(pt.round(3).to_markdown() + "\n")
        print(f"\nmarkdown -> {out}")


if __name__ == "__main__":
    main()
