#!/usr/bin/env python3
"""
Re-score existing experiment summaries against an oracle table.

Adds, per run: percentile_val, percentile_test, regret_test, regret_vs_val_oracle,
excess_over_median_test. Prints a leaderboard grouped by --group (model or framework_type).

  .venv/bin/python scripts/score_against_oracle.py docs/experiments/phase1_models --group model
  .venv/bin/python scripts/score_against_oracle.py docs/experiments/phase2_infras docs/experiments/baselines --group framework_type
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from agent.oracle import load_oracle, OracleScorer
from agent.spec import StrategySpec
from agent.space import spec_key


def _spec_from_row(row: pd.Series, trace_best: dict | None) -> StrategySpec:
    if trace_best:
        return StrategySpec.from_dict(trace_best)
    return StrategySpec(
        spec_id=str(row.get("best_spec_id", "")),
        hypothesis="",
        signal_type=str(row["best_signal_type"]),
        params=json.loads(row["best_params"]) if isinstance(row["best_params"], str) else {},
    )


def _load_trace_best_specs(exp_dir: Path) -> dict:
    """(model, framework_type, seed) -> best_spec dict from runs.jsonl, so vol-scaling fields are preserved."""
    out = {}
    p = exp_dir / "runs.jsonl"
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        c = r["config"]
        out[(c["model"], c["framework_type"], int(c["seed"]))] = r["orchestrator_result"].get("best_spec")
    return out


def score_dir(exp_dir: Path, scorer: OracleScorer, evaluator=None) -> pd.DataFrame:
    df = pd.read_csv(exp_dir / "summary.csv")
    df = df[df["status"] == "success"].copy()
    traces = _load_trace_best_specs(exp_dir)
    rows = []
    for _, r in df.iterrows():
        spec = _spec_from_row(r, traces.get((r["model"], r["framework_type"], int(r["seed"]))))
        try:
            s = scorer.score_spec(spec, evaluator)
        except KeyError:
            s = scorer.score(float(r["val_sharpe"]), float(r["test_sharpe"]))
            s["on_lattice"] = False
        s["spec_key"] = spec_key(spec)
        s["budget_utilization"] = r["actual_backtests"] / r["max_backtests"] if r["max_backtests"] else np.nan
        rows.append(s)
    scored = pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    scored["source_dir"] = exp_dir.name
    return scored


def leaderboard(scored: pd.DataFrame, group: str) -> pd.DataFrame:
    g = scored.groupby(group)
    lb = pd.DataFrame({
        "n": g.size(),
        "percentile_val": g["percentile_val"].mean(),
        "percentile_test": g["percentile_test"].mean(),
        "regret_test": g["regret_test"].mean(),
        "regret_vs_val_oracle": g["regret_vs_val_oracle"].mean(),
        "test_sharpe": g["test_sharpe"].mean(),
        "test_sharpe_std": g["test_sharpe"].std(),
        "budget_util": g["budget_utilization"].mean(),
        "parse_fail": g["parse_failures"].sum(),
        "rejections": g["reviewer_rejections"].sum(),
        "distinct_picks": g["spec_key"].nunique(),
        "elapsed_s": g["elapsed_seconds"].mean(),
    })
    return lb.sort_values("percentile_val", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_dirs", nargs="+")
    ap.add_argument("--oracle", default=str(REPO_ROOT / "docs/experiments/oracle/SPY_full.csv.gz"))
    ap.add_argument("--group", default="framework_type", choices=["model", "framework_type"])
    ap.add_argument("--ticker", default="SPY", help="Used only to evaluate off-lattice picks")
    ap.add_argument("--out", default=None, help="Write scored rows to this CSV")
    args = ap.parse_args()

    scorer = OracleScorer(load_oracle(Path(args.oracle)))
    evaluator = None
    try:
        from src.data import load_market_data
        from agent.evaluator import StrategyEvaluator
        evaluator = StrategyEvaluator(load_market_data(args.ticker))
    except Exception as e:
        print(f"(no evaluator for off-lattice picks: {e})")

    print("ORACLE REFERENCE")
    for k, v in scorer.summary().items():
        print(f"  {k}: {v}")

    scored = pd.concat([score_dir(Path(d), scorer, evaluator) for d in args.exp_dirs], ignore_index=True)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print("\nLEADERBOARD by", args.group)
    print(leaderboard(scored, args.group).to_string())
    off = scored[~scored["on_lattice"].astype(bool)]
    if len(off):
        print(f"\n{len(off)} picks were off-lattice (evaluated directly): {sorted(off['spec_key'].unique())}")
    if args.out:
        scored.to_csv(args.out, index=False)
        print(f"\nscored rows -> {args.out}")


if __name__ == "__main__":
    main()
