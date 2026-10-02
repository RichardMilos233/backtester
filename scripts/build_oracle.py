#!/usr/bin/env python3
"""
Exhaustively evaluate the canonical StrategySpec lattice on the Val and Test
splits of a task and persist the table. Required before scoring any agent run.

  .venv/bin/python scripts/build_oracle.py --task SPY_full
  .venv/bin/python scripts/build_oracle.py --suite core   # ~5 min per task
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent.evaluator import StrategyEvaluator
from agent.oracle import build_oracle_table, save_oracle, OracleScorer
from agent.tasks import get_task, SUITE_SMOKE, SUITE_CORE, SUITE_FULL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", nargs="+", default=None, help="Task ids from agent/tasks.py (default SPY_full)")
    ap.add_argument("--suite", choices=["smoke", "core", "full"], default=None, help="Build every oracle in a suite")
    ap.add_argument("--no-vol", action="store_true", help="Exclude vol-scaling overlays from the lattice")
    ap.add_argument("--force", action="store_true", help="Rebuild even if the table exists")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    suites = {"smoke": SUITE_SMOKE, "core": SUITE_CORE, "full": SUITE_FULL}
    task_ids = args.task or suites.get(args.suite or "", None) or ["SPY_full"]
    for task_id in task_ids:
        task = get_task(task_id)
        out = task.oracle_path(REPO_ROOT)
        if out.exists() and not args.force:
            print(f"{task_id}: oracle exists, skip ({out})")
            continue
        evaluator = StrategyEvaluator(task.load())
        s, v, t = evaluator.split_data()
        print(f"{task_id} ({task.ticker}): search {s.index[0].date()}..{s.index[-1].date()} "
              f"| val {v.index[0].date()}..{v.index[-1].date()} | test {t.index[0].date()}..{t.index[-1].date()}", flush=True)
        table = build_oracle_table(
            evaluator, include_vol=not args.no_vol,
            progress=None if args.quiet else (lambda i, n: print(f"  {i}/{n}", flush=True)),
        )
        save_oracle(table, out)
        print(f"  saved {len(table)} specs -> {out} ({table.attrs.get('elapsed_seconds')}s)")
        for k, val in OracleScorer(table).summary().items():
            print(f"    {k}: {val}")


if __name__ == "__main__":
    main()
