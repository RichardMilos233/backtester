#!/usr/bin/env python3
"""
Exhaustively evaluate the canonical StrategySpec lattice on the Val and Test
splits of a task and persist the table. Required before scoring any agent run.

  .venv/bin/python scripts/build_oracle.py --task SPY_full
  .venv/bin/python scripts/build_oracle.py --suite core   # ~5 min per task
  .venv/bin/python scripts/build_oracle.py --suite core --regimes --sample 20000

Default output directory is docs/experiments/oracle for v1 and
docs/experiments/oracle_v2 for v2 (driven by LATTICE_VERSION from agent/space.py).
Pass --oracle-dir to override. This ensures existing v1 tables under
docs/experiments/oracle/ are never overwritten by a v2 build.
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent.evaluator import StrategyEvaluator
from agent.oracle import build_oracle_table, save_oracle, OracleScorer
from agent.space import _count_space, LATTICE_VERSION
from agent.tasks import get_task, SUITE_SMOKE, SUITE_CORE, SUITE_FULL

_AUTO_SAMPLE_THRESHOLD = 20_000
_AUTO_SAMPLE_DEFAULT = 20_000

_DEFAULT_ORACLE_DIRS = {
    "v1": "docs/experiments/oracle",
    "v2": "docs/experiments/oracle_v2",
}


def main():
    default_oracle_dir = _DEFAULT_ORACLE_DIRS.get(LATTICE_VERSION, f"docs/experiments/oracle_{LATTICE_VERSION}")
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", nargs="+", default=None, help="Task ids from agent/tasks.py (default SPY_full)")
    ap.add_argument("--suite", choices=["smoke", "core", "full"], default=None, help="Build every oracle in a suite")
    ap.add_argument("--no-vol", action="store_true", help="Exclude vol-scaling overlays from the lattice")
    ap.add_argument("--regimes", action="store_true", help="Include regime filter options in the lattice")
    ap.add_argument("--sample", type=int, default=None,
                    help="Evaluate a random sample of N specs instead of the full lattice. "
                         "Defaults to 20000 automatically when lattice > 20000 specs.")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for --sample (default: 42)")
    ap.add_argument("--force", action="store_true", help="Rebuild even if the table exists")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument(
        "--oracle-dir",
        default=default_oracle_dir,
        help=(
            f"Directory to write oracle tables into. "
            f"Defaults to '{default_oracle_dir}' (driven by LATTICE_VERSION={LATTICE_VERSION!r}). "
            f"v1 tables under docs/experiments/oracle/ are never overwritten by a v2 build."
        ),
    )
    args = ap.parse_args()

    oracle_dir = REPO_ROOT / args.oracle_dir
    include_vol = not args.no_vol
    include_regimes = args.regimes

    # Safety guard: never write v2 into the v1 directory
    v1_dir = REPO_ROOT / _DEFAULT_ORACLE_DIRS["v1"]
    if LATTICE_VERSION != "v1" and oracle_dir.resolve() == v1_dir.resolve():
        print(
            f"[build_oracle] ERROR: refusing to write lattice {LATTICE_VERSION!r} tables into the v1 "
            f"oracle directory ({v1_dir}). Pass --oracle-dir explicitly to a different path."
        )
        sys.exit(1)

    # Auto-sample logic: if user didn't pass --sample but the lattice is large, default to 20k
    total_lattice = _count_space(include_vol=include_vol, include_regimes=include_regimes)
    sample = args.sample
    if sample is None and total_lattice > _AUTO_SAMPLE_THRESHOLD:
        sample = _AUTO_SAMPLE_DEFAULT
        print(
            f"[build_oracle] Lattice has {total_lattice:,} specs (> {_AUTO_SAMPLE_THRESHOLD:,}). "
            f"Defaulting to --sample {sample}. Pass --sample explicitly to suppress."
        )

    print(f"[build_oracle] lattice_version={LATTICE_VERSION!r}, oracle_dir={oracle_dir}")

    suites = {"smoke": SUITE_SMOKE, "core": SUITE_CORE, "full": SUITE_FULL}
    task_ids = args.task or suites.get(args.suite or "", None) or ["SPY_full"]
    for task_id in task_ids:
        task = get_task(task_id)
        # Use the --oracle-dir argument; fall back to task.oracle_path() only for legacy
        out = oracle_dir / f"{task_id}.csv.gz"
        if out.exists() and not args.force:
            print(f"{task_id}: oracle exists, skip ({out})")
            continue
        evaluator = StrategyEvaluator(task.load())
        s, v, t = evaluator.split_data()
        print(f"{task_id} ({task.ticker}): search {s.index[0].date()}..{s.index[-1].date()} "
              f"| val {v.index[0].date()}..{v.index[-1].date()} | test {t.index[0].date()}..{t.index[-1].date()}",
              flush=True)
        table = build_oracle_table(
            evaluator,
            include_vol=include_vol,
            include_regimes=include_regimes,
            sample=sample,
            seed=args.seed,
            progress=None if args.quiet else (lambda i, n: print(f"  {i}/{n}", flush=True)),
        )
        save_oracle(table, out)
        n_eval = table.attrs.get("n_evaluated", len(table))
        n_total = table.attrs.get("n_total_lattice", "?")
        sampled = table.attrs.get("sampled", False)
        tag = f"sampled {n_eval}/{n_total}" if sampled else f"exhaustive {n_eval}"
        lv = table.attrs.get("lattice_version", "?")
        print(f"  saved {len(table)} specs ({tag}, lattice_version={lv!r}) -> {out} ({table.attrs.get('elapsed_seconds')}s)")
        for k, val in OracleScorer(table).summary().items():
            print(f"    {k}: {val}")


if __name__ == "__main__":
    main()

