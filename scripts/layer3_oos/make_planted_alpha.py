#!/usr/bin/env python3
"""
Generate and verify the 4 deterministic Layer 3 Planted Alpha synthetic series:
  - SYNTH_MOM: AR(1) / persistent regime trend where momentum(20..100) has Sharpe > 1.2 and mean_reversion < 0
  - SYNTH_MREV: Ornstein-Uhlenbeck mean reversion where mean_reversion / rsi(reversion) has Sharpe > 1.2 and momentum < 0
  - SYNTH_REGIME_Q1: High-vol/high-volume Q1 trend bursts where allowed_regimes=['Q1'] has Sharpe > 1.2
  - SYNTH_NULL: Zero-drift pure random walk control (true Sharpe = 0)

Usage:
  .venv/bin/python scripts/layer3_oos/make_planted_alpha.py
  .venv/bin/python scripts/layer3_oos/make_planted_alpha.py --out-dir data/planted_alpha
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from benchmark.common.spec import StrategySpec
from benchmark.layer3_oos.planted_alpha import (
    generate_planted_alpha_suite,
    compute_planted_series_sharpe,
)


def build_verification_table(suite: dict[str, pd.DataFrame]) -> pd.DataFrame:
    probe_specs = [
        ("mom_w20", StrategySpec("mom_20", "Momentum 20", "momentum", {"window": 20})),
        ("mom_w50", StrategySpec("mom_50", "Momentum 50", "momentum", {"window": 50})),
        ("mom_w100", StrategySpec("mom_100", "Momentum 100", "momentum", {"window": 100})),
        ("mrev_w20", StrategySpec("mrev_20", "MeanRev 20", "mean_reversion", {"window": 20, "entry_z": -1.5, "exit_z": 0.0})),
        ("rsi_rev14", StrategySpec("rsi_14", "RSI Rev 14", "rsi", {"window": 14, "lower": 30.0, "upper": 70.0, "mode": "reversion"})),
        ("mom_w20_rQ1", StrategySpec("mom_q1", "Momentum 20 Q1", "momentum", {"window": 20}, allowed_regimes=["Q1"])),
        ("mom_w20_rQ3", StrategySpec("mom_q3", "Momentum 20 Q3", "momentum", {"window": 20}, allowed_regimes=["Q3"])),
    ]

    rows = []
    for task_id, df in suite.items():
        row = {"task_id": task_id, "bars": len(df)}
        for col_name, spec in probe_specs:
            row[col_name] = compute_planted_series_sharpe(df, spec)
        rows.append(row)
    return pd.DataFrame(rows).set_index("task_id")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Layer 3 Planted Alpha synthetic OHLCV suite")
    ap.add_argument("--bars", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=20261015)
    ap.add_argument("--out-dir", default=None, help="Optional directory to write <task_id>.csv files")
    args = ap.parse_args()

    suite = generate_planted_alpha_suite(n_bars=args.bars, seed=args.seed)
    ver_df = build_verification_table(suite)

    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print("=== Layer 3 Planted Alpha Ground-Truth Sharpe Matrix ===")
    print(ver_df.to_string())

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for task_id, df in suite.items():
            out_path = out_dir / f"{task_id}.csv"
            df[["open", "high", "low", "close", "volume"]].to_csv(out_path, float_format="%.6f")
        print(f"\nSaved 4 planted alpha CSV files to {out_dir}")


if __name__ == "__main__":
    main()
