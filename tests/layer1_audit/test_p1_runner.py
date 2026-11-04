from __future__ import annotations

from pathlib import Path
import pandas as pd

from scripts.layer1_audit.run_p1_suites import run_layer1_benchmark


def test_run_layer1_benchmark_golden_and_mock_end_to_end(tmp_path: Path):
    out_dir = tmp_path / "layer1_out"
    report = run_layer1_benchmark(
        models=["golden", "mock"],
        seeds=[0],
        task="all",
        out_dir=out_dir,
    )

    assert (out_dir / "summary.csv").exists()
    assert (out_dir / "runs.jsonl").exists()
    assert (out_dir / "REPORT.md").exists()

    df = pd.read_csv(out_dir / "summary.csv")
    # 2 models * (18 bad_alpha + 18 systematic) = 72 rows
    assert len(df) == 72

    golden_ba = df[(df["model"] == "golden") & (df["task_type"] == "bad_alpha")]
    assert len(golden_ba) == 18
    assert golden_ba["category_f1"].mean() == 1.0

    golden_sys = df[(df["model"] == "golden") & (df["task_type"] == "systematic")]
    assert len(golden_sys) == 18
    assert golden_sys["passed"].mean() == 1.0
    assert golden_sys["position_agreement"].mean() == 1.0

    assert "bad_alpha_leaderboard" in report
    assert "systematic_leaderboard" in report
