# scripts/

| directory | entry points |
|---|---|
| `layer1_impl/` | `run_harness_matrix.py` (one harness x models x task set), `run_full_matrix.sh` (whole matrix), `report_full_matrix.py` (aggregate a run dir) |
| `layer1_audit/` | `run_canonical40.py` (20 audit + 20 impl, 8-model leaderboard), `export_canonical40_table.py`, `run_v2_suites.py`, `run_p1_suites.py`, `demo_examples.py` |
| `layer2_search/` | `build_oracle.py`, `run_experiment_grid.py`, `score_against_oracle.py`, `benchmark_report.py`, `score_layer2_extensions.py`, live-loop runners |
| `layer3_oos/` | `make_planted_alpha.py` |
| `engine/` | course-project baselines, regime plots, ML evaluation |
| `data/` | `fetch_ohlcv.py`, `make_synthetic.py` |

All scripts are run from the repo root with `.venv/bin/python scripts/<dir>/<name>.py`.
