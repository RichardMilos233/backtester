# results/

Experiment outputs, grouped like the code. Tracked: `summary.csv`, leaderboards, `REPORT.md`.
Ignored: `runs.jsonl` (raw LLM traces), `logs/`, `_archive/`.

| directory | what |
|---|---|
| `layer1_impl/full_matrix_20261023/` | 7 models x 10 harnesses x 22 tasks, the main harness result (`REPORT.md`, `HARNESS_WALKTHROUGH.md`) |
| `layer1_impl/harness_pilot_v2/` | earlier pilot on 3 cheap models, 5 harnesses |
| `layer1_impl/g4_v2_20261029/` | G4_v2 implementer + supervisor run with draft-vs-final comparison |
| `layer1_impl/qcb_screen.md` | screening of 56 QuantCode-Bench tasks and the pinning decisions for the 22 kept |
| `layer1_audit/canonical40/` | 40-case x 8-model canonical run (audit + impl), xlsx and long/wide csv |
| `layer2_search/core_v2/`, `core_noes/` | Layer 2 search runs, oracle-normalised |
| `layer2_search/oracle/`, `oracle_v2/` | exhaustive lattice tables used to normalise Layer 2 scores |
