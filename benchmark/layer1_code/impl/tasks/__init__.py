"""
Task bank for Layer 1B.

  schema.py            HighFreedomImplCase (prompt + golden)
  indicators.py        shared indicator conventions every golden uses
  qcb_cases.py         22 tasks ported from QuantCode-Bench, pinned + original wording
  v3_hard_cases.py     14 unhinted "real-world failure zone" tasks (Track B of canonical20)
  systematic_cases.py  factor-library / friction tasks (Track A of canonical20)
  hf_cases.py          10 hand-written stateful prototypes
  hf_generator.py      deterministic 60-case scaled suite
  task_bank.py         named sets: canonical20, qcb22, all (import this one lazily:
                       it pulls in canonical.py, which assembles cases from the files above)
"""
from benchmark.layer1_code.impl.tasks.schema import HighFreedomImplCase

__all__ = ["HighFreedomImplCase"]
