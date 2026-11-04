"""Named task sets for the Layer 1 implementation benchmark."""
from __future__ import annotations

from typing import Dict, List

from benchmark.layer1_code.canonical import CANONICAL_IMPL_CASES
from benchmark.layer1_code.impl.tasks.qcb_cases import QCB_CASES, QCB_CASES_BATCH2, QCB_CASES_ALL
from benchmark.layer1_code.impl.tasks.schema import HighFreedomImplCase

TASK_SETS: Dict[str, List[HighFreedomImplCase]] = {
    "canonical20": list(CANONICAL_IMPL_CASES),
    "qcb12": list(QCB_CASES),
    "qcb10": list(QCB_CASES_BATCH2),
    "qcb22": list(QCB_CASES_ALL),
}
TASK_SETS["all"] = TASK_SETS["canonical20"] + TASK_SETS["qcb22"]


def get_task_set(name: str) -> List[HighFreedomImplCase]:
    if name not in TASK_SETS:
        raise ValueError(f"Unknown task set '{name}'. Known: {sorted(TASK_SETS)}")
    return TASK_SETS[name]
