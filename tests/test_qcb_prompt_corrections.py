"""Task wording corrections must retain the reference strategy's meaning."""
from agent.layer1.task_bank import get_task_set


def test_qcb101_original_uses_standard_macd_parameter_labels():
    case = next(c for c in get_task_set("qcb22") if c.case_id == "qcb_101_atr_trail_macd_exit")
    assert "fast 12, slow 26, signal 9" in case.original_prompt
    assert "fast 12, slow 9, signal 26" not in case.original_prompt
    assert "MACD(12,26,9)" in case.prompt
