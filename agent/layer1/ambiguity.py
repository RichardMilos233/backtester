from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from agent.llm import BaseLLMClient, LLMUsage
from agent.layer1.schema import (
    VALID_AMBIGUITY_SLOTS,
    AmbiguityCase,
    AmbiguityEvalResult,
)
from agent.layer1.cases.ambiguity_cases import AMBIGUITY_CASES

DEFAULT_SLOT_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "lookback_window": (
        "lookback_window",
        "lookback",
        "window",
        "period",
        "span",
        "窗口",
        "周期",
        "多少天",
        "多少日",
        "天数",
    ),
    "entry_exit_threshold": (
        "entry_exit_threshold",
        "threshold",
        "entry_z",
        "exit_z",
        "lower",
        "upper",
        "cutoff",
        "阈值",
        "超买",
        "超卖",
        "入场",
        "出场",
        "平仓条件",
    ),
    "volatility_definition": (
        "volatility_definition",
        "target_vol",
        "max_leverage",
        "rolling_vol",
        "volatility",
        "波动率",
        "目标波动率",
        "最大杠杆",
        "恐慌",
    ),
    "execution_timing": (
        "execution_timing",
        "execution",
        "timing",
        "holding",
        "rebalance",
        "成交时点",
        "执行时点",
        "持仓天数",
        "持有多久",
        "何时平仓",
        "次日开盘",
    ),
}

AMBIGUITY_SYSTEM_PROMPT = """You are a senior quantitative trading assistant.
Given a user's strategy specification, decide whether the specification is complete enough to implement deterministically ("implement") or whether critical parameters/definitions are missing and require clarification first ("clarify").

Valid missing slot categories (`missing_slots`):
- "lookback_window": Missing rolling window / lookback period (e.g., unspecified SMA window or Donchian channel window).
- "entry_exit_threshold": Missing numerical entry/exit thresholds or stop/exit rules (e.g., unspecified RSI oversold/overbought levels or Z-score cutoffs).
- "volatility_definition": Missing volatility metric definition, target volatility (`target_vol`), or leverage cap (`max_leverage`).
- "execution_timing": Missing execution timing, holding period, or rebalance schedule.

If all critical parameters are explicitly specified, set `"action": "implement"`, `"questions": []`, and `"missing_slots": []`.

Output ONLY valid JSON matching this schema:
{
  "action": "clarify" | "implement",
  "questions": ["<concise clarifying question 1>", ...],
  "missing_slots": ["<zero or more slot names from the valid slot categories>"]
}
"""


def build_ambiguity_prompt(case: AmbiguityCase) -> str:
    return (
        f"Case ID: {case.case_id}\n"
        f"Title: {case.title}\n"
        f"User Strategy Request:\n{case.prompt}\n\n"
        f"Decide whether to clarify or implement and return the JSON verdict."
    )


def _parse_ambiguity_json(raw_text: str) -> Dict[str, Any]:
    text = raw_text.strip()
    if "```" in text:
        lines = text.splitlines()
        cleaned = []
        in_fence = False
        for line in lines:
            if line.strip().startswith("```"):
                in_fence = not in_fence
                continue
            cleaned.append(line)
        text = "\n".join(cleaned).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


def _slot_is_hit(
    slot: str,
    case: AmbiguityCase,
    pred_slots: Set[str],
    combined_question_text: str,
) -> bool:
    if slot in pred_slots:
        return True
    keywords = list(DEFAULT_SLOT_KEYWORDS.get(slot, ()))
    if case.slot_keywords and slot in case.slot_keywords:
        keywords.extend(case.slot_keywords[slot])
    lower_text = combined_question_text.lower()
    return any(kw.lower() in lower_text for kw in keywords)


def score_ambiguity_prediction(case: AmbiguityCase, raw_response: str) -> AmbiguityEvalResult:
    try:
        data = _parse_ambiguity_json(raw_response)
        raw_action = str(data.get("action", "")).strip().lower()
        if raw_action not in ("clarify", "implement"):
            raw_action = "clarify" if data.get("questions") or data.get("missing_slots") else "implement"

        raw_questions = data.get("questions", [])
        if not isinstance(raw_questions, list):
            raw_questions = [str(raw_questions)] if raw_questions else []
        questions = tuple(str(q).strip() for q in raw_questions if str(q).strip())

        raw_slots = data.get("missing_slots", [])
        if not isinstance(raw_slots, list):
            raw_slots = [str(raw_slots)] if raw_slots else []
        pred_slots = tuple(
            sorted({str(s).strip() for s in raw_slots if str(s).strip() in VALID_AMBIGUITY_SLOTS})
        )
        parse_ok = True
        err_msg = ""
    except Exception as e:
        raw_action = "implement"
        questions = ()
        pred_slots = ()
        parse_ok = False
        err_msg = str(e)

    expected_action = "clarify" if case.is_ambiguous else "implement"
    clarify_correct = 1.0 if (raw_action == expected_action) else 0.0

    if case.is_ambiguous:
        false_clarify = 0.0
        if raw_action != "clarify":
            hit_slots: Tuple[str, ...] = ()
            slot_recall: Optional[float] = 0.0
        else:
            combined_text = " ".join(list(questions) + list(pred_slots))
            hits = [
                s
                for s in case.required_slots
                if _slot_is_hit(s, case, set(pred_slots), combined_text)
            ]
            hit_slots = tuple(hits)
            slot_recall = float(len(hit_slots) / len(case.required_slots)) if case.required_slots else 1.0
    else:
        hit_slots = ()
        slot_recall = None
        false_clarify = 1.0 if raw_action == "clarify" else 0.0

    return AmbiguityEvalResult(
        case_id=case.case_id,
        is_ambiguous_gt=case.is_ambiguous,
        required_slots_gt=case.required_slots,
        action_pred=raw_action,
        missing_slots_pred=pred_slots,
        questions_pred=questions,
        hit_slots=hit_slots,
        parse_ok=parse_ok,
        clarify_correct=clarify_correct,
        slot_recall=slot_recall,
        false_clarify=false_clarify,
        raw_response=raw_response,
        error_message=err_msg,
    )


def evaluate_ambiguity_suite(
    llm_client: BaseLLMClient,
    cases: Optional[List[AmbiguityCase]] = None,
) -> List[AmbiguityEvalResult]:
    suite = cases if cases is not None else AMBIGUITY_CASES
    results: List[AmbiguityEvalResult] = []
    for case in suite:
        prompt = build_ambiguity_prompt(case)
        raw = llm_client.generate(prompt, system_prompt=AMBIGUITY_SYSTEM_PROMPT)
        results.append(score_ambiguity_prediction(case, raw))
    return results


def summarize_ambiguity_results(results: List[AmbiguityEvalResult]) -> Dict[str, Any]:
    if not results:
        return {}
    amb_results = [r for r in results if r.is_ambiguous_gt]
    ctrl_results = [r for r in results if not r.is_ambiguous_gt]
    recalls = [r.slot_recall for r in amb_results if r.slot_recall is not None]
    return {
        "total_cases": len(results),
        "ambiguous_cases": len(amb_results),
        "control_cases": len(ctrl_results),
        "parse_ok_rate": float(np.mean([1.0 if r.parse_ok else 0.0 for r in results])),
        "clarify_accuracy": float(np.mean([r.clarify_correct for r in results])),
        "ambiguous_clarify_rate": (
            float(np.mean([r.clarify_correct for r in amb_results])) if amb_results else 0.0
        ),
        "slot_recall": float(np.mean(recalls)) if recalls else 0.0,
        "false_clarify_rate": (
            float(np.mean([r.false_clarify for r in ctrl_results])) if ctrl_results else 0.0
        ),
    }


class GoldenAmbiguityLLMClient(BaseLLMClient):
    """Oracle client that returns the ground-truth clarify/implement verdict and questions."""

    def __init__(self, cases: Optional[List[AmbiguityCase]] = None):
        self.usage = LLMUsage()
        self._by_id = {c.case_id: c for c in (cases or AMBIGUITY_CASES)}

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        self.usage.calls += 1
        for cid, case in self._by_id.items():
            if f"Case ID: {cid}" in prompt:
                return json.dumps(
                    {
                        "action": "clarify" if case.is_ambiguous else "implement",
                        "questions": list(case.reference_questions),
                        "missing_slots": list(case.required_slots),
                    },
                    ensure_ascii=False,
                )
        return json.dumps({"action": "implement", "questions": [], "missing_slots": []})
