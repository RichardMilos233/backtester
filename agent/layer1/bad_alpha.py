from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
import numpy as np

from agent.llm import BaseLLMClient, LLMUsage
from agent.layer1.schema import (
    VALID_FLAW_TYPES,
    BadAlphaCase,
    BadAlphaEvalResult,
)
from agent.layer1.cases.bad_alpha_cases import BAD_ALPHA_CASES

BAD_ALPHA_SYSTEM_PROMPT = """You are a senior quantitative strategy auditor.
Given a strategy description and its Python implementation, identify whether it contains any fundamental quantitative flaws.

Valid flaw categories (choose ONLY from this exact list):
- "lookahead_bias": Using future information not available at decision/execution time (e.g., full-sample mean/std/quantile, centered rolling window, unshifted same-bar close signal executed at same-bar open, shift(-1)).
- "friction_erosion": Excessive turnover or zero rebalance tolerance where realistic slippage and commission specified in the problem eat up the strategy's edge.
- "data_snooping": Overfitting via arbitrary conjunctive magic numbers, hardcoded historical crash date exclusions, or tuning parameters on the hold-out test split.
- "logical_inconsistency": Contradictory entry/exit conditions that never trigger or immediately exit, or formulas that implement the exact opposite of the stated economic logic.

If the strategy is sound and free of these flaws, set "has_flaw" to false and "flaw_types" to [].

Output ONLY valid JSON matching this schema:
{
  "has_flaw": <bool>,
  "flaw_types": [<zero or more strings from the valid flaw categories>],
  "explanation": "<concise explanation of the flaw(s) or why the strategy is clean>"
}
"""


def build_bad_alpha_prompt(case: BadAlphaCase) -> str:
    return (
        f"Case ID: {case.case_id}\n"
        f"Title: {case.title}\n"
        f"Strategy Description:\n{case.description}\n\n"
        f"Python Implementation:\n```python\n{case.code.strip()}\n```\n\n"
        f"Audit this strategy and return the JSON verdict."
    )


def _parse_bad_alpha_json(raw_text: str) -> Dict[str, Any]:
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


def score_bad_alpha_prediction(case: BadAlphaCase, raw_response: str) -> BadAlphaEvalResult:
    gt_set = set(case.flaw_types)
    try:
        data = _parse_bad_alpha_json(raw_response)
        raw_types = data.get("flaw_types", [])
        if not isinstance(raw_types, list):
            raw_types = []
        pred_set = {str(t).strip() for t in raw_types if str(t).strip() in VALID_FLAW_TYPES}
        has_flaw_pred = bool(data.get("has_flaw", len(pred_set) > 0))
        if not has_flaw_pred:
            pred_set = set()
        explanation = str(data.get("explanation", ""))
        parse_ok = True
        err_msg = ""
    except Exception as e:
        pred_set = set()
        has_flaw_pred = False
        explanation = ""
        parse_ok = False
        err_msg = str(e)

    binary_correct = 1.0 if (has_flaw_pred == case.has_flaw) else 0.0
    exact_match = 1.0 if (pred_set == gt_set and has_flaw_pred == case.has_flaw) else 0.0

    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)

    if len(gt_set) > 0:
        recall: Optional[float] = tp / len(gt_set)
    else:
        recall = None

    if len(pred_set) == 0 and len(gt_set) == 0:
        precision = 1.0 if not has_flaw_pred else 0.0
        f1 = 1.0 if not has_flaw_pred else 0.0
    elif len(pred_set) == 0 or len(gt_set) == 0:
        precision = 0.0
        f1 = 0.0
    else:
        precision = tp / len(pred_set)
        rec_val = tp / len(gt_set)
        f1 = (2.0 * precision * rec_val / (precision + rec_val)) if (precision + rec_val) > 0 else 0.0

    # If a clean case is flagged as has_flaw=True even with empty flaw_types, count at least 1 FP
    if not case.has_flaw and has_flaw_pred and fp == 0:
        fp = 1

    return BadAlphaEvalResult(
        case_id=case.case_id,
        has_flaw_gt=case.has_flaw,
        flaw_types_gt=tuple(sorted(gt_set)),
        has_flaw_pred=has_flaw_pred,
        flaw_types_pred=tuple(sorted(pred_set)),
        parse_ok=parse_ok,
        binary_correct=binary_correct,
        exact_match=exact_match,
        category_recall=recall,
        category_precision=precision,
        category_f1=f1,
        false_positive_labels=fp,
        explanation=explanation,
        raw_response=raw_response,
        error_message=err_msg,
    )


def evaluate_bad_alpha_suite(
    llm_client: BaseLLMClient,
    cases: Optional[List[BadAlphaCase]] = None,
) -> List[BadAlphaEvalResult]:
    suite = cases if cases is not None else BAD_ALPHA_CASES
    results: List[BadAlphaEvalResult] = []
    for case in suite:
        prompt = build_bad_alpha_prompt(case)
        raw = llm_client.generate(prompt, system_prompt=BAD_ALPHA_SYSTEM_PROMPT)
        results.append(score_bad_alpha_prediction(case, raw))
    return results


def summarize_bad_alpha_results(results: List[BadAlphaEvalResult]) -> Dict[str, Any]:
    if not results:
        return {}
    recalls = [r.category_recall for r in results if r.category_recall is not None]
    clean_results = [r for r in results if not r.has_flaw_gt]
    clean_fp_rate = (
        float(np.mean([1.0 if (r.has_flaw_pred or r.false_positive_labels > 0) else 0.0 for r in clean_results]))
        if clean_results
        else 0.0
    )
    return {
        "total_cases": len(results),
        "parse_ok_rate": float(np.mean([1.0 if r.parse_ok else 0.0 for r in results])),
        "binary_accuracy": float(np.mean([r.binary_correct for r in results])),
        "exact_match_rate": float(np.mean([r.exact_match for r in results])),
        "category_recall": float(np.mean(recalls)) if recalls else 0.0,
        "category_precision": float(np.mean([r.category_precision for r in results])),
        "category_f1": float(np.mean([r.category_f1 for r in results])),
        "clean_false_positive_rate": clean_fp_rate,
        "total_false_positive_labels": int(sum(r.false_positive_labels for r in results)),
    }


class GoldenBadAlphaLLMClient(BaseLLMClient):
    """Oracle client that returns the ground-truth flaw labels for testing and verification."""

    def __init__(self, cases: Optional[List[BadAlphaCase]] = None):
        self.usage = LLMUsage()
        self._by_id = {c.case_id: c for c in (cases or BAD_ALPHA_CASES)}

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        self.usage.calls += 1
        for cid, case in self._by_id.items():
            if f"Case ID: {cid}" in prompt:
                return json.dumps({
                    "has_flaw": case.has_flaw,
                    "flaw_types": list(case.flaw_types),
                    "explanation": case.reference_notes,
                })
        return json.dumps({"has_flaw": False, "flaw_types": [], "explanation": "Unknown case"})
