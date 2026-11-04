from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, Optional, Set, Tuple
import pandas as pd

VALID_FLAW_TYPES: Set[str] = {
    "lookahead_bias",
    "friction_erosion",
    "data_snooping",
    "logical_inconsistency",
}


@dataclass(frozen=True)
class BadAlphaCase:
    case_id: str
    title: str
    description: str
    code: str
    has_flaw: bool
    flaw_types: Tuple[str, ...]
    reference_notes: str

    def validate(self) -> None:
        unknown = set(self.flaw_types) - VALID_FLAW_TYPES
        if unknown:
            raise ValueError(f"Case {self.case_id} has unknown flaw_types: {unknown}")
        if self.has_flaw and len(self.flaw_types) == 0:
            raise ValueError(f"Case {self.case_id} has_flaw=True but empty flaw_types")
        if not self.has_flaw and len(self.flaw_types) > 0:
            raise ValueError(f"Case {self.case_id} has_flaw=False but non-empty flaw_types")


@dataclass
class BadAlphaEvalResult:
    case_id: str
    has_flaw_gt: bool
    flaw_types_gt: Tuple[str, ...]
    has_flaw_pred: bool
    flaw_types_pred: Tuple[str, ...]
    parse_ok: bool
    binary_correct: float
    exact_match: float
    category_recall: Optional[float]
    category_precision: float
    category_f1: float
    false_positive_labels: int
    explanation: str = ""
    raw_response: str = ""
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["flaw_types_gt"] = list(self.flaw_types_gt)
        d["flaw_types_pred"] = list(self.flaw_types_pred)
        return d


@dataclass(frozen=True)
class SystematicCase:
    case_id: str
    title: str
    category: str
    prompt: str
    golden_fn: Callable[[pd.DataFrame], Dict[str, Any]]
    expected_factor_calls: Tuple[str, ...] = ()


@dataclass
class SystematicEvalResult:
    case_id: str
    category: str
    exec_pass: bool
    no_lookahead: bool
    signal_agreement: float
    position_agreement: float
    position_corr: float
    friction_match: float
    sharpe_ref: float
    sharpe_agent: float
    sharpe_abs_error: float
    passed: bool
    lookahead_reason: str = ""
    error_message: str = ""
    extracted_code: str = ""
    raw_response: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FactorUsageEvalResult:
    case_id: str
    category: str
    expected_factor_calls: Tuple[str, ...]
    ast_factor_calls: Tuple[str, ...]
    runtime_factor_calls: Tuple[str, ...]
    factor_call_recall: float
    no_reinvented_wheel: bool
    signature_valid: bool
    signature_errors: Tuple[str, ...] = ()
    manual_rolling_ops: Tuple[str, ...] = ()
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["expected_factor_calls"] = list(self.expected_factor_calls)
        d["ast_factor_calls"] = list(self.ast_factor_calls)
        d["runtime_factor_calls"] = list(self.runtime_factor_calls)
        d["signature_errors"] = list(self.signature_errors)
        d["manual_rolling_ops"] = list(self.manual_rolling_ops)
        return d


VALID_AMBIGUITY_SLOTS: Set[str] = {
    "lookback_window",
    "entry_exit_threshold",
    "volatility_definition",
    "execution_timing",
}


@dataclass(frozen=True)
class AmbiguityCase:
    case_id: str
    title: str
    prompt: str
    is_ambiguous: bool
    required_slots: Tuple[str, ...] = ()
    slot_keywords: Dict[str, Tuple[str, ...]] = None  # type: ignore[assignment]
    reference_questions: Tuple[str, ...] = ()

    def validate(self) -> None:
        unknown = set(self.required_slots) - VALID_AMBIGUITY_SLOTS
        if unknown:
            raise ValueError(f"AmbiguityCase {self.case_id} has unknown required_slots: {unknown}")
        if self.is_ambiguous and len(self.required_slots) == 0:
            raise ValueError(f"AmbiguityCase {self.case_id} is_ambiguous=True but empty required_slots")
        if not self.is_ambiguous and len(self.required_slots) > 0:
            raise ValueError(f"AmbiguityCase {self.case_id} is_ambiguous=False but non-empty required_slots")


@dataclass
class AmbiguityEvalResult:
    case_id: str
    is_ambiguous_gt: bool
    required_slots_gt: Tuple[str, ...]
    action_pred: str
    missing_slots_pred: Tuple[str, ...]
    questions_pred: Tuple[str, ...]
    hit_slots: Tuple[str, ...]
    parse_ok: bool
    clarify_correct: float
    slot_recall: Optional[float]
    false_clarify: float
    raw_response: str = ""
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["required_slots_gt"] = list(self.required_slots_gt)
        d["missing_slots_pred"] = list(self.missing_slots_pred)
        d["questions_pred"] = list(self.questions_pred)
        d["hit_slots"] = list(self.hit_slots)
        return d

