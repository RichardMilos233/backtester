from dataclasses import dataclass
from typing import List
from agent.spec import StrategySpec


@dataclass
class ReviewResult:
    passed: bool
    reason: str = ""


class Reviewer:
    """
    Centralized verification agent that screens strategy specs prior to backtesting.
    Filters invalid parameters, prevents duplicate re-evaluation, and enforces consistency.
    """
    def __init__(self, allow_duplicates: bool = False):
        self.allow_duplicates = allow_duplicates

    def review(self, spec: StrategySpec, history: List[StrategySpec]) -> ReviewResult:
        # Gate 1: Spec validation
        is_valid, reason = spec.validate()
        if not is_valid:
            return ReviewResult(passed=False, reason=f"Spec Validation Failed: {reason}")

        # Gate 2: Duplicate detection
        if not self.allow_duplicates:
            for past_spec in history:
                if (
                    past_spec.signal_type == spec.signal_type and
                    past_spec.params == spec.params
                ):
                    return ReviewResult(
                        passed=False,
                        reason=f"Duplicate strategy parameters already evaluated: {spec.signal_type} {spec.params}"
                    )

        # Gate 3: Hypothesis sanity check
        if not spec.hypothesis or len(spec.hypothesis.strip()) < 5:
            return ReviewResult(passed=False, reason="Insufficient hypothesis justification.")

        return ReviewResult(passed=True, reason="")
