from dataclasses import dataclass
from typing import List, Optional, Any
import pandas as pd
from agent.spec import StrategySpec


@dataclass
class ReviewResult:
    passed: bool
    reason: str = ""


def _same_strategy(a: StrategySpec, b: StrategySpec) -> bool:
    """Two specs are duplicates only if signal, params AND the vol-scaling overlay all match."""
    if a.signal_type != b.signal_type or a.params != b.params:
        return False
    if bool(a.use_vol_scaling) != bool(b.use_vol_scaling):
        return False
    if not a.use_vol_scaling:
        return True
    return (a.vol_target or 0.15) == (b.vol_target or 0.15) and a.max_leverage == b.max_leverage


class Reviewer:
    """
    Centralized verification agent that screens strategy specs prior to backtesting.
    Filters invalid parameters, prevents duplicate re-evaluation, enforces consistency,
    and optionally screens against excessive signal correlation with prior strategies.
    """
    def __init__(
        self,
        allow_duplicates: bool = False,
        check_correlation: bool = False,
        correlation_threshold: float = 0.95,
        evaluator: Optional[Any] = None,
    ):
        self.allow_duplicates = allow_duplicates
        self.check_correlation = check_correlation
        self.correlation_threshold = correlation_threshold
        self.evaluator = evaluator

    def review(self, spec: StrategySpec, history: List[StrategySpec]) -> ReviewResult:
        # Gate 1: Spec validation
        is_valid, reason = spec.validate()
        if not is_valid:
            return ReviewResult(passed=False, reason=f"Spec Validation Failed: {reason}")

        # Gate 2: Duplicate detection
        if not self.allow_duplicates:
            for past_spec in history:
                if _same_strategy(past_spec, spec):
                    return ReviewResult(
                        passed=False,
                        reason=f"Duplicate strategy parameters already evaluated: {spec.signal_type} {spec.params}"
                    )

        # Gate 3: Hypothesis sanity check
        if not spec.hypothesis or len(spec.hypothesis.strip()) < 5:
            return ReviewResult(passed=False, reason="Insufficient hypothesis justification.")

        # Gate 4: Signal correlation check (for C-corr topology)
        if self.check_correlation and self.evaluator is not None and history:
            try:
                _, candidate_sig = self.evaluator._generate_signals_for_split(
                    spec, self.evaluator._search_df
                )
                for past_spec in history:
                    _, past_sig = self.evaluator._generate_signals_for_split(
                        past_spec, self.evaluator._search_df
                    )
                    if (candidate_sig == past_sig).all():
                        corr = 1.0
                    else:
                        c = candidate_sig.corr(past_sig)
                        corr = 0.0 if pd.isna(c) else float(c)
                    if abs(corr) >= self.correlation_threshold:
                        return ReviewResult(
                            passed=False,
                            reason=(
                                f"High signal correlation ({corr:.4f} >= {self.correlation_threshold}) "
                                f"with past strategy {past_spec.spec_id}"
                            )
                        )
            except Exception as e:
                return ReviewResult(
                    passed=False,
                    reason=f"Signal generation error during review: {str(e)}"
                )

        return ReviewResult(passed=True, reason="")
