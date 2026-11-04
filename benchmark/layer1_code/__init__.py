"""
Layer 1: the agent's strategy-code ability.

  audit/      1A  identify the flawed strategy
  impl/       1B  write the correct strategy (scoring / harness / tasks)
  ambiguity/      ask instead of guessing when a parameter is missing
  canonical.py    the 40-case canonical suite (20 audit + 20 impl) used for the 8-model leaderboard
"""
from __future__ import annotations

from benchmark.layer1_code.schema import (
    VALID_FLAW_TYPES,
    VALID_AMBIGUITY_SLOTS,
    BadAlphaCase,
    BadAlphaEvalResult,
    SystematicCase,
    SystematicEvalResult,
    FactorUsageEvalResult,
    AmbiguityCase,
    AmbiguityEvalResult,
)
from benchmark.layer1_code.impl.scoring.sandbox import (
    extract_python_code,
    compile_build_strategy,
    normalize_strategy_output,
    check_no_lookahead,
)
from benchmark.layer1_code.audit.bad_alpha import (
    BAD_ALPHA_SYSTEM_PROMPT,
    build_bad_alpha_prompt,
    score_bad_alpha_prediction,
    evaluate_bad_alpha_suite,
    summarize_bad_alpha_results,
    GoldenBadAlphaLLMClient,
)
from benchmark.layer1_code.impl.scoring.systematic import (
    SYSTEMATIC_SYSTEM_PROMPT,
    build_systematic_prompt,
    evaluate_systematic_case,
    evaluate_systematic_suite,
    summarize_systematic_results,
    GoldenSystematicLLMClient,
)
from benchmark.layer1_code.impl.scoring.factor_usage import (
    KNOWN_FACTOR_FUNCTIONS,
    evaluate_factor_usage_case,
    evaluate_factor_usage_suite,
    summarize_factor_usage_results,
    score_runs_jsonl_factor_usage,
)
from benchmark.layer1_code.ambiguity.ambiguity import (
    AMBIGUITY_SYSTEM_PROMPT,
    build_ambiguity_prompt,
    score_ambiguity_prediction,
    evaluate_ambiguity_suite,
    summarize_ambiguity_results,
    GoldenAmbiguityLLMClient,
)
from benchmark.layer1_code.audit.bad_alpha_cases import BAD_ALPHA_CASES
from benchmark.layer1_code.impl.tasks.systematic_cases import SYSTEMATIC_CASES
from benchmark.layer1_code.ambiguity.ambiguity_cases import AMBIGUITY_CASES

__all__ = [
    "VALID_FLAW_TYPES", "VALID_AMBIGUITY_SLOTS", "BadAlphaCase", "BadAlphaEvalResult",
    "SystematicCase", "SystematicEvalResult", "FactorUsageEvalResult", "AmbiguityCase", "AmbiguityEvalResult",
    "extract_python_code", "compile_build_strategy", "normalize_strategy_output", "check_no_lookahead",
    "BAD_ALPHA_SYSTEM_PROMPT", "build_bad_alpha_prompt", "score_bad_alpha_prediction",
    "evaluate_bad_alpha_suite", "summarize_bad_alpha_results", "GoldenBadAlphaLLMClient",
    "SYSTEMATIC_SYSTEM_PROMPT", "build_systematic_prompt", "evaluate_systematic_case",
    "evaluate_systematic_suite", "summarize_systematic_results", "GoldenSystematicLLMClient",
    "KNOWN_FACTOR_FUNCTIONS", "evaluate_factor_usage_case", "evaluate_factor_usage_suite",
    "summarize_factor_usage_results", "score_runs_jsonl_factor_usage",
    "AMBIGUITY_SYSTEM_PROMPT", "build_ambiguity_prompt", "score_ambiguity_prediction",
    "evaluate_ambiguity_suite", "summarize_ambiguity_results", "GoldenAmbiguityLLMClient",
    "BAD_ALPHA_CASES", "SYSTEMATIC_CASES", "AMBIGUITY_CASES",
]
