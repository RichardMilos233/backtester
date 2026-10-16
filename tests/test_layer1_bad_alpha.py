from __future__ import annotations

import json
import pytest

from agent.layer1.schema import VALID_FLAW_TYPES, BadAlphaCase
from agent.layer1.cases.bad_alpha_cases import BAD_ALPHA_CASES
from agent.layer1.bad_alpha import (
    build_bad_alpha_prompt,
    score_bad_alpha_prediction,
    evaluate_bad_alpha_suite,
    summarize_bad_alpha_results,
    GoldenBadAlphaLLMClient,
)


def test_bad_alpha_cases_count_and_distribution():
    assert len(BAD_ALPHA_CASES) == 18

    ids = [c.case_id for c in BAD_ALPHA_CASES]
    assert len(ids) == len(set(ids)), "Case IDs must be unique"

    for c in BAD_ALPHA_CASES:
        c.validate()
        assert len(c.description.strip()) > 10
        assert len(c.code.strip()) > 20

    clean_cases = [c for c in BAD_ALPHA_CASES if not c.has_flaw]
    single_flaw = [c for c in BAD_ALPHA_CASES if c.has_flaw and len(c.flaw_types) == 1]
    compound_flaw = [c for c in BAD_ALPHA_CASES if c.has_flaw and len(c.flaw_types) >= 2]

    assert len(clean_cases) == 3
    assert len(single_flaw) == 12
    assert len(compound_flaw) == 3

    # Each of the 4 flaw types appears 3 times among single-flaw cases
    for ft in VALID_FLAW_TYPES:
        cnt = sum(1 for c in single_flaw if c.flaw_types == (ft,))
        assert cnt == 3, f"Expected 3 single-flaw cases for {ft}, got {cnt}"


def test_score_bad_alpha_prediction_exact_and_partial():
    case_compound = BadAlphaCase(
        case_id="test_comp",
        title="Compound lookahead and snooping",
        description="Test description",
        code="df['sig'] = df['close'].shift(-1)",
        has_flaw=True,
        flaw_types=("lookahead_bias", "data_snooping"),
        reference_notes="Both flaws present",
    )

    # Exact hit
    raw_exact = json.dumps({
        "has_flaw": True,
        "flaw_types": ["lookahead_bias", "data_snooping"],
        "explanation": "Shifts future price and overfits",
    })
    res_exact = score_bad_alpha_prediction(case_compound, raw_exact)
    assert res_exact.parse_ok is True
    assert res_exact.binary_correct == 1.0
    assert res_exact.exact_match == 1.0
    assert res_exact.category_recall == 1.0
    assert res_exact.category_precision == 1.0
    assert res_exact.category_f1 == 1.0
    assert res_exact.false_positive_labels == 0

    # Partial hit + 1 false positive label
    raw_partial = json.dumps({
        "has_flaw": True,
        "flaw_types": ["lookahead_bias", "friction_erosion"],
        "explanation": "Partial",
    })
    res_partial = score_bad_alpha_prediction(case_compound, raw_partial)
    assert res_partial.binary_correct == 1.0
    assert res_partial.exact_match == 0.0
    assert res_partial.category_recall == 0.5
    assert res_partial.category_precision == 0.5
    assert res_partial.category_f1 == pytest.approx(0.5)
    assert res_partial.false_positive_labels == 1


def test_score_bad_alpha_clean_case_false_positive():
    clean_case = [c for c in BAD_ALPHA_CASES if not c.has_flaw][0]

    # Correct clean prediction
    res_ok = score_bad_alpha_prediction(
        clean_case,
        '```json\n{"has_flaw": false, "flaw_types": [], "explanation": "Clean strategy"}\n```',
    )
    assert res_ok.binary_correct == 1.0
    assert res_ok.exact_match == 1.0
    assert res_ok.category_f1 == 1.0
    assert res_ok.false_positive_labels == 0

    # False alarm on clean case
    res_fp = score_bad_alpha_prediction(
        clean_case,
        '{"has_flaw": true, "flaw_types": ["lookahead_bias"], "explanation": "False alarm"}',
    )
    assert res_fp.binary_correct == 0.0
    assert res_fp.exact_match == 0.0
    assert res_fp.category_f1 == 0.0
    assert res_fp.false_positive_labels == 1


def test_golden_bad_alpha_client_achieves_perfect_score():
    client = GoldenBadAlphaLLMClient()
    results = evaluate_bad_alpha_suite(client, BAD_ALPHA_CASES)
    summary = summarize_bad_alpha_results(results)

    assert summary["total_cases"] == 18
    assert summary["binary_accuracy"] == 1.0
    assert summary["exact_match_rate"] == 1.0
    assert summary["category_recall"] == 1.0
    assert summary["category_f1"] == 1.0
    assert summary["clean_false_positive_rate"] == 0.0
