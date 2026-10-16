from __future__ import annotations

import pandas as pd
import pytest

from scripts.run_layer1_benchmark import load_layer1_market_df
from agent.layer1.canonical import (
    CANONICAL_AUDIT_CASES,
    CANONICAL_IMPL_CASES,
    evaluate_canonical_audit_case,
    evaluate_canonical_impl_case,
    GoldenCanonicalAuditClient,
    GoldenCanonicalImplClient,
)


def test_canonical_suite_counts_and_uniqueness():
    assert len(CANONICAL_AUDIT_CASES) == 20
    assert len(CANONICAL_IMPL_CASES) == 20

    audit_ids = [c.case_id for c in CANONICAL_AUDIT_CASES]
    impl_ids = [c.case_id for c in CANONICAL_IMPL_CASES]
    assert len(audit_ids) == len(set(audit_ids))
    assert len(impl_ids) == len(set(impl_ids))

    flawed = [c for c in CANONICAL_AUDIT_CASES if c.has_flaw]
    clean = [c for c in CANONICAL_AUDIT_CASES if not c.has_flaw]
    assert len(flawed) == 16
    assert len(clean) == 4


def test_golden_clients_achieve_100_percent_on_canonical_suite():
    df = load_layer1_market_df(n_bars=600)
    audit_client = GoldenCanonicalAuditClient()
    impl_client = GoldenCanonicalImplClient()

    for c in CANONICAL_AUDIT_CASES:
        raw = audit_client.generate(f"Case ID: {c.case_id}")
        res = evaluate_canonical_audit_case(c, raw)
        assert res.strict_pass is True, f"Audit {c.case_id} failed golden check"

    for c in CANONICAL_IMPL_CASES:
        raw = impl_client.generate(f"Case ID: {c.case_id}")
        res = evaluate_canonical_impl_case(c, raw, df)
        assert res.passed is True, f"Impl {c.case_id} failed golden check: {res.error_message}"
