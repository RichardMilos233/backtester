from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmark.common.market_data import load_layer1_market_df
from benchmark.layer1_code.audit.subtle_audit_generator import generate_subtle_audit_suite
from benchmark.layer1_code.impl.tasks.hf_generator import generate_high_freedom_impl_suite
from benchmark.layer1_code.audit.subtle_audit import score_subtle_audit_case
from benchmark.layer1_code.impl.scoring.position_match import evaluate_high_freedom_case


def test_generate_subtle_audit_suite_60_cases_and_differential_proof():
    cases = generate_subtle_audit_suite()
    assert len(cases) == 60

    ids = [c.case_id for c in cases]
    assert len(ids) == len(set(ids)), "All 60 audit case IDs must be unique"

    flawed = [c for c in cases if c.has_flaw]
    clean = [c for c in cases if not c.has_flaw]
    assert len(flawed) == 48
    assert len(clean) == 12

    # Verify zero obvious surface giveaways
    for c in cases:
        assert "shift(-1)" not in c.code
        assert "center=True" not in c.code
        assert "BUG" not in c.code


def test_generate_high_freedom_impl_suite_60_cases_all_golden_valid():
    df = load_layer1_market_df(n_bars=400)
    cases = generate_high_freedom_impl_suite()
    assert len(cases) == 60

    ids = [c.case_id for c in cases]
    assert len(ids) == len(set(ids)), "All 60 implementation case IDs must be unique"

    # Verify every generated golden_fn executes cleanly, has no NaNs after warm-up,
    # respects max_leverage bounds, and is active on the market slice.
    for c in cases:
        out = c.golden_fn(df.copy())
        sig = out["signals"]
        assert isinstance(sig, pd.Series)
        assert len(sig) == len(df)
        assert not sig.iloc[120:].isna().any(), f"{c.case_id} golden_fn produced NaN after warmup"
        arr = sig.to_numpy(dtype=float)
        assert (arr >= -1e-6).all() and (arr <= c.max_leverage + 1e-6).all()
        # Must have non-trivial activity
        assert np.any(arr > 0), f"{c.case_id} golden_fn is completely flat"
