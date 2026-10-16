from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.run_layer1_benchmark import load_layer1_market_df
from agent.layer1.v3_hard_impl import (
    V3_HARD_IMPL_CASES,
    evaluate_v3_hard_impl_case,
)


def test_v3_hard_impl_cases_no_code_hints_and_golden_valid():
    assert len(V3_HARD_IMPL_CASES) == 14
    ids = [c.case_id for c in V3_HARD_IMPL_CASES]
    assert len(ids) == len(set(ids))

    df = load_layer1_market_df(n_bars=600)

    # 1. Verify prompts contain ZERO pandas/python code hints
    forbidden_tokens = [
        "rolling(",
        ".ewm(",
        ".shift(",
        "min_periods",
        "iloc[",
        "to_numpy",
        "is_month_end",
        "resample(",
    ]
    for case in V3_HARD_IMPL_CASES:
        for tok in forbidden_tokens:
            assert tok not in case.prompt, f"{case.case_id} prompt leaked code token {tok!r}"

        # 2. Verify golden_fn executes, is causal, has active trades, and passes self-evaluation
        out = case.golden_fn(df.copy())
        sig = out["signals"]
        assert isinstance(sig, pd.Series)
        assert len(sig) == len(df)
        assert not sig.iloc[120:].isna().any()
        assert np.any(sig.to_numpy() > 0), f"{case.case_id} golden_fn has zero active bars"
