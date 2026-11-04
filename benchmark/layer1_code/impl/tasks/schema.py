"""
Task schema for the Layer 1B implementation benchmark.

A task is a natural-language strategy description plus a hand-written golden
`build_strategy(df)`. `prompt` is the pinned (unambiguous) wording; `original_prompt`
keeps the loose real-world wording when the task was ported from an external source.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional
import pandas as pd


@dataclass(frozen=True)
class HighFreedomImplCase:
    case_id: str
    title: str
    state_complexity: str  # "medium_state" | "long_state_ratchet" | "multi_tf_sizing"
    prompt: str
    golden_fn: Callable[[pd.DataFrame], Dict[str, Any]]
    max_leverage: float = 1.0
    source: str = ""                      # e.g. "quantcode-bench:101"
    original_prompt: Optional[str] = None  # loose real-world wording, same golden as `prompt`
    eval_ticker: str = "SPY"              # daily OHLCV file used for scoring (data/ohlcv_daily/<ticker>.csv)

