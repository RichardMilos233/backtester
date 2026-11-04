#!/usr/bin/env python3
"""Deterministic zero-drift GBM series for the SYNTH_RW benchmark task."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
out = REPO_ROOT / "data/ohlcv_daily/SYNTH_RW.csv"
rng = np.random.default_rng(20261002)
n = 2950
dates = pd.bdate_range("2015-01-02", periods=n)
daily_vol = 0.15 / np.sqrt(252)
log_ret = rng.normal(-0.5 * daily_vol**2, daily_vol, n)   # zero expected simple return
close = 100 * np.exp(np.cumsum(log_ret))
gap = rng.normal(0, daily_vol * 0.3, n)
open_ = np.r_[100.0, close[:-1]] * np.exp(gap)
intra = np.abs(rng.normal(0, daily_vol * 0.5, n))
high = np.maximum(open_, close) * np.exp(intra)
low = np.minimum(open_, close) * np.exp(-intra)
df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                   "volume": rng.integers(5_000_000, 50_000_000, n)}, index=dates)
df.index.name = "date"
df.to_csv(out, float_format="%.6f")
print(f"{len(df)} rows -> {out}; total return {close[-1]/100-1:+.1%}")
