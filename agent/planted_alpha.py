from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.backtest import Backtest
from src.metrics import compute_strategy_metrics
from src.regimes import classify_market_regimes
from src.signals import (
    compute_moving_average_signal,
    compute_macd_signal,
    compute_volatility_scaled_signal,
    compute_mean_reversion_signal,
    compute_breakout_signal,
    compute_rsi_signal,
    combine_signals,
)
from agent.spec import StrategySpec


PLANTED_ALPHA_SPECS: Dict[str, Dict[str, Any]] = {
    "SYNTH_MOM": {
        "description": "Persistent regime-switching trend where momentum (window 20..100) has true Sharpe > 1.2 while mean_reversion loses money.",
        "expected_signal_types": ("momentum", "macd", "breakout"),
        "required_regime": None,
        "is_null_control": False,
    },
    "SYNTH_MREV": {
        "description": "Ornstein-Uhlenbeck mean-reverting series where mean_reversion and rsi(reversion) have true Sharpe > 1.2 while momentum loses money.",
        "expected_signal_types": ("mean_reversion", "rsi"),
        "required_regime": None,
        "is_null_control": False,
    },
    "SYNTH_REGIME_Q1": {
        "description": "Positive trend drift exclusively during high-vol/high-volume Q1 regime bursts and negative whip-saw drift elsewhere.",
        "expected_signal_types": ("momentum", "macd", "breakout", "rsi", "combo"),
        "required_regime": "Q1",
        "is_null_control": False,
    },
    "SYNTH_NULL": {
        "description": "Zero-drift pure random walk control (true Sharpe = 0).",
        "expected_signal_types": (),
        "required_regime": None,
        "is_null_control": True,
    },
}


@dataclass
class PlantedAlphaPickEval:
    task_id: str
    spec_id: str
    signal_type: str
    allowed_regimes: Optional[Tuple[str, ...]]
    is_null_control: bool
    family_match: bool
    regime_match: bool
    planted_hit: bool
    false_positive: bool
    true_sharpe: float
    total_return: float

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["allowed_regimes"] = list(self.allowed_regimes) if self.allowed_regimes else None
        return d


def _build_ohlcv_from_returns(
    rets: np.ndarray,
    volume: np.ndarray,
    dates: pd.DatetimeIndex,
    rng: np.random.Generator,
    intra_scale: float = 0.003,
) -> pd.DataFrame:
    close = 100.0 * np.cumprod(1.0 + rets)
    open_ = np.r_[100.0, close[:-1]]
    spread = np.abs(rng.normal(0.0, intra_scale, len(rets)))
    high = np.maximum(open_, close) * (1.0 + spread)
    low = np.minimum(open_, close) * np.maximum(0.5, 1.0 - spread)
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )
    df.index.name = "date"
    df["return"] = df["close"].pct_change()
    df = classify_market_regimes(df)
    return df


def _generate_synth_mom(n_bars: int, rng: np.random.Generator, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Persistent bull/bear cycle where bull regimes last 120 bars (+0.32%/day drift)
    and bear regimes last 70 bars (-0.30%/day drift) with low daily noise (0.55%).
    Momentum (windows 20..100) rides bull phases and sits flat in bear phases (Sharpe > 1.8),
    while mean_reversion buys into the start of bear phases and loses money (Sharpe < 0).
    """
    rets = np.zeros(n_bars, dtype=float)
    cycle = 190
    for i in range(n_bars):
        phase = i % cycle
        if phase < 120:
            drift = 0.0032
        else:
            drift = -0.0030
        rets[i] = drift + rng.normal(0.0, 0.0055)
    volume = rng.integers(8_000_000, 25_000_000, n_bars)
    return _build_ohlcv_from_returns(rets, volume, dates, rng)


def _generate_synth_mrev(n_bars: int, rng: np.random.Generator, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Ornstein-Uhlenbeck mean-reverting log-price process around 0 with periodic
    oversold capitulation bars and rapid snap-backs. Both `mean_reversion`
    (Z-score <= -1.5) and `rsi(reversion)` (RSI < 30) trigger at the capitulation
    trough and exit on the rebound (Z-score >= 0 and RSI > 70, both Sharpe > 2.5),
    whereas `momentum` turns long near the rebound peak and gets hit by the
    subsequent mean-reverting drop (Sharpe < -1.5).
    """
    rets = np.zeros(n_bars, dtype=float)
    for i in range(1, n_bars):
        mod = i % 26
        if mod == 0:
            # Capitulation bar: drives 20-day Z-score < -1.8 and 14-day Wilder RSI < 28
            rets[i] = -0.045
        elif 1 <= mod <= 10:
            # Mean-reverting recovery: drives Z-score > 0.0
            rets[i] = +0.0045
        elif 11 <= mod <= 12:
            # Late rebound burst: drives 14-day Wilder RSI > 72 so RSI reversion exits at top
            rets[i] = +0.0265
        else:
            # Gentle 13-day pullback (-0.435%/day): RSI stays ~38 (> 30, so RSI stays flat=0)
            # while momentum(20/50) is long=1 after the rebound peak and loses money
            rets[i] = -0.00435

    volume = rng.integers(8_000_000, 25_000_000, n_bars)
    return _build_ohlcv_from_returns(rets, volume, dates, rng)


def _generate_synth_regime_q1(n_bars: int, rng: np.random.Generator, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Positive trend drift occurs exclusively during high-volatility, high-volume Q1
    regime blocks. Outside Q1, false breakouts climb above SMA20 and then suffer
    sharp single-bar reversals while momentum is long, so unfiltered momentum and
    non-Q1 regimes lose money.
    """
    rets = np.zeros(n_bars, dtype=float)
    volume = np.zeros(n_bars, dtype=float)

    for i in range(n_bars):
        if i < 275:
            # Warm-up period for 252-day rolling median: gentle downtrend with alternating vol/volume
            in_q1_seed = (i % 80) < 35
            volume[i] = 35_000_000 if in_q1_seed else 8_000_000
            sigma = 0.025 if in_q1_seed else 0.003
            rets[i] = -0.0015 + (sigma if (i % 2 == 0) else -sigma)
            continue

        cycle_pos = (i - 275) % 100
        if cycle_pos < 32:
            # High-vol, high-volume Q1 expansion block: high 20-day vol + positive net drift
            volume[i] = 48_000_000
            rets[i] = 0.024 if (cycle_pos % 2 == 0) else -0.011
        else:
            # Low-vol, low-volume Q3 bull-trap cycle:
            # Slow climb above SMA20 followed by sudden drops while momentum(20) is long=1
            volume[i] = 6_000_000
            sub = (cycle_pos - 32) % 17
            if sub < 16:
                rets[i] = 0.0015
            else:
                rets[i] = -0.038

    return _build_ohlcv_from_returns(rets, volume, dates, rng)


def _generate_synth_null(n_bars: int, rng: np.random.Generator, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Zero-drift pure random walk control (true Sharpe = 0)."""
    daily_vol = 0.15 / np.sqrt(252.0)
    log_ret = rng.normal(-0.5 * daily_vol**2, daily_vol, n_bars)
    rets = np.exp(log_ret) - 1.0
    volume = rng.integers(5_000_000, 35_000_000, n_bars)
    return _build_ohlcv_from_returns(rets, volume, dates, rng)


def generate_planted_alpha_suite(
    n_bars: int = 1500,
    seed: int = 20261015,
) -> Dict[str, pd.DataFrame]:
    """
     deterministically generate the 4 Layer-3 planted-alpha OHLCV series:
      - `SYNTH_MOM`
      - `SYNTH_MREV`
      - `SYNTH_REGIME_Q1`
      - `SYNTH_NULL`
    """
    dates = pd.bdate_range("2018-01-02", periods=n_bars)
    return {
        "SYNTH_MOM": _generate_synth_mom(n_bars, np.random.default_rng(seed + 1), dates),
        "SYNTH_MREV": _generate_synth_mrev(n_bars, np.random.default_rng(seed + 2), dates),
        "SYNTH_REGIME_Q1": _generate_synth_regime_q1(n_bars, np.random.default_rng(seed + 3), dates),
        "SYNTH_NULL": _generate_synth_null(n_bars, np.random.default_rng(seed + 4), dates),
    }


def _build_signal_for_df(df: pd.DataFrame, spec: StrategySpec) -> pd.Series:
    ok, msg = spec.validate()
    if not ok:
        raise ValueError(f"Invalid StrategySpec {spec.spec_id}: {msg}")

    st = spec.signal_type
    p = spec.params
    if st == "momentum":
        sig = compute_moving_average_signal(df, window=p["window"])
    elif st == "macd":
        sig = compute_macd_signal(df, fast=p["fast"], slow=p["slow"], signal_span=p["signal_span"])
    elif st == "mean_reversion":
        sig = compute_mean_reversion_signal(df, window=p["window"], entry_z=p["entry_z"], exit_z=p["exit_z"])
    elif st == "breakout":
        sig = compute_breakout_signal(df, entry_window=p["entry_window"], exit_window=p["exit_window"])
    elif st == "rsi":
        sig = compute_rsi_signal(
            df, window=p["window"], lower=p["lower"], upper=p["upper"], mode=p["mode"]
        )
    elif st == "combo":
        comp_sigs = []
        for c in p["components"]:
            c_spec = StrategySpec("c", "c", c["signal_type"], dict(c.get("params", {})))
            comp_sigs.append(_build_signal_for_df(df, c_spec))
        sig = combine_signals(comp_sigs, mode=p.get("mode", "all"))
    else:
        raise ValueError(f"Unsupported signal_type: {st}")

    if spec.allowed_regimes:
        mask = df["regime"].isna() | df["regime"].isin(spec.allowed_regimes)
        sig = sig.where(mask, 0)

    if spec.use_vol_scaling:
        sig = compute_volatility_scaled_signal(
            df,
            sig,
            target_vol=spec.vol_target if spec.vol_target is not None else 0.15,
            max_leverage=spec.max_leverage,
        )
    return sig


def compute_planted_series_sharpe(
    df: pd.DataFrame,
    spec: StrategySpec,
    return_total: bool = False,
) -> Union[float, Tuple[float, float]]:
    """Run causal Backtest on `df` for `spec` and return Sharpe (or `(total_return, sharpe)`)."""
    sig = _build_signal_for_df(df, spec)
    bt = Backtest()
    ledger = bt.run(df, sig)
    metrics = compute_strategy_metrics(ledger)
    raw_sharpe = float(metrics.get("sharpe_ratio", 0.0))
    sharpe = 0.0 if (math.isnan(raw_sharpe) or math.isinf(raw_sharpe)) else raw_sharpe
    total_ret = float(metrics.get("total_return", 0.0))
    if return_total:
        return total_ret, sharpe
    return sharpe


def evaluate_planted_alpha_pick(
    task_id: str,
    spec: Optional[StrategySpec],
    df: Optional[pd.DataFrame] = None,
) -> PlantedAlphaPickEval:
    """
    Evaluate whether an agent's selected `StrategySpec` hits the planted ground-truth
    mechanism on `task_id` (`SYNTH_MOM`, `SYNTH_MREV`, `SYNTH_REGIME_Q1`, `SYNTH_NULL`).
    """
    if task_id not in PLANTED_ALPHA_SPECS:
        raise KeyError(f"Unknown planted alpha task_id {task_id!r}. Known: {sorted(PLANTED_ALPHA_SPECS)}")

    meta = PLANTED_ALPHA_SPECS[task_id]
    is_null = bool(meta["is_null_control"])

    if df is None:
        df = generate_planted_alpha_suite()[task_id]

    if spec is None:
        return PlantedAlphaPickEval(
            task_id=task_id,
            spec_id="abstain",
            signal_type="none",
            allowed_regimes=None,
            is_null_control=is_null,
            family_match=is_null,
            regime_match=is_null,
            planted_hit=False,
            false_positive=False,
            true_sharpe=0.0,
            total_return=0.0,
        )

    spec.validate()
    tot_ret, sharpe = compute_planted_series_sharpe(df, spec, return_total=True)
    regimes_tuple = tuple(spec.allowed_regimes) if spec.allowed_regimes else None

    if is_null:
        return PlantedAlphaPickEval(
            task_id=task_id,
            spec_id=spec.spec_id,
            signal_type=spec.signal_type,
            allowed_regimes=regimes_tuple,
            is_null_control=True,
            family_match=False,
            regime_match=False,
            planted_hit=False,
            false_positive=True,
            true_sharpe=sharpe,
            total_return=tot_ret,
        )

    expected_fams = set(meta["expected_signal_types"])
    if spec.signal_type == "rsi" and task_id == "SYNTH_MREV":
        family_match = spec.params.get("mode", "reversion") == "reversion"
    elif spec.signal_type == "combo":
        comp_types = {c.get("signal_type") for c in spec.params.get("components", [])}
        family_match = bool(comp_types & expected_fams) or ("combo" in expected_fams)
    else:
        family_match = spec.signal_type in expected_fams

    req_regime = meta["required_regime"]
    if req_regime is None:
        regime_match = True
    else:
        regime_match = bool(spec.allowed_regimes and req_regime in spec.allowed_regimes)

    planted_hit = bool(family_match and regime_match and sharpe > 1.0)

    return PlantedAlphaPickEval(
        task_id=task_id,
        spec_id=spec.spec_id,
        signal_type=spec.signal_type,
        allowed_regimes=regimes_tuple,
        is_null_control=False,
        family_match=family_match,
        regime_match=regime_match,
        planted_hit=planted_hit,
        false_positive=False,
        true_sharpe=sharpe,
        total_return=tot_ret,
    )


def summarize_planted_alpha_picks(results: List[PlantedAlphaPickEval]) -> Dict[str, Any]:
    if not results:
        return {}
    planted = [r for r in results if not r.is_null_control]
    nulls = [r for r in results if r.is_null_control]
    return {
        "total_tasks": len(results),
        "planted_tasks": len(planted),
        "null_tasks": len(nulls),
        "planted_hit_rate": float(np.mean([1.0 if r.planted_hit else 0.0 for r in planted])) if planted else 0.0,
        "family_match_rate": float(np.mean([1.0 if r.family_match else 0.0 for r in planted])) if planted else 0.0,
        "regime_match_rate": float(np.mean([1.0 if r.regime_match else 0.0 for r in planted])) if planted else 0.0,
        "mean_planted_sharpe": float(np.mean([r.true_sharpe for r in planted])) if planted else 0.0,
        "null_false_positive_rate": float(np.mean([1.0 if r.false_positive else 0.0 for r in nulls])) if nulls else 0.0,
    }
