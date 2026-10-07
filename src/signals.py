import numpy as np
import pandas as pd


def compute_moving_average_signal(df: pd.DataFrame, window: int = 50) -> pd.Series:
    sma = df['close'].rolling(window).mean()
    return (df['close'] > sma).astype(int)


def compute_macd_signal(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal_span: int = 9,
) -> pd.Series:
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal_span, adjust=False).mean()
    return (dif > dea).astype(int)


def compute_volatility_scaled_signal(
    df: pd.DataFrame,
    base_signal: pd.Series,
    target_vol: float = 0.12,
    max_leverage: float = 1.0,
) -> pd.Series:
    vol = df['rolling_vol'].replace(0, np.nan)
    scale = (target_vol / vol).clip(lower=0.0, upper=max_leverage).fillna(1.0)
    return scale * base_signal


def compute_mean_reversion_signal(
    df: pd.DataFrame,
    window: int = 20,
    entry_z: float = -1.0,
    exit_z: float = 0.0,
) -> pd.Series:
    """Long-only mean reversion: enter when the z-score of close vs its
    rolling mean/std drops to entry_z or below, exit once it recovers to
    exit_z or above. Stateful; uses only data up to and including day t.
    """
    if entry_z >= exit_z:
        raise ValueError("entry_z must be strictly less than exit_z")

    mean = df['close'].rolling(window).mean()
    std = df['close'].rolling(window).std()
    z = (df['close'] - mean) / std
    z_values = z.to_numpy()

    out = np.zeros(len(df), dtype=int)
    state = 0
    for i in range(len(df)):
        zi = z_values[i]
        if np.isnan(zi):
            out[i] = 0
            continue
        if state == 0:
            if zi <= entry_z:
                state = 1
        else:
            if zi >= exit_z:
                state = 0
        out[i] = state

    return pd.Series(out, index=df.index, dtype=int)


def compute_breakout_signal(
    df: pd.DataFrame,
    entry_window: int = 55,
    exit_window: int = 20,
) -> pd.Series:
    """Donchian channel breakout. Enters long when close breaks above the
    highest high of the prior entry_window days (today's bar excluded via
    shift(1)); exits when close breaks below the lowest low of the prior
    exit_window days. Stateful long/flat.
    """
    if entry_window <= 0 or exit_window <= 0:
        raise ValueError("entry_window and exit_window must be positive")

    entry_high = df['high'].rolling(entry_window).max().shift(1)
    exit_low = df['low'].rolling(exit_window).min().shift(1)

    close_values = df['close'].to_numpy()
    entry_high_values = entry_high.to_numpy()
    exit_low_values = exit_low.to_numpy()

    out = np.zeros(len(df), dtype=int)
    state = 0
    for i in range(len(df)):
        c = close_values[i]
        eh = entry_high_values[i]
        el = exit_low_values[i]
        if state == 0:
            if not np.isnan(eh) and c > eh:
                state = 1
        else:
            if not np.isnan(el) and c < el:
                state = 0
        out[i] = state

    return pd.Series(out, index=df.index, dtype=int)


def compute_rsi_signal(
    df: pd.DataFrame,
    window: int = 14,
    lower: float = 30.0,
    upper: float = 70.0,
    mode: str = "reversion",
) -> pd.Series:
    """Wilder RSI signal. mode="reversion" is stateful: long when RSI drops
    below `lower`, flat again once RSI rises above `upper`. mode="trend" is
    stateless: long whenever RSI > 50.
    """
    if mode not in ("reversion", "trend"):
        raise ValueError("mode must be 'reversion' or 'trend'")
    if lower >= upper:
        raise ValueError("lower must be strictly less than upper")

    delta = df['close'].diff()
    gain = delta.clip(lower=0.0).fillna(0.0)
    loss = (-delta.clip(upper=0.0)).fillna(0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False).mean()

    avg_loss_safe = avg_loss.replace(0.0, np.nan)
    rs = avg_gain / avg_loss_safe
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi = rsi.fillna(100.0)
    rsi_values = rsi.to_numpy()

    out = np.zeros(len(df), dtype=int)
    state = 0
    for i in range(len(df)):
        if i < window:
            out[i] = 0
            continue
        r = rsi_values[i]
        if mode == "trend":
            out[i] = 1 if r > 50.0 else 0
            continue
        if state == 0:
            if r < lower:
                state = 1
        else:
            if r > upper:
                state = 0
        out[i] = state

    return pd.Series(out, index=df.index, dtype=int)


def combine_signals(signals: list, mode: str = "all") -> pd.Series:
    """Combine 0/1 long-only signals. mode="all" requires every component to
    be long; "any" requires at least one; "majority" requires more than half.
    Output is aligned on the union of all input indices, missing values
    filled with 0.
    """
    if not signals:
        raise ValueError("signals must be a non-empty list of Series")
    if mode not in ("all", "any", "majority"):
        raise ValueError("mode must be one of 'all', 'any', 'majority'")

    idx = signals[0].index
    for s in signals[1:]:
        idx = idx.union(s.index)

    aligned = [s.reindex(idx).fillna(0).astype(int) for s in signals]
    stacked = pd.concat(aligned, axis=1)

    if mode == "all":
        out = stacked.astype(bool).all(axis=1)
    elif mode == "any":
        out = stacked.astype(bool).any(axis=1)
    else:
        out = stacked.sum(axis=1) > (len(signals) / 2.0)

    return out.astype(int)
