"""
Canonical discrete StrategySpec search space for the benchmark.

LATTICE_VERSION tracks the legal lattice and key encoding. Bump it whenever:
  - a new signal type or parameter dimension is added, or
  - the way a spec is encoded in spec_key changes.
Any code that compares percentiles against an oracle table must verify that
the oracle's lattice_version matches LATTICE_VERSION before using the scores.

v1: momentum + macd only (12,020 specs with vol, no regimes).
v2: adds mean_reversion, breakout, rsi, combo.

enumerate_space() defaults to signal_types=("momentum","macd") so that
existing 12,020-spec wave-1 oracle tables remain valid and reproducible.
To work with the full v2 lattice pass signal_types=None (= all types) or use
baseline_space() / full_lattice_size().
"""
from __future__ import annotations

import random as _random
from itertools import product
from typing import Iterator, List, Optional, Tuple

from benchmark.common.spec import StrategySpec

LATTICE_VERSION = "v2"  # bump whenever the legal lattice or key encoding changes

MOMENTUM_WINDOWS: List[int] = list(range(2, 261))                       # 259
MACD_FAST: List[int] = list(range(4, 64, 4))                            # 15
MACD_SLOW: List[int] = list(range(20, 261, 8))                          # 31
MACD_SPAN: List[int] = [5, 9, 12, 16, 20]                               # 5
VOL_TARGETS: List[Optional[float]] = [None, 0.10, 0.12, 0.15, 0.18]     # None = no scaling
VOL_MAX_LEVERAGE: float = 1.5

# New signal type parameter grids (v2)
MR_WINDOWS: List[int] = [5, 10, 20, 30, 60, 120]
MR_ENTRY_Z: List[float] = [-3.0, -2.0, -1.5, -1.0, -0.5]
MR_EXIT_Z: List[float] = [-1.0, -0.5, 0.0, 0.5]

BO_ENTRY_WINDOWS: List[int] = [10, 20, 55, 100, 150, 200, 260]
BO_EXIT_WINDOWS: List[int] = [5, 10, 20, 55]

RSI_WINDOWS: List[int] = [5, 9, 14, 21, 60]
RSI_LOWERS: List[float] = [10.0, 20.0, 30.0, 40.0, 45.0]
RSI_UPPERS: List[float] = [55.0, 60.0, 70.0, 80.0, 90.0]
RSI_MODES: List[str] = ["reversion", "trend"]

# Regime filter options. None means "no regime filter" (strategy is always in).
# Each list entry is a distinct filter; together with None they cover the nine
# regime-dimension options added by Layer 1.
REGIME_OPTIONS: List[Optional[List[str]]] = [
    None,
    ["Q1", "Q2"],
    ["Q3", "Q4"],
    ["Q1", "Q4"],
    ["Q2", "Q3"],
    ["Q1", "Q2", "Q3"],
    ["Q1", "Q2", "Q4"],
    ["Q1", "Q3", "Q4"],
    ["Q2", "Q3", "Q4"],
]  # 9 options

# All signal types in v2 lattice
ALL_SIGNAL_TYPES: Tuple[str, ...] = ("momentum", "macd", "mean_reversion", "breakout", "rsi", "combo")
# Wave-1 signal types (preserve 12,020-spec oracle compatibility)
LEGACY_SIGNAL_TYPES: Tuple[str, ...] = ("momentum", "macd")


def spec_key(spec: StrategySpec) -> str:
    """Stable identity of a spec that ignores spec_id / hypothesis text."""
    if spec.signal_type == "momentum":
        core = f"mom_w{spec.params['window']}"
    elif spec.signal_type == "macd":
        p = spec.params
        core = f"macd_f{p['fast']}_s{p['slow']}_x{p['signal_span']}"
    elif spec.signal_type == "mean_reversion":
        p = spec.params
        core = f"mr_w{p['window']}_ez{p['entry_z']:g}_xz{p['exit_z']:g}"
    elif spec.signal_type == "breakout":
        p = spec.params
        core = f"bo_en{p['entry_window']}_ex{p['exit_window']}"
    elif spec.signal_type == "rsi":
        p = spec.params
        core = f"rsi_w{p['window']}_lo{p['lower']:g}_up{p['upper']:g}_m{p['mode']}"
    elif spec.signal_type == "combo":
        p = spec.params
        # Sort components by their individual key for order-insensitivity
        comp_keys = sorted(
            f"{c['signal_type']}:{sorted(c.get('params', {}).items())}"
            for c in p["components"]
        )
        core = f"combo_{'|'.join(comp_keys)}_m{p.get('mode', 'all')}"
    else:
        core = f"{spec.signal_type}_{sorted(spec.params.items())}"
    if spec.use_vol_scaling:
        vt = spec.vol_target if spec.vol_target is not None else 0.15
        core += f"_vol{int(round(vt * 100))}_lev{spec.max_leverage:g}"
    if spec.allowed_regimes:
        regimes_tag = "".join(sorted(spec.allowed_regimes))
        core += f"_r{regimes_tag}"
    return core


def _with_vol(spec_id: str, signal_type: str, params: dict, vol_target: Optional[float]) -> StrategySpec:
    if vol_target is None:
        return StrategySpec(spec_id, "oracle enumeration", signal_type, dict(params))
    return StrategySpec(
        f"{spec_id}_vol{int(round(vol_target * 100))}",
        "oracle enumeration",
        signal_type,
        dict(params),
        use_vol_scaling=True,
        vol_target=vol_target,
        max_leverage=VOL_MAX_LEVERAGE,
    )


def _with_regime(base: StrategySpec, allowed_regimes: Optional[List[str]]) -> StrategySpec:
    """Return a copy of base with allowed_regimes set (or the original if None)."""
    if allowed_regimes is None:
        return base
    return StrategySpec(
        spec_id=f"{base.spec_id}_r{''.join(sorted(allowed_regimes))}",
        hypothesis=base.hypothesis,
        signal_type=base.signal_type,
        params=dict(base.params),
        allowed_regimes=list(allowed_regimes),
        use_vol_scaling=base.use_vol_scaling,
        vol_target=base.vol_target,
        max_leverage=base.max_leverage,
    )


def _iter_momentum(vols: list, regimes: list) -> Iterator[StrategySpec]:
    for w in MOMENTUM_WINDOWS:
        for vt in vols:
            base = _with_vol(f"mom_w{w}", "momentum", {"window": w}, vt)
            for r in regimes:
                yield _with_regime(base, r)


def _iter_macd(vols: list, regimes: list) -> Iterator[StrategySpec]:
    for f, s, x in product(MACD_FAST, MACD_SLOW, MACD_SPAN):
        if not (1 < f < s):
            continue
        for vt in vols:
            base = _with_vol(
                f"macd_f{f}_s{s}_x{x}", "macd",
                {"fast": f, "slow": s, "signal_span": x}, vt,
            )
            for r in regimes:
                yield _with_regime(base, r)


def _iter_mean_reversion(vols: list, regimes: list) -> Iterator[StrategySpec]:
    for w in MR_WINDOWS:
        for ez in MR_ENTRY_Z:
            for xz in MR_EXIT_Z:
                if ez >= xz:
                    continue
                p = {"window": w, "entry_z": ez, "exit_z": xz}
                label = f"mr_w{w}_ez{ez:g}_xz{xz:g}"
                for vt in vols:
                    base = _with_vol(label, "mean_reversion", p, vt)
                    for r in regimes:
                        yield _with_regime(base, r)


def _iter_breakout(vols: list, regimes: list) -> Iterator[StrategySpec]:
    for en in BO_ENTRY_WINDOWS:
        for ex in BO_EXIT_WINDOWS:
            if ex > en:
                continue
            p = {"entry_window": en, "exit_window": ex}
            label = f"bo_en{en}_ex{ex}"
            for vt in vols:
                base = _with_vol(label, "breakout", p, vt)
                for r in regimes:
                    yield _with_regime(base, r)


def _iter_rsi(vols: list, regimes: list) -> Iterator[StrategySpec]:
    for w in RSI_WINDOWS:
        for mode in RSI_MODES:
            if mode == "trend":
                # Normalised to lower=30, upper=70 by validate(); enumerate once per window
                p = {"window": w, "lower": 30.0, "upper": 70.0, "mode": "trend"}
                label = f"rsi_w{w}_lo30_up70_mtrend"
                for vt in vols:
                    base = _with_vol(label, "rsi", p, vt)
                    for r in regimes:
                        yield _with_regime(base, r)
            else:
                for lo in RSI_LOWERS:
                    for up in RSI_UPPERS:
                        p = {"window": w, "lower": lo, "upper": up, "mode": "reversion"}
                        label = f"rsi_w{w}_lo{lo:g}_up{up:g}_mreversion"
                        for vt in vols:
                            base = _with_vol(label, "rsi", p, vt)
                            for r in regimes:
                                yield _with_regime(base, r)


def _combo_base_configs() -> List[dict]:
    """Enumerate combo (signal_type, params) pairs for two-component combos."""
    primitives = []
    for w in MOMENTUM_WINDOWS[:10]:  # sample subset for combo lattice
        primitives.append({"signal_type": "momentum", "params": {"window": w}})
    for w in MR_WINDOWS[:3]:
        primitives.append({"signal_type": "mean_reversion",
                            "params": {"window": w, "entry_z": -1.5, "exit_z": 0.0}})
    for en in BO_ENTRY_WINDOWS[:3]:
        primitives.append({"signal_type": "breakout",
                            "params": {"entry_window": en, "exit_window": min(en, 20)}})
    for w in RSI_WINDOWS[:3]:
        primitives.append({"signal_type": "rsi",
                            "params": {"window": w, "lower": 30.0, "upper": 70.0, "mode": "reversion"}})

    configs = []
    # 1-component combos
    for mode in ("all", "any", "mean"):
        for c in primitives:
            configs.append({"components": [c], "mode": mode})
    # 2-component combos (distinct, order-sorted by key for uniqueness)
    for i in range(len(primitives)):
        for j in range(i + 1, len(primitives)):
            for mode in ("all", "any", "majority", "mean"):
                configs.append({"components": [primitives[i], primitives[j]], "mode": mode})
    return configs


def _iter_combo(vols: list, regimes: list) -> Iterator[StrategySpec]:
    for idx, combo_params in enumerate(_combo_base_configs()):
        label = f"combo_{idx}"
        for vt in vols:
            base = _with_vol(label, "combo", combo_params, vt)
            for r in regimes:
                yield _with_regime(base, r)


def iter_space(
    include_vol: bool = True,
    include_regimes: bool = False,
    signal_types: Optional[Tuple[str, ...]] = None,
) -> Iterator[StrategySpec]:
    """Iterate the canonical lattice.

    include_vol=True adds vol-scaling overlays (5 vol targets per signal config).
    include_regimes=True multiplies each spec by each of REGIME_OPTIONS (9 options).
    signal_types: restrict to a specific subset of signal types. Defaults to
        ("momentum","macd") to preserve the 12,020-spec wave-1 oracle lattice.
        Pass signal_types=None or signal_types=ALL_SIGNAL_TYPES for the full v2 lattice.
    """
    if signal_types is None:
        # Keep the legacy default so wave-1 oracle tables remain reproducible
        signal_types = LEGACY_SIGNAL_TYPES

    regimes = REGIME_OPTIONS if include_regimes else [None]
    vols = VOL_TARGETS if include_vol else [None]

    if "momentum" in signal_types:
        yield from _iter_momentum(vols, regimes)
    if "macd" in signal_types:
        yield from _iter_macd(vols, regimes)
    if "mean_reversion" in signal_types:
        yield from _iter_mean_reversion(vols, regimes)
    if "breakout" in signal_types:
        yield from _iter_breakout(vols, regimes)
    if "rsi" in signal_types:
        yield from _iter_rsi(vols, regimes)
    if "combo" in signal_types:
        yield from _iter_combo(vols, regimes)


def enumerate_space(
    include_vol: bool = True,
    include_regimes: bool = False,
    signal_types: Optional[Tuple[str, ...]] = None,
) -> List[StrategySpec]:
    """Return the full lattice as a list.

    signal_types defaults to ("momentum","macd") so existing 12,020-spec oracle
    tables remain valid and reproducible. For the v2 lattice (all signal types)
    prefer sample_space() or baseline_space() instead of materialising the full
    list, as it is now in the millions.
    """
    return list(iter_space(
        include_vol=include_vol,
        include_regimes=include_regimes,
        signal_types=signal_types,
    ))


def _count_signal_type(st: str, include_vol: bool, include_regimes: bool) -> int:
    """Analytical count for a single signal type on the lattice."""
    n_vols = len(VOL_TARGETS) if include_vol else 1
    n_regimes = len(REGIME_OPTIONS) if include_regimes else 1

    if st == "momentum":
        n_base = len(MOMENTUM_WINDOWS)
    elif st == "macd":
        n_base = sum(1 for f in MACD_FAST for s in MACD_SLOW if 1 < f < s) * len(MACD_SPAN)
    elif st == "mean_reversion":
        n_base = sum(
            1
            for w in MR_WINDOWS
            for ez in MR_ENTRY_Z
            for xz in MR_EXIT_Z
            if ez < xz
        )
    elif st == "breakout":
        n_base = sum(
            1
            for en in BO_ENTRY_WINDOWS
            for ex in BO_EXIT_WINDOWS
            if ex <= en
        )
    elif st == "rsi":
        # trend: 1 per window; reversion: all (lower, upper) combos per window
        n_base = len(RSI_WINDOWS) * (1 + len(RSI_LOWERS) * len(RSI_UPPERS))
    elif st == "combo":
        n_base = len(_combo_base_configs())
    else:
        return 0
    return n_base * n_vols * n_regimes


def _count_space(
    include_vol: bool = True,
    include_regimes: bool = False,
    signal_types: Optional[Tuple[str, ...]] = None,
) -> int:
    """Compute the total lattice size analytically (no materialisation).

    signal_types defaults to ("momentum","macd") matching enumerate_space().
    """
    if signal_types is None:
        signal_types = LEGACY_SIGNAL_TYPES
    return sum(_count_signal_type(st, include_vol, include_regimes) for st in signal_types)


def full_lattice_size(include_vol: bool = True, include_regimes: bool = False) -> int:
    """Return the total number of specs in the full v2 lattice (all signal types)."""
    return _count_space(
        include_vol=include_vol,
        include_regimes=include_regimes,
        signal_types=ALL_SIGNAL_TYPES,
    )


def _build_base_configs(
    signal_types: Optional[Tuple[str, ...]] = None,
) -> List[Tuple[str, dict, str]]:
    """Build the ordered list of (signal_type, params, label) without vol/regime axes."""
    if signal_types is None:
        signal_types = LEGACY_SIGNAL_TYPES
    configs: List[Tuple[str, dict, str]] = []
    if "momentum" in signal_types:
        for w in MOMENTUM_WINDOWS:
            configs.append(("momentum", {"window": w}, f"mom_w{w}"))
    if "macd" in signal_types:
        for f, s, x in product(MACD_FAST, MACD_SLOW, MACD_SPAN):
            if not (1 < f < s):
                continue
            configs.append(("macd", {"fast": f, "slow": s, "signal_span": x}, f"macd_f{f}_s{s}_x{x}"))
    if "mean_reversion" in signal_types:
        for w in MR_WINDOWS:
            for ez in MR_ENTRY_Z:
                for xz in MR_EXIT_Z:
                    if ez >= xz:
                        continue
                    configs.append(("mean_reversion",
                                    {"window": w, "entry_z": ez, "exit_z": xz},
                                    f"mr_w{w}_ez{ez:g}_xz{xz:g}"))
    if "breakout" in signal_types:
        for en in BO_ENTRY_WINDOWS:
            for ex in BO_EXIT_WINDOWS:
                if ex > en:
                    continue
                configs.append(("breakout",
                                 {"entry_window": en, "exit_window": ex},
                                 f"bo_en{en}_ex{ex}"))
    if "rsi" in signal_types:
        for w in RSI_WINDOWS:
            # trend normalised
            configs.append(("rsi",
                             {"window": w, "lower": 30.0, "upper": 70.0, "mode": "trend"},
                             f"rsi_w{w}_lo30_up70_mtrend"))
            for lo in RSI_LOWERS:
                for up in RSI_UPPERS:
                    configs.append(("rsi",
                                     {"window": w, "lower": lo, "upper": up, "mode": "reversion"},
                                     f"rsi_w{w}_lo{lo:g}_up{up:g}_mreversion"))
    if "combo" in signal_types:
        for idx, cp in enumerate(_combo_base_configs()):
            configs.append(("combo", cp, f"combo_{idx}"))
    return configs


def sample_space(
    n: int,
    seed: int,
    include_vol: bool = True,
    include_regimes: bool = True,
    signal_types: Optional[Tuple[str, ...]] = None,
) -> List[StrategySpec]:
    """Draw n distinct specs uniformly from the lattice WITHOUT materialising it.

    The lattice is structured as a product of independent axes:
        (signal_config) x (vol_target) x (regime_option)

    We compute the total size analytically, draw n distinct integer indices in
    [0, total), then decode each index into a spec on the fly.  This is O(n) in
    both time and memory regardless of total lattice size.

    Parameters
    ----------
    n : int
        Number of distinct specs to draw.
    seed : int
        RNG seed for reproducibility.
    include_vol : bool
        Whether to include vol-scaling overlays.
    include_regimes : bool
        Whether to include regime filter options.
    signal_types : tuple or None
        Restrict to these signal types. Defaults to LEGACY_SIGNAL_TYPES
        (momentum+macd) when None, matching the enumerate_space default.
        Pass ALL_SIGNAL_TYPES for the full v2 lattice.

    Returns
    -------
    List[StrategySpec]  length exactly min(n, total_lattice_size)
    """
    vols = VOL_TARGETS if include_vol else [None]
    regimes = REGIME_OPTIONS if include_regimes else [None]
    n_vols = len(vols)
    n_regimes = len(regimes)

    base_configs = _build_base_configs(signal_types)
    n_base = len(base_configs)
    total = n_base * n_vols * n_regimes

    n = min(n, total)
    rng = _random.Random(seed)
    indices = rng.sample(range(total), n)

    specs: List[StrategySpec] = []
    for idx in indices:
        # Decode: index = base_i * (n_vols * n_regimes) + vol_i * n_regimes + reg_i
        reg_i = idx % n_regimes
        idx //= n_regimes
        vol_i = idx % n_vols
        base_i = idx // n_vols

        signal_type, params, label = base_configs[base_i]
        vt = vols[vol_i]
        r = regimes[reg_i]

        base = _with_vol(label, signal_type, params, vt)
        spec = _with_regime(base, r)
        specs.append(spec)

    return specs


def baseline_space(
    n: Optional[int] = None,
    seed: int = 42,
    include_vol: bool = True,
    include_regimes: bool = True,
) -> List[StrategySpec]:
    """Return the canonical lattice the current code considers a baseline.

    Uses ALL_SIGNAL_TYPES (full v2 lattice). When the lattice is too large to
    materialise (typically the case with all signal types), pass n to draw a
    uniform sample of that size.  When n is None and the lattice fits in memory
    it is returned in full; otherwise a 20k sample is returned automatically.

    Baselines (R-random, R-grid) and the oracle must use the same lattice so
    their percentile comparisons are apples-to-apples.
    """
    total = full_lattice_size(include_vol=include_vol, include_regimes=include_regimes)
    if n is None:
        _threshold = 20_000
        if total <= _threshold:
            return list(iter_space(
                include_vol=include_vol,
                include_regimes=include_regimes,
                signal_types=ALL_SIGNAL_TYPES,
            ))
        n = _threshold
    return sample_space(
        n=n,
        seed=seed,
        include_vol=include_vol,
        include_regimes=include_regimes,
        signal_types=ALL_SIGNAL_TYPES,
    )
