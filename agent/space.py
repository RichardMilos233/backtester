"""
Canonical discrete StrategySpec search space for the benchmark.

Every spec an agent can legally propose (see StrategySpec.validate) is covered
for momentum. MACD is continuous-ish (three free integers), so the benchmark
uses a fixed lattice; agent proposals off-lattice are still valid, but oracle
percentiles are computed against this lattice.

The same enumeration is used by:
  - the oracle (exhaustive Val/Test metrics per spec), and
  - non-LLM baselines (random search, grid search) so they sample from the
    exact space the LLM agents compete in.

Lattice sizes (with regimes):
  - enumerate_space(include_vol=False, include_regimes=False) : 2,404  (no vol, no regimes)
  - enumerate_space(include_vol=True,  include_regimes=False) : 12,020 (with vol, no regimes)
  - With include_regimes=True the regime dimension multiplies the base lattice
    by len(REGIME_OPTIONS) = 9, giving ~108,180 specs. Use sample_space() for that.
"""
from __future__ import annotations

import random as _random
from itertools import product
from typing import Iterator, List, Optional

from agent.spec import StrategySpec

MOMENTUM_WINDOWS: List[int] = list(range(2, 261))                       # 259
MACD_FAST: List[int] = list(range(4, 64, 4))                            # 15
MACD_SLOW: List[int] = list(range(20, 261, 8))                          # 31
MACD_SPAN: List[int] = [5, 9, 12, 16, 20]                               # 5
VOL_TARGETS: List[Optional[float]] = [None, 0.10, 0.12, 0.15, 0.18]     # None = no scaling
VOL_MAX_LEVERAGE: float = 1.5

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


def spec_key(spec: StrategySpec) -> str:
    """Stable identity of a spec that ignores spec_id / hypothesis text."""
    if spec.signal_type == "momentum":
        core = f"mom_w{spec.params['window']}"
    elif spec.signal_type == "macd":
        p = spec.params
        core = f"macd_f{p['fast']}_s{p['slow']}_x{p['signal_span']}"
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


def iter_space(include_vol: bool = True, include_regimes: bool = False) -> Iterator[StrategySpec]:
    """Iterate the canonical lattice.

    include_vol=True adds vol-scaling overlays (5 vol targets per signal config).
    include_regimes=True multiplies each spec by each of REGIME_OPTIONS (9 options).
    Default include_regimes=False preserves the existing 12,020-spec lattice for
    oracle tables already stored on disk.
    """
    regimes = REGIME_OPTIONS if include_regimes else [None]
    vols = VOL_TARGETS if include_vol else [None]
    for w in MOMENTUM_WINDOWS:
        for vt in vols:
            base = _with_vol(f"mom_w{w}", "momentum", {"window": w}, vt)
            for r in regimes:
                yield _with_regime(base, r)
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


def enumerate_space(include_vol: bool = True, include_regimes: bool = False) -> List[StrategySpec]:
    """Return the full lattice as a list.

    include_regimes defaults to False so existing 12,020-spec oracle tables
    remain valid and reproducible. For the regime-extended lattice (~108k specs)
    prefer sample_space() instead of materialising the full list.
    """
    return list(iter_space(include_vol=include_vol, include_regimes=include_regimes))


def _count_space(include_vol: bool = True, include_regimes: bool = False) -> int:
    """Compute the total lattice size analytically (no materialisation)."""
    n_vols = len(VOL_TARGETS) if include_vol else 1
    n_regimes = len(REGIME_OPTIONS) if include_regimes else 1

    # Momentum: one per window
    n_mom = len(MOMENTUM_WINDOWS)

    # MACD: valid (fast, slow) pairs × signal_spans
    n_macd_pairs = sum(
        1
        for f in MACD_FAST
        for s in MACD_SLOW
        if 1 < f < s
    )
    n_macd = n_macd_pairs * len(MACD_SPAN)

    return (n_mom + n_macd) * n_vols * n_regimes


def sample_space(
    n: int,
    seed: int,
    include_vol: bool = True,
    include_regimes: bool = True,
) -> List[StrategySpec]:
    """Draw n distinct specs uniformly from the lattice WITHOUT materialising it.

    The lattice is structured as a product of independent axes:
        (signal_config) × (vol_target) × (regime_option)

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

    Returns
    -------
    List[StrategySpec]  length exactly min(n, total_lattice_size)
    """
    vols = VOL_TARGETS if include_vol else [None]
    regimes = REGIME_OPTIONS if include_regimes else [None]
    n_vols = len(vols)
    n_regimes = len(regimes)

    # Build the ordered list of signal configs (mom/macd × params) without
    # the vol/regime axes — these are the "base" configs.
    base_configs: List[tuple] = []  # (signal_type, params_dict, label)
    for w in MOMENTUM_WINDOWS:
        base_configs.append(("momentum", {"window": w}, f"mom_w{w}"))
    for f, s, x in product(MACD_FAST, MACD_SLOW, MACD_SPAN):
        if not (1 < f < s):
            continue
        base_configs.append(("macd", {"fast": f, "slow": s, "signal_span": x}, f"macd_f{f}_s{s}_x{x}"))

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

