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
"""
from itertools import product
from typing import Iterator, List, Optional

from agent.spec import StrategySpec

MOMENTUM_WINDOWS: List[int] = list(range(2, 261))                       # 259
MACD_FAST: List[int] = list(range(4, 64, 4))                            # 15
MACD_SLOW: List[int] = list(range(20, 261, 8))                          # 31
MACD_SPAN: List[int] = [5, 9, 12, 16, 20]                               # 5
VOL_TARGETS: List[Optional[float]] = [None, 0.10, 0.12, 0.15, 0.18]     # None = no scaling
VOL_MAX_LEVERAGE: float = 1.5


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


def iter_space(include_vol: bool = True) -> Iterator[StrategySpec]:
    vols = VOL_TARGETS if include_vol else [None]
    for w in MOMENTUM_WINDOWS:
        for vt in vols:
            yield _with_vol(f"mom_w{w}", "momentum", {"window": w}, vt)
    for f, s, x in product(MACD_FAST, MACD_SLOW, MACD_SPAN):
        if not (1 < f < s):
            continue
        for vt in vols:
            yield _with_vol(
                f"macd_f{f}_s{s}_x{x}", "macd",
                {"fast": f, "slow": s, "signal_span": x}, vt,
            )


def enumerate_space(include_vol: bool = True) -> List[StrategySpec]:
    return list(iter_space(include_vol=include_vol))
