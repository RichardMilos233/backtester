from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Tuple

VALID_REGIMES = {"Q1", "Q2", "Q3", "Q4"}
# All primitive signal types (no nested combos allowed inside combo).
PRIMITIVE_SIGNAL_TYPES = {"momentum", "macd", "mean_reversion", "breakout", "rsi"}
VALID_SIGNAL_TYPES = PRIMITIVE_SIGNAL_TYPES | {"combo"}


def _validate_mean_reversion_params(params: Dict[str, Any]) -> Tuple[bool, str]:
    window = params.get("window")
    entry_z = params.get("entry_z")
    exit_z = params.get("exit_z")
    if not isinstance(window, int) or window < 5 or window > 120:
        return False, f"mean_reversion window must be int in [5, 120], got {window!r}."
    if not isinstance(entry_z, (int, float)) or not (-3.0 <= entry_z <= -0.5):
        return False, f"mean_reversion entry_z must be float in [-3, -0.5], got {entry_z!r}."
    if not isinstance(exit_z, (int, float)) or not (-1.0 <= exit_z <= 0.5):
        return False, f"mean_reversion exit_z must be float in [-1, 0.5], got {exit_z!r}."
    if entry_z >= exit_z:
        return False, f"mean_reversion requires entry_z < exit_z, got entry_z={entry_z}, exit_z={exit_z}."
    return True, ""


def _validate_breakout_params(params: Dict[str, Any]) -> Tuple[bool, str]:
    entry_window = params.get("entry_window")
    exit_window = params.get("exit_window")
    if not isinstance(entry_window, int) or entry_window < 10 or entry_window > 260:
        return False, f"breakout entry_window must be int in [10, 260], got {entry_window!r}."
    if not isinstance(exit_window, int) or exit_window < 5 or exit_window > 130:
        return False, f"breakout exit_window must be int in [5, 130], got {exit_window!r}."
    if exit_window > entry_window:
        return False, f"breakout requires exit_window <= entry_window, got exit_window={exit_window}, entry_window={entry_window}."
    return True, ""


def _validate_rsi_params(params: Dict[str, Any]) -> Tuple[bool, str]:
    window = params.get("window")
    mode = params.get("mode", "reversion")
    if not isinstance(window, int) or window < 5 or window > 60:
        return False, f"rsi window must be int in [5, 60], got {window!r}."
    if mode not in ("reversion", "trend"):
        return False, f"rsi mode must be 'reversion' or 'trend', got {mode!r}."
    if mode == "reversion":
        lower = params.get("lower")
        upper = params.get("upper")
        if not isinstance(lower, (int, float)) or not (10.0 <= lower <= 45.0):
            return False, f"rsi lower must be float in [10, 45], got {lower!r}."
        if not isinstance(upper, (int, float)) or not (55.0 <= upper <= 90.0):
            return False, f"rsi upper must be float in [55, 90], got {upper!r}."
    return True, ""


def _normalize_rsi_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise rsi params: mode='trend' gets default lower/upper so spec_key stays stable."""
    params = dict(params)
    if params.get("mode") == "trend":
        params.setdefault("lower", 30.0)
        params.setdefault("upper", 70.0)
    return params


def _validate_combo_params(params: Dict[str, Any]) -> Tuple[bool, str]:
    components = params.get("components")
    mode = params.get("mode", "all")
    if not isinstance(components, list):
        return False, "combo params must include 'components' as a list."
    if len(components) == 0 or len(components) > 2:
        return False, f"combo requires 1 or 2 components, got {len(components)}."
    if mode not in ("all", "any", "majority", "mean"):
        return False, f"combo mode must be one of 'all', 'any', 'majority', 'mean', got {mode!r}."
    seen_keys: List[str] = []
    for i, comp in enumerate(components):
        if not isinstance(comp, dict):
            return False, f"combo component {i} must be a dict."
        st = comp.get("signal_type")
        if st not in PRIMITIVE_SIGNAL_TYPES:
            return False, (
                f"combo component {i} signal_type must be a primitive "
                f"(no nested combos), got {st!r}. Allowed: {sorted(PRIMITIVE_SIGNAL_TYPES)}."
            )
        p = comp.get("params", {})
        # Validate each component's params using the respective validator
        if st == "momentum":
            ok, msg = (True, "") if (isinstance(p.get("window"), int) and 2 <= p["window"] <= 260) \
                else (False, f"combo component {i}: Invalid momentum window: {p.get('window')}. Must be int in [2, 260].")
        elif st == "macd":
            fast, slow, span = p.get("fast"), p.get("slow"), p.get("signal_span")
            if not (isinstance(fast, int) and isinstance(slow, int) and isinstance(span, int)):
                ok, msg = False, f"combo component {i}: MACD parameters must be integers."
            elif not (1 < fast < slow) or span <= 1:
                ok, msg = False, f"combo component {i}: Invalid MACD parameters: fast={fast}, slow={slow}, span={span}."
            else:
                ok, msg = True, ""
        elif st == "mean_reversion":
            ok, msg = _validate_mean_reversion_params(p)
            if not ok:
                msg = f"combo component {i}: {msg}"
        elif st == "breakout":
            ok, msg = _validate_breakout_params(p)
            if not ok:
                msg = f"combo component {i}: {msg}"
        elif st == "rsi":
            # normalise before validating
            p = _normalize_rsi_params(p)
            ok, msg = _validate_rsi_params(p)
            if not ok:
                msg = f"combo component {i}: {msg}"
        else:
            ok, msg = False, f"combo component {i}: unsupported signal_type {st!r}."
        if not ok:
            return False, msg
        # Build a stable component key for duplicate detection
        comp_key = _component_key({"signal_type": st, "params": p})
        seen_keys.append(comp_key)
    if len(seen_keys) == 2 and seen_keys[0] == seen_keys[1]:
        return False, "combo components must not be duplicates."
    return True, ""


def _component_key(comp: Dict[str, Any]) -> str:
    """Stable, order-insensitive string key for a combo component."""
    st = comp["signal_type"]
    p = dict(comp.get("params", {}))
    if st == "rsi":
        p = _normalize_rsi_params(p)
    return f"{st}:{sorted(p.items())}"


@dataclass
class StrategySpec:
    spec_id: str
    hypothesis: str
    signal_type: str
    params: Dict[str, Any]
    allowed_regimes: Optional[List[str]] = None
    use_vol_scaling: bool = False
    vol_target: Optional[float] = None
    max_leverage: float = 1.0
    is_fallback: bool = False

    def validate(self) -> Tuple[bool, str]:
        if self.signal_type == "momentum":
            window = self.params.get("window")
            if not isinstance(window, int) or window <= 1 or window > 260:
                return False, f"Invalid momentum window: {window}. Must be int in [2, 260]."
        elif self.signal_type == "macd":
            fast = self.params.get("fast")
            slow = self.params.get("slow")
            span = self.params.get("signal_span")
            if not (isinstance(fast, int) and isinstance(slow, int) and isinstance(span, int)):
                return False, f"MACD parameters must be integers: fast={fast}, slow={slow}, signal_span={span}"
            if not (1 < fast < slow) or span <= 1:
                return False, f"Invalid MACD parameters: fast={fast}, slow={slow}, span={span}. Must satisfy 1 < fast < slow and span > 1."
        elif self.signal_type == "mean_reversion":
            ok, msg = _validate_mean_reversion_params(self.params)
            if not ok:
                return False, msg
        elif self.signal_type == "breakout":
            ok, msg = _validate_breakout_params(self.params)
            if not ok:
                return False, msg
        elif self.signal_type == "rsi":
            # Normalise trend mode in-place so spec_key is stable
            self.params = _normalize_rsi_params(self.params)
            ok, msg = _validate_rsi_params(self.params)
            if not ok:
                return False, msg
        elif self.signal_type == "combo":
            ok, msg = _validate_combo_params(self.params)
            if not ok:
                return False, msg
            # Normalise rsi params inside combo components
            components = self.params.get("components", [])
            for comp in components:
                if comp.get("signal_type") == "rsi":
                    comp["params"] = _normalize_rsi_params(comp.get("params", {}))
        else:
            return False, (
                f"Unsupported signal_type: '{self.signal_type}'. "
                f"Supported: {sorted(VALID_SIGNAL_TYPES)}."
            )

        if self.allowed_regimes is not None:
            if not isinstance(self.allowed_regimes, (list, tuple)) or len(self.allowed_regimes) == 0:
                return False, (
                    f"allowed_regimes must be None or a non-empty list drawn from "
                    f"{sorted(VALID_REGIMES)}, got {self.allowed_regimes!r}."
                )
            deduped = sorted(set(self.allowed_regimes))
            unknown = [r for r in deduped if r not in VALID_REGIMES]
            if unknown:
                return False, (
                    f"Unknown regime(s) {unknown} in allowed_regimes. "
                    f"Must be a subset of {sorted(VALID_REGIMES)}."
                )
            # Normalize in place: deduped, sorted. Downstream (evaluator, spec_key)
            # relies on this canonical form.
            self.allowed_regimes = deduped

        if self.use_vol_scaling:
            effective_vt = self.vol_target if self.vol_target is not None else 0.15
            if not (0.05 <= effective_vt <= 0.40):
                return False, (
                    f"vol_target must be in [0.05, 0.40] (annualised) when use_vol_scaling=True, "
                    f"got {self.vol_target}."
                )
            if not (0.5 <= self.max_leverage <= 2.0):
                return False, (
                    f"max_leverage must be in [0.5, 2.0] when use_vol_scaling=True, "
                    f"got {self.max_leverage}."
                )
        else:
            if self.max_leverage != 1.0:
                return False, (
                    f"max_leverage must be 1.0 when use_vol_scaling=False, got {self.max_leverage}."
                )

        return True, ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategySpec":
        return cls(
            spec_id=data["spec_id"],
            hypothesis=data.get("hypothesis", ""),
            signal_type=data["signal_type"],
            params=dict(data.get("params", {})),
            allowed_regimes=data.get("allowed_regimes"),
            use_vol_scaling=data.get("use_vol_scaling", False),
            vol_target=data.get("vol_target"),
            max_leverage=float(data.get("max_leverage", 1.0)),
            is_fallback=bool(data.get("is_fallback", False)),
        )
