from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Tuple

VALID_REGIMES = {"Q1", "Q2", "Q3", "Q4"}

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
        else:
            return False, f"Unsupported signal_type: '{self.signal_type}'. Supported: ['momentum', 'macd']."

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
