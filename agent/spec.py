from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Tuple

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

    def validate(self) -> Tuple[bool, str]:
        if self.signal_type == "momentum":
            window = self.params.get("window")
            if not isinstance(window, int) or window <= 1 or window > 250:
                return False, f"Invalid momentum window: {window}. Must be int in [2, 250]."
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
        )
