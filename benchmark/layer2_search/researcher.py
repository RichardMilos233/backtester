import json
from typing import Optional, List, Dict, Any
from benchmark.common.spec import StrategySpec
from benchmark.common.llm import BaseLLMClient

PROMPT_VERSION = "v3"

SYSTEM_PROMPT = """You are a quantitative CTA research agent.
Your objective is to propose profitable, sound CTA strategies avoiding overfitting.
You must output ONLY valid JSON matching the following schema:
{
  "spec_id": "<string>",
  "hypothesis": "<string explaining economic/statistical rationale>",
  "signal_type": "<'momentum' | 'macd' | 'mean_reversion' | 'breakout' | 'rsi' | 'combo'>",
  "params": {"<param_name>": <value>},
  "use_vol_scaling": <bool, optional, default false>,
  "vol_target": <float, optional, annualised volatility target e.g. 0.10-0.20>,
  "max_leverage": <float, optional, max leverage multiplier <= 2.0, only used with vol scaling>,
  "allowed_regimes": <list of "Q1".."Q4" or null, optional>
}

Signal type rules:
- momentum params: {"window": int in [2, 260]}
- macd params: {"fast": int, "slow": int, "signal_span": int} with 1 < fast < slow and signal_span > 1
- mean_reversion params: {"window": int in [5, 120], "entry_z": float in [-3, -0.5], "exit_z": float in [-1, 0.5]}
  Requires entry_z < exit_z. Enters long when z-score drops to entry_z, exits once z-score recovers to exit_z.
- breakout params: {"entry_window": int in [10, 260], "exit_window": int in [5, 130]}
  Requires exit_window <= entry_window. Donchian channel breakout (enters on upper breakout, exits on lower).
- rsi params: {"window": int in [5, 60], "lower": float in [10, 45], "upper": float in [55, 90], "mode": "reversion"|"trend"}
  mode="reversion" enters long when RSI < lower, exits when RSI > upper.
  mode="trend" is long whenever RSI > 50; simplified form {"window": int, "mode": "trend"} is accepted
  (lower/upper default to 30/70 internally).
- combo: combine 1 or 2 primitive signals. No nested combos.
  params: {"components": [<component>, ...], "mode": "all"|"any"|"majority"|"mean"}
  Each component: {"signal_type": "<primitive type>", "params": {...}}
  mode "all" = long only when ALL components are long; "any" = at least one; "majority" = more than half;
  "mean" = fractional position averaging component weights.
  Example: {"components": [{"signal_type": "momentum", "params": {"window": 50}},
                            {"signal_type": "rsi", "params": {"window": 14, "lower": 30, "upper": 70, "mode": "reversion"}}],
            "mode": "all"}

Vol scaling and regime rules:
- use_vol_scaling (optional): if true, signal is scaled so realised vol targets vol_target; omit or false to disable
- vol_target (optional): annualised volatility target, e.g. 0.10 to 0.20; ignored unless use_vol_scaling=true
- max_leverage (optional): maximum position scale factor <= 2.0; only applies with vol_scaling; default 1.5
- allowed_regimes (optional): filter which market regimes the strategy trades in.
  Regimes are volume x volatility quadrants relative to rolling 1-year medians:
    Q1 = high-volatility / high-volume (trending, liquid)
    Q2 = high-volatility / low-volume  (volatile, thin)
    Q3 = low-volatility  / low-volume  (quiet, thin)
    Q4 = low-volatility  / high-volume (calm, liquid)
  When set, the strategy is FLAT (signal=0) outside the allowed regimes.
  Null or omitted means trade in all regimes.
"""

class Researcher:
    def __init__(
        self,
        llm_client: BaseLLMClient,
        memory_length: int = 5,
        researcher_id: str = "researcher_0"
    ):
        self.llm_client = llm_client
        self.memory_length = memory_length
        self.researcher_id = researcher_id
        self.memory: List[Dict[str, Any]] = []
        self.proposal_count = 0
        self.last_parse_failed = False

    @property
    def usage(self):
        """Cumulative LLMUsage of the underlying LLM client."""
        return self.llm_client.usage

    def update_memory(self, spec: StrategySpec, metrics: Dict[str, Any], reflection: Optional[str] = None):
        entry = {"spec": spec, "metrics": metrics}
        if reflection:
            entry["reflection"] = reflection
        self.memory.append(entry)
        if len(self.memory) > self.memory_length:
            self.memory.pop(0)

    def _build_prompt(self, rejection_feedback: Optional[str] = None) -> str:
        prompt_lines = [
            f"Researcher ID: {self.researcher_id}",
            f"Proposal Attempt: #{self.proposal_count}",
            "\n--- Historical Memory (Most Recent Strategies & Performance) ---"
        ]
        if not self.memory:
            prompt_lines.append("No prior history. Propose initial baseline candidate.")
        else:
            for item in self.memory:
                spec = item["spec"]
                metrics = item["metrics"]
                # Build overlay description for vol/regime settings
                overlay_parts = []
                if spec.use_vol_scaling:
                    vt = spec.vol_target if spec.vol_target is not None else 0.15
                    overlay_parts.append(f"vol_target={vt}, max_leverage={spec.max_leverage}")
                if spec.allowed_regimes:
                    overlay_parts.append(f"allowed_regimes={spec.allowed_regimes}")
                overlay_str = f" | Overlay: {', '.join(overlay_parts)}" if overlay_parts else ""
                # Render combo params readably
                if spec.signal_type == "combo":
                    comps = spec.params.get("components", [])
                    mode = spec.params.get("mode", "all")
                    comp_strs = [
                        f"{c['signal_type']}({c.get('params', {})})" for c in comps
                    ]
                    params_str = f"combo({' + '.join(comp_strs)}, mode={mode})"
                else:
                    params_str = str(spec.params)
                line = (
                    f"- Spec {spec.spec_id} ({spec.signal_type}, {params_str}{overlay_str}): "
                    f"Metrics: {metrics}"
                )
                if "reflection" in item:
                    line += f" | Reflection: {item['reflection']}"
                prompt_lines.append(line)

        if rejection_feedback:
            prompt_lines.append(
                f"\n[ALERT - PRIOR PROPOSAL REJECTED BY REVIEWER]:\n"
                f"Reason: {rejection_feedback}\n"
                f"Please fix the issue and propose a revised specification."
            )
        else:
            prompt_lines.append("\nPlease propose a refined quantitative strategy.")

        return "\n".join(prompt_lines)

    def propose(self, rejection_feedback: Optional[str] = None) -> StrategySpec:
        self.proposal_count += 1
        prompt = self._build_prompt(rejection_feedback)
        response_text = self.llm_client.generate(prompt, system_prompt=SYSTEM_PROMPT)
        
        # Robustly extract and parse JSON (strip markdown fences if present)
        try:
            text = response_text.strip()
            if "```" in text:
                # Strip markdown code blocks
                lines = text.splitlines()
                clean_lines = []
                inside_block = False
                for line in lines:
                    if line.strip().startswith("```"):
                        inside_block = not inside_block
                        continue
                    clean_lines.append(line)
                text = "\n".join(clean_lines).strip()
            
            # Locate outer JSON braces
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                text = text[start:end + 1]

            data = json.loads(text)
            spec = StrategySpec.from_dict(data)
            spec.spec_id = f"{self.researcher_id}_iter_{self.proposal_count}"
            spec.is_fallback = False
            self.last_parse_failed = False
            return spec
        except Exception as e:
            # Fallback safe spec if LLM fails formatting
            self.last_parse_failed = True
            return StrategySpec(
                spec_id=f"{self.researcher_id}_iter_{self.proposal_count}",
                hypothesis=f"Fallback baseline due to parsing exception: {str(e)}",
                signal_type="momentum",
                params={"window": 50},
                is_fallback=True
            )
