import json
from typing import Optional, List, Dict, Any
from agent.spec import StrategySpec
from agent.llm import BaseLLMClient

SYSTEM_PROMPT = """You are a quantitative CTA research agent.
Your objective is to propose profitable, sound CTA strategies avoiding overfitting.
You must output ONLY valid JSON matching the following schema:
{
  "spec_id": "<string>",
  "hypothesis": "<string explaining economic/statistical rationale>",
  "signal_type": "<'momentum' or 'macd'>",
  "params": {"<param_name>": <value>}
}
Rules:
- momentum params: {"window": integer between 2 and 260 (e.g. 20, 50, 120, 252)}
- macd params: {"fast": int, "slow": int, "signal_span": int} with 1 < fast < slow and signal_span > 1
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
                line = (
                    f"- Spec {spec.spec_id} ({spec.signal_type}, {spec.params}): "
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
