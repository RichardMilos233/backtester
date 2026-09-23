import abc
import json
import random
from typing import Optional, Dict, Any


class BaseLLMClient(abc.ABC):
    @abc.abstractmethod
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generates response text from the LLM given prompt."""
        pass


class MockLLMClient(BaseLLMClient):
    """
    Deterministic Mock LLM client for rapid testing, ablation studies,
    and seed reproducibility without API costs or network latency.
    """
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        self.call_count = 0

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        self.call_count += 1
        # Flip between momentum and macd based on call count and seed
        pick_momentum = (self.call_count % 2 == 1)

        if pick_momentum:
            # Pick a safe window between 10 and 200
            window = self.rng.choice([10, 20, 30, 50, 60, 100, 120, 150, 200])
            spec_dict = {
                "spec_id": f"mock_spec_{self.call_count}",
                "hypothesis": f"Trend momentum hypothesis #{self.call_count} with window {window}",
                "signal_type": "momentum",
                "params": {"window": window}
            }
        else:
            fast = self.rng.choice([8, 10, 12, 15])
            slow = self.rng.choice([20, 24, 26, 30])
            span = self.rng.choice([7, 9, 10])
            spec_dict = {
                "spec_id": f"mock_spec_{self.call_count}",
                "hypothesis": f"MACD crossover hypothesis #{self.call_count} (fast={fast}, slow={slow}, span={span})",
                "signal_type": "macd",
                "params": {"fast": fast, "slow": slow, "signal_span": span}
            }
        return json.dumps(spec_dict)


class OpenAICompatibleClient(BaseLLMClient):
    """
    Client for OpenAI-compatible REST APIs (DeepSeek, Qwen, Ollama, OpenAI).
    """
    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o",
        temperature: float = 0.7
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.temperature = temperature

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        # Placeholder for external HTTP request library (e.g. urllib or httpx/requests)
        # Raises clear error if used without API key
        if not self.api_key:
            raise ValueError("OpenAICompatibleClient requires a non-empty api_key.")
        raise NotImplementedError("Live API calls to be plugged in when external key is provided.")
