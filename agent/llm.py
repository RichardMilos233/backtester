import abc
import json
import random


class BaseLLMClient(abc.ABC):
    @abc.abstractmethod
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generates response text from the LLM given prompt."""
        pass


REJECTION_ALERT = "[ALERT - PRIOR PROPOSAL REJECTED BY REVIEWER]"


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
        rejected = REJECTION_ALERT in prompt
        # Flip between momentum and macd based on call count and seed
        pick_momentum = (self.call_count % 2 == 1)

        if rejected:
            # Revised proposal: flip signal type and offset parameters
            pick_momentum = not pick_momentum
            revision_note = "Revised proposal addressing reviewer feedback"
        else:
            revision_note = None

        if pick_momentum:
            # Pick a safe window between 10 and 200
            window = self.rng.choice([10, 20, 30, 50, 60, 100, 120, 150, 200])
            if rejected:
                window = min(250, window + 5)
            hypothesis = (
                f"{revision_note}: trend momentum #{self.call_count} with window {window}"
                if revision_note
                else f"Trend momentum hypothesis #{self.call_count} with window {window}"
            )
            spec_dict = {
                "spec_id": f"mock_spec_{self.call_count}",
                "hypothesis": hypothesis,
                "signal_type": "momentum",
                "params": {"window": window}
            }
        else:
            fast = self.rng.choice([8, 10, 12, 15])
            slow = self.rng.choice([20, 24, 26, 30])
            span = self.rng.choice([7, 9, 10])
            if rejected:
                fast = max(2, fast - 1)
                if not (1 < fast < slow):
                    slow = fast + 5
            hypothesis = (
                f"{revision_note}: MACD #{self.call_count} (fast={fast}, slow={slow}, span={span})"
                if revision_note
                else f"MACD crossover hypothesis #{self.call_count} (fast={fast}, slow={slow}, span={span})"
            )
            spec_dict = {
                "spec_id": f"mock_spec_{self.call_count}",
                "hypothesis": hypothesis,
                "signal_type": "macd",
                "params": {"fast": fast, "slow": slow, "signal_span": span}
            }
        return json.dumps(spec_dict)


class OpenAICompatibleClient(BaseLLMClient):
    """
    Client for OpenAI-compatible REST APIs (DeepSeek, Qwen, Amber AI Gateway, OpenAI).
    Uses standard library urllib for zero external dependencies.
    """
    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://api.openai.com/v1",
        model: str = "deepseek-chat",
        temperature: float = 0.7,
        timeout: float = 60.0
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        import re
        import urllib.request
        import urllib.error

        if not self.api_key:
            raise ValueError("OpenAICompatibleClient requires a non-empty api_key.")

        endpoint = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature
        }

        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                content = result["choices"][0]["message"]["content"]
                # Clean up reasoning <think>...</think> tags if model is reasoning-based
                cleaned_content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                return cleaned_content
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTPError {e.code}: {e.reason}\nBody: {error_body}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to call LLM API: {str(e)}") from e
