import abc
import json
import random
import time


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

        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                req = urllib.request.Request(
                    endpoint,
                    data=json.dumps(payload).encode("utf-8"),
                    headers=headers,
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
                    content = result["choices"][0]["message"]["content"]
                    # Clean up reasoning <think>...</think> tags if model is reasoning-based
                    cleaned_content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                    return cleaned_content
            except urllib.error.HTTPError as e:
                error_body = e.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"HTTPError {e.code}: {e.reason}\nBody: {error_body}")
                if e.code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                    continue
                raise last_error from e
            except Exception as e:
                last_error = RuntimeError(f"Failed to call LLM API: {str(e)}")
                # Retry on socket timeouts or connection drops
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                    continue
                raise last_error from e

        raise last_error or RuntimeError("Max retries exceeded")


def get_llm_client(
    model: str = "deepseek-chat",
    seed: int = 42,
    temperature: float = 0.7,
    timeout: float = 90.0
) -> BaseLLMClient:
    """
    Unified factory for LLM clients.
    Returns deterministic MockLLMClient when model='mock', otherwise connects
    to OpenAI-compatible gateway using credentials from environment or .env file.
    """
    if model == "mock":
        return MockLLMClient(seed=seed)

    import os
    from pathlib import Path

    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "") or os.environ.get("AI_PROXY_BASE_URL", "")

    # Lazy-load .env if env vars are unset
    if not api_key:
        candidate_paths = [
            Path(".env"),
            Path(__file__).resolve().parents[1] / ".env",
        ]
        for p in candidate_paths:
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip("'\"")
                            if k not in os.environ:
                                os.environ[k] = v
                break
        api_key = os.environ.get("OPENAI_API_KEY", "")
        base_url = os.environ.get("OPENAI_BASE_URL", "") or os.environ.get("AI_PROXY_BASE_URL", "")

    if not base_url:
        base_url = "https://api.openai.com/v1"
    elif "amberainsider" in base_url and not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"

    return OpenAICompatibleClient(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temperature,
        timeout=timeout
    )
