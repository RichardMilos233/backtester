import abc
import json
import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class LLMUsage:
    """Cumulative process metrics for an LLM client (token usage, call counts)."""
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    retries: int = 0
    failures: int = 0
    latency_seconds: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def add(self, other: "LLMUsage") -> "LLMUsage":
        """Accumulates `other` into self in place and returns self."""
        self.calls += other.calls
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens
        self.retries += other.retries
        self.failures += other.failures
        self.latency_seconds += other.latency_seconds
        return self


class BaseLLMClient(abc.ABC):
    @abc.abstractmethod
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generates response text from the LLM given prompt."""
        pass

    @property
    def usage(self) -> LLMUsage:
        # Lazily created so subclasses need not call super().__init__().
        u = self.__dict__.get("_usage")
        if u is None:
            u = LLMUsage()
            self.__dict__["_usage"] = u
        return u

    @usage.setter
    def usage(self, value: LLMUsage) -> None:
        self.__dict__["_usage"] = value

    def record(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        latency: float,
        retries: int = 0,
        total_tokens: Optional[int] = None,
    ) -> None:
        """Records one successful generate() call into self.usage."""
        u = self.usage
        prompt_tokens = int(prompt_tokens or 0)
        completion_tokens = int(completion_tokens or 0)
        if total_tokens is None:
            total_tokens = prompt_tokens + completion_tokens
        u.calls += 1
        u.prompt_tokens += prompt_tokens
        u.completion_tokens += completion_tokens
        u.total_tokens += int(total_tokens or 0)
        u.retries += int(retries)
        u.latency_seconds += float(latency)


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

        # Every 3rd call: emit a spec with vol scaling and/or allowed_regimes
        emit_overlay = (self.call_count % 3 == 0)

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

        if emit_overlay:
            # Deterministically alternate between vol-only, regime-only, and combined
            overlay_mode = (self.call_count // 3) % 3
            if overlay_mode == 0:
                spec_dict["use_vol_scaling"] = True
                spec_dict["vol_target"] = 0.15
                spec_dict["max_leverage"] = 1.5
            elif overlay_mode == 1:
                spec_dict["allowed_regimes"] = ["Q1", "Q4"]
            else:
                spec_dict["use_vol_scaling"] = True
                spec_dict["vol_target"] = 0.12
                spec_dict["max_leverage"] = 1.5
                spec_dict["allowed_regimes"] = ["Q1", "Q2"]

        response = json.dumps(spec_dict)
        # Deterministic token estimate (~4 chars/token); latency fixed at 0 for reproducibility.
        self.record(
            prompt_tokens=len(prompt) // 4,
            completion_tokens=len(response) // 4,
            latency=0.0,
        )
        return response


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
        timeout: float = 60.0,
        max_tokens: int = 32768,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        # The gateway defaults Anthropic models to 4096 completion tokens, which truncates
        # "reason first, then code" answers mid-function. Always send an explicit cap.
        self.max_tokens = max_tokens
        self.last_finish_reason: Optional[str] = None

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
            "temperature": 1.0 if self.model.startswith("gpt-6") else self.temperature,
            "max_tokens": self.max_tokens,
        }

        max_retries = 3
        last_error = None
        start = time.perf_counter()

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
                    choice = result["choices"][0]
                    content = choice["message"].get("content") or ""
                    self.last_finish_reason = choice.get("finish_reason")
                    # Clean up reasoning <think>...</think> tags if model is reasoning-based
                    cleaned_content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                    usage = result.get("usage") or {}
                    prompt_tokens = int(usage.get("prompt_tokens") or 0)
                    completion_tokens = int(usage.get("completion_tokens") or 0)
                    total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
                    self.record(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        latency=time.perf_counter() - start,
                        retries=attempt,
                        total_tokens=total_tokens,
                    )
                    return cleaned_content
            except urllib.error.HTTPError as e:
                error_body = e.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"HTTPError {e.code}: {e.reason}\nBody: {error_body}")
                if e.code == 400 and "temperature" in error_body and payload.get("temperature") != 1.0:
                    payload["temperature"] = 1.0
                    continue
                if (
                    e.code == 400
                    and "max_completion_tokens" in error_body
                    and "max_tokens" in payload
                ):
                    # Some models (e.g. gpt-6.1-sol as of 2026-11) reject `max_tokens` and
                    # require `max_completion_tokens` instead. Self-correct once and retry.
                    payload["max_completion_tokens"] = payload.pop("max_tokens")
                    continue
                if e.code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                    continue
                self._record_failure(attempt, start)
                raise last_error from e
            except Exception as e:
                last_error = RuntimeError(f"Failed to call LLM API: {str(e)}")
                # Retry on socket timeouts or connection drops
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                    continue
                self._record_failure(attempt, start)
                raise last_error from e

        self._record_failure(max_retries - 1, start)
        raise last_error or RuntimeError("Max retries exceeded")

    def _record_failure(self, attempt: int, start: float) -> None:
        """Records a generate() call that ultimately raised (counts as a call)."""
        u = self.usage
        u.calls += 1
        u.failures += 1
        u.retries += attempt
        u.latency_seconds += time.perf_counter() - start


MODEL_ALIASES: Dict[str, str] = {
    "deepseek-4-pro": "deepseek-v4-pro",
    "deepseek-4-flash": "deepseek-v4-flash",
}


def get_llm_client(
    model: str = "deepseek-chat",
    seed: int = 42,
    temperature: float = 0.7,
    timeout: float = 600.0
) -> BaseLLMClient:
    """
    Unified factory for LLM clients.
    Returns deterministic MockLLMClient when model='mock', otherwise connects
    to OpenAI-compatible gateway using credentials from environment or .env file.
    """
    if model == "mock":
        return MockLLMClient(seed=seed)

    resolved_model = MODEL_ALIASES.get(model, model)

    import os
    from pathlib import Path

    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "") or os.environ.get("AI_PROXY_BASE_URL", "")

    # Lazy-load .env if env vars are unset
    if not api_key:
        candidate_paths = [
            Path(".env"),
            Path(__file__).resolve().parents[2] / ".env",
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
        model=resolved_model,
        temperature=temperature,
        timeout=timeout
    )
