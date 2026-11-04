import json
from benchmark.common.llm import MockLLMClient, REJECTION_ALERT


def test_mock_llm_client_returns_valid_json():
    client = MockLLMClient(seed=42)
    response = client.generate("Generate strategy with MA window")
    parsed = json.loads(response)

    assert "signal_type" in parsed
    assert "params" in parsed
    assert parsed["signal_type"] in ("momentum", "macd")


def test_mock_llm_deterministic_across_runs():
    client1 = MockLLMClient(seed=99)
    client2 = MockLLMClient(seed=99)

    res1 = client1.generate("Prompt 1")
    res2 = client2.generate("Prompt 1")
    assert res1 == res2


def test_mock_llm_reacts_to_rejection():
    client = MockLLMClient(seed=42)
    prompt_with_rejection = (
        f"{REJECTION_ALERT}\n"
        "Reason: Invalid momentum window: 0. Fix it."
    )
    response = client.generate(prompt_with_rejection)
    parsed = json.loads(response)
    assert "Revised proposal" in parsed["hypothesis"]
    if parsed["signal_type"] == "momentum":
        assert parsed["params"]["window"] > 1


def test_get_llm_client_mock():
    from benchmark.common.llm import get_llm_client, MockLLMClient
    client = get_llm_client("mock", seed=123)
    assert isinstance(client, MockLLMClient)
    assert client.seed == 123


def test_get_llm_client_real_env(monkeypatch):
    from benchmark.common.llm import get_llm_client, OpenAICompatibleClient
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-12345")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.example.com/v1")
    client = get_llm_client("claude-sonnet-4-6")
    assert isinstance(client, OpenAICompatibleClient)
    assert client.model == "claude-sonnet-4-6"
    assert client.api_key == "test-key-12345"
    assert client.base_url == "https://api.example.com/v1"



# ---------------------------------------------------------------------------
# Usage / process-metric accounting
# ---------------------------------------------------------------------------
import io
import urllib.error
import urllib.request

import pytest

from benchmark.common.llm import BaseLLMClient, LLMUsage, OpenAICompatibleClient


def test_llm_usage_add_and_as_dict():
    a = LLMUsage(calls=1, prompt_tokens=10, completion_tokens=5, total_tokens=15, latency_seconds=0.5)
    b = LLMUsage(calls=2, prompt_tokens=1, completion_tokens=2, total_tokens=3, retries=1, failures=1, latency_seconds=0.25)
    out = a.add(b)
    assert out is a
    assert a.as_dict() == {
        "calls": 3, "prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18,
        "retries": 1, "failures": 1, "latency_seconds": 0.75,
    }


def test_usage_available_on_subclass_without_super_init():
    class Bare(BaseLLMClient):
        def __init__(self):
            pass

        def generate(self, prompt, system_prompt=""):
            return "x"

    c = Bare()
    assert c.usage == LLMUsage()
    c.record(3, 4, 0.1)
    assert c.usage.calls == 1 and c.usage.total_tokens == 7


def test_mock_usage_increments_and_is_deterministic():
    c1 = MockLLMClient(seed=7)
    c2 = MockLLMClient(seed=7)
    assert c1.usage.calls == 0
    for i in range(3):
        c1.generate(f"Prompt number {i} " * 5)
        c2.generate(f"Prompt number {i} " * 5)
        assert c1.usage.calls == i + 1
    assert c1.usage.prompt_tokens > 0
    assert c1.usage.completion_tokens > 0
    assert c1.usage.total_tokens == c1.usage.prompt_tokens + c1.usage.completion_tokens
    assert c1.usage.as_dict() == c2.usage.as_dict()


def test_get_llm_client_returns_object_with_usage():
    from benchmark.common.llm import get_llm_client
    client = get_llm_client("mock", seed=1)
    client.generate("hello world prompt")
    assert client.usage.calls == 1


class _FakeResp:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _chat_payload(content, usage=None):
    payload = {"choices": [{"message": {"content": content}}]}
    if usage is not None:
        payload["usage"] = usage
    return payload


@pytest.fixture
def no_sleep(monkeypatch):
    import benchmark.common.llm as llm_mod
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: None)


def test_openai_client_records_usage_block(monkeypatch, no_sleep):
    payload = _chat_payload(
        "<think>reasoning</think>  answer",
        usage={"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150},
    )
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: _FakeResp(payload))
    client = OpenAICompatibleClient(api_key="k", base_url="http://gw.test/v1", model="m")
    assert client.generate("hi", system_prompt="sys") == "answer"
    u = client.usage
    assert (u.calls, u.prompt_tokens, u.completion_tokens, u.total_tokens) == (1, 120, 30, 150)
    assert u.retries == 0 and u.failures == 0
    assert u.latency_seconds >= 0.0


def test_openai_client_without_usage_block_records_call_only(monkeypatch, no_sleep):
    payload = _chat_payload("plain answer")
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: _FakeResp(payload))
    client = OpenAICompatibleClient(api_key="k", base_url="http://gw.test/v1", model="m")
    assert client.generate("hi") == "plain answer"
    u = client.usage
    assert u.calls == 1
    assert (u.prompt_tokens, u.completion_tokens, u.total_tokens) == (0, 0, 0)


def test_openai_client_counts_retry_after_502(monkeypatch, no_sleep):
    payload = _chat_payload("ok", usage={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7})
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError(req.full_url, 502, "Bad Gateway", {}, io.BytesIO(b"upstream"))
        return _FakeResp(payload)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = OpenAICompatibleClient(api_key="k", base_url="http://gw.test/v1", model="m")
    assert client.generate("hi") == "ok"
    u = client.usage
    assert calls["n"] == 2
    assert (u.calls, u.retries, u.failures) == (1, 1, 0)
    assert u.total_tokens == 7


def test_openai_client_counts_final_failure(monkeypatch, no_sleep):
    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 503, "Unavailable", {}, io.BytesIO(b"down"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = OpenAICompatibleClient(api_key="k", base_url="http://gw.test/v1", model="m")
    with pytest.raises(RuntimeError, match="HTTPError 503"):
        client.generate("hi")
    u = client.usage
    assert (u.calls, u.retries, u.failures) == (1, 2, 1)
    assert u.total_tokens == 0
