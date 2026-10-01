import json
from agent.llm import MockLLMClient, REJECTION_ALERT


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
    from agent.llm import get_llm_client, MockLLMClient
    client = get_llm_client("mock", seed=123)
    assert isinstance(client, MockLLMClient)
    assert client.seed == 123


def test_get_llm_client_real_env(monkeypatch):
    from agent.llm import get_llm_client, OpenAICompatibleClient
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-12345")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.example.com/v1")
    client = get_llm_client("claude-sonnet-4-6")
    assert isinstance(client, OpenAICompatibleClient)
    assert client.model == "claude-sonnet-4-6"
    assert client.api_key == "test-key-12345"
    assert client.base_url == "https://api.example.com/v1"

