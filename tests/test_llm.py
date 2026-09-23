import json
import pytest
from agent.llm import MockLLMClient, BaseLLMClient


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
    prompt_with_rejection = "Previous proposal rejected: Invalid momentum window: 0. Fix it."
    response = client.generate(prompt_with_rejection)
    parsed = json.loads(response)
    # Generated window should be valid
    if parsed["signal_type"] == "momentum":
        assert parsed["params"]["window"] > 1
