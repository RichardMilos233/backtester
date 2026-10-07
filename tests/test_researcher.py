import pytest
from agent.llm import MockLLMClient
from agent.researcher import Researcher, PROMPT_VERSION, SYSTEM_PROMPT
from agent.spec import StrategySpec

def test_researcher_proposes_spec():
    client = MockLLMClient(seed=42)
    researcher = Researcher(llm_client=client, memory_length=3)

    spec = researcher.propose()
    assert isinstance(spec, StrategySpec)
    assert spec.signal_type in ("momentum", "macd")

def test_researcher_first_proposal_attempt_numbering():
    client = MockLLMClient(seed=42)
    captured = []
    original_generate = client.generate

    def capture_generate(prompt, system_prompt=""):
        captured.append(prompt)
        return original_generate(prompt, system_prompt=system_prompt)

    client.generate = capture_generate
    researcher = Researcher(llm_client=client, memory_length=3)
    spec = researcher.propose()

    assert "Proposal Attempt: #1" in captured[0]
    assert spec.spec_id == "researcher_0_iter_1"

def test_researcher_rolling_memory():
    client = MockLLMClient(seed=42)
    researcher = Researcher(llm_client=client, memory_length=2)

    for i in range(4):
        s = StrategySpec(f"s_{i}", f"hypo_{i}", "momentum", {"window": 20 + i * 10})
        researcher.update_memory(s, {"sharpe_ratio": 0.5 + i * 0.1})

    assert len(researcher.memory) == 2
    assert researcher.memory[0]["spec"].spec_id == "s_2"
    assert researcher.memory[-1]["spec"].spec_id == "s_3"

def test_researcher_fallback_flag_on_invalid_json():
    from agent.llm import BaseLLMClient
    class BadLLM(BaseLLMClient):
        def generate(self, prompt: str, system_prompt: str = "") -> str:
            return "This is completely malformed text and definitely not JSON."

    researcher = Researcher(BadLLM(), researcher_id="bad_res")
    spec = researcher.propose()
    assert spec.is_fallback is True
    assert researcher.last_parse_failed is True
    assert "Fallback baseline due to parsing exception" in spec.hypothesis


def test_researcher_usage_property_exposes_client_usage():
    client = MockLLMClient(seed=3)
    researcher = Researcher(llm_client=client)
    assert researcher.usage is client.usage
    researcher.propose()
    assert researcher.usage.calls == 1
    assert researcher.usage.total_tokens > 0


# --- Layer 1: researcher prompt v2 / v3 ---

def test_prompt_version_is_v3():
    assert PROMPT_VERSION == "v3"


def test_system_prompt_contains_new_fields():
    for field in ("use_vol_scaling", "vol_target", "max_leverage", "allowed_regimes"):
        assert field in SYSTEM_PROMPT, f"SYSTEM_PROMPT missing field: {field}"


def test_system_prompt_describes_regimes():
    for q in ("Q1", "Q2", "Q3", "Q4"):
        assert q in SYSTEM_PROMPT, f"SYSTEM_PROMPT missing regime: {q}"


def test_build_prompt_shows_vol_settings():
    client = MockLLMClient(seed=42)
    researcher = Researcher(llm_client=client, memory_length=5)
    spec = StrategySpec("x", "vol test", "momentum", {"window": 50},
                        use_vol_scaling=True, vol_target=0.12, max_leverage=1.5)
    researcher.update_memory(spec, {"sharpe_ratio": 0.8})
    prompt = researcher._build_prompt()
    assert "vol_target=0.12" in prompt
    assert "max_leverage=1.5" in prompt


def test_build_prompt_shows_regime_settings():
    client = MockLLMClient(seed=42)
    researcher = Researcher(llm_client=client, memory_length=5)
    spec = StrategySpec("y", "regime test", "momentum", {"window": 50},
                        allowed_regimes=["Q1", "Q4"])
    researcher.update_memory(spec, {"sharpe_ratio": 0.5})
    prompt = researcher._build_prompt()
    assert "allowed_regimes" in prompt
    assert "Q1" in prompt


def test_mock_llm_emits_vol_and_regime_on_third_call():
    """Every 3rd call from MockLLMClient must include overlay fields."""
    import json
    client = MockLLMClient(seed=42)
    # call 1, 2: plain; call 3: overlay
    client.generate("prompt 1")
    client.generate("prompt 2")
    response = client.generate("prompt 3")
    data = json.loads(response)
    has_vol = data.get("use_vol_scaling") or data.get("allowed_regimes")
    assert has_vol, "3rd call from MockLLMClient should emit vol or regime overlay"

