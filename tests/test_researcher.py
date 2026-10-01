import pytest
from agent.llm import MockLLMClient
from agent.researcher import Researcher
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

