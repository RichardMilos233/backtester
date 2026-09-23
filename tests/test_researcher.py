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

def test_researcher_rolling_memory():
    client = MockLLMClient(seed=42)
    researcher = Researcher(llm_client=client, memory_length=2)
    
    for i in range(4):
        s = StrategySpec(f"s_{i}", f"hypo_{i}", "momentum", {"window": 20 + i * 10})
        researcher.update_memory(s, {"sharpe_ratio": 0.5 + i * 0.1})
        
    assert len(researcher.memory) == 2
    assert researcher.memory[-1]["spec"].spec_id == "s_3"
