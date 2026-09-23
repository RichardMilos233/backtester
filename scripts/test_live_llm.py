import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agent.llm import OpenAICompatibleClient
from agent.researcher import Researcher
from agent.spec import StrategySpec


def load_env_file(filepath: str = ".env"):
    if not os.path.exists(filepath):
        return
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'").strip('"')
                os.environ.setdefault(k, v)


def run_live_agent():
    load_env_file(".env")

    api_key = os.getenv("OPENAI_API_KEY", "vk-31b9b9f3-e0d3-40f7-a878-f6f507e17514")
    base_url = os.getenv("OPENAI_BASE_URL", "https://chatai-api.amberainsider.com/v1")
    model = os.getenv("LLM_MODEL", "deepseek-chat")

    print(f"Connecting to AI Gateway: {base_url}")
    print(f"Target LLM Model: {model}")
    print(f"API Key (masked): {api_key[:6]}...{api_key[-4:]}\n")

    client = OpenAICompatibleClient(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=0.7
    )

    researcher = Researcher(llm_client=client, memory_length=3, researcher_id="live_researcher_0")

    # Round 1: Initial cold proposal
    print("=" * 60)
    print(">>> [Round 1] Asking Live LLM to generate initial CTA strategy...")
    print("=" * 60)
    spec1 = researcher.propose()

    print(f"Spec ID:    {spec1.spec_id}")
    print(f"Signal:     {spec1.signal_type}")
    print(f"Params:     {spec1.params}")
    print(f"Hypothesis: {spec1.hypothesis}")

    is_valid, reason = spec1.validate()
    print(f"Validation: {'PASSED' if is_valid else 'FAILED: ' + reason}\n")

    # Simulate Evaluation Feedback: Suppose this strategy had low Sharpe and high drawdown
    simulated_metrics = {
        "sharpe_ratio": -0.15,
        "max_drawdown": -0.22,
        "turnover": 4.5
    }
    print(f"Simulating poor backtest feedback: {simulated_metrics}")
    researcher.update_memory(spec1, simulated_metrics)

    # Round 2: Adaptive proposal following feedback
    print("\n" + "=" * 60)
    print(">>> [Round 2] Asking Live LLM to evolve strategy based on feedback...")
    print("=" * 60)
    spec2 = researcher.propose()

    print(f"Spec ID:    {spec2.spec_id}")
    print(f"Signal:     {spec2.signal_type}")
    print(f"Params:     {spec2.params}")
    print(f"Hypothesis: {spec2.hypothesis}")

    is_valid2, reason2 = spec2.validate()
    print(f"Validation: {'PASSED' if is_valid2 else 'FAILED: ' + reason2}\n")


if __name__ == "__main__":
    run_live_agent()
