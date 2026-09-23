import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import load_market_data
from agent.config import ExperimentConfig
from agent.evaluator import StrategyEvaluator
from agent.orchestrator import StrategyOrchestrator
from agent.llm import OpenAICompatibleClient


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


def run_live_framework(framework_type: str = "C", iterations: int = 4):
    load_env_file(".env")
    api_key = os.getenv("OPENAI_API_KEY", "vk-31b9b9f3-e0d3-40f7-a878-f6f507e17514")
    base_url = os.getenv("OPENAI_BASE_URL", "https://chatai-api.amberainsider.com/v1")
    model = os.getenv("LLM_MODEL", "deepseek-chat")

    print("=" * 70)
    print(f"  RUNNING LIVE AGENT (Framework {framework_type}) WITH MODEL: {model}")
    print("=" * 70)

    # Factory that creates live LLM client
    def live_llm_factory(seed: int):
        return OpenAICompatibleClient(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=0.7
        )

    df = load_market_data("data/SPY.csv")
    evaluator = StrategyEvaluator(df)

    cfg = ExperimentConfig(
        framework_type=framework_type,
        max_iterations=iterations,
        memory_length=3,
        seed=42
    )

    orch = StrategyOrchestrator(cfg, evaluator, llm_client_factory=live_llm_factory)
    print(f"Starting {iterations} iterations of research on SPY data (Search 60% / Select 20% / Test 20%)...\n")
    res = orch.run()

    print("\n" + "=" * 70)
    print("                      LIVE EXPERIMENT RESULTS")
    print("=" * 70)
    print(f"Total Candidates Explored: {res['total_candidates']}")
    print(f"Winning Strategy ID:       {res['best_spec']['spec_id']}")
    print(f"Signal Type:               {res['best_spec']['signal_type']}")
    print(f"Parameters:                {res['best_spec']['params']}")
    print(f"Hypothesis:                {res['best_spec']['hypothesis']}")
    print("-" * 70)
    print(f"Search Set Sharpe:         {res['search_metrics'].get('sharpe_ratio', 'N/A'):.4f}")
    print(f"Selection Set Sharpe:      {res['selection_metrics']['sharpe_ratio']:.4f}")
    print(f"Test Set Sharpe (Holdout): {res['test_metrics']['sharpe_ratio']:.4f}")
    print(f"Overfitting Gap:           {res['overfitting_gap']:.4f}")
    print(f"Test Max Drawdown:         {res['test_metrics']['max_drawdown'] * 100:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    fw = sys.argv[1] if len(sys.argv) > 1 else "C"
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    run_live_framework(framework_type=fw, iterations=iters)
