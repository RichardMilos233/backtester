# scripts/run_agent_experiment.py
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import load_market_data
from agent.config import ExperimentConfig
from agent.evaluator import StrategyEvaluator
from agent.orchestrator import StrategyOrchestrator

def run_multi_seed_experiment(ticker: str = "SPY", seeds=(42, 43, 44, 45, 46)):
    print(f"Loading data for {ticker}...")
    df = load_market_data(ticker)
    evaluator = StrategyEvaluator(df)

    frameworks = ["A", "B", "C", "D"]
    all_results = []

    print("\nStarting Cross-Framework Ablation Experiment across 5 seeds...")
    for fw in frameworks:
        print(f"\n>>> Running Framework {fw}...")
        for seed in (seeds if fw != "A" else [42]):
            cfg = ExperimentConfig(framework_type=fw, seed=seed, max_iterations=8)
            orch = StrategyOrchestrator(cfg, evaluator)
            res = orch.run()
            all_results.append({
                "Framework": fw,
                "Seed": seed,
                "Best Spec": res["best_spec"]["signal_type"] + str(res["best_spec"]["params"]),
                "Select Sharpe": round(res["selection_metrics"]["sharpe_ratio"], 4),
                "Test Sharpe": round(res["test_metrics"]["sharpe_ratio"], 4),
                "Overfitting Gap": round(res["overfitting_gap"], 4),
                "Test MDD (%)": round(res["test_metrics"]["max_drawdown"] * 100, 2),
            })

    summary_df = pd.DataFrame(all_results)
    print("\n" + "=" * 80)
    print("                    AGENT INFRA EXPERIMENT SUMMARY")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80)

    # Average metrics per framework
    grouped = summary_df.groupby("Framework")[["Select Sharpe", "Test Sharpe", "Overfitting Gap"]].mean()
    print("\nMean Performance by Framework:")
    print(grouped.round(4).to_string())

if __name__ == "__main__":
    run_multi_seed_experiment()
