import json
import os
import sys
import time
import pandas as pd
import numpy as np

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


def run_benchmark():
    load_env_file(".env")
    api_key = os.getenv("OPENAI_API_KEY", "vk-31b9b9f3-e0d3-40f7-a878-f6f507e17514")
    base_url = os.getenv("OPENAI_BASE_URL", "https://chatai-api.amberainsider.com/v1")
    model = os.getenv("LLM_MODEL", "deepseek-chat")

    print("=" * 80)
    print("      AGENT INFRA TOPOLOGY BENCHMARK: SCIENTIFIC EXPERIMENT (PHASE 1)")
    print("=" * 80)
    print(f"Fixed LLM Backend:      {model}")
    print(f"Fixed LLM Base URL:     {base_url}")
    print(f"Fixed Temperature:      0.7")
    print(f"Evaluation Asset:       SPY (2015-01 to 2026-09)")
    print(f"Time Splits:            Search (60%) | Selection (20%) | Holdout Test (20%)")
    print(f"Frictions:              Slippage 5bps (0.0005) + Commission 5bps (0.0005)")
    print(f"Exploration Iterations: 4 iterations per seed")
    print(f"Seeds:                  [42, 43, 44, 45, 46] (5 seeds)")
    print("=" * 80)

    # Factory that returns live LLM client with identical configuration
    def live_llm_factory(seed: int):
        return OpenAICompatibleClient(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=0.7
        )

    df = load_market_data("data/SPY.csv")
    evaluator = StrategyEvaluator(df)

    frameworks = ["A", "B", "C", "D"]
    seeds = [42, 43, 44, 45, 46]

    all_results = []
    start_time = time.time()

    for fw in frameworks:
        run_seeds = [42] if fw == "A" else seeds
        print(f"\n>>> Running Framework {fw} ({len(run_seeds)} run(s))...")

        for seed in run_seeds:
            t0 = time.time()
            print(f"    - Running Seed {seed}...", end="", flush=True)

            cfg = ExperimentConfig(
                framework_type=fw,
                seed=seed,
                max_iterations=4,
                memory_length=3
            )
            orch = StrategyOrchestrator(cfg, evaluator, llm_client_factory=live_llm_factory)
            res = orch.run()
            elapsed = time.time() - t0

            best_spec = res["best_spec"]
            sel_m = res["selection_metrics"]
            test_m = res["test_metrics"]
            gap = res["overfitting_gap"]

            spec_desc = f"{best_spec['signal_type']}{best_spec['params']}"
            print(f" Done ({elapsed:.1f}s) -> Best: {spec_desc} | Sel Sharpe: {sel_m['sharpe_ratio']:.3f} | Test Sharpe: {test_m['sharpe_ratio']:.3f} | Gap: {gap:.3f}")

            all_results.append({
                "framework": fw,
                "seed": seed,
                "spec_id": best_spec.get("spec_id", ""),
                "signal_type": best_spec.get("signal_type", ""),
                "params": str(best_spec.get("params", {})),
                "hypothesis": best_spec.get("hypothesis", ""),
                "search_sharpe": round(res.get("search_metrics", {}).get("sharpe_ratio", 0.0), 4),
                "select_sharpe": round(sel_m.get("sharpe_ratio", 0.0), 4),
                "test_sharpe": round(test_m.get("sharpe_ratio", 0.0), 4),
                "overfitting_gap": round(gap, 4),
                "test_total_return": round(test_m.get("total_return", 0.0) * 100, 2),
                "test_mdd": round(test_m.get("max_drawdown", 0.0) * 100, 2),
                "test_calmar": round(test_m.get("calmar_ratio", 0.0), 4),
                "test_turnover": round(test_m.get("turnover", 0.0), 2),
                "test_trades": int(test_m.get("trade_count", 0)),
                "total_candidates": res.get("total_candidates", 0),
                "elapsed_sec": round(elapsed, 2)
            })

    total_time = time.time() - start_time
    print(f"\nAll benchmark runs completed in {total_time:.1f} seconds.")

    # Save raw records
    out_dir = "docs/experiments"
    os.makedirs(out_dir, exist_ok=True)
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(os.path.join(out_dir, "live_topology_benchmark.csv"), index=False)
    with open(os.path.join(out_dir, "live_topology_benchmark.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 95)
    print("                      DETAILED BENCHMARK RESULTS TABLE")
    print("=" * 95)
    cols_display = [
        "framework", "seed", "params", "select_sharpe", "test_sharpe", 
        "overfitting_gap", "test_total_return", "test_mdd", "test_trades"
    ]
    print(df_results[cols_display].to_string(index=False))
    print("=" * 95)

    # Aggregate Statistics per Framework
    stats = df_results.groupby("framework").agg(
        select_sharpe_mean=("select_sharpe", "mean"),
        select_sharpe_std=("select_sharpe", "std"),
        test_sharpe_mean=("test_sharpe", "mean"),
        test_sharpe_std=("test_sharpe", "std"),
        gap_mean=("overfitting_gap", "mean"),
        gap_std=("overfitting_gap", "std"),
        test_return_mean=("test_total_return", "mean"),
        test_mdd_mean=("test_mdd", "mean"),
        test_trades_mean=("test_trades", "mean"),
        test_turnover_mean=("test_turnover", "mean")
    ).round(4)

    print("\n" + "=" * 95)
    print("              AGGREGATED FRAMEWORK PERFORMANCE (MEAN ± STD)")
    print("=" * 95)
    print(stats.to_string())
    print("=" * 95)


if __name__ == "__main__":
    run_benchmark()
