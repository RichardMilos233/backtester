import json
import time
from pathlib import Path
from typing import Callable, Dict, Any, List, Optional, Set, Tuple
import pandas as pd

from agent.config import ExperimentConfig
from agent.evaluator import StrategyEvaluator
from agent.orchestrator import StrategyOrchestrator
from agent.llm import BaseLLMClient, MockLLMClient


class ExperimentGridRunner:
    """
    Executes matrix of (Model x Infra x Seed) experiment configurations.
    Enforces idempotency, trace persistence to runs.jsonl, and tabular aggregation in summary.csv.
    """
    def __init__(
        self,
        output_dir: Path,
        evaluator: StrategyEvaluator,
        llm_factory: Optional[Callable[[str, int], BaseLLMClient]] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.evaluator = evaluator
        self.llm_factory = llm_factory
        self.summary_csv = self.output_dir / "summary.csv"
        self.runs_jsonl = self.output_dir / "runs.jsonl"

    def _get_completed_keys(self) -> Set[Tuple[str, str, int]]:
        if not self.summary_csv.exists():
            return set()
        try:
            df = pd.read_csv(self.summary_csv)
            if df.empty or "status" not in df.columns:
                return set()
            success_df = df[df["status"] == "success"]
            keys = set()
            for _, row in success_df.iterrows():
                keys.add((str(row["model"]), str(row["framework_type"]), int(row["seed"])))
            return keys
        except Exception:
            return set()

    def _create_llm_client_factory(self, model: str) -> Callable[[int], BaseLLMClient]:
        if self.llm_factory is not None:
            return lambda seed: self.llm_factory(model, seed)

        # Default fallback
        if model == "mock":
            return lambda seed: MockLLMClient(seed=seed)
        else:
            from agent.llm import get_llm_client
            return lambda seed: get_llm_client(model, seed=seed)

    def run_single(self, config: ExperimentConfig) -> Dict[str, Any]:
        start_time = time.time()
        client_factory = self._create_llm_client_factory(config.model)
        orchestrator = StrategyOrchestrator(
            config=config,
            evaluator=self.evaluator,
            llm_client_factory=client_factory
        )

        try:
            orch_res = orchestrator.run()
            elapsed = round(time.time() - start_time, 2)

            best_spec = orch_res.get("best_spec", {})
            val_m = orch_res.get("selection_metrics", {})
            test_m = orch_res.get("test_metrics", {})
            search_m = orch_res.get("search_metrics", {})

            summary_row = {
                "model": config.model,
                "framework_type": config.framework_type,
                "seed": config.seed,
                "max_backtests": config.max_backtests,
                "total_proposals": orch_res.get("total_proposals", 0),
                "parse_failures": orch_res.get("parse_failures", 0),
                "reviewer_rejections": orch_res.get("reviewer_rejections", 0),
                "actual_backtests": orch_res.get("actual_backtests", 0),
                "best_spec_id": best_spec.get("spec_id", ""),
                "best_signal_type": best_spec.get("signal_type", ""),
                "best_params": json.dumps(best_spec.get("params", {})),
                "best_is_fallback": best_spec.get("is_fallback", False),
                "search_sharpe": search_m.get("sharpe_ratio", None),
                "val_sharpe": val_m.get("sharpe_ratio", None),
                "val_total_return": val_m.get("total_return", None),
                "val_max_drawdown": val_m.get("max_drawdown", None),
                "val_turnover": val_m.get("turnover", None),
                "test_sharpe": test_m.get("sharpe_ratio", None),
                "test_total_return": test_m.get("total_return", None),
                "test_annualized_return": test_m.get("annualized_return", None),
                "test_max_drawdown": test_m.get("max_drawdown", None),
                "test_win_rate": test_m.get("daily_win_rate", None),
                "test_trades_count": test_m.get("trades_count", None),
                "test_turnover": test_m.get("turnover", None),
                "overfitting_gap": orch_res.get("overfitting_gap", None),
                "elapsed_seconds": elapsed,
                "status": "success",
                "error_message": "",
            }

            # Write trace to jsonl
            trace_entry = {
                "config": {
                    "model": config.model,
                    "framework_type": config.framework_type,
                    "seed": config.seed,
                    "max_backtests": config.max_backtests,
                },
                "elapsed_seconds": elapsed,
                "orchestrator_result": orch_res,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            with open(self.runs_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(trace_entry, default=str) + "\n")

            self._append_summary_row(summary_row)
            return summary_row

        except Exception as e:
            elapsed = round(time.time() - start_time, 2)
            fail_row = {
                "model": config.model,
                "framework_type": config.framework_type,
                "seed": config.seed,
                "max_backtests": config.max_backtests,
                "total_proposals": getattr(orchestrator, "total_proposals", 0),
                "parse_failures": getattr(orchestrator, "parse_failures", 0),
                "reviewer_rejections": getattr(orchestrator, "reviewer_rejections", 0),
                "actual_backtests": len(getattr(orchestrator, "evaluated_specs", [])),
                "best_spec_id": "",
                "best_signal_type": "",
                "best_params": "",
                "best_is_fallback": False,
                "search_sharpe": None,
                "val_sharpe": None,
                "val_total_return": None,
                "val_max_drawdown": None,
                "val_turnover": None,
                "test_sharpe": None,
                "test_total_return": None,
                "test_annualized_return": None,
                "test_max_drawdown": None,
                "test_win_rate": None,
                "test_trades_count": None,
                "test_turnover": None,
                "overfitting_gap": None,
                "elapsed_seconds": elapsed,
                "status": "failed",
                "error_message": str(e),
            }
            self._append_summary_row(fail_row)
            return fail_row

    def _append_summary_row(self, row: Dict[str, Any]):
        df_row = pd.DataFrame([row])
        if not self.summary_csv.exists():
            df_row.to_csv(self.summary_csv, index=False)
        else:
            df_row.to_csv(self.summary_csv, mode="a", header=False, index=False)

    def run_grid(self, configs: List[ExperimentConfig], resume: bool = True) -> pd.DataFrame:
        completed = self._get_completed_keys() if resume else set()

        for idx, cfg in enumerate(configs, 1):
            key = (cfg.model, cfg.framework_type, cfg.seed)
            if resume and key in completed:
                print(f"[{idx}/{len(configs)}] Skipping already completed run: {key}")
                continue

            print(f"[{idx}/{len(configs)}] Running experiment: model={cfg.model}, fw={cfg.framework_type}, seed={cfg.seed}...")
            self.run_single(cfg)

        if self.summary_csv.exists():
            return pd.read_csv(self.summary_csv)
        return pd.DataFrame()
