import json
import time
from pathlib import Path
from typing import Callable, Dict, Any, List, Optional, Set, Tuple
import pandas as pd

from agent.config import ExperimentConfig
from agent.evaluator import StrategyEvaluator
from agent.orchestrator import StrategyOrchestrator
from agent.llm import BaseLLMClient, MockLLMClient
from agent.oracle import OracleScorer, load_oracle
from agent.space import spec_key
from agent.spec import StrategySpec
from agent.tasks import get_task


class ExperimentGridRunner:
    """
    Executes matrix of (Model x Infra x Seed) experiment configurations.
    Enforces idempotency, trace persistence to runs.jsonl, and tabular aggregation in summary.csv.
    """
    def __init__(
        self,
        output_dir: Path,
        evaluator: Optional[StrategyEvaluator] = None,
        llm_factory: Optional[Callable[[str, int], BaseLLMClient]] = None,
        oracle_dir: Optional[Path] = None,
    ):
        """
        evaluator: if given, used for every config regardless of task_id (legacy / tests).
                   Otherwise one evaluator is built per task_id from agent.tasks.
        oracle_dir: directory holding <task_id>.csv oracle tables; when a table exists the
                    summary row gets percentile/regret columns.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._fixed_evaluator = evaluator
        self._evaluators: Dict[str, StrategyEvaluator] = {}
        self.oracle_dir = Path(oracle_dir) if oracle_dir else None
        self._scorers: Dict[str, Optional[OracleScorer]] = {}
        self.llm_factory = llm_factory
        self.summary_csv = self.output_dir / "summary.csv"
        self.runs_jsonl = self.output_dir / "runs.jsonl"

    def evaluator_for(self, task_id: str) -> StrategyEvaluator:
        if self._fixed_evaluator is not None:
            return self._fixed_evaluator
        if task_id not in self._evaluators:
            self._evaluators[task_id] = StrategyEvaluator(get_task(task_id).load())
        return self._evaluators[task_id]

    def scorer_for(self, task_id: str) -> Optional[OracleScorer]:
        if self.oracle_dir is None:
            return None
        if task_id not in self._scorers:
            path = self.oracle_dir / f"{task_id}.csv.gz"
            self._scorers[task_id] = OracleScorer(load_oracle(path)) if path.exists() else None
        return self._scorers[task_id]

    def _get_completed_keys(self) -> Set[Tuple[str, str, str, int]]:
        if not self.summary_csv.exists():
            return set()
        try:
            df = pd.read_csv(self.summary_csv)
            if df.empty or "status" not in df.columns:
                return set()
            success_df = df[df["status"] == "success"]
            keys = set()
            for _, row in success_df.iterrows():
                task = str(row["task_id"]) if "task_id" in row and pd.notna(row["task_id"]) else "SPY_full"
                keys.add((task, str(row["model"]), str(row["framework_type"]), int(row["seed"])))
            return keys
        except Exception:
            return set()

    def _create_llm_client_factory(self, model: str) -> Callable[[int], BaseLLMClient]:
        if model == "none":
            # Non-LLM baselines (R-random / R-grid) never request a client.
            def _no_llm(seed: int) -> BaseLLMClient:
                raise RuntimeError("model='none' has no LLM client; only non-LLM baselines may use it")
            return _no_llm

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
        task_id = getattr(config, "task_id", "SPY_full")
        evaluator = self.evaluator_for(task_id)
        orchestrator = StrategyOrchestrator(
            config=config,
            evaluator=evaluator,
            llm_client_factory=client_factory
        )

        try:
            orch_res = orchestrator.run()
            elapsed = round(time.time() - start_time, 2)

            best_spec = orch_res.get("best_spec", {})
            val_m = orch_res.get("selection_metrics", {})
            test_m = orch_res.get("test_metrics", {})
            search_m = orch_res.get("search_metrics", {})

            usage = orch_res.get("llm_usage", {}) or {}
            summary_row = {
                "task_id": task_id,
                "model": config.model,
                "framework_type": config.framework_type,
                "seed": config.seed,
                "max_backtests": config.max_backtests,
                "early_stop": config.early_stop,
                "prompt_version": orch_res.get("prompt_version", ""),
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
                "budget_utilization": (orch_res.get("actual_backtests", 0) or 0) / config.max_backtests if config.max_backtests else None,
                "llm_calls": usage.get("calls", 0),
                "llm_prompt_tokens": usage.get("prompt_tokens", 0),
                "llm_completion_tokens": usage.get("completion_tokens", 0),
                "llm_total_tokens": usage.get("total_tokens", 0),
                "llm_retries": usage.get("retries", 0),
                "llm_failures": usage.get("failures", 0),
                "llm_latency_seconds": round(usage.get("latency_seconds", 0.0) or 0.0, 2),
                "elapsed_seconds": elapsed,
                "status": "success",
                "error_message": "",
            }
            summary_row.update(self._oracle_columns(task_id, best_spec, evaluator))

            # Write trace to jsonl
            trace_entry = {
                "config": {
                    "task_id": task_id,
                    "model": config.model,
                    "framework_type": config.framework_type,
                    "seed": config.seed,
                    "max_backtests": config.max_backtests,
                    "early_stop": config.early_stop,
                    "prompt_version": orch_res.get("prompt_version", ""),
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
                "task_id": task_id,
                "model": config.model,
                "framework_type": config.framework_type,
                "seed": config.seed,
                "max_backtests": config.max_backtests,
                "early_stop": config.early_stop,
                "prompt_version": "",
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
                "budget_utilization": None,
                "llm_calls": None, "llm_prompt_tokens": None, "llm_completion_tokens": None,
                "llm_total_tokens": None, "llm_retries": None, "llm_failures": None, "llm_latency_seconds": None,
                "elapsed_seconds": elapsed,
                "status": "failed",
                "error_message": str(e),
            }
            self._append_summary_row(fail_row)
            return fail_row

    _ORACLE_COLS = ("spec_key", "on_lattice", "percentile_val", "percentile_test",
                    "regret_test", "regret_vs_val_oracle", "excess_over_median_test")

    def _oracle_columns(self, task_id: str, best_spec: Dict[str, Any], evaluator: StrategyEvaluator) -> Dict[str, Any]:
        cols: Dict[str, Any] = {k: None for k in self._ORACLE_COLS}
        if not best_spec:
            return cols
        try:
            spec = StrategySpec.from_dict(best_spec)
            cols["spec_key"] = spec_key(spec)
            scorer = self.scorer_for(task_id)
            if scorer is not None:
                cols.update(scorer.score_spec(spec, evaluator))
        except Exception as e:  # scoring must never fail a run
            cols["oracle_error"] = str(e)
        return cols

    def _append_summary_row(self, row: Dict[str, Any]):
        df_row = pd.DataFrame([row])
        if not self.summary_csv.exists():
            df_row.to_csv(self.summary_csv, index=False)
            return
        existing_cols = list(pd.read_csv(self.summary_csv, nrows=0).columns)
        if set(existing_cols) == set(df_row.columns):
            df_row[existing_cols].to_csv(self.summary_csv, mode="a", header=False, index=False)
        else:  # schema changed (e.g. new columns): rewrite with the union of columns
            full = pd.concat([pd.read_csv(self.summary_csv), df_row], ignore_index=True)
            full.to_csv(self.summary_csv, index=False)

    def run_grid(self, configs: List[ExperimentConfig], resume: bool = True) -> pd.DataFrame:
        completed = self._get_completed_keys() if resume else set()

        for idx, cfg in enumerate(configs, 1):
            task_id = getattr(cfg, "task_id", "SPY_full")
            key = (task_id, cfg.model, cfg.framework_type, cfg.seed)
            if resume and key in completed:
                print(f"[{idx}/{len(configs)}] Skipping already completed run: {key}")
                continue

            print(f"[{idx}/{len(configs)}] Running: task={task_id} model={cfg.model} fw={cfg.framework_type} seed={cfg.seed}...")
            self.run_single(cfg)

        if self.summary_csv.exists():
            return pd.read_csv(self.summary_csv)
        return pd.DataFrame()
