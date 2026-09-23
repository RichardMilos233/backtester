from typing import Dict, Any, List, Optional
import pandas as pd

from agent.config import ExperimentConfig
from agent.evaluator import StrategyEvaluator
from agent.spec import StrategySpec
from agent.llm import MockLLMClient
from agent.researcher import Researcher
from agent.reviewer import Reviewer

class StrategyOrchestrator:
    def __init__(self, config: ExperimentConfig, evaluator: StrategyEvaluator):
        self.config = config
        self.evaluator = evaluator
        self.evaluated_specs: List[StrategySpec] = []
        self.search_results: List[Dict[str, Any]] = []

    def run(self) -> Dict[str, Any]:
        if self.config.framework_type == "A":
            return self._run_framework_a()
        elif self.config.framework_type == "B":
            return self._run_framework_b()
        elif self.config.framework_type == "C":
            return self._run_framework_c()
        elif self.config.framework_type == "D":
            return self._run_framework_d()
        else:
            raise ValueError(f"Unknown framework_type {self.config.framework_type}")

    def _run_framework_a(self) -> Dict[str, Any]:
        """Framework A: Handwritten Baseline (no agent search)."""
        baselines = [
            StrategySpec("baseline_sma50", "Classic 50-day SMA", "momentum", {"window": 50}),
            StrategySpec("baseline_macd", "Classic MACD", "macd", {"fast": 12, "slow": 26, "signal_span": 9}),
        ]
        for spec in baselines:
            res = self.evaluator.evaluate_search(spec, self.config.feedback_metrics)
            self.evaluated_specs.append(spec)
            self.search_results.append(res)
        return self._finalize_selection_and_test()

    def _run_framework_b(self) -> Dict[str, Any]:
        """Framework B: Single Researcher, No Reviewer."""
        client = MockLLMClient(seed=self.config.seed)
        researcher = Researcher(client, memory_length=self.config.memory_length, researcher_id="res_b")
        
        for iteration in range(self.config.max_iterations):
            spec = researcher.propose()
            is_valid, _ = spec.validate()
            if not is_valid:
                continue
            metrics = self.evaluator.evaluate_search(spec, self.config.feedback_metrics)
            researcher.update_memory(spec, metrics)
            self.evaluated_specs.append(spec)
            self.search_results.append(metrics)
            
            if self._check_early_stop():
                break

        return self._finalize_selection_and_test()

    def _run_framework_c(self) -> Dict[str, Any]:
        """Framework C: Researcher + Reviewer Verification Gate."""
        client = MockLLMClient(seed=self.config.seed)
        researcher = Researcher(client, memory_length=self.config.memory_length, researcher_id="res_c")
        reviewer = Reviewer(allow_duplicates=False)
        
        rejection_reason = None
        for iteration in range(self.config.max_iterations):
            spec = researcher.propose(rejection_feedback=rejection_reason)
            review_result = reviewer.review(spec, self.evaluated_specs)
            
            if not review_result.passed:
                rejection_reason = review_result.reason
                continue
            
            rejection_reason = None
            metrics = self.evaluator.evaluate_search(spec, self.config.feedback_metrics)
            researcher.update_memory(spec, metrics)
            self.evaluated_specs.append(spec)
            self.search_results.append(metrics)

            if self._check_early_stop():
                break

        return self._finalize_selection_and_test()

    def _run_framework_d(self) -> Dict[str, Any]:
        """Framework D: Dual-Researcher Competition."""
        client1 = MockLLMClient(seed=self.config.seed)
        client2 = MockLLMClient(seed=self.config.seed + 1000)
        res1 = Researcher(client1, memory_length=self.config.memory_length, researcher_id="res_d1")
        res2 = Researcher(client2, memory_length=self.config.memory_length, researcher_id="res_d2")
        
        for iteration in range(self.config.max_iterations // 2):
            for res_agent in (res1, res2):
                spec = res_agent.propose()
                is_valid, _ = spec.validate()
                if not is_valid:
                    continue
                metrics = self.evaluator.evaluate_search(spec, self.config.feedback_metrics)
                res_agent.update_memory(spec, metrics)
                self.evaluated_specs.append(spec)
                self.search_results.append(metrics)

        return self._finalize_selection_and_test()

    def _check_early_stop(self) -> bool:
        if len(self.search_results) < self.config.patience + 1:
            return False
        metric = self.config.selection_metric
        recent = [r.get(metric, 0.0) for r in self.search_results[-self.config.patience:]]
        best_prior = max(r.get(metric, 0.0) for r in self.search_results[:-self.config.patience])
        if all((r - best_prior) < self.config.min_improvement for r in recent):
            return True
        return False

    def _finalize_selection_and_test(self) -> Dict[str, Any]:
        """
        Selection Gate: Evaluate all search candidates strictly on the selection set.
        Pick single best candidate.
        Holdout Test: Evaluate ONLY the winning candidate on the test set.
        """
        if not self.evaluated_specs:
            # Fallback baseline if all rejected
            fallback = StrategySpec("fallback", "Default SMA 50", "momentum", {"window": 50})
            self.evaluated_specs.append(fallback)

        selection_scores = []
        for spec in self.evaluated_specs:
            sel_metrics = self.evaluator.evaluate_selection(spec)
            selection_scores.append((sel_metrics.get(self.config.selection_metric, -999.0), spec, sel_metrics))

        # Sort descending
        selection_scores.sort(key=lambda x: x[0], reverse=True)
        best_score, best_spec, best_sel_metrics = selection_scores[0]

        # Holdout test on winner only
        test_metrics = self.evaluator.evaluate_test(best_spec)

        metric_name = self.config.selection_metric
        overfitting_gap = best_sel_metrics.get(metric_name, 0.0) - test_metrics.get(metric_name, 0.0)

        return {
            "framework_type": self.config.framework_type,
            "seed": self.config.seed,
            "total_candidates": len(self.evaluated_specs),
            "best_spec": best_spec.to_dict(),
            "selection_metrics": best_sel_metrics,
            "test_metrics": test_metrics,
            "overfitting_gap": overfitting_gap,
            "search_candidates_count": len(self.evaluated_specs)
        }
