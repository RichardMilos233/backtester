import json
import math
import random
from typing import Callable, Dict, Any, List, Optional

from agent.config import ExperimentConfig
from agent.evaluator import StrategyEvaluator
from agent.spec import StrategySpec
from agent.llm import BaseLLMClient, MockLLMClient, LLMUsage
from agent.researcher import Researcher
from agent.reviewer import Reviewer
from agent.space import enumerate_space, spec_key


def _prompt_version() -> str:
    try:
        from agent.researcher import PROMPT_VERSION
        return str(PROMPT_VERSION)
    except ImportError:
        return "v1"


class StrategyOrchestrator:
    def __init__(
        self,
        config: ExperimentConfig,
        evaluator: StrategyEvaluator,
        llm_client_factory: Optional[Callable[[int], BaseLLMClient]] = None,
    ):
        self.config = config
        self.evaluator = evaluator
        self._raw_llm_factory = llm_client_factory or MockLLMClient
        self.llm_clients: List[BaseLLMClient] = []
        self.evaluated_specs: List[StrategySpec] = []
        self.search_results: List[Dict[str, Any]] = []

        # Process tracking metrics
        self.total_proposals: int = 0
        self.parse_failures: int = 0
        self.reviewer_rejections: int = 0

    def llm_client_factory(self, seed: int) -> BaseLLMClient:
        """Creates a client and keeps a handle so token usage can be summed at the end."""
        client = self._raw_llm_factory(seed)
        self.llm_clients.append(client)
        return client

    def llm_usage(self) -> Dict[str, Any]:
        total = LLMUsage()
        for c in self.llm_clients:
            u = getattr(c, "usage", None)
            if isinstance(u, LLMUsage):
                total.add(u)
        return total.as_dict()

    def search_trajectory(self) -> List[Dict[str, Any]]:
        """Ordered record of every backtested spec with its search-split metrics.
        Needed for best-so-far / sample-efficiency curves."""
        metric = self.config.selection_metric
        out, best = [], -float("inf")
        for i, (spec, m) in enumerate(zip(self.evaluated_specs, self.search_results)):
            score = m.get(metric)
            if score is not None and not (isinstance(score, float) and math.isnan(score)):
                best = max(best, score)
            out.append({
                "step": i + 1,
                "spec_id": spec.spec_id,
                "spec_key": spec_key(spec),
                "spec": spec.to_dict(),
                "search_metrics": m,
                "best_so_far": best if best != -float("inf") else None,
            })
        return out

    @property
    def budget(self) -> int:
        return getattr(self.config, "max_backtests", 8)

    @property
    def max_proposals(self) -> int:
        multiplier = getattr(self.config, "max_proposals_multiplier", 3)
        return self.budget * multiplier

    def run(self) -> Dict[str, Any]:
        fw = self.config.framework_type
        if fw == "A":
            return self._run_framework_a()
        elif fw == "B":
            return self._run_framework_b()
        elif fw == "C":
            return self._run_framework_c()
        elif fw == "D":
            return self._run_framework_d()
        elif fw == "C-sota":
            return self._run_framework_c_sota()
        elif fw == "C-corr":
            return self._run_framework_c_corr()
        elif fw == "C-reflect":
            return self._run_framework_c_reflect()
        elif fw == "C-hier":
            return self._run_framework_c_hier()
        elif fw == "R-random":
            return self._run_random_search()
        elif fw == "R-grid":
            return self._run_grid_search()
        else:
            raise ValueError(f"Unknown framework_type {fw}")

    def _run_framework_a(self) -> Dict[str, Any]:
        """Framework A: Handwritten Baseline (no agent search)."""
        baselines = [
            StrategySpec("baseline_sma50", "Classic 50-day SMA", "momentum", {"window": 50}),
            StrategySpec("baseline_macd", "Classic MACD", "macd", {"fast": 12, "slow": 26, "signal_span": 9}),
        ]
        for spec in baselines:
            self.total_proposals += 1
            res = self.evaluator.evaluate_search(spec, self.config.feedback_metrics)
            self.evaluated_specs.append(spec)
            self.search_results.append(res)
        return self._finalize_selection_and_test()

    def _run_framework_b(self) -> Dict[str, Any]:
        """Framework B: Single Researcher, No Reviewer."""
        client = self.llm_client_factory(self.config.seed)
        researcher = Researcher(client, memory_length=self.config.memory_length, researcher_id="res_b")

        while len(self.evaluated_specs) < self.budget and self.total_proposals < self.max_proposals:
            self.total_proposals += 1
            spec = researcher.propose()
            if getattr(spec, "is_fallback", False) or researcher.last_parse_failed:
                self.parse_failures += 1

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
        client = self.llm_client_factory(self.config.seed)
        researcher = Researcher(client, memory_length=self.config.memory_length, researcher_id="res_c")
        reviewer = Reviewer(allow_duplicates=False)

        rejection_reason = None
        while len(self.evaluated_specs) < self.budget and self.total_proposals < self.max_proposals:
            self.total_proposals += 1
            spec = researcher.propose(rejection_feedback=rejection_reason)
            if getattr(spec, "is_fallback", False) or researcher.last_parse_failed:
                self.parse_failures += 1

            review_result = reviewer.review(spec, self.evaluated_specs)
            if not review_result.passed:
                self.reviewer_rejections += 1
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
        client1 = self.llm_client_factory(self.config.seed)
        client2 = self.llm_client_factory(self.config.seed + 1000)
        res1 = Researcher(client1, memory_length=self.config.memory_length, researcher_id="res_d1")
        res2 = Researcher(client2, memory_length=self.config.memory_length, researcher_id="res_d2")

        agents = [res1, res2]
        idx = 0
        while len(self.evaluated_specs) < self.budget and self.total_proposals < self.max_proposals:
            current_res = agents[idx % 2]
            idx += 1
            self.total_proposals += 1

            spec = current_res.propose()
            if getattr(spec, "is_fallback", False) or current_res.last_parse_failed:
                self.parse_failures += 1

            is_valid, _ = spec.validate()
            if not is_valid:
                continue

            metrics = self.evaluator.evaluate_search(spec, self.config.feedback_metrics)
            current_res.update_memory(spec, metrics)
            self.evaluated_specs.append(spec)
            self.search_results.append(metrics)

        return self._finalize_selection_and_test()

    def _run_framework_c_sota(self) -> Dict[str, Any]:
        """Framework C-sota: Researcher + Reviewer, with SOTA Anchor in Memory."""
        client = self.llm_client_factory(self.config.seed)
        researcher = Researcher(client, memory_length=self.config.memory_length, researcher_id="res_sota")
        reviewer = Reviewer(allow_duplicates=False)

        best_score = -float("inf")
        best_anchor_spec: Optional[StrategySpec] = None
        best_anchor_metrics: Optional[Dict[str, Any]] = None

        rejection_reason = None
        while len(self.evaluated_specs) < self.budget and self.total_proposals < self.max_proposals:
            self.total_proposals += 1
            spec = researcher.propose(rejection_feedback=rejection_reason)
            if getattr(spec, "is_fallback", False) or researcher.last_parse_failed:
                self.parse_failures += 1

            review_result = reviewer.review(spec, self.evaluated_specs)
            if not review_result.passed:
                self.reviewer_rejections += 1
                rejection_reason = review_result.reason
                continue

            rejection_reason = None
            metrics = self.evaluator.evaluate_search(spec, self.config.feedback_metrics)
            self.evaluated_specs.append(spec)
            self.search_results.append(metrics)

            score = metrics.get(self.config.selection_metric, -999.0)
            if score > best_score:
                best_score = score
                best_anchor_spec = spec
                best_anchor_metrics = metrics

            # Regular update
            researcher.update_memory(spec, metrics)

            # Enforce SOTA pinning at index 0
            if best_anchor_spec is not None and len(researcher.memory) > 0:
                if researcher.memory[0]["spec"].spec_id != best_anchor_spec.spec_id:
                    researcher.memory.insert(0, {"spec": best_anchor_spec, "metrics": best_anchor_metrics})
                    if len(researcher.memory) > researcher.memory_length:
                        researcher.memory.pop(1)

            if self._check_early_stop():
                break

        return self._finalize_selection_and_test()

    def _run_framework_c_corr(self) -> Dict[str, Any]:
        """Framework C-corr: Researcher + Reviewer with Signal Correlation Gate."""
        client = self.llm_client_factory(self.config.seed)
        researcher = Researcher(client, memory_length=self.config.memory_length, researcher_id="res_corr")
        reviewer = Reviewer(
            allow_duplicates=False,
            check_correlation=True,
            correlation_threshold=self.config.correlation_threshold,
            evaluator=self.evaluator
        )

        rejection_reason = None
        while len(self.evaluated_specs) < self.budget and self.total_proposals < self.max_proposals:
            self.total_proposals += 1
            spec = researcher.propose(rejection_feedback=rejection_reason)
            if getattr(spec, "is_fallback", False) or researcher.last_parse_failed:
                self.parse_failures += 1

            review_result = reviewer.review(spec, self.evaluated_specs)
            if not review_result.passed:
                self.reviewer_rejections += 1
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

    def _run_framework_c_reflect(self) -> Dict[str, Any]:
        """Framework C-reflect: Researcher + Reviewer with Post-Hoc Reflexion Log."""
        client = self.llm_client_factory(self.config.seed)
        researcher = Researcher(client, memory_length=self.config.memory_length, researcher_id="res_reflect")
        reviewer = Reviewer(allow_duplicates=False)

        rejection_reason = None
        while len(self.evaluated_specs) < self.budget and self.total_proposals < self.max_proposals:
            self.total_proposals += 1
            spec = researcher.propose(rejection_feedback=rejection_reason)
            if getattr(spec, "is_fallback", False) or researcher.last_parse_failed:
                self.parse_failures += 1

            review_result = reviewer.review(spec, self.evaluated_specs)
            if not review_result.passed:
                self.reviewer_rejections += 1
                rejection_reason = review_result.reason
                continue

            rejection_reason = None
            metrics = self.evaluator.evaluate_search(spec, self.config.feedback_metrics)

            # Generate reflection from LLM
            prompt = (
                f"Strategy {spec.spec_id} ({spec.signal_type}, {spec.params}) achieved Sharpe: "
                f"{metrics.get('sharpe_ratio')}, Total Return: {metrics.get('total_return')}, "
                f"MaxDD: {metrics.get('max_drawdown')}. "
                f"In 1 sentence, explain why this parameter set performed well or poorly."
            )
            reflection = client.generate(prompt)
            researcher.update_memory(spec, metrics, reflection=reflection.strip())

            self.evaluated_specs.append(spec)
            self.search_results.append(metrics)

            if self._check_early_stop():
                break

        return self._finalize_selection_and_test()

    def _run_framework_c_hier(self) -> Dict[str, Any]:
        """Framework C-hier: Hierarchical Search (Core Indicator alternating with Volatility Scaling)."""
        client = self.llm_client_factory(self.config.seed)
        researcher = Researcher(client, memory_length=self.config.memory_length, researcher_id="res_hier")
        reviewer = Reviewer(allow_duplicates=False)

        rejection_reason = None
        vol_targets = [0.10, 0.12, 0.15, 0.18]
        vol_attempts = 0  # advances on every vol-overlay proposal so a rejection never re-proposes the same overlay

        while len(self.evaluated_specs) < self.budget and self.total_proposals < self.max_proposals:
            self.total_proposals += 1
            step = len(self.evaluated_specs)

            # Odd step: optimize volatility scaling on current best *core* candidate.
            # Fall back to a core proposal once every vol target for that base has been tried.
            core_specs = [(i, s) for i, s in enumerate(self.evaluated_specs) if not s.use_vol_scaling]
            do_vol = step > 0 and step % 2 == 1 and core_specs and vol_attempts < len(vol_targets)
            if do_vol:
                metric = self.config.selection_metric
                best_idx = max(core_specs, key=lambda t: self.search_results[t[0]].get(metric, -999.0))[0]
                base_spec = self.evaluated_specs[best_idx]

                vol_target = vol_targets[vol_attempts % len(vol_targets)]
                vol_attempts += 1
                spec = StrategySpec(
                    spec_id=f"{base_spec.spec_id}_vol_{int(vol_target*100)}",
                    hypothesis=f"Hierarchical vol-scaling on {base_spec.spec_id} with target {vol_target}",
                    signal_type=base_spec.signal_type,
                    params=dict(base_spec.params),
                    use_vol_scaling=True,
                    vol_target=vol_target,
                    max_leverage=1.5
                )
            else:
                # Even step: propose core strategy
                spec = researcher.propose(rejection_feedback=rejection_reason)
                if getattr(spec, "is_fallback", False) or researcher.last_parse_failed:
                    self.parse_failures += 1

            review_result = reviewer.review(spec, self.evaluated_specs)
            if not review_result.passed:
                self.reviewer_rejections += 1
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

    # ------------------------------------------------------------------
    # Non-LLM baselines: same search budget, same Val selection + single Test.
    # ------------------------------------------------------------------
    def _evaluate_baseline_specs(self, specs: List[StrategySpec]) -> Dict[str, Any]:
        for spec in specs:
            self.total_proposals += 1
            metrics = self.evaluator.evaluate_search(spec, self.config.feedback_metrics)
            self.evaluated_specs.append(spec)
            self.search_results.append(metrics)
        return self._finalize_selection_and_test()

    @staticmethod
    def _relabel(spec: StrategySpec, spec_id: str, hypothesis: str) -> StrategySpec:
        return StrategySpec(
            spec_id=spec_id,
            hypothesis=hypothesis,
            signal_type=spec.signal_type,
            params=dict(spec.params),
            use_vol_scaling=spec.use_vol_scaling,
            vol_target=spec.vol_target,
            max_leverage=spec.max_leverage,
        )

    def _run_random_search(self) -> Dict[str, Any]:
        """R-random: uniform sample (without replacement) of `budget` specs from the full lattice
        (incl. vol-scaling overlays). No LLM, no early stop."""
        space = enumerate_space(include_vol=True)
        rng = random.Random(self.config.seed)
        picks = rng.sample(space, min(self.budget, len(space)))
        specs = [
            self._relabel(s, f"rand_{i}", f"Random search sample {i}: {spec_key(s)}")
            for i, s in enumerate(picks)
        ]
        return self._evaluate_baseline_specs(specs)

    def _run_grid_search(self) -> Dict[str, Any]:
        """R-grid: deterministic, seed-independent, evenly spaced subset of the no-vol lattice
        (sorted by spec_key) with exactly `budget` entries. No LLM, no early stop."""
        space = sorted(enumerate_space(include_vol=False), key=spec_key)
        n, k = len(space), min(self.budget, len(space))
        if k <= 0:
            indices: List[int] = []
        elif k == 1:
            indices = [(n - 1) // 2]
        else:
            indices = [round(i * (n - 1) / (k - 1)) for i in range(k)]
        specs = [
            self._relabel(space[j], f"grid_{i}", f"Coarse grid point {i}: {spec_key(space[j])}")
            for i, j in enumerate(indices)
        ]
        return self._evaluate_baseline_specs(specs)

    def _check_early_stop(self) -> bool:
        if not getattr(self.config, "early_stop", True):
            return False
        if len(self.search_results) < self.config.patience + 1:
            return False
        metric = self.config.selection_metric
        if any(metric not in r for r in self.search_results):
            return False
        recent = [r[metric] for r in self.search_results[-self.config.patience:]]
        best_prior = max(r[metric] for r in self.search_results[:-self.config.patience])
        if all((r - best_prior) < self.config.min_improvement for r in recent):
            return True
        return False

    def _finalize_selection_and_test(self) -> Dict[str, Any]:
        """
        Selection Gate: Evaluate all search candidates strictly on the selection set.
        Pick single best candidate (excluding fallbacks if valid candidates exist).
        Holdout Test: Evaluate ONLY the winning candidate on the test set.
        """
        if not self.evaluated_specs:
            fallback = StrategySpec("fallback", "Default SMA 50", "momentum", {"window": 50}, is_fallback=True)
            self.evaluated_specs.append(fallback)
            self.search_results.append(
                self.evaluator.evaluate_search(fallback, self.config.feedback_metrics)
            )

        search_by_spec_id = {
            spec.spec_id: metrics
            for spec, metrics in zip(self.evaluated_specs, self.search_results)
        }

        # Filter out fallbacks from selection if any legitimate candidates exist
        non_fallbacks = [s for s in self.evaluated_specs if not getattr(s, "is_fallback", False)]
        candidates_to_score = non_fallbacks if non_fallbacks else self.evaluated_specs

        selection_scores = []
        for spec in candidates_to_score:
            sel_metrics = self.evaluator.evaluate_selection(spec)
            score = sel_metrics.get(self.config.selection_metric, -999.0)
            # Degenerate specs (e.g. all-flat signal -> zero vol) yield NaN Sharpe; NaN breaks sort
            # ordering and could be picked as winner, so rank it last.
            if score is None or (isinstance(score, float) and math.isnan(score)):
                score = -float("inf")
            selection_scores.append((score, spec, sel_metrics))

        selection_scores.sort(key=lambda x: x[0], reverse=True)
        best_score, best_spec, best_sel_metrics = selection_scores[0]
        search_metrics = search_by_spec_id.get(best_spec.spec_id, {})

        # Holdout test on winner only
        test_metrics = self.evaluator.evaluate_test(best_spec)

        metric_name = self.config.selection_metric
        overfitting_gap = best_sel_metrics.get(metric_name, 0.0) - test_metrics.get(metric_name, 0.0)

        return {
            "framework_type": self.config.framework_type,
            "seed": self.config.seed,
            "model": getattr(self.config, "model", "mock"),
            "total_candidates": len(self.evaluated_specs),
            "total_proposals": self.total_proposals,
            "parse_failures": self.parse_failures,
            "reviewer_rejections": self.reviewer_rejections,
            "actual_backtests": len(self.evaluated_specs),
            "best_spec": best_spec.to_dict(),
            "search_metrics": search_metrics,
            "selection_metrics": best_sel_metrics,
            "test_metrics": test_metrics,
            "overfitting_gap": overfitting_gap,
            "search_candidates_count": len(self.evaluated_specs),
            "prompt_version": _prompt_version(),
            "llm_usage": self.llm_usage(),
            "trajectory": self.search_trajectory(),
        }
