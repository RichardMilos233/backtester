from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from agent.evaluator import StrategyEvaluator
from agent.oracle import OracleScorer, load_oracle
from agent.space import spec_key
from agent.spec import StrategySpec
from agent.tasks import get_task


SIGNAL_FAMILY_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "momentum": ("momentum", "sma", "moving average", "trend", "window", "mom_w"),
    "macd": ("macd", "ema", "fast", "slow", "signal_span", "crossover", "macd_f"),
    "mean_reversion": ("reversion", "mean-reversion", "mean reversion", "z-score", "z score", "entry_z", "mr_w"),
    "breakout": ("breakout", "donchian", "channel", "entry_window", "high", "bo_en"),
    "rsi": ("rsi", "wilder", "relative strength", "oversold", "overbought", "rsi_w"),
    "combo": ("combo", "combine", "ensemble", "confluenc", "dual", "and", "mean", "all", "any"),
}


@dataclass
class TrajectoryEvalResult:
    n_steps: int
    step_percentile_val: Tuple[float, ...]
    best_percentile_val_so_far: Tuple[float, ...]
    auc_percentile_val: float
    step1_percentile_val: float
    final_best_percentile_val: float
    trajectory_val_gain: float
    step_improvement_rate: float
    median_improvement_rate: float
    val_generalization_corr: float
    hypothesis_alignment: float
    n_off_lattice: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["step_percentile_val"] = list(self.step_percentile_val)
        d["best_percentile_val_so_far"] = list(self.best_percentile_val_so_far)
        return d


def compute_hypothesis_alignment(spec: StrategySpec) -> float:
    """
    Check whether a strategy's natural-language `hypothesis` aligns with its
    structured `signal_type`, `use_vol_scaling`, and `allowed_regimes`.
    """
    text = (spec.hypothesis or "").lower()
    if not text.strip():
        return 0.0

    checks: List[float] = []

    # 1. Signal family alignment
    fam_kws = list(SIGNAL_FAMILY_KEYWORDS.get(spec.signal_type, (spec.signal_type,)))
    if spec.signal_type == "combo":
        for comp in spec.params.get("components", []):
            st = comp.get("signal_type", "")
            fam_kws.extend(SIGNAL_FAMILY_KEYWORDS.get(st, (st,)))
    checks.append(1.0 if any(kw in text for kw in fam_kws) else 0.0)

    # 2. Volatility scaling alignment (when enabled)
    if spec.use_vol_scaling:
        vol_kws = ("vol", "volatility", "leverage", "risk", "scale", "scaling", "target")
        checks.append(1.0 if any(kw in text for kw in vol_kws) else 0.0)

    # 3. Regime filter alignment (when enabled)
    if spec.allowed_regimes:
        reg_kws = ["regime", "quadrant"] + [r.lower() for r in spec.allowed_regimes]
        checks.append(1.0 if any(kw in text for kw in reg_kws) else 0.0)

    return float(np.mean(checks))


def _spearman_corr(x: List[float], y: List[float]) -> float:
    if len(x) < 2 or len(x) != len(y):
        return 0.0
    arr_x = np.asarray(x, dtype=float)
    arr_y = np.asarray(y, dtype=float)
    if np.std(arr_x) < 1e-12 or np.std(arr_y) < 1e-12:
        return 0.0
    corr = float(pd.Series(arr_x).corr(pd.Series(arr_y), method="spearman"))
    return 0.0 if math.isnan(corr) else corr


def score_trajectory(
    trajectory: List[Dict[str, Any]],
    scorer: OracleScorer,
    evaluator: Optional[StrategyEvaluator] = None,
    spec_val_cache: Optional[Dict[str, float]] = None,
) -> TrajectoryEvalResult:
    """
    Score a multi-step search trajectory on:
      - Sample efficiency (`step_percentile_val`, `best_percentile_val_so_far`, `auc_percentile_val`)
      - Hypothesis quality (`step_improvement_rate`, `median_improvement_rate`,
        `val_generalization_corr`, `hypothesis_alignment`)
    """
    if not trajectory:
        return TrajectoryEvalResult(
            n_steps=0,
            step_percentile_val=(),
            best_percentile_val_so_far=(),
            auc_percentile_val=0.0,
            step1_percentile_val=0.0,
            final_best_percentile_val=0.0,
            trajectory_val_gain=0.0,
            step_improvement_rate=0.0,
            median_improvement_rate=0.0,
            val_generalization_corr=0.0,
            hypothesis_alignment=0.0,
            n_off_lattice=0,
        )

    cache = spec_val_cache if spec_val_cache is not None else {}
    step_pvs: List[float] = []
    best_pvs: List[float] = []
    search_sharpes: List[float] = []
    alignments: List[float] = []
    n_off_lattice = 0
    running_best_pv = 0.0

    for item in trajectory:
        spec_dict = item.get("spec") or {}
        spec = StrategySpec.from_dict(spec_dict)
        key = item.get("spec_key") or spec_key(spec)

        if key in scorer.oracle.index:
            val_sharpe = float(scorer.oracle.loc[key, f"val_{scorer.metric}"])
        elif key in cache:
            val_sharpe = cache[key]
            n_off_lattice += 1
        elif evaluator is not None:
            n_off_lattice += 1
            try:
                val_sharpe = float(evaluator.evaluate_selection(spec).get(scorer.metric, float("nan")))
            except Exception:
                val_sharpe = float("nan")
            cache[key] = val_sharpe
        else:
            n_off_lattice += 1
            val_sharpe = float("nan")

        if val_sharpe is None or math.isnan(val_sharpe):
            pv = 0.0
        else:
            pv = float(scorer.percentile(val_sharpe, "val"))
            if math.isnan(pv):
                pv = 0.0

        running_best_pv = max(running_best_pv, pv)
        step_pvs.append(pv)
        best_pvs.append(running_best_pv)

        s_metrics = item.get("search_metrics") or {}
        raw_ss = s_metrics.get(scorer.metric, 0.0)
        ss = 0.0 if (raw_ss is None or (isinstance(raw_ss, float) and math.isnan(raw_ss))) else float(raw_ss)
        search_sharpes.append(ss)

        alignments.append(compute_hypothesis_alignment(spec))

    # Step-to-step improvement rates for steps t >= 2
    if len(search_sharpes) >= 2:
        improved_prev = []
        improved_median = []
        for idx in range(1, len(search_sharpes)):
            curr = search_sharpes[idx]
            prev = search_sharpes[idx - 1]
            prior_med = float(np.median(search_sharpes[:idx]))
            improved_prev.append(1.0 if curr > prev else 0.0)
            improved_median.append(1.0 if curr > prior_med else 0.0)
        step_imp_rate = float(np.mean(improved_prev))
        med_imp_rate = float(np.mean(improved_median))
    else:
        step_imp_rate = 0.0
        med_imp_rate = 0.0

    val_gen_corr = _spearman_corr(search_sharpes, step_pvs)
    auc_pv = float(np.mean(best_pvs))
    step1_pv = step_pvs[0]
    final_pv = best_pvs[-1]

    return TrajectoryEvalResult(
        n_steps=len(trajectory),
        step_percentile_val=tuple(step_pvs),
        best_percentile_val_so_far=tuple(best_pvs),
        auc_percentile_val=auc_pv,
        step1_percentile_val=step1_pv,
        final_best_percentile_val=final_pv,
        trajectory_val_gain=final_pv - step1_pv,
        step_improvement_rate=step_imp_rate,
        median_improvement_rate=med_imp_rate,
        val_generalization_corr=val_gen_corr,
        hypothesis_alignment=float(np.mean(alignments)),
        n_off_lattice=n_off_lattice,
    )


def bootstrap_ci(task_means: pd.Series, n: int = 2000, seed: int = 0) -> Tuple[float, float]:
    x = task_means.dropna().to_numpy()
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boots = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return (float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)))


def score_experiment_trajectories(
    exp_dir: Path,
    oracle_dir: Path,
) -> pd.DataFrame:
    """
    Score all runs in `exp_dir / runs.jsonl` against the per-task oracle tables
    in `oracle_dir / <task_id>.csv.gz`.
    """
    exp_dir = Path(exp_dir)
    oracle_dir = Path(oracle_dir)
    runs_path = exp_dir / "runs.jsonl"
    summary_path = exp_dir / "summary.csv"

    summary_lookup: Dict[Tuple[str, str, str, int], Dict[str, Any]] = {}
    if summary_path.exists():
        sum_df = pd.read_csv(summary_path)
        for _, r in sum_df.iterrows():
            k = (str(r["task_id"]), str(r["model"]), str(r["framework_type"]), int(r["seed"]))
            summary_lookup[k] = r.to_dict()

    scorers: Dict[str, OracleScorer] = {}
    evaluators: Dict[str, Optional[StrategyEvaluator]] = {}
    val_caches: Dict[str, Dict[str, float]] = {}

    rows: List[Dict[str, Any]] = []
    for line in runs_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        cfg = rec.get("config", {})
        task_id = str(cfg.get("task_id", "SPY_full"))
        model = str(cfg.get("model", "unknown"))
        fw = str(cfg.get("framework_type", "unknown"))
        seed = int(cfg.get("seed", 0))

        if task_id not in scorers:
            oracle_path = oracle_dir / f"{task_id}.csv.gz"
            scorers[task_id] = OracleScorer(load_oracle(oracle_path))
            val_caches[task_id] = {}
            try:
                task_obj = get_task(task_id)
                evaluators[task_id] = StrategyEvaluator(task_obj.load())
            except Exception:
                evaluators[task_id] = None

        traj = rec.get("orchestrator_result", {}).get("trajectory", [])
        t_res = score_trajectory(
            traj,
            scorer=scorers[task_id],
            evaluator=evaluators[task_id],
            spec_val_cache=val_caches[task_id],
        )

        sum_row = summary_lookup.get((task_id, model, fw, seed), {})
        selected_pv = sum_row.get("percentile_val", np.nan)
        if selected_pv is None or (isinstance(selected_pv, float) and math.isnan(selected_pv)):
            # Fall back to scoring best_spec directly if summary had NaN
            best_spec_dict = rec.get("orchestrator_result", {}).get("best_spec")
            if best_spec_dict:
                best_res = score_trajectory(
                    [{"spec": best_spec_dict, "search_metrics": rec.get("orchestrator_result", {}).get("search_metrics", {})}],
                    scorer=scorers[task_id],
                    evaluator=evaluators[task_id],
                    spec_val_cache=val_caches[task_id],
                )
                selected_pv = best_res.step1_percentile_val

        row: Dict[str, Any] = {
            "task_id": task_id,
            "model": model,
            "framework_type": fw,
            "seed": seed,
            "n_steps": t_res.n_steps,
            "selected_percentile_val": float(selected_pv) if selected_pv is not None else np.nan,
            "auc_percentile_val": t_res.auc_percentile_val,
            "step1_percentile_val": t_res.step1_percentile_val,
            "final_best_percentile_val": t_res.final_best_percentile_val,
            "trajectory_val_gain": t_res.trajectory_val_gain,
            "step_improvement_rate": t_res.step_improvement_rate,
            "median_improvement_rate": t_res.median_improvement_rate,
            "val_generalization_corr": t_res.val_generalization_corr,
            "hypothesis_alignment": t_res.hypothesis_alignment,
            "n_off_lattice": t_res.n_off_lattice,
        }
        for step_i, pv_i in enumerate(t_res.best_percentile_val_so_far, 1):
            row[f"best_pv_step_{step_i}"] = pv_i
        rows.append(row)

    return pd.DataFrame(rows)


def aggregate_layer2_extensions(
    df: pd.DataFrame,
    group: str = "framework_type",
) -> pd.DataFrame:
    """
    Aggregate trajectory-level Layer 2 extension metrics following the two-stage
    benchmark protocol (mean over seeds per task, then equal-weight mean over tasks).
    """
    metrics = [
        "selected_percentile_val",
        "auc_percentile_val",
        "step1_percentile_val",
        "final_best_percentile_val",
        "trajectory_val_gain",
        "step_improvement_rate",
        "median_improvement_rate",
        "val_generalization_corr",
        "hypothesis_alignment",
    ]
    per_task = df.groupby(["task_id", group])[metrics].mean().reset_index()

    rows = []
    for g, sub in per_task.groupby(group):
        row: Dict[str, Any] = {
            group: g,
            "tasks": int(sub["task_id"].nunique()),
            "runs": int((df[group] == g).sum()),
        }
        for m in metrics:
            row[m] = float(sub[m].mean())
        auc_lo, auc_hi = bootstrap_ci(sub.set_index("task_id")["auc_percentile_val"])
        row["auc_ci_lo"] = auc_lo
        row["auc_ci_hi"] = auc_hi
        rows.append(row)

    lb = pd.DataFrame(rows).set_index(group)
    return lb.sort_values("auc_percentile_val", ascending=False)
