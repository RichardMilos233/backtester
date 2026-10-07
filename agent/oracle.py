"""
Oracle: exhaustive evaluation of the canonical spec lattice on the Val ("select")
and Test splits of a task.

Why: raw Test Sharpe is dominated by the market regime of the split, not by the
agent. Scoring an agent's pick against the full distribution of the space it was
searching (percentile on Val, regret vs. the best Test spec) removes that.

When the lattice is too large (e.g. with regimes, ~108k specs), use
build_oracle_table(..., sample=N) to evaluate a uniform random sample instead of
exhaustive enumeration. Percentile estimates from a sampled table are
approximations; see OracleScorer docstring.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

import numpy as np
import pandas as pd

from agent.evaluator import StrategyEvaluator
from agent.space import enumerate_space, sample_space, spec_key, _count_space
from agent.spec import StrategySpec

ORACLE_METRICS = ("sharpe_ratio", "total_return", "max_drawdown", "turnover", "calmar_ratio")

_AUTO_SAMPLE_THRESHOLD = 20_000
_AUTO_SAMPLE_DEFAULT = 20_000


def build_oracle_table(
    evaluator: StrategyEvaluator,
    include_vol: bool = True,
    include_regimes: bool = False,
    sample: Optional[int] = None,
    seed: int = 42,
    progress: Optional[Callable[[int, int], None]] = None,
) -> pd.DataFrame:
    """Evaluate specs from the canonical lattice and return a DataFrame indexed by spec_key.

    Parameters
    ----------
    include_vol : bool
        Whether to include vol-scaling overlays.
    include_regimes : bool
        Whether to include regime filter options.
    sample : int or None
        If given, evaluate a random sample of this size from the lattice instead
        of exhaustive enumeration. When the lattice exceeds 20k specs and sample
        is not given, defaults to 20k with a printed warning.
    seed : int
        RNG seed for sample_space (ignored when sample is None).
    progress : callable or None
        Called with (i, n) every 500 specs.
    """
    total_lattice = _count_space(include_vol=include_vol, include_regimes=include_regimes)

    if sample is None and total_lattice > _AUTO_SAMPLE_THRESHOLD:
        sample = _AUTO_SAMPLE_DEFAULT
        print(
            f"[oracle] Lattice has {total_lattice:,} specs (> {_AUTO_SAMPLE_THRESHOLD:,}). "
            f"Defaulting to --sample {sample}. Pass sample= explicitly to suppress."
        )

    sampled = sample is not None
    if sampled:
        specs = sample_space(n=sample, seed=seed, include_vol=include_vol, include_regimes=include_regimes)
    else:
        specs = enumerate_space(include_vol=include_vol, include_regimes=include_regimes)

    n_evaluated = len(specs)
    rows = []
    t0 = time.time()
    for i, spec in enumerate(specs, 1):
        row: Dict = {
            "spec_key": spec_key(spec),
            "signal_type": spec.signal_type,
            "params": str(spec.params),
            "use_vol_scaling": spec.use_vol_scaling,
            "vol_target": spec.vol_target,
            "allowed_regimes": str(spec.allowed_regimes) if spec.allowed_regimes else None,
        }
        for split, prefix in (("select", "val"), ("test", "test")):
            try:
                m = evaluator.evaluate_spec(spec, split)
            except Exception as e:  # keep the table complete; a broken spec scores NaN
                m = {}
                row[f"{prefix}_error"] = str(e)
            for k in ORACLE_METRICS:
                row[f"{prefix}_{k}"] = m.get(k, np.nan)
        rows.append(row)
        if progress and (i % 500 == 0 or i == n_evaluated):
            progress(i, n_evaluated)

    df = pd.DataFrame(rows).set_index("spec_key")
    df.attrs["elapsed_seconds"] = round(time.time() - t0, 1)
    df.attrs["n_total_lattice"] = total_lattice
    df.attrs["n_evaluated"] = n_evaluated
    df.attrs["sampled"] = sampled
    df.attrs["include_vol"] = include_vol
    df.attrs["include_regimes"] = include_regimes
    df.attrs["seed"] = seed
    return df


def save_oracle(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    # Write sidecar meta JSON alongside the CSV (or CSV.GZ)
    meta = {
        "n_total_lattice": df.attrs.get("n_total_lattice"),
        "n_evaluated": df.attrs.get("n_evaluated"),
        "sampled": df.attrs.get("sampled"),
        "include_vol": df.attrs.get("include_vol"),
        "include_regimes": df.attrs.get("include_regimes"),
        "seed": df.attrs.get("seed"),
        "elapsed_seconds": df.attrs.get("elapsed_seconds"),
    }
    meta_path = path.parent / (path.name.split(".")[0] + ".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def load_oracle(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col="spec_key")


class OracleScorer:
    """Converts an agent's final pick into regime-free benchmark scores.

    When built from a *sampled* oracle table (sampled=True in df.attrs),
    percentile estimates are approximations based on the sample distribution.
    The larger the sample relative to the lattice, the more accurate they are.
    """

    def __init__(self, oracle: pd.DataFrame, metric: str = "sharpe_ratio"):
        self.oracle = oracle
        self.metric = metric
        self._val = oracle[f"val_{metric}"].dropna().to_numpy()
        self._test = oracle[f"test_{metric}"].dropna().to_numpy()
        self._val_sorted = np.sort(self._val)
        self._test_sorted = np.sort(self._test)
        # Oracle references
        self.best_test = float(self._test.max())
        self.best_val_key = oracle[f"val_{metric}"].idxmax()
        # Test score of the spec an omniscient Val-selector would have picked:
        # the ceiling for any method that selects on Val.
        self.val_oracle_test = float(oracle.loc[self.best_val_key, f"test_{metric}"])
        self.median_test = float(np.median(self._test))

    def percentile(self, value: float, split: str) -> float:
        arr = self._val_sorted if split == "val" else self._test_sorted
        if value is None or np.isnan(value):
            return float("nan")
        return float(np.searchsorted(arr, value, side="right") / len(arr))

    def score(self, val_value: float, test_value: float) -> Dict[str, float]:
        return {
            "percentile_val": self.percentile(val_value, "val"),
            "percentile_test": self.percentile(test_value, "test"),
            "regret_test": self.best_test - test_value if test_value is not None else float("nan"),
            "regret_vs_val_oracle": self.val_oracle_test - test_value if test_value is not None else float("nan"),
            "excess_over_median_test": test_value - self.median_test if test_value is not None else float("nan"),
        }

    def score_spec(self, spec: StrategySpec, evaluator: Optional[StrategyEvaluator] = None) -> Dict[str, float]:
        key = spec_key(spec)
        if key in self.oracle.index:
            v = float(self.oracle.loc[key, f"val_{self.metric}"])
            t = float(self.oracle.loc[key, f"test_{self.metric}"])
        elif evaluator is not None:  # off-lattice (e.g. odd MACD ints): evaluate directly
            v = float(evaluator.evaluate_selection(spec)[self.metric])
            t = float(evaluator.evaluate_test(spec)[self.metric])
        else:
            raise KeyError(f"{key} not in oracle and no evaluator provided")
        out = self.score(v, t)
        out["on_lattice"] = key in self.oracle.index
        return out

    def summary(self) -> Dict[str, float]:
        return {
            "n_specs": int(len(self.oracle)),
            "best_test": self.best_test,
            "best_val_key": self.best_val_key,
            "val_oracle_test": self.val_oracle_test,
            "median_test": self.median_test,
            "median_val": float(np.median(self._val)),
            "p90_test": float(np.quantile(self._test, 0.9)),
            "p90_val": float(np.quantile(self._val, 0.9)),
            "val_test_rank_corr": float(pd.Series(self._val).corr(pd.Series(self._test), method="spearman"))
            if len(self._val) == len(self._test) else float("nan"),
        }
