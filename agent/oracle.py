"""
Oracle: exhaustive evaluation of the canonical spec lattice on the Val ("select")
and Test splits of a task.

Why: raw Test Sharpe is dominated by the market regime of the split, not by the
agent. Scoring an agent's pick against the full distribution of the space it was
searching (percentile on Val, regret vs. the best Test spec) removes that.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

import numpy as np
import pandas as pd

from agent.evaluator import StrategyEvaluator
from agent.space import enumerate_space, spec_key
from agent.spec import StrategySpec

ORACLE_METRICS = ("sharpe_ratio", "total_return", "max_drawdown", "turnover", "calmar_ratio")


def build_oracle_table(
    evaluator: StrategyEvaluator,
    include_vol: bool = True,
    progress: Optional[Callable[[int, int], None]] = None,
) -> pd.DataFrame:
    specs = enumerate_space(include_vol=include_vol)
    rows = []
    t0 = time.time()
    for i, spec in enumerate(specs, 1):
        row: Dict = {
            "spec_key": spec_key(spec),
            "signal_type": spec.signal_type,
            "params": str(spec.params),
            "use_vol_scaling": spec.use_vol_scaling,
            "vol_target": spec.vol_target,
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
        if progress and (i % 500 == 0 or i == len(specs)):
            progress(i, len(specs))
    df = pd.DataFrame(rows).set_index("spec_key")
    df.attrs["elapsed_seconds"] = round(time.time() - t0, 1)
    return df


def save_oracle(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)


def load_oracle(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col="spec_key")


class OracleScorer:
    """Converts an agent's final pick into regime-free benchmark scores."""

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
