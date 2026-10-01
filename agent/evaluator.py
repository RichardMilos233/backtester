import os
from typing import Tuple, Dict, Any, List, Optional
import pandas as pd

from agent.spec import StrategySpec
from agent.config import DataSplitConfig
from src.signals import (
    compute_moving_average_signal,
    compute_macd_signal,
    compute_volatility_scaled_signal,
)
from src.backtest import Backtest
from src.metrics import compute_strategy_metrics

class StrategyEvaluator:
    def __init__(
        self,
        df: pd.DataFrame,
        split_config: Optional[DataSplitConfig] = None,
        initial_capital: float = 100000.0,
    ):
        self.df = df.sort_index()
        self.split_config = split_config or DataSplitConfig()
        self.split_config.validate()
        self.initial_capital = initial_capital
        self._search_df, self._select_df, self._test_df = self._create_splits()

    def _create_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        n = len(self.df)
        n_search = int(n * self.split_config.search_ratio)
        n_select = int(n * self.split_config.select_ratio)
        
        search_df = self.df.iloc[:n_search]
        select_df = self.df.iloc[n_search:n_search + n_select]
        test_df = self.df.iloc[n_search + n_select:]
        return search_df, select_df, test_df

    def split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return self._search_df, self._select_df, self._test_df

    def _generate_signals_for_split(self, spec: StrategySpec, split_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Calculates signals strictly forward in time.
        Uses historical lookback buffer from full df to warm up indicators without lookahead leakage.
        """
        # Determine warm-up window needed
        if spec.signal_type == "momentum":
            window = spec.params["window"]
        elif spec.signal_type == "macd":
            window = spec.params.get("slow", 26) + spec.params.get("signal_span", 9)
        else:
            raise ValueError(f"Unsupported signal_type: {spec.signal_type}")

        start_loc = self.df.index.get_loc(split_df.index[0])
        buffer_start = max(0, start_loc - window * 2)
        warmup_df = self.df.iloc[buffer_start: start_loc + len(split_df)]

        if spec.signal_type == "momentum":
            full_signal = compute_moving_average_signal(warmup_df, window=spec.params["window"])
        elif spec.signal_type == "macd":
            full_signal = compute_macd_signal(
                warmup_df,
                fast=spec.params["fast"],
                slow=spec.params["slow"],
                signal_span=spec.params["signal_span"]
            )
        
        if spec.use_vol_scaling:
            if "rolling_vol" not in warmup_df.columns:
                warmup_df = warmup_df.copy()
                warmup_df["rolling_vol"] = warmup_df["close"].pct_change().rolling(20).std() * (252 ** 0.5)
            full_signal = compute_volatility_scaled_signal(
                warmup_df,
                full_signal,
                target_vol=spec.vol_target if spec.vol_target is not None else 0.15,
                max_leverage=spec.max_leverage,
            )

        if not spec.use_vol_scaling:
            split_signal = full_signal.loc[split_df.index].fillna(0).astype(int)
        else:
            split_signal = full_signal.loc[split_df.index].fillna(0).astype(float)
        return split_df, split_signal

    def evaluate_spec(self, spec: StrategySpec, split_name: str) -> Dict[str, Any]:
        if split_name == "search":
            split_df = self._search_df
        elif split_name == "select":
            split_df = self._select_df
        elif split_name == "test":
            split_df = self._test_df
        else:
            raise ValueError(f"Unknown split_name: {split_name}")

        is_valid, reason = spec.validate()
        if not is_valid:
            raise ValueError(f"Invalid spec {spec.spec_id}: {reason}")

        data_slice, signals = self._generate_signals_for_split(spec, split_df)
        bt = Backtest(initial_capital=self.initial_capital)
        bt_result = bt.run(data_slice, signals)
        metrics_series = compute_strategy_metrics(bt_result)

        metrics_dict = metrics_series.to_dict()
        metrics_dict["initial_capital"] = self.initial_capital
        metrics_dict["spec_id"] = spec.spec_id
        metrics_dict["split_name"] = split_name
        return metrics_dict

    def evaluate_search(self, spec: StrategySpec, feedback_metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        full_res = self.evaluate_spec(spec, "search")
        if not feedback_metrics:
            return full_res
        filtered = {k: full_res[k] for k in feedback_metrics if k in full_res}
        filtered["spec_id"] = spec.spec_id
        return filtered

    def evaluate_selection(self, spec: StrategySpec) -> Dict[str, Any]:
        return self.evaluate_spec(spec, "select")

    def evaluate_test(self, spec: StrategySpec) -> Dict[str, Any]:
        return self.evaluate_spec(spec, "test")
