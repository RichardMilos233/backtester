from dataclasses import dataclass, field
from typing import List

@dataclass
class DataSplitConfig:
    search_ratio: float = 0.6
    select_ratio: float = 0.2
    test_ratio: float = 0.2

    def validate(self) -> bool:
        total = round(self.search_ratio + self.select_ratio + self.test_ratio, 6)
        if total != 1.0:
            raise ValueError(f"Sum of ratios must be 1.0, got {total}")
        if any(r <= 0 for r in (self.search_ratio, self.select_ratio, self.test_ratio)):
            raise ValueError("All split ratios must be positive numbers")
        return True

ALLOWED_FRAMEWORKS = (
    "A", "B", "C", "D",
    "C-sota", "C-corr", "C-reflect", "C-hier",
    "R-random", "R-grid",               # Non-LLM baselines (same search budget, no LLM calls)
)

@dataclass
class ExperimentConfig:
    framework_type: str = "B"           # Allowed: A, B, C, D, C-sota, C-corr, C-reflect, C-hier, R-random, R-grid
    model: str = "mock"                 # Foundation model name or "mock"
    task_id: str = "SPY_full"           # Benchmark task (see benchmark/layer2_search/tasks.py); determines data + oracle
    max_iterations: int = 10            # Stop condition: max iterations (legacy)
    max_backtests: int = 8              # Fair budget: max evaluated backtests on search set
    max_proposals_multiplier: int = 3   # Safety limit: max proposals = max_backtests * multiplier
    memory_length: int = 5              # Knob: past iterations visible to Researcher
    feedback_metrics: List[str] = field(default_factory=lambda: [
        "sharpe_ratio", "max_drawdown", "turnover", "total_return"
    ])                                  # Knob: metrics fed back in prompt
    selection_metric: str = "sharpe_ratio" # Knob: metric used by Orchestrator on selection set
    early_stop: bool = True             # Benchmark runs set False so every method spends the full budget
    min_improvement: float = 0.01       # Stop condition: early stopping threshold
    patience: int = 3                   # Stop condition: patience rounds
    seed: int = 42                      # Reproducibility seed
    initial_capital: float = 100000.0   # Backtester capital
    correlation_threshold: float = 0.95 # Gate threshold for C-corr framework
    data_split: DataSplitConfig = field(default_factory=DataSplitConfig)

    def __post_init__(self):
        if self.framework_type not in ALLOWED_FRAMEWORKS:
            raise ValueError(
                f"Invalid framework_type: '{self.framework_type}'. "
                f"Must be one of {list(ALLOWED_FRAMEWORKS)}."
            )
        self.data_split.validate()
        # Ensure selection_metric is always available in Researcher feedback / early-stop
        if self.selection_metric not in self.feedback_metrics:
            self.feedback_metrics.append(self.selection_metric)
