import pytest
from agent.config import DataSplitConfig, ExperimentConfig

def test_data_split_config_defaults():
    split = DataSplitConfig()
    assert split.search_ratio == 0.6
    assert split.select_ratio == 0.2
    assert split.test_ratio == 0.2
    assert split.validate() is True

def test_data_split_config_invalid():
    split = DataSplitConfig(search_ratio=0.5, select_ratio=0.5, test_ratio=0.5)
    with pytest.raises(ValueError, match="Sum of ratios must be 1.0"):
        split.validate()

def test_experiment_config_options():
    cfg = ExperimentConfig(
        framework_type="C",
        max_iterations=15,
        memory_length=3,
        seed=100
    )
    assert cfg.framework_type == "C"
    assert cfg.patience == 3
    assert "sharpe_ratio" in cfg.feedback_metrics
    assert cfg.selection_metric == "sharpe_ratio"
