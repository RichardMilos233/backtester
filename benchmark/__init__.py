"""
TIA benchmark: does an LLM agent turn a trader's strategy description into correct code?

Layout mirrors the three-layer design (see README.md at the repo root):
  common/         LLM client, StrategySpec, split config, StrategyEvaluator (used by every layer)
  layer1_code/    Layer 1  - strategy-code ability: audit/ (spot the flawed strategy),
                  impl/ (write the correct strategy: scoring, harness, tasks), ambiguity/
  layer2_search/  Layer 2  - in-space strategy search, oracle-normalised scoring
  layer3_oos/     Layer 3  - planted-alpha / out-of-sample validation
The backtest engine itself lives in `src/` (its import path is part of the task prompts).
"""
