# TIA benchmark: can an LLM agent turn a trader's words into the right strategy code?

Repo for the Trading-Intelligence-Agent (TIA) benchmark. It started as a CTA backtesting course
project (`src/`), grew a three-layer benchmark design, and has since narrowed to one question:
**given a strategy described in plain language, does the agent write code whose positions match the
reference implementation?** The directory tree follows that narrowing.

```
benchmark/                 the benchmark package (was agent/)
  common/                  LLM client, StrategySpec, split config, StrategyEvaluator, market loaders
  layer1_code/             LAYER 1  strategy-code ability        <- current focus
    audit/                   1A  spot the flawed strategy
    impl/                    1B  write the correct strategy
      scoring/                   how a submission is judged
      harness/                   model x harness experiment configs
      tasks/                     the task bank
    ambiguity/               side task: ask instead of guessing
    canonical.py             the 40-case suite (20 audit + 20 impl) behind the 8-model leaderboard
  layer2_search/           LAYER 2  in-space strategy search vs an exhaustive oracle   (paused)
  layer3_oos/              LAYER 3  planted-alpha / out-of-sample validation            (paused)
src/                       backtest engine: data, signals, regimes, backtest, metrics, ml
scripts/<layer>/           CLI entry points, same split
tests/<layer>/             pytest, same split
results/<layer>/           experiment outputs (summary.csv, leaderboards, REPORT.md; raw runs.jsonl ignored)
docs/                      design notes and HTML reports
data/                      daily OHLCV (not committed; see data/README.md)
```

## The three layers

| Layer | Question | Where | Status |
|---|---|---|---|
| 1 | Can the agent read and write strategy code correctly? | `benchmark/layer1_code/` | active |
| 2 | Given a fixed strategy space and a backtest budget, does the agent find a good strategy? | `benchmark/layer2_search/` | paused: LLM search did not beat random search at equal budget |
| 3 | Does what it finds hold out of sample? | `benchmark/layer3_oos/` | paused: synthetic planted-alpha series built, not run |

`docs/design/BENCHMARK_DESIGN_DISCUSSION.html` is the full three-layer discussion;
`docs/design/BENCHMARK_PROTOCOL.md` has the scoring rules and commands for all three.

## Layer 1 in two halves

**1A, audit (`layer1_code/audit/`).** The model gets a strategy plus its backtest wiring and must say
whether there is a flaw, which kind, and the mechanism. Twenty canonical cases: ten subtle
cross-module / pandas-default traps, six classic flaws, four clean controls. A pass needs the right
flaw label and the right mechanism in the explanation.

**1B, implementation (`layer1_code/impl/`).** The model gets a natural-language strategy description
and must write `build_strategy(df)`. It is scored bar by bar against a hand-written golden. This is
where the work is now, and it has three parts that map to three subdirectories:

### How a submission is judged: `impl/scoring/`

- `sandbox.py` extracts the code block, compiles it in a restricted namespace, and runs a look-ahead probe
  (truncate the frame, perturb future prices, the positions up to the cut must not change).
- `position_match.py` is the scorer. A task passes only if all hold: executes, no look-ahead, no NaN after
  warm-up, weights inside bounds, friction parameters match, ActiveActionMatch >= 0.95 on bars where either
  side holds a position, and round-trip TradeF1 >= 0.90.
- `systematic.py` / `factor_usage.py` are the older Track A scorer (factor-library API and friction) kept
  for the canonical suite.

Walkthrough with examples: `docs/layer1_impl/LAYER1_IMPL_SCORING.html`.

### How experiments are configured: `impl/harness/`

A harness is everything the agent does between receiving the prompt and returning code, with one model
and no access to the golden. Each is a (generator, decider) pair in `harness.py`:

| name | what the model does |
|---|---|
| G0 | one shot |
| G1 / G1c | list the ambiguities, decide each, then code; `c` adds a house-conventions system prompt |
| G2 | write a spec in one call, code in a second |
| G3 / G3c / G3c_qN | ask a simulated client up to N questions, then code; the client holds the pinned spec |
| G3w | same, but the client only has the loose wording too |
| G4_v2 | implementer + supervisor, same model, up to two revise rounds |

Run the full model x harness matrix with `scripts/layer1_impl/run_full_matrix.sh`; one cell with
`scripts/layer1_impl/run_harness_matrix.py --harness G3c --models ... --task-set qcb22 --variant original`;
aggregate with `scripts/layer1_impl/report_full_matrix.py <run_dir>`. Rows are resumable and every LLM call
is counted so cost curves can be drawn.

### How the tasks are designed: `impl/tasks/`

Every task is a `HighFreedomImplCase` (`schema.py`): a prompt, a golden `build_strategy`, and optionally an
`original_prompt`. `indicators.py` fixes what SMA, EMA, Wilder RSI/ATR, Bollinger and MACD mean, and every
golden uses those helpers so the conventions in the prompt and in the answer agree.

- `qcb_cases.py`: 22 strategies ported from QuantCode-Bench. Each carries the original loose wording and a
  pinned, unambiguous rewrite; both share one golden. The screening and pinning decisions are in
  `results/layer1_impl/qcb_screen.md`.
- `v3_hard_cases.py`: 14 unhinted tasks across seven real-world failure zones (formula substitution,
  calendar vs bar, coupled state, multi-state automata, pulse vs level triggers, intrabar execution,
  substrategy renormalisation). Track B of the canonical 20.
- `systematic_cases.py`: factor-library / friction tasks. Six of them are Track A of the canonical 20.
- `hf_cases.py`, `hf_generator.py`: the earlier 10 hand-written prototypes and a deterministic 60-case
  scaled suite.
- `task_bank.py`: named sets `canonical20`, `qcb22`, `all`.

## Headline results so far

Full matrix, 7 models x 10 harnesses x 22 QuantCode tasks, original wording
(`results/layer1_impl/full_matrix_20261023/REPORT.md`):

- With the spec pinned, every model passes 0.91 to 1.00. The difficulty is in the loose wording, not the code.
- Single shot on loose wording, every model lands at 0.23 to 0.32, cheap or expensive.
- Asking the client (G3 family) lifts the cheapest model to 0.95, above every expensive model's single shot.
  House conventions alone (G1c) get 0.45 to 0.73. Reasoning harder about ambiguities without asking (G1, G2)
  does not help.

Canonical 40 (20 audit + 20 impl), 8 models: `docs/layer1_audit/LAYER1_BENCHMARK_SHOWCASE.html`, full
320-row table in `docs/layer1_audit/LAYER1_FULL_EXPERIMENT_TABLE.html`.

## Running

```bash
python3.9 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/data/fetch_ohlcv.py SPY QQQ TLT GLD IWM EFA
.venv/bin/python -m pytest -q                      # 234 tests, no network
set -a; source .env; set +a                        # LLM gateway keys for real runs
.venv/bin/python scripts/layer1_impl/run_harness_matrix.py --harness G0 --models deepseek-v4.1-flash \
    --task-set qcb22 --variant original --out results/layer1_impl/my_run
```

`src/` is the backtest engine the tasks run on (next-open execution, slippage and commission, rebalance
tolerance band, four-quadrant regime labels). Its course-project write-up is `docs/engine/FINAL_RESEARCH_REPORT.md`.
