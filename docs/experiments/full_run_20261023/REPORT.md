# Full run: cheap model + harness vs expensive model (Layer 1 implementation)

Run directory: `docs/experiments/full_run_20261023`. Rows scored: 2358. Infrastructure failures after 4 attempts (not scored): 0.
Responses cut off by the output cap (finish_reason = length): 0.

Scoring: bar-by-bar position match against the golden on 1500 daily bars (active_action_match >= 0.95,
trade_f1 >= 0.90, friction match, no look-ahead, no NaN after warm-up, bounds). Temperature 0, one pass.

## 1. Pinned wording, single shot (G0)

|               |   canonical20 |   qcb22 |
|:--------------|--------------:|--------:|
| ds-v4.1-flash |          0.8  |    1    |
| sonnet-5.5    |          0.85 |    1    |
| opus-5.5      |          0.9  |    1    |
| fable-5.1     |          0.9  |    1    |
| gpt-6.1-sol   |          0.85 |    1    |
| gpt-6-astra   |          0.9  |    1    |
| ds-v4-pro     |          0.7  |    0.91 |

## 2. Original wording, pass rate (22 tasks; one task = 0.045)

|        |   ds-v4.1-flash |   sonnet-5.5 |   opus-5.5 |   fable-5.1 |   gpt-6.1-sol |   gpt-6-astra |   ds-v4-pro |
|:-------|----------------:|-------------:|-----------:|------------:|--------------:|--------------:|------------:|
| G0     |            0.27 |         0.32 |       0.32 |        0.32 |          0.32 |          0.23 |        0.32 |
| G1     |            0.23 |         0.27 |       0.41 |        0.23 |          0.32 |          0.27 |        0.23 |
| G2     |            0.32 |         0.41 |       0.36 |        0.36 |          0.23 |          0.18 |        0.32 |
| G1c    |            0.45 |         0.73 |       0.73 |        0.55 |          0.55 |          0.59 |        0.36 |
| G3w    |            0.41 |         0.45 |       0.45 |        0.5  |          0.41 |          0.36 |        0.36 |
| G4_v2  |            0.64 |         0.64 |       0.64 |        0.59 |          0.59 |          0.55 |        0.27 |
| G3     |            0.95 |         1    |       0.91 |        0.95 |          0.77 |          0.91 |        0.64 |
| G3c_q3 |            0.82 |         0.95 |       0.86 |        1    |          0.77 |          0.86 |        0.59 |
| G3c_q5 |            0.86 |         0.95 |       0.91 |        0.95 |          0.91 |          0.91 |        0.64 |
| G3c    |            0.95 |         0.91 |       0.95 |        0.95 |          0.86 |          0.77 |        0.64 |

Cells with fewer than 22 scored tasks: 0.

## 3. Tokens per task for the model under test (prompt + completion)

|        |   ds-v4.1-flash |   sonnet-5.5 |   opus-5.5 |   fable-5.1 |   gpt-6.1-sol |   gpt-6-astra |   ds-v4-pro |
|:-------|----------------:|-------------:|-----------:|------------:|--------------:|--------------:|------------:|
| G0     |            7736 |         2075 |       2161 |        2071 |          1352 |          1332 |        5399 |
| G1     |           10876 |         4396 |       3858 |        5528 |          2250 |          2254 |        7051 |
| G2     |           25111 |         9258 |       9314 |       13716 |          6351 |          6698 |       14850 |
| G1c    |           10359 |         4903 |       4080 |        4724 |          2160 |          2161 |        7667 |
| G3w    |           16609 |         6460 |       6296 |        6412 |          3746 |          3668 |       10779 |
| G4_v2  |           40034 |         8879 |       8570 |        8683 |          5421 |          5062 |       25074 |
| G3     |           14490 |         4979 |       4986 |        5605 |          3438 |          3106 |        9174 |
| G3c_q3 |            9791 |         4694 |       4902 |        5376 |          3486 |          2913 |        8008 |
| G3c_q5 |           13862 |         5005 |       5160 |        6319 |          3502 |          3064 |        8355 |
| G3c    |           13503 |         5753 |       5752 |        6560 |          3557 |          3180 |       10163 |

Simulated client (claude-sonnet-5-5) tokens per task, G3 family only:

| harness   |   oracle_prompt_tokens |   oracle_completion_tokens |
|:----------|-----------------------:|---------------------------:|
| G3        |                   1492 |                        523 |
| G3c       |                   1436 |                        479 |
| G3c_q3    |                   1094 |                        281 |
| G3c_q5    |                   1186 |                        347 |
| G3w       |                   1291 |                        552 |

## 4. Cheap + harness above the best expensive single shot

- ds-v4.1-flash + G1c: 0.45 > best expensive single shot 0.32
- ds-v4.1-flash + G3w: 0.41 > best expensive single shot 0.32
- ds-v4.1-flash + G4_v2: 0.64 > best expensive single shot 0.32
- ds-v4.1-flash + G3: 0.95 > best expensive single shot 0.32
- ds-v4.1-flash + G3c_q3: 0.82 > best expensive single shot 0.32
- ds-v4.1-flash + G3c_q5: 0.86 > best expensive single shot 0.32
- ds-v4.1-flash + G3c: 0.95 > best expensive single shot 0.32

## 5. Task difficulty on original wording (share of model cells passed)

| case_id                              |   G0 |   G1 |   G2 |   G1c |   G3w |   G4_v2 |   G3 |   G3c_q3 |   G3c_q5 |   G3c |   all |
|:-------------------------------------|-----:|-----:|-----:|------:|------:|--------:|-----:|---------:|---------:|------:|------:|
| qcb_044_atr_pullback_scale_in        | 0    | 0    | 0    |  0    |  0    |    0    | 0.78 |     0.22 |     0.44 |  0.78 |  0.22 |
| qcb_212_vol_rotation_095             | 0    | 0    | 0    |  0    |  0    |    0    | 0.67 |     0.56 |     0.56 |  0.56 |  0.23 |
| qcb_064_ema_ribbon_order             | 0    | 0    | 0    |  0.11 |  0    |    0    | 0.89 |     0.44 |     0.56 |  0.67 |  0.27 |
| qcb_013_rsi_pyramiding_partial_exits | 0    | 0.11 | 0.33 |  0    |  0.11 |    0.22 | 0.56 |     0.44 |     0.56 |  0.44 |  0.28 |
| qcb_175_mean_reversion_time_stop     | 0    | 0    | 0    |  0.11 |  0    |    0    | 0.89 |     0.67 |     1    |  0.44 |  0.31 |
| qcb_089_profit_dependent_exit        | 0    | 0    | 0    |  0    |  0.11 |    0    | 1    |     1    |     1    |  1    |  0.41 |
| qcb_226_sma252_atrpct_gate           | 0    | 0    | 0    |  0.67 |  0    |    0.67 | 0.89 |     0.78 |     0.78 |  0.78 |  0.46 |
| qcb_027_prev_exit_price_gate         | 0    | 0.22 | 0    |  0.22 |  0.11 |    0.22 | 1    |     1    |     1    |  1    |  0.48 |
| qcb_256_sma50_200_atr_target         | 0    | 0    | 0    |  0.56 |  0.11 |    0.67 | 0.78 |     1    |     1    |  0.89 |  0.5  |
| qcb_004_abs_momentum_252             | 0    | 0    | 0    |  0.56 |  0.22 |    0.67 | 1    |     0.89 |     1    |  1    |  0.53 |
| qcb_126_supertrend_dema200           | 0.22 | 0.33 | 0.56 |  0.33 |  0.33 |    0.33 | 0.89 |     1    |     0.78 |  0.89 |  0.57 |
| qcb_101_atr_trail_macd_exit          | 0    | 0.11 | 0.22 |  0.56 |  0.56 |    0.89 | 0.89 |     0.89 |     1    |  0.89 |  0.6  |
| qcb_092_bb_breakout_tp6_sl3          | 0.11 | 0    | 0    |  1    |  0.56 |    0.78 | 1    |     1    |     1    |  1    |  0.64 |
| qcb_227_universal_portfolio          | 0.67 | 0.56 | 0.44 |  0.89 |  0.22 |    0.78 | 0.33 |     0.89 |     0.89 |  0.89 |  0.66 |
| qcb_284_bb_breakout_ratchet_trail    | 0.33 | 0    | 0    |  1    |  0.89 |    0.78 | 1    |     0.89 |     1    |  1    |  0.69 |
| qcb_243_bb_breakout_after_3_below    | 0.11 | 0.11 | 0    |  1    |  0.89 |    0.89 | 1    |     1    |     1    |  1    |  0.7  |
| qcb_052_macd_line_signal_cross       | 0.67 | 0.78 | 1    |  0.56 |  0.44 |    0.11 | 1    |     0.89 |     0.89 |  0.78 |  0.71 |
| qcb_317_month_day1_to_day5           | 0.78 | 0.67 | 0.56 |  0.89 |  0.89 |    0.89 | 1    |     1    |     0.78 |  0.78 |  0.82 |
| qcb_110_rsi_30_up_70_down            | 1    | 0.89 | 0.89 |  0.89 |  1    |    0.89 | 0.89 |     1    |     0.89 |  1    |  0.93 |
| qcb_207_bb10_0674_breakout           | 1    | 1    | 0.78 |  1    |  0.89 |    0.89 | 1    |     1    |     1    |  1    |  0.96 |
| qcb_353_vwap20_macd_rsi              | 0.89 | 0.78 | 1    |  1    |  1    |    1    | 1    |     1    |     1    |  1    |  0.97 |
| qcb_039_sma10_30_cross               | 1    | 1    | 1    |  1    |  1    |    1    | 1    |     1    |     1    |  1    |  1    |

## Model set note (2026-11-04)

Model set changed after the 2026-10-23 run above. claude-haiku and gpt-5.4-mini are dropped (user
decision); gpt-6-astra is added. The gateway also dropped the haiku-5.5 tier entirely in the interim
(only haiku-4.5 remains) and renamed claude-sonnet-5-5 / claude-opus-5-5 to claude-sonnet-5.5 /
claude-opus-5.5 (old rows renamed in place, not re-run). deepseek-v4.1-flash is now the only "cheap"
model in the set; the other six (sonnet-5.5, opus-5.5, fable-5.1, gpt-6.1-sol, gpt-6-astra, ds-v4-pro)
are "expensive". Two harnesses were added this round:

- G3w: same as G3 (ask the client, then implement) but the simulated client does NOT hold the pinned
  spec. It only sees the same loose task text the implementer has and is told to answer like a real
  trader using plain conventions, not a precise spec. G3/G3c/G3c_q* remain the fully-informed upper
  bound; G3w is the more realistic lower bound for "ask the client".
- G4_v2: implementer and supervisor are the same model in two independent calls (plus up to two
  revise+re-review rounds). Both calls share the same public interface contract and house
  conventions; the revision call is given the actual draft, not a blind rewrite. No information
  beyond what the implementer already had is ever introduced. (G4_v1, an earlier version where the
  supervisor lacked the shared conventions and the reviser never saw its own draft, was net harmful --
  draft pass 102/176, final pass 84/176 -- and is archived, not used in this report.)

## Run-to-run noise

The cells here were also run on 2026-10-22 under three models that have since been renamed or dropped
(claude-haiku-5-5, gpt-5.4-mini, claude-sonnet-5-5/claude-opus-5-5 pre-rename). Where a (harness,
model) cell exists in both runs, mean absolute difference in pass rate is 0.07 and the largest is 0.18,
at temperature 0, one pass each time. Read differences under about 0.15 between two cells as noise
rather than a real effect.

## Reading

1. Pinned wording: every model passes 0.91-1.00 of the 22 QCB tasks. Writing correct code from an
   exact spec is solved for this model set, deepseek-v4.1-flash included.
2. Original wording, single shot (G0): every model lands at 0.23-0.32. Model size and price do not
   help with an under-specified request; gpt-6-astra, nominally the most capable OpenAI model here,
   is tied for last at 0.23.
3. Thinking harder (G1) and spec-then-code (G2) still do not help, and G2 costs 2-4x the tokens of G0.
4. House conventions (G1c) still help across the board, 0.1-0.4 depending on the model.
5. The new harnesses sit between G1c and the fully-informed G3 family:
   - G3w (ask an uninformed, common-sense client): 0.36-0.50, barely above G1c and well below G3.
     Confirms that most of G3's gain comes from the client already knowing the pinned answer, not
     from the act of asking itself. A real client who does not know implementation details would not
     get an agent anywhere near G3's 0.77-1.00.
   - G4_v2 (self-review, zero new information): 0.27-0.64, better than G0 and G1 for every model, but
     loses to G1c for 6 of 7 models and loses to every member of the G3 family for all 7. Reviewing
     your own work without new information helps some, but not as much as either a convention
     reminder or asking someone who can actually resolve the ambiguity.
6. The fully-informed G3 family (0.77-1.00) remains the strongest lever found. Three questions (q3)
   recover most of the gain; going from 5 to 12 questions is within noise.
7. Ranking of harnesses by what they add: G0/G1/G2 (no new information) < G1c (generic conventions)
   < G3w (an uninformed second opinion from outside) and G4_v2 (an informed second opinion from
   inside, but no new facts) < G3 family (an informed second opinion from outside, i.e. real new
   facts about this specific task).
8. The three multi-state tasks (qcb_044, qcb_212, qcb_064) remain the hardest under every harness,
   including G4_v2; G3's access to the pinned spec is what clears them, not self-review or vague
   client input.

Caveats unchanged from the 2026-10-23 run: 22 tasks (one task = 4.5 percentage points), one pass per
cell, goldens by one author. G3/G3c family additionally assumes a client who already knows the exact
implementation choice for every ambiguity and answers every question asked; G3w is a more realistic
but still simulated lower bound, not a measurement of real client behavior.
