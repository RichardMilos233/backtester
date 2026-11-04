# Cheap model + harness vs expensive model single-shot (Layer 1 implementation)

Task set: 22 strategies ported from QuantCode-Bench (`benchmark/layer1_code/impl/tasks/qcb_cases.py`), each with a pinned
specification and the original loose wording, one golden per task. Scored on 1500 daily bars
(SPY; TLT for qcb_027, IWM for qcb_004) by bar-by-bar position match against the golden
(active_action_match >= 0.95, trade_f1 >= 0.90, friction match, no look-ahead, no NaN, bounds).
Pass rate = fraction of the 22 tasks passed. All runs temperature 0, one seed.

## 1. Pinned wording, single shot (saturated)

| harness   |   claude-haiku-5-5 |   deepseek-v4.1-flash |   gpt-5.4-mini |
|:----------|-------------------:|----------------------:|---------------:|
| G0        |                  1 |                     1 |              1 |

## 2. Original wording: cheap models under each harness

| harness   |   claude-haiku-5-5 |   deepseek-v4.1-flash |   gpt-5.4-mini |   calls/task |
|:----------|-------------------:|----------------------:|---------------:|-------------:|
| G0        |               0.41 |                  0.36 |           0.23 |            1 |
| G1c       |               0.64 |                  0.55 |           0.41 |            1 |
| G3c_q3    |               0.77 |                  0.73 |           0.77 |            2 |
| G3c_q5    |               0.86 |                  0.91 |           0.77 |            2 |
| G3c       |               0.95 |                  0.91 |           0.82 |            2 |

Harnesses (one model throughout; the model never sees the golden or any sandbox output):
- G0: bare single shot.
- G1c: single shot with a house-conventions system prompt (indicator formulas, close-based tests,
  exit-before-entry ordering, sizing -> weight 1.0, warm-up flat).
- G3c_qN: the model first asks up to N clarifying questions; a fixed simulated client
  (claude-sonnet-5-5 holding the pinned spec, answering only what is asked) replies; the model then
  implements with the Q&A. Conventions prompt included.

Tried and dropped (12-task subset): G1 = enumerate ambiguities then code, no conventions (0.17 for
all three models); G2 = spec-then-code in two calls (0.00-0.25, more exec failures, 2x tokens).

| harness   |   claude-haiku-5-5 |   deepseek-v4.1-flash |   gpt-5.4-mini |
|:----------|-------------------:|----------------------:|---------------:|
| G1        |               0.17 |                  0.17 |           0.17 |
| G2        |               0.17 |                  0    |           0.25 |

## 3. Original wording: expensive models, single shot

| harness   |   claude-sonnet-5-5 |   claude-opus-5-5 |   claude-fable-5.1 |   gpt-6.1-sol |   deepseek-v4-pro |
|:----------|--------------------:|------------------:|-------------------:|--------------:|------------------:|
| G0        |                0.32 |              0.27 |               0.27 |          0.27 |              0.27 |

gpt-6.1-sol refused to guess on two tasks (raised "supply sell_below10 / longcondition"), which is
scored as a failure here but is the behaviour G3 formalises.

## 4. Tokens per task (prompt + completion, cheap models, original wording)

| harness   |   claude-haiku-5-5 |   deepseek-v4.1-flash |   gpt-5.4-mini |
|:----------|-------------------:|----------------------:|---------------:|
| G0        |               2917 |                  8502 |           1160 |
| G1c       |               6401 |                  9799 |           2054 |
| G3c_q3    |               7077 |                 10351 |           2837 |
| G3c_q5    |               7123 |                 12658 |           2959 |
| G3c       |               7907 |                 13604 |           3517 |

## 5. Reading

- With the spec pinned, every model in the set passes essentially everything; the benchmark's
  difficulty lives in turning loose wording into a definite implementation.
- Reasoning harder about the ambiguity (G1) does not help; stating house conventions (G1c) does;
  asking the client (G3) does most. Three questions recover most of the gain; the multi-state tasks
  (qcb_013, qcb_044, qcb_212) need five or more.
- Cheap + G1c already exceeds every expensive model's single shot; cheap + G3c (0.82-0.95) is about
  three times the expensive single-shot pass rate (0.27-0.32).
- Open: expensive models were not run under G1c/G3c; the comparison that matters for the product
  decision is cheap+G3c vs expensive+G3c.
- Caveats: 22 tasks (one task = 0.045), one seed, oracle fully informed and patient, goldens written
  by one author.
