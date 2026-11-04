# 八种 harness 逐一说明 + 结果

本文档把上一份 `REPORT.md` 里的每一行数字还原成"这一步到底让模型做了什么"。顺序是先讲共享的实验设置，再逐个 harness 过一遍配置和该 harness 下八个模型的横向对比，最后把八个 harness 放在一起纵向比较。

## 0. 所有实验共享的设置

- **题目**：22 道从 QuantCode-Bench 移植的策略题（`benchmark/layer1_code/impl/tasks/qcb_cases.py`）。每道题有两份文本：**钉死版**（指标公式、价格判断用 close 还是 high/low、预热期根数、规则检查顺序全部写明）和**原帖版**（QuantCode 原始的英文描述，留着真实用户常见的留白和未定义术语）。两份文本共用同一份人工写的 golden 实现。本文档里的 8 个 harness 全部用**原帖版**跑，因为这是翻译成本真正体现的地方；钉死版只在最前面跑了一次单发作对照（全模型 0.91 到 1.00，基本饱和）。
- **数据**：某只 ETF 最近 1500 根日线（默认 SPY，两道题换成 TLT 和 IWM 让交易次数够多）。
- **评分**：模型写的 `build_strategy(df)` 在数据上跑出仓位序列，逐根 bar 和 golden 的仓位序列比。通过线是仓位匹配度 ≥ 0.95、买卖时点的 F1 ≥ 0.90、摩擦成本参数一致、无未来数据、无 NaN、仓位不越界。全部满足才算这道题通过。
- **模型**：8 个，分两组。

  | 分组 | 模型 |
  |---|---|
  | 便宜 | claude-haiku-5-5、gpt-5.4-mini、deepseek-v4.1-flash |
  | 贵 | claude-sonnet-5-5、claude-opus-5-5、claude-fable-5.1、gpt-6.1-sol、deepseek-v4-pro |

- **温度**：全部调用温度为 0（gpt-6 系列网关强制 1.0，代码里有特判），每个格子只跑一次，不是多次取平均。
- **harness 之间唯一的变量**：被测模型拿到题目之后、交出代码之前的流程。题目、数据、评分代码在 8 个 harness 之间完全不变。
- **harness 能看到什么，不能看到什么**：被测模型从头到尾看不到钉死版文本、golden 仓位序列、或者评分结果。它能看到的只有原帖文本，以及它自己在流程里主动要来的信息（比如下面 G3 系列里"客户"的回答）。

下面每个 harness 先说流程和用到的原文提示词，再给结果表。表里的列含义：

- **通过率**：22 题里通过的比例。
- **能跑通率**（exec_pass）：代码能执行、不报错、能提取出仓位序列的比例，不要求仓位对得上。
- **仓位匹配**（am = active_action_match）：和 golden 仓位序列的平均匹配度，即使没达到 0.95 的通过线也算进来，所以它比通过率更能看出"差多少"而不是"过没过"。
- **调用次数**：被测模型被调用几次（不含下面 G3 系列里扮演客户的那次调用）。
- **输出 token（均值）**：每道题被测模型生成的 token 数均值，衡量这个 harness 的运行成本。

---

## 1. G0：裸单发

**流程**：一次调用。系统提示固定为下面这段（后面所有 harness 都在这段基础上加东西，这是共同的底座），用户消息就是题目原文，模型直接输出代码。

```
You are a quantitative strategy developer.
Translate the user's natural-language strategy specification into a causal Python function named `build_strategy(df)`.

Environment & Available Imports (already imported or importable from `src.signals`):
- `pandas as pd`, `numpy as np`
- `from src.signals import compute_moving_average_signal, compute_macd_signal, ...`

Input `df` columns:
- 'open', 'high', 'low', 'close', 'volume', 'return', 'rolling_vol', 'regime'
- Indexed by pd.DatetimeIndex sorted ascending.

Contract for build_strategy(df):
- Must return a dict: {"signals": pd.Series, "slippage_rate": float, "commission_rate": float, "rebalance_tolerance": float}
- IMPORTANT: Do NOT call .shift(1) on signals (Backtest.run() already shifts by 1 bar).
- Output ONLY a single ```python ... ``` code block.
```

这是基线,没有任何额外提示、没有二次调用、没有外部信息。

**结果（22 题，原帖版）**

| 模型 | 通过率 | 能跑通率 | 仓位匹配 | 输出 token |
|---|---|---|---|---|
| claude-haiku-5-5 | 0.36 | 1.00 | 0.54 | 2398 |
| gpt-5.4-mini | 0.32 | 1.00 | 0.46 | 643 |
| deepseek-v4.1-flash | 0.27 | 1.00 | 0.54 | 7188 |
| claude-sonnet-5-5 | 0.32 | 1.00 | 0.59 | 1300 |
| claude-opus-5-5 | 0.32 | 1.00 | 0.59 | 1386 |
| claude-fable-5.1 | 0.32 | 1.00 | 0.54 | 1296 |
| gpt-6.1-sol | 0.32 | 0.95 | 0.54 | 867 |
| deepseek-v4-pro | 0.32 | 0.95 | 0.53 | 4877 |

八个模型全部挤在 0.27 到 0.36 之间,几乎所有模型都卡在 0.32。haiku 反而最高,贵模型没有体现出优势。代码基本都能跑(能跑通率接近或等于 1.00),卡住的是仓位对不上,也就是模型按自己的理解填了留白,填的和 golden 不一致。

---

## 2. G1:先列留白再写码

**流程**:一次调用,但在题目原文后面多拼一段指令,要求模型在同一次回复里先列出规格里每一处没说清的地方并逐条写下自己的决定,再把策略写成逐 bar 的状态机,最后才输出代码。系统提示和 G0 相同,不包含任何约定。

拼在题目后面的指令原文:

```
Before writing code, do the following inside the same reply:

1. List every point where the specification is ambiguous or under-specified (indicator
   formula variants, warm-up length, whether a test uses close/high/low, cross vs level
   conditions, order in which exit and entry/add rules are checked on the same bar, whether
   thresholds are frozen at entry or recomputed, how position size maps to a weight in
   [0, 1], re-entry timing). For each one, state the decision you take in one line.
2. Write the strategy as an explicit per-bar state machine: state variables, and the ordered
   list of rules evaluated on each bar.
3. Then output the final ```python ... ``` block implementing exactly those decisions.
   Put the assumptions as a comment block at the top of the function.
```

关键是这一步**没有告诉模型我们的约定是什么**,只是让它自己想、自己拿主意。目的是看"让模型想得更细"这件事本身有没有用,和下面 G1c(同样的流程,但告诉它约定)做对照。

**结果(22 题,原帖版)**

| 模型 | 通过率 | 能跑通率 | 仓位匹配 | 输出 token |
|---|---|---|---|---|
| claude-haiku-5-5 | 0.36 | 1.00 | 0.66 | 4605 |
| gpt-5.4-mini | 0.36 | 0.91 | 0.56 | 1293 |
| deepseek-v4.1-flash | 0.23 | 1.00 | 0.53 | 10154 |
| claude-sonnet-5-5 | 0.27 | 0.86 | 0.51 | 3367 |
| claude-opus-5-5 | 0.41 | 0.95 | 0.60 | 2829 |
| claude-fable-5.1 | 0.23 | 1.00 | 0.62 | 4143 |
| gpt-6.1-sol | 0.32 | 1.00 | 0.57 | 1592 |
| deepseek-v4-pro | 0.23 | 0.95 | 0.44 | 6355 |

和 G0 比几乎没有系统性提升,八个模型里有三个(sonnet、fable、ds-pro)反而比 G0 低。token 用量涨了一到三倍。"让模型自己想清楚哪里有歧义"不会让它猜中我们这边默认的填法,留白终究是靠约定定下来的,不是靠推理推出来的。

---

## 3. G1c:在 G1 基础上告诉模型我们的约定

**流程**:和 G1 完全一样的指令,但系统提示里多加一段"桌面约定",把我们这边默认的填法直接写明。

加在系统提示末尾的约定原文:

```
House conventions for this desk (apply unless the specification says otherwise):
- Decisions are made on the close of bar t; signals[t] is the target weight at the end of bar t.
- When holding, exit rules are checked before any add/entry rule; a bar on which the position
  is closed is flat, and re-entry is possible from the next bar at the earliest.
- All price tests use close unless the spec names high/low explicitly.
- SMA(n) = rolling(n).mean(); EMA(n) = ewm(span=n, adjust=False).mean();
  RSI(n) and ATR(n) use Wilder smoothing ewm(alpha=1/n, adjust=False); the first TR is high-low;
  Bollinger std uses pandas default ddof=1; MACD = (12, 26, 9), histogram = line - signal.
- Position sizes given as a fraction of capital or a fixed number of shares map to weight 1.0;
  partial/multi-unit positions map to equal fractions of 1.0.
- Stay flat during warm-up (while any required indicator is undefined).
- Use default friction (slippage 0.0005, commission 0.0005, rebalance tolerance 0.05).
```

这段话不含任何题目信息,是通用的"我们这边的默认习惯"。调用次数和 G0 一样是 1 次。

**结果(22 题,原帖版)**

| 模型 | 通过率 | 能跑通率 | 仓位匹配 | 输出 token |
|---|---|---|---|---|
| claude-haiku-5-5 | 0.68 | 1.00 | 0.78 | 4892 |
| gpt-5.4-mini | 0.41 | 1.00 | 0.57 | 1191 |
| deepseek-v4.1-flash | 0.45 | 1.00 | 0.64 | 9375 |
| claude-sonnet-5-5 | 0.73 | 0.95 | 0.84 | 3482 |
| claude-opus-5-5 | 0.73 | 1.00 | 0.88 | 2658 |
| claude-fable-5.1 | 0.55 | 0.95 | 0.74 | 3302 |
| gpt-6.1-sol | 0.55 | 0.91 | 0.68 | 1245 |
| deepseek-v4-pro | 0.36 | 0.95 | 0.49 | 6709 |

这是第一个明显起作用的 harness。每个模型都比自己的 G0 高,sonnet 和 opus 涨了 0.4,haiku 涨了 0.3。只用一次调用、不额外问任何人、不碰 golden,单靠把约定写清楚就把通过率平均拉高了十几到四十个百分点。说明 G0 和 G1 里很大一部分失分不是"模型不会写"，而是"模型不知道我们这边的默认习惯"。

---

## 4. G2:先出规格,再照规格写码(两次调用)

**流程**:两次独立调用,用两个独立的系统提示。

第一次调用,系统提示要求模型**只写规格,不写代码**:

```
You are a quantitative analyst writing an implementation-ready specification
for a backtest engineer. You do NOT write code. Output a specification with these sections:

1. Data and warm-up: which columns are used; the first bar index on which trading may occur.
2. Indicators: each indicator with its exact formula (smoothing type, parameters, ddof, etc.).
3. State variables: every variable carried across bars, with its initial value.
4. Per-bar procedure: an ordered list of steps evaluated on each bar t using only data up to t.
   Make every comparison explicit (<, <=, >, >=), say whether each price test uses close/high/low,
   whether thresholds are frozen at entry or recomputed, and what happens on the same bar when
   several rules fire.
5. Position mapping: how the described sizing maps to a target weight in [0, 1] per state.
6. Ambiguities: list each under-specified point in the original text and the decision taken.
```

第二次调用,把第一次产出的规格原样塞进第二条提示的结尾,系统提示换回 G0 的那段,要求模型"严格按这份规格实现,不要再自己判断":

```
An analyst has already converted the specification above into the following implementation-ready
specification. Implement EXACTLY this specification (it resolves every ambiguity in the original text):
----- PINNED SPECIFICATION -----
{spec}
----- END -----
Write the complete Python function `build_strategy(df)` now.
```

两次调用都是**同一个被测模型**,没有外部信息进来,第二次看到的规格就是第一次自己写的。

**结果(22 题,原帖版)**

| 模型 | 通过率 | 能跑通率 | 仓位匹配 | 输出 token |
|---|---|---|---|---|
| claude-haiku-5-5 | 0.27 | 0.95 | 0.61 | 8732 |
| gpt-5.4-mini | 0.32 | 1.00 | 0.54 | 3417 |
| deepseek-v4.1-flash | 0.32 | 1.00 | 0.60 | 21247 |
| claude-sonnet-5-5 | 0.41 | 0.95 | 0.64 | 5189 |
| claude-opus-5-5 | 0.36 | 1.00 | 0.63 | 5043 |
| claude-fable-5.1 | 0.36 | 1.00 | 0.67 | 6182 |
| gpt-6.1-sol | 0.23 | 0.91 | 0.54 | 3335 |
| deepseek-v4-pro | 0.32 | 0.86 | 0.53 | 12065 |

跟 G0 比基本是原地踏步,haiku 和 gpt-6.1-sol 甚至略降,代价是 token 翻了三到四倍,因为两次调用都要生成完整内容。原因是第一次调用里模型自己给自己留白填的坑,第二次调用一样会踩进去,等于把同一个理解错误多走了一遍流程,没有引入任何新信息去纠正它。

---

## 5. G3:先向客户提问,再写码(两次调用 + 一个固定的"客户")

**流程**:这是第一个引入**题目之外的新信息**的 harness,也是本轮效果最大的一步。

第一次调用,被测模型只写提问,不写代码。拼在题目后面的指令:

```
Do NOT write code yet. You are about to implement this strategy for a client and you may ask
the client clarifying questions first. List every question whose answer would change the
position series: undefined terms or functions, indicator formula variants, warm-up length,
close vs high/low for each price test, cross vs level conditions, the order in which exit,
add and entry rules are checked on the same bar, whether thresholds are frozen at entry or
recomputed, how the described sizing maps to a target weight in [0, 1], and re-entry timing.
Ask at most 12 questions, numbered, one line each, most important first. Output only the questions.
```

这些问题交给一个**固定不变的"客户"模型**(全程用 claude-sonnet-5-5,和被测模型是谁无关)。客户的系统提示里私藏着这道题的钉死版规格,规则是只答被问到的,不主动补充,规格里没写的就说"你定":

```
You are the client who wrote a trading-strategy request. You hold the exact
implementation-ready specification of what you want (below). An engineer will ask you questions.
Rules:
- Answer each question directly from the specification, one short answer per question, numbered the same way.
- Answer ONLY what is asked. Do not volunteer other details, do not paste the specification, do not write code.
- If a question is already answered in the engineer's own original request, answer it anyway from the specification.
- If the specification does not settle a question, say "your choice" and nothing else.
----- SPECIFICATION (private) -----
{钉死版规格}
----- END -----
```

客户回答之后,第二次调用回到被测模型,把问答原样拼进提示,要求照着实现:

```
You asked the client the following questions and received these answers. Implement the strategy
so that it follows the original request AND every answer below exactly. Where the client said
"your choice", pick the simplest reading and state it in a comment.
----- QUESTIONS -----
{questions}
----- ANSWERS -----
{answers}
----- END -----
Write the complete Python function `build_strategy(df)` now.
```

被测模型**从未直接看到钉死版规格**,它看到的只是自己提的问题对应的答案。客户用的 token 不计入下表(单独统计:平均每题提示 1500 token、回答 507 token)。这一版的系统提示不含桌面约定,问题数上限 12 个。

**结果(22 题,原帖版)**

| 模型 | 通过率 | 能跑通率 | 仓位匹配 | 输出 token |
|---|---|---|---|---|
| claude-haiku-5-5 | 0.95 | 1.00 | 1.00 | 4005 |
| gpt-5.4-mini | 0.86 | 1.00 | 0.97 | 1125 |
| deepseek-v4.1-flash | 0.95 | 1.00 | 1.00 | 12306 |
| claude-sonnet-5-5 | 1.00 | 1.00 | 1.00 | 1915 |
| claude-opus-5-5 | 0.91 | 1.00 | 1.00 | 2022 |
| claude-fable-5.1 | 0.95 | 1.00 | 1.00 | 2524 |
| gpt-6.1-sol | 0.77 | 1.00 | 0.87 | 1488 |
| deepseek-v4-pro | 0.64 | 0.91 | 0.84 | 7071 |

这一步把大部分模型从 0.3 左右拉到了 0.8 以上,haiku 和 deepseek-v4.1-flash 这两个便宜模型追到了 0.95,跟 sonnet(1.00)、opus(0.91)、fable(0.95)已经分不出高下。gpt-6.1-sol(0.77)和 deepseek-v4-pro(0.64)明显落后,它们的失败代码大多能跑,错在逻辑上,不是没问清楚。

---

## 6. G3c:G3 加上桌面约定

**流程**:和 G3 一模一样,只是被测模型两次调用用的系统提示都换成"G0 的系统提示 + G1c 那段桌面约定"。问题上限仍是 12 个。想法是看"问客户"和"知道我们默认习惯"这两件事加在一起是不是比单独任何一个都好。

**结果(22 题,原帖版)**

| 模型 | 通过率 | 能跑通率 | 仓位匹配 | 输出 token |
|---|---|---|---|---|
| claude-haiku-5-5 | 0.86 | 1.00 | 0.89 | 4287 |
| gpt-5.4-mini | 0.77 | 0.91 | 0.89 | 1083 |
| deepseek-v4.1-flash | 0.95 | 1.00 | 0.99 | 10816 |
| claude-sonnet-5-5 | 0.91 | 0.95 | 0.96 | 1940 |
| claude-opus-5-5 | 0.95 | 1.00 | 1.00 | 2030 |
| claude-fable-5.1 | 0.95 | 1.00 | 1.00 | 2524 |
| gpt-6.1-sol | 0.86 | 1.00 | 0.93 | 1265 |
| deepseek-v4-pro | 0.64 | 0.95 | 0.86 | 7561 |

跟不加约定的 G3 比,互有高低(haiku 从 0.95 降到 0.86,gpt-6.1-sol 从 0.77 升到 0.86),整体没有看出谁系统性地更好。这和后面"运行两次测噪声"量出来的结果一致:同一个格子重复跑,通过率平均能差 0.07、最多差 0.18,G3 和 G3c 之间的这种涨跌大概率是噪声,不是真实差别。

---

## 7 和 8. G3c_q3、G3c_q5:限制只能问 3 个 / 5 个问题

**流程**:和 G3c 完全一样(两次调用,带约定,客户设定相同),唯一的差别是第一次调用里"最多问 12 个"换成"最多问 3 个"或"最多问 5 个"。目的是看模型自己能不能挑出最要紧的几个问题,而不是靠问满 12 个把钉死版规格的内容基本搬一遍过来。

限到 3 个和 5 个之后,客户平均回答的字数确实变短了(12 个问题时客户平均答 1465 token,限 3 个时降到 1094 token,限 5 个时 1187 token),说明问题数的限制确实生效了,模型没有靠简写问题绕开限制。

**G3c_q3 结果(22 题,原帖版)**

| 模型 | 通过率 | 能跑通率 | 仓位匹配 | 输出 token |
|---|---|---|---|---|
| claude-haiku-5-5 | 0.91 | 1.00 | 0.96 | 3806 |
| gpt-5.4-mini | 0.82 | 1.00 | 0.91 | 788 |
| deepseek-v4.1-flash | 0.82 | 1.00 | 0.90 | 7579 |
| claude-sonnet-5-5 | 0.95 | 1.00 | 1.00 | 1532 |
| claude-opus-5-5 | 0.86 | 1.00 | 0.98 | 1818 |
| claude-fable-5.1 | 1.00 | 1.00 | 1.00 | 1823 |
| gpt-6.1-sol | 0.77 | 1.00 | 0.87 | 1215 |
| deepseek-v4-pro | 0.59 | 1.00 | 0.85 | 5947 |

**G3c_q5 结果(22 题,原帖版)**

| 模型 | 通过率 | 能跑通率 | 仓位匹配 | 输出 token |
|---|---|---|---|---|
| claude-haiku-5-5 | 0.91 | 1.00 | 0.96 | 4181 |
| gpt-5.4-mini | 0.77 | 0.95 | 0.93 | 898 |
| deepseek-v4.1-flash | 0.86 | 1.00 | 0.97 | 11523 |
| claude-sonnet-5-5 | 0.95 | 1.00 | 1.00 | 1656 |
| claude-opus-5-5 | 0.91 | 1.00 | 1.00 | 1889 |
| claude-fable-5.1 | 0.95 | 1.00 | 1.00 | 2388 |
| gpt-6.1-sol | 0.91 | 1.00 | 0.99 | 1209 |
| deepseek-v4-pro | 0.64 | 0.96 | 0.81 | 6215 |

只问 3 个问题就已经拿到了问 12 个时大部分的提升(fable 甚至在只问 3 个时拿到了全组最高的 1.00),说明每道题真正卡脖子的留白数量不多,前几个问题就能问到点上。问到 5 个之后已经很接近问 12 个的效果。这组实验量的是"模型会不会抓重点",答案是:大部分模型会,deepseek-v4-pro 不太会(限 3 个和限 12 个它都是 0.6 左右,没怎么受益于多问)。

---

## 9. 纵向对比:把八个 harness 放在一起看

下表是 22 题通过率,按 harness 分行、模型分列,调用次数标在最左列。

| harness | 调用 | haiku-5-5 | gpt-5.4-mini | ds-v4.1-flash | sonnet-5-5 | opus-5-5 | fable-5.1 | gpt-6.1-sol | ds-v4-pro |
|---|---|---|---|---|---|---|---|---|---|
| G0 裸单发 | 1 | 0.36 | 0.32 | 0.27 | 0.32 | 0.32 | 0.32 | 0.32 | 0.32 |
| G1 自己想留白 | 1 | 0.36 | 0.36 | 0.23 | 0.27 | 0.41 | 0.23 | 0.32 | 0.23 |
| G2 先出规格再写码 | 2 | 0.27 | 0.32 | 0.32 | 0.41 | 0.36 | 0.36 | 0.23 | 0.32 |
| G1c 告知约定 | 1 | 0.68 | 0.41 | 0.45 | 0.73 | 0.73 | 0.55 | 0.55 | 0.36 |
| G3 先问客户 | 2 | 0.95 | 0.86 | 0.95 | 1.00 | 0.91 | 0.95 | 0.77 | 0.64 |
| G3c_q3 问 3 个 | 2 | 0.91 | 0.82 | 0.82 | 0.95 | 0.86 | 1.00 | 0.77 | 0.59 |
| G3c_q5 问 5 个 | 2 | 0.91 | 0.77 | 0.86 | 0.95 | 0.91 | 0.95 | 0.91 | 0.64 |
| G3c 问 12 个+约定 | 2 | 0.86 | 0.77 | 0.95 | 0.91 | 0.95 | 0.95 | 0.86 | 0.64 |

按这张表能看出三层递进:

1. **G0 到 G1、G2**:几乎没变化。单纯让模型"想更久"或者"先写规格"不解决留白问题,因为模型手里的信息量没有增加,它还是在猜。
2. **G1c**:信息量第一次真正增加,但增加的是"我们这边的通用习惯",不是这道题特有的信息。平均提升 0.1 到 0.4,便宜模型和贵模型都有感。
3. **G3 系列**:信息量再次增加,这次是"这道题特有的留白"的答案,来自一个固定的、手握正确答案的"客户"。这一步把六个模型(两个例外是 gpt-6.1-sol 和 deepseek-v4-pro)都推到了 0.77 到 1.00 的区间,便宜模型和贵模型基本落在同一个带里。

**关于噪声**:同样的设置在 2026-10-22 跑过一次,这次(10-23)又完整重跑了一遍。22 个重叠的格子里,通过率平均相差 0.07,最大相差 0.18。所以上表里差距小于 0.15 的两个格子不能认定谁真的更好,比如 G3 和 G3c 之间、问 3 个和问 12 个之间的来回涨跌大概率是噪声。能稳定复现的结论只有两条:加约定(G1c)有用,问客户(G3 系列)比加约定更有用。

**两个掉队的模型**:deepseek-v4-pro 在所有 harness 下都明显低于其他模型,它的失败代码基本都能跑,错在逻辑,不是没弄清楚留白。gpt-6.1-sol 在 G3(0.77)下也偏低,但在 G3c_q5(0.91)下又追平了主流,这个波动本身也可能是噪声,需要更多次重复才能分清楚是模型本身的问题还是运气。

**对核心问题的回答**:便宜模型(haiku-5-5、deepseek-v4.1-flash)只要配上"先问清楚"这个 harness,通过率能到 0.95,追平乃至反超大部分贵模型的同款 harness 成绩,远远超过任何贵模型的裸单发成绩(最高 0.32)。harness 带来的提升比换模型大一个量级。

---

## 几个要记住的限制

- 客户模型手握正确答案,有问必答,真实客户做不到这么配合,G3 系列的分数要当上限看,不是下限。
- 22 道题,每道权重 4.5 个百分点,单个格子的数字很粗,前面说的噪声范围(±0.07 到 0.18)比这还重要,看趋势不要看单元格的小数点后两位。
- 评分用的 golden 全部由同一个人写,对留白的默认填法带着这个人的习惯。
- 温度 0、单次运行,不是多次取平均后的结果。
