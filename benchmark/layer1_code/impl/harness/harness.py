"""
Harness definitions for the cheap-model + harness experiments (Layer 1 implementation only).

A harness is everything the agent does between receiving the task prompt and returning
code, with ONE model. It is a (generator, decider) pair:

  generator: task -> list of candidate raw responses
  decider:   candidates -> index of the chosen candidate

Rules:
  * A harness may only use information a real user could compute without the golden
    answer. The golden position series is scorer-only and is never exposed here.
  * Every LLM call is counted so cost curves can be built later.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from benchmark.common.llm import BaseLLMClient
from benchmark.layer1_code.impl.scoring.systematic import SYSTEMATIC_SYSTEM_PROMPT


@dataclass
class Candidate:
    raw: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HarnessOutput:
    chosen: str
    candidates: List[Candidate]
    chosen_index: int
    llm_calls: int
    prompt_tokens: int
    completion_tokens: int
    trace: Dict[str, Any] = field(default_factory=dict)


ClientFactory = Callable[[float], BaseLLMClient]  # temperature -> client


class Harness:
    name: str = "base"

    def run(self, task_prompt: str, make_client: ClientFactory, context: Optional[Dict[str, Any]] = None) -> HarnessOutput:
        raise NotImplementedError


class InfraError(RuntimeError):
    """Gateway/network failure after client retries. Never scored as a model failure."""


def _gen(client: BaseLLMClient, prompt: str, system_prompt: str = "") -> str:
    try:
        return client.generate(prompt, system_prompt=system_prompt)
    except Exception as e:  # urllib / HTTP errors after the client's own retries
        raise InfraError(str(e)) from e


def _usage(client: BaseLLMClient) -> Dict[str, int]:
    u = getattr(client, "usage", None)
    if u is None:
        return {"calls": 1, "prompt_tokens": 0, "completion_tokens": 0}
    return {
        "calls": int(u.calls),
        "prompt_tokens": int(u.prompt_tokens),
        "completion_tokens": int(u.completion_tokens),
    }


class G0SingleShot(Harness):
    """G0 x D0: the task prompt goes in once, the first answer comes out. Temperature 0."""
    name = "G0_single_shot"

    def __init__(self, system_prompt: str = SYSTEMATIC_SYSTEM_PROMPT, temperature: float = 0.0):
        self.system_prompt = system_prompt
        self.temperature = temperature

    def run(self, task_prompt: str, make_client: ClientFactory, context: Optional[Dict[str, Any]] = None) -> HarnessOutput:
        client = make_client(self.temperature)
        raw = _gen(client, task_prompt, self.system_prompt)
        u = _usage(client)
        cand = Candidate(raw=raw, meta={"temperature": self.temperature})
        return HarnessOutput(
            chosen=raw,
            candidates=[cand],
            chosen_index=0,
            llm_calls=u["calls"],
            prompt_tokens=u["prompt_tokens"],
            completion_tokens=u["completion_tokens"],
        )


# ---------------------------------------------------------------------------
# G1: single call, richer instructions. The model must first enumerate every
# under-specified point in the spec and commit to a decision, then write code.
# G1 does NOT tell the model our house conventions; G1c does (a product-style
# system prompt), so the difference between them is pure convention alignment.
# ---------------------------------------------------------------------------
G1_INSTRUCTIONS = """
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
"""

HOUSE_CONVENTIONS = """
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
"""


class G1EnumerateAndDecide(Harness):
    name = "G1_enumerate_decide"

    def __init__(self, with_conventions: bool = False, temperature: float = 0.0):
        self.with_conventions = with_conventions
        self.temperature = temperature
        self.system_prompt = SYSTEMATIC_SYSTEM_PROMPT + (HOUSE_CONVENTIONS if with_conventions else "")

    def run(self, task_prompt: str, make_client: ClientFactory, context: Optional[Dict[str, Any]] = None) -> HarnessOutput:
        client = make_client(self.temperature)
        prompt = task_prompt + "\n" + G1_INSTRUCTIONS
        raw = _gen(client, prompt, self.system_prompt)
        u = _usage(client)
        return HarnessOutput(
            chosen=raw,
            candidates=[Candidate(raw=raw, meta={"with_conventions": self.with_conventions})],
            chosen_index=0,
            llm_calls=u["calls"],
            prompt_tokens=u["prompt_tokens"],
            completion_tokens=u["completion_tokens"],
        )


# ---------------------------------------------------------------------------
# G2: two calls. Call 1 turns the loose description into a pinned specification
# (no code). Call 2 implements that specification. The model never sees sandbox
# output or the golden; it only sees its own spec.
# ---------------------------------------------------------------------------
G2_SPEC_SYSTEM = """You are a quantitative analyst writing an implementation-ready specification
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

Be literal and complete; the engineer will implement exactly what you write and nothing else.
"""

G2_IMPL_SUFFIX = """

An analyst has already converted the specification above into the following implementation-ready
specification. Implement EXACTLY this specification (it resolves every ambiguity in the original text):

----- PINNED SPECIFICATION -----
{spec}
----- END -----

Write the complete Python function `build_strategy(df)` now.
"""


class G2SpecThenCode(Harness):
    name = "G2_spec_then_code"

    def __init__(self, temperature: float = 0.0):
        self.temperature = temperature

    def run(self, task_prompt: str, make_client: ClientFactory, context: Optional[Dict[str, Any]] = None) -> HarnessOutput:
        c1 = make_client(self.temperature)
        spec = _gen(c1, task_prompt + "\n\nWrite the implementation-ready specification now.", G2_SPEC_SYSTEM)
        c2 = make_client(self.temperature)
        raw = _gen(c2, task_prompt + G2_IMPL_SUFFIX.format(spec=spec), SYSTEMATIC_SYSTEM_PROMPT)
        u1, u2 = _usage(c1), _usage(c2)
        return HarnessOutput(
            chosen=raw,
            candidates=[Candidate(raw=raw, meta={"spec": spec})],
            chosen_index=0,
            llm_calls=u1["calls"] + u2["calls"],
            prompt_tokens=u1["prompt_tokens"] + u2["prompt_tokens"],
            completion_tokens=u1["completion_tokens"] + u2["completion_tokens"],
            trace={"spec": spec},
        )


# ---------------------------------------------------------------------------
# G3: ask-then-implement. Call 1: the model reads the loose description and writes
# the clarifying questions it would put to the client. The client is simulated by a
# FIXED model (part of the environment, not of the harness under test) that holds the
# pinned specification and answers only what is asked. Call 2: implement with the Q&A.
# The agent never sees the pinned spec or the golden; it sees only answers to its own
# questions, which a real client could equally have given.
# ---------------------------------------------------------------------------
import os

ORACLE_MODEL = os.environ.get("HARNESS_ORACLE_MODEL", "claude-sonnet-5-5")

G3_ASK_INSTRUCTIONS = """
Do NOT write code yet. You are about to implement this strategy for a client and you may ask
the client clarifying questions first. List every question whose answer would change the
position series: undefined terms or functions, indicator formula variants, warm-up length,
close vs high/low for each price test, cross vs level conditions, the order in which exit,
add and entry rules are checked on the same bar, whether thresholds are frozen at entry or
recomputed, how the described sizing maps to a target weight in [0, 1], and re-entry timing.
Ask at most {max_q} questions, numbered, one line each, most important first. Output only the questions.
"""

G3_ORACLE_SYSTEM = """You are the client who wrote a trading-strategy request. You hold the exact
implementation-ready specification of what you want (below). An engineer will ask you questions.
Rules:
- Answer each question directly from the specification, one short answer per question, numbered the same way.
- Answer ONLY what is asked. Do not volunteer other details, do not paste the specification, do not write code.
- If a question is already answered in the engineer's own original request, answer it anyway from the specification.
- If the specification does not settle a question, say "your choice" and nothing else.

----- SPECIFICATION (private) -----
{spec}
----- END -----
"""

G3_IMPL_SUFFIX = """

You asked the client the following questions and received these answers. Implement the strategy
so that it follows the original request AND every answer below exactly. Where the client said
"your choice", pick the simplest reading and state it in a comment.

----- QUESTIONS -----
{questions}
----- ANSWERS -----
{answers}
----- END -----

Write the complete Python function `build_strategy(df)` now.
"""


class G3AskThenImplement(Harness):
    name = "G3_ask_then_implement"

    def __init__(self, with_conventions: bool = False, max_questions: int = 12, temperature: float = 0.0):
        self.with_conventions = with_conventions
        self.max_questions = max_questions
        self.temperature = temperature
        self.system_prompt = SYSTEMATIC_SYSTEM_PROMPT + (HOUSE_CONVENTIONS if with_conventions else "")

    def run(self, task_prompt: str, make_client: ClientFactory, context: Optional[Dict[str, Any]] = None) -> HarnessOutput:
        spec = (context or {}).get("pinned_spec")
        if not spec:
            raise ValueError("G3 needs context['pinned_spec'] for the simulated client")
        c1 = make_client(self.temperature)
        questions = _gen(c1, task_prompt + "\n" + G3_ASK_INSTRUCTIONS.format(max_q=self.max_questions), self.system_prompt)
        # simulated client: fixed model, not the model under test
        from benchmark.common.llm import get_llm_client
        oracle = get_llm_client(model=ORACLE_MODEL, seed=0, temperature=0.0)
        answers = _gen(oracle, "Engineer's questions:\n" + questions, G3_ORACLE_SYSTEM.format(spec=spec))
        c2 = make_client(self.temperature)
        raw = _gen(c2, task_prompt + G3_IMPL_SUFFIX.format(questions=questions, answers=answers), self.system_prompt)
        u1, u2, uo = _usage(c1), _usage(c2), _usage(oracle)
        return HarnessOutput(
            chosen=raw,
            candidates=[Candidate(raw=raw, meta={"questions": questions, "answers": answers})],
            chosen_index=0,
            llm_calls=u1["calls"] + u2["calls"],
            prompt_tokens=u1["prompt_tokens"] + u2["prompt_tokens"],
            completion_tokens=u1["completion_tokens"] + u2["completion_tokens"],
            trace={"questions": questions, "answers": answers, "oracle_model": ORACLE_MODEL,
                   "oracle_prompt_tokens": uo["prompt_tokens"], "oracle_completion_tokens": uo["completion_tokens"]},
        )


# ---------------------------------------------------------------------------
# G3w: same as G3c, but the simulated client does NOT hold the pinned specification.
# It only has the same loose task text the implementer has, and is told to answer like
# a real trader using ordinary conventions, not a secret correct answer. This isolates
# how much of G3's gain was "the oracle leaking the pinned spec" vs "a second voice
# applying common trading conventions". Fully-informed G3/G3c/G3c_q* are the upper bound;
# G3w is the realistic lower bound for the "ask the client" idea.
# ---------------------------------------------------------------------------
G3W_ORACLE_SYSTEM = """You are the client who wrote the trading-strategy request below. You understand
your own trading idea (what market behavior you want to capture, your rough intent for risk and sizing)
but you are NOT a programmer and you never specified low-level implementation choices such as exact
indicator smoothing formulas, which bar a comparison is evaluated on, close vs high/low for a given test,
the exact order in which rules are checked, or precise warm-up length.
Rules:
- Answer each question directly, one short answer per question, numbered the same way.
- If the question is about something your original request already states or clearly implies
  (direction of entry/exit, which indicators, approximate thresholds, sizing), answer from your request.
- If the question is about a low-level implementation choice you never thought about, do NOT invent a
  precise answer. Say what you'd expect in plain trader language (e.g. "whichever is standard", "just
  use the closing price, I don't track intrabar highs") or say "your call, use whatever is simplest."
- Never claim precision you do not have. Do not paste or imply a formula you were not given.

----- YOUR ORIGINAL REQUEST -----
{spec}
----- END -----
"""


class G3WeakClient(Harness):
    """G3c with an uninformed, common-sense-only simulated client (no pinned spec)."""
    name = "G3_weak_client"

    def __init__(self, with_conventions: bool = True, max_questions: int = 12, temperature: float = 0.0):
        self.with_conventions = with_conventions
        self.max_questions = max_questions
        self.temperature = temperature
        self.system_prompt = SYSTEMATIC_SYSTEM_PROMPT + (HOUSE_CONVENTIONS if with_conventions else "")

    def run(self, task_prompt: str, make_client: ClientFactory, context: Optional[Dict[str, Any]] = None) -> HarnessOutput:
        c1 = make_client(self.temperature)
        questions = _gen(c1, task_prompt + "\n" + G3_ASK_INSTRUCTIONS.format(max_q=self.max_questions), self.system_prompt)
        from benchmark.common.llm import get_llm_client
        oracle = get_llm_client(model=ORACLE_MODEL, seed=0, temperature=0.0)
        # the weak client only gets the SAME loose text the implementer already had, not the pinned spec
        answers = _gen(oracle, "Engineer's questions:\n" + questions, G3W_ORACLE_SYSTEM.format(spec=task_prompt))
        c2 = make_client(self.temperature)
        raw = _gen(c2, task_prompt + G3_IMPL_SUFFIX.format(questions=questions, answers=answers), self.system_prompt)
        u1, u2, uo = _usage(c1), _usage(c2), _usage(oracle)
        return HarnessOutput(
            chosen=raw,
            candidates=[Candidate(raw=raw, meta={"questions": questions, "answers": answers})],
            chosen_index=0,
            llm_calls=u1["calls"] + u2["calls"],
            prompt_tokens=u1["prompt_tokens"] + u2["prompt_tokens"],
            completion_tokens=u1["completion_tokens"] + u2["completion_tokens"],
            trace={"questions": questions, "answers": answers, "oracle_model": ORACLE_MODEL,
                   "oracle_prompt_tokens": uo["prompt_tokens"], "oracle_completion_tokens": uo["completion_tokens"]},
        )


# ---------------------------------------------------------------------------
# G4: implementer + supervisor, same model playing both roles. No external information
# at all: all roles receive the same public environment and conventions; the supervisor
# also sees the original task and latest draft, never the pinned spec or golden. It can catch
# things the implementer could in principle have caught itself: an assumption that
# contradicts the original text, an internal inconsistency, an unresolved undefined term
# that got silently guessed. This tests whether a second look helps even with zero new
# information, as a contrast to G3 (which adds information from outside).
# ---------------------------------------------------------------------------
G4_DRAFT_INSTRUCTIONS = """
Before writing code, list every point where the specification is ambiguous or under-specified,
and state the decision you take for each in one line (same list as before: formula variants,
warm-up length, close/high/low, cross vs level, rule order, sizing mapping, re-entry timing).
Then output the ```python ... ``` block.
"""

G4_REVIEW_ROLE = """
For this call, act as the supervisor reviewing the supplied draft. This role overrides the
code-generation and code-only output instructions above: return only review concerns or exactly
APPROVED. The environment, interface contract and house conventions above are supplied facts shared
with the implementer, not guesses. Do not challenge those facts as missing from the original task.
"""

G4_SUPERVISOR_SYSTEM = """You are a second engineer reviewing a colleague's draft implementation before
it ships. You share the original task, public environment and conventions with the implementer.
Check only for concrete problems supported by those materials:
- Does the code contradict something explicitly stated in the original text?
- Did the colleague silently guess the meaning of an undefined term or function without flagging it?
- Is there an internal inconsistency (the stated assumptions disagree with what the code does)?
- Does the code violate the supplied interface contract or applicable house conventions?
For a genuinely unspecified point, accept an explicitly stated, consistent assumption unless it
contradicts the task or supplied conventions. Do not request a change just because you prefer another
reasonable interpretation. You have no execution results; do not claim to have run the code.
Do NOT invent new requirements. If you have no concrete concern,
say exactly "APPROVED" and nothing else. Otherwise list your concerns as short numbered points,
most important first; do not rewrite the code yourself.

----- ORIGINAL TASK TEXT -----
{task_prompt}
----- END -----

----- COLLEAGUE'S DRAFT (assumptions + code) -----
{draft}
----- END -----
"""

G4_REVISE_SUFFIX = """

A second engineer reviewed your draft against the same original task text (no extra information) and
raised the following. Edit the supplied draft to address concerns justified by the task, public
environment or conventions. Preserve correct and unrelated code and keep reasonable stated assumptions
where no supplied rule settles the matter. Reject unsupported review claims rather than implementing
them. Briefly state which concerns you addressed or rejected, then return the complete code block.

----- CURRENT DRAFT (assumptions + code) -----
{draft}
----- END -----

----- REVIEWER FEEDBACK -----
{feedback}
----- END -----

Write the complete, final Python function `build_strategy(df)` now.
"""


class G4ImplementerSupervisor(Harness):
    """One model in two roles, with at most two revisions and a review after each.

    All calls share the public contract. Private scoring context is deliberately unused.
    A final rejection is recorded, not treated as approval or a golden-based selection.
    """
    name = "G4_implementer_supervisor"
    version = "g4_v2"

    def __init__(self, with_conventions: bool = True, temperature: float = 0.0, max_revisions: int = 2):
        if max_revisions < 0:
            raise ValueError("max_revisions must be nonnegative")
        self.with_conventions = with_conventions
        self.temperature = temperature
        self.max_revisions = max_revisions
        self.system_prompt = SYSTEMATIC_SYSTEM_PROMPT + (HOUSE_CONVENTIONS if with_conventions else "")

    def run(self, task_prompt: str, make_client: ClientFactory, context: Optional[Dict[str, Any]] = None) -> HarnessOutput:
        c1 = make_client(self.temperature)
        draft = _gen(c1, task_prompt + "\n" + G4_DRAFT_INSTRUCTIONS, self.system_prompt)
        clients = [c1]
        raw = draft
        reviews = []
        revisions = 0
        previous_feedback = ""
        while True:
            reviewer = make_client(self.temperature)
            clients.append(reviewer)
            review_prompt = G4_SUPERVISOR_SYSTEM.format(task_prompt=task_prompt, draft=raw)
            if previous_feedback:
                review_prompt += "\nPrevious review concerns (check whether resolved):\n" + previous_feedback
            feedback = _gen(reviewer, review_prompt, self.system_prompt + G4_REVIEW_ROLE)
            approved = feedback.strip().upper() == "APPROVED"
            reviews.append({"draft": raw, "feedback": feedback, "approved": approved})
            if approved or revisions >= self.max_revisions:
                break
            implementer = make_client(self.temperature)
            clients.append(implementer)
            raw = _gen(implementer, task_prompt + G4_REVISE_SUFFIX.format(draft=raw, feedback=feedback), self.system_prompt)
            revisions += 1
            previous_feedback = feedback

        usage = [_usage(client) for client in clients]

        return HarnessOutput(
            chosen=raw,
            candidates=[Candidate(raw=raw, meta={"draft": draft, "feedback": feedback, "approved": approved})],
            chosen_index=0,
            llm_calls=sum(u["calls"] for u in usage),
            prompt_tokens=sum(u["prompt_tokens"] for u in usage),
            completion_tokens=sum(u["completion_tokens"] for u in usage),
            trace={"draft": draft, "feedback": feedback, "approved": approved,
                   "harness_version": self.version, "reviews": reviews, "revisions": revisions,
                   "max_revisions": self.max_revisions,
                   "stop_reason": "approved" if approved else "revision_limit"},
        )


HARNESSES: Dict[str, Callable[[], Harness]] = {
    "G3": lambda: G3AskThenImplement(with_conventions=False),
    "G3c": lambda: G3AskThenImplement(with_conventions=True),
    # question budget ablation: how well does the model prioritise what to ask?
    "G3c_q3": lambda: G3AskThenImplement(with_conventions=True, max_questions=3),
    "G3c_q5": lambda: G3AskThenImplement(with_conventions=True, max_questions=5),
    "G3w": lambda: G3WeakClient(with_conventions=True),
    "G4": lambda: G4ImplementerSupervisor(with_conventions=True),
    "G4_v2": lambda: G4ImplementerSupervisor(with_conventions=True),
    "G0": G0SingleShot,
    "G1": lambda: G1EnumerateAndDecide(with_conventions=False),
    "G1c": lambda: G1EnumerateAndDecide(with_conventions=True),
    "G2": G2SpecThenCode,
}


def get_harness(name: str) -> Harness:
    if name not in HARNESSES:
        raise ValueError(f"Unknown harness '{name}'. Known: {sorted(HARNESSES)}")
    return HARNESSES[name]()
