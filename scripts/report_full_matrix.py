#!/usr/bin/env python3
"""
Merge every stage directory of a full-matrix run and write REPORT.md + tables.

Usage: .venv/bin/python scripts/report_full_matrix.py docs/experiments/full_run_20261023
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# Model set as of 2026-11-04: haiku (no 5.5 tier on the gateway) and gpt-5.4-mini dropped
# at the user's request; gpt-6-astra added. CHEAP now has one member (deepseek-v4.1-flash);
# kept as its own list so the "cheap vs expensive" framing in the report still has meaning.
CHEAP = ["deepseek-v4.1-flash"]
EXPENSIVE = ["claude-sonnet-5.5", "claude-opus-5.5", "claude-fable-5.1", "gpt-6.1-sol", "gpt-6-astra", "deepseek-v4-pro"]
MODELS = CHEAP + EXPENSIVE
HARNESS_ORDER = ["G0", "G1", "G2", "G1c", "G3w", "G4_v2", "G3", "G3c_q3", "G3c_q5", "G3c"]
CALLS = {"G0": 1, "G1": 1, "G1c": 1, "G2": 2, "G3w": 2, "G4_v2": "2-6", "G3": 2, "G3c": 2, "G3c_q3": 2, "G3c_q5": 2}


def load(out: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(out.glob("*/runs.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                r["stage"] = f.parent.name
                trace = r.get("trace") or {}
                r["oracle_prompt_tokens"] = trace.get("oracle_prompt_tokens", 0)
                r["oracle_completion_tokens"] = trace.get("oracle_completion_tokens", 0)
                for k in ("chosen_raw", "candidates", "trace"):
                    r.pop(k, None)
                rows.append(r)
    df = pd.DataFrame(rows)
    df["infra_error"] = df.get("infra_error", False).fillna(False).astype(bool)
    return df


def short(m: str) -> str:
    return m.replace("claude-", "").replace("deepseek-", "ds-")


def main(out: Path) -> None:
    raw = load(out)
    infra = raw[raw.infra_error]
    df = raw[~raw.infra_error].copy()
    df["group"] = df.case_id.map(lambda c: "qcb22" if c.startswith("qcb_") else "canonical20")

    # 1. pinned wording
    pin = df[df.variant == "pinned"]
    t_pin = pin.pivot_table(index="model", columns="group", values="passed", aggfunc="mean").reindex(MODELS).round(2)

    # 2. original wording, harness x model
    org = df[df.variant == "original"]
    t_main = org.pivot_table(index="harness", columns="model", values="passed", aggfunc="mean")
    t_main = t_main.reindex([h for h in HARNESS_ORDER if h in t_main.index])[[m for m in MODELS if m in t_main.columns]].round(2)
    t_n = org.pivot_table(index="harness", columns="model", values="passed", aggfunc="count")

    # 3. cost: tokens per task, model-under-test only, and oracle separately
    org = org.assign(tok=org.prompt_tokens + org.completion_tokens)
    t_tok = org.pivot_table(index="harness", columns="model", values="tok", aggfunc="mean")
    t_tok = t_tok.reindex(t_main.index)[t_main.columns].round(0)
    t_oracle = org[org.harness.str.startswith("G3")].groupby("harness")[["oracle_prompt_tokens", "oracle_completion_tokens"]].mean().round(0)

    # 4. headline comparisons
    best_exp_g0 = t_main.loc["G0", [m for m in EXPENSIVE if m in t_main.columns]].max()
    lines = []
    for h in t_main.index:
        for m in CHEAP:
            if m in t_main.columns and t_main.loc[h, m] > best_exp_g0:
                lines.append(f"- {short(m)} + {h}: {t_main.loc[h, m]:.2f} > best expensive single shot {best_exp_g0:.2f}")

    # 5. per-task difficulty on original wording (fraction of model x harness cells passed)
    t_task = org.pivot_table(index="case_id", columns="harness", values="passed", aggfunc="mean")
    t_task = t_task[[h for h in HARNESS_ORDER if h in t_task.columns]].round(2)
    t_task["all"] = org.groupby("case_id").passed.mean().round(2)
    t_task = t_task.sort_values("all")

    # 6. hygiene
    trunc = df[df.get("finish_reason").eq("length")] if "finish_reason" in df.columns else df.iloc[0:0]
    t_trunc = trunc.groupby(["model", "harness"]).size().rename("n").reset_index() if len(trunc) else None

    def md(t: pd.DataFrame) -> str:
        t = t.copy()
        t.columns = [short(c) if isinstance(c, str) else c for c in t.columns]
        if t.index.name == "model" or t.index.dtype == object:
            t.index = [short(i) if isinstance(i, str) else i for i in t.index]
        return t.to_markdown()

    rep = f"""# Full run: cheap model + harness vs expensive model (Layer 1 implementation)

Run directory: `{out}`. Rows scored: {len(df)}. Infrastructure failures after 4 attempts (not scored): {len(infra)}.
Responses cut off by the output cap (finish_reason = length): {len(trunc)}.

Scoring: bar-by-bar position match against the golden on 1500 daily bars (active_action_match >= 0.95,
trade_f1 >= 0.90, friction match, no look-ahead, no NaN after warm-up, bounds). Temperature 0, one pass.

## 1. Pinned wording, single shot (G0)

{md(t_pin)}

## 2. Original wording, pass rate (22 tasks; one task = 0.045)

{md(t_main)}

Cells with fewer than 22 scored tasks: {int((t_n < 22).sum().sum())}.

## 3. Tokens per task for the model under test (prompt + completion)

{md(t_tok)}

Simulated client (claude-sonnet-5-5) tokens per task, G3 family only:

{t_oracle.to_markdown()}

## 4. Cheap + harness above the best expensive single shot

{chr(10).join(lines) if lines else "- none"}

## 5. Task difficulty on original wording (share of model cells passed)

{t_task.to_markdown()}
"""
    if t_trunc is not None:
        rep += "\n## 6. Truncated responses\n\n" + t_trunc.to_markdown(index=False) + "\n"
    if len(infra):
        rep += "\n## 7. Infrastructure failures (not scored)\n\n" + infra.groupby(["model", "harness"]).size().rename("n").reset_index().to_markdown(index=False) + "\n"

    (out / "REPORT.md").write_text(rep, encoding="utf-8")
    df.to_csv(out / "all_rows.csv", index=False)
    t_main.to_csv(out / "pass_rate_original.csv")
    print(rep)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
