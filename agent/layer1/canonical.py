from __future__ import annotations

import inspect
import json
from typing import List
import pandas as pd

from agent.llm import BaseLLMClient, LLMUsage
from agent.layer1.cases.bad_alpha_cases import BAD_ALPHA_CASES
from agent.layer1.cases.systematic_cases import SYSTEMATIC_CASES
from agent.layer1.v2_hard import (
    SUBTLE_AUDIT_CASES,
    SubtleAuditCase,
    SubtleAuditEvalResult,
    HighFreedomImplCase,
    HighFreedomEvalResult,
    score_subtle_audit_case,
    evaluate_high_freedom_case,
)
from agent.layer1.v3_hard_impl import V3_HARD_IMPL_CASES


def _build_canonical_audit_cases() -> List[SubtleAuditCase]:
    """
    Consolidated 20-Case High-Discriminative Audit Suite (Identify Bad Alpha):
      - 10 Subtle Cross-Module / Implicit-Default Traps (`SUBTLE_AUDIT_CASES` flawed)
      - 6 Classic & Compound Strategy Traps (`ba_friction_02`, `ba_logic_02`, `ba_snooping_02`,
        `ba_snooping_03`, `ba_compound_01`, `ba_compound_02`)
      - 4 Clean Negative Controls (`ba_clean_01`, `ba_clean_03`, `sa_clean_subtle_11`, `sa_clean_subtle_12`)
    """
    by_ba = {c.case_id: c for c in BAD_ALPHA_CASES}
    by_sa = {c.case_id: c for c in SUBTLE_AUDIT_CASES}

    selected_sa_flawed = [
        by_sa["sa_nan_lock_01"],
        by_sa["sa_resample_left_02"],
        by_sa["sa_double_shift_03"],
        by_sa["sa_ewm_warmup_04"],
        by_sa["sa_trailing_reanchor_05"],
        by_sa["sa_sizing_chain_uncapped_06"],
        by_sa["sa_groupby_transform_leak_07"],
        by_sa["sa_rebalance_tol_drift_08"],
        by_sa["sa_silent_zero_merge_09"],
        by_sa["sa_sell_overshoot_10"],
    ]

    # Convert the 6 highest-signal classic/compound BadAlphaCases into SubtleAuditCase with mechanism keywords
    classic_specs = [
        ("ba_friction_02", (("rebalance_tolerance", "tolerance", "0.0", "daily", "turnover", "15bps", "30bps", "摩擦", "容忍"),)),
        ("ba_logic_02", (("rolling_vol", "target_vol", "invert", "opposite", "increase", "反", "颠倒"),)),
        ("ba_snooping_02", (("2020", "2022", "date", "calendar", "hardcode", "covid", "日期", "硬编码"),)),
        ("ba_snooping_03", (("test_df", "test", "hold-out", "holdout", "select", "测试集"),)),
        ("ba_compound_01", (("shift(-1)", "tomorrow", "next", "lookahead", "未来", "前视"), ("0.001", "rebalance_tolerance", "turnover", "friction", "cost", "摩擦", "成本"))),
        ("ba_compound_02", (("quantile", "mean()", "std()", "full", "entire", "全样本"), ("2020-03", "march", "0.83", "-0.314", "1.271", "magic", "thursday"))),
    ]
    converted_ba_flawed = []
    for cid, kw_groups in classic_specs:
        c = by_ba[cid]
        converted_ba_flawed.append(
            SubtleAuditCase(
                case_id=c.case_id,
                title=c.title,
                description=c.description,
                code=c.code,
                has_flaw=True,
                flaw_types=c.flaw_types,
                required_mechanism_keywords=kw_groups,
                reference_notes=c.reference_notes,
            )
        )

    clean_controls = [
        SubtleAuditCase(
            case_id=by_ba["ba_clean_01"].case_id,
            title=by_ba["ba_clean_01"].title,
            description=by_ba["ba_clean_01"].description,
            code=by_ba["ba_clean_01"].code,
            has_flaw=False,
            flaw_types=(),
            required_mechanism_keywords=(),
            reference_notes=by_ba["ba_clean_01"].reference_notes,
        ),
        SubtleAuditCase(
            case_id=by_ba["ba_clean_03"].case_id,
            title=by_ba["ba_clean_03"].title,
            description=by_ba["ba_clean_03"].description,
            code=by_ba["ba_clean_03"].code,
            has_flaw=False,
            flaw_types=(),
            required_mechanism_keywords=(),
            reference_notes=by_ba["ba_clean_03"].reference_notes,
        ),
        by_sa["sa_clean_subtle_11"],
        by_sa["sa_clean_subtle_12"],
    ]

    return selected_sa_flawed + converted_ba_flawed + clean_controls


def _build_canonical_impl_cases() -> List[HighFreedomImplCase]:
    """
    Consolidated 20-Case High-Discriminative Implementation Suite (Systematic Implementation):
      - Track A (6 cases): Factor Library API Contract & Friction Integration
        (`sys_mom_01`, `sys_rsi_trd_06`, `sys_combo_all_07`, `sys_combo_mean_08`, `sys_regime_vol_11`, `sys_fric_zero_14`)
      - Track B (14 cases): Unhinted Colloquial 7 Real-World Coding Failure Zones (`V3_HARD_IMPL_CASES`)
    """
    by_sys = {c.case_id: c for c in SYSTEMATIC_CASES}
    track_a_ids = [
        ("sys_mom_01", 1.0),
        ("sys_rsi_trd_06", 1.0),
        ("sys_combo_all_07", 1.0),
        ("sys_combo_mean_08", 1.0),
        ("sys_regime_vol_11", 2.0),
        ("sys_fric_zero_14", 1.0),
    ]
    track_a: List[HighFreedomImplCase] = []
    for cid, mlev in track_a_ids:
        sc = by_sys[cid]
        track_a.append(
            HighFreedomImplCase(
                case_id=sc.case_id,
                title=f"Track A (Factor API & Friction): {sc.title}",
                state_complexity="track_a_factor_api_friction",
                prompt=sc.prompt,
                golden_fn=sc.golden_fn,
                max_leverage=mlev,
            )
        )

    return track_a + list(V3_HARD_IMPL_CASES)


CANONICAL_AUDIT_CASES: List[SubtleAuditCase] = _build_canonical_audit_cases()
CANONICAL_IMPL_CASES: List[HighFreedomImplCase] = _build_canonical_impl_cases()

evaluate_canonical_audit_case = score_subtle_audit_case
evaluate_canonical_impl_case = evaluate_high_freedom_case


class GoldenCanonicalAuditClient(BaseLLMClient):
    def __init__(self):
        self.usage = LLMUsage()
        self._by_id = {c.case_id: c for c in CANONICAL_AUDIT_CASES}

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        self.usage.calls += 1
        for cid, c in self._by_id.items():
            if cid in prompt:
                kw_str = " ".join(g[0] for g in c.required_mechanism_keywords)
                return json.dumps({
                    "has_flaw": c.has_flaw,
                    "flaw_types": list(c.flaw_types),
                    "flawed_location": kw_str,
                    "explanation": f"{c.reference_notes} ({kw_str})",
                })
        return json.dumps({"has_flaw": False, "flaw_types": [], "flawed_location": "none", "explanation": ""})


class GoldenCanonicalImplClient(BaseLLMClient):
    def __init__(self):
        self.usage = LLMUsage()
        self._by_id = {c.case_id: c for c in CANONICAL_IMPL_CASES}

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        self.usage.calls += 1
        for cid, c in self._by_id.items():
            if cid in prompt:
                lines = inspect.getsource(c.golden_fn).splitlines()
                lines[0] = "def build_strategy(df):"
                return "```python\n" + "\n".join(lines) + "\n```"
        return "```python\ndef build_strategy(df):\n    return {'signals': df['close']*0}\n```"
