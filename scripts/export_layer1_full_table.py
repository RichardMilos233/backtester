#!/usr/bin/env python3
"""
Deterministic exporter for the Canonical 40-Case × 8-Model Layer 1 Benchmark Full Data Table.

Generates:
  1. docs/experiments/layer1_canonical/LAYER1_FULL_RESULTS.xlsx (4 sheets)
  2. docs/experiments/layer1_canonical/full_qa_wide_40cases.csv (40 rows, wide format)
  3. docs/experiments/layer1_canonical/full_qa_long_320runs.csv (320 rows, long format)
  4. docs/LAYER1_FULL_EXPERIMENT_TABLE.html (interactive browsable full-data HTML page)

Usage:
  .venv/bin/python scripts/export_layer1_full_table.py
"""
from __future__ import annotations

import html
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from agent.layer1.canonical import CANONICAL_AUDIT_CASES, CANONICAL_IMPL_CASES

MODELS: List[str] = [
    "gpt-6-astra",
    "claude-opus-5.5",
    "claude-sonnet-4-6",
    "gpt-5.4-mini",
    "deepseek-v4-pro",
    "claude-haiku-4-5",
    "deepseek-v3.2",
    "qwen-plus",
]

SUBGROUP_LABELS_ZH: Dict[str, str] = {
    "subtle_cross_module_or_default": "隐式默认参数与跨模块契约陷阱 (10题)",
    "classic_or_compound": "经典与复合策略陷阱 (6题)",
    "clean_control": "无缺陷干净对照组 (4题)",
    "track_a_factor_api": "Track A: 因子库 API 契约与摩擦成本集成 (6题)",
    "track_b_unhinted_stateful": "Track B: 无提示实盘级复杂状态机实现 (14题)",
}


def get_audit_sub_group(case) -> str:
    if not case.has_flaw:
        return "clean_control"
    return "subtle_cross_module_or_default" if case.case_id.startswith("sa_") else "classic_or_compound"


def get_impl_sub_group(case) -> str:
    return "track_a_factor_api" if case.case_id.startswith("sys_") else "track_b_unhinted_stateful"


def extract_golden_code(case) -> str:
    lines = inspect.getsource(case.golden_fn).splitlines()
    if lines:
        lines[0] = "def build_strategy(df):"
    return "\n".join(lines)


def load_canonical_runs(runs_path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    by_mc: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for line in runs_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        model = str(rec["model"])
        if model not in MODELS:
            continue
        res = rec["result"]
        cid = str(res["case_id"])
        by_mc[(model, cid)] = rec
    return by_mc


def diagnose_audit_result(case, res: Dict[str, Any]) -> Tuple[bool, str, str]:
    """
    Returns (passed, matrix_short_text, detailed_diagnosis).
    """
    passed = bool(res.get("strict_pass", False))
    has_gt = bool(res.get("has_flaw_gt", case.has_flaw))
    has_pred = bool(res.get("has_flaw_pred", False))
    gt_types = list(res.get("flaw_types_gt", case.flaw_types))
    pred_types = list(res.get("flaw_types_pred", []))
    f1 = float(res.get("category_f1", 0.0))
    mech = bool(res.get("mechanism_hit", False))

    if passed:
        if not has_gt:
            return True, "PASS (1.00 | 正确放行干净代码)", "正确判定无缺陷 (has_flaw=False)"
        return (
            True,
            f"PASS (1.00 | F1={f1:.2f}, 命中机制)",
            f"正确识别有缺陷，类别 F1={f1:.2f}，准确命中底层机制关键词",
        )

    # Failure diagnosis
    if not has_gt and has_pred:
        short = f"FAIL (0.00 | 干净代码误报: {','.join(pred_types)})"
        detail = f"误报 (False Positive)：该题为完全合规的干净对照组，模型却误判存在 {pred_types} 缺陷"
        return False, short, detail

    if has_gt and not has_pred:
        short = f"FAIL (0.00 | 漏报真实缺陷: {','.join(gt_types)})"
        detail = f"漏报 (False Negative)：未能发现代码中隐藏的 {gt_types} 陷阱，错误判定为 has_flaw=False"
        return False, short, detail

    reasons = []
    short_reasons = []
    if f1 < 0.5:
        short_reasons.append(f"类别错判(F1={f1:.2f})")
        reasons.append(f"缺陷类别错判（标准类别为 {gt_types}，模型预测为 {pred_types}，F1={f1:.2f} < 0.50）")
    if not mech:
        short_reasons.append("未命中核心机制")
        req_kw = " & ".join(["/".join(g[:3]) for g in case.required_mechanism_keywords])
        reasons.append(f"未定位到真正致错机制（缺少核心机制解析：{req_kw}）")

    short = f"FAIL (0.00 | {', '.join(short_reasons)})"
    detail = "；".join(reasons) if reasons else "未达到严格通过标准"
    return False, short, detail


def diagnose_impl_result(case, res: Dict[str, Any]) -> Tuple[bool, str, str]:
    """
    Returns (passed, matrix_short_text, detailed_diagnosis).
    """
    passed = bool(res.get("passed", False))
    exec_pass = bool(res.get("exec_pass", False))
    no_la = bool(res.get("no_lookahead", False))
    nan_ok = bool(res.get("nan_hygiene", False))
    bnd_ok = bool(res.get("bounds_ok", False))
    act = float(res.get("active_action_match", 0.0))
    tf1 = float(res.get("trade_f1", 0.0))
    fric = float(res.get("friction_match", 0.0))
    err = str(res.get("error_message", "")).strip()
    la_reason = str(res.get("lookahead_reason", "")).strip()

    if passed:
        return (
            True,
            f"PASS (1.00 | Act={act:.1%}, F1={tf1:.1%})",
            f"全部契约与等价性测试通过（ActiveMatch={act:.3f}, TradeF1={tf1:.3f}, 无未来函数）",
        )

    if not exec_pass:
        short_err = err.splitlines()[0][:42] if err else "RuntimeError"
        short = f"FAIL (0.00 | 运行报错: {short_err})"
        detail = f"代码执行报错 (Exec Fail)：{err}"
        return False, short, detail

    reasons = []
    short_tags = []
    if not no_la:
        short_tags.append("未来函数泄露")
        reasons.append(f"未来函数因果性检验未通过 ({la_reason or '截断对比发现历史信号随未来数据改变'})")
    if not nan_ok:
        short_tags.append("预热期后含NaN")
        reasons.append("NaN 卫生性未通过（预热期后仍输出 NaN 导致回测引擎锁仓）")
    if not bnd_ok:
        short_tags.append("杠杆越界")
        reasons.append(f"仓位越界或含 Inf（超出 [0, {case.max_leverage}] 范围）")
    if fric < 1.0:
        short_tags.append("摩擦参数错误")
        reasons.append("摩擦成本参数（slippage_rate / commission_rate / rebalance_tolerance）与题目要求不符")
    if act < 0.95 or tf1 < 0.90:
        short_tags.append(f"逻辑偏差 Act={act:.0%}/F1={tf1:.0%}")
        reasons.append(
            f"策略信号与标准答案不一致（ActiveActionMatch={act:.3f} [需>=0.95], TradeF1={tf1:.3f} [需>=0.90]）"
        )

    short = f"FAIL (0.00 | {', '.join(short_tags)})"
    detail = "；".join(reasons)
    return False, short, detail


def build_all_datasets(runs_by_mc: Dict[Tuple[str, str], Dict[str, Any]]) -> Dict[str, Any]:
    """
    Constructs structured tables for:
      - leaderboard_rows (8 models)
      - matrix_rows (40 cases × 8 models)
      - audit_detail_rows (20 audit cases wide)
      - impl_detail_rows (20 impl cases wide)
      - wide_40_rows (all 40 cases wide for full_qa_wide_40cases.csv)
      - long_320_rows (all 320 runs long for Sheet 4 and full_qa_long_320runs.csv)
      - html_cases_data (rich JSON structure for interactive HTML)
    """
    # 1. Leaderboard stats per model
    leaderboard_rows: List[Dict[str, Any]] = []
    for rank, m in enumerate(MODELS, start=1):
        aud_passes = 0
        aud_subtle_passes = 0
        aud_classic_passes = 0
        aud_clean_fp = 0
        aud_mech_hits = 0
        aud_f1_sum = 0.0

        for c in CANONICAL_AUDIT_CASES:
            res = runs_by_mc[(m, c.case_id)]["result"]
            sp = bool(res["strict_pass"])
            aud_passes += int(sp)
            sg = get_audit_sub_group(c)
            if sg == "subtle_cross_module_or_default":
                aud_subtle_passes += int(sp)
            elif sg == "classic_or_compound":
                aud_classic_passes += int(sp)
            elif sg == "clean_control":
                aud_clean_fp += int(bool(res["has_flaw_pred"]))
            aud_mech_hits += int(bool(res["mechanism_hit"]))
            aud_f1_sum += float(res["category_f1"])

        impl_passes = 0
        impl_ta_passes = 0
        impl_tb_passes = 0
        impl_exec_passes = 0
        impl_nola_passes = 0
        impl_act_sum = 0.0
        impl_tf1_sum = 0.0

        for c in CANONICAL_IMPL_CASES:
            res = runs_by_mc[(m, c.case_id)]["result"]
            ps = bool(res["passed"])
            impl_passes += int(ps)
            sg = get_impl_sub_group(c)
            if sg == "track_a_factor_api":
                impl_ta_passes += int(ps)
            else:
                impl_tb_passes += int(ps)
            impl_exec_passes += int(bool(res["exec_pass"]))
            impl_nola_passes += int(bool(res["no_lookahead"]))
            impl_act_sum += float(res["active_action_match"])
            impl_tf1_sum += float(res["trade_f1"])

        total_pass = aud_passes + impl_passes
        leaderboard_rows.append(
            {
                "排名": rank,
                "模型名称 (model)": m,
                "总通过题数 (40题)": f"{total_pass}/40",
                "总分 (overall_score)": round(total_pass / 40.0, 4),
                "挑错题通过率 (20题)": round(aud_passes / 20.0, 4),
                "挑错题通过数": f"{aud_passes}/20",
                "隐蔽陷阱通过率 (10题)": round(aud_subtle_passes / 10.0, 4),
                "经典复合通过率 (6题)": round(aud_classic_passes / 6.0, 4),
                "干净代码误报率 (4题FPR)": round(aud_clean_fp / 4.0, 4),
                "挑错机制命中率": round(aud_mech_hits / 20.0, 4),
                "写代码通过率 (20题)": round(impl_passes / 20.0, 4),
                "写代码通过数": f"{impl_passes}/20",
                "TrackA因子库通过率 (6题)": round(impl_ta_passes / 6.0, 4),
                "TrackB复杂状态机通过率 (14题)": round(impl_tb_passes / 14.0, 4),
                "代码平均ActiveMatch": round(impl_act_sum / 20.0, 4),
                "代码平均TradeF1": round(impl_tf1_sum / 20.0, 4),
            }
        )

    matrix_rows: List[Dict[str, Any]] = []
    audit_detail_rows: List[Dict[str, Any]] = []
    impl_detail_rows: List[Dict[str, Any]] = []
    wide_40_rows: List[Dict[str, Any]] = []
    long_320_rows: List[Dict[str, Any]] = []
    html_cases_data: List[Dict[str, Any]] = []

    # 2. Process 20 Audit Cases
    for idx, c in enumerate(CANONICAL_AUDIT_CASES, start=1):
        sg = get_audit_sub_group(c)
        gt_flaw_str = "有缺陷 (True)" if c.has_flaw else "无缺陷-干净对照 (False)"
        gt_types_str = ", ".join(c.flaw_types) if c.flaw_types else "none (无缺陷)"
        req_kw_str = (
            " AND ".join(["(" + " | ".join(g) + ")" for g in c.required_mechanism_keywords])
            if c.required_mechanism_keywords
            else "无 (保持 has_flaw=False)"
        )
        gt_summary = f"[{gt_flaw_str}] 类别: {gt_types_str} | 核心机制: {c.reference_notes}"

        mat_row: Dict[str, Any] = {
            "序号": idx,
            "case_id": c.case_id,
            "task_type": "audit (挑错题)",
            "sub_group": sg,
            "title": c.title,
            "question_summary": c.description,
            "ground_truth_summary": gt_summary,
        }

        aud_row: Dict[str, Any] = {
            "题号 (case_id)": c.case_id,
            "子类别 (sub_group)": f"{sg} ({SUBGROUP_LABELS_ZH[sg]})",
            "标题 (title)": c.title,
            "策略描述 (description)": c.description,
            "待审代码 (input_code)": c.code.strip(),
            "标准答案：是否有坑 (gt_has_flaw)": gt_flaw_str,
            "标准答案：缺陷类别 (gt_flaw_types)": gt_types_str,
            "标准答案：底层机制解析 (gt_reference_notes)": f"{c.reference_notes}\n[必中机制关键词组: {req_kw_str}]",
        }

        wide_row: Dict[str, Any] = {
            "question_index": idx,
            "case_id": c.case_id,
            "task_type": "audit",
            "sub_group": sg,
            "title": c.title,
            "question_text": c.description,
            "question_code": c.code.strip(),
            "ground_truth_label": f"has_flaw={c.has_flaw}; flaw_types={list(c.flaw_types)}",
            "ground_truth_detail_or_code": c.reference_notes,
        }

        html_models: List[Dict[str, Any]] = []
        pass_count = 0

        for m in MODELS:
            rec = runs_by_mc[(m, c.case_id)]
            res = rec["result"]
            passed, short_cell, diag_detail = diagnose_audit_result(c, res)
            pass_count += int(passed)

            mat_row[m] = short_cell

            pred_flaw = res.get("has_flaw_pred", False)
            pred_types = res.get("flaw_types_pred", [])
            f1 = float(res.get("category_f1", 0.0))
            mech = bool(res.get("mechanism_hit", False))
            loc = str(res.get("flawed_location", "")).strip()
            expl = str(res.get("explanation", "")).strip()
            raw_json = str(res.get("raw_response", "")).strip()

            aud_row[f"[{m}] 是否通过"] = "PASS (对)" if passed else f"FAIL (错: {diag_detail})"
            aud_row[f"[{m}] 预测标签 & 机制命中"] = (
                f"has_flaw={pred_flaw} | flaw_types={pred_types}\n"
                f"Category_F1={f1:.2f} | Mechanism_Hit={mech}\n"
                f"定位位置(flawed_location): {loc or 'none'}\n"
                f"判定诊断: {diag_detail}"
            )
            aud_row[f"[{m}] 模型给出的完整回答 (explanation + raw JSON)"] = (
                f"【机制解释 Explanation】:\n{expl}\n\n【原始输出 Raw JSON】:\n{raw_json}"
            )

            wide_row[f"{m}_passed"] = "PASS" if passed else "FAIL"
            wide_row[f"{m}_score_summary"] = short_cell
            wide_row[f"{m}_diagnosis"] = diag_detail
            wide_row[f"{m}_full_response"] = raw_json

            long_320_rows.append(
                {
                    "run_index": len(long_320_rows) + 1,
                    "question_index": idx,
                    "case_id": c.case_id,
                    "task_type": "audit",
                    "sub_group": sg,
                    "title": c.title,
                    "model": m,
                    "pass_fail_zh": "是对 (PASS)" if passed else "是错 (FAIL)",
                    "passed": 1 if passed else 0,
                    "score": 1.0 if passed else 0.0,
                    "short_verdict": short_cell,
                    "diagnosis": diag_detail,
                    "question_text": c.description,
                    "question_code": c.code.strip(),
                    "ground_truth_label": f"has_flaw={c.has_flaw}; flaw_types={list(c.flaw_types)}",
                    "ground_truth_answer": f"{c.reference_notes} (Keywords: {req_kw_str})",
                    "model_pred_summary": f"has_flaw={pred_flaw}, flaw_types={pred_types}, F1={f1:.2f}, mech_hit={mech}, loc={loc}",
                    "model_explanation": expl,
                    "model_full_output": raw_json,
                    "audit_binary_correct": float(res.get("binary_correct", 0.0)),
                    "audit_category_f1": f1,
                    "audit_mechanism_hit": 1 if mech else 0,
                    "impl_exec_pass": "",
                    "impl_no_lookahead": "",
                    "impl_nan_hygiene": "",
                    "impl_bounds_ok": "",
                    "impl_active_action_match": "",
                    "impl_trade_f1": "",
                    "impl_friction_match": "",
                    "impl_sharpe_ref": "",
                    "impl_sharpe_agent": "",
                }
            )

            html_models.append(
                {
                    "model": m,
                    "passed": passed,
                    "short_cell": short_cell,
                    "diagnosis": diag_detail,
                    "metrics_badges": [
                        f"预测 has_flaw: {pred_flaw}",
                        f"预测类别: {', '.join(pred_types) if pred_types else '[]'}",
                        f"类别 F1: {f1:.2f}",
                        f"机制命中: {'✓ 是' if mech else '✗ 否'}",
                    ],
                    "flawed_location": loc,
                    "explanation": expl,
                    "raw_output": raw_json,
                }
            )

        mat_row["通过模型数"] = f"{pass_count}/8"
        matrix_rows.append(mat_row)
        audit_detail_rows.append(aud_row)
        wide_40_rows.append(wide_row)

        html_cases_data.append(
            {
                "idx": idx,
                "case_id": c.case_id,
                "task_type": "audit",
                "sub_group": sg,
                "sub_group_zh": SUBGROUP_LABELS_ZH[sg],
                "title": c.title,
                "pass_count": pass_count,
                "question_text": c.description,
                "question_code": c.code.strip(),
                "gt_has_flaw": c.has_flaw,
                "gt_flaw_types": list(c.flaw_types),
                "gt_notes": c.reference_notes,
                "gt_keywords": req_kw_str,
                "gt_code": "",
                "models": html_models,
            }
        )

    # 3. Process 20 Implementation Cases
    for offset, c in enumerate(CANONICAL_IMPL_CASES, start=1):
        idx = 20 + offset
        sg = get_impl_sub_group(c)
        golden_code = extract_golden_code(c)
        gt_summary = (
            f"[Golden 标准实现] 最大杠杆上限={c.max_leverage}x | "
            f"验收标准: 无未来函数 + 预热后无NaN + ActiveActionMatch>=95% + TradeF1>=90% + 摩擦成本完全匹配"
        )

        mat_row = {
            "序号": idx,
            "case_id": c.case_id,
            "task_type": "implementation (写代码题)",
            "sub_group": sg,
            "title": c.title,
            "question_summary": c.prompt,
            "ground_truth_summary": gt_summary,
        }

        imp_row: Dict[str, Any] = {
            "题号 (case_id)": c.case_id,
            "子类别 (sub_group)": f"{sg} ({SUBGROUP_LABELS_ZH[sg]})",
            "标题 (title)": c.title,
            "题目要求 (prompt)": c.prompt,
            "标准答案代码 (golden_code)": golden_code,
        }

        wide_row = {
            "question_index": idx,
            "case_id": c.case_id,
            "task_type": "implementation",
            "sub_group": sg,
            "title": c.title,
            "question_text": c.prompt,
            "question_code": "",
            "ground_truth_label": f"max_leverage={c.max_leverage}; state_complexity={c.state_complexity}",
            "ground_truth_detail_or_code": golden_code,
        }

        html_models = []
        pass_count = 0

        for m in MODELS:
            rec = runs_by_mc[(m, c.case_id)]
            res = rec["result"]
            passed, short_cell, diag_detail = diagnose_impl_result(c, res)
            pass_count += int(passed)

            mat_row[m] = short_cell

            exec_p = bool(res.get("exec_pass", False))
            no_la = bool(res.get("no_lookahead", False))
            nan_ok = bool(res.get("nan_hygiene", False))
            bnd_ok = bool(res.get("bounds_ok", False))
            act = float(res.get("active_action_match", 0.0))
            tf1 = float(res.get("trade_f1", 0.0))
            fric = float(res.get("friction_match", 0.0))
            s_ref = float(res.get("sharpe_ref", 0.0))
            s_agt = float(res.get("sharpe_agent", 0.0))
            err = str(res.get("error_message", "")).strip()
            la_r = str(res.get("lookahead_reason", "")).strip()
            code_out = str(res.get("extracted_code", "")).strip()

            imp_row[f"[{m}] 是否通过 & 核心指标 (ActiveMatch / TradeF1 / 错误原因)"] = (
                f"{'PASS (对)' if passed else 'FAIL (错)'}\n"
                f"ActiveMatch={act:.3f} | TradeF1={tf1:.3f} | FrictionMatch={fric:.1f}\n"
                f"ExecPass={exec_p} | NoLookahead={no_la} | NaNHygiene={nan_ok} | BoundsOK={bnd_ok}\n"
                f"Sharpe(Ref={s_ref:.3f}, Agent={s_agt:.3f})\n"
                f"诊断结果: {diag_detail}"
            )
            imp_row[f"[{m}] 模型写出的完整代码 (extracted_code)"] = code_out

            wide_row[f"{m}_passed"] = "PASS" if passed else "FAIL"
            wide_row[f"{m}_score_summary"] = short_cell
            wide_row[f"{m}_diagnosis"] = diag_detail
            wide_row[f"{m}_full_response"] = code_out

            long_320_rows.append(
                {
                    "run_index": len(long_320_rows) + 1,
                    "question_index": idx,
                    "case_id": c.case_id,
                    "task_type": "implementation",
                    "sub_group": sg,
                    "title": c.title,
                    "model": m,
                    "pass_fail_zh": "是对 (PASS)" if passed else "是错 (FAIL)",
                    "passed": 1 if passed else 0,
                    "score": 1.0 if passed else 0.0,
                    "short_verdict": short_cell,
                    "diagnosis": diag_detail,
                    "question_text": c.prompt,
                    "question_code": "",
                    "ground_truth_label": f"ActiveMatch>=0.95 & TradeF1>=0.90 & NoLookahead & MaxLev={c.max_leverage}",
                    "ground_truth_answer": golden_code,
                    "model_pred_summary": (
                        f"exec={exec_p}, no_lookahead={no_la}, nan_ok={nan_ok}, bounds_ok={bnd_ok}, "
                        f"active_match={act:.3f}, trade_f1={tf1:.3f}, friction_match={fric:.1f}"
                    ),
                    "model_explanation": err or la_r or diag_detail,
                    "model_full_output": code_out,
                    "audit_binary_correct": "",
                    "audit_category_f1": "",
                    "audit_mechanism_hit": "",
                    "impl_exec_pass": 1 if exec_p else 0,
                    "impl_no_lookahead": 1 if no_la else 0,
                    "impl_nan_hygiene": 1 if nan_ok else 0,
                    "impl_bounds_ok": 1 if bnd_ok else 0,
                    "impl_active_action_match": round(act, 4),
                    "impl_trade_f1": round(tf1, 4),
                    "impl_friction_match": round(fric, 4),
                    "impl_sharpe_ref": round(s_ref, 4),
                    "impl_sharpe_agent": round(s_agt, 4),
                }
            )

            html_models.append(
                {
                    "model": m,
                    "passed": passed,
                    "short_cell": short_cell,
                    "diagnosis": diag_detail,
                    "metrics_badges": [
                        f"可执行: {'✓' if exec_p else '✗'}",
                        f"无未来函数: {'✓' if no_la else '✗'}",
                        f"ActiveMatch: {act:.1%}",
                        f"TradeF1: {tf1:.1%}",
                        f"Sharpe(基准/模型): {s_ref:.2f} / {s_agt:.2f}",
                    ],
                    "flawed_location": "",
                    "explanation": err if err else (la_r if la_r else ""),
                    "raw_output": code_out,
                }
            )

        mat_row["通过模型数"] = f"{pass_count}/8"
        matrix_rows.append(mat_row)
        impl_detail_rows.append(imp_row)
        wide_40_rows.append(wide_row)

        html_cases_data.append(
            {
                "idx": idx,
                "case_id": c.case_id,
                "task_type": "implementation",
                "sub_group": sg,
                "sub_group_zh": SUBGROUP_LABELS_ZH[sg],
                "title": c.title,
                "pass_count": pass_count,
                "question_text": c.prompt,
                "question_code": "",
                "gt_has_flaw": None,
                "gt_flaw_types": [],
                "gt_notes": gt_summary,
                "gt_keywords": f"最大杠杆: {c.max_leverage}x | 复杂度分类: {c.state_complexity}",
                "gt_code": golden_code,
                "models": html_models,
            }
        )

    return {
        "leaderboard_rows": leaderboard_rows,
        "matrix_rows": matrix_rows,
        "audit_detail_rows": audit_detail_rows,
        "impl_detail_rows": impl_detail_rows,
        "wide_40_rows": wide_40_rows,
        "long_320_rows": long_320_rows,
        "html_cases_data": html_cases_data,
    }


def write_excel_workbook(data: Dict[str, Any], out_xlsx: Path) -> None:
    wb = openpyxl.Workbook()
    # Remove default sheet
    wb.remove(wb.active)

    font_family = "Arial"
    title_font = Font(name=font_family, size=13, bold=True, color="0F172A")
    section_font = Font(name=font_family, size=11, bold=True, color="0F172A")
    header_font = Font(name=font_family, size=10, bold=True, color="FFFFFF")
    sub_header_font = Font(name=font_family, size=10, bold=True, color="1E293B")
    cell_font = Font(name=font_family, size=9, bold=False, color="1E293B")
    code_font = Font(name="Consolas", size=9, bold=False, color="0F172A")
    pass_font = Font(name=font_family, size=9, bold=True, color="065F46")
    fail_font = Font(name=font_family, size=9, bold=True, color="9F1239")

    header_fill = PatternFill(start_color="1E293B", end_color="1E293B", fill_type="solid")
    sub_header_fill = PatternFill(start_color="E2E8F0", end_color="E2E8F0", fill_type="solid")
    zebra_fill = PatternFill(start_color="F8FAFC", end_color="F8FAFC", fill_type="solid")
    pass_fill = PatternFill(start_color="D1FAE5", end_color="D1FAE5", fill_type="solid")
    fail_fill = PatternFill(start_color="FFE4E6", end_color="FFE4E6", fill_type="solid")

    thin_side = Side(border_style="thin", color="CBD5E1")
    grid_border = Border(left=thin_side, right=thin_side, top=thin_side, bottom=thin_side)

    align_left_wrap = Alignment(horizontal="left", vertical="top", wrap_text=True)
    align_center_wrap = Alignment(horizontal="center", vertical="center", wrap_text=True)

    # =========================================================================
    # Sheet 1: 1_总榜与40题通过矩阵
    # =========================================================================
    ws1 = wb.create_sheet(title="1_总榜与40题通过矩阵")
    ws1.views.sheetView[0].showGridLines = True

    ws1["A1"] = "TIA Layer 1 策略代码评测基准 — 8 模型总榜与 40 题通过矩阵 (Canonical 40 Cases × 8 LLMs = 320 Runs)"
    ws1["A1"].font = title_font
    ws1["A2"] = "表 1A：8 模型核心得分总榜（按 40 题总分降序排列）"
    ws1["A2"].font = section_font

    lb_rows = data["leaderboard_rows"]
    lb_headers = list(lb_rows[0].keys())
    for col_idx, h in enumerate(lb_headers, start=1):
        c = ws1.cell(row=3, column=col_idx, value=h)
        c.font = header_font
        c.fill = header_fill
        c.alignment = align_center_wrap
        c.border = grid_border
    ws1.row_dimensions[3].height = 28

    pct_cols_lb = {
        "总分 (overall_score)",
        "挑错题通过率 (20题)",
        "隐蔽陷阱通过率 (10题)",
        "经典复合通过率 (6题)",
        "干净代码误报率 (4题FPR)",
        "挑错机制命中率",
        "写代码通过率 (20题)",
        "TrackA因子库通过率 (6题)",
        "TrackB复杂状态机通过率 (14题)",
        "代码平均ActiveMatch",
        "代码平均TradeF1",
    }

    for r_offset, row_data in enumerate(lb_rows, start=4):
        ws1.row_dimensions[r_offset].height = 22
        for col_idx, h in enumerate(lb_headers, start=1):
            val = row_data[h]
            c = ws1.cell(row=r_offset, column=col_idx, value=val)
            c.font = cell_font
            c.border = grid_border
            c.alignment = align_center_wrap if col_idx != 2 else Alignment(horizontal="left", vertical="center")
            if r_offset % 2 == 1:
                c.fill = zebra_fill
            if h in pct_cols_lb and isinstance(val, (int, float)):
                c.number_format = "0.0%"

    matrix_start_row = 4 + len(lb_rows) + 2
    ws1.cell(
        row=matrix_start_row - 1,
        column=1,
        value="表 1B：40 道标准真题 × 8 模型通过与失分原因全景矩阵（绿色 PASS = 满分通过，红色 FAIL = 失分及具体原因）",
    ).font = section_font

    mat_rows = data["matrix_rows"]
    mat_headers = list(mat_rows[0].keys())
    header_row_idx = matrix_start_row
    ws1.row_dimensions[header_row_idx].height = 30
    for col_idx, h in enumerate(mat_headers, start=1):
        c = ws1.cell(row=header_row_idx, column=col_idx, value=h)
        c.font = header_font
        c.fill = header_fill
        c.alignment = align_center_wrap
        c.border = grid_border

    for idx_m, row_data in enumerate(mat_rows, start=header_row_idx + 1):
        ws1.row_dimensions[idx_m].height = 42
        for col_idx, h in enumerate(mat_headers, start=1):
            val = row_data[h]
            c = ws1.cell(row=idx_m, column=col_idx, value=val)
            c.border = grid_border
            if h in MODELS:
                s_val = str(val)
                if s_val.startswith("PASS"):
                    c.fill = pass_fill
                    c.font = pass_font
                else:
                    c.fill = fail_fill
                    c.font = fail_font
                c.alignment = align_left_wrap
            elif h in ("序号", "通过模型数", "task_type"):
                c.font = cell_font
                c.alignment = align_center_wrap
            else:
                c.font = cell_font
                c.alignment = align_left_wrap

    ws1.freeze_panes = f"C{header_row_idx + 1}"

    ws1_widths = {
        1: 8,
        2: 32,
        3: 22,
        4: 30,
        5: 36,
        6: 46,
        7: 46,
        8: 28,
        9: 28,
        10: 28,
        11: 28,
        12: 28,
        13: 28,
        14: 28,
        15: 28,
        16: 14,
    }
    for col_idx, w in ws1_widths.items():
        ws1.column_dimensions[get_column_letter(col_idx)].width = w

    # =========================================================================
    # Sheet 2: 2_挑错题20道全量问答明细
    # =========================================================================
    ws2 = wb.create_sheet(title="2_挑错题20道全量问答明细")
    ws2.views.sheetView[0].showGridLines = True
    aud_rows = data["audit_detail_rows"]
    aud_headers = list(aud_rows[0].keys())

    ws2.row_dimensions[1].height = 34
    for col_idx, h in enumerate(aud_headers, start=1):
        c = ws2.cell(row=1, column=col_idx, value=h)
        c.font = header_font
        c.fill = header_fill
        c.alignment = align_center_wrap
        c.border = grid_border

    for r_idx, row_data in enumerate(aud_rows, start=2):
        ws2.row_dimensions[r_idx].height = 135
        for col_idx, h in enumerate(aud_headers, start=1):
            val = row_data[h]
            c = ws2.cell(row=r_idx, column=col_idx, value=val)
            c.border = grid_border
            c.alignment = align_left_wrap
            if h.endswith("是否通过"):
                if str(val).startswith("PASS"):
                    c.fill = pass_fill
                    c.font = pass_font
                else:
                    c.fill = fail_fill
                    c.font = fail_font
            elif "代码" in h or "JSON" in h:
                c.font = code_font
            else:
                c.font = cell_font

    ws2.freeze_panes = "B2"
    for col_idx, h in enumerate(aud_headers, start=1):
        letter = get_column_letter(col_idx)
        if col_idx == 1:
            ws2.column_dimensions[letter].width = 28
        elif col_idx in (2, 3):
            ws2.column_dimensions[letter].width = 30
        elif col_idx == 4:
            ws2.column_dimensions[letter].width = 42
        elif col_idx == 5:
            ws2.column_dimensions[letter].width = 55
        elif col_idx in (6, 7):
            ws2.column_dimensions[letter].width = 24
        elif col_idx == 8:
            ws2.column_dimensions[letter].width = 48
        elif h.endswith("是否通过"):
            ws2.column_dimensions[letter].width = 26
        elif h.endswith("预测标签 & 机制命中"):
            ws2.column_dimensions[letter].width = 36
        else:
            ws2.column_dimensions[letter].width = 55

    # =========================================================================
    # Sheet 3: 3_写代码题20道全量问答明细
    # =========================================================================
    ws3 = wb.create_sheet(title="3_写代码题20道全量问答明细")
    ws3.views.sheetView[0].showGridLines = True
    imp_rows = data["impl_detail_rows"]
    imp_headers = list(imp_rows[0].keys())

    ws3.row_dimensions[1].height = 34
    for col_idx, h in enumerate(imp_headers, start=1):
        c = ws3.cell(row=1, column=col_idx, value=h)
        c.font = header_font
        c.fill = header_fill
        c.alignment = align_center_wrap
        c.border = grid_border

    for r_idx, row_data in enumerate(imp_rows, start=2):
        ws3.row_dimensions[r_idx].height = 150
        for col_idx, h in enumerate(imp_headers, start=1):
            val = row_data[h]
            c = ws3.cell(row=r_idx, column=col_idx, value=val)
            c.border = grid_border
            c.alignment = align_left_wrap
            if "是否通过 & 核心指标" in h:
                if str(val).startswith("PASS"):
                    c.fill = pass_fill
                    c.font = pass_font
                else:
                    c.fill = fail_fill
                    c.font = fail_font
            elif "代码" in h:
                c.font = code_font
            else:
                c.font = cell_font

    ws3.freeze_panes = "B2"
    for col_idx, h in enumerate(imp_headers, start=1):
        letter = get_column_letter(col_idx)
        if col_idx == 1:
            ws3.column_dimensions[letter].width = 34
        elif col_idx in (2, 3):
            ws3.column_dimensions[letter].width = 32
        elif col_idx == 4:
            ws3.column_dimensions[letter].width = 45
        elif col_idx == 5:
            ws3.column_dimensions[letter].width = 58
        elif "是否通过 & 核心指标" in h:
            ws3.column_dimensions[letter].width = 38
        else:
            ws3.column_dimensions[letter].width = 58

    # =========================================================================
    # Sheet 4: 4_全量320条逐次运行长表
    # =========================================================================
    ws4 = wb.create_sheet(title="4_全量320条逐次运行长表")
    ws4.views.sheetView[0].showGridLines = True
    long_rows = data["long_320_rows"]
    long_headers = list(long_rows[0].keys())

    ws4.row_dimensions[1].height = 30
    for col_idx, h in enumerate(long_headers, start=1):
        c = ws4.cell(row=1, column=col_idx, value=h)
        c.font = header_font
        c.fill = header_fill
        c.alignment = align_center_wrap
        c.border = grid_border

    for r_idx, row_data in enumerate(long_rows, start=2):
        ws4.row_dimensions[r_idx].height = 68
        passed_bool = row_data["passed"] == 1
        for col_idx, h in enumerate(long_headers, start=1):
            val = row_data[h]
            c = ws4.cell(row=r_idx, column=col_idx, value=val)
            c.border = grid_border
            if h in ("pass_fail_zh", "short_verdict"):
                c.fill = pass_fill if passed_bool else fail_fill
                c.font = pass_font if passed_bool else fail_font
                c.alignment = align_left_wrap
            elif h in ("question_code", "ground_truth_answer", "model_full_output"):
                c.font = code_font
                c.alignment = align_left_wrap
            elif h in ("run_index", "question_index", "passed", "score", "task_type", "model"):
                c.font = cell_font
                c.alignment = align_center_wrap
            else:
                c.font = cell_font
                c.alignment = align_left_wrap

    ws4.freeze_panes = "D2"
    ws4_widths = {
        "run_index": 10,
        "question_index": 12,
        "case_id": 32,
        "task_type": 15,
        "sub_group": 28,
        "title": 34,
        "model": 18,
        "pass_fail_zh": 14,
        "passed": 10,
        "score": 10,
        "short_verdict": 32,
        "diagnosis": 42,
        "question_text": 44,
        "question_code": 50,
        "ground_truth_label": 32,
        "ground_truth_answer": 50,
        "model_pred_summary": 38,
        "model_explanation": 44,
        "model_full_output": 54,
    }
    for col_idx, h in enumerate(long_headers, start=1):
        ws4.column_dimensions[get_column_letter(col_idx)].width = ws4_widths.get(h, 16)

    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_xlsx)


def write_html_page(data: Dict[str, Any], out_html: Path) -> None:
    lb_rows = data["leaderboard_rows"]
    cases_data = data["html_cases_data"]

    # Build top leaderboard cards HTML
    lb_tr_html = []
    for r in lb_rows:
        m = r["模型名称 (model)"]
        overall_pct = r["总分 (overall_score)"] * 100
        aud_pct = r["挑错题通过率 (20题)"] * 100
        imp_pct = r["写代码通过率 (20题)"] * 100
        lb_tr_html.append(
            f"""
            <tr>
              <td><span class="rank-badge">#{r['排名']}</span> <strong>{html.escape(m)}</strong></td>
              <td><span class="score-pill">{overall_pct:.1f}%</span> <span class="muted">({r['总通过题数 (40题)']})</span></td>
              <td><strong>{aud_pct:.0f}%</strong> <span class="muted">({r['挑错题通过数']})</span></td>
              <td><strong>{r['隐蔽陷阱通过率 (10题)']*100:.0f}%</strong></td>
              <td><strong>{r['干净代码误报率 (4题FPR)']*100:.0f}%</strong></td>
              <td><strong>{imp_pct:.0f}%</strong> <span class="muted">({r['写代码通过数']})</span></td>
              <td><strong>{r['代码平均ActiveMatch']*100:.1f}%</strong></td>
              <td><strong>{r['代码平均TradeF1']*100:.1f}%</strong></td>
            </tr>
            """
        )

    # Build 40 x 8 Matrix rows HTML
    matrix_tr_html = []
    for c in cases_data:
        idx = c["idx"]
        cid = c["case_id"]
        ttype = "挑错题" if c["task_type"] == "audit" else "实现题"
        ttype_cls = "tag-audit" if c["task_type"] == "audit" else "tag-impl"
        cells_html = []
        for m_obj in c["models"]:
            passed = m_obj["passed"]
            short_reason = m_obj["short_cell"].replace("PASS (1.00 | ", "").replace("FAIL (0.00 | ", "").rstrip(")")
            if passed:
                cells_html.append(
                    f'<td class="m-cell pass" onclick="jumpToCase(\'{cid}\', \'{m_obj["model"]}\')" title="{html.escape(m_obj["diagnosis"])}">'
                    f'<div class="m-status">✓ 对</div><div class="m-sub">{html.escape(short_reason)}</div></td>'
                )
            else:
                cells_html.append(
                    f'<td class="m-cell fail" onclick="jumpToCase(\'{cid}\', \'{m_obj["model"]}\')" title="{html.escape(m_obj["diagnosis"])}">'
                    f'<div class="m-status">✗ 错</div><div class="m-sub">{html.escape(short_reason)}</div></td>'
                )

        matrix_tr_html.append(
            f"""
            <tr class="matrix-row" data-task="{c['task_type']}" data-case="{cid}">
              <td class="col-idx">{idx}</td>
              <td class="col-id">
                <a href="#card-{cid}" onclick="jumpToCase('{cid}', ''); return false;" class="case-link">{html.escape(cid)}</a>
                <div class="sub-title">{html.escape(c['title'])}</div>
              </td>
              <td><span class="task-tag {ttype_cls}">{ttype}</span></td>
              <td class="col-passcount"><strong>{c['pass_count']}/8</strong></td>
              {''.join(cells_html)}
            </tr>
            """
        )

    # Build 40 Full Deep-Dive Question Cards HTML
    cards_html = []
    for c in cases_data:
        idx = c["idx"]
        cid = c["case_id"]
        is_audit = c["task_type"] == "audit"
        ttype_label = "第一部分：代码挑错题 (Identify Bad Alpha)" if is_audit else "第二部分：按规则写代码题 (Systematic Implementation)"
        ttype_cls = "tag-audit" if is_audit else "tag-impl"

        # Question block
        q_code_block = ""
        if c["question_code"]:
            q_code_block = f"""
            <div class="code-label">题面待审代码 (Input Code Under Audit)：</div>
            <pre class="code-box"><code>{html.escape(c["question_code"])}</code></pre>
            """

        # Ground truth block
        if is_audit:
            flaw_badge = (
                '<span class="verdict-badge fail-badge">标准答案：有缺陷 (has_flaw = True)</span>'
                if c["gt_has_flaw"]
                else '<span class="verdict-badge pass-badge">标准答案：无缺陷干净代码 (has_flaw = False)</span>'
            )
            types_str = ", ".join(c["gt_flaw_types"]) if c["gt_flaw_types"] else "[] (无缺陷)"
            gt_block = f"""
            <div class="gt-meta">
              {flaw_badge}
              <span class="meta-pill">标准缺陷类别: <code>{html.escape(types_str)}</code></span>
              <span class="meta-pill">必中机制词组: <code>{html.escape(c["gt_keywords"])}</code></span>
            </div>
            <div class="gt-notes">
              <strong>标准底层机制解析：</strong>{html.escape(c["gt_notes"])}
            </div>
            """
        else:
            gt_block = f"""
            <div class="gt-meta">
              <span class="verdict-badge pass-badge">Golden 标准参考实现</span>
              <span class="meta-pill">{html.escape(c["gt_keywords"])}</span>
            </div>
            <div class="gt-notes" style="margin-bottom:8px;">
              <strong>验收标准：</strong>{html.escape(c["gt_notes"])}
            </div>
            <pre class="code-box"><code>{html.escape(c["gt_code"])}</code></pre>
            """

        # 8 Model tabs + panels + summary table
        model_rows_summary = []
        model_tabs_buttons = []
        model_panels = []

        for m_idx, m_obj in enumerate(c["models"]):
            m_name = m_obj["model"]
            passed = m_obj["passed"]
            status_badge = (
                '<span class="verdict-badge pass-badge">✓ 对 (PASS)</span>'
                if passed
                else '<span class="verdict-badge fail-badge">✗ 错 (FAIL)</span>'
            )
            tab_cls = "pass-tab" if passed else "fail-tab"
            active_cls = "active" if m_idx == 0 else ""
            badges_html = "".join(f'<span class="meta-pill">{html.escape(b)}</span>' for b in m_obj["metrics_badges"])

            model_tabs_buttons.append(
                f'<button type="button" class="model-tab-btn {tab_cls} {active_cls}" '
                f'data-model="{m_name}" onclick="switchModelTab(\'{cid}\', \'{m_name}\')">'
                f'{"✓" if passed else "✗"} {html.escape(m_name)}</button>'
            )

            expl_block = ""
            if m_obj["explanation"]:
                label_txt = "模型机制解析 (Explanation)：" if is_audit else "报错 / 因果检验诊断信息："
                expl_block = f"""
                <div class="model-expl">
                  <strong>{label_txt}</strong> {html.escape(m_obj["explanation"])}
                </div>
                """

            out_label = "模型实际输出完整 JSON：" if is_audit else "模型实际写出的完整 Python 代码："
            model_panels.append(
                f"""
                <div class="model-panel {active_cls}" id="panel-{cid}-{m_name}" data-model="{m_name}">
                  <div class="model-panel-header">
                    <div class="model-panel-title">
                      <strong>{html.escape(m_name)}</strong> {status_badge}
                      <span class="diag-text">{html.escape(m_obj["diagnosis"])}</span>
                    </div>
                    <div class="model-badges">{badges_html}</div>
                  </div>
                  {expl_block}
                  <div class="code-label">{out_label}</div>
                  <pre class="code-box"><code>{html.escape(m_obj["raw_output"])}</code></pre>
                </div>
                """
            )

            model_rows_summary.append(
                f"""
                <tr class="model-summary-row {'row-pass' if passed else 'row-fail'}" data-model="{m_name}" onclick="switchModelTab('{cid}', '{m_name}')" style="cursor:pointer;">
                  <td><strong>{html.escape(m_name)}</strong></td>
                  <td>{status_badge}</td>
                  <td>{badges_html}</td>
                  <td>{html.escape(m_obj["diagnosis"])}</td>
                </tr>
                """
            )

        cards_html.append(
            f"""
            <div class="case-card" id="card-{cid}" data-task="{c['task_type']}" data-search="{html.escape((cid + ' ' + c['title'] + ' ' + c['question_text']).lower())}">
              <div class="case-card-header">
                <div>
                  <span class="case-num">第 {idx} 题</span>
                  <span class="task-tag {ttype_cls}">{ttype_label}</span>
                  <span class="subgroup-tag">{html.escape(c['sub_group_zh'])}</span>
                  <h3 class="case-title"><code>{html.escape(cid)}</code> — {html.escape(c['title'])}</h3>
                </div>
                <div class="case-pass-pill">通过模型：<strong>{c['pass_count']} / 8</strong></div>
              </div>

              <div class="qa-grid">
                <div class="qa-col">
                  <div class="section-banner q-banner">1. 我们提的问题 (Question & Prompt)</div>
                  <div class="q-desc">{html.escape(c['question_text'])}</div>
                  {q_code_block}
                </div>
                <div class="qa-col">
                  <div class="section-banner gt-banner">2. 标准答案 (Ground Truth Reference)</div>
                  {gt_block}
                </div>
              </div>

              <div class="models-section">
                <div class="section-banner m-banner">3. 8 个模型给出的真实结果对比（点击任意行或页签切换查看完整原始输出）</div>
                <table class="mini-compare-table">
                  <thead>
                    <tr>
                      <th style="width:16%">模型</th>
                      <th style="width:13%">对错判定</th>
                      <th style="width:36%">核心得分指标</th>
                      <th style="width:35%">判定依据与错误诊断</th>
                    </tr>
                  </thead>
                  <tbody>
                    {''.join(model_rows_summary)}
                  </tbody>
                </table>

                <div class="model-tabs-bar" id="tabs-{cid}">
                  {''.join(model_tabs_buttons)}
                </div>
                <div class="model-panels-wrap" id="panels-{cid}">
                  {''.join(model_panels)}
                </div>
              </div>
            </div>
            """
        )

    page_html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TIA 策略代码评测基准 — 40 题 × 8 模型全量原始实验数据大表</title>
<style>
  :root {{
    --slate-950: #090d16;
    --slate-900: #0f172a;
    --slate-800: #1e293b;
    --slate-700: #334155;
    --slate-600: #475569;
    --slate-500: #64748b;
    --slate-300: #cbd5e1;
    --slate-200: #e2e8f0;
    --slate-100: #f1f5f9;
    --slate-50:  #f8fafc;
    --blue-600:  #2563eb;
    --blue-50:   #eff6ff;
    --emerald-700: #047857;
    --emerald-600: #059669;
    --emerald-100: #d1fae5;
    --emerald-50:  #ecfdf5;
    --rose-700:  #be123c;
    --rose-600:  #e11d48;
    --rose-100:  #ffe4e6;
    --rose-50:   #fff1f2;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", "Microsoft YaHei", sans-serif;
    background: var(--slate-100);
    color: var(--slate-800);
    line-height: 1.55;
    font-size: 14px;
    padding: 24px 18px 60px;
  }}
  .container {{
    max-width: 1440px;
    margin: 0 auto;
  }}
  .top-card {{
    background: #fff;
    border: 1px solid var(--slate-200);
    border-radius: 12px;
    padding: 28px 32px;
    margin-bottom: 20px;
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
  }}
  .nav-bar {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
    margin-bottom: 16px;
    padding-bottom: 14px;
    border-bottom: 1px solid var(--slate-200);
  }}
  .btn-link {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 8px 14px;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 600;
    text-decoration: none;
    border: 1px solid var(--slate-300);
    color: var(--slate-800);
    background: var(--slate-50);
  }}
  .btn-link.primary {{
    background: var(--blue-600);
    color: #fff;
    border-color: var(--blue-600);
  }}
  h1 {{
    font-size: 24px;
    font-weight: 800;
    color: var(--slate-900);
    margin-bottom: 6px;
  }}
  .subtitle {{
    color: var(--slate-600);
    font-size: 14.5px;
    margin-bottom: 18px;
  }}
  .lb-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    margin-top: 10px;
  }}
  .lb-table th, .lb-table td {{
    padding: 9px 12px;
    border-bottom: 1px solid var(--slate-200);
    text-align: left;
  }}
  .lb-table th {{
    background: var(--slate-800);
    color: #fff;
    font-weight: 600;
    font-size: 12px;
  }}
  .rank-badge {{
    display: inline-block;
    width: 24px;
    height: 22px;
    line-height: 22px;
    text-align: center;
    background: var(--slate-200);
    border-radius: 5px;
    font-weight: 700;
    font-size: 11.5px;
    margin-right: 6px;
  }}
  .score-pill {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 6px;
    background: var(--emerald-100);
    color: var(--emerald-700);
    font-weight: 700;
  }}
  .muted {{ color: var(--slate-500); font-size: 12px; }}

  /* Matrix table */
  .matrix-wrap {{
    overflow-x: auto;
    margin-top: 12px;
    border: 1px solid var(--slate-200);
    border-radius: 10px;
  }}
  .matrix-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
    min-width: 1180px;
  }}
  .matrix-table th {{
    background: var(--slate-900);
    color: #fff;
    padding: 10px 8px;
    font-weight: 600;
    text-align: center;
    position: sticky;
    top: 0;
    z-index: 2;
  }}
  .matrix-table td {{
    border: 1px solid var(--slate-200);
    padding: 6px 8px;
    vertical-align: middle;
  }}
  .col-idx {{ text-align: center; font-weight: 700; color: var(--slate-500); width: 42px; }}
  .col-id {{ width: 230px; }}
  .case-link {{ font-weight: 700; color: var(--blue-600); text-decoration: none; font-family: monospace; font-size: 12.5px; }}
  .case-link:hover {{ text-decoration: underline; }}
  .sub-title {{ font-size: 11px; color: var(--slate-500); line-height: 1.3; margin-top: 2px; }}
  .col-passcount {{ text-align: center; width: 68px; }}
  .task-tag {{
    display: inline-block;
    padding: 2px 7px;
    border-radius: 5px;
    font-size: 11px;
    font-weight: 700;
  }}
  .tag-audit {{ background: #e0f2fe; color: #0369a1; }}
  .tag-impl {{ background: #ede9fe; color: #5b21b6; }}

  .m-cell {{
    cursor: pointer;
    text-align: center;
    width: 108px;
    transition: transform 0.08s ease;
  }}
  .m-cell:hover {{ filter: brightness(0.95); }}
  .m-cell.pass {{ background: var(--emerald-50); color: var(--emerald-700); }}
  .m-cell.fail {{ background: var(--rose-50); color: var(--rose-700); }}
  .m-status {{ font-weight: 800; font-size: 12px; }}
  .m-sub {{ font-size: 10.5px; opacity: 0.88; line-height: 1.25; margin-top: 2px; }}

  /* Filter controls */
  .filter-bar {{
    position: sticky;
    top: 0;
    z-index: 10;
    background: rgba(255,255,255,0.96);
    backdrop-filter: blur(8px);
    border: 1px solid var(--slate-200);
    border-radius: 10px;
    padding: 12px 18px;
    margin-bottom: 18px;
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.05);
  }}
  .filter-group {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }}
  .filter-btn {{
    padding: 6px 12px;
    border-radius: 6px;
    border: 1px solid var(--slate-300);
    background: #fff;
    font-size: 12.5px;
    font-weight: 600;
    cursor: pointer;
    color: var(--slate-700);
  }}
  .filter-btn.active {{
    background: var(--slate-900);
    color: #fff;
    border-color: var(--slate-900);
  }}
  .search-input, .model-select {{
    padding: 6px 11px;
    border-radius: 6px;
    border: 1px solid var(--slate-300);
    font-size: 13px;
  }}

  /* Deep dive case cards */
  .case-card {{
    background: #fff;
    border: 1px solid var(--slate-300);
    border-radius: 12px;
    padding: 22px 26px;
    margin-bottom: 22px;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
  }}
  .case-card-header {{
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 14px;
    padding-bottom: 14px;
    border-bottom: 2px solid var(--slate-100);
    margin-bottom: 16px;
  }}
  .case-num {{
    display: inline-block;
    background: var(--slate-900);
    color: #fff;
    font-weight: 700;
    font-size: 12px;
    padding: 2px 8px;
    border-radius: 5px;
    margin-right: 6px;
  }}
  .subgroup-tag {{
    display: inline-block;
    background: var(--slate-100);
    color: var(--slate-700);
    font-size: 11.5px;
    padding: 2px 8px;
    border-radius: 5px;
    margin-left: 6px;
  }}
  .case-title {{
    font-size: 17px;
    margin-top: 6px;
    color: var(--slate-900);
  }}
  .case-pass-pill {{
    background: var(--slate-100);
    padding: 6px 12px;
    border-radius: 8px;
    font-size: 13px;
    white-space: nowrap;
  }}
  .qa-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 18px;
    margin-bottom: 18px;
  }}
  .qa-col {{
    background: var(--slate-50);
    border: 1px solid var(--slate-200);
    border-radius: 10px;
    padding: 14px 16px;
    min-width: 0;
  }}
  .section-banner {{
    font-size: 12.5px;
    font-weight: 800;
    text-transform: uppercase;
    padding: 4px 10px;
    border-radius: 6px;
    margin-bottom: 10px;
    display: inline-block;
  }}
  .q-banner {{ background: #dbeafe; color: #1e40af; }}
  .gt-banner {{ background: #d1fae5; color: #065f46; }}
  .m-banner {{ background: #e2e8f0; color: #0f172a; }}
  .q-desc, .gt-notes {{
    font-size: 13.5px;
    color: var(--slate-800);
    margin-bottom: 10px;
    line-height: 1.6;
  }}
  .code-label {{
    font-size: 12px;
    font-weight: 700;
    color: var(--slate-600);
    margin: 8px 0 4px;
  }}
  .code-box {{
    background: var(--slate-950);
    color: #f8fafc;
    padding: 12px 14px;
    border-radius: 8px;
    font-family: "SFMono-Regular", Consolas, Menlo, monospace;
    font-size: 12px;
    line-height: 1.45;
    overflow-x: auto;
    max-height: 400px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-word;
  }}
  .gt-meta {{
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 10px;
  }}
  .verdict-badge {{
    display: inline-block;
    padding: 3px 9px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 700;
  }}
  .pass-badge {{ background: var(--emerald-100); color: var(--emerald-700); }}
  .fail-badge {{ background: var(--rose-100); color: var(--rose-700); }}
  .meta-pill {{
    display: inline-block;
    background: #fff;
    border: 1px solid var(--slate-300);
    padding: 2px 8px;
    border-radius: 5px;
    font-size: 11.5px;
    color: var(--slate-700);
    margin-right: 4px;
    margin-bottom: 3px;
  }}
  .mini-compare-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 12.5px;
    margin-bottom: 12px;
  }}
  .mini-compare-table th {{
    background: var(--slate-100);
    padding: 7px 10px;
    text-align: left;
    border: 1px solid var(--slate-200);
  }}
  .mini-compare-table td {{
    padding: 6px 10px;
    border: 1px solid var(--slate-200);
  }}
  .row-pass {{ background: #f0fdf4; }}
  .row-fail {{ background: #fff1f2; }}

  .model-tabs-bar {{
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 10px;
  }}
  .model-tab-btn {{
    padding: 6px 12px;
    border-radius: 7px;
    font-size: 12.5px;
    font-weight: 700;
    cursor: pointer;
    border: 1px solid var(--slate-300);
    background: #fff;
  }}
  .model-tab-btn.pass-tab {{ color: var(--emerald-700); border-color: #6ee7b7; background: var(--emerald-50); }}
  .model-tab-btn.fail-tab {{ color: var(--rose-700); border-color: #fda4af; background: var(--rose-50); }}
  .model-tab-btn.active {{
    outline: 2px solid var(--slate-900);
    box-shadow: 0 2px 6px rgba(0,0,0,0.12);
  }}
  .model-panel {{
    display: none;
    background: var(--slate-50);
    border: 1px solid var(--slate-300);
    border-radius: 9px;
    padding: 14px 16px;
  }}
  .model-panel.active {{ display: block; }}
  .model-panel-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 10px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--slate-200);
  }}
  .diag-text {{ font-size: 13px; color: var(--slate-700); margin-left: 8px; }}
  .model-expl {{
    background: #fff;
    border-left: 3px solid var(--blue-600);
    padding: 8px 12px;
    font-size: 13px;
    margin-bottom: 8px;
    border-radius: 4px;
  }}
  @media (max-width: 980px) {{
    .qa-grid {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>
<div class="container">

  <div class="top-card">
    <div class="nav-bar">
      <div>
        <a href="LAYER1_BENCHMARK_SHOWCASE.html" class="btn-link">← 返回执行摘要报告 (Showcase Report)</a>
      </div>
      <div style="display:flex; gap:10px; flex-wrap:wrap;">
        <a href="experiments/layer1_canonical/LAYER1_FULL_RESULTS.xlsx" class="btn-link primary" download>⬇ 下载完整 Excel 工作簿 (.xlsx 含4张表)</a>
        <a href="experiments/layer1_canonical/full_qa_wide_40cases.csv" class="btn-link" download>⬇ 下载 40 题宽表 (.csv)</a>
        <a href="experiments/layer1_canonical/full_qa_long_320runs.csv" class="btn-link" download>⬇ 下载 320 次运行长表 (.csv)</a>
      </div>
    </div>

    <h1>TIA 策略代码评测基准 — 40 题 × 8 模型全量实验问答与代码明细表</h1>
    <p class="subtitle">
      本页面完整收录全部 40 道评测真题（20 道代码挑错题 + 20 道策略代码实现题）的<strong>完整题面、标准答案、8 个参测大模型的原始输出/代码、核心指标与逐题对错诊断</strong>（共 40 × 8 = 320 条完整实验记录）。
    </p>

    <h3 style="font-size:15px; margin-bottom:6px;">表 1：8 模型核心得分总榜</h3>
    <table class="lb-table">
      <thead>
        <tr>
          <th>排名 & 模型</th>
          <th>40题总分 (通过数)</th>
          <th>挑错题总通过率 (20题)</th>
          <th>隐蔽陷阱通过率 (10题)</th>
          <th>干净代码误报率 (4题FPR)</th>
          <th>写代码总通过率 (20题)</th>
          <th>平均 ActiveMatch</th>
          <th>平均 TradeF1</th>
        </tr>
      </thead>
      <tbody>
        {''.join(lb_tr_html)}
      </tbody>
    </table>
  </div>

  <div class="top-card">
    <h3 style="font-size:16px; margin-bottom:4px;">表 2：40 题 × 8 模型对错全貌热力矩阵（点击任意单元格可直接定位到该题与对应模型的完整回答）</h3>
    <p class="muted" style="margin-bottom:8px;">绿色 <strong>✓ 对</strong> 表示完全通过严格判分标准；红色 <strong>✗ 错</strong> 显示核心失分原因（误报 / 漏报 / 类别错判 / 未来函数 / 运行报错 / 信号偏差）。</p>
    <div class="matrix-wrap">
      <table class="matrix-table">
        <thead>
          <tr>
            <th>#</th>
            <th>题号 (case_id) & 标题</th>
            <th>类型</th>
            <th>通过数</th>
            {''.join(f'<th>{html.escape(m)}</th>' for m in MODELS)}
          </tr>
        </thead>
        <tbody>
          {''.join(matrix_tr_html)}
        </tbody>
      </table>
    </div>
  </div>

  <div class="filter-bar">
    <div class="filter-group">
      <strong>筛选题型：</strong>
      <button type="button" class="filter-btn active" data-filter="all" onclick="filterTask('all', this)">全部 40 题</button>
      <button type="button" class="filter-btn" data-filter="audit" onclick="filterTask('audit', this)">第一部分：挑错题 (20 题)</button>
      <button type="button" class="filter-btn" data-filter="implementation" onclick="filterTask('implementation', this)">第二部分：写代码题 (20 题)</button>
    </div>
    <div class="filter-group">
      <label for="globalModelSelect"><strong>统一锁定查看模型页签：</strong></label>
      <select id="globalModelSelect" class="model-select" onchange="switchAllModelTabs(this.value)">
        <option value="">-- 保持各题当前页签 --</option>
        {''.join(f'<option value="{html.escape(m)}">{html.escape(m)}</option>' for m in MODELS)}
      </select>
      <input type="text" id="caseSearchInput" class="search-input" placeholder="搜索题号 / 标题 / 关键词..." oninput="applyFilters()">
    </div>
  </div>

  <div id="cardsContainer">
    {''.join(cards_html)}
  </div>

</div>

<script>
  let currentTaskFilter = 'all';

  function filterTask(task, btn) {{
    currentTaskFilter = task;
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    applyFilters();
  }}

  function applyFilters() {{
    const q = (document.getElementById('caseSearchInput').value || '').toLowerCase().trim();
    document.querySelectorAll('.case-card').forEach(card => {{
      const t = card.getAttribute('data-task');
      const s = card.getAttribute('data-search') || '';
      const taskOk = (currentTaskFilter === 'all' || t === currentTaskFilter);
      const searchOk = (!q || s.includes(q));
      card.style.display = (taskOk && searchOk) ? 'block' : 'none';
    }});
    document.querySelectorAll('.matrix-row').forEach(row => {{
      const t = row.getAttribute('data-task');
      const taskOk = (currentTaskFilter === 'all' || t === currentTaskFilter);
      row.style.display = taskOk ? '' : 'none';
    }});
  }}

  function switchModelTab(caseId, modelName) {{
    const tabsWrap = document.getElementById('tabs-' + caseId);
    const panelsWrap = document.getElementById('panels-' + caseId);
    if (!tabsWrap || !panelsWrap) return;
    tabsWrap.querySelectorAll('.model-tab-btn').forEach(btn => {{
      btn.classList.toggle('active', btn.getAttribute('data-model') === modelName);
    }});
    panelsWrap.querySelectorAll('.model-panel').forEach(panel => {{
      panel.classList.toggle('active', panel.getAttribute('data-model') === modelName);
    }});
  }}

  function switchAllModelTabs(modelName) {{
    if (!modelName) return;
    document.querySelectorAll('.case-card').forEach(card => {{
      const cid = card.id.replace('card-', '');
      switchModelTab(cid, modelName);
    }});
  }}

  function jumpToCase(caseId, modelName) {{
    const card = document.getElementById('card-' + caseId);
    if (!card) return;
    if (card.style.display === 'none') {{
      currentTaskFilter = 'all';
      document.getElementById('caseSearchInput').value = '';
      document.querySelectorAll('.filter-btn').forEach(b => b.classList.toggle('active', b.getAttribute('data-filter') === 'all'));
      applyFilters();
    }}
    if (modelName) {{
      switchModelTab(caseId, modelName);
    }}
    card.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
  }}
</script>
</body>
</html>
"""
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(page_html, encoding="utf-8")


def main() -> None:
    out_dir = REPO_ROOT / "docs/experiments/layer1_canonical"
    runs_path = out_dir / "runs.jsonl"
    xlsx_path = out_dir / "LAYER1_FULL_RESULTS.xlsx"
    wide_csv_path = out_dir / "full_qa_wide_40cases.csv"
    long_csv_path = out_dir / "full_qa_long_320runs.csv"
    html_path = REPO_ROOT / "docs/LAYER1_FULL_EXPERIMENT_TABLE.html"

    runs_by_mc = load_canonical_runs(runs_path)
    expected_pairs = (len(CANONICAL_AUDIT_CASES) + len(CANONICAL_IMPL_CASES)) * len(MODELS)
    if len(runs_by_mc) != expected_pairs:
        raise RuntimeError(f"Expected {expected_pairs} (model, case_id) runs, found {len(runs_by_mc)}")

    data = build_all_datasets(runs_by_mc)

    # 1. Write CSVs
    pd.DataFrame(data["wide_40_rows"]).to_csv(wide_csv_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(data["long_320_rows"]).to_csv(long_csv_path, index=False, encoding="utf-8-sig")

    # 2. Write Excel Workbook
    write_excel_workbook(data, xlsx_path)

    # 3. Write Standalone Interactive HTML Table
    write_html_page(data, html_path)

    print(f"[OK] Wrote Excel workbook : {xlsx_path} ({xlsx_path.stat().st_size / 1024:.1f} KB)")
    print(f"[OK] Wrote Wide 40 CSV    : {wide_csv_path} ({len(data['wide_40_rows'])} rows)")
    print(f"[OK] Wrote Long 320 CSV   : {long_csv_path} ({len(data['long_320_rows'])} rows)")
    print(f"[OK] Wrote HTML Data Table: {html_path} ({html_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
