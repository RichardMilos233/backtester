"""Offline regression tests for G4's role context and bounded review loop."""
import pytest

from agent.llm import BaseLLMClient
from agent.layer1.harness import G4ImplementerSupervisor, HOUSE_CONVENTIONS, get_harness
from agent.layer1.systematic import SYSTEMATIC_SYSTEM_PROMPT
from scripts.run_harness_impl import _load_done


class ScriptedClient(BaseLLMClient):
    def __init__(self, response, calls):
        self.response = response
        self.calls = calls

    def generate(self, prompt, system_prompt=""):
        self.calls.append((prompt, system_prompt))
        self.record(prompt_tokens=10, completion_tokens=20, latency=0)
        return self.response


def run_script(responses, **kwargs):
    calls = []
    pending = iter(responses)

    def factory(temperature):
        return ScriptedClient(next(pending), calls)

    out = G4ImplementerSupervisor(**kwargs).run(
        "PUBLIC_TASK", factory, context={"pinned_spec": "PRIVATE_SPEC_SENTINEL"}
    )
    return out, calls


def test_supervisor_receives_same_environment_as_implementer():
    out, calls = run_script(["DRAFT", "APPROVED"])
    assert SYSTEMATIC_SYSTEM_PROMPT in calls[1][1]
    assert HOUSE_CONVENTIONS in calls[1][1]
    assert calls[1][1].startswith(calls[0][1])
    assert out.chosen == "DRAFT"
    assert out.llm_calls == 2


def test_revision_sees_draft_and_feedback_and_is_reviewed_again():
    out, calls = run_script(["DRAFT_V1", "FIX_EXIT", "DRAFT_V2", "APPROVED"])
    assert "DRAFT_V1" in calls[2][0]
    assert "FIX_EXIT" in calls[2][0]
    assert "DRAFT_V2" in calls[3][0]
    assert "FIX_EXIT" in calls[3][0]
    assert out.chosen == "DRAFT_V2"
    assert out.trace["approved"] is True
    assert out.trace["stop_reason"] == "approved"
    assert len(out.trace["reviews"]) == 2
    assert out.llm_calls == 4
    assert out.prompt_tokens == 40
    assert out.completion_tokens == 80


def test_revision_limit_still_reviews_latest_code_and_marks_unapproved():
    out, calls = run_script(
        ["DRAFT_0", "ISSUE_0", "DRAFT_1", "ISSUE_1", "DRAFT_2", "ISSUE_2"]
    )
    assert len(calls) == 6
    assert out.chosen == "DRAFT_2"
    assert out.trace["approved"] is False
    assert out.trace["stop_reason"] == "revision_limit"
    assert out.trace["revisions"] == 2
    assert [r["draft"] for r in out.trace["reviews"]] == ["DRAFT_0", "DRAFT_1", "DRAFT_2"]


def test_approval_requires_exact_marker():
    out, calls = run_script(["DRAFT_0", "APPROVED, except fix the exit", "DRAFT_1", "APPROVED"])
    assert len(calls) == 4
    assert out.chosen == "DRAFT_1"


def test_no_private_spec_and_no_conventions_when_disabled():
    out, calls = run_script(["DRAFT", "APPROVED"], with_conventions=False)
    assert all("PRIVATE_SPEC_SENTINEL" not in prompt + system for prompt, system in calls)
    assert all(HOUSE_CONVENTIONS not in system for _, system in calls)
    assert all(SYSTEMATIC_SYSTEM_PROMPT in system for _, system in calls)


def test_g4_alias_and_explicit_version_identify_fixed_harness():
    assert get_harness("G4").version == "g4_v2"
    assert get_harness("G4_v2").version == "g4_v2"


def test_resume_does_not_reuse_legacy_g4_results(tmp_path):
    import json

    path = tmp_path / "runs.jsonl"
    rows = [
        {"harness": "G4", "variant": "original", "model": "m", "case_id": "old"},
        {"harness": "G4", "variant": "original", "model": "m", "case_id": "new", "harness_version": "g4_v2"},
        {"harness": "G0", "variant": "original", "model": "m", "case_id": "baseline"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError, match="version"):
        _load_done(path)
    path.write_text("\n".join(json.dumps(r) for r in rows[1:]))
    done = _load_done(path)
    assert ("G4", "original", "m", "new") in done
    assert ("G0", "original", "m", "baseline") in done


def test_negative_revision_limit_is_rejected():
    with pytest.raises(ValueError):
        G4ImplementerSupervisor(max_revisions=-1)
