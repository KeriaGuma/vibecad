from __future__ import annotations

import pytest

from app.agent_eval import load_agent_eval_dataset, run_agent_eval
from app.models import AgentEvalRequest


def test_agent_eval_dataset_is_versioned_and_covers_core_categories():
    version, cases = load_agent_eval_dataset()

    assert version == "agent-tasks-v1.1"
    assert len(cases) == 12
    assert {case.category for case in cases} == {
        "planning",
        "editing",
        "semantic",
        "safety",
        "recovery",
        "export",
    }
    assert len({case.id for case in cases}) == len(cases)


def test_deterministic_agent_eval_baseline_passes_all_cases():
    report = run_agent_eval(AgentEvalRequest(mode="deterministic"))

    assert report.case_count == 12
    assert report.passed_count == 12
    assert report.metrics.task_success_rate == 1
    assert report.metrics.tool_selection_precision == 1
    assert report.metrics.tool_selection_recall == 1
    assert report.metrics.argument_accuracy == 1
    assert report.metrics.assertion_accuracy == 1
    assert report.metrics.rollback_success_rate == 1
    assert report.metrics.invalid_action_rate > 0
    assert report.metrics.average_policy_injected_steps > 0
    assert all(case.passed for case in report.cases)
    assert all(case.task_success for case in report.cases)


def test_agent_eval_supports_case_selection_and_rejects_unknown_ids():
    report = run_agent_eval(
        AgentEvalRequest(
            mode="deterministic",
            case_ids=["semantic_drive_dimension", "safety_locked_reference"],
        )
    )

    assert [case.case_id for case in report.cases] == [
        "semantic_drive_dimension",
        "safety_locked_reference",
    ]

    with pytest.raises(ValueError, match="Unknown agent eval cases"):
        run_agent_eval(AgentEvalRequest(case_ids=["missing_case"]))


def test_agent_eval_api_persists_report_without_mutating_project_ir(client):
    created = client.post(
        "/api/projects",
        json={"name": "eval host", "prompt": "创建 100 60 8 两个孔"},
    ).json()
    project_id = created["project_id"]
    original_ir = created["ir"]

    dataset = client.get("/api/agent/evals/dataset")
    assert dataset.status_code == 200
    assert dataset.json()["version"] == "agent-tasks-v1.1"

    response = client.post(
        f"/api/projects/{project_id}/agent/evals",
        json={"mode": "deterministic", "max_cases": 4},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["report"]["case_count"] == 4
    assert body["report"]["passed_count"] == 4
    assert body["project"]["ir"] == original_ir
    assert body["project"]["agent_eval_reports"][-1]["id"] == body["report"]["id"]
