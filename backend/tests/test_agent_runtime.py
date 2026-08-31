from __future__ import annotations

from datetime import datetime, timezone

from app.agent_runtime import run_agent_task
from app.agent_tools import build_default_tool_registry
from app.models import (
    AgentPlannedStep,
    AgentTaskRequest,
    AgentToolArguments,
    ProjectState,
    default_ir,
)


def _project(project_id: str = "agent_runtime") -> ProjectState:
    now = datetime.now(timezone.utc)
    return ProjectState(
        project_id=project_id,
        name="agent runtime",
        created_at=now,
        updated_at=now,
        ir=default_ir(),
    )


def _entity(project, entity_id):
    return next(entity for entity in project.ir.entities if entity.id == entity_id)


def test_tool_registry_exposes_typed_contracts():
    registry = build_default_tool_registry()
    definitions = {tool.name: tool for tool in registry.definitions()}

    assert set(definitions) == {
        "inspect_drawing",
        "evaluate_drawing",
        "evaluate_dimensions",
        "repair_dimensions",
        "drive_dimension",
        "edit_cad",
        "export_dxf",
    }
    assert definitions["drive_dimension"].mutating is True
    assert definitions["drive_dimension"].reversible is True
    assert definitions["drive_dimension"].parameters["properties"]["target_value"]["type"] == "number"
    assert definitions["inspect_drawing"].validator == "read_only"
    assert definitions["evaluate_drawing"].validator == "drawing_integrity"


def test_deterministic_task_executes_edit_and_export():
    project, run = run_agent_task(
        _project(),
        AgentTaskRequest(goal="把左边孔直径改成 10 并导出 DXF", use_llm=False),
    )

    assert run.status == "completed"
    assert [step.tool for step in run.steps] == [
        "inspect_drawing",
        "edit_cad",
        "evaluate_drawing",
        "export_dxf",
    ]
    assert all(step.status == "accepted" for step in run.steps)
    assert _entity(project, "hole_1").r == 5
    assert run.artifacts["dxf_url"].endswith("/output.dxf")
    assert run.llm_calls == 0
    assert run.policy_injected_steps == 1


def test_ambiguous_generic_edit_requests_one_focused_clarification():
    initial = _project()
    project, run = run_agent_task(
        initial,
        AgentTaskRequest(goal="把孔直径改成 10", use_llm=True),
    )

    assert run.status == "needs_clarification"
    assert run.llm_calls == 0
    assert [step.tool for step in run.steps] == ["inspect_drawing"]
    assert project.ir == initial.ir
    assert run.clarification is not None
    assert any(label.startswith("hole_1") for label in run.clarification.candidates)
    assert any(label.startswith("hole_2") for label in run.clarification.candidates)
    assert "确认" in run.summary


def test_export_task_round_trips_real_artifacts_before_reporting_success():
    _, run = run_agent_task(
        _project(),
        AgentTaskRequest(goal="检查当前图纸并导出 DXF，不修改几何", use_llm=False),
    )

    export_step = next(step for step in run.steps if step.tool == "export_dxf")
    assert export_step.validation["validator"] == "export_artifacts_round_trip"
    assert export_step.output["dxf_entity_count"] > 0
    assert export_step.output["dxf_bytes"] > 0
    assert export_step.output["svg_bytes"] > 0
    assert export_step.output["svg_root_verified"] is True


def test_failed_mutation_is_rolled_back_and_deepseek_replans(monkeypatch):
    from app import agent_runtime

    calls = 0

    def plan(goal, project, catalog, max_tool_calls, execution_context=None):
        nonlocal calls
        calls += 1
        if execution_context is None:
            return (
                [
                    AgentPlannedStep(
                        call_id="bad_edit",
                        tool="edit_cad",
                        arguments=AgentToolArguments(message="删除 missing_999"),
                    )
                ],
                "try requested edit",
                "deepseek-v4-flash",
            )
        assert execution_context[-1]["status"] == "rolled_back"
        return (
            [
                AgentPlannedStep(
                    call_id="replanned_edit",
                    tool="edit_cad",
                    arguments=AgentToolArguments(message=goal),
                )
            ],
            "use a valid deterministic edit tool call",
            "deepseek-v4-flash",
        )

    monkeypatch.setattr(agent_runtime, "plan_agent_task_llm", plan)
    project, run = run_agent_task(
        _project(),
        AgentTaskRequest(goal="添加孔 50 30 6", use_llm=True, max_replans=1),
    )

    assert calls == 2
    assert run.status == "completed"
    assert run.replan_count == 1
    assert run.llm_calls == 2
    assert [step.tool for step in run.steps] == [
        "edit_cad",
        "evaluate_drawing",
        "edit_cad",
        "evaluate_drawing",
    ]
    assert run.steps[0].status == "rolled_back"
    assert all(step.status == "accepted" for step in run.steps[1:])
    assert run.policy_injected_steps >= 2
    assert len([entity for entity in project.ir.entities if entity.type == "circle"]) == 3


def test_failed_post_mutation_evaluator_rolls_back_the_accepted_edit(monkeypatch):
    from app.agent_tools import AgentToolError

    registry = build_default_tool_registry()
    execute = registry.execute
    evaluator_calls = 0

    def fail_evaluator(name, project, arguments, context):
        nonlocal evaluator_calls
        if name == "evaluate_drawing":
            evaluator_calls += 1
            if evaluator_calls == 1:
                raise AgentToolError("forced drawing integrity failure")
        return execute(name, project, arguments, context)

    monkeypatch.setattr(registry, "execute", fail_evaluator)
    initial = _project()
    project, run = run_agent_task(
        initial,
        AgentTaskRequest(goal="把左边孔直径改成 10", use_llm=False),
        registry,
    )

    assert run.status == "partial"
    assert [step.status for step in run.steps] == ["accepted", "rolled_back", "error", "accepted"]
    assert project.ir == initial.ir
    assert run.steps[1].validation["post_validation_passed"] is False


def test_tool_budget_never_allows_an_unvalidated_mutation():
    initial = _project()
    project, run = run_agent_task(
        initial,
        AgentTaskRequest(
            goal="把左边孔直径改成 10",
            use_llm=False,
            max_tool_calls=2,
        ),
    )

    assert run.status == "partial"
    assert [step.tool for step in run.steps] == ["inspect_drawing"]
    assert project.ir == initial.ir


def test_runtime_policy_adds_missing_read_only_evaluator(monkeypatch):
    from app import agent_runtime

    def plan(*args, **kwargs):
        del args, kwargs
        return (
            [AgentPlannedStep(call_id="inspect", tool="inspect_drawing")],
            "inspect only",
            "deepseek-v4-flash",
        )

    monkeypatch.setattr(agent_runtime, "plan_agent_task_llm", plan)
    _, run = run_agent_task(
        _project(),
        AgentTaskRequest(goal="检查当前图纸，不修改几何", use_llm=True),
    )

    assert run.status == "completed"
    assert [step.tool for step in run.steps] == ["inspect_drawing", "evaluate_dimensions"]
    assert run.policy_injected_steps == 1


def test_runtime_policy_canonicalizes_generic_edit_message(monkeypatch):
    from app import agent_runtime

    def plan(*args, **kwargs):
        del args, kwargs
        return (
            [
                AgentPlannedStep(call_id="inspect", tool="inspect_drawing"),
                AgentPlannedStep(
                    call_id="edit",
                    tool="edit_cad",
                    arguments=AgentToolArguments(message="把右孔向右移动 12"),
                ),
            ],
            "edit with paraphrased arguments",
            "deepseek-v4-flash",
        )

    monkeypatch.setattr(agent_runtime, "plan_agent_task_llm", plan)
    _, run = run_agent_task(
        _project(),
        AgentTaskRequest(goal="把右边孔右移 12", use_llm=True),
    )

    assert run.status == "completed"
    assert run.steps[1].arguments.message == "把右边孔右移 12"
    assert [step.tool for step in run.steps] == [
        "inspect_drawing",
        "edit_cad",
        "evaluate_drawing",
    ]


def test_unknown_goal_is_not_reported_as_completed():
    _, run = run_agent_task(
        _project(),
        AgentTaskRequest(goal="请施展一种不存在的操作", use_llm=False),
    )

    assert run.status == "partial"
    assert run.steps[0].tool == "inspect_drawing"


def test_read_only_negation_does_not_require_an_edit():
    _, run = run_agent_task(
        _project(),
        AgentTaskRequest(goal="检查图纸并导出 DXF，不修改几何", use_llm=False),
    )

    assert run.status == "completed"
    assert [step.tool for step in run.steps] == ["inspect_drawing", "evaluate_dimensions", "export_dxf"]
    assert all(step.mutating is False for step in run.steps)


def test_agent_task_api_catalog_snapshot_and_rollback(client):
    created = client.post(
        "/api/projects",
        json={"name": "agent", "prompt": "创建 100 60 8 两个孔"},
    ).json()
    project_id = created["project_id"]

    catalog = client.get("/api/agent/tools")
    assert catalog.status_code == 200
    assert len(catalog.json()["tools"]) == 7

    response = client.post(
        f"/api/projects/{project_id}/agent/tasks",
        json={
            "goal": "把左边孔直径改成 10 并导出 DXF",
            "use_llm": False,
            "max_tool_calls": 6,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["run"]["status"] == "completed"
    assert body["run"]["snapshot_file"]
    assert next(entity for entity in body["project"]["ir"]["entities"] if entity["id"] == "hole_1")["r"] == 5

    run_id = body["run"]["id"]
    rollback = client.post(f"/api/projects/{project_id}/agent/tasks/{run_id}/rollback")

    assert rollback.status_code == 200
    restored = rollback.json()
    assert restored["run"]["status"] == "rolled_back"
    assert next(entity for entity in restored["project"]["ir"]["entities"] if entity["id"] == "hole_1")["r"] == 4
