from __future__ import annotations

from datetime import datetime, timezone

from app.dimension_benchmark import evaluate_dimension_benchmark
from app.models import (
    DimensionBinding,
    DimensionGroundTruth,
    DrawingIR,
    LineEntity,
    MechanicalDimensionObject,
    MechanicalDrawingIR,
    ParsedDimensionValue,
    PolylineEntity,
    ProjectState,
    RectangleEntity,
    SemanticRepairRequest,
)
from app.semantic_repair import run_semantic_repair_agent


def _repair_project(project_id: str = "semantic_repair") -> ProjectState:
    now = datetime.now(timezone.utc)
    parsed = ParsedDimensionValue(kind="linear", raw_text="244", nominal=244)
    binding = DimensionBinding(
        id="binding_244",
        dimension_line_id="wrong_line",
        text_id="ocr_244",
        text="244",
        parsed=parsed,
        confidence=0.88,
        kind="linear",
        line_x1=100,
        line_y1=110,
        line_x2=108,
        line_y2=118,
    )
    dimension = MechanicalDimensionObject(
        id="dimension_244",
        binding_id=binding.id,
        kind="linear",
        text="244",
        parsed=parsed,
        confidence=0.88,
        dimension_line_id="wrong_line",
        text_id="ocr_244",
        status="partial",
    )
    return ProjectState(
        project_id=project_id,
        name="semantic repair fixture",
        created_at=now,
        updated_at=now,
        ir=DrawingIR(
            entities=[
                RectangleEntity(id="sheet", layer="TITLE_BLOCK", x=0, y=0, width=300, height=200),
                LineEntity(
                    id="correct_dimension_line",
                    layer="OUTLINE",
                    x1=20,
                    y1=60,
                    x2=220,
                    y2=60,
                    group="promoted_geometry",
                ),
                LineEntity(
                    id="left_surface",
                    layer="OUTLINE",
                    x1=20,
                    y1=90,
                    x2=20,
                    y2=130,
                    group="promoted_geometry",
                ),
                LineEntity(
                    id="right_surface",
                    layer="OUTLINE",
                    x1=220,
                    y1=90,
                    x2=220,
                    y2=130,
                    group="promoted_geometry",
                ),
                LineEntity(id="wrong_line", layer="OUTLINE", x1=100, y1=110, x2=108, y2=118),
            ]
        ),
        dimension_bindings=[binding],
        mechanical_dimensions=[dimension.model_copy(deep=True)],
        mechanical_ir=MechanicalDrawingIR(dimensions=[dimension]),
        dimension_ground_truth=[
            DimensionGroundTruth(
                id="gt_linear_244",
                label="总长 244",
                expected_text="244",
                kind="linear",
                nominal=244,
                source="seed",
            )
        ],
    )


def test_semantic_repair_closes_linear_dimension_with_monotonic_eval():
    project = _repair_project()
    before = evaluate_dimension_benchmark(project)

    repaired, report, run = run_semantic_repair_agent(
        project,
        SemanticRepairRequest(use_llm=False, max_steps=1),
    )

    assert before.complete_count == 0
    assert report.complete_count == 1
    assert report.overall_score > before.overall_score
    assert run.planner_source == "deterministic"
    assert run.accepted_steps == 1
    assert run.stopped_reason == "repair_pass_complete"
    assert run.steps[0].status == "accepted"
    dimension = repaired.mechanical_ir.dimensions[0]
    assert dimension.dimension_line_id == "correct_dimension_line"
    assert set(dimension.extension_line_ids) == {
        "auto_extension_gt_linear_244_start",
        "auto_extension_gt_linear_244_end",
    }
    assert set(dimension.measured_geometry_ids) == {"left_surface", "right_surface"}
    assert dimension.export_ready is True
    arrows = [
        entity
        for entity in repaired.ir.entities
        if isinstance(entity, PolylineEntity) and "auto_repair" in entity.tags
    ]
    assert len(arrows) == 2
    assert all(entity.closed and entity.metadata["solid_fill"] for entity in arrows)


def test_semantic_repair_uses_llm_only_for_target_order(monkeypatch):
    from app import semantic_repair

    monkeypatch.setattr(
        semantic_repair,
        "plan_dimension_repair_order_llm",
        lambda report, ids, max_steps: (["gt_linear_244"], "优先修复已匹配总长", "deepseek-v4-flash"),
    )

    _, _, run = run_semantic_repair_agent(
        _repair_project(),
        SemanticRepairRequest(use_llm=True, max_steps=1),
    )

    assert run.planner_source == "deepseek"
    assert run.planner_model == "deepseek-v4-flash"
    assert run.llm_calls == 1


def test_semantic_repair_api_and_snapshot_rollback(client):
    from app.storage import load_project, save_project

    project_id = client.post("/api/projects", json={"name": "repair", "prompt": ""}).json()["project_id"]
    fixture = _repair_project(project_id)
    save_project(fixture)

    response = client.post(
        f"/api/projects/{project_id}/agent/dimensions/repair",
        json={"use_llm": False, "max_steps": 1, "min_gain": 0.01},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["report"]["complete_count"] == 1
    assert body["run"]["accepted_steps"] == 1
    run_id = body["run"]["id"]
    stored = load_project(project_id)
    assert stored.semantic_repair_runs[-1].snapshot_file

    rollback = client.post(
        f"/api/projects/{project_id}/agent/dimensions/repair/{run_id}/rollback"
    )

    assert rollback.status_code == 200
    restored = rollback.json()
    assert restored["report"]["complete_count"] == 0
    assert restored["run"]["rolled_back_at"] is not None


def test_semantic_repair_requires_ground_truth(client):
    project_id = client.post("/api/projects", json={"name": "repair", "prompt": ""}).json()["project_id"]

    response = client.post(
        f"/api/projects/{project_id}/agent/dimensions/repair",
        json={"use_llm": False},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Dimension ground truth is not initialized"
