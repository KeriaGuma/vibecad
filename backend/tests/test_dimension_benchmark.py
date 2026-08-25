from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.dimension_benchmark import (
    apply_dimension_correction,
    default_output_shaft_targets,
    evaluate_dimension_benchmark,
    seed_dimension_ground_truth,
)
from app.models import (
    DimensionCorrectionRequest,
    DimensionGroundTruth,
    DrawingIR,
    LineEntity,
    MechanicalDrawingIR,
    ProjectState,
    TextEntity,
)


def _project() -> ProjectState:
    now = datetime.now(timezone.utc)
    return ProjectState(
        project_id="dimension_benchmark",
        name="dimension benchmark",
        created_at=now,
        updated_at=now,
        ir=DrawingIR(
            entities=[
                LineEntity(id="outline", layer="OUTLINE", x1=0, y1=0, x2=244, y2=0),
                LineEntity(id="ext_start", layer="DIMENSION", x1=0, y1=0, x2=0, y2=20),
                LineEntity(id="ext_end", layer="DIMENSION", x1=244, y1=0, x2=244, y2=20),
                LineEntity(id="dim_line", layer="DIMENSION", x1=0, y1=20, x2=244, y2=20),
                LineEntity(id="arrow_start", layer="DIMENSION", x1=0, y1=20, x2=4, y2=22),
                LineEntity(id="arrow_end", layer="DIMENSION", x1=244, y1=20, x2=240, y2=22),
                TextEntity(id="dim_text", layer="TEXT", x=119, y=23, text="244"),
            ]
        ),
        mechanical_ir=MechanicalDrawingIR(),
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


def _correction(**updates) -> DimensionCorrectionRequest:
    payload = {
        "ground_truth_id": "gt_linear_244",
        "text_id": "dim_text",
        "dimension_line_id": "dim_line",
        "arrow_entity_ids": ["arrow_start", "arrow_end"],
        "extension_line_ids": ["ext_start", "ext_end"],
        "measured_geometry_ids": ["outline"],
    }
    payload.update(updates)
    return DimensionCorrectionRequest(**payload)


def test_seed_default_output_shaft_ground_truth():
    seeded = seed_dimension_ground_truth(_project(), replace=True)

    assert len(seeded.dimension_ground_truth) == 10
    assert {target.expected_text for target in seeded.dimension_ground_truth} >= {"244", "φ65", "φ176"}
    assert len(default_output_shaft_targets()) == 10


def test_manual_correction_completes_dimension_semantic_loop():
    project = _project()
    before = evaluate_dimension_benchmark(project)

    assert before.matched_count == 0
    assert before.complete_count == 0

    corrected = apply_dimension_correction(project, _correction())
    report = evaluate_dimension_benchmark(corrected)

    assert report.matched_count == 1
    assert report.complete_count == 1
    assert report.overall_score == 1.0
    assert all(value == 1.0 for value in report.metrics.values())
    dimension = corrected.mechanical_ir.dimensions[0]
    assert dimension.status == "complete"
    assert dimension.export_ready is True
    assert dimension.dxf_dimension_type == "linear"
    assert dimension.measurement_points == [[0.0, 0.0], [244.0, 0.0]]
    assert dimension.extension_line_ids == ["ext_start", "ext_end"]
    assert dimension.measured_geometry_ids == ["outline"]
    assert corrected.dimension_corrections[0].ground_truth_id == "gt_linear_244"


def test_manual_correction_rejects_wrong_entity_role():
    with pytest.raises(ValueError, match="Invalid dimension line entity"):
        apply_dimension_correction(_project(), _correction(dimension_line_id="dim_text"))


def test_benchmark_does_not_match_similar_but_wrong_nominal():
    corrected = apply_dimension_correction(_project(), _correction())
    corrected.dimension_ground_truth = [
        DimensionGroundTruth(
            id="gt_linear_127",
            label="轴段长度 127",
            expected_text="127",
            kind="linear",
            nominal=127,
            source="seed",
        )
    ]

    report = evaluate_dimension_benchmark(corrected)

    assert report.matched_count == 0


def test_manual_correction_materializes_missing_text_entity():
    corrected = apply_dimension_correction(_project(), _correction(text_id=None))

    report = evaluate_dimension_benchmark(corrected)

    assert report.complete_count == 1
    assert report.metrics["text"] == 1.0
    text = next(entity for entity in corrected.ir.entities if entity.id == "manual_text_gt_linear_244")
    assert text.type == "text"
    assert text.text == "244"


def test_dimension_benchmark_api_round_trip(client):
    from app.storage import load_project, save_project

    project_id = client.post("/api/projects", json={"name": "benchmark", "prompt": ""}).json()["project_id"]
    project = load_project(project_id)
    fixture = _project()
    project.ir = fixture.ir
    project.mechanical_ir = fixture.mechanical_ir
    project.mechanical_dimensions = []
    save_project(project)

    seeded = client.post(
        f"/api/projects/{project_id}/benchmark/dimensions/seed",
        json={"targets": [fixture.dimension_ground_truth[0].model_dump(mode="json")], "replace": True},
    )
    assert seeded.status_code == 200
    assert seeded.json()["report"]["target_count"] == 1

    corrected = client.put(
        f"/api/projects/{project_id}/benchmark/dimensions/correction",
        json=_correction().model_dump(mode="json"),
    )
    assert corrected.status_code == 200
    body = corrected.json()
    assert body["report"]["complete_count"] == 1
    assert body["project"]["mechanical_ir"]["dimensions"][0]["export_ready"] is True

    evaluated = client.get(f"/api/projects/{project_id}/eval/dimensions")
    assert evaluated.status_code == 200
    assert evaluated.json()["report"]["overall_score"] == 1.0


def test_dimension_benchmark_api_returns_400_for_invalid_selection(client):
    project_id = client.post("/api/projects", json={"name": "benchmark", "prompt": ""}).json()["project_id"]
    fixture = _project()
    client.post(
        f"/api/projects/{project_id}/benchmark/dimensions/seed",
        json={"targets": [fixture.dimension_ground_truth[0].model_dump(mode="json")], "replace": True},
    )

    response = client.put(
        f"/api/projects/{project_id}/benchmark/dimensions/correction",
        json=_correction(dimension_line_id="missing").model_dump(mode="json"),
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid dimension line entity: missing"
