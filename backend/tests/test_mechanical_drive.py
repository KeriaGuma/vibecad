from __future__ import annotations

from datetime import datetime, timezone

import ezdxf
import pytest

from app.exporter import export_dxf
from app.mechanical_drive import (
    MechanicalDriveError,
    execute_mechanical_operation,
    plan_mechanical_drive_deterministic,
    undo_last_mechanical_transaction,
)
from app.models import (
    CircleEntity,
    DimensionBinding,
    DrawingIR,
    Layer,
    LineEntity,
    MechanicalArrowhead,
    MechanicalDimensionObject,
    MechanicalDrawingIR,
    MechanicalOperation,
    ParsedDimensionValue,
    PolylineEntity,
    ProjectState,
    TextEntity,
)


def _linear_project(*, measured_layer: str = "OUTLINE") -> ProjectState:
    parsed = ParsedDimensionValue(kind="linear", raw_text="244", nominal=244)
    binding = DimensionBinding(
        id="dim_binding_244",
        dimension_line_id="dim_line_244",
        arrow_ids=["arrow_source_start", "arrow_source_end"],
        text_id="dim_text_244",
        text="244",
        parsed=parsed,
        confidence=0.98,
        kind="linear",
        line_x1=0,
        line_y1=20,
        line_x2=244,
        line_y2=20,
        text_x=120,
        text_y=23,
    )
    dimension = MechanicalDimensionObject(
        id="mechanical_dimension_dim_binding_244",
        binding_id=binding.id,
        kind="linear",
        text="244",
        parsed=parsed,
        confidence=0.97,
        dimension_line_id=binding.dimension_line_id,
        text_id=binding.text_id,
        arrowheads=[
            MechanicalArrowhead(
                candidate_id="a0",
                source_entity_id="arrow_source_start",
                render_entity_id="arrow_render_start",
                tip_x=0,
                tip_y=20,
                direction_x=1,
                direction_y=0,
                score=0.95,
                endpoint="start",
                endpoint_distance=0,
            ),
            MechanicalArrowhead(
                candidate_id="a1",
                source_entity_id="arrow_source_end",
                render_entity_id="arrow_render_end",
                tip_x=244,
                tip_y=20,
                direction_x=-1,
                direction_y=0,
                score=0.95,
                endpoint="end",
                endpoint_distance=0,
            ),
        ],
        extension_line_ids=["extension_start", "extension_end"],
        measured_geometry_ids=["measured_outline"],
        target_geometry_ids=["measured_outline"],
        measurement_points=[[0, 0], [244, 0]],
        dimension_line_point=[122, 20],
        orientation="horizontal",
        dxf_dimension_type="linear",
        export_ready=True,
        status="complete",
    )
    layers = [
        Layer(name="OUTLINE", lineweight=0.5),
        Layer(name="DIMENSION", lineweight=0.18),
        Layer(name="REFERENCE_TRACE", locked=True, editable=False),
    ]
    ir = DrawingIR(
        layers=layers,
        entities=[
            LineEntity(id="measured_outline", layer=measured_layer, x1=0, y1=0, x2=244, y2=0),
            LineEntity(id="extension_start", layer="DIMENSION", x1=0, y1=0, x2=0, y2=21),
            LineEntity(id="extension_end", layer="DIMENSION", x1=244, y1=0, x2=244, y2=21),
            LineEntity(id="dim_line_244", layer="DIMENSION", x1=0, y1=20, x2=244, y2=20),
            PolylineEntity(
                id="arrow_render_start",
                layer="DIMENSION",
                points=[[0, 20], [3, 19], [3, 21]],
                closed=True,
            ),
            PolylineEntity(
                id="arrow_render_end",
                layer="DIMENSION",
                points=[[244, 20], [241, 19], [241, 21]],
                closed=True,
            ),
            TextEntity(id="dim_text_244", layer="DIMENSION", x=120, y=23, text="244", height=2.5),
        ],
    )
    now = datetime.now(timezone.utc)
    return ProjectState(
        project_id="drive_test",
        name="drive",
        created_at=now,
        updated_at=now,
        ir=ir,
        dimension_bindings=[binding],
        mechanical_dimensions=[dimension.model_copy(deep=True)],
        mechanical_ir=MechanicalDrawingIR(dimensions=[dimension.model_copy(deep=True)]),
    )


def _entity(project, entity_id):
    return next(entity for entity in project.ir.entities if entity.id == entity_id)


def test_linear_dimension_drive_changes_geometry_semantics_and_native_dxf(tmp_path):
    project = _linear_project()
    plan = plan_mechanical_drive_deterministic("把 244 改成 250", project)

    assert plan is not None
    result = execute_mechanical_operation(project, plan, "把 244 改成 250")

    assert _entity(result.project, "measured_outline").x2 == 250
    assert _entity(result.project, "extension_end").x1 == 250
    assert _entity(result.project, "dim_line_244").x2 == 250
    assert _entity(result.project, "arrow_render_end").points[0] == [250, 20]
    assert _entity(result.project, "dim_text_244").text == "250"
    assert _entity(result.project, "dim_text_244").x == 123
    dimension = result.project.mechanical_ir.dimensions[0]
    assert dimension.measurement_points == [[0.0, 0.0], [250.0, 0.0]]
    assert dimension.measured_value == 250
    assert dimension.edit_mode == "driving"
    assert dimension.validation_status == "passed"
    assert result.validation.passed is True
    assert result.project.mechanical_transactions[-1].validation.measured_value == 250
    assert "244 → 250" in result.reply

    path = tmp_path / "driven.dxf"
    export_dxf(result.project.ir, path, result.project.mechanical_ir)
    native = list(ezdxf.readfile(path).modelspace().query("DIMENSION"))[0]
    assert native.dxf.text == "250"
    assert native.get_measurement() == pytest.approx(250)


def test_mechanical_transaction_undo_restores_geometry_and_semantics():
    project = _linear_project()
    plan = plan_mechanical_drive_deterministic("把244改成250", project)
    result = execute_mechanical_operation(project, plan, "把244改成250")

    restored, reply, diffs = undo_last_mechanical_transaction(result.project)

    assert "已撤销" in reply
    assert diffs[0].after == "rolled_back"
    assert _entity(restored, "measured_outline").x2 == 244
    assert restored.mechanical_ir.dimensions[0].parsed.nominal == 244
    assert restored.mechanical_ir.dimensions[0].edit_mode == "annotation_override"
    assert restored.mechanical_transactions == []


def test_drive_refuses_locked_reference_geometry():
    project = _linear_project(measured_layer="REFERENCE_TRACE")
    plan = MechanicalOperation(dimension_id="dim_binding_244", target_value=250)

    with pytest.raises(MechanicalDriveError, match="locked reference"):
        execute_mechanical_operation(project, plan, "把244改成250")
    assert _entity(project, "measured_outline").x2 == 244


def test_diameter_drive_resizes_bound_circle():
    project = _linear_project()
    project.ir.entities.append(CircleEntity(id="hole", layer="OUTLINE", cx=50, cy=50, r=25))
    parsed = ParsedDimensionValue(kind="diameter", raw_text="φ50", nominal=50)
    dimension = project.mechanical_ir.dimensions[0]
    dimension.kind = "diameter"
    dimension.text = "φ50"
    dimension.parsed = parsed
    dimension.measured_geometry_ids = ["hole"]
    dimension.target_geometry_ids = ["hole"]
    dimension.measurement_points = [[25, 50], [75, 50]]
    dimension.orientation = "horizontal"
    dimension.dxf_dimension_type = "diameter"
    project.mechanical_dimensions = [dimension.model_copy(deep=True)]
    binding = project.dimension_bindings[0]
    binding.kind = "diameter"
    binding.text = "φ50"
    binding.parsed = parsed

    result = execute_mechanical_operation(
        project,
        MechanicalOperation(dimension_id=dimension.id, target_value=60),
        "把孔直径改成60",
    )

    assert _entity(result.project, "hole").r == 30
    assert result.project.mechanical_ir.dimensions[0].measurement_points == [[20.0, 50.0], [80.0, 50.0]]
    assert result.validation.measured_value == 60
