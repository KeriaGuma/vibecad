from __future__ import annotations

from datetime import datetime

import ezdxf

from app.cad_ops import apply_operations
from app.exporter import export_dxf
from app.mechanical_edit import plan_mechanical_dimension_edit, sync_mechanical_dimension_edit
from app.models import (
    DimensionBinding,
    DrawingIR,
    MechanicalArrowhead,
    MechanicalDimensionObject,
    MechanicalDrawingIR,
    ParsedDimensionValue,
    ProjectState,
    TextEntity,
)


def _project(text_entity: TextEntity | None = None) -> ProjectState:
    parsed = ParsedDimensionValue(kind="diameter", raw_text="φ49 -0.2", nominal=49, lower_tol=-0.2)
    dimension = MechanicalDimensionObject(
        id="mechanical_dimension_dim_binding_00000",
        binding_id="dim_binding_00000",
        kind="diameter",
        text="φ49 -0.2",
        parsed=parsed,
        confidence=0.91,
        dimension_line_id="dim_line_1",
        text_id=text_entity.id if text_entity else "ocr_49",
        arrowheads=[
            MechanicalArrowhead(
                candidate_id="arrow_left",
                source_entity_id="arrow_1",
                render_entity_id="solid_arrow_1",
                tip_x=10,
                tip_y=20,
                direction_x=-1,
                direction_y=0,
                score=0.9,
                endpoint="start",
                endpoint_distance=0.1,
            )
        ],
    )
    return ProjectState(
        project_id="p1",
        name="demo",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        ir=DrawingIR(entities=[text_entity] if text_entity else []),
        dimension_bindings=[
            DimensionBinding(
                id="dim_binding_00000",
                dimension_line_id="dim_line_1",
                arrow_ids=["arrow_1", "arrow_2"],
                text_id=text_entity.id if text_entity else "ocr_49",
                text="φ49 -0.2",
                parsed=parsed,
                confidence=0.91,
                kind="diameter",
                line_x1=10,
                line_y1=20,
                line_x2=50,
                line_y2=20,
                text_x=28,
                text_y=24,
            )
        ],
        mechanical_dimensions=[dimension.model_copy(deep=True)],
        mechanical_ir=MechanicalDrawingIR(dimensions=[dimension.model_copy(deep=True)]),
    )


def test_plan_mechanical_dimension_edit_updates_bound_text_and_snapshot():
    project = _project(
        TextEntity(
            id="dim_text_1",
            layer="dimensions",
            x=28,
            y=24,
            text="φ49 -0.2",
            height=2.4,
        )
    )

    edit = plan_mechanical_dimension_edit("把 49 改成 50", project)

    assert edit is not None
    assert edit.operations[0].operation == "modify_entity"
    assert edit.operations[0].entity_id == "dim_text_1"
    assert edit.operations[0].changes["text"] == "φ50 -0.2"
    project.ir, _ = apply_operations(project.ir, edit.operations)
    sync_mechanical_dimension_edit(project, edit)

    entity = project.ir.entities[0]
    assert isinstance(entity, TextEntity)
    assert entity.text == "φ50 -0.2"
    assert project.dimension_bindings[0].text == "φ50 -0.2"
    assert project.dimension_bindings[0].parsed.nominal == 50
    assert project.mechanical_dimensions[0].text == "φ50 -0.2"
    assert project.mechanical_dimensions[0].parsed.nominal == 50
    assert project.mechanical_ir.dimensions[0].text == "φ50 -0.2"
    assert project.mechanical_ir.dimensions[0].parsed.nominal == 50


def test_plan_mechanical_dimension_edit_adds_text_when_only_ocr_text_exists():
    project = _project()

    edit = plan_mechanical_dimension_edit("dim_binding_00000 改成 52", project)

    assert edit is not None
    assert edit.operations[0].operation == "add_entity"
    assert edit.text_id is not None
    project.ir, _ = apply_operations(project.ir, edit.operations)
    sync_mechanical_dimension_edit(project, edit)

    entity = project.ir.entities[0]
    assert isinstance(entity, TextEntity)
    assert entity.text == "φ52 -0.2"
    assert entity.x == 28
    assert entity.y == 24
    assert project.dimension_bindings[0].text_id == edit.text_id
    assert project.mechanical_dimensions[0].text_id == edit.text_id


def test_plan_mechanical_dimension_edit_returns_clean_noop_for_missing_match():
    project = _project()

    edit = plan_mechanical_dimension_edit("把 88 改成 50", project)

    assert edit is not None
    assert edit.handled is True
    assert edit.operations == []
    assert "没有" in edit.reply


def test_plan_mechanical_dimension_edit_supports_244_to_250(tmp_path):
    project = _project(
        TextEntity(
            id="dim_text_244",
            layer="dimensions",
            x=28,
            y=24,
            text="244",
            height=2.4,
        )
    )
    for dimension in [*project.mechanical_dimensions, *project.mechanical_ir.dimensions]:
        dimension.kind = "linear"
        dimension.text = "244"
        dimension.parsed = ParsedDimensionValue(kind="linear", raw_text="244", nominal=244)
        dimension.measurement_points = [[0, 0], [244, 0]]
        dimension.dimension_line_point = [122, 18]
        dimension.orientation = "horizontal"
        dimension.dxf_dimension_type = "linear"
        dimension.export_ready = True
        dimension.status = "complete"
    binding = project.dimension_bindings[0]
    binding.kind = "linear"
    binding.text = "244"
    binding.parsed = ParsedDimensionValue(kind="linear", raw_text="244", nominal=244)

    edit = plan_mechanical_dimension_edit("把 244 改成 250", project)

    assert edit is not None
    assert edit.operations[0].changes["text"] == "250"
    project.ir, _ = apply_operations(project.ir, edit.operations)
    sync_mechanical_dimension_edit(project, edit)
    assert project.mechanical_ir.dimensions[0].text == "250"
    assert project.mechanical_ir.dimensions[0].parsed.nominal == 250

    dxf_path = tmp_path / "edited_dimension.dxf"
    export_dxf(project.ir, dxf_path, project.mechanical_ir)
    native = list(ezdxf.readfile(dxf_path).modelspace().query("DIMENSION"))[0]
    assert native.dxf.text == "250"
