from __future__ import annotations

from datetime import datetime, timezone

from .models import (
    DimensionBinding,
    DimensionGroundTruth,
    DrawingIR,
    Layer,
    LineEntity,
    MechanicalArrowhead,
    MechanicalDimensionObject,
    MechanicalDrawingIR,
    ParsedDimensionValue,
    PolylineEntity,
    ProjectState,
    RectangleEntity,
    TextEntity,
    default_ir,
)


def build_agent_eval_fixture(name: str, case_id: str) -> ProjectState:
    if name == "baseline_plate":
        return _project(case_id, default_ir())
    if name == "locked_reference":
        project = _project(case_id, default_ir())
        hole = next(entity for entity in project.ir.entities if entity.id == "hole_1")
        hole.layer = "REFERENCE_TRACE"
        project.ir.layers.append(Layer(name="REFERENCE_TRACE", locked=True, editable=False))
        return project
    if name == "complete_dimension":
        return _complete_dimension_project(case_id)
    if name == "repairable_dimension":
        return _repairable_dimension_project(case_id)
    raise ValueError(f"Unknown agent eval fixture: {name}")


def _project(case_id: str, ir: DrawingIR) -> ProjectState:
    now = datetime.now(timezone.utc)
    return ProjectState(
        project_id=f"eval_{case_id}",
        name=f"Agent eval: {case_id}",
        created_at=now,
        updated_at=now,
        ir=ir,
    )


def _complete_dimension_project(case_id: str) -> ProjectState:
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
    ir = DrawingIR(
        layers=[
            Layer(name="OUTLINE", lineweight=0.5),
            Layer(name="DIMENSION", lineweight=0.18),
            Layer(name="REFERENCE_TRACE", locked=True, editable=False),
        ],
        entities=[
            LineEntity(id="measured_outline", layer="OUTLINE", x1=0, y1=0, x2=244, y2=0),
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
    project = _project(case_id, ir)
    project.dimension_bindings = [binding]
    project.mechanical_dimensions = [dimension.model_copy(deep=True)]
    project.mechanical_ir = MechanicalDrawingIR(dimensions=[dimension.model_copy(deep=True)])
    project.dimension_ground_truth = [
        DimensionGroundTruth(
            id="gt_linear_244",
            label="总长 244",
            expected_text="244",
            kind="linear",
            nominal=244,
            matched_dimension_id=dimension.id,
            source="seed",
        )
    ]
    return project


def _repairable_dimension_project(case_id: str) -> ProjectState:
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
    project = _project(
        case_id,
        DrawingIR(
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
    )
    project.dimension_bindings = [binding]
    project.mechanical_dimensions = [dimension.model_copy(deep=True)]
    project.mechanical_ir = MechanicalDrawingIR(dimensions=[dimension])
    project.dimension_ground_truth = [
        DimensionGroundTruth(
            id="gt_linear_244",
            label="总长 244",
            expected_text="244",
            kind="linear",
            nominal=244,
            matched_dimension_id=dimension.id,
            source="seed",
        )
    ]
    return project
