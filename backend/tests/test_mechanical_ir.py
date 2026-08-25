from __future__ import annotations

from app.mechanical_ir import build_mechanical_drawing_ir
from app.models import (
    DimensionBinding,
    DrawingIR,
    Layer,
    LineEntity,
    MechanicalArrowhead,
    MechanicalDimensionObject,
    ParsedDimensionValue,
    TextEntity,
)


def _binding() -> DimensionBinding:
    parsed = ParsedDimensionValue(kind="linear", raw_text="20", nominal=20)
    return DimensionBinding(
        id="dim_binding_00000",
        dimension_line_id="dim_line",
        arrow_ids=["arrow_source_left", "arrow_source_right"],
        text_id="dim_text",
        text="20",
        parsed=parsed,
        confidence=0.92,
        kind="linear",
        line_x1=0,
        line_y1=10,
        line_x2=20,
        line_y2=10,
        text_x=9,
        text_y=12,
        graph_path=["text:dim_text", "arrow:arrow_source_left", "line:dim_line"],
        graph_score=2.1,
    )


def _arrow(candidate_id: str, source_id: str, render_id: str, tip_x: float, endpoint: str) -> MechanicalArrowhead:
    return MechanicalArrowhead(
        candidate_id=candidate_id,
        source_entity_id=source_id,
        render_entity_id=render_id,
        tip_x=tip_x,
        tip_y=10,
        direction_x=1 if endpoint == "start" else -1,
        direction_y=0,
        score=0.93,
        endpoint=endpoint,
        endpoint_distance=0.1,
    )


def test_build_mechanical_ir_binds_extension_lines_and_measured_outline():
    binding = _binding()
    rendered = MechanicalDimensionObject(
        id="mechanical_dimension_dim_binding_00000",
        binding_id=binding.id,
        kind="linear",
        text="20",
        parsed=binding.parsed,
        confidence=0.92,
        dimension_line_id="dim_line",
        text_id="dim_text",
        arrowheads=[
            _arrow("left", "arrow_source_left", "solid_arrow_left", 0, "start"),
            _arrow("right", "arrow_source_right", "solid_arrow_right", 20, "end"),
        ],
    )
    ir = DrawingIR(
        layers=[Layer(name="dimensions"), Layer(name="outline")],
        entities=[
            LineEntity(id="dim_line", layer="dimensions", x1=0, y1=10, x2=20, y2=10),
            TextEntity(id="dim_text", layer="dimensions", x=9, y=12, text="20"),
            LineEntity(id="extension_left", layer="dimensions", x1=0, y1=0, x2=0, y2=10),
            LineEntity(id="extension_right", layer="dimensions", x1=20, y1=0, x2=20, y2=10),
            LineEntity(id="measured_outline", layer="outline", x1=0, y1=0, x2=20, y2=0),
            LineEntity(id="arrow_source_left", layer="dimensions", x1=0, y1=10, x2=2, y2=11),
            LineEntity(id="arrow_source_right", layer="dimensions", x1=20, y1=10, x2=18, y2=11),
        ],
    )

    mechanical_ir = build_mechanical_drawing_ir(ir, [binding], [rendered])

    assert mechanical_ir.schema_version == "1.0"
    assert mechanical_ir.unresolved_binding_ids == []
    assert len(mechanical_ir.dimensions) == 1
    dimension = mechanical_ir.dimensions[0]
    assert dimension.status == "complete"
    assert dimension.orientation == "horizontal"
    assert set(dimension.extension_line_ids) == {"extension_left", "extension_right"}
    assert dimension.measured_geometry_ids == ["measured_outline"]
    assert dimension.target_geometry_ids == dimension.measured_geometry_ids
    assert dimension.measurement_points == [[0.0, 0.0], [20.0, 0.0]]
    assert dimension.dimension_line_point == [10.0, 10.0]
    assert dimension.dxf_dimension_type == "linear"
    assert dimension.export_ready is True
    assert mechanical_ir.entity_roles["dimension_line"] == ["dim_line"]
    assert set(mechanical_ir.entity_roles["extension_line"]) == {"extension_left", "extension_right"}
    assert mechanical_ir.entity_roles["measured_geometry"] == ["measured_outline"]


def test_build_mechanical_ir_keeps_incomplete_binding_inspectable():
    binding = _binding().model_copy(update={"text_id": None, "text": ""})
    ir = DrawingIR(
        entities=[LineEntity(id="dim_line", layer="dimensions", x1=0, y1=10, x2=20, y2=10)]
    )

    mechanical_ir = build_mechanical_drawing_ir(ir, [binding], [])

    dimension = mechanical_ir.dimensions[0]
    assert dimension.status == "unresolved"
    assert "missing_text" in dimension.issues
    assert "missing_arrowhead" in dimension.issues
    assert "missing_definition_points" in dimension.issues
    assert dimension.export_ready is False
    assert binding.id in mechanical_ir.unresolved_binding_ids
