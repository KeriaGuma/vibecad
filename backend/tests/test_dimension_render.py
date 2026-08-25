from __future__ import annotations

from app.dimension_render import DIMENSION_ARROW_RENDER_TAG, render_dimension_binding_arrowheads
from app.models import DimensionBinding, DrawingIR, Layer, LineEntity, ParsedDimensionValue, PolylineEntity


def _arrow_wing(
    entity_id: str,
    candidate_id: str,
    tip_x: float,
    tip_y: float,
    direction_x: float,
    direction_y: float,
) -> LineEntity:
    return LineEntity(
        id=entity_id,
        layer="dimensions",
        x1=tip_x,
        y1=tip_y,
        x2=tip_x - direction_x * 2.0,
        y2=tip_y - direction_y * 2.0 + 0.8,
        group="promoted_geometry",
        tags=["dimension_arrow", "arrowhead", "template_arrow"],
        metadata={
            "arrow_candidate_id": candidate_id,
            "tip_x": tip_x,
            "tip_y": tip_y,
            "direction_x": direction_x,
            "direction_y": direction_y,
            "score": 0.91,
            "size_mm": 2.0,
        },
    )


def _binding(arrow_ids: list[str]) -> DimensionBinding:
    return DimensionBinding(
        id="binding_1",
        dimension_line_id="dim_line",
        arrow_ids=arrow_ids,
        text="40",
        parsed=ParsedDimensionValue(kind="linear", raw_text="40", nominal=40, unit="mm"),
        confidence=0.92,
        kind="linear",
        line_x1=10,
        line_y1=20,
        line_x2=50,
        line_y2=20,
    )


def test_render_dimension_binding_arrowheads_adds_solid_arrows_from_verified_candidates():
    ir = DrawingIR(
        layers=[
            Layer(name="promoted_geometry", color="white"),
            Layer(name="dimensions", color="green"),
        ],
        entities=[
            LineEntity(
                id="dim_line",
                layer="promoted_geometry",
                x1=10,
                y1=20,
                x2=50,
                y2=20,
                group="promoted_geometry",
            ),
            _arrow_wing("arrow_left_a", "arrow_left", 10, 20, -1, 0),
            _arrow_wing("arrow_left_b", "arrow_left", 10, 20, -1, 0),
            _arrow_wing("arrow_right_a", "arrow_right", 50, 20, 1, 0),
            _arrow_wing("arrow_right_b", "arrow_right", 50, 20, 1, 0),
        ],
    )
    binding = _binding(["arrow_left_a", "arrow_left_b", "arrow_right_a", "arrow_right_b"])

    result = render_dimension_binding_arrowheads(ir, [binding])

    arrows = [entity for entity in result.ir.entities if DIMENSION_ARROW_RENDER_TAG in entity.tags]
    assert result.arrow_line_count == 2
    assert len(arrows) == 2
    assert all(isinstance(entity, PolylineEntity) for entity in arrows)
    assert all(entity.closed for entity in arrows if isinstance(entity, PolylineEntity))
    assert all(entity.layer == "DIMENSION" for entity in arrows)
    assert all("dimension_arrow" in entity.tags and "arrowhead" in entity.tags for entity in arrows)
    assert all("solid_fill" in entity.tags for entity in arrows)
    assert all(entity.metadata.get("fill") is True for entity in arrows)
    assert {entity.metadata["arrow_candidate_id"] for entity in arrows} == {"arrow_left", "arrow_right"}
    assert {
        tuple(entity.points[0])
        for entity in arrows
        if isinstance(entity, PolylineEntity)
    } == {(10, 20), (50, 20)}
    assert len(result.mechanical_dimensions) == 1
    assert result.mechanical_dimensions[0].binding_id == "binding_1"
    assert {arrow.candidate_id for arrow in result.mechanical_dimensions[0].arrowheads} == {"arrow_left", "arrow_right"}
    assert next(layer for layer in result.ir.layers if layer.name == "DIMENSION").color == "white"


def test_render_dimension_binding_arrowheads_skips_bindings_without_verified_candidates():
    result = render_dimension_binding_arrowheads(DrawingIR(), [_binding([])])

    assert result.arrow_line_count == 0
    assert sum(1 for entity in result.ir.entities if DIMENSION_ARROW_RENDER_TAG in entity.tags) == 0
    assert result.mechanical_dimensions == []
    assert any("without verified arrowhead" in warning for warning in result.warnings)


def test_render_dimension_binding_arrowheads_is_idempotent():
    ir = DrawingIR(
        entities=[
            _arrow_wing("arrow_left_a", "arrow_left", 10, 20, -1, 0),
            _arrow_wing("arrow_left_b", "arrow_left", 10, 20, -1, 0),
            _arrow_wing("arrow_right_a", "arrow_right", 50, 20, 1, 0),
            _arrow_wing("arrow_right_b", "arrow_right", 50, 20, 1, 0),
        ]
    )
    binding = _binding(["arrow_left_a", "arrow_left_b", "arrow_right_a", "arrow_right_b"])
    first = render_dimension_binding_arrowheads(ir, [binding])
    second = render_dimension_binding_arrowheads(first.ir, [binding])

    assert sum(1 for entity in first.ir.entities if DIMENSION_ARROW_RENDER_TAG in entity.tags) == 2
    assert sum(1 for entity in second.ir.entities if DIMENSION_ARROW_RENDER_TAG in entity.tags) == 2
