from __future__ import annotations

from datetime import datetime, timezone

from app.dimension_semantics import (
    DimensionGraphBuilder,
    _collect_line_candidates,
    _collect_text_candidates,
    detect_dimension_bindings,
    parse_dimension_text,
)
from app.models import DrawingIR, Layer, LineEntity, OcrRegion, ProjectState, RectangleEntity, TextEntity


def _project(ir: DrawingIR) -> ProjectState:
    now = datetime.now(timezone.utc)
    return ProjectState(project_id="pid", name="demo", created_at=now, updated_at=now, ir=ir)


def _dimension_ir(text: TextEntity | None = None) -> DrawingIR:
    entities = [
        LineEntity(
            id="dim_line",
            layer="promoted_geometry",
            x1=0,
            y1=0,
            x2=20,
            y2=0,
            group="promoted_geometry",
            tags=["promoted_geometry", "line_fit"],
        ),
        LineEntity(
            id="arrow_left",
            layer="promoted_geometry",
            x1=0,
            y1=0,
            x2=2.2,
            y2=1.0,
            group="promoted_geometry",
            tags=["dimension_arrow", "arrowhead"],
        ),
        LineEntity(
            id="arrow_right",
            layer="promoted_geometry",
            x1=20,
            y1=0,
            x2=17.8,
            y2=1.0,
            group="promoted_geometry",
            tags=["dimension_arrow", "arrowhead"],
        ),
    ]
    if text is not None:
        entities.append(text)
    return DrawingIR(
        units="mm",
        layers=[Layer(name="promoted_geometry"), Layer(name="text")],
        entities=entities,
    )


def test_detect_dimension_bindings_pairs_line_arrows_and_text():
    ir = _dimension_ir(TextEntity(id="txt_phi", layer="text", x=9.0, y=2.0, text="φ25 +0.021", height=2.5))

    result = detect_dimension_bindings(_project(ir))

    assert result.warnings == []
    assert len(result.bindings) == 1
    binding = result.bindings[0]
    assert binding.dimension_line_id == "dim_line"
    assert set(binding.arrow_ids) == {"arrow_left", "arrow_right"}
    assert binding.text_id == "txt_phi"
    assert binding.kind == "diameter"
    assert binding.parsed.nominal == 25.0
    assert binding.parsed.upper_tol == 0.021
    assert binding.confidence > 0.8
    assert binding.binding_method == "graph_text_arrow_line"
    assert binding.graph_path[0] == "text:txt_phi"
    assert binding.graph_path[-1] == "line:dim_line"
    assert binding.graph_score is not None


def test_dimension_graph_builder_creates_text_arrow_line_path():
    ir = _dimension_ir(TextEntity(id="txt_phi", layer="text", x=9.0, y=2.0, text="φ25 +0.021", height=2.5))
    project = _project(ir)
    dimension_lines, arrow_lines = _collect_line_candidates(project.ir.entities)
    text_candidates = _collect_text_candidates(project)
    builder = DimensionGraphBuilder(dimension_lines, arrow_lines, text_candidates)

    graph = builder.build()
    candidates = builder.binding_candidates()

    assert graph.has_node("text:txt_phi")
    assert graph.has_node("line:dim_line")
    assert any(node.startswith("arrow:") for node in graph.neighbors("text:txt_phi"))
    assert candidates
    assert candidates[0].line.entity.id == "dim_line"
    assert candidates[0].text and candidates[0].text.id == "txt_phi"
    assert candidates[0].path[0] == "text:txt_phi"
    assert candidates[0].path[-1] == "line:dim_line"
    assert any(node.startswith("arrow:") for node in candidates[0].path)


def test_detect_dimension_bindings_uses_ocr_dimension_region_as_fallback_text():
    ir = _dimension_ir()
    ir.entities.insert(
        0,
        RectangleEntity(
            id="scan_sheet_border",
            layer="sheet",
            x=0,
            y=0,
            width=100,
            height=50,
            group="sheet",
            tags=["sheet"],
        ),
    )
    project = _project(ir)
    project.ocr_regions = [
        OcrRegion(
            target="dimensions",
            label="尺寸标注",
            text="8±0.018",
            confidence=0.72,
            x=0.35,
            y=0.42,
            width=0.3,
            height=0.1,
            source="test_ocr",
        )
    ]

    result = detect_dimension_bindings(project)

    assert len(result.bindings) == 1
    binding = result.bindings[0]
    assert binding.text_id and binding.text_id.startswith("ocr_dimensions_")
    assert binding.text == "8±0.018"
    assert binding.kind == "linear"
    assert binding.parsed.nominal == 8.0
    assert binding.parsed.upper_tol == 0.018
    assert binding.parsed.lower_tol == -0.018


def test_detect_dimension_bindings_warns_without_arrows():
    ir = DrawingIR(
        units="mm",
        layers=[Layer(name="dimensions"), Layer(name="text")],
        entities=[
            LineEntity(id="line", layer="dimensions", x1=0, y1=0, x2=20, y2=0, tags=["dimensions"]),
            TextEntity(id="txt", layer="text", x=10, y=2, text="25"),
        ],
    )

    result = detect_dimension_bindings(_project(ir))

    assert result.bindings == []
    assert any("arrowhead" in warning for warning in result.warnings)


def test_parse_dimension_text_variants():
    assert parse_dimension_text("R3").kind == "radius"
    assert parse_dimension_text("Ra3.2").unit == "um"
    parsed = parse_dimension_text("φ62 0 -0.2")
    assert parsed.kind == "diameter"
    assert parsed.nominal == 62.0
    assert parsed.upper_tol == 0.0
    assert parsed.lower_tol == -0.2
    assert parse_dimension_text("0.01 A").kind == "tolerance"
