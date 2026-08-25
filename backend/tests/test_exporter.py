"""DXF/SVG export coverage, including the new polyline + arc primitives."""
from __future__ import annotations

import ezdxf

from app.exporter import _bounds, export_dxf, export_svg
from app.models import (
    ArcEntity,
    CircleEntity,
    DrawingIR,
    LineEntity,
    MechanicalDimensionObject,
    MechanicalDrawingIR,
    ParsedDimensionValue,
    PolylineEntity,
    RectangleEntity,
    TextEntity,
)
from app.templates import spur_gear_drawing_ir


def _all_primitives_ir() -> DrawingIR:
    return DrawingIR(
        entities=[
            LineEntity(id="l", x1=0, y1=0, x2=10, y2=0),
            PolylineEntity(id="p", points=[[0, 0], [5, 5], [10, 0]], closed=True),
            CircleEntity(id="c", cx=20, cy=20, r=5),
            ArcEntity(id="a", cx=30, cy=30, r=4, start_angle=0, end_angle=270),
            RectangleEntity(id="r", x=0, y=0, width=8, height=4),
            TextEntity(id="t", x=1, y=1, text="REV A", rotation=90),
        ]
    )


def test_export_dxf_writes_openable_file(tmp_path):
    path = tmp_path / "out.dxf"
    export_dxf(_all_primitives_ir(), path)
    assert path.exists()
    doc = ezdxf.readfile(path)  # round-trips as a valid DXF
    types = {e.dxftype() for e in doc.modelspace()}
    assert {"LINE", "LWPOLYLINE", "CIRCLE", "ARC", "TEXT"} <= types


def test_export_svg_renders_each_primitive(tmp_path):
    path = tmp_path / "out.svg"
    export_svg(_all_primitives_ir(), path)
    svg = path.read_text(encoding="utf-8")
    assert svg.startswith("<svg")
    assert "<polygon" in svg          # closed polyline
    assert "<circle" in svg
    assert "<path" in svg             # arc
    assert "<rect" in svg
    assert "REV A" in svg             # text content
    assert 'rotate(' in svg           # rotated text transform


def test_export_svg_fills_solid_closed_polyline(tmp_path):
    ir = DrawingIR(
        entities=[
            PolylineEntity(
                id="arrow",
                layer="dimensions",
                points=[[0, 0], [2, 0.4], [2, -0.4]],
                closed=True,
                tags=["solid_fill", "dimension_arrow"],
                metadata={"fill": True},
            )
        ]
    )
    path = tmp_path / "solid.svg"
    export_svg(ir, path)

    svg = path.read_text(encoding="utf-8")
    assert '<polygon points="' in svg
    assert 'fill="#111827"' in svg


def test_export_dxf_writes_solid_for_filled_closed_polyline(tmp_path):
    ir = DrawingIR(
        entities=[
            PolylineEntity(
                id="arrow",
                layer="dimensions",
                points=[[0, 0], [2, 0.4], [2, -0.4]],
                closed=True,
                tags=["solid_fill", "dimension_arrow"],
                metadata={"fill": True},
            )
        ]
    )
    path = tmp_path / "solid.dxf"
    export_dxf(ir, path)

    doc = ezdxf.readfile(path)
    assert {entity.dxftype() for entity in doc.modelspace()} == {"SOLID"}


def test_export_dxf_writes_native_dimension_with_semantic_xdata(tmp_path):
    ir = DrawingIR(
        entities=[LineEntity(id="measured_outline", layer="outline", x1=0, y1=0, x2=244, y2=0)]
    )
    dimension = MechanicalDimensionObject(
        id="mechanical_dimension_dim_binding_244",
        binding_id="dim_binding_244",
        kind="linear",
        text="250",
        parsed=ParsedDimensionValue(kind="linear", raw_text="250", nominal=250),
        confidence=0.96,
        dimension_line_id="dim_line_244",
        extension_line_ids=["extension_left", "extension_right"],
        measured_geometry_ids=["measured_outline"],
        measurement_points=[[0, 0], [244, 0]],
        dimension_line_point=[122, 18],
        orientation="horizontal",
        dxf_dimension_type="linear",
        export_ready=True,
        status="complete",
    )
    path = tmp_path / "native_dimension.dxf"

    export_dxf(ir, path, MechanicalDrawingIR(dimensions=[dimension]))

    doc = ezdxf.readfile(path)
    native_dimensions = list(doc.modelspace().query("DIMENSION"))
    assert len(native_dimensions) == 1
    native = native_dimensions[0]
    assert native.dxf.text == "250"
    assert native.dxf.layer == "DIMENSION"
    xdata = native.get_xdata("VIBECAD")
    assert [tag.value for tag in xdata[:3]] == [
        "mechanical_dimension_dim_binding_244",
        "dim_binding_244",
        "linear",
    ]


def test_export_svg_escapes_text(tmp_path):
    ir = DrawingIR(entities=[TextEntity(id="t", x=0, y=0, text="<b>&hack")])
    path = tmp_path / "x.svg"
    export_svg(ir, path)
    svg = path.read_text(encoding="utf-8")
    assert "<b>&hack" not in svg
    assert "&lt;b&gt;&amp;hack" in svg


def test_layer_color_falls_back_to_white():
    from app.exporter import layer_color

    ir = DrawingIR(entities=[LineEntity(id="l", layer="unknown_layer", x1=0, y1=0, x2=1, y2=1)])
    assert layer_color(ir, "unknown_layer") == "white"


def test_bounds_empty_drawing_has_safe_default():
    assert _bounds(DrawingIR()) == (0, 0, 100, 100)


def test_bounds_wraps_all_geometry():
    ir = DrawingIR(
        entities=[
            CircleEntity(id="c", cx=0, cy=0, r=5),
            RectangleEntity(id="r", x=10, y=-3, width=20, height=8),
        ]
    )
    assert _bounds(ir) == (-5, -5, 30, 5)


def test_gear_template_exports_both_formats(tmp_path):
    ir = spur_gear_drawing_ir()
    export_dxf(ir, tmp_path / "gear.dxf")
    export_svg(ir, tmp_path / "gear.svg")
    assert (tmp_path / "gear.dxf").stat().st_size > 0
    assert (tmp_path / "gear.svg").stat().st_size > 0
