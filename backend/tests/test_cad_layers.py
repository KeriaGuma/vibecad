from __future__ import annotations

import ezdxf

from app.cad_layers import CAD_LAYER_SPECS, normalize_cad_layers
from app.exporter import export_dxf, export_svg
from app.models import DrawingIR, LineEntity, RectangleEntity, TextEntity


def _legacy_ir() -> DrawingIR:
    return DrawingIR(
        entities=[
            LineEntity(id="trace", layer="reference_trace", group="reference_trace", x1=0, y1=0, x2=10, y2=0),
            LineEntity(id="outline", layer="promoted_geometry", group="promoted_geometry", x1=0, y1=2, x2=10, y2=2),
            LineEntity(id="dimension", layer="dimensions", tags=["dimensions"], x1=0, y1=4, x2=10, y2=4),
            LineEntity(id="center", layer="centerline", tags=["centerline"], x1=0, y1=6, x2=10, y2=6),
            LineEntity(id="hatch", layer="hatch", tags=["hatch"], x1=0, y1=8, x2=10, y2=8),
            TextEntity(id="note", layer="notes", x=0, y=10, text="NOTE"),
            RectangleEntity(id="drawing_frame", layer="sheet", group="sheet", x=0, y=0, width=20, height=12),
        ]
    )


def test_normalize_cad_layers_maps_legacy_roles_and_locks_reference():
    ir = normalize_cad_layers(_legacy_ir())

    assert [layer.name for layer in ir.layers] == [spec.name for spec in CAD_LAYER_SPECS]
    assert [entity.layer for entity in ir.entities] == [
        "REFERENCE_TRACE",
        "OUTLINE",
        "DIMENSION",
        "CENTER",
        "HATCH",
        "TEXT",
        "TITLE_BLOCK",
    ]
    reference = ir.entities[0]
    assert reference.stroke_width == 0.13
    assert reference.metadata["locked"] is True
    assert reference.metadata["editable"] is False
    assert reference.metadata["legacy_layer"] == "reference_trace"
    assert ir.entities[1].stroke_width == 0.5
    assert ir.entities[-1].stroke_width == 0.5
    assert next(layer for layer in ir.layers if layer.name == "REFERENCE_TRACE").locked is True


def test_normalize_cad_layers_is_idempotent():
    ir = normalize_cad_layers(_legacy_ir())
    first = ir.model_dump()

    normalize_cad_layers(ir)

    assert ir.model_dump() == first


def test_dxf_and_svg_use_monochrome_cad_layer_contract(tmp_path):
    dxf_path = tmp_path / "layers.dxf"
    svg_path = tmp_path / "layers.svg"
    ir = _legacy_ir()

    export_dxf(ir, dxf_path)
    export_svg(ir, svg_path)

    doc = ezdxf.readfile(dxf_path)
    assert next(layer for layer in doc.layers if layer.dxf.name == "REFERENCE_TRACE").is_locked()
    assert doc.layers.get("CENTER").dxf.linetype == "CENTER2"
    assert {entity.dxf.layer for entity in doc.modelspace()} <= {
        "REFERENCE_TRACE",
        "OUTLINE",
        "DIMENSION",
        "CENTER",
        "HATCH",
        "TEXT",
        "TITLE_BLOCK",
    }
    lineweights = {entity.dxf.layer: entity.dxf.lineweight for entity in doc.modelspace() if entity.dxftype() == "LINE"}
    assert lineweights["REFERENCE_TRACE"] == 13
    assert lineweights["OUTLINE"] == 50
    assert lineweights["DIMENSION"] == 25

    svg = svg_path.read_text(encoding="utf-8")
    assert "#2563eb" not in svg
    assert 'data-id="center"' in svg and 'stroke-dasharray="2.5 0.8 0.4 0.8"' in svg
    assert 'data-id="trace"' in svg and 'stroke-opacity="0.48"' in svg
