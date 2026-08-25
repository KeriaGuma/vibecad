"""Line A vector-PDF -> DrawingIR extraction tests."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import ezdxf
import pymupdf
import pytest

from app.exporter import export_dxf
from app.models import LineEntity, RectangleEntity, TextEntity
from app.structure_eval import evaluate_structure
from app.svg_dxf import (
    _append_point,
    _has_visible_stroke,
    _sample_curve,
    export_svg_geometry_to_dxf,
    render_dxf_to_svg,
    svg_geometry_to_polylines,
)
from app.vector_external import (
    _dxf_has_entities,
    _find_inkscape,
    _try_inkscape_dxf,
    _try_pstoedit_dxf,
    _try_svg_dxf,
    _write_dxf_preview_or_fallback,
    _write_mupdf_svg_preview,
    export_vector_pdf_assets,
)
from app.vector_extract import PT_TO_MM, extract_drawing_ir


def _synthetic_pdf(path: Path) -> None:
    document = pymupdf.open()
    page = document.new_page(width=200, height=100)  # points
    page.draw_line(pymupdf.Point(10, 10), pymupdf.Point(190, 10))  # near the top in pdf space
    page.draw_rect(pymupdf.Rect(20, 30, 120, 80))
    page.insert_text((30, 60), "PART-1", fontsize=12)
    document.save(path)
    document.close()


def test_extract_maps_primitives_and_flips_y(tmp_path):
    pdf = tmp_path / "v.pdf"
    _synthetic_pdf(pdf)

    ir = extract_drawing_ir(pdf)

    assert ir.units == "mm"
    kinds = {e.type for e in ir.entities}
    assert {"line", "rectangle", "text"} <= kinds

    line = next(e for e in ir.entities if isinstance(e, LineEntity))
    # The line sat at pdf-y=10 (near the top); after the y-flip it lands near the
    # top of the y-up page, i.e. close to the full height in mm.
    page_height_mm = 100 * PT_TO_MM
    assert line.y1 == pytest.approx(page_height_mm - 10 * PT_TO_MM, abs=0.05)

    text = next(e for e in ir.entities if isinstance(e, TextEntity))
    assert text.text == "PART-1"
    assert text.height == pytest.approx(12 * PT_TO_MM, abs=0.05)


def test_extract_dxf_round_trips(tmp_path):
    pdf = tmp_path / "v.pdf"
    _synthetic_pdf(pdf)

    ir = extract_drawing_ir(pdf)
    out = tmp_path / "out.dxf"
    export_dxf(ir, out)

    doc = ezdxf.readfile(out)
    types = {e.dxftype() for e in doc.modelspace()}
    assert "LINE" in types
    assert "TEXT" in types


def test_extract_rejects_out_of_range_page(tmp_path):
    pdf = tmp_path / "v.pdf"
    _synthetic_pdf(pdf)

    with pytest.raises(ValueError, match="out of range"):
        extract_drawing_ir(pdf, page_number=5)


def test_extract_real_reference_is_rich():
    sample = Path(__file__).resolve().parents[2] / "data" / "demo_data" / "test.pdf"
    if not sample.exists():
        pytest.skip("sample drawing not available")

    ir = extract_drawing_ir(sample)

    # The real gear drawing is geometry-dense; the template only managed ~180.
    assert len(ir.entities) > 1000
    assert any(isinstance(e, RectangleEntity) for e in ir.entities)
    assert any(isinstance(e, TextEntity) and e.text for e in ir.entities)
    assert {e.group for e in ir.entities if e.group} >= {
        "title_block",
        "parameter_table",
        "section_view",
        "circular_view",
        "dimensions",
    }


def test_extract_real_reference_scores_structure_eval():
    sample = Path(__file__).resolve().parents[2] / "data" / "demo_data" / "test.pdf"
    if not sample.exists():
        pytest.skip("sample drawing not available")

    report = evaluate_structure(extract_drawing_ir(sample))

    assert report.passed is True
    assert report.overall_score == 1.0
    assert all(target.evidence["semantic_import"] for target in report.targets)


def test_vector_pdf_assets_use_mupdf_svg_and_svg_dxf_fallback(tmp_path, monkeypatch):
    pdf = tmp_path / "v.pdf"
    _synthetic_pdf(pdf)
    monkeypatch.setattr("app.vector_external.shutil.which", lambda name: None)

    result = export_vector_pdf_assets(pdf, tmp_path)

    assert result.preview_source == "dxf_render_svg"
    assert result.dxf_source == "svg_dxf"
    assert "Inkscape" in result.warnings[0]
    assert "svgelements+ezdxf" in result.warnings[1]
    assert "DXF preview rendered" in result.warnings[-1]
    assert (tmp_path / "source_preview.svg").read_text(encoding="utf-8").lstrip().startswith("<svg")
    assert (tmp_path / "preview.svg").read_text(encoding="utf-8").lstrip().startswith("<?xml")
    doc = ezdxf.readfile(tmp_path / "output.dxf")
    assert len(list(doc.modelspace())) > 0
    assert {entity.dxf.layer for entity in doc.modelspace()} == {"REFERENCE_TRACE"}
    assert doc.layers.get("REFERENCE_TRACE").is_locked()


def test_vector_pdf_assets_reject_bad_page_or_pdf(tmp_path):
    pdf = tmp_path / "v.pdf"
    _synthetic_pdf(pdf)
    with pytest.raises(ValueError, match="out of range"):
        _write_mupdf_svg_preview(pdf, tmp_path / "preview.svg", page_number=5)

    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"not a pdf")
    with pytest.raises(ValueError, match="Could not open PDF"):
        _write_mupdf_svg_preview(broken, tmp_path / "broken.svg", page_number=0)


def test_pstoedit_adapter_success_and_failure(tmp_path, monkeypatch):
    pdf = tmp_path / "v.pdf"
    _synthetic_pdf(pdf)
    dxf = tmp_path / "out.dxf"
    monkeypatch.setattr("app.vector_external.shutil.which", lambda name: f"/usr/local/bin/{name}")

    def fake_success(args, **kwargs):
        Path(args[-1]).write_text("0\nSECTION\n2\nENTITIES\n0\nLINE\n8\n0\n0\nENDSEC\n0\nEOF\n", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("app.vector_external.subprocess.run", fake_success)
    warnings: list[str] = []
    assert _try_pstoedit_dxf(pdf, dxf, warnings) == "pstoedit"
    assert warnings == []
    assert dxf.exists()

    dxf.write_text("fallback", encoding="utf-8")

    def fake_empty(args, **kwargs):
        Path(args[-1]).write_text("0\nSECTION\n2\nENTITIES\n0\nENDSEC\n0\nEOF\n", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="empty")

    monkeypatch.setattr("app.vector_external.subprocess.run", fake_empty)
    assert _try_pstoedit_dxf(pdf, dxf, warnings) == "ir_fallback"
    assert dxf.read_text(encoding="utf-8") == "fallback"
    assert "empty DXF" in warnings[-1]

    dxf.unlink()

    def fake_failure(args, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr("app.vector_external.subprocess.run", fake_failure)
    assert _try_pstoedit_dxf(pdf, dxf, warnings) == "ir_fallback"
    assert "boom" in warnings[-1]


def test_inkscape_adapter_success_and_empty_output(tmp_path, monkeypatch):
    svg = tmp_path / "preview.svg"
    svg.write_text("<svg/>", encoding="utf-8")
    dxf = tmp_path / "out.dxf"
    monkeypatch.setattr("app.vector_external._find_inkscape", lambda: "/usr/local/bin/inkscape")

    def fake_success(args, **kwargs):
        Path(args[2].removeprefix("--export-filename=")).write_text(
            "0\nSECTION\n2\nENTITIES\n0\nLWPOLYLINE\n8\n0\n0\nENDSEC\n0\nEOF\n",
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("app.vector_external.subprocess.run", fake_success)
    warnings: list[str] = []
    assert _try_inkscape_dxf(svg, dxf, warnings) == "inkscape"
    assert warnings == []
    assert dxf.exists()

    dxf.write_text("fallback", encoding="utf-8")

    def fake_empty(args, **kwargs):
        Path(args[2].removeprefix("--export-filename=")).write_text(
            "0\nSECTION\n2\nENTITIES\n0\nENDSEC\n0\nEOF\n",
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="empty")

    monkeypatch.setattr("app.vector_external.subprocess.run", fake_empty)
    assert _try_inkscape_dxf(svg, dxf, warnings) == "ir_fallback"
    assert dxf.read_text(encoding="utf-8") == "fallback"
    assert "empty DXF" in warnings[-1]


def test_svg_dxf_export_keeps_geometry_and_skips_text_paths(tmp_path):
    svg = tmp_path / "drawing.svg"
    svg.write_text(
        """<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 100 100">
  <path d="M10 10 L90 10 L90 90 Z" fill="none" stroke="black" />
  <path d="M10 90 C20 70 40 70 50 90" fill="none" stroke="black" />
  <path d="M20 80 L30 75 L30 85 Z" fill="black" stroke="none" />
  <path data-text="A" d="M40 40 L50 40 L50 50 Z" fill="black" stroke="none" />
  <path d="" fill="none" stroke="black" />
  <path d="M5 5" fill="none" stroke="black" />
</svg>
""",
        encoding="utf-8",
    )
    dxf = tmp_path / "drawing.dxf"

    result = export_svg_geometry_to_dxf(svg, dxf)

    assert result.entity_count == 3
    doc = ezdxf.readfile(dxf)
    entities = list(doc.modelspace())
    assert [entity.dxftype() for entity in entities] == ["LWPOLYLINE", "LWPOLYLINE", "LWPOLYLINE"]


def test_svg_geometry_to_polylines_scales_and_limits_entities(tmp_path):
    svg = tmp_path / "drawing.svg"
    svg.write_text(
        """<svg xmlns="http://www.w3.org/2000/svg" width="100" height="50" viewBox="0 0 100 50">
  <path d="M10 10 L90 10 L90 40 Z" fill="none" stroke="black" />
  <path d="M20 20 L80 20" fill="none" stroke="black" />
  <path data-text="A" d="M1 1 L2 2" fill="none" stroke="black" />
</svg>
""",
        encoding="utf-8",
    )

    entities = svg_geometry_to_polylines(
        svg,
        layer="reference_trace",
        group="reference_trace",
        id_prefix="trace",
        tags=["unit"],
        stroke_width=0.2,
        max_entities=1,
        target_width=200,
    )

    assert len(entities) == 1
    assert entities[0].closed is True
    assert entities[0].layer == "reference_trace"
    assert entities[0].tags == ["unit", "reference_trace", "external_vectorizer"]
    assert entities[0].stroke_width == 0.2
    assert entities[0].points[0] == [20.0, 80.0]


def test_svg_geometry_to_polylines_uses_width_without_viewbox(tmp_path):
    svg = tmp_path / "drawing.svg"
    svg.write_text(
        """<svg xmlns="http://www.w3.org/2000/svg" width="40" height="20">
  <path d="M0 0 L40 20" fill="none" stroke="black" />
</svg>
""",
        encoding="utf-8",
    )

    entities = svg_geometry_to_polylines(
        svg,
        layer="editable_linework",
        group="editable_linework",
        id_prefix="editable",
        target_width=80,
    )

    assert len(entities) == 1
    assert entities[0].points[-1] == [80.0, 0.0]


def test_svg_dxf_adapter_replaces_nonempty_output(tmp_path):
    svg = tmp_path / "drawing.svg"
    svg.write_text(
        """<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 20 20">
  <path d="M1 1 L19 19" fill="none" stroke="black" />
</svg>
""",
        encoding="utf-8",
    )
    dxf = tmp_path / "out.dxf"
    warnings: list[str] = []

    assert _try_svg_dxf(svg, dxf, warnings) == "svg_dxf"
    assert "svgelements+ezdxf" in warnings[-1]
    assert dxf.exists()

    preview = tmp_path / "preview.svg"
    rendered = render_dxf_to_svg(dxf, preview)
    assert rendered.entity_count == 1
    assert rendered.width_mm > 0
    assert preview.read_text(encoding="utf-8").lstrip().startswith("<?xml")


def test_svg_dxf_adapter_handles_empty_and_failed_exports(tmp_path, monkeypatch):
    svg = tmp_path / "empty.svg"
    svg.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"10\" height=\"10\" />", encoding="utf-8")
    dxf = tmp_path / "out.dxf"
    warnings: list[str] = []

    assert _try_svg_dxf(svg, dxf, warnings) == "ir_fallback"
    assert "empty DXF" in warnings[-1]

    def boom(svg_path, dxf_path):
        raise RuntimeError("broken converter")

    monkeypatch.setattr("app.vector_external.export_svg_geometry_to_dxf", boom)
    assert _try_svg_dxf(svg, dxf, warnings) == "ir_fallback"
    assert "broken converter" in warnings[-1]


def test_render_dxf_to_svg_rejects_empty_modelspace(tmp_path):
    dxf = tmp_path / "empty.dxf"
    preview = tmp_path / "preview.svg"
    doc = ezdxf.new("R2010")
    doc.saveas(dxf)

    with pytest.raises(ValueError, match="no modelspace entities"):
        render_dxf_to_svg(dxf, preview)


def test_dxf_preview_falls_back_to_source_svg(tmp_path):
    dxf = tmp_path / "missing.dxf"
    preview = tmp_path / "preview.svg"
    source = tmp_path / "source_preview.svg"
    source.write_text("<svg>source</svg>", encoding="utf-8")
    warnings: list[str] = []

    assert _write_dxf_preview_or_fallback(dxf, preview, source, warnings) == "mupdf_svg_fallback"
    assert preview.read_text(encoding="utf-8") == "<svg>source</svg>"
    assert "DXF preview render failed" in warnings[-1]


def test_svg_dxf_small_helpers_cover_edge_cases(monkeypatch, tmp_path):
    class BadStroke:
        stroke = "#000000"
        stroke_width = "bad"

    assert _has_visible_stroke(BadStroke())
    points = []
    _append_point(points, None)
    assert points == []

    class OddCurve:
        start = None

        def length(self, **kwargs):
            raise RuntimeError("no length")

        def point(self, position):
            from svgelements import Point

            return Point(position, position)

    assert len(list(_sample_curve(OddCurve()))) >= 4
    assert not _dxf_has_entities(tmp_path / "missing.dxf")

    monkeypatch.setattr("app.vector_external.shutil.which", lambda name: None)
    monkeypatch.setattr("app.vector_external.Path.exists", lambda self: str(self).endswith("/inkscape"))
    assert _find_inkscape() in {
        "/Applications/Inkscape.app/Contents/MacOS/inkscape",
        "/opt/homebrew/bin/inkscape",
        "/usr/local/bin/inkscape",
    }
