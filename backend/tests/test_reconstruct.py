"""Table/title-block reconstruction from reference image grids."""
from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

from app import table_ocr as table_ocr_module
from app.models import LineEntity, PolylineEntity, ProjectState, default_ir
from app.reconstruct import reconstruct_tables_from_reference
from app.scan_cad import (
    EDITABLE_LINEWORK_STROKE_MM,
    REFERENCE_TRACE_STROKE_MM,
    TABLE_GRID_STROKE_MM,
    _accept_section_reconstruction,
    _base_scan_ir,
    _external_vectorizer_entities,
    _hough_fallback_entities,
    _mask_trace_entities_in_regions,
    _quantized_line_key,
    _run_autotrace_entities,
    _run_vtracer_entities,
    _scan_table_entities,
    reconstruct_scan_cad_from_reference,
)
from app.section_cv import (
    Segment,
    _clip_segment_to_polygon,
    _clipped_hatches,
    _cluster_values,
    _detect_section_segments,
    _estimate_section_topology,
    _filtered_hatches,
    _first_horizontal,
    _left_outline_x,
    _point_in_polygon,
    _raw_line_to_segment,
    _right_extension_x,
    _row_segment_start,
    _segment_intersection_t,
    _segments_to_entities,
    _unique_sorted,
    reconstruct_section_from_reference,
)
from app.table_ocr import extract_table_ocr_from_reference
from app.templates import spur_gear_drawing_ir


def test_reconstruct_tables_from_reference_detects_non_overlapping_regions(tmp_path):
    image_path = tmp_path / "pid_reference.png"
    _write_table_page(image_path)
    now = datetime.now(timezone.utc)
    project = ProjectState(
        project_id="pid",
        name="demo",
        created_at=now,
        updated_at=now,
        source_image="/api/uploads/pid_reference.png",
        ir=default_ir(),
    )

    result = reconstruct_tables_from_reference(project, tmp_path)

    assert result.layout_passed is True
    assert result.warnings == []
    assert {region.target for region in result.regions} == {"parameter_table", "title_block"}
    table_lines = [entity for entity in result.ir.entities if entity.group in {"parameter_table", "title_block"}]
    assert len(table_lines) >= 20
    assert any(entity.group == "parameter_table" and entity.type == "text" for entity in result.ir.entities)
    assert any(entity.group == "title_block" and entity.type == "text" for entity in result.ir.entities)


def test_extract_table_ocr_from_reference_returns_structured_cells(tmp_path, monkeypatch):
    image_path = tmp_path / "pid_reference.png"
    _write_table_page(image_path)
    now = datetime.now(timezone.utc)
    project = ProjectState(
        project_id="pid",
        name="demo",
        created_at=now,
        updated_at=now,
        source_image="/api/uploads/pid_reference.png",
        ir=default_ir(),
    )
    monkeypatch.setattr("app.table_ocr._paddleocr_available", lambda: True)
    monkeypatch.setattr("app.table_ocr._run_paddle_ocr", lambda crop, language: ("cell text", 0.88))

    result = extract_table_ocr_from_reference(project, tmp_path, language_hint="zh")

    assert result.warnings == []
    assert result.cells
    assert {cell.target for cell in result.cells} == {"parameter_table", "title_block"}
    assert all(cell.text == "cell text" for cell in result.cells)
    assert all(cell.confidence == 0.88 for cell in result.cells)
    assert all(cell.engine == "paddleocr" for cell in result.cells)


def test_table_reconstruction_and_ocr_normalize_rotated_portrait_scan(tmp_path, monkeypatch):
    source_path = tmp_path / "source.png"
    image_path = tmp_path / "pid_reference.png"
    _write_table_page(source_path)
    Image.open(source_path).rotate(-90, expand=True).save(image_path)
    now = datetime.now(timezone.utc)
    project = ProjectState(
        project_id="pid",
        name="demo",
        created_at=now,
        updated_at=now,
        source_image="/api/uploads/pid_reference.png",
        ir=default_ir(),
    )

    reconstruction = reconstruct_tables_from_reference(project, tmp_path)

    assert reconstruction.layout_passed is True
    assert {region.target for region in reconstruction.regions} == {"parameter_table", "title_block"}

    monkeypatch.setattr("app.table_ocr._paddleocr_available", lambda: True)
    monkeypatch.setattr("app.table_ocr._run_paddle_ocr", lambda crop, language: ("cell text", 0.88))

    ocr = extract_table_ocr_from_reference(project, tmp_path, language_hint="zh")

    assert ocr.warnings == []
    assert ocr.cells
    assert {cell.target for cell in ocr.cells} == {"parameter_table", "title_block"}


def test_reconstruct_scan_cad_generates_trace_and_structured_layers_from_rotated_scan(tmp_path, monkeypatch):
    source_path = tmp_path / "source.png"
    image_path = tmp_path / "pid_reference.png"
    _write_scan_cad_page(source_path)
    Image.open(source_path).rotate(-90, expand=True).save(image_path)
    now = datetime.now(timezone.utc)
    project = ProjectState(
        project_id="pid",
        name="demo",
        created_at=now,
        updated_at=now,
        source_kind="scanned_pdf",
        source_image="/api/uploads/pid_reference.png",
        ir=default_ir(),
    )
    monkeypatch.setattr(
        "app.scan_cad._run_vtracer_entities",
        lambda image_path, output_dir, warnings: [
            PolylineEntity(
                id="ref_trace_00000",
                layer="reference_trace",
                points=[[0, 0], [10, 0]],
                group="reference_trace",
                tags=["vtracer"],
            )
        ],
    )
    monkeypatch.setattr(
        "app.scan_cad._run_autotrace_entities",
        lambda image_path, output_dir, warnings: [
            PolylineEntity(
                id="editable_line_00000",
                layer="editable_linework",
                points=[[0, 0], [0, 10]],
                group="editable_linework",
                tags=["autotrace"],
            )
        ],
    )

    result = reconstruct_scan_cad_from_reference(project, tmp_path)

    assert result.entity_count == len(result.ir.entities)
    assert result.trace_count == 2
    assert result.structured_counts["reference_trace"] == 1
    assert result.structured_counts["editable_linework"] == 1
    assert result.structured_counts["tables"] > 0
    assert result.structured_counts["section_view"] > 0
    assert any(entity.group == "reference_trace" for entity in result.ir.entities)
    assert any(entity.group == "editable_linework" for entity in result.ir.entities)
    assert any(entity.group == "title_block" for entity in result.ir.entities)
    assert any(entity.group == "section_view" for entity in result.ir.entities)


def test_scan_cad_rejects_low_confidence_section_overlay():
    assert _accept_section_reconstruction(12, []) is True
    assert _accept_section_reconstruction(2, []) is False
    assert _accept_section_reconstruction(12, ["Low hatch line count: 0"]) is False


def test_scan_cad_masks_trace_inside_reconstructed_table_regions():
    entities = [
        PolylineEntity(
            id="ref_inside",
            layer="reference_trace",
            points=[[10, 10], [12, 12]],
            group="reference_trace",
        ),
        PolylineEntity(
            id="editable_inside",
            layer="editable_linework",
            points=[[18, 18], [22, 18]],
            group="editable_linework",
        ),
        PolylineEntity(
            id="ref_outside",
            layer="reference_trace",
            points=[[80, 80], [90, 90]],
            group="reference_trace",
        ),
        LineEntity(
            id="manual_line",
            layer="geometry",
            x1=10,
            y1=10,
            x2=12,
            y2=12,
            group="section_view",
        ),
    ]

    masked = _mask_trace_entities_in_regions(entities, [(8, 8, 24, 24)])

    assert [entity.id for entity in masked] == ["ref_outside", "manual_line"]


def test_scan_cad_table_entities_drop_stub_text_and_apply_thin_grid_weight():
    entities = [
        LineEntity(
            id="grid",
            layer="table",
            x1=0,
            y1=0,
            x2=10,
            y2=0,
            group="title_block",
            tags=["title_block", "grid"],
        ),
        PolylineEntity(
            id="stub_text",
            layer="text",
            points=[[0, 0], [1, 1]],
            group="title_block",
            tags=["title_block", "text_stub"],
        ),
    ]

    clean = _scan_table_entities(entities)

    assert [entity.id for entity in clean] == ["grid"]
    assert clean[0].stroke_width == TABLE_GRID_STROKE_MM


def test_scan_cad_rejects_missing_or_wrong_source_kind(tmp_path):
    now = datetime.now(timezone.utc)
    project = ProjectState(project_id="pid", name="demo", created_at=now, updated_at=now, ir=default_ir())

    try:
        reconstruct_scan_cad_from_reference(project, tmp_path)
    except ValueError as exc:
        assert "Upload a PDF or image" in str(exc)
    else:
        raise AssertionError("expected missing source error")

    project.source_image = "/api/uploads/pid_reference.png"
    project.source_kind = "vector_pdf"
    try:
        reconstruct_scan_cad_from_reference(project, tmp_path)
    except ValueError as exc:
        assert "Use vector extraction" in str(exc)
    else:
        raise AssertionError("expected vector PDF routing error")

    project.source_kind = "scanned_pdf"
    try:
        reconstruct_scan_cad_from_reference(project, tmp_path)
    except FileNotFoundError as exc:
        assert "Reference image not found" in str(exc)
    else:
        raise AssertionError("expected missing image error")


def test_scan_cad_uses_hough_fallback_when_centerline_missing(tmp_path, monkeypatch):
    image_path = tmp_path / "pid_reference.png"
    Image.new("L", (300, 200), 255).save(image_path)
    now = datetime.now(timezone.utc)
    project = ProjectState(
        project_id="pid",
        name="demo",
        created_at=now,
        updated_at=now,
        source_kind="scanned_pdf",
        source_image="/api/uploads/pid_reference.png",
        ir=default_ir(),
    )
    monkeypatch.setattr(
        "app.scan_cad._external_vectorizer_entities",
        lambda image, output_dir, warnings: [
            PolylineEntity(
                id="ref_trace_00000",
                layer="reference_trace",
                points=[[0, 0], [10, 0]],
                group="reference_trace",
                tags=["vtracer"],
            )
        ],
    )
    monkeypatch.setattr(
        "app.scan_cad._hough_fallback_entities",
        lambda image, image_width, image_height, canvas_height: [
            LineEntity(
                id="editable_hough_00000",
                layer="editable_linework",
                x1=0,
                y1=0,
                x2=10,
                y2=0,
                group="editable_linework",
                tags=["hough_fallback"],
            )
        ],
    )
    monkeypatch.setattr(
        "app.scan_cad.reconstruct_tables_from_reference",
        lambda project, uploads_dir: (_ for _ in ()).throw(ValueError("no tables")),
    )
    monkeypatch.setattr(
        "app.scan_cad.reconstruct_section_from_reference",
        lambda project, uploads_dir: (_ for _ in ()).throw(ValueError("no section")),
    )

    result = reconstruct_scan_cad_from_reference(project, tmp_path)

    assert result.structured_counts["reference_trace"] == 1
    assert result.structured_counts["editable_linework"] == 1
    assert result.structured_counts["section_view"] == 0
    assert any("Editable centerline fallback used" in warning for warning in result.warnings)
    assert any("Table reconstruction skipped" in warning for warning in result.warnings)
    assert any("Section reconstruction skipped" in warning for warning in result.warnings)


def test_scan_cad_masks_table_trace_during_reconstruction(tmp_path, monkeypatch):
    image_path = tmp_path / "pid_reference.png"
    Image.new("L", (300, 200), 255).save(image_path)
    now = datetime.now(timezone.utc)
    project = ProjectState(
        project_id="pid",
        name="demo",
        created_at=now,
        updated_at=now,
        source_kind="scanned_pdf",
        source_image="/api/uploads/pid_reference.png",
        ir=default_ir(),
    )
    monkeypatch.setattr(
        "app.scan_cad._external_vectorizer_entities",
        lambda image, output_dir, warnings: [
            PolylineEntity(
                id="ref_inside_table",
                layer="reference_trace",
                points=[[10, 10], [12, 12]],
                group="reference_trace",
                tags=["vtracer"],
            ),
            PolylineEntity(
                id="ref_outside_table",
                layer="reference_trace",
                points=[[80, 80], [90, 90]],
                group="reference_trace",
                tags=["vtracer"],
            ),
            PolylineEntity(
                id="editable_outside_table",
                layer="editable_linework",
                points=[[100, 80], [110, 90]],
                group="editable_linework",
                tags=["autotrace"],
            ),
        ],
    )

    def fake_tables(project, uploads_dir):
        table_ir = default_ir()
        table_ir.entities.append(
            LineEntity(
                id="recon_title_h0",
                layer="table",
                x1=8,
                y1=8,
                x2=24,
                y2=8,
                group="title_block",
                tags=["title_block", "grid"],
            )
        )
        return SimpleNamespace(
            ir=table_ir,
            regions=[SimpleNamespace(target="title_block", bbox=(8, 8, 24, 24))],
            warnings=["table warning"],
        )

    monkeypatch.setattr("app.scan_cad.reconstruct_tables_from_reference", fake_tables)
    monkeypatch.setattr(
        "app.scan_cad.reconstruct_section_from_reference",
        lambda project, uploads_dir: (_ for _ in ()).throw(ValueError("no section")),
    )

    result = reconstruct_scan_cad_from_reference(project, tmp_path)

    assert result.trace_count == 2
    assert result.structured_counts["reference_trace"] == 1
    assert result.structured_counts["editable_linework"] == 1
    assert not any(entity.id == "ref_inside_table" for entity in result.ir.entities)
    assert any(entity.id == "recon_title_h0" for entity in result.ir.entities)
    assert any("Masked 1 trace entities" in warning for warning in result.warnings)
    assert "table warning" in result.warnings


def test_scan_cad_low_confidence_section_is_not_merged(tmp_path, monkeypatch):
    image_path = tmp_path / "pid_reference.png"
    Image.new("L", (300, 200), 255).save(image_path)
    now = datetime.now(timezone.utc)
    project = ProjectState(
        project_id="pid",
        name="demo",
        created_at=now,
        updated_at=now,
        source_kind="scanned_pdf",
        source_image="/api/uploads/pid_reference.png",
        ir=default_ir(),
    )
    monkeypatch.setattr(
        "app.scan_cad._external_vectorizer_entities",
        lambda image, output_dir, warnings: [
            PolylineEntity(
                id="editable_line_00000",
                layer="editable_linework",
                points=[[0, 0], [10, 0]],
                group="editable_linework",
                tags=["autotrace"],
            )
        ],
    )
    monkeypatch.setattr(
        "app.scan_cad.reconstruct_tables_from_reference",
        lambda project, uploads_dir: (_ for _ in ()).throw(ValueError("no tables")),
    )

    def fake_section(project, uploads_dir):
        next_ir = project.ir.model_copy(deep=True)
        next_ir.entities.append(
            PolylineEntity(
                id="cv_section_bad",
                layer="geometry",
                points=[[0, 0], [1, 1]],
                group="section_view",
                tags=["cv_section"],
            )
        )
        return SimpleNamespace(ir=next_ir, warnings=["Low section primitive count: 1"])

    monkeypatch.setattr("app.scan_cad.reconstruct_section_from_reference", fake_section)

    result = reconstruct_scan_cad_from_reference(project, tmp_path)

    assert result.structured_counts["section_view"] == 0
    assert not any(entity.group == "section_view" for entity in result.ir.entities)
    assert any("low-confidence CV result" in warning for warning in result.warnings)


def test_scan_cad_base_ir_external_vectorizers_and_hough_helpers(tmp_path, monkeypatch):
    ir = _base_scan_ir(1000, 500, 210, (100, 80, 900, 430))
    assert [entity.id for entity in ir.entities] == ["scan_sheet_border", "scan_drawing_frame"]

    warnings: list[str] = []
    monkeypatch.setattr(
        "app.scan_cad._run_vtracer_entities",
        lambda image_path, output_dir, warnings: [
            PolylineEntity(
                id="ref_trace_00000",
                layer="reference_trace",
                points=[[0, 0], [1, 0]],
                group="reference_trace",
                tags=["vtracer"],
            )
        ],
    )
    monkeypatch.setattr(
        "app.scan_cad._run_autotrace_entities",
        lambda image_path, output_dir, warnings: [
            PolylineEntity(
                id="editable_line_00000",
                layer="editable_linework",
                points=[[0, 0], [0, 1]],
                group="editable_linework",
                tags=["autotrace"],
            )
        ],
    )
    entities = _external_vectorizer_entities(Image.new("L", (40, 30), 255), tmp_path, warnings)
    assert [entity.group for entity in entities] == ["reference_trace", "editable_linework"]
    assert (tmp_path / "scan_normalized_binary.png").exists()
    assert (tmp_path / "scan_normalized_binary.pbm").exists()

    monkeypatch.setattr(
        "app.scan_cad.cv2.HoughLinesP",
        lambda *args, **kwargs: np.array([[[0, 0, 80, 0]], [[0, 0, 80, 0]], [[0, 0, 2, 0]]]),
    )
    hough = _hough_fallback_entities(Image.new("L", (100, 80), 255), 100, 80, 336)
    assert len(hough) == 1
    assert hough[0].group == "editable_linework"
    assert _quantized_line_key(8, 4, 0, 0) == (0, 0, 2, 1)


def test_scan_cad_vtracer_and_autotrace_runners(tmp_path, monkeypatch):
    image = tmp_path / "input.png"
    Image.new("L", (20, 20), 255).save(image)
    warnings: list[str] = []

    def fake_vtracer(input_path, output_path, **kwargs):
        Path(output_path).write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 20 20">'
            '<path d="M1 1 L19 1" fill="none" stroke="black"/></svg>',
            encoding="utf-8",
        )

    monkeypatch.setattr("vtracer.convert_image_to_svg_py", fake_vtracer)
    vtracer_entities = _run_vtracer_entities(image, tmp_path, warnings)
    assert len(vtracer_entities) == 1
    assert vtracer_entities[0].group == "reference_trace"
    assert vtracer_entities[0].stroke_width == REFERENCE_TRACE_STROKE_MM

    monkeypatch.setattr(
        "vtracer.convert_image_to_svg_py",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("vtracer boom")),
    )
    assert _run_vtracer_entities(image, tmp_path, warnings) == []
    assert "vtracer boom" in warnings[-1]

    monkeypatch.setattr("app.scan_cad.shutil.which", lambda name: None)
    assert _run_autotrace_entities(image, tmp_path, warnings) == []
    assert "autotrace not installed" in warnings[-1]

    monkeypatch.setattr("app.scan_cad.shutil.which", lambda name: "/bin/autotrace")
    monkeypatch.setattr(
        "app.scan_cad.subprocess.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(subprocess.TimeoutExpired("autotrace", 90)),
    )
    assert _run_autotrace_entities(image, tmp_path, warnings) == []
    assert "timed out" in warnings[-1]

    monkeypatch.setattr(
        "app.scan_cad.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=2, stdout="", stderr="bad trace"),
    )
    assert _run_autotrace_entities(image, tmp_path, warnings) == []
    assert "bad trace" in warnings[-1]

    def fake_autotrace(command, **kwargs):
        svg_path = Path([part for part in command if part.startswith("--output-file=")][0].split("=", 1)[1])
        svg_path.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 20 20">'
            '<path d="M1 1 L1 19" fill="none" stroke="black"/></svg>',
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("app.scan_cad.subprocess.run", fake_autotrace)
    autotrace_entities = _run_autotrace_entities(image, tmp_path, warnings)
    assert len(autotrace_entities) == 1
    assert autotrace_entities[0].group == "editable_linework"
    assert autotrace_entities[0].stroke_width == EDITABLE_LINEWORK_STROKE_MM


def test_table_ocr_reports_missing_sources_and_grids(tmp_path, monkeypatch):
    now = datetime.now(timezone.utc)
    project = ProjectState(project_id="pid", name="demo", created_at=now, updated_at=now, ir=default_ir())

    try:
        extract_table_ocr_from_reference(project, tmp_path)
    except ValueError as exc:
        assert "before running table OCR" in str(exc)
    else:
        raise AssertionError("expected missing upload error")

    project.source_image = "/api/uploads/missing.png"
    try:
        extract_table_ocr_from_reference(project, tmp_path)
    except FileNotFoundError as exc:
        assert "Reference image not found" in str(exc)
    else:
        raise AssertionError("expected missing image error")

    image_path = tmp_path / "pid_reference.png"
    Image.new("L", (300, 220), 255).save(image_path)
    project.source_image = "/api/uploads/pid_reference.png"
    try:
        extract_table_ocr_from_reference(project, tmp_path)
    except ValueError as exc:
        assert "drawing frame" in str(exc)
    else:
        raise AssertionError("expected missing frame error")

    _write_section_page(image_path)
    try:
        extract_table_ocr_from_reference(project, tmp_path, language_hint="en", engine_hint="tesseract")
    except ValueError as exc:
        assert "table grids" in str(exc)
    else:
        raise AssertionError("expected missing table grid error")

    _write_table_page(image_path)
    monkeypatch.setattr("app.table_ocr.shutil.which", lambda name: None)
    try:
        extract_table_ocr_from_reference(project, tmp_path, language_hint="en", engine_hint="tesseract")
    except ValueError as exc:
        assert "No table OCR engine" in str(exc)
    else:
        raise AssertionError("expected missing OCR engine error")


def test_table_ocr_engine_selection_and_cell_fallbacks(monkeypatch):
    monkeypatch.setattr("app.table_ocr._paddleocr_available", lambda: True)
    assert table_ocr_module._select_table_engine("zh", "auto") == "paddle"
    assert table_ocr_module._select_table_engine("en", "auto") == "tesseract"
    assert table_ocr_module._select_table_engine("zh", "tesseract") == "tesseract"
    assert table_ocr_module._select_table_engine("en", "paddle") == "paddle"

    image = Image.new("RGB", (120, 80), "white")
    warnings: list[str] = []
    monkeypatch.setattr("app.table_ocr._run_paddle_ocr", lambda crop, language: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr("app.table_ocr._select_languages", lambda tesseract: ("chi_sim+eng", ["lang warning"]))
    monkeypatch.setattr("app.table_ocr._run_tesseract_cell", lambda tesseract, crop, languages, warnings: ("fallback", 0.77))

    text, confidence, engine, language, source = table_ocr_module._ocr_cell(
        image,
        (0, 0, 80, 60),
        "zh",
        "paddle",
        "/usr/bin/tesseract",
        warnings,
    )

    assert text == "fallback"
    assert confidence == 0.77
    assert engine == "tesseract"
    assert language == "chi_sim+eng"
    assert source == "table_cell_ocr_tesseract"
    assert any("PaddleOCR failed" in warning for warning in warnings)
    assert "lang warning" in warnings

    text, confidence, engine, language, source = table_ocr_module._ocr_cell(
        image,
        (0, 0, 80, 60),
        "zh",
        "tesseract",
        None,
        [],
    )
    assert (text, confidence, engine, language, source) == ("", 0.0, "none", "zh", "table_cell_ocr_unavailable")


def test_table_ocr_grid_limits_and_tesseract_paths(monkeypatch):
    image = Image.new("RGB", (120, 80), "white")
    warnings: list[str] = []
    grid = table_ocr_module.PixelGrid(
        target="parameter_table",
        label="参数表",
        bbox=(0, 0, 120, 80),
        columns=[0, 40, 80, 120],
        rows=[0, 40, 80],
    )
    monkeypatch.setattr(table_ocr_module, "MAX_TABLE_CELLS", 2)
    monkeypatch.setattr("app.table_ocr._ocr_cell", lambda *args, **kwargs: ("x", 0.5, "mock", "zh", "mock_source"))

    cells = table_ocr_module._ocr_grids(image, 120, 80, [grid], "zh", "paddle", None, warnings)

    assert len(cells) == 2
    assert warnings == ["Table OCR stopped after 2 cells."]

    empty_cells = table_ocr_module._ocr_grids(
        image,
        120,
        80,
        [table_ocr_module.PixelGrid("title_block", "标题栏", (0, 0, 10, 10), [1], [1])],
        "zh",
        "paddle",
        None,
        [],
    )
    assert empty_cells == []

    def timeout_run(*args, **kwargs):
        raise table_ocr_module.subprocess.TimeoutExpired(cmd="tesseract", timeout=20)

    warnings = []
    monkeypatch.setattr("app.table_ocr.subprocess.run", timeout_run)
    assert table_ocr_module._run_tesseract_cell("/usr/bin/tesseract", image, "eng", warnings) == ("", 0.0)
    assert "timed out" in warnings[0]

    def failed_run(*args, **kwargs):
        return type("Result", (), {"returncode": 1, "stdout": "", "stderr": "bad cell"})()

    warnings = []
    monkeypatch.setattr("app.table_ocr.subprocess.run", failed_run)
    assert table_ocr_module._run_tesseract_cell("/usr/bin/tesseract", image, "eng", warnings) == ("", 0.0)
    assert "bad cell" in warnings[0]


def test_reconstruct_section_from_reference_replaces_template_section_group(tmp_path):
    image_path = tmp_path / "pid_reference.png"
    _write_section_page(image_path)
    now = datetime.now(timezone.utc)
    project = ProjectState(
        project_id="pid",
        name="demo",
        created_at=now,
        updated_at=now,
        source_image="/api/uploads/pid_reference.png",
        ir=spur_gear_drawing_ir(),
    )

    result = reconstruct_section_from_reference(project, tmp_path)

    assert result.line_count >= 12
    assert result.hatch_count >= 4
    assert result.region.target == "section_view"
    section_entities = [entity for entity in result.ir.entities if entity.group == "section_view"]
    assert section_entities
    assert all(entity.id.startswith("cv_section_") for entity in section_entities)
    assert any(entity.layer == "hatch" for entity in section_entities)
    assert not any(entity.id == "section_profile" for entity in section_entities)


def test_section_cv_segments_are_topologized_into_closed_primitives():
    segments = _topology_fixture_segments()

    entities = _segments_to_entities(segments, (0, 0, 300, 600), 1000, 700, 294.0)

    assert [entity.id for entity in entities[:2]] == ["cv_section_outer_profile", "cv_section_bore"]
    assert entities[0].type == "polyline"
    assert entities[0].closed is True
    assert entities[1].type == "polyline"
    assert any(entity.layer == "hatch" and "cut_hatch" in entity.tags for entity in entities)
    assert not any("raw_segment" in entity.tags for entity in entities)


def test_section_cv_falls_back_to_non_dimension_raw_segments_when_topology_is_sparse():
    segments = [
        Segment(20, 80, 120, 80, "horizontal"),
        Segment(40, 100, 90, 150, "diag_pos"),
        Segment(2, 20, 2, 280, "vertical"),
    ]

    entities = _segments_to_entities(segments, (10, 20, 160, 340), 1000, 700, 294.0)

    assert entities
    assert all(entity.id.startswith("cv_section_raw_") for entity in entities)
    assert any(entity.layer == "hatch" for entity in entities)
    assert not any(entity.layer == "dimensions" for entity in entities)


def test_section_cv_helper_edges_and_empty_detection_paths():
    assert _detect_section_segments(np.asarray(Image.new("L", (20, 20), 255)), (0, 0, 0, 0)) == []
    assert _detect_section_segments(np.asarray(Image.new("L", (80, 80), 255)), (0, 0, 80, 80)) == []
    assert _raw_line_to_segment(0, 0, 5, 0) is None

    reversed_diag = _raw_line_to_segment(60, 10, 10, 60)
    assert reversed_diag is not None
    assert reversed_diag.kind == "diag_neg"
    assert reversed_diag.x1 == 10

    assert _estimate_section_topology([], 300, 600) is None
    assert _cluster_values([], tolerance=5) == []
    assert _first_horizontal([], min_y=10, min_x=10) is None
    assert _left_outline_x([], 300) == 12
    assert _filtered_hatches(
        [
            Segment(0, 0, 20, 20, "diag_pos"),
            Segment(60, 60, 80, 80, "diag_pos"),
            Segment(60, 60, 80, 60, "horizontal"),
        ],
        (50, 50, 100, 100),
    ) == [Segment(60, 60, 80, 80, "diag_pos")]


def test_section_cv_topology_helpers_pick_body_edges_not_dimension_edges():
    segments = [
        Segment(4, 10, 4, 590, "vertical"),
        Segment(11, 104, 110, 104, "horizontal"),
        Segment(14, 202, 170, 202, "horizontal"),
        Segment(20, 372, 170, 372, "horizontal"),
        Segment(170, 286, 205, 286, "horizontal"),
        Segment(132, 429, 273, 429, "horizontal"),
    ]

    assert _left_outline_x(segments, 300) == 14
    assert _right_extension_x(segments, 286, 168, 300, fallback=180) == 205
    assert _right_extension_x(segments, 320, 168, 300, fallback=180) == 180
    assert _row_segment_start(segments, 429, min_x=50, max_x=220, fallback=150) == 132
    assert _row_segment_start(segments, 500, min_x=50, max_x=220, fallback=150) == 150


def test_section_cv_clips_hatches_to_material_polygons():
    topology = {
        "bbox": (0, 0, 100, 100),
        "hatch_regions": [[(0, 0), (100, 0), (100, 100), (0, 100)]],
    }
    segments = [
        Segment(-10, 20, 30, 60, "diag_pos"),
        Segment(20, 20, 40, 20, "horizontal"),
        Segment(120, 20, 150, 50, "diag_pos"),
    ]

    clipped = _clipped_hatches(segments, topology)

    assert len(clipped) == 1
    assert clipped[0].x1 == 0
    assert clipped[0].y1 == 30
    assert clipped[0].x2 == 30
    assert clipped[0].y2 == 60
    assert _clip_segment_to_polygon(Segment(20, 20, 40, 40, "diag_pos"), topology["hatch_regions"][0]) == [
        Segment(20, 20, 40, 40, "diag_pos")
    ]
    assert _clipped_hatches([Segment(60, 60, 80, 80, "diag_pos")], {"bbox": (50, 50, 100, 100)}) == [
        Segment(60, 60, 80, 80, "diag_pos")
    ]
    assert _clipped_hatches([Segment(60, 60, 80, 80, "diag_pos")], {"bbox": (50, 50, 100, 100), "hatch_regions": [{}]}) == []
    assert _point_in_polygon((50, 50), topology["hatch_regions"][0]) is True
    assert _point_in_polygon((0, 50), topology["hatch_regions"][0]) is True
    assert _point_in_polygon((150, 50), topology["hatch_regions"][0]) is False
    assert _segment_intersection_t(0, 0, 10, 0, 5, -5, 0, 10) == 0.5
    assert _segment_intersection_t(0, 0, 10, 0, 0, 5, 10, 0) is None
    assert _unique_sorted([0.0, 0.000001, 0.4, 0.4, 1.0], tolerance=0.001) == [0.0, 0.4, 1.0]


def _topology_fixture_segments() -> list[Segment]:
    return [
        Segment(20, 180, 20, 390, "vertical"),
        Segment(112, 20, 112, 95, "vertical"),
        Segment(170, 180, 170, 390, "vertical"),
        Segment(180, 130, 180, 540, "vertical"),
        Segment(70, 86, 170, 86, "horizontal"),
        Segment(30, 104, 112, 104, "horizontal"),
        Segment(30, 176, 176, 176, "horizontal"),
        Segment(20, 202, 170, 202, "horizontal"),
        Segment(20, 372, 170, 372, "horizontal"),
        Segment(20, 466, 108, 466, "horizontal"),
        Segment(18, 498, 104, 498, "horizontal"),
        Segment(42, 534, 172, 534, "horizontal"),
        Segment(42, 568, 172, 568, "horizontal"),
        Segment(30, 176, 100, 106, "diag_neg"),
        Segment(50, 176, 120, 106, "diag_neg"),
        Segment(30, 466, 100, 396, "diag_neg"),
        Segment(50, 466, 120, 396, "diag_neg"),
    ]


def _write_table_page(path: Path) -> None:
    img = Image.new("L", (1000, 700), 255)
    draw = ImageDraw.Draw(img)
    draw.rectangle((40, 40, 960, 680), outline=0, width=2)
    draw.rectangle((120, 120, 900, 650), outline=0, width=2)

    draw.rectangle((690, 130, 890, 350), outline=0, width=2)
    for x in range(730, 890, 40):
        draw.line((x, 130, x, 350), fill=0, width=1)
    for y in range(160, 350, 30):
        draw.line((690, y, 890, y), fill=0, width=1)

    draw.rectangle((520, 545, 900, 650), outline=0, width=2)
    for x in range(570, 900, 70):
        draw.line((x, 545, x, 650), fill=0, width=1)
    for y in range(575, 650, 25):
        draw.line((520, y, 900, y), fill=0, width=1)
    img.save(path)


def _write_section_page(path: Path) -> None:
    img = Image.new("L", (1000, 700), 255)
    draw = ImageDraw.Draw(img)
    draw.rectangle((40, 40, 960, 680), outline=0, width=2)
    draw.rectangle((120, 120, 900, 650), outline=0, width=2)

    # Main section view sits in the same frame-relative band used by the CV ROI.
    body = [(250, 245), (310, 245), (325, 270), (365, 270), (365, 455), (325, 455), (310, 485), (250, 485)]
    draw.line(body + [body[0]], fill=0, width=4)
    draw.rectangle((270, 315, 348, 405), outline=0, width=4)
    draw.line((235, 365, 380, 365), fill=0, width=2)
    draw.line((300, 225, 300, 505), fill=0, width=2)
    for offset in range(0, 90, 16):
        draw.line((255 + offset, 300, 305 + offset, 250), fill=0, width=2)
        draw.line((255 + offset, 480, 305 + offset, 430), fill=0, width=2)
    img.save(path)


def _write_scan_cad_page(path: Path) -> None:
    img = Image.new("L", (1000, 700), 255)
    draw = ImageDraw.Draw(img)
    draw.rectangle((40, 40, 960, 680), outline=0, width=2)
    draw.rectangle((120, 120, 900, 650), outline=0, width=2)

    draw.rectangle((690, 130, 890, 350), outline=0, width=2)
    for x in range(730, 890, 40):
        draw.line((x, 130, x, 350), fill=0, width=1)
    for y in range(160, 350, 30):
        draw.line((690, y, 890, y), fill=0, width=1)

    draw.rectangle((520, 545, 900, 650), outline=0, width=2)
    for x in range(570, 900, 70):
        draw.line((x, 545, x, 650), fill=0, width=1)
    for y in range(575, 650, 25):
        draw.line((520, y, 900, y), fill=0, width=1)

    body = [(250, 245), (310, 245), (325, 270), (365, 270), (365, 455), (325, 455), (310, 485), (250, 485)]
    draw.line(body + [body[0]], fill=0, width=4)
    draw.rectangle((270, 315, 348, 405), outline=0, width=4)
    draw.line((235, 365, 380, 365), fill=0, width=2)
    draw.line((300, 225, 300, 505), fill=0, width=2)
    for offset in range(0, 90, 16):
        draw.line((255 + offset, 300, 305 + offset, 250), fill=0, width=2)
        draw.line((255 + offset, 480, 305 + offset, 430), fill=0, width=2)
    img.save(path)
