"""Reference PDF/image ingestion and first-pass region analysis tests."""
from __future__ import annotations

import struct
from datetime import datetime, timezone
from pathlib import Path

import pymupdf
import pytest
from PIL import Image, ImageDraw

from app import reference
from app.ingest import SourceClassification, SourceKind
from app.models import ProjectState, default_ir


def png_bytes(width: int = 320, height: int = 200) -> bytes:
    return b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + struct.pack(">II", width, height) + b"\x08\x02\x00\x00\x00"


def gif_bytes(width: int = 320, height: int = 200) -> bytes:
    return b"GIF89a" + struct.pack("<HH", width, height) + b"\x00\x00"


def jpeg_bytes(width: int = 320, height: int = 200) -> bytes:
    return b"\xff\xd8\xff\xc0\x00\x11\x08" + struct.pack(">HH", height, width) + b"\x03\x01\x11\x00"


def test_save_image_upload_creates_source_and_preview(tmp_path):
    upload = reference.save_reference_upload("pid", "drawing.png", png_bytes(640, 480), tmp_path)

    assert upload.source_file == "/api/uploads/pid_source.png"
    assert upload.source_image == "/api/uploads/pid_reference.png"
    assert upload.image_width == 640
    assert upload.image_height == 480
    assert (tmp_path / "pid_source.png").exists()
    assert (tmp_path / "pid_reference.png").exists()


def test_save_pdf_upload_renders_first_page_to_png(tmp_path, monkeypatch):
    def fake_render(source_path, preview_path):
        assert source_path.name == "pid_source.pdf"
        preview_path.write_bytes(png_bytes(1200, 900))

    monkeypatch.setattr(reference, "_render_pdf_first_page", fake_render)
    monkeypatch.setattr(
        reference,
        "classify_source",
        lambda path: SourceClassification(SourceKind.VECTOR_PDF, 999, 0, 0.0, "stub"),
    )
    upload = reference.save_reference_upload("pid", "drawing.pdf", b"%PDF fake", tmp_path)

    assert upload.source_file == "/api/uploads/pid_source.pdf"
    assert upload.source_image == "/api/uploads/pid_reference.png"
    assert upload.image_width == 1200
    assert upload.image_height == 900
    assert upload.source_kind == "vector_pdf"


def test_unsupported_reference_file_type_raises(tmp_path):
    with pytest.raises(ValueError, match="Unsupported reference"):
        reference.save_reference_upload("pid", "notes.txt", b"hello", tmp_path)


def test_analyze_reference_returns_five_candidate_boxes(tmp_path):
    now = datetime.now(timezone.utc)
    project = ProjectState(project_id="pid", name="demo", created_at=now, updated_at=now, ir=default_ir())
    image_path = tmp_path / "pid_reference.png"
    image_path.write_bytes(png_bytes(1000, 800))
    project.source_image = "/api/uploads/pid_reference.png"

    report = reference.analyze_reference(project, tmp_path)

    assert report.project_id == "pid"
    assert report.image_width == 1000
    assert report.image_height == 800
    assert {box.target for box in report.boxes} == {
        "title_block",
        "parameter_table",
        "section_view",
        "circular_view",
        "dimensions",
    }
    assert all(0 <= box.x <= 1 and 0 <= box.y <= 1 for box in report.boxes)


def test_analyze_reference_uses_image_projection_for_valid_drawing(tmp_path):
    now = datetime.now(timezone.utc)
    project = ProjectState(project_id="pid", name="demo", created_at=now, updated_at=now, ir=default_ir())
    image_path = tmp_path / "pid_reference.png"
    _write_synthetic_mechanical_page(image_path)
    project.source_image = "/api/uploads/pid_reference.png"

    report = reference.analyze_reference(project, tmp_path)
    boxes = {box.target: box for box in report.boxes}

    assert set(boxes) == {
        "title_block",
        "parameter_table",
        "section_view",
        "circular_view",
        "dimensions",
    }
    assert all(box.source == "scan_preprocess_layout_v3" for box in boxes.values())
    assert report.frame is not None
    assert report.deskew_angle == 0
    assert boxes["parameter_table"].x > 0.58
    assert boxes["parameter_table"].y < 0.35
    assert boxes["title_block"].x > 0.45
    assert boxes["title_block"].y > 0.72
    assert boxes["section_view"].x < 0.30
    assert boxes["circular_view"].x > boxes["section_view"].x


def test_analyze_reference_normalizes_rotated_portrait_scan(tmp_path):
    now = datetime.now(timezone.utc)
    project = ProjectState(project_id="pid", name="demo", created_at=now, updated_at=now, ir=default_ir())
    source_path = tmp_path / "source.png"
    image_path = tmp_path / "pid_reference.png"
    _write_synthetic_mechanical_page(source_path)
    Image.open(source_path).rotate(-90, expand=True).save(image_path)
    project.source_image = "/api/uploads/pid_reference.png"

    report = reference.analyze_reference(project, tmp_path)
    boxes = {box.target: box for box in report.boxes}

    assert report.image_width > report.image_height
    assert report.deskew_angle == -90.0
    assert boxes["parameter_table"].x > 0.58
    assert boxes["title_block"].y > 0.72
    assert boxes["section_view"].x < boxes["circular_view"].x


def test_analyze_reference_does_not_label_view_as_parameter_table(tmp_path):
    now = datetime.now(timezone.utc)
    project = ProjectState(project_id="pid", name="demo", created_at=now, updated_at=now, ir=default_ir())
    source_path = tmp_path / "source.png"
    image_path = tmp_path / "pid_reference.png"
    _write_title_only_mechanical_page(source_path)
    Image.open(source_path).rotate(-90, expand=True).save(image_path)
    project.source_image = "/api/uploads/pid_reference.png"

    report = reference.analyze_reference(project, tmp_path)
    boxes = {box.target: box for box in report.boxes}

    assert report.deskew_angle == -90.0
    assert "title_block" in boxes
    assert "parameter_table" not in boxes


def test_analyze_reference_requires_upload(tmp_path):
    now = datetime.now(timezone.utc)
    project = ProjectState(
        project_id="pid",
        name="demo",
        created_at=now,
        updated_at=now,
        ir=default_ir(),
    )

    with pytest.raises(ValueError, match="Upload a PDF or image"):
        reference.analyze_reference(project, tmp_path)


def test_analyze_reference_missing_file_raises(tmp_path):
    now = datetime.now(timezone.utc)
    project = ProjectState(
        project_id="pid",
        name="demo",
        created_at=now,
        updated_at=now,
        source_image="/api/uploads/missing.png",
        ir=default_ir(),
    )

    with pytest.raises(FileNotFoundError, match="Reference image not found"):
        reference.analyze_reference(project, tmp_path)


def test_analyze_reference_rejects_invalid_upload_url(tmp_path):
    now = datetime.now(timezone.utc)
    project = ProjectState(
        project_id="pid",
        name="demo",
        created_at=now,
        updated_at=now,
        source_image="/bad/path.png",
        ir=default_ir(),
    )

    with pytest.raises(ValueError, match="Invalid reference image URL"):
        reference.analyze_reference(project, tmp_path)


def test_image_size_supports_gif_and_jpeg(tmp_path):
    gif = tmp_path / "sample.gif"
    jpg = tmp_path / "sample.jpg"
    gif.write_bytes(gif_bytes(111, 222))
    jpg.write_bytes(jpeg_bytes(333, 444))

    assert reference.image_size(gif) == (111, 222)
    assert reference.image_size(jpg) == (333, 444)


def test_image_size_returns_none_for_unknown_or_invalid_headers(tmp_path):
    files = {
        "bad.bin": b"not an image",
        "bad.png": b"\x89PNG\r\n\x1a\n",
        "bad.gif": b"GIF89a",
        "bad.jpg": b"\xff\xd8\xff\xc0\x00\x01",
    }
    for name, payload in files.items():
        path = tmp_path / name
        path.write_bytes(payload)
        assert reference.image_size(path) == (None, None)


def _write_single_page_pdf(path: Path) -> None:
    document = pymupdf.open()
    page = document.new_page(width=400, height=300)
    page.insert_text((72, 144), "reference drawing")
    document.save(path)
    document.close()


def test_render_pdf_first_page_renders_png(tmp_path):
    source = tmp_path / "source.pdf"
    preview = tmp_path / "preview.png"
    _write_single_page_pdf(source)

    reference._render_pdf_first_page(source, preview)

    assert preview.exists()
    # 300 dpi over a 400x300pt page -> 1667x1250px (PyMuPDF rounds to the rendered pixmap).
    assert reference.image_size(preview) == (1667, 1250)


def test_render_pdf_first_page_surfaces_open_error(tmp_path):
    source = tmp_path / "source.pdf"
    source.write_bytes(b"not really a pdf")

    with pytest.raises(ValueError, match="Could not render PDF first page"):
        reference._render_pdf_first_page(source, tmp_path / "preview.png")


def _write_synthetic_mechanical_page(path: Path) -> None:
    img = Image.new("L", (1000, 700), 255)
    draw = ImageDraw.Draw(img)
    draw.rectangle((40, 40, 960, 680), outline=0, width=2)
    draw.rectangle((120, 120, 900, 650), outline=0, width=2)

    # Parameter table: dense grid in the upper-right drawing frame.
    draw.rectangle((690, 130, 890, 350), outline=0, width=2)
    for x in range(730, 890, 40):
        draw.line((x, 130, x, 350), fill=0, width=1)
    for y in range(160, 350, 30):
        draw.line((690, y, 890, y), fill=0, width=1)

    # Title block: wide grid in the lower-right drawing frame.
    draw.rectangle((520, 545, 900, 650), outline=0, width=2)
    for x in range(570, 900, 70):
        draw.line((x, 545, x, 650), fill=0, width=1)
    for y in range(575, 650, 25):
        draw.line((520, y, 900, y), fill=0, width=1)

    # Section view and circular view stand-ins.
    draw.rectangle((185, 235, 380, 555), outline=0, width=4)
    draw.line((170, 395, 410, 395), fill=0, width=2)
    draw.ellipse((455, 285, 615, 445), outline=0, width=4)
    draw.line((435, 365, 635, 365), fill=0, width=2)
    img.save(path)


def _write_title_only_mechanical_page(path: Path) -> None:
    img = Image.new("L", (1000, 700), 255)
    draw = ImageDraw.Draw(img)
    draw.rectangle((40, 40, 960, 680), outline=0, width=2)
    draw.rectangle((120, 120, 900, 650), outline=0, width=2)

    # Upper-right view-like geometry, not a table grid.
    draw.ellipse((610, 180, 820, 380), outline=0, width=4)
    draw.ellipse((660, 230, 760, 330), outline=0, width=3)
    for offset in range(0, 160, 32):
        draw.line((610 + offset, 370, 660 + offset, 410), fill=0, width=2)

    # Main left view and a real title block.
    draw.rectangle((180, 220, 440, 510), outline=0, width=4)
    draw.line((155, 365, 470, 365), fill=0, width=2)
    draw.rectangle((520, 545, 900, 650), outline=0, width=2)
    for x in range(570, 900, 70):
        draw.line((x, 545, x, 650), fill=0, width=1)
    for y in range(575, 650, 25):
        draw.line((520, y, 900, y), fill=0, width=1)
    img.save(path)
