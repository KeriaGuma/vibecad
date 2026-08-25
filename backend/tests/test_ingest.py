"""Source classification (vector vs scanned vs image) tests."""
from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from app.ingest import SourceKind, classify_source


def _vector_pdf(path: Path) -> None:
    document = pymupdf.open()
    page = document.new_page(width=400, height=300)
    # Plenty of real vector geometry plus selectable text.
    for i in range(30):
        y = 20 + i * 8
        page.draw_line(pymupdf.Point(20, y), pymupdf.Point(380, y))
    page.insert_text((40, 280), "cylindrical spur gear LJT01.01")
    document.save(path)
    document.close()


def _scanned_pdf(path: Path, tmp_path: Path) -> None:
    # A single raster image stretched across the whole page, no vector content.
    raster = tmp_path / "scan.png"
    pix_src = pymupdf.open()
    p = pix_src.new_page(width=400, height=300)
    p.draw_rect(p.rect, fill=(0.9, 0.9, 0.9))
    p.get_pixmap(dpi=72).save(raster)
    pix_src.close()

    document = pymupdf.open()
    page = document.new_page(width=400, height=300)
    page.insert_image(page.rect, filename=str(raster))
    document.save(path)
    document.close()


def test_classify_vector_pdf(tmp_path):
    pdf = tmp_path / "vector.pdf"
    _vector_pdf(pdf)

    result = classify_source(pdf)

    assert result.kind is SourceKind.VECTOR_PDF
    assert result.path_count >= 30
    assert result.image_cover == 0.0


def test_classify_scanned_pdf(tmp_path):
    pdf = tmp_path / "scanned.pdf"
    _scanned_pdf(pdf, tmp_path)

    result = classify_source(pdf)

    assert result.kind is SourceKind.SCANNED_PDF
    assert result.image_cover >= 0.8


def test_classify_real_reference_pdf_is_vector():
    sample = Path(__file__).resolve().parents[2] / "data" / "demo_data" / "test.pdf"
    if not sample.exists():
        pytest.skip("sample drawing not available")

    result = classify_source(sample)

    assert result.kind is SourceKind.VECTOR_PDF


def test_classify_image_upload(tmp_path):
    img = tmp_path / "ref.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")  # suffix-based, content not inspected

    result = classify_source(img)

    assert result.kind is SourceKind.IMAGE


def test_classify_rejects_unknown_suffix(tmp_path):
    bad = tmp_path / "ref.txt"
    bad.write_text("hello")

    with pytest.raises(ValueError, match="Unsupported reference file type"):
        classify_source(bad)


def test_classify_surfaces_corrupt_pdf(tmp_path):
    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"not really a pdf")

    with pytest.raises(ValueError, match="Could not classify PDF"):
        classify_source(bad)
