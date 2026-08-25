"""Source classification: decide how an uploaded reference should be processed.

A reference file flows down one of two pipelines:

* ``vector_pdf``  -> geometry is extracted directly from the PDF vector content
  (lines/arcs/text), bypassing the raster computer-vision path entirely.
* ``scanned_pdf`` / ``image`` -> the file is rasterised and handed to the
  existing CV region-detection + reconstruction path.

This module only answers the *which pipeline* question. The branches themselves
live elsewhere.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pymupdf

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif", ".tiff"}

# A page is treated as vector once it carries at least this many path primitives
# (single line/curve/rect segments). Real drawings have hundreds-to-thousands;
# a scanned page wrapped in a PDF has essentially none.
VECTOR_PATH_MIN = 40
# ...or this many characters of selectable text.
VECTOR_TEXT_MIN = 20
# A single embedded image covering at least this fraction of the page marks a
# scan, even if an OCR text layer is present.
SCAN_IMAGE_COVER = 0.80
# Below this path count we don't trust an otherwise image-dominated page to be
# vector, regardless of any OCR text layer.
SCAN_PATH_MAX = 50


class SourceKind(str, Enum):
    VECTOR_PDF = "vector_pdf"
    SCANNED_PDF = "scanned_pdf"
    IMAGE = "image"


@dataclass(frozen=True)
class SourceClassification:
    kind: SourceKind
    path_count: int
    text_len: int
    image_cover: float
    reason: str


def classify_source(path: Path) -> SourceClassification:
    """Classify a saved reference file into one of the processing pipelines."""
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return SourceClassification(
            kind=SourceKind.IMAGE,
            path_count=0,
            text_len=0,
            image_cover=1.0,
            reason="raster image upload",
        )
    if suffix != ".pdf":
        raise ValueError("Unsupported reference file type. Upload a PDF or image.")

    try:
        document = pymupdf.open(path)
    except Exception as exc:  # noqa: BLE001 - surface any PyMuPDF open failure to the caller
        raise ValueError(f"Could not classify PDF: {exc}") from exc

    try:
        if document.page_count < 1:
            raise ValueError("Could not classify PDF: the document has no pages.")
        page = document.load_page(0)
        path_count = _count_path_primitives(page)
        text_len = len(page.get_text("text").strip())
        image_cover = _max_image_cover(page)
    finally:
        document.close()

    kind, reason = _decide(path_count, text_len, image_cover)
    return SourceClassification(
        kind=kind,
        path_count=path_count,
        text_len=text_len,
        image_cover=round(image_cover, 4),
        reason=reason,
    )


def _decide(path_count: int, text_len: int, image_cover: float) -> tuple[SourceKind, str]:
    # Check the full-page-image signal first so a scan carrying an OCR text
    # layer is still routed to the raster pipeline.
    if image_cover >= SCAN_IMAGE_COVER and path_count < SCAN_PATH_MAX:
        return SourceKind.SCANNED_PDF, (
            f"page-covering image ({image_cover:.0%}) with few vector paths ({path_count})"
        )
    if path_count >= VECTOR_PATH_MIN:
        return SourceKind.VECTOR_PDF, f"{path_count} vector path primitives"
    if text_len >= VECTOR_TEXT_MIN:
        return SourceKind.VECTOR_PDF, f"{text_len} characters of selectable text"
    return SourceKind.SCANNED_PDF, (
        f"insufficient vector content (paths={path_count}, text={text_len}); defaulting to scanned"
    )


def _count_path_primitives(page: pymupdf.Page) -> int:
    return sum(len(path["items"]) for path in page.get_drawings())


def _max_image_cover(page: pymupdf.Page) -> float:
    page_area = abs(page.rect.width * page.rect.height)
    if page_area <= 0:
        return 0.0
    best = 0.0
    for info in page.get_image_info():
        bbox = pymupdf.Rect(info["bbox"])
        cover = abs(bbox.width * bbox.height) / page_area
        best = max(best, cover)
    return min(best, 1.0)
