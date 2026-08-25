"""Line A: extract a DrawingIR directly from a vector PDF.

This is the high-fidelity pipeline used when :func:`app.ingest.classify_source`
reports ``vector_pdf``. Every line, curve, rectangle and text span in the PDF is
mapped straight onto an IR entity, so dimension arrows, roughness symbols and
tolerance frames come through as the real geometry the draughtsman drew -- no
computer-vision guessing.

PDF coordinates are y-down with the origin at the top-left; the IR (and the SVG
exporter's ``scale(1,-1)``) is y-up. We therefore flip y, and convert points to
millimetres so the resulting DXF carries honest mm units.
"""
from __future__ import annotations

from pathlib import Path

import pymupdf

from .models import DrawingIR, Layer, LineEntity, PolylineEntity, ProjectState, RectangleEntity, TextEntity
from .reference import _upload_url_to_path
from .vector_semantics import annotate_vector_semantics

# 1 PostScript point = 1/72 inch; the IR is in millimetres.
PT_TO_MM = 25.4 / 72.0
# Cubic Bezier flattening resolution.
BEZIER_STEPS = 16
# Drop sub-micron noise segments that some PDFs emit.
MIN_SEGMENT_MM = 1e-3
MIN_STROKE_WIDTH_MM = 0.05
TEXT_DUPLICATE_PATH_THRESHOLD = 1000
LARGE_TEXT_HEIGHT_MM = 3.0

GEOMETRY_LAYER = "geometry"
TEXT_LAYER = "text"


def extract_drawing_ir(pdf_path: Path, page_number: int = 0) -> DrawingIR:
    """Build a DrawingIR from the vector content of one PDF page."""
    try:
        document = pymupdf.open(pdf_path)
    except Exception as exc:  # noqa: BLE001 - surface any PyMuPDF open failure to the caller
        raise ValueError(f"Could not open PDF for vector extraction: {exc}") from exc

    try:
        if not 0 <= page_number < document.page_count:
            raise ValueError(f"Page {page_number} is out of range for this PDF.")
        page = document.load_page(page_number)
        page_height = page.rect.height
        entities: list = []
        counter = _Counter()
        _extract_paths(page, page_height, entities, counter)
        _extract_text(
            page,
            page_height,
            entities,
            counter,
            drop_small_text=len(entities) >= TEXT_DUPLICATE_PATH_THRESHOLD,
        )
    finally:
        document.close()

    ir = DrawingIR(
        units="mm",
        layers=[Layer(name=GEOMETRY_LAYER, color="white"), Layer(name=TEXT_LAYER, color="white")],
        entities=entities,
        notes=[f"Imported from vector PDF: {Path(pdf_path).name} (page {page_number + 1})."],
    )
    return annotate_vector_semantics(ir)


def reconstruct_vector_from_reference(project: ProjectState, uploads_dir: Path) -> DrawingIR:
    """Extract a DrawingIR from the project's uploaded vector PDF.

    Mirrors the raster ``reconstruct_*_from_reference`` helpers so the API layer
    can dispatch uniformly. Only valid for ``vector_pdf`` sources.
    """
    if not project.source_file:
        raise ValueError("Upload a PDF before running vector extraction.")
    if project.source_kind != "vector_pdf":
        raise ValueError("Vector extraction only applies to vector PDFs.")

    source_path = _upload_url_to_path(project.source_file, uploads_dir)
    if not source_path.exists():
        raise FileNotFoundError("Reference PDF not found")
    return extract_drawing_ir(source_path)


class _Counter:
    def __init__(self) -> None:
        self._n = 0

    def next(self, prefix: str) -> str:
        self._n += 1
        return f"{prefix}_{self._n:05d}"


def _pt(x: float, y: float, page_height: float) -> tuple[float, float]:
    """Flip y and convert points -> millimetres, rounded for compact output."""
    return (round(x * PT_TO_MM, 3), round((page_height - y) * PT_TO_MM, 3))


def _seg_len(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _stroke_width(path: dict) -> float:
    width_pt = float(path.get("width") or 0)
    return round(max(width_pt * PT_TO_MM, MIN_STROKE_WIDTH_MM), 3)


def _extract_paths(page: pymupdf.Page, page_height: float, entities: list, counter: _Counter) -> None:
    for path in page.get_drawings():
        stroke_width = _stroke_width(path)
        for item in path["items"]:
            kind = item[0]
            if kind == "l":
                a = _pt(item[1].x, item[1].y, page_height)
                b = _pt(item[2].x, item[2].y, page_height)
                if _seg_len(a, b) < MIN_SEGMENT_MM:
                    continue
                entities.append(
                    LineEntity(
                        id=counter.next("v_line"),
                        layer=GEOMETRY_LAYER,
                        x1=a[0],
                        y1=a[1],
                        x2=b[0],
                        y2=b[1],
                        stroke_width=stroke_width,
                    )
                )
            elif kind == "re":
                rect = item[1]
                x0, y0 = _pt(rect.x0, rect.y1, page_height)  # y1 is the larger pdf-y -> bottom edge once flipped
                entities.append(
                    RectangleEntity(
                        id=counter.next("v_rect"),
                        layer=GEOMETRY_LAYER,
                        x=x0,
                        y=y0,
                        width=round(rect.width * PT_TO_MM, 3),
                        height=round(rect.height * PT_TO_MM, 3),
                        stroke_width=stroke_width,
                    )
                )
            elif kind == "c":
                pts = _flatten_cubic(item[1], item[2], item[3], item[4], page_height)
                if len(pts) >= 2:
                    entities.append(
                        PolylineEntity(
                            id=counter.next("v_curve"),
                            layer=GEOMETRY_LAYER,
                            points=pts,
                            closed=False,
                            stroke_width=stroke_width,
                        )
                    )
            elif kind == "qu":
                quad = item[1]
                pts = [_pt(p.x, p.y, page_height) for p in (quad.ul, quad.ur, quad.lr, quad.ll)]
                entities.append(
                    PolylineEntity(
                        id=counter.next("v_quad"),
                        layer=GEOMETRY_LAYER,
                        points=[list(p) for p in pts],
                        closed=True,
                        stroke_width=stroke_width,
                    )
                )


def _flatten_cubic(p0, p1, p2, p3, page_height: float) -> list[list[float]]:
    points: list[list[float]] = []
    for i in range(BEZIER_STEPS + 1):
        t = i / BEZIER_STEPS
        mt = 1 - t
        x = mt**3 * p0.x + 3 * mt**2 * t * p1.x + 3 * mt * t**2 * p2.x + t**3 * p3.x
        y = mt**3 * p0.y + 3 * mt**2 * t * p1.y + 3 * mt * t**2 * p2.y + t**3 * p3.y
        points.append(list(_pt(x, y, page_height)))
    return points


def _extract_text(
    page: pymupdf.Page,
    page_height: float,
    entities: list,
    counter: _Counter,
    drop_small_text: bool = False,
) -> None:
    data = page.get_text("dict")
    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            rotation = _line_rotation(line.get("dir", (1.0, 0.0)))
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if not text:
                    continue
                height = round(span.get("size", 10.0) * PT_TO_MM, 3)
                if drop_small_text and height < LARGE_TEXT_HEIGHT_MM:
                    continue
                ox, oy = span.get("origin", (span["bbox"][0], span["bbox"][3]))
                x, y = _pt(ox, oy, page_height)
                entities.append(
                    TextEntity(
                        id=counter.next("v_text"),
                        layer=TEXT_LAYER,
                        x=x,
                        y=y,
                        text=text,
                        height=height,
                        rotation=rotation,
                    )
                )


def _line_rotation(direction: tuple[float, float]) -> float:
    import math

    # PDF writing direction is in y-down space; negate dy to match the y-up IR.
    angle = math.degrees(math.atan2(-direction[1], direction[0]))
    return round(angle, 2)
