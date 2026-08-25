from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import ezdxf
import ezdxf.units
from ezdxf import bbox
from ezdxf.addons.drawing import Frontend, RenderContext, layout
from ezdxf.addons.drawing.svg import SVGBackend
from svgelements import SVG, Close, Line, Move, Point
from svgelements import Path as SvgPath

from .models import PolylineEntity

PT_TO_MM = 25.4 / 72.0
DEFAULT_CURVE_STEP_PT = 1.5
MAX_CURVE_STEPS = 96
POINT_TOLERANCE = 1e-4
GEOMETRY_LAYER = "svg_geometry"


@dataclass(frozen=True)
class SvgDxfExport:
    entity_count: int
    source_path_count: int


@dataclass(frozen=True)
class DxfSvgPreview:
    entity_count: int
    width_mm: float
    height_mm: float


def svg_geometry_to_polylines(
    svg_path: Path,
    *,
    layer: str,
    group: str,
    id_prefix: str,
    tags: list[str] | None = None,
    stroke_width: float | None = None,
    max_entities: int | None = None,
    target_width: float | None = None,
) -> list[PolylineEntity]:
    svg = SVG.parse(svg_path, reify=True, ppi=72)
    min_x, max_y = _svg_bounds(svg)
    scale = _svg_scale(svg, target_width)
    entities: list[PolylineEntity] = []
    base_tags = tags or []
    for path in _iter_exportable_paths(svg):
        for points, closed in _iter_polylines(path):
            dxf_points = [_to_dxf_point(point, min_x, max_y, scale) for point in points]
            dxf_points = _dedupe_points(dxf_points)
            if len(dxf_points) < 2:
                continue
            if closed and _same_point(dxf_points[0], dxf_points[-1]):
                dxf_points = dxf_points[:-1]
            if len(dxf_points) < 2:
                continue
            entities.append(
                PolylineEntity(
                    id=f"{id_prefix}_{len(entities):05d}",
                    layer=layer,
                    points=[[round(x, 4), round(y, 4)] for x, y in dxf_points],
                    closed=closed,
                    group=group,
                    tags=[*base_tags, group, "external_vectorizer"],
                    stroke_width=stroke_width,
                )
            )
            if max_entities is not None and len(entities) >= max_entities:
                return entities
    return entities


def export_svg_geometry_to_dxf(svg_path: Path, dxf_path: Path) -> SvgDxfExport:
    """Convert MuPDF-style SVG geometry into a CAD-editable DXF.

    PDF interpretation stays with MuPDF. This converter only consumes the SVG
    geometry that MuPDF emitted, keeps visible stroked paths and non-text filled
    shapes, flattens curves into LWPOLYLINE entities, and writes a DXF in mm.
    """
    svg = SVG.parse(svg_path, reify=True, ppi=72)
    min_x, max_y = _svg_bounds(svg)

    doc = ezdxf.new("R2010")
    doc.units = ezdxf.units.MM
    if GEOMETRY_LAYER not in doc.layers:
        doc.layers.add(GEOMETRY_LAYER, color=7)
    msp = doc.modelspace()

    entity_count = 0
    source_path_count = 0
    for path in _iter_exportable_paths(svg):
        source_path_count += 1
        for points, closed in _iter_polylines(path):
            dxf_points = [_to_dxf_point(point, min_x, max_y) for point in points]
            dxf_points = _dedupe_points(dxf_points)
            if len(dxf_points) < 2:
                continue
            if closed and _same_point(dxf_points[0], dxf_points[-1]):
                dxf_points = dxf_points[:-1]
            if len(dxf_points) < 2:
                continue
            msp.add_lwpolyline(
                dxf_points,
                format="xy",
                close=closed,
                dxfattribs={"layer": GEOMETRY_LAYER, "color": 7},
            )
            entity_count += 1

    doc.saveas(dxf_path)
    return SvgDxfExport(entity_count=entity_count, source_path_count=source_path_count)


def render_dxf_to_svg(dxf_path: Path, svg_path: Path) -> DxfSvgPreview:
    """Render the generated DXF back to SVG for the UI preview panel."""
    doc = ezdxf.readfile(dxf_path)
    modelspace = doc.modelspace()
    entities = list(modelspace)
    if not entities:
        raise ValueError("DXF contains no modelspace entities.")

    extents = bbox.extents(modelspace)
    width = max(float(extents.size.x), 1.0)
    height = max(float(extents.size.y), 1.0)
    backend = SVGBackend()
    Frontend(RenderContext(doc), backend).draw_layout(modelspace, finalize=True)
    page = layout.Page(
        width=width,
        height=height,
        units=layout.Units.mm,
        margins=layout.Margins.all(2),
    )
    svg_path.write_text(backend.get_string(page), encoding="utf-8")
    return DxfSvgPreview(entity_count=len(entities), width_mm=width, height_mm=height)


def _iter_exportable_paths(svg: SVG) -> Iterable[SvgPath]:
    for element in svg.elements():
        if not isinstance(element, SvgPath):
            continue
        if element.values.get("data-text"):
            continue
        if _has_visible_stroke(element) or _has_visible_nonwhite_fill(element):
            yield element


def _svg_bounds(svg: SVG) -> tuple[float, float]:
    viewbox = svg.viewbox
    if viewbox is not None:
        return float(viewbox.x), float(viewbox.y + viewbox.height)
    return 0.0, float(svg.height or 0)


def _has_visible_stroke(path: SvgPath) -> bool:
    stroke = str(getattr(path, "stroke", "") or "").strip().lower()
    if stroke in {"", "none", "null"}:
        return False
    try:
        return float(getattr(path, "stroke_width", 1.0) or 1.0) > 0
    except (TypeError, ValueError):
        return True


def _has_visible_nonwhite_fill(path: SvgPath) -> bool:
    fill = str(getattr(path, "fill", "") or "").strip().lower()
    return fill not in {"", "none", "null", "#fff", "#ffffff", "white"}


def _iter_polylines(path: SvgPath) -> Iterable[tuple[list[Point], bool]]:
    for subpath in path.as_subpaths():
        segments = list(subpath.segments())
        if not segments:
            continue
        closed = any(isinstance(segment, Close) for segment in segments)
        points: list[Point] = []
        for segment in segments:
            if isinstance(segment, Move):
                _append_point(points, segment.end)
                continue
            if isinstance(segment, (Line, Close)):
                if not points and segment.start is not None:
                    _append_point(points, segment.start)
                _append_point(points, segment.end)
                continue
            for point in _sample_curve(segment):
                _append_point(points, point)
        if len(points) >= 2:
            yield points, closed


def _sample_curve(segment) -> Iterable[Point]:
    try:
        length = float(segment.length(error=1e-4))
    except Exception:  # noqa: BLE001 - third-party segment types can fail length estimation
        length = DEFAULT_CURVE_STEP_PT * 8
    steps = max(4, min(MAX_CURVE_STEPS, int(length / DEFAULT_CURVE_STEP_PT) + 1))
    if getattr(segment, "start", None) is not None:
        yield segment.start
    for index in range(1, steps + 1):
        yield segment.point(index / steps)


def _append_point(points: list[Point], point: Point | None) -> None:
    if point is None:
        return
    if not points or abs(points[-1].x - point.x) > POINT_TOLERANCE or abs(points[-1].y - point.y) > POINT_TOLERANCE:
        points.append(point)


def _svg_scale(svg: SVG, target_width: float | None) -> float:
    if target_width is None:
        return PT_TO_MM
    if svg.viewbox is not None and float(svg.viewbox.width) > 0:
        return target_width / float(svg.viewbox.width)
    try:
        width = float(svg.width or 0)
    except (TypeError, ValueError):
        width = 0.0
    return target_width / width if width > 0 else PT_TO_MM


def _to_dxf_point(point: Point, min_x: float, max_y: float, scale: float = PT_TO_MM) -> tuple[float, float]:
    return ((point.x - min_x) * scale, (max_y - point.y) * scale)


def _dedupe_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    deduped: list[tuple[float, float]] = []
    for point in points:
        if not deduped or not _same_point(deduped[-1], point):
            deduped.append(point)
    return deduped


def _same_point(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return abs(a[0] - b[0]) <= POINT_TOLERANCE and abs(a[1] - b[1]) <= POINT_TOLERANCE
