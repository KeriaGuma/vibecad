from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees
from pathlib import Path

import cv2
import numpy as np

from .models import DrawingIR, Layer, LineEntity, PolylineEntity, ProjectState
from .reconstruct import CANVAS_WIDTH_MM, LayoutRegion, _px_x_to_cad, _px_y_to_cad
from .reference import _detect_inner_frame, _preprocess_reference_image, _upload_url_to_path


@dataclass(frozen=True)
class SectionReconstruction:
    ir: DrawingIR
    region: LayoutRegion
    line_count: int
    hatch_count: int
    warnings: list[str]


@dataclass(frozen=True)
class Segment:
    x1: float
    y1: float
    x2: float
    y2: float
    kind: str


def reconstruct_section_from_reference(project: ProjectState, uploads_dir: Path) -> SectionReconstruction:
    if not project.source_image:
        raise ValueError("Upload a PDF or image before reconstructing the section view.")

    image_path = _upload_url_to_path(project.source_image, uploads_dir)
    if not image_path.exists():
        raise FileNotFoundError("Reference image not found")

    processed = _preprocess_reference_image(image_path)
    gray = np.asarray(processed.image.convert("L"))
    image_height, image_width = gray.shape
    canvas_height = CANVAS_WIDTH_MM * image_height / image_width
    dark = processed.dark
    frame = _detect_inner_frame(dark)
    if frame is None:
        raise ValueError("Could not detect drawing frame.")

    roi = _section_roi(frame, image_width, image_height)
    segments = _detect_section_segments(gray, roi)
    if not segments:
        raise ValueError("Could not detect section-view linework.")

    entities = _segments_to_entities(segments, roi, image_width, image_height, canvas_height)
    warnings: list[str] = []
    hatch_count = sum(1 for entity in entities if "cut_hatch" in entity.tags)
    if len(entities) < 6:
        warnings.append(f"Low section primitive count: {len(entities)}")
    if hatch_count < 4:
        warnings.append(f"Low hatch line count: {hatch_count}")

    next_ir = project.ir.model_copy(deep=True)
    next_ir.entities = [entity for entity in next_ir.entities if entity.group != "section_view"]
    _ensure_layers(next_ir)
    next_ir.entities.extend(entities)
    next_ir.notes = [
        *next_ir.notes,
        "Section view reconstructed from reference image with OpenCV Hough line detection.",
    ]

    x1, y1, x2, y2 = roi
    region = LayoutRegion(
        target="section_view",
        bbox=(
            _px_x_to_cad(x1, image_width),
            _px_y_to_cad(y2, image_height, canvas_height),
            _px_x_to_cad(x2, image_width),
            _px_y_to_cad(y1, image_height, canvas_height),
        ),
    )
    return SectionReconstruction(
        ir=next_ir,
        region=region,
        line_count=len(entities),
        hatch_count=hatch_count,
        warnings=warnings,
    )


def _section_roi(frame: tuple[int, int, int, int], image_width: int, image_height: int) -> tuple[int, int, int, int]:
    fx1, fy1, fx2, fy2 = frame
    fw = max(fx2 - fx1, 1)
    fh = max(fy2 - fy1, 1)
    return (
        max(0, min(image_width - 1, round(fx1 + fw * 0.14))),
        max(0, min(image_height - 1, round(fy1 + fh * 0.17))),
        max(1, min(image_width, round(fx1 + fw * 0.34))),
        max(1, min(image_height, round(fy1 + fh * 0.78))),
    )


def _detect_section_segments(gray: np.ndarray, roi: tuple[int, int, int, int]) -> list[Segment]:
    x1, y1, x2, y2 = roi
    crop = gray[y1:y2, x1:x2]
    if crop.size == 0:
        return []

    blurred = cv2.GaussianBlur(crop, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    raw_lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=35, minLineLength=22, maxLineGap=5)
    if raw_lines is None:
        return []

    raw_segments: list[Segment] = []
    for x1_raw, y1_raw, x2_raw, y2_raw in raw_lines[:, 0, :]:
        segment = _raw_line_to_segment(float(x1_raw), float(y1_raw), float(x2_raw), float(y2_raw))
        if segment is not None:
            raw_segments.append(segment)
    return _merge_segments(raw_segments)


def _raw_line_to_segment(x1: float, y1: float, x2: float, y2: float) -> Segment | None:
    dx = x2 - x1
    dy = y2 - y1
    length = (dx * dx + dy * dy) ** 0.5
    if length < 28:
        return None

    angle = abs(degrees(atan2(dy, dx))) % 180
    if min(angle, 180 - angle) <= 8:
        y = (y1 + y2) / 2
        return Segment(min(x1, x2), y, max(x1, x2), y, "horizontal")
    if abs(angle - 90) <= 8:
        x = (x1 + x2) / 2
        return Segment(x, min(y1, y2), x, max(y1, y2), "vertical")
    if abs(angle - 45) <= 9 or abs(angle - 135) <= 9:
        if x2 < x1:
            x1, y1, x2, y2 = x2, y2, x1, y1
        kind = "diag_pos" if y2 > y1 else "diag_neg"
        return Segment(x1, y1, x2, y2, kind)
    return None


def _merge_segments(segments: list[Segment]) -> list[Segment]:
    merged: list[Segment] = []
    for kind in ["horizontal", "vertical", "diag_pos", "diag_neg"]:
        group = [segment for segment in segments if segment.kind == kind]
        if kind == "horizontal":
            merged.extend(_merge_axis_aligned(group, axis="y"))
        elif kind == "vertical":
            merged.extend(_merge_axis_aligned(group, axis="x"))
        else:
            merged.extend(_merge_diagonal(group))
    return sorted(merged, key=lambda item: (item.y1, item.x1, item.y2, item.x2, item.kind))


def _merge_axis_aligned(segments: list[Segment], axis: str) -> list[Segment]:
    if not segments:
        return []
    tolerance = 3.5
    interval_gap = 8.0
    if axis == "y":
        sorted_segments = sorted(segments, key=lambda item: (item.y1, item.x1))

        def coord(item: Segment) -> float:
            return item.y1

        def interval(item: Segment) -> tuple[float, float]:
            return item.x1, item.x2

    else:
        sorted_segments = sorted(segments, key=lambda item: (item.x1, item.y1))

        def coord(item: Segment) -> float:
            return item.x1

        def interval(item: Segment) -> tuple[float, float]:
            return item.y1, item.y2

    clusters: list[list[Segment]] = []
    for segment in sorted_segments:
        if not clusters or abs(coord(segment) - np.median([coord(item) for item in clusters[-1]])) > tolerance:
            clusters.append([segment])
        else:
            clusters[-1].append(segment)

    out: list[Segment] = []
    for cluster in clusters:
        center = float(np.median([coord(item) for item in cluster]))
        intervals = sorted(interval(item) for item in cluster)
        start, end = intervals[0]
        for next_start, next_end in intervals[1:]:
            if next_start <= end + interval_gap:
                end = max(end, next_end)
            else:
                out.append(_axis_segment(axis, center, start, end))
                start, end = next_start, next_end
        out.append(_axis_segment(axis, center, start, end))
    return [segment for segment in out if _segment_length(segment) >= 18]


def _axis_segment(axis: str, center: float, start: float, end: float) -> Segment:
    if axis == "y":
        return Segment(start, center, end, center, "horizontal")
    return Segment(center, start, center, end, "vertical")


def _merge_diagonal(segments: list[Segment]) -> list[Segment]:
    if not segments:
        return []
    tolerance = 4.0
    interval_gap = 7.0
    key = (lambda item: item.y1 - item.x1) if segments[0].kind == "diag_pos" else (lambda item: item.y1 + item.x1)
    sorted_segments = sorted(segments, key=lambda item: (key(item), item.x1))
    clusters: list[list[Segment]] = []
    for segment in sorted_segments:
        if not clusters or abs(key(segment) - np.median([key(item) for item in clusters[-1]])) > tolerance:
            clusters.append([segment])
        else:
            clusters[-1].append(segment)

    out: list[Segment] = []
    for cluster in clusters:
        c = float(np.median([key(item) for item in cluster]))
        intervals = sorted((min(item.x1, item.x2), max(item.x1, item.x2)) for item in cluster)
        start, end = intervals[0]
        for next_start, next_end in intervals[1:]:
            if next_start <= end + interval_gap:
                end = max(end, next_end)
            else:
                out.append(_diagonal_segment(cluster[0].kind, c, start, end))
                start, end = next_start, next_end
        out.append(_diagonal_segment(cluster[0].kind, c, start, end))
    return [segment for segment in out if _segment_length(segment) >= 18]


def _diagonal_segment(kind: str, c: float, x1: float, x2: float) -> Segment:
    if kind == "diag_pos":
        return Segment(x1, x1 + c, x2, x2 + c, kind)
    return Segment(x1, c - x1, x2, c - x2, kind)


def _segment_length(segment: Segment) -> float:
    dx = segment.x2 - segment.x1
    dy = segment.y2 - segment.y1
    return (dx * dx + dy * dy) ** 0.5


def _segments_to_entities(
    segments: list[Segment],
    roi: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    canvas_height: float,
) -> list[LineEntity | PolylineEntity]:
    x_offset, y_offset, x2, y2 = roi
    crop_width = max(x2 - x_offset, 1)
    crop_height = max(y2 - y_offset, 1)
    topology = _estimate_section_topology(segments, crop_width, crop_height)
    if topology is None:
        return _raw_segments_to_entities(segments, roi, image_width, image_height, canvas_height)

    def cad_point(point: tuple[float, float]) -> list[float]:
        return [
            _px_x_to_cad(x_offset + point[0], image_width),
            _px_y_to_cad(y_offset + point[1], image_height, canvas_height),
        ]

    entities: list[LineEntity | PolylineEntity] = [
        PolylineEntity(
            id="cv_section_outer_profile",
            layer="geometry",
            points=[cad_point(point) for point in topology["outer"]],
            closed=True,
            group="section_view",
            tags=["section_view", "cv_section", "outline", "topology"],
        ),
        PolylineEntity(
            id="cv_section_bore",
            layer="geometry",
            points=[cad_point(point) for point in topology["bore"]],
            closed=True,
            group="section_view",
            tags=["section_view", "cv_section", "outline", "topology", "bore"],
        ),
    ]
    for idx, segment in enumerate(_clipped_hatches(segments, topology)):
        entities.append(
            LineEntity(
                id=f"cv_section_hatch_{idx:03d}",
                layer="hatch",
                x1=_px_x_to_cad(x_offset + segment.x1, image_width),
                y1=_px_y_to_cad(y_offset + segment.y1, image_height, canvas_height),
                x2=_px_x_to_cad(x_offset + segment.x2, image_width),
                y2=_px_y_to_cad(y_offset + segment.y2, image_height, canvas_height),
                group="section_view",
                tags=["section_view", "cv_section", "cut_hatch", "topology"],
            )
        )
    return entities


def _raw_segments_to_entities(
    segments: list[Segment],
    roi: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    canvas_height: float,
) -> list[LineEntity]:
    x_offset, y_offset, x2, y2 = roi
    crop_width = max(x2 - x_offset, 1)
    crop_height = max(y2 - y_offset, 1)
    entities: list[LineEntity] = []
    for idx, segment in enumerate(segments):
        classification = _classify_segment(segment, crop_width, crop_height)
        if classification == "dimension_like":
            continue
        layer = "hatch" if classification == "cut_hatch" else "geometry"
        entities.append(
            LineEntity(
                id=f"cv_section_raw_{idx:03d}",
                layer=layer,
                x1=_px_x_to_cad(x_offset + segment.x1, image_width),
                y1=_px_y_to_cad(y_offset + segment.y1, image_height, canvas_height),
                x2=_px_x_to_cad(x_offset + segment.x2, image_width),
                y2=_px_y_to_cad(y_offset + segment.y2, image_height, canvas_height),
                group="section_view",
                tags=["section_view", "cv_section", classification, "raw_segment"],
            )
        )
    return entities


def _estimate_section_topology(
    segments: list[Segment],
    crop_width: int,
    crop_height: int,
) -> dict[str, object] | None:
    verticals = [
        segment
        for segment in segments
        if segment.kind == "vertical"
        and _segment_length(segment) >= 30
        and not (segment.x1 < crop_width * 0.05 and _segment_length(segment) > crop_height * 0.75)
        and segment.x1 < crop_width * 0.75
    ]
    horizontals = [
        segment
        for segment in segments
        if segment.kind == "horizontal"
        and _segment_length(segment) >= 35
        and segment.x1 < crop_width * 0.75
        and segment.y1 > crop_height * 0.08
    ]
    if len(verticals) < 3 or len(horizontals) < 6:
        return None

    vertical_xs = _cluster_values([segment.x1 for segment in verticals], tolerance=5)
    if len(vertical_xs) < 3:
        return None

    top_line = _first_horizontal(horizontals, min_y=crop_height * 0.10, min_x=crop_width * 0.12)
    if top_line is None:
        top_line = min(horizontals, key=lambda item: item.y1)

    rows = _cluster_values([segment.y1 for segment in horizontals if segment.y1 >= top_line.y1 - 3], tolerance=7)
    if len(rows) < 6:
        return None

    x_outer_left = _left_outline_x(horizontals, crop_width)
    x_bore_left = vertical_xs[0]
    x_bore_right = vertical_xs[-2] if len(vertical_xs) >= 4 else vertical_xs[-1]
    x_outer_right = vertical_xs[-1]
    top_x1 = max(top_line.x1, x_outer_left + 8)
    top_x2 = min(top_line.x2, x_outer_right)

    y_top = rows[0]
    y_top_inner = _row_after(rows, y_top + crop_height * 0.018, fallback=y_top + crop_height * 0.035)
    y_upper_shoulder = _row_after(rows, y_top_inner + crop_height * 0.035, fallback=y_top_inner + crop_height * 0.07)
    y_upper_step = _row_after(rows, y_upper_shoulder + crop_height * 0.035, fallback=y_upper_shoulder + crop_height * 0.06)
    y_bore_top = _row_after(rows, y_upper_step + crop_height * 0.025, fallback=y_upper_step + crop_height * 0.05)
    y_bore_bottom = _row_after(rows, y_bore_top + crop_height * 0.23, fallback=y_bore_top + crop_height * 0.30)
    y_side_bottom = _row_after(rows, y_bore_top + crop_height * 0.11, fallback=y_bore_top + crop_height * 0.15)
    y_lower_shoulder = _row_after(rows, y_bore_bottom + crop_height * 0.035, fallback=y_bore_bottom + crop_height * 0.07)
    y_lower_step = _row_after(rows, y_lower_shoulder + crop_height * 0.055, fallback=y_lower_shoulder + crop_height * 0.09)
    y_lower_foot = _row_after(rows, y_lower_step + crop_height * 0.035, fallback=y_lower_step + crop_height * 0.06)
    y_tab_top = _row_after(rows, y_lower_foot + crop_height * 0.035, fallback=y_lower_foot + crop_height * 0.06)
    y_bottom = rows[-1]
    if y_bottom <= y_tab_top:
        y_bottom = y_tab_top + crop_height * 0.06

    x_side_right = _right_extension_x(horizontals, y_side_bottom, x_bore_right, crop_width, fallback=x_outer_right)
    x_lower_shoulder_left = _row_segment_start(
        horizontals,
        y_lower_shoulder,
        min_x=x_bore_left + 30,
        max_x=crop_width * 0.78,
        fallback=x_bore_right - max(18.0, crop_width * 0.12),
    )
    x_lower_foot_right = min(max(x_lower_shoulder_left, x_bore_left + 20), x_bore_right)
    bottom_line = max(horizontals, key=lambda item: (item.y1, _segment_length(item)))
    tab_left = max(bottom_line.x1, x_outer_left + 18)
    tab_right = min(bottom_line.x2, x_outer_right)
    if tab_right <= tab_left + 12:
        tab_left = x_outer_left + 20
        tab_right = x_outer_right - 4

    outer = [
        (top_x1, y_top),
        (top_x2, y_top),
        (top_x2, y_top_inner),
        (x_outer_right, y_upper_shoulder),
        (x_outer_right, y_upper_step),
        (x_side_right, y_bore_top),
        (x_side_right, y_side_bottom),
        (x_bore_right, y_side_bottom),
        (x_bore_right, y_bore_bottom),
        (x_outer_right, y_bore_bottom),
        (x_outer_right, y_lower_shoulder),
        (x_lower_shoulder_left, y_lower_shoulder),
        (x_lower_foot_right, y_lower_foot),
        (tab_right, y_tab_top),
        (tab_right, y_bottom),
        (tab_left, y_bottom),
        (tab_left, y_tab_top),
        (x_outer_left, y_lower_foot),
        (x_outer_left, y_lower_step),
        (x_outer_left, y_bore_bottom),
        (x_outer_left, y_bore_top),
        (x_outer_left, y_upper_step),
        (top_x1, y_top_inner),
    ]
    bore = [
        (x_bore_left, y_bore_top),
        (x_bore_right, y_bore_top),
        (x_bore_right, y_bore_bottom),
        (x_bore_left, y_bore_bottom),
    ]
    return {
        "outer": outer,
        "bore": bore,
        "bbox": (
            min(point[0] for point in outer),
            min(point[1] for point in outer),
            max(point[0] for point in outer),
            max(point[1] for point in outer),
        ),
        "hatch_regions": [
            [
                (top_x1, y_top),
                (top_x2, y_top),
                (top_x2, y_top_inner),
                (x_outer_right, y_upper_shoulder),
                (x_outer_right, y_upper_step),
                (x_side_right, y_bore_top),
                (x_bore_right, y_bore_top),
                (x_outer_left, y_bore_top),
                (x_outer_left, y_upper_step),
                (top_x1, y_top_inner),
            ],
            [
                (x_outer_left, y_bore_bottom),
                (x_bore_right, y_bore_bottom),
                (x_outer_right, y_bore_bottom),
                (x_outer_right, y_lower_shoulder),
                (x_lower_shoulder_left, y_lower_shoulder),
                (x_lower_foot_right, y_lower_foot),
                (tab_right, y_tab_top),
                (tab_right, y_bottom),
                (tab_left, y_bottom),
                (tab_left, y_tab_top),
                (x_outer_left, y_lower_foot),
                (x_outer_left, y_lower_step),
            ],
        ],
    }


def _first_horizontal(segments: list[Segment], min_y: float, min_x: float) -> Segment | None:
    candidates = [segment for segment in segments if segment.y1 >= min_y and segment.x1 >= min_x]
    if not candidates:
        return None
    return min(candidates, key=lambda item: item.y1)


def _left_outline_x(segments: list[Segment], crop_width: int) -> float:
    candidates = [
        segment.x1
        for segment in segments
        if _segment_length(segment) >= 70 and segment.x1 < crop_width * 0.22 and segment.y1 > 70
    ]
    if not candidates:
        return crop_width * 0.04
    return float(np.median(candidates))


def _right_extension_x(
    segments: list[Segment],
    row: float,
    min_x: float,
    crop_width: int,
    fallback: float,
) -> float:
    candidates = [
        segment.x2
        for segment in segments
        if abs(segment.y1 - row) <= 8
        and segment.x1 >= min_x - 8
        and segment.x2 <= crop_width * 0.82
        and _segment_length(segment) >= 24
    ]
    if not candidates:
        return fallback
    return max(candidates)


def _row_segment_start(
    segments: list[Segment],
    row: float,
    min_x: float,
    max_x: float,
    fallback: float,
) -> float:
    candidates = [
        segment.x1
        for segment in segments
        if abs(segment.y1 - row) <= 8 and min_x <= segment.x1 <= max_x and _segment_length(segment) >= 24
    ]
    if not candidates:
        return fallback
    return min(candidates)


def _row_after(rows: list[float], threshold: float, fallback: float) -> float:
    for row in rows:
        if row >= threshold:
            return row
    return fallback


def _cluster_values(values: list[float], tolerance: float) -> list[float]:
    if not values:
        return []
    out: list[float] = []
    cluster: list[float] = []
    for value in sorted(values):
        if not cluster or abs(value - float(np.median(cluster))) <= tolerance:
            cluster.append(value)
            continue
        out.append(float(np.median(cluster)))
        cluster = [value]
    out.append(float(np.median(cluster)))
    return out


def _filtered_hatches(segments: list[Segment], bbox: object) -> list[Segment]:
    x1, y1, x2, y2 = bbox
    hatches: list[Segment] = []
    for segment in segments:
        if segment.kind not in {"diag_pos", "diag_neg"}:
            continue
        mid_x = (segment.x1 + segment.x2) / 2
        mid_y = (segment.y1 + segment.y2) / 2
        if x1 - 2 <= mid_x <= x2 + 2 and y1 - 2 <= mid_y <= y2 + 2:
            hatches.append(segment)
    return hatches


def _clipped_hatches(segments: list[Segment], topology: dict[str, object]) -> list[Segment]:
    regions = topology.get("hatch_regions")
    if not isinstance(regions, list):
        return _filtered_hatches(segments, topology["bbox"])

    clipped: list[Segment] = []
    for segment in segments:
        if segment.kind not in {"diag_pos", "diag_neg"}:
            continue
        for region in regions:
            if not isinstance(region, list):
                continue
            clipped.extend(_clip_segment_to_polygon(segment, region))
    return [segment for segment in clipped if _segment_length(segment) >= 8]


def _clip_segment_to_polygon(segment: Segment, polygon: list[tuple[float, float]]) -> list[Segment]:
    dx = segment.x2 - segment.x1
    dy = segment.y2 - segment.y1
    values = [0.0, 1.0]
    for start, end in zip(polygon, polygon[1:] + polygon[:1], strict=True):
        value = _segment_intersection_t(
            segment.x1,
            segment.y1,
            dx,
            dy,
            start[0],
            start[1],
            end[0] - start[0],
            end[1] - start[1],
        )
        if value is not None:
            values.append(value)

    values = _unique_sorted([max(0.0, min(1.0, value)) for value in values], tolerance=1e-5)
    out: list[Segment] = []
    for left, right in zip(values, values[1:], strict=False):
        if right - left < 1e-5:
            continue
        mid = (left + right) / 2
        if not _point_in_polygon((segment.x1 + dx * mid, segment.y1 + dy * mid), polygon):
            continue
        out.append(
            Segment(
                segment.x1 + dx * left,
                segment.y1 + dy * left,
                segment.x1 + dx * right,
                segment.y1 + dy * right,
                segment.kind,
            )
        )
    return out


def _segment_intersection_t(
    px: float,
    py: float,
    rx: float,
    ry: float,
    qx: float,
    qy: float,
    sx: float,
    sy: float,
) -> float | None:
    denominator = rx * sy - ry * sx
    if abs(denominator) < 1e-9:
        return None
    qpx = qx - px
    qpy = qy - py
    t = (qpx * sy - qpy * sx) / denominator
    u = (qpx * ry - qpy * rx) / denominator
    if -1e-6 <= t <= 1 + 1e-6 and -1e-6 <= u <= 1 + 1e-6:
        return t
    return None


def _point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    x, y = point
    for start, end in zip(polygon, polygon[1:] + polygon[:1], strict=True):
        x1, y1 = start
        x2, y2 = end
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        if (
            abs(cross) < 1e-6
            and min(x1, x2) - 1e-6 <= x <= max(x1, x2) + 1e-6
            and min(y1, y2) - 1e-6 <= y <= max(y1, y2) + 1e-6
        ):
            return True

    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
        if (y1 > y) != (y2 > y):
            x_at_y = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < x_at_y:
                inside = not inside
        previous = current
    return inside


def _unique_sorted(values: list[float], tolerance: float) -> list[float]:
    result: list[float] = []
    for value in sorted(values):
        if not result or abs(value - result[-1]) > tolerance:
            result.append(value)
    return result


def _classify_segment(segment: Segment, crop_width: int, crop_height: int) -> str:
    if segment.kind in {"diag_pos", "diag_neg"}:
        return "cut_hatch"
    border_x = min(segment.x1, segment.x2) < 14 or max(segment.x1, segment.x2) > crop_width - 14
    border_y = min(segment.y1, segment.y2) < 42 or max(segment.y1, segment.y2) > crop_height - 32
    if border_x or border_y:
        return "dimension_like"
    return "outline"


def _ensure_layers(ir: DrawingIR) -> None:
    existing = {layer.name for layer in ir.layers}
    for layer in [
        Layer(name="geometry", color="white"),
        Layer(name="hatch", color="white"),
        Layer(name="dimensions", color="white"),
    ]:
        if layer.name not in existing:
            ir.layers.append(layer)
