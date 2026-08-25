from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .models import CircleEntity, DrawingIR, Entity, Layer, LineEntity, PolylineEntity, ProjectState

PROMOTED_LAYER = "promoted_geometry"
PROMOTED_GROUP = "promoted_geometry"
PROMOTED_STROKE_MM = 0.35
MAX_PROMOTED_LINES = 700
MAX_PROMOTED_CIRCLES = 120
MAX_PROMOTED_ARROWS = 220
MIN_LINE_LENGTH_MM = 8.0
MIN_DETAIL_LINE_LENGTH_MM = 4.0
MIN_ARROW_LENGTH_MM = 1.1
MAX_ARROW_LENGTH_MM = 7.5
MIN_CIRCLE_RADIUS_MM = 1.2
LINE_MAX_RELATIVE_ERROR = 0.018
LINE_MAX_ABSOLUTE_ERROR_MM = 0.22
CIRCLE_MAX_RELATIVE_ERROR = 0.055
CIRCLE_MIN_COVERAGE_RAD = math.radians(300)
MERGE_ANGLE_TOLERANCE_RAD = math.radians(3.0)
MERGE_COLLINEAR_DISTANCE_MM = 0.65
MERGE_GAP_MM = 2.0
ARROW_TIP_TOLERANCE_MM = 1.0
ARROW_NEAR_LINE_ENDPOINT_MM = 3.5
ARROW_MIN_PAIR_ANGLE_RAD = math.radians(18)
ARROW_MAX_PAIR_ANGLE_RAD = math.radians(110)
ARROW_MIN_LINE_ANGLE_RAD = math.radians(18)
ARROW_MAX_LINE_ANGLE_RAD = math.radians(75)


@dataclass(frozen=True)
class ScanPromotion:
    ir: DrawingIR
    promoted_counts: dict[str, int]
    source_count: int
    warnings: list[str]


@dataclass(frozen=True)
class LineCandidate:
    source_id: str
    start: np.ndarray
    end: np.ndarray
    length: float
    angle: float


def promote_scan_primitives(project: ProjectState) -> ScanPromotion:
    source_entities = [
        entity
        for entity in project.ir.entities
        if isinstance(entity, PolylineEntity) and entity.group == "editable_linework"
    ]
    if not source_entities:
        raise ValueError("No editable_linework polylines found. Run scan CAD reconstruction first.")

    next_ir = project.ir.model_copy(deep=True)
    next_ir.entities = [
        entity
        for entity in next_ir.entities
        if entity.group != PROMOTED_GROUP and entity.layer != PROMOTED_LAYER
    ]
    _ensure_promoted_layer(next_ir)

    promoted: list[Entity] = []
    counts = {"line": 0, "circle": 0, "arrow": 0}
    line_candidates: list[LineCandidate] = []
    skipped_sources: set[str] = set()
    for entity in source_entities:
        if counts["circle"] < MAX_PROMOTED_CIRCLES:
            circle = _promote_circle(entity, counts["circle"])
            if circle is not None:
                promoted.append(circle)
                counts["circle"] += 1
                continue
        candidate = _fit_line_candidate(entity, MIN_ARROW_LENGTH_MM)
        if candidate is not None:
            line_candidates.append(candidate)
        else:
            skipped_sources.add(entity.id)

    merged_lines = _merge_line_candidates(
        [candidate for candidate in line_candidates if candidate.length >= MIN_DETAIL_LINE_LENGTH_MM],
        MAX_PROMOTED_LINES,
    )
    for index, line in enumerate(merged_lines):
        promoted.append(_line_from_candidate(line, f"promoted_line_{index:05d}", ["line_fit", "merged_collinear"]))
    counts["line"] = len(merged_lines)

    arrow_candidates = _dimension_arrow_candidates(line_candidates, merged_lines)
    for index, arrow in enumerate(arrow_candidates[:MAX_PROMOTED_ARROWS]):
        promoted.append(
            _line_from_candidate(
                arrow,
                f"promoted_arrow_{index:05d}",
                ["dimension_arrow", "arrowhead", "line_fit"],
                stroke_width=0.25,
            )
        )
    counts["arrow"] = min(len(arrow_candidates), MAX_PROMOTED_ARROWS)

    promoted_source_ids = {
        tag
        for entity in promoted
        for tag in entity.tags
        if tag.startswith(("editable_", "ref_", "line_", "circle_"))
    }
    skipped = len([entity for entity in source_entities if entity.id in skipped_sources and entity.id not in promoted_source_ids])

    next_ir.entities.extend(promoted)
    next_ir.notes = [
        *next_ir.notes,
        (
            "Promoted editable linework into CAD primitives: "
            f"{counts['line']} merged lines, {counts['circle']} circles, {counts['arrow']} arrow strokes."
        ),
    ]
    warnings: list[str] = []
    if not promoted:
        warnings.append("No high-confidence editable primitives were promoted.")
    if counts["line"] >= MAX_PROMOTED_LINES:
        warnings.append(f"Line promotion capped at {MAX_PROMOTED_LINES} entities.")
    if counts["circle"] >= MAX_PROMOTED_CIRCLES:
        warnings.append(f"Circle promotion capped at {MAX_PROMOTED_CIRCLES} entities.")
    if len(arrow_candidates) > MAX_PROMOTED_ARROWS:
        warnings.append(f"Dimension arrow promotion capped at {MAX_PROMOTED_ARROWS} strokes.")
    if skipped:
        warnings.append(f"Skipped {skipped} editable polylines below primitive-fit confidence thresholds.")
    return ScanPromotion(ir=next_ir, promoted_counts=counts, source_count=len(source_entities), warnings=warnings)


def _ensure_promoted_layer(ir: DrawingIR) -> None:
    if not any(layer.name == PROMOTED_LAYER for layer in ir.layers):
        ir.layers.append(Layer(name=PROMOTED_LAYER, color="white"))


def _promote_line(entity: PolylineEntity, index: int) -> LineEntity | None:
    candidate = _fit_line_candidate(entity, MIN_LINE_LENGTH_MM)
    if candidate is None:
        return None
    return _line_from_candidate(candidate, f"promoted_line_{index:05d}", ["line_fit"])


def _fit_line_candidate(entity: PolylineEntity, min_length: float) -> LineCandidate | None:
    points = _points_array(entity)
    if len(points) < 2 or _is_closed_polyline(entity, points):
        return None

    start = points[0].astype(float)
    end = points[-1].astype(float)
    vector = end - start
    length = float(np.linalg.norm(vector))
    if length < min_length:
        return None

    distances = _point_line_distances(points, start, end)
    max_allowed = max(LINE_MAX_ABSOLUTE_ERROR_MM, length * LINE_MAX_RELATIVE_ERROR)
    if float(np.max(distances)) > max_allowed:
        return None
    if float(np.mean(distances)) > max_allowed * 0.45:
        return None
    angle = math.atan2(float(vector[1]), float(vector[0])) % math.pi
    return LineCandidate(source_id=entity.id, start=start, end=end, length=length, angle=angle)


def _line_from_candidate(
    candidate: LineCandidate,
    entity_id: str,
    tags: list[str],
    stroke_width: float = PROMOTED_STROKE_MM,
) -> LineEntity:
    return LineEntity(
        id=entity_id,
        layer=PROMOTED_LAYER,
        x1=round(float(candidate.start[0]), 4),
        y1=round(float(candidate.start[1]), 4),
        x2=round(float(candidate.end[0]), 4),
        y2=round(float(candidate.end[1]), 4),
        group=PROMOTED_GROUP,
        tags=["promoted_geometry", "from_editable_linework", *tags, candidate.source_id],
        stroke_width=stroke_width,
    )


def _promote_circle(entity: PolylineEntity, index: int) -> CircleEntity | None:
    points = _points_array(entity)
    if len(points) < 10 or not _is_closed_polyline(entity, points):
        return None

    x = points[:, 0]
    y = points[:, 1]
    width = float(np.max(x) - np.min(x))
    height = float(np.max(y) - np.min(y))
    if width <= 0 or height <= 0:
        return None
    ratio = width / height
    if ratio < 0.72 or ratio > 1.38:
        return None

    try:
        center_x, center_y, radius = _fit_circle(points)
    except np.linalg.LinAlgError:
        return None
    if radius < MIN_CIRCLE_RADIUS_MM:
        return None

    distances = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    relative_error = float(np.mean(np.abs(distances - radius)) / max(radius, 1e-6))
    if relative_error > CIRCLE_MAX_RELATIVE_ERROR:
        return None
    if _angle_coverage(points, center_x, center_y) < CIRCLE_MIN_COVERAGE_RAD:
        return None

    return CircleEntity(
        id=f"promoted_circle_{index:05d}",
        layer=PROMOTED_LAYER,
        cx=round(float(center_x), 4),
        cy=round(float(center_y), 4),
        r=round(float(radius), 4),
        group=PROMOTED_GROUP,
        tags=["promoted_geometry", "from_editable_linework", "circle_fit", entity.id],
        stroke_width=PROMOTED_STROKE_MM,
    )


def _points_array(entity: PolylineEntity) -> np.ndarray:
    return np.asarray(entity.points, dtype=float)


def _merge_line_candidates(candidates: list[LineCandidate], limit: int) -> list[LineCandidate]:
    clusters: list[list[LineCandidate]] = []
    for candidate in sorted(candidates, key=lambda item: item.length, reverse=True):
        for cluster in clusters:
            if _candidate_fits_cluster(candidate, cluster):
                cluster.append(candidate)
                break
        else:
            clusters.append([candidate])

    merged: list[LineCandidate] = []
    for cluster in clusters:
        merged.extend(_merge_cluster_intervals(cluster))
    return sorted(merged, key=lambda item: item.length, reverse=True)[:limit]


def _candidate_fits_cluster(candidate: LineCandidate, cluster: list[LineCandidate]) -> bool:
    reference = cluster[0]
    if _angle_delta(candidate.angle, reference.angle) > MERGE_ANGLE_TOLERANCE_RAD:
        return False
    direction = _unit_vector(reference)
    normal = np.asarray([-direction[1], direction[0]])
    reference_mid = (reference.start + reference.end) / 2
    candidate_mid = (candidate.start + candidate.end) / 2
    return abs(float(np.dot(candidate_mid - reference_mid, normal))) <= MERGE_COLLINEAR_DISTANCE_MM


def _merge_cluster_intervals(cluster: list[LineCandidate]) -> list[LineCandidate]:
    reference = cluster[0]
    direction = _unit_vector(reference)
    origin = reference.start
    intervals: list[tuple[float, float, list[str]]] = []
    for candidate in cluster:
        t1 = float(np.dot(candidate.start - origin, direction))
        t2 = float(np.dot(candidate.end - origin, direction))
        intervals.append((min(t1, t2), max(t1, t2), [candidate.source_id]))
    intervals.sort(key=lambda item: item[0])

    merged: list[LineCandidate] = []
    current_start, current_end, source_ids = intervals[0]
    for start, end, ids in intervals[1:]:
        if start <= current_end + MERGE_GAP_MM:
            current_end = max(current_end, end)
            source_ids.extend(ids)
        else:
            merged.append(_interval_to_candidate(origin, direction, current_start, current_end, source_ids))
            current_start, current_end, source_ids = start, end, ids
    merged.append(_interval_to_candidate(origin, direction, current_start, current_end, source_ids))
    return [candidate for candidate in merged if candidate.length >= MIN_DETAIL_LINE_LENGTH_MM]


def _interval_to_candidate(
    origin: np.ndarray,
    direction: np.ndarray,
    start: float,
    end: float,
    source_ids: list[str],
) -> LineCandidate:
    p1 = origin + direction * start
    p2 = origin + direction * end
    length = float(np.linalg.norm(p2 - p1))
    angle = math.atan2(float(direction[1]), float(direction[0])) % math.pi
    return LineCandidate(source_id=",".join(source_ids[:6]), start=p1, end=p2, length=length, angle=angle)


def _dimension_arrow_candidates(
    candidates: list[LineCandidate],
    promoted_lines: list[LineCandidate],
) -> list[LineCandidate]:
    short = [candidate for candidate in candidates if MIN_ARROW_LENGTH_MM <= candidate.length <= MAX_ARROW_LENGTH_MM]
    if not short or not promoted_lines:
        return []

    selected: dict[str, LineCandidate] = {}
    long_endpoint_records = [
        (point, line.angle)
        for line in promoted_lines
        if line.length >= MIN_LINE_LENGTH_MM
        for point in (line.start, line.end)
    ]
    long_endpoints = [point for point, _angle in long_endpoint_records]
    for index, first in enumerate(short):
        for second in short[index + 1 :]:
            if _arrow_pair(first, second, long_endpoints) is None:
                continue
            selected[first.source_id] = first
            selected[second.source_id] = second
    for candidate in short:
        if candidate.source_id in selected:
            continue
        if _single_arrow_stroke(candidate, long_endpoint_records):
            selected[candidate.source_id] = candidate
    return sorted(selected.values(), key=lambda candidate: candidate.source_id)


def _arrow_pair(
    first: LineCandidate,
    second: LineCandidate,
    long_endpoints: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    tip, first_tail, second_tail = _shared_tip(first, second)
    if tip is None or first_tail is None or second_tail is None:
        return None
    if not any(float(np.linalg.norm(tip - endpoint)) <= ARROW_NEAR_LINE_ENDPOINT_MM for endpoint in long_endpoints):
        return None
    v1 = first_tail - tip
    v2 = second_tail - tip
    norm_product = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if norm_product <= 1e-9:
        return None
    angle = math.acos(float(np.clip(np.dot(v1, v2) / norm_product, -1.0, 1.0)))
    if not (ARROW_MIN_PAIR_ANGLE_RAD <= angle <= ARROW_MAX_PAIR_ANGLE_RAD):
        return None
    return tip, first_tail, second_tail


def _single_arrow_stroke(
    candidate: LineCandidate,
    long_endpoint_records: list[tuple[np.ndarray, float]],
) -> bool:
    nearest_distance = math.inf
    nearest_angle = 0.0
    for endpoint, angle in long_endpoint_records:
        distance = min(float(np.linalg.norm(candidate.start - endpoint)), float(np.linalg.norm(candidate.end - endpoint)))
        if distance < nearest_distance:
            nearest_distance = distance
            nearest_angle = angle
    if nearest_distance > ARROW_NEAR_LINE_ENDPOINT_MM:
        return False
    delta = _angle_delta(candidate.angle, nearest_angle)
    return ARROW_MIN_LINE_ANGLE_RAD <= delta <= ARROW_MAX_LINE_ANGLE_RAD


def _shared_tip(
    first: LineCandidate,
    second: LineCandidate,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    first_points = (first.start, first.end)
    second_points = (second.start, second.end)
    for first_index, first_point in enumerate(first_points):
        for second_index, second_point in enumerate(second_points):
            if float(np.linalg.norm(first_point - second_point)) <= ARROW_TIP_TOLERANCE_MM:
                tip = (first_point + second_point) / 2
                first_tail = first_points[1 - first_index]
                second_tail = second_points[1 - second_index]
                return tip, first_tail, second_tail
    return None, None, None


def _unit_vector(candidate: LineCandidate) -> np.ndarray:
    vector = candidate.end - candidate.start
    length = float(np.linalg.norm(vector))
    if length <= 1e-9:
        return np.asarray([1.0, 0.0])
    return vector / length


def _angle_delta(first: float, second: float) -> float:
    delta = abs(first - second) % math.pi
    return min(delta, math.pi - delta)


def _is_closed_polyline(entity: PolylineEntity, points: np.ndarray) -> bool:
    if entity.closed:
        return True
    if len(points) < 2:
        return False
    return float(np.linalg.norm(points[0] - points[-1])) <= 0.9


def _point_line_distances(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    vector = end - start
    length = float(np.linalg.norm(vector))
    if length <= 1e-9:
        return np.full(len(points), np.inf)
    offsets = points - start
    cross = vector[0] * offsets[:, 1] - vector[1] * offsets[:, 0]
    return np.abs(cross / length)


def _fit_circle(points: np.ndarray) -> tuple[float, float, float]:
    x = points[:, 0]
    y = points[:, 1]
    matrix = np.column_stack((x, y, np.ones_like(x)))
    target = x**2 + y**2
    a, b, c = np.linalg.lstsq(matrix, target, rcond=None)[0]
    center_x = a / 2
    center_y = b / 2
    radius = math.sqrt(max(c + center_x**2 + center_y**2, 0.0))
    return float(center_x), float(center_y), float(radius)


def _angle_coverage(points: np.ndarray, center_x: float, center_y: float) -> float:
    angles = np.sort(np.arctan2(points[:, 1] - center_y, points[:, 0] - center_x))
    if len(angles) < 2:
        return 0.0
    gaps = np.diff(np.concatenate([angles, [angles[0] + math.tau]]))
    return float(math.tau - np.max(gaps))
