from __future__ import annotations

import math
from dataclasses import dataclass

from .cad_layers import CENTER, DIMENSION, HATCH, OUTLINE, REFERENCE_TRACE, TEXT, TITLE_BLOCK, canonical_layer_name
from .models import (
    ArcEntity,
    CircleEntity,
    DimensionBinding,
    DrawingIR,
    Entity,
    LineEntity,
    MechanicalDimensionObject,
    MechanicalDrawingIR,
    PolylineEntity,
    RectangleEntity,
    TextEntity,
)

MECHANICAL_IR_SOURCE = "mechanical_ir_v1"
EXTENSION_ENDPOINT_TOLERANCE_MM = 4.5
EXTENSION_PERPENDICULAR_TOLERANCE_RAD = math.radians(24)
MEASURED_GEOMETRY_TOLERANCE_MM = 6.0
RADIAL_GEOMETRY_TOLERANCE_MM = 12.0


@dataclass(frozen=True)
class RelationMatch:
    entity_id: str
    confidence: float
    distance: float


def build_mechanical_drawing_ir(
    ir: DrawingIR,
    bindings: list[DimensionBinding],
    rendered_dimensions: list[MechanicalDimensionObject],
    warnings: list[str] | None = None,
) -> MechanicalDrawingIR:
    """Build one inspectable semantic snapshot from graph bindings and CAD entities.

    Bindings remain the low-level detector output. This function promotes every
    binding into a Dimension object and adds the geometric relationships needed
    by later agent tools: witness/extension lines and measured geometry.
    """

    entities_by_id = {entity.id: entity for entity in ir.entities}
    rendered_by_binding = {dimension.binding_id: dimension for dimension in rendered_dimensions}
    reserved_ids = {
        entity_id
        for binding in bindings
        for entity_id in [binding.dimension_line_id, *binding.arrow_ids]
    }

    dimensions: list[MechanicalDimensionObject] = []
    entity_roles: dict[str, list[str]] = {
        "dimension_line": [],
        "dimension_text": [],
        "arrowhead": [],
        "extension_line": [],
        "measured_geometry": [],
    }
    unresolved: list[str] = []
    semantic_warnings = list(warnings or [])

    for binding in bindings:
        rendered = rendered_by_binding.get(binding.id)
        arrowheads = list(rendered.arrowheads) if rendered else []
        extension_matches = _find_extension_lines(ir.entities, binding, reserved_ids)
        extension_ids = [match.entity_id for match in extension_matches]
        measured_matches = _find_measured_geometry(
            ir.entities,
            binding,
            extension_ids,
            reserved_ids,
        )
        measured_ids = [match.entity_id for match in measured_matches]
        measurement_points = _dimension_measurement_points(
            entities_by_id,
            binding,
            extension_ids,
            measured_ids,
        )
        dimension_line_point = [
            round((binding.line_x1 + binding.line_x2) * 0.5, 6),
            round((binding.line_y1 + binding.line_y2) * 0.5, 6),
        ]
        dxf_dimension_type = _dxf_dimension_type(binding, measurement_points, entities_by_id, measured_ids)
        issues = _dimension_issues(
            binding,
            arrowheads,
            extension_ids,
            measured_ids,
            measurement_points,
        )
        status = _dimension_status(binding, arrowheads, issues)
        export_ready = status == "complete" and dxf_dimension_type is not None
        if status != "complete":
            unresolved.append(binding.id)

        relation_confidence = {
            binding.dimension_line_id: round(binding.confidence, 4),
            **({binding.text_id: round(binding.confidence, 4)} if binding.text_id else {}),
            **{arrow.render_entity_id: round(arrow.score, 4) for arrow in arrowheads},
            **{match.entity_id: round(match.confidence, 4) for match in extension_matches},
            **{match.entity_id: round(match.confidence, 4) for match in measured_matches},
        }
        completeness = max(0.0, 1.0 - len(issues) * 0.16)
        confidence = min(1.0, binding.confidence * 0.72 + completeness * 0.28)
        evidence = list(rendered.evidence) if rendered else [binding.binding_method, *binding.graph_path[:6]]
        evidence.extend(
            [
                *[f"extension_line:{match.entity_id} distance={match.distance:.3f}" for match in extension_matches],
                *[f"measured_geometry:{match.entity_id} distance={match.distance:.3f}" for match in measured_matches],
                *[f"definition_point:{point[0]:.3f},{point[1]:.3f}" for point in measurement_points],
            ]
        )
        dimension = MechanicalDimensionObject(
            id=rendered.id if rendered else f"mechanical_dimension_{binding.id}",
            binding_id=binding.id,
            kind=binding.kind,
            text=binding.text,
            parsed=binding.parsed,
            confidence=round(confidence, 4),
            dimension_line_id=binding.dimension_line_id,
            text_id=binding.text_id,
            arrowheads=arrowheads,
            extension_line_ids=extension_ids,
            measured_geometry_ids=measured_ids,
            target_geometry_ids=measured_ids,
            measurement_points=measurement_points,
            dimension_line_point=dimension_line_point,
            relation_confidence=relation_confidence,
            orientation=_dimension_orientation(binding),
            dxf_dimension_type=dxf_dimension_type,
            export_ready=export_ready,
            status=status,
            issues=issues,
            evidence=evidence,
            source=MECHANICAL_IR_SOURCE,
        )
        dimensions.append(dimension)

        _append_role(entity_roles, "dimension_line", binding.dimension_line_id, entities_by_id)
        if binding.text_id:
            _append_role(entity_roles, "dimension_text", binding.text_id, entities_by_id, allow_missing=True)
        for arrow in arrowheads:
            _append_role(entity_roles, "arrowhead", arrow.render_entity_id, entities_by_id)
        for entity_id in extension_ids:
            _append_role(entity_roles, "extension_line", entity_id, entities_by_id)
        for entity_id in measured_ids:
            _append_role(entity_roles, "measured_geometry", entity_id, entities_by_id)

    if unresolved:
        semantic_warnings.append(
            f"{len(unresolved)} of {len(bindings)} dimensions are partial or unresolved; inspect their missing relations."
        )
    return MechanicalDrawingIR(
        units=ir.units,
        dimensions=dimensions,
        entity_roles={role: ids for role, ids in entity_roles.items() if ids},
        unresolved_binding_ids=unresolved,
        warnings=_dedupe(semantic_warnings),
    )


def _find_extension_lines(
    entities: list[Entity],
    binding: DimensionBinding,
    reserved_ids: set[str],
) -> list[RelationMatch]:
    dimension_direction = _unit(binding.line_x2 - binding.line_x1, binding.line_y2 - binding.line_y1)
    if dimension_direction is None:
        return []

    endpoints = [(binding.line_x1, binding.line_y1), (binding.line_x2, binding.line_y2)]
    selected: list[RelationMatch] = []
    used: set[str] = set()
    for endpoint in endpoints:
        matches: list[tuple[float, RelationMatch]] = []
        for entity in entities:
            if not isinstance(entity, LineEntity) or entity.id in reserved_ids or entity.id in used:
                continue
            if not _can_be_extension_line(entity):
                continue
            candidate_direction = _unit(entity.x2 - entity.x1, entity.y2 - entity.y1)
            if candidate_direction is None:
                continue
            perpendicular_error = abs(math.pi / 2 - math.acos(min(1.0, abs(_dot(dimension_direction, candidate_direction)))))
            if perpendicular_error > EXTENSION_PERPENDICULAR_TOLERANCE_RAD:
                continue
            distance = _point_to_segment_distance(endpoint, (entity.x1, entity.y1), (entity.x2, entity.y2))
            if distance > EXTENSION_ENDPOINT_TOLERANCE_MM:
                continue
            semantic_bonus = -0.7 if _is_dimension_related(entity) else 0.0
            score = distance + perpendicular_error * 4.0 + semantic_bonus
            confidence = 1.0 - min(0.62, distance / EXTENSION_ENDPOINT_TOLERANCE_MM * 0.38)
            confidence -= min(0.28, perpendicular_error / EXTENSION_PERPENDICULAR_TOLERANCE_RAD * 0.28)
            matches.append((score, RelationMatch(entity.id, max(0.35, confidence), distance)))
        if matches:
            match = min(matches, key=lambda item: (item[0], item[1].entity_id))[1]
            selected.append(match)
            used.add(match.entity_id)
    return selected


def _find_measured_geometry(
    entities: list[Entity],
    binding: DimensionBinding,
    extension_ids: list[str],
    reserved_ids: set[str],
) -> list[RelationMatch]:
    by_id = {entity.id: entity for entity in entities}
    excluded = {*reserved_ids, *extension_ids}
    probes: list[tuple[float, float]] = []
    for extension_id in extension_ids:
        entity = by_id.get(extension_id)
        if not isinstance(entity, LineEntity):
            continue
        endpoints = [(entity.x1, entity.y1), (entity.x2, entity.y2)]
        probes.append(max(endpoints, key=lambda point: _point_to_binding_line_distance(point, binding)))

    tolerance = MEASURED_GEOMETRY_TOLERANCE_MM
    if not probes and binding.kind in {"diameter", "radius"}:
        probes = [
            (binding.line_x1, binding.line_y1),
            (binding.line_x2, binding.line_y2),
        ]
        tolerance = RADIAL_GEOMETRY_TOLERANCE_MM
    if not probes:
        return []

    selected: list[RelationMatch] = []
    used: set[str] = set()
    for probe in probes:
        candidates: list[tuple[float, RelationMatch]] = []
        for entity in entities:
            if entity.id in excluded or entity.id in used or not _can_be_measured_geometry(entity):
                continue
            distance = _point_to_entity_distance(probe, entity)
            if distance > tolerance:
                continue
            priority = _geometry_priority(entity)
            score = distance + priority
            confidence = max(0.32, 1.0 - distance / tolerance * 0.58 - min(priority, 1.5) * 0.08)
            candidates.append((score, RelationMatch(entity.id, confidence, distance)))
        if candidates:
            match = min(candidates, key=lambda item: (item[0], item[1].entity_id))[1]
            selected.append(match)
            used.add(match.entity_id)
    return selected


def _can_be_extension_line(entity: LineEntity) -> bool:
    if _line_length(entity) < 2.0:
        return False
    if canonical_layer_name(entity.layer) in {TITLE_BLOCK, REFERENCE_TRACE, HATCH, CENTER}:
        return False
    if entity.group in {"sheet", "title_block", "parameter_table", "reference_trace"}:
        return False
    return not set(entity.tags).intersection({"dimension_arrow", "arrowhead", "dimension_arrow_render", "hatch"})


def _can_be_measured_geometry(entity: Entity) -> bool:
    if isinstance(entity, TextEntity):
        return False
    if canonical_layer_name(entity.layer) in {TITLE_BLOCK, REFERENCE_TRACE, HATCH, CENTER, TEXT}:
        return False
    if entity.group in {"sheet", "title_block", "parameter_table", "reference_trace"}:
        return False
    return not set(entity.tags).intersection({"dimension_arrow", "arrowhead", "dimension_arrow_render", "hatch"})


def _is_dimension_related(entity: Entity) -> bool:
    return canonical_layer_name(entity.layer) == DIMENSION or entity.group == "dimensions" or "dimensions" in entity.tags


def _geometry_priority(entity: Entity) -> float:
    if isinstance(entity, CircleEntity | ArcEntity):
        return -0.35
    if canonical_layer_name(entity.layer) == OUTLINE:
        return 0.0
    if entity.group in {"section_view", "circular_view", "promoted_geometry"}:
        return 0.25
    return 0.9


def _dimension_issues(binding, arrowheads, extension_ids, measured_ids, measurement_points) -> list[str]:
    issues: list[str] = []
    if not binding.text_id and not binding.text:
        issues.append("missing_text")
    expected_arrows = 2 if binding.kind in {"linear", "diameter"} else 1
    if len(arrowheads) < expected_arrows:
        issues.append("missing_arrowhead")
    if binding.kind == "linear" and len(extension_ids) < 2:
        issues.append("missing_extension_lines")
    if binding.kind in {"linear", "diameter", "radius"} and not measured_ids:
        issues.append("missing_measured_geometry")
    expected_points = 1 if binding.kind == "radius" else 2
    if binding.kind in {"linear", "diameter", "radius"} and len(measurement_points) < expected_points:
        issues.append("missing_definition_points")
    return issues


def _dimension_status(binding, arrowheads, issues: list[str]) -> str:
    if not binding.dimension_line_id or (not arrowheads and not binding.text):
        return "unresolved"
    return "complete" if not issues else "partial"


def _dimension_orientation(binding: DimensionBinding) -> str:
    if binding.kind == "radius":
        return "radial"
    dx = abs(binding.line_x2 - binding.line_x1)
    dy = abs(binding.line_y2 - binding.line_y1)
    if dy <= max(dx, 1e-9) * 0.18:
        return "horizontal"
    if dx <= max(dy, 1e-9) * 0.18:
        return "vertical"
    return "aligned"


def _dimension_measurement_points(
    entities_by_id: dict[str, Entity],
    binding: DimensionBinding,
    extension_ids: list[str],
    measured_ids: list[str],
) -> list[list[float]]:
    radial_geometry = next(
        (
            entity
            for entity_id in measured_ids
            if isinstance((entity := entities_by_id.get(entity_id)), CircleEntity | ArcEntity)
        ),
        None,
    )
    if radial_geometry is not None and binding.kind in {"diameter", "radius"}:
        center = (radial_geometry.cx, radial_geometry.cy)
        direction = _radial_direction(binding, center)
        if binding.kind == "diameter":
            return [
                _rounded_point((center[0] - direction[0] * radial_geometry.r, center[1] - direction[1] * radial_geometry.r)),
                _rounded_point((center[0] + direction[0] * radial_geometry.r, center[1] + direction[1] * radial_geometry.r)),
            ]
        return [
            _rounded_point(center),
            _rounded_point((center[0] + direction[0] * radial_geometry.r, center[1] + direction[1] * radial_geometry.r)),
        ]

    points: list[list[float]] = []
    for extension_id in extension_ids:
        extension = entities_by_id.get(extension_id)
        if not isinstance(extension, LineEntity):
            continue
        endpoints = [(extension.x1, extension.y1), (extension.x2, extension.y2)]
        definition_point = max(endpoints, key=lambda point: _point_to_binding_line_distance(point, binding))
        rounded = _rounded_point(definition_point)
        if rounded not in points:
            points.append(rounded)
    return points[:2]


def _radial_direction(binding: DimensionBinding, center: tuple[float, float]) -> tuple[float, float]:
    endpoints = [(binding.line_x1, binding.line_y1), (binding.line_x2, binding.line_y2)]
    target = max(endpoints, key=lambda point: math.dist(point, center))
    return _unit(target[0] - center[0], target[1] - center[1]) or (1.0, 0.0)


def _rounded_point(point: tuple[float, float]) -> list[float]:
    return [round(float(point[0]), 6), round(float(point[1]), 6)]


def _dxf_dimension_type(
    binding: DimensionBinding,
    measurement_points: list[list[float]],
    entities_by_id: dict[str, Entity],
    measured_ids: list[str],
) -> str | None:
    has_radial_geometry = any(
        isinstance(entities_by_id.get(entity_id), CircleEntity | ArcEntity)
        for entity_id in measured_ids
    )
    if binding.kind == "radius" and has_radial_geometry and len(measurement_points) >= 2:
        return "radius"
    if binding.kind == "diameter" and has_radial_geometry and len(measurement_points) >= 2:
        return "diameter"
    if binding.kind in {"linear", "diameter"} and len(measurement_points) >= 2:
        return "aligned" if _dimension_orientation(binding) == "aligned" else "linear"
    return None


def _append_role(roles, role: str, entity_id: str, entities_by_id, allow_missing: bool = False) -> None:
    if not allow_missing and entity_id not in entities_by_id:
        return
    if entity_id not in roles[role]:
        roles[role].append(entity_id)


def _point_to_binding_line_distance(point, binding: DimensionBinding) -> float:
    return _point_to_segment_distance(
        point,
        (binding.line_x1, binding.line_y1),
        (binding.line_x2, binding.line_y2),
    )


def _point_to_entity_distance(point: tuple[float, float], entity: Entity) -> float:
    if isinstance(entity, LineEntity):
        return _point_to_segment_distance(point, (entity.x1, entity.y1), (entity.x2, entity.y2))
    if isinstance(entity, CircleEntity | ArcEntity):
        return abs(math.dist(point, (entity.cx, entity.cy)) - entity.r)
    if isinstance(entity, PolylineEntity):
        segments = list(zip(entity.points, entity.points[1:]))
        if entity.closed and len(entity.points) > 2:
            segments.append((entity.points[-1], entity.points[0]))
        return min((_point_to_segment_distance(point, tuple(a), tuple(b)) for a, b in segments), default=math.inf)
    if isinstance(entity, RectangleEntity):
        corners = [
            (entity.x, entity.y),
            (entity.x + entity.width, entity.y),
            (entity.x + entity.width, entity.y + entity.height),
            (entity.x, entity.y + entity.height),
        ]
        return min(
            _point_to_segment_distance(point, corners[index], corners[(index + 1) % 4])
            for index in range(4)
        )
    return math.inf


def _point_to_segment_distance(point, start, end) -> float:
    vx = end[0] - start[0]
    vy = end[1] - start[1]
    length_sq = vx * vx + vy * vy
    if length_sq <= 1e-12:
        return math.dist(point, start)
    t = max(0.0, min(1.0, ((point[0] - start[0]) * vx + (point[1] - start[1]) * vy) / length_sq))
    projection = (start[0] + t * vx, start[1] + t * vy)
    return math.dist(point, projection)


def _line_length(entity: LineEntity) -> float:
    return math.hypot(entity.x2 - entity.x1, entity.y2 - entity.y1)


def _unit(dx: float, dy: float) -> tuple[float, float] | None:
    length = math.hypot(dx, dy)
    if length <= 1e-9:
        return None
    return (dx / length, dy / length)


def _dot(first, second) -> float:
    return first[0] * second[0] + first[1] * second[1]


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))
