from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from difflib import SequenceMatcher

from .dimension_semantics import parse_dimension_text
from .models import (
    ArcEntity,
    CircleEntity,
    DimensionBenchmarkReport,
    DimensionCorrection,
    DimensionCorrectionRequest,
    DimensionGroundTruth,
    DimensionTargetEval,
    Entity,
    LineEntity,
    MechanicalArrowhead,
    MechanicalDimensionObject,
    ParsedDimensionValue,
    PolylineEntity,
    ProjectState,
    RectangleEntity,
    TextEntity,
)

PASS_SCORE = 0.999
MATCH_THRESHOLD = 0.72


def default_output_shaft_targets() -> list[DimensionGroundTruth]:
    relations = [
        "text",
        "dimension_line",
        "arrowheads",
        "extension_lines",
        "measured_geometry",
        "definition_points",
    ]
    radial_relations = [
        "text",
        "dimension_line",
        "arrowheads",
        "measured_geometry",
        "definition_points",
    ]
    specs = [
        ("gt_linear_244", "总长 244", "244", "linear", 244.0, relations),
        ("gt_linear_197", "轴段长度 197", "197", "linear", 197.0, relations),
        ("gt_linear_127", "轴段长度 127", "127", "linear", 127.0, relations),
        ("gt_diameter_65", "直径 65", "φ65", "diameter", 65.0, radial_relations),
        ("gt_diameter_75", "直径 75", "φ75", "diameter", 75.0, radial_relations),
        ("gt_diameter_80", "直径 80", "φ80", "diameter", 80.0, radial_relations),
        ("gt_diameter_140", "直径 140", "φ140", "diameter", 140.0, radial_relations),
        ("gt_diameter_176", "直径 176", "φ176", "diameter", 176.0, radial_relations),
        ("gt_diameter_60", "直径 60 公差", "φ60", "diameter", 60.0, radial_relations),
        ("gt_diameter_50", "直径 50", "φ50", "diameter", 50.0, radial_relations),
    ]
    return [
        DimensionGroundTruth(
            id=target_id,
            label=label,
            expected_text=text,
            kind=kind,
            nominal=nominal,
            required_relations=required,
            source="seed",
        )
        for target_id, label, text, kind, nominal, required in specs
    ]


def seed_dimension_ground_truth(
    project: ProjectState,
    targets: list[DimensionGroundTruth] | None = None,
    *,
    replace: bool = False,
) -> ProjectState:
    working = project.model_copy(deep=True)
    seeds = [item.model_copy(deep=True) for item in (targets or default_output_shaft_targets())]
    if replace:
        working.dimension_ground_truth = seeds
        valid_ids = {item.id for item in seeds}
        working.dimension_corrections = [
            item for item in working.dimension_corrections if item.ground_truth_id in valid_ids
        ]
        return working
    by_id = {item.id: item for item in working.dimension_ground_truth}
    for seed in seeds:
        by_id[seed.id] = seed
    working.dimension_ground_truth = list(by_id.values())
    return working


def evaluate_dimension_benchmark(project: ProjectState) -> DimensionBenchmarkReport:
    dimensions = list(project.mechanical_ir.dimensions or project.mechanical_dimensions)
    corrections = {item.ground_truth_id: item for item in project.dimension_corrections}
    used_dimension_ids: set[str] = set()
    targets: list[DimensionTargetEval] = []

    for truth in project.dimension_ground_truth:
        correction = corrections.get(truth.id)
        preferred_id = correction.dimension_id if correction else truth.matched_dimension_id
        dimension = _dimension_by_id(dimensions, preferred_id)
        if dimension is None:
            dimension = _best_dimension_match(truth, dimensions, used_dimension_ids)
        if dimension is not None:
            used_dimension_ids.add(dimension.id)
        targets.append(_evaluate_target(project, truth, dimension, correction is not None))

    target_count = len(targets)
    metric_names = [
        "text",
        "kind",
        "dimension_line",
        "arrowheads",
        "extension_lines",
        "measured_geometry",
        "definition_points",
        "native_ready",
    ]
    metrics = {
        name: round(sum(target.metrics.get(name, 0.0) for target in targets) / max(target_count, 1), 4)
        for name in metric_names
    }
    overall = sum(target.score for target in targets) / max(target_count, 1)
    return DimensionBenchmarkReport(
        project_id=project.project_id,
        target_count=target_count,
        matched_count=sum(target.matched_dimension_id is not None for target in targets),
        complete_count=sum(target.passed for target in targets),
        overall_score=round(overall, 4),
        metrics=metrics,
        targets=targets,
    )


def apply_dimension_correction(
    project: ProjectState,
    request: DimensionCorrectionRequest,
) -> ProjectState:
    working = project.model_copy(deep=True)
    truth = next(
        (item for item in working.dimension_ground_truth if item.id == request.ground_truth_id),
        None,
    )
    if truth is None:
        raise ValueError(f"Ground truth target not found: {request.ground_truth_id}")
    entities = {entity.id: entity for entity in working.ir.entities}
    dimension_line = _require_entity(entities, request.dimension_line_id, LineEntity, "dimension line")
    existing = _dimension_by_id(working.mechanical_ir.dimensions, request.dimension_id)
    if request.text_id:
        text_entity = entities.get(request.text_id)
        synthetic_existing_text = existing is not None and request.text_id == existing.text_id
        if text_entity is not None and not isinstance(text_entity, TextEntity):
            raise ValueError(f"Invalid dimension text entity: {request.text_id}")
        if text_entity is None and not synthetic_existing_text:
            raise ValueError(f"Invalid dimension text entity: {request.text_id}")
    for entity_id in request.arrow_entity_ids:
        _require_entity(entities, entity_id, (LineEntity, PolylineEntity), "arrowhead")
    for entity_id in request.extension_line_ids:
        _require_entity(entities, entity_id, LineEntity, "extension line")
    for entity_id in request.measured_geometry_ids:
        entity = entities.get(entity_id)
        if entity is None or isinstance(entity, TextEntity):
            raise ValueError(f"Invalid measured geometry entity: {entity_id}")

    dimension_id = existing.id if existing else f"manual_dimension_{truth.id}"
    binding_id = existing.binding_id if existing else f"manual_binding_{truth.id}"
    text_entity = _ensure_dimension_text(working, entities, truth, dimension_line, existing, request.text_id)
    parsed = _ground_truth_parsed(truth)
    orientation = _line_orientation(dimension_line, truth.kind)
    measurement_points = _measurement_points(
        truth.kind,
        dimension_line,
        [entities[entity_id] for entity_id in request.extension_line_ids],
        [entities[entity_id] for entity_id in request.measured_geometry_ids],
    )
    arrowheads = [
        _manual_arrowhead(entities[entity_id], dimension_line, index)
        for index, entity_id in enumerate(request.arrow_entity_ids)
    ]
    issues = _correction_issues(
        truth,
        text_entity.id,
        arrowheads,
        request.extension_line_ids,
        request.measured_geometry_ids,
        measurement_points,
    )
    status = "complete" if not issues else "partial"
    dxf_type = _dxf_type(truth.kind, orientation, request.measured_geometry_ids, entities)
    export_ready = status == "complete" and dxf_type is not None
    dimension = MechanicalDimensionObject(
        id=dimension_id,
        binding_id=binding_id,
        kind=truth.kind,
        text=truth.expected_text,
        parsed=parsed,
        confidence=1.0,
        dimension_line_id=dimension_line.id,
        text_id=text_entity.id,
        arrowheads=arrowheads,
        extension_line_ids=_dedupe(request.extension_line_ids),
        measured_geometry_ids=_dedupe(request.measured_geometry_ids),
        target_geometry_ids=_dedupe(request.measured_geometry_ids),
        measurement_points=measurement_points,
        dimension_line_point=[
            round((dimension_line.x1 + dimension_line.x2) * 0.5, 6),
            round((dimension_line.y1 + dimension_line.y2) * 0.5, 6),
        ],
        relation_confidence={
            entity_id: 1.0
            for entity_id in [
                dimension_line.id,
                *request.arrow_entity_ids,
                *request.extension_line_ids,
                *request.measured_geometry_ids,
            ]
        },
        orientation=orientation,
        dxf_dimension_type=dxf_type,
        export_ready=export_ready,
        status=status,
        issues=issues,
        evidence=[
            "manual_override",
            f"ground_truth:{truth.id}",
            *[f"manual_relation:{entity_id}" for entity_id in request.arrow_entity_ids],
        ],
        source="dimension_benchmark_manual_v1",
    )
    _upsert_dimension(working, dimension)
    _upsert_binding(working, dimension_line, dimension, request.arrow_entity_ids)
    _sync_roles(working, dimension)

    truth.matched_dimension_id = dimension.id
    correction = DimensionCorrection(
        ground_truth_id=truth.id,
        dimension_id=dimension.id,
        text_id=dimension.text_id,
        dimension_line_id=dimension.dimension_line_id,
        arrow_entity_ids=_dedupe(request.arrow_entity_ids),
        extension_line_ids=_dedupe(request.extension_line_ids),
        measured_geometry_ids=_dedupe(request.measured_geometry_ids),
        updated_at=datetime.now(timezone.utc),
    )
    working.dimension_corrections = [
        item for item in working.dimension_corrections if item.ground_truth_id != truth.id
    ]
    working.dimension_corrections.append(correction)
    return working


def _evaluate_target(project, truth, dimension, corrected):
    if dimension is None:
        metrics = {
            "text": 0.0,
            "kind": 0.0,
            "dimension_line": 0.0,
            "arrowheads": 0.0,
            "extension_lines": 0.0,
            "measured_geometry": 0.0,
            "definition_points": 0.0,
            "native_ready": 0.0,
        }
        return DimensionTargetEval(
            ground_truth=truth,
            score=0,
            corrected=corrected,
            metrics=metrics,
            missing_relations=list(truth.required_relations),
        )

    entity_ids = {entity.id for entity in project.ir.entities}
    expected_arrows = 1 if truth.kind == "radius" else 2
    expected_extensions = 2 if truth.kind == "linear" else 0
    text_score = _text_match_score(truth, dimension) if dimension.text_id else 0.0
    valid_arrow_count = sum(
        arrow.render_entity_id in entity_ids or arrow.source_entity_id in entity_ids
        for arrow in dimension.arrowheads
    )
    valid_extension_count = sum(entity_id in entity_ids for entity_id in dimension.extension_line_ids)
    valid_measured_count = sum(entity_id in entity_ids for entity_id in dimension.measured_geometry_ids)
    metrics = {
        "text": text_score,
        "kind": 1.0 if dimension.kind == truth.kind else 0.0,
        "dimension_line": 1.0 if dimension.dimension_line_id in entity_ids else 0.0,
        "arrowheads": min(1.0, valid_arrow_count / max(expected_arrows, 1)),
        "extension_lines": (
            min(1.0, valid_extension_count / expected_extensions)
            if expected_extensions
            else 1.0
        ),
        "measured_geometry": 1.0 if valid_measured_count else 0.0,
        "definition_points": 1.0 if len(dimension.measurement_points) >= 2 else 0.0,
        "native_ready": 1.0 if dimension.export_ready and dimension.dxf_dimension_type else 0.0,
    }
    relation_to_metric = {
        "text": "text",
        "dimension_line": "dimension_line",
        "arrowheads": "arrowheads",
        "extension_lines": "extension_lines",
        "measured_geometry": "measured_geometry",
        "definition_points": "definition_points",
    }
    missing = [
        relation
        for relation in truth.required_relations
        if metrics[relation_to_metric[relation]] < PASS_SCORE
    ]
    score_keys = ["kind", *[relation_to_metric[relation] for relation in truth.required_relations]]
    score = sum(metrics[key] for key in score_keys) / len(score_keys)
    passed = not missing and metrics["kind"] == 1.0 and metrics["native_ready"] == 1.0
    return DimensionTargetEval(
        ground_truth=truth,
        matched_dimension_id=dimension.id,
        matched_text=dimension.text,
        score=round(score, 4),
        passed=passed,
        corrected=corrected,
        metrics={key: round(value, 4) for key, value in metrics.items()},
        missing_relations=missing,
    )


def _best_dimension_match(truth, dimensions, used_ids):
    candidates = [
        (_candidate_match_score(truth, dimension), dimension)
        for dimension in dimensions
        if dimension.id not in used_ids
    ]
    if not candidates:
        return None
    score, dimension = max(candidates, key=lambda item: (item[0], item[1].confidence))
    return dimension if score >= MATCH_THRESHOLD else None


def _candidate_match_score(truth, dimension):
    expected = _normalize_dimension_text(truth.expected_text)
    actual = _normalize_dimension_text(dimension.text or dimension.parsed.raw_text)
    ratio = SequenceMatcher(None, expected, actual).ratio() if expected and actual else 0.0
    nominal_match = (
        truth.nominal is not None
        and dimension.parsed.nominal is not None
        and math.isclose(float(truth.nominal), float(dimension.parsed.nominal), rel_tol=0.002, abs_tol=0.04)
    )
    kind_match = truth.kind == dimension.kind
    if expected == actual:
        return 1.0
    if expected and actual.startswith(expected):
        return 0.94
    return ratio * 0.62 + (0.23 if kind_match else 0.0) + (0.15 if nominal_match else 0.0)


def _text_match_score(truth, dimension):
    score = _candidate_match_score(truth, dimension)
    return 1.0 if score >= 0.78 else round(score, 4)


def _normalize_dimension_text(value):
    return re.sub(r"[^0-9a-zφr.+-]", "", (value or "").lower().replace("ø", "φ").replace("Φ", "φ"))


def _ground_truth_parsed(truth):
    parsed = parse_dimension_text(truth.expected_text)
    return ParsedDimensionValue(
        kind=truth.kind,
        raw_text=truth.expected_text,
        nominal=truth.nominal if truth.nominal is not None else parsed.nominal,
        upper_tol=parsed.upper_tol,
        lower_tol=parsed.lower_tol,
        unit=truth.unit,
    )


def _measurement_points(kind, dimension_line, extensions, measured):
    radial = next((entity for entity in measured if isinstance(entity, CircleEntity | ArcEntity)), None)
    if radial is not None and kind in {"diameter", "radius"}:
        dx, dy = dimension_line.x2 - dimension_line.x1, dimension_line.y2 - dimension_line.y1
        length = math.hypot(dx, dy) or 1.0
        direction = (dx / length, dy / length)
        center = [radial.cx, radial.cy]
        edge = [radial.cx + direction[0] * radial.r, radial.cy + direction[1] * radial.r]
        if kind == "radius":
            return [_rounded(center), _rounded(edge)]
        other = [radial.cx - direction[0] * radial.r, radial.cy - direction[1] * radial.r]
        return [_rounded(other), _rounded(edge)]

    points = []
    for extension in extensions:
        endpoints = [(extension.x1, extension.y1), (extension.x2, extension.y2)]
        point = max(endpoints, key=lambda item: _point_to_line_distance(item, dimension_line))
        points.append(_rounded(point))
    if len(points) >= 2:
        dx, dy = dimension_line.x2 - dimension_line.x1, dimension_line.y2 - dimension_line.y1
        points.sort(key=lambda point: point[0] * dx + point[1] * dy)
    return _dedupe_points(points)[:2]


def _manual_arrowhead(entity, dimension_line, index):
    points = _entity_points(entity)
    if not points:
        raise ValueError(f"Arrowhead entity has no usable points: {entity.id}")
    endpoints = [(dimension_line.x1, dimension_line.y1), (dimension_line.x2, dimension_line.y2)]
    tip, endpoint_index = min(
        ((point, endpoint_index) for point in points for endpoint_index, endpoint in enumerate(endpoints)),
        key=lambda item: math.dist(item[0], endpoints[item[1]]),
    )
    cx = sum(point[0] for point in points) / len(points)
    cy = sum(point[1] for point in points) / len(points)
    dx, dy = tip[0] - cx, tip[1] - cy
    length = math.hypot(dx, dy) or 1.0
    return MechanicalArrowhead(
        candidate_id=f"manual_arrow_{index}_{entity.id}",
        source_entity_id=entity.id,
        render_entity_id=entity.id,
        tip_x=round(tip[0], 6),
        tip_y=round(tip[1], 6),
        direction_x=round(dx / length, 6),
        direction_y=round(dy / length, 6),
        score=1.0,
        endpoint="start" if endpoint_index == 0 else "end",
        endpoint_distance=round(math.dist(tip, endpoints[endpoint_index]), 6),
    )


def _correction_issues(truth, text_id, arrows, extensions, measured, points):
    issues = []
    if "text" in truth.required_relations and not text_id:
        issues.append("missing_text")
    expected_arrows = 1 if truth.kind == "radius" else 2
    if len(arrows) < expected_arrows:
        issues.append("missing_arrowhead")
    if truth.kind == "linear" and len(extensions) < 2:
        issues.append("missing_extension_lines")
    if truth.kind in {"linear", "diameter", "radius"} and not measured:
        issues.append("missing_measured_geometry")
    if truth.kind in {"linear", "diameter", "radius"} and len(points) < 2:
        issues.append("missing_definition_points")
    return issues


def _dxf_type(kind, orientation, measured_ids, entities):
    radial = any(isinstance(entities.get(entity_id), CircleEntity | ArcEntity) for entity_id in measured_ids)
    if kind == "diameter" and radial:
        return "diameter"
    if kind == "radius" and radial:
        return "radius"
    if kind == "linear":
        return "aligned" if orientation == "aligned" else "linear"
    return None


def _line_orientation(line, kind):
    if kind == "radius":
        return "radial"
    dx, dy = abs(line.x2 - line.x1), abs(line.y2 - line.y1)
    if dy <= max(dx, 1e-9) * 0.18:
        return "horizontal"
    if dx <= max(dy, 1e-9) * 0.18:
        return "vertical"
    return "aligned"


def _upsert_dimension(project, dimension):
    dimensions = [item for item in project.mechanical_ir.dimensions if item.id != dimension.id]
    dimensions.append(dimension)
    project.mechanical_ir.dimensions = dimensions
    project.mechanical_dimensions = project.mechanical_ir.dimensions
    if dimension.status == "complete":
        project.mechanical_ir.unresolved_binding_ids = [
            item for item in project.mechanical_ir.unresolved_binding_ids if item != dimension.binding_id
        ]
    elif dimension.binding_id not in project.mechanical_ir.unresolved_binding_ids:
        project.mechanical_ir.unresolved_binding_ids.append(dimension.binding_id)


def _upsert_binding(project, line, dimension, arrow_ids):
    from .models import DimensionBinding

    binding = DimensionBinding(
        id=dimension.binding_id,
        dimension_line_id=line.id,
        arrow_ids=_dedupe(arrow_ids),
        text_id=dimension.text_id,
        text=dimension.text,
        parsed=dimension.parsed,
        confidence=1.0,
        kind=dimension.kind,
        line_x1=line.x1,
        line_y1=line.y1,
        line_x2=line.x2,
        line_y2=line.y2,
        binding_method="manual_override",
        graph_path=[f"ground_truth:{dimension.id}", f"line:{line.id}"],
        graph_score=0.0,
        source="dimension_benchmark_manual_v1",
    )
    project.dimension_bindings = [item for item in project.dimension_bindings if item.id != binding.id]
    project.dimension_bindings.append(binding)


def _sync_roles(project, dimension):
    role_ids = {
        "dimension_line": [dimension.dimension_line_id],
        "dimension_text": [dimension.text_id] if dimension.text_id else [],
        "arrowhead": [item.render_entity_id for item in dimension.arrowheads],
        "extension_line": dimension.extension_line_ids,
        "measured_geometry": dimension.measured_geometry_ids,
    }
    for role, ids in role_ids.items():
        current = project.mechanical_ir.entity_roles.setdefault(role, [])
        project.mechanical_ir.entity_roles[role] = _dedupe([*current, *ids])


def _ensure_dimension_text(project, entities, truth, dimension_line, existing, requested_id):
    text_id = requested_id or (existing.text_id if existing else None) or f"manual_text_{truth.id}"
    entity = entities.get(text_id)
    if isinstance(entity, TextEntity):
        entity.text = truth.expected_text
        return entity

    midpoint_x = (dimension_line.x1 + dimension_line.x2) * 0.5
    midpoint_y = (dimension_line.y1 + dimension_line.y2) * 0.5
    binding = next(
        (
            item
            for item in project.dimension_bindings
            if existing is not None and item.id == existing.binding_id
        ),
        None,
    )
    text_x = binding.text_x if binding and binding.text_x is not None else midpoint_x
    text_y = binding.text_y if binding and binding.text_y is not None else midpoint_y + 2.5
    if math.dist((text_x, text_y), (midpoint_x, midpoint_y)) > 32.0:
        text_x, text_y = midpoint_x, midpoint_y + 2.5
    entity = TextEntity(
        id=text_id,
        layer="TEXT",
        x=round(text_x, 6),
        y=round(text_y, 6),
        text=truth.expected_text,
        height=2.5,
        group="dimensions",
        tags=["dimension_text", "manual_semantic"],
        metadata={"ground_truth_id": truth.id, "source": "dimension_benchmark_manual_v1"},
    )
    project.ir.entities.append(entity)
    entities[entity.id] = entity
    return entity


def _dimension_by_id(dimensions, dimension_id):
    if not dimension_id:
        return None
    return next(
        (
            item
            for item in dimensions
            if item.id == dimension_id or item.binding_id == dimension_id
        ),
        None,
    )


def _require_entity(entities, entity_id, entity_type, label):
    entity = entities.get(entity_id)
    if entity is None or not isinstance(entity, entity_type):
        raise ValueError(f"Invalid {label} entity: {entity_id}")
    return entity


def _entity_points(entity: Entity):
    if isinstance(entity, LineEntity):
        return [(entity.x1, entity.y1), (entity.x2, entity.y2)]
    if isinstance(entity, PolylineEntity):
        return [tuple(point[:2]) for point in entity.points if len(point) >= 2]
    if isinstance(entity, RectangleEntity):
        return [
            (entity.x, entity.y),
            (entity.x + entity.width, entity.y),
            (entity.x + entity.width, entity.y + entity.height),
            (entity.x, entity.y + entity.height),
        ]
    return []


def _point_to_line_distance(point, line):
    vx, vy = line.x2 - line.x1, line.y2 - line.y1
    length_sq = vx * vx + vy * vy
    if length_sq <= 1e-12:
        return math.dist(point, (line.x1, line.y1))
    t = max(0.0, min(1.0, ((point[0] - line.x1) * vx + (point[1] - line.y1) * vy) / length_sq))
    projection = (line.x1 + t * vx, line.y1 + t * vy)
    return math.dist(point, projection)


def _rounded(point):
    return [round(float(point[0]), 6), round(float(point[1]), 6)]


def _dedupe(items):
    return list(dict.fromkeys(items))


def _dedupe_points(points):
    result = []
    for point in points:
        if point not in result:
            result.append(point)
    return result
