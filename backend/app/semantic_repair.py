from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone

from .cad_layers import CENTER, HATCH, REFERENCE_TRACE, TEXT, TITLE_BLOCK, canonical_layer_name
from .dimension_benchmark import apply_dimension_correction, evaluate_dimension_benchmark
from .llm_agent import LlmUnavailable, plan_dimension_repair_order_llm
from .models import (
    ArcEntity,
    CircleEntity,
    DimensionBenchmarkReport,
    DimensionCorrectionRequest,
    DimensionTargetEval,
    Entity,
    LineEntity,
    PolylineEntity,
    ProjectState,
    RectangleEntity,
    SemanticRepairRequest,
    SemanticRepairRun,
    SemanticRepairStep,
    TextEntity,
    new_id,
)

MAX_REPAIR_RUNS = 10
MIN_LINE_LENGTH_MM = 6.0
MAX_RAY_DISTANCE_MM = 90.0
RAY_LATERAL_TOLERANCE_MM = 3.2


@dataclass(frozen=True)
class RayHit:
    entity_id: str
    point: tuple[float, float]
    distance: float
    score: float


def run_semantic_repair_agent(
    project: ProjectState,
    request: SemanticRepairRequest,
) -> tuple[ProjectState, DimensionBenchmarkReport, SemanticRepairRun]:
    """Repair semantic dimensions with local tools under a monotonic eval guard."""

    working = project.model_copy(deep=True)
    baseline = evaluate_dimension_benchmark(working)
    assignments = _linear_line_assignments(working, baseline)
    repairable = [
        target
        for target in baseline.targets
        if not target.passed and target.ground_truth.id in assignments
    ]
    deterministic_order = [
        target.ground_truth.id
        for target in sorted(
            repairable,
            key=lambda item: (
                item.matched_dimension_id is None,
                -item.score,
                -(item.ground_truth.nominal or 0.0),
            ),
        )
    ]

    planner_source = "deterministic"
    planner_model = None
    planner_reason = "Local benchmark priority: matched targets first, then score and nominal size."
    llm_calls = 0
    order: list[str] = []
    if repairable and request.use_llm:
        llm_calls = 1
        try:
            llm_order, planner_reason, planner_model = plan_dimension_repair_order_llm(
                baseline,
                deterministic_order,
                request.max_steps,
            )
        except LlmUnavailable as exc:
            planner_source = "deterministic_fallback"
            planner_reason = f"DeepSeek unavailable; local order used. {exc}"
        else:
            planner_source = "deepseek"
            order.extend(llm_order)
    for target_id in deterministic_order:
        if target_id not in order:
            order.append(target_id)
    order = order[: request.max_steps]

    current_report = baseline
    steps: list[SemanticRepairStep] = []
    accepted = 0
    rejected = 0
    reserved_dimension_lines = set(assignments.values())
    for index, target_id in enumerate(order):
        before_target = _target_by_id(current_report, target_id)
        if before_target is None:
            continue
        overall_before = current_report.overall_score
        trial = working.model_copy(deep=True)
        try:
            correction, selected, tool_calls = _build_linear_repair(
                trial,
                before_target,
                assignments[target_id],
                reserved_dimension_lines,
            )
            trial = apply_dimension_correction(trial, correction)
            _mark_auto_repair(trial, target_id, tool_calls)
            trial_report = evaluate_dimension_benchmark(trial)
            after_target = _target_by_id(trial_report, target_id)
            if after_target is None:
                raise ValueError("Target disappeared after repair evaluation")
        except ValueError as exc:
            rejected += 1
            steps.append(
                SemanticRepairStep(
                    index=index,
                    ground_truth_id=target_id,
                    label=before_target.ground_truth.label,
                    status="error",
                    score_before=before_target.score,
                    score_after=before_target.score,
                    overall_before=overall_before,
                    overall_after=overall_before,
                    missing_before=before_target.missing_relations,
                    missing_after=before_target.missing_relations,
                    detail=str(exc),
                )
            )
            continue

        target_gain = after_target.score - before_target.score
        monotonic = trial_report.overall_score + 1e-9 >= current_report.overall_score
        accept = target_gain + 1e-9 >= request.min_gain and monotonic
        if accept:
            status = "accepted"
            detail = f"Accepted monotonic repair; target gain {target_gain:.4f}."
            working = trial
            current_report = trial_report
            accepted += 1
        else:
            status = "rejected"
            detail = (
                f"Rolled back trial; target gain {target_gain:.4f}, "
                f"overall {current_report.overall_score:.4f}->{trial_report.overall_score:.4f}."
            )
            rejected += 1
        steps.append(
            SemanticRepairStep(
                index=index,
                ground_truth_id=target_id,
                label=before_target.ground_truth.label,
                status=status,
                tool_calls=tool_calls,
                score_before=before_target.score,
                score_after=after_target.score,
                overall_before=overall_before,
                overall_after=trial_report.overall_score if accept else overall_before,
                missing_before=before_target.missing_relations,
                missing_after=after_target.missing_relations,
                selected_entities=selected,
                detail=detail,
            )
        )

    remaining_repairable = any(
        not target.passed and target.ground_truth.id in assignments
        for target in current_report.targets
    )
    if not repairable:
        stopped_reason = "no_repairable_targets"
    elif not accepted:
        stopped_reason = "no_monotonic_improvement"
    elif not remaining_repairable:
        stopped_reason = "repair_pass_complete"
    elif len(order) >= request.max_steps:
        stopped_reason = "step_budget_reached"
    else:
        stopped_reason = "repair_pass_complete"
    run = SemanticRepairRun(
        id=new_id("semantic_repair"),
        created_at=datetime.now(timezone.utc),
        planner_source=planner_source,
        planner_model=planner_model,
        planner_reason=planner_reason,
        budget=request.max_steps,
        llm_calls=llm_calls,
        before_score=baseline.overall_score,
        after_score=current_report.overall_score,
        accepted_steps=accepted,
        rejected_steps=rejected,
        stopped_reason=stopped_reason,
        steps=steps,
    )
    return working, current_report, run


def append_semantic_repair_run(project: ProjectState, run: SemanticRepairRun) -> None:
    project.semantic_repair_runs = [*project.semantic_repair_runs, run][-MAX_REPAIR_RUNS:]


def _build_linear_repair(
    project: ProjectState,
    target: DimensionTargetEval,
    dimension_line_id: str,
    reserved_dimension_lines: set[str],
) -> tuple[DimensionCorrectionRequest, dict[str, list[str]], list[str]]:
    entities = {entity.id: entity for entity in project.ir.entities}
    dimension_line = entities.get(dimension_line_id)
    if not isinstance(dimension_line, LineEntity):
        raise ValueError(f"Repair dimension line is unavailable: {dimension_line_id}")
    dimension = next(
        (
            item
            for item in project.mechanical_ir.dimensions
            if item.id == target.matched_dimension_id or item.binding_id == target.matched_dimension_id
        ),
        None,
    )
    arrow_ids = _materialize_arrowheads(project, target.ground_truth.id, dimension_line)
    extension_ids, measured_ids = _materialize_extension_relations(
        project,
        target.ground_truth.id,
        dimension_line,
        reserved_dimension_lines,
    )
    selected = {
        "dimension_line": [dimension_line.id],
        "arrowheads": arrow_ids,
        "extension_lines": extension_ids,
        "measured_geometry": measured_ids,
    }
    tool_calls = [
        "rank_dimension_line_candidates",
        "materialize_solid_arrowheads",
        "trace_extension_rays",
        "bind_measured_geometry",
        "evaluate_dimension_trial",
    ]
    return (
        DimensionCorrectionRequest(
            ground_truth_id=target.ground_truth.id,
            dimension_id=dimension.id if dimension else None,
            text_id=dimension.text_id if dimension else None,
            dimension_line_id=dimension_line.id,
            arrow_entity_ids=arrow_ids,
            extension_line_ids=extension_ids,
            measured_geometry_ids=measured_ids,
        ),
        selected,
        tool_calls,
    )


def _linear_line_assignments(
    project: ProjectState,
    report: DimensionBenchmarkReport,
) -> dict[str, str]:
    sheet = _sheet_bounds(project)
    candidates: list[tuple[float, LineEntity]] = []
    for entity in project.ir.entities:
        if not isinstance(entity, LineEntity) or not _is_dimension_line_candidate(entity, sheet):
            continue
        length = _line_length(entity)
        candidates.append((length, entity))
    candidates.sort(key=lambda item: (-item[0], item[1].id))
    targets = sorted(
        [target for target in report.targets if target.ground_truth.kind == "linear"],
        key=lambda item: (-(item.ground_truth.nominal or 0.0), item.ground_truth.id),
    )
    return {
        target.ground_truth.id: candidate.id
        for target, (_, candidate) in zip(targets, candidates)
    }


def _is_dimension_line_candidate(entity: LineEntity, sheet: tuple[float, float, float, float]) -> bool:
    layer = canonical_layer_name(entity.layer)
    if layer in {REFERENCE_TRACE, TITLE_BLOCK, HATCH, CENTER, TEXT}:
        return False
    if entity.group in {"sheet", "title_block", "parameter_table", "reference_trace"}:
        return False
    if set(entity.tags).intersection({"dimension_arrow", "arrowhead", "grid", "sheet", "drawing_frame"}):
        return False
    length = _line_length(entity)
    if length < MIN_LINE_LENGTH_MM:
        return False
    dx = abs(entity.x2 - entity.x1)
    dy = abs(entity.y2 - entity.y1)
    if dy > max(dx, 1e-9) * 0.08:
        return False
    sheet_x, sheet_y, sheet_width, sheet_height = sheet
    midpoint_x = (entity.x1 + entity.x2) * 0.5
    midpoint_y = (entity.y1 + entity.y2) * 0.5
    if not (sheet_x + sheet_width * 0.12 <= midpoint_x <= sheet_x + sheet_width * 0.72):
        return False
    if not (sheet_y + sheet_height * 0.25 <= midpoint_y <= sheet_y + sheet_height * 0.78):
        return False
    # Overall dimensions can span most of the part. Sheet/frame lines are
    # excluded above by their semantic group, tags, and drawing-zone bounds.
    return length <= sheet_width * 0.70


def _materialize_arrowheads(project: ProjectState, target_id: str, line: LineEntity) -> list[str]:
    dx, dy = line.x2 - line.x1, line.y2 - line.y1
    length = math.hypot(dx, dy)
    if length <= 1e-9:
        raise ValueError("Cannot create arrows on a zero-length dimension line")
    tangent = (dx / length, dy / length)
    perpendicular = (-tangent[1], tangent[0])
    arrow_length = min(2.8, max(1.4, length * 0.035))
    half_width = arrow_length * 0.42
    ids: list[str] = []
    for endpoint_name, tip, inward in [
        ("start", (line.x1, line.y1), tangent),
        ("end", (line.x2, line.y2), (-tangent[0], -tangent[1])),
    ]:
        entity_id = f"auto_arrow_{target_id}_{endpoint_name}"
        base = (tip[0] + inward[0] * arrow_length, tip[1] + inward[1] * arrow_length)
        points = [
            [round(tip[0], 6), round(tip[1], 6)],
            [round(base[0] + perpendicular[0] * half_width, 6), round(base[1] + perpendicular[1] * half_width, 6)],
            [round(base[0] - perpendicular[0] * half_width, 6), round(base[1] - perpendicular[1] * half_width, 6)],
        ]
        _upsert_entity(
            project,
            PolylineEntity(
                id=entity_id,
                layer="DIMENSION",
                points=points,
                closed=True,
                group="dimensions",
                tags=["dimension_arrow", "arrowhead", "solid_fill", "auto_repair"],
                stroke_width=0.25,
                metadata={"solid_fill": True, "ground_truth_id": target_id, "source": "semantic_repair_v1"},
            ),
        )
        ids.append(entity_id)
    return ids


def _materialize_extension_relations(
    project: ProjectState,
    target_id: str,
    line: LineEntity,
    reserved_dimension_lines: set[str],
) -> tuple[list[str], list[str]]:
    dx, dy = line.x2 - line.x1, line.y2 - line.y1
    length = math.hypot(dx, dy)
    if length <= 1e-9:
        return [], []
    tangent = (dx / length, dy / length)
    normal = (-tangent[1], tangent[0])
    midpoint = ((line.x1 + line.x2) * 0.5, (line.y1 + line.y2) * 0.5)
    body_center = _body_center(project, reserved_dimension_lines)
    if _dot((body_center[0] - midpoint[0], body_center[1] - midpoint[1]), normal) < 0:
        normal = (-normal[0], -normal[1])

    extension_ids: list[str] = []
    measured_ids: list[str] = []
    excluded = {line.id, *reserved_dimension_lines}
    for endpoint_name, origin in [("start", (line.x1, line.y1)), ("end", (line.x2, line.y2))]:
        hit = _nearest_ray_hit(project.ir.entities, origin, normal, tangent, excluded)
        if hit is None:
            continue
        extension_id = f"auto_extension_{target_id}_{endpoint_name}"
        _upsert_entity(
            project,
            LineEntity(
                id=extension_id,
                layer="DIMENSION",
                x1=round(origin[0], 6),
                y1=round(origin[1], 6),
                x2=round(hit.point[0], 6),
                y2=round(hit.point[1], 6),
                group="dimensions",
                tags=["extension_line", "auto_repair"],
                stroke_width=0.18,
                metadata={"ground_truth_id": target_id, "source": "semantic_repair_v1"},
            ),
        )
        extension_ids.append(extension_id)
        measured_ids.append(hit.entity_id)
        excluded.add(hit.entity_id)
    return extension_ids, measured_ids


def _nearest_ray_hit(
    entities: list[Entity],
    origin: tuple[float, float],
    normal: tuple[float, float],
    tangent: tuple[float, float],
    excluded: set[str],
) -> RayHit | None:
    hits: list[RayHit] = []
    for entity in entities:
        if entity.id in excluded or not _can_bind_measured_geometry(entity):
            continue
        entity_penalty = 0.0 if entity.group == "promoted_geometry" else 1.5
        for first, second in _entity_segments(entity):
            segment = (second[0] - first[0], second[1] - first[1])
            segment_length = math.hypot(*segment)
            if segment_length <= 1e-9:
                continue
            segment_unit = (segment[0] / segment_length, segment[1] / segment_length)
            alignment = abs(_dot(segment_unit, normal))
            points = [first, second]
            lateral_first = _dot((first[0] - origin[0], first[1] - origin[1]), tangent)
            lateral_second = _dot((second[0] - origin[0], second[1] - origin[1]), tangent)
            denominator = lateral_first - lateral_second
            if lateral_first * lateral_second <= 0 and abs(denominator) > 1e-9:
                ratio = lateral_first / denominator
                if 0 <= ratio <= 1:
                    points.append(
                        (
                            first[0] + ratio * segment[0],
                            first[1] + ratio * segment[1],
                        )
                    )
            for point in points:
                relative = (point[0] - origin[0], point[1] - origin[1])
                distance = _dot(relative, normal)
                lateral = abs(_dot(relative, tangent))
                if not (3.5 <= distance <= MAX_RAY_DISTANCE_MM):
                    continue
                if lateral > RAY_LATERAL_TOLERANCE_MM:
                    continue
                score = distance + lateral * 4.0 + (1.0 - alignment) * 18.0 + entity_penalty
                hits.append(RayHit(entity.id, point, distance, score))
    return min(hits, key=lambda item: (item.score, item.entity_id)) if hits else None


def _can_bind_measured_geometry(entity: Entity) -> bool:
    if isinstance(entity, TextEntity):
        return False
    layer = canonical_layer_name(entity.layer)
    if layer in {REFERENCE_TRACE, TITLE_BLOCK, HATCH, TEXT}:
        return False
    if layer == CENTER and entity.group != "editable_linework":
        return False
    if entity.group in {"sheet", "title_block", "parameter_table", "reference_trace", "dimensions"}:
        return False
    if set(entity.tags).intersection({"dimension_arrow", "arrowhead", "grid", "sheet", "drawing_frame"}):
        return False
    return isinstance(entity, LineEntity | PolylineEntity | RectangleEntity | CircleEntity | ArcEntity)


def _entity_segments(entity: Entity) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    if isinstance(entity, LineEntity):
        return [((entity.x1, entity.y1), (entity.x2, entity.y2))]
    if isinstance(entity, PolylineEntity):
        points = [tuple(point[:2]) for point in entity.points if len(point) >= 2]
        pairs = list(zip(points, points[1:]))
        if entity.closed and len(points) > 2:
            pairs.append((points[-1], points[0]))
        return pairs
    if isinstance(entity, RectangleEntity):
        points = [
            (entity.x, entity.y),
            (entity.x + entity.width, entity.y),
            (entity.x + entity.width, entity.y + entity.height),
            (entity.x, entity.y + entity.height),
        ]
        return list(zip(points, [*points[1:], points[0]]))
    if isinstance(entity, CircleEntity | ArcEntity):
        start = entity.start_angle if isinstance(entity, ArcEntity) else 0.0
        end = entity.end_angle if isinstance(entity, ArcEntity) else 360.0
        steps = 24
        points = [
            (
                entity.cx + math.cos(math.radians(start + (end - start) * index / steps)) * entity.r,
                entity.cy + math.sin(math.radians(start + (end - start) * index / steps)) * entity.r,
            )
            for index in range(steps + 1)
        ]
        return list(zip(points, points[1:]))
    return []


def _body_center(project: ProjectState, reserved_dimension_lines: set[str]) -> tuple[float, float]:
    sheet_x, sheet_y, sheet_width, sheet_height = _sheet_bounds(project)
    points = [
        ((entity.x1 + entity.x2) * 0.5, (entity.y1 + entity.y2) * 0.5)
        for entity in project.ir.entities
        if isinstance(entity, LineEntity)
        and entity.id not in reserved_dimension_lines
        and entity.group == "promoted_geometry"
        and sheet_x + sheet_width * 0.12 <= (entity.x1 + entity.x2) * 0.5 <= sheet_x + sheet_width * 0.72
        and sheet_y + sheet_height * 0.25 <= (entity.y1 + entity.y2) * 0.5 <= sheet_y + sheet_height * 0.78
    ]
    if points:
        return (statistics.median(point[0] for point in points), statistics.median(point[1] for point in points))
    return (sheet_x + sheet_width * 0.42, sheet_y + sheet_height * 0.52)


def _sheet_bounds(project: ProjectState) -> tuple[float, float, float, float]:
    rectangles = [entity for entity in project.ir.entities if isinstance(entity, RectangleEntity)]
    if rectangles:
        sheet = max(rectangles, key=lambda item: item.width * item.height)
        return (sheet.x, sheet.y, max(sheet.width, 1.0), max(sheet.height, 1.0))
    points = [point for entity in project.ir.entities for segment in _entity_segments(entity) for point in segment]
    if not points:
        return (0.0, 0.0, 420.0, 297.0)
    min_x = min(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_x = max(point[0] for point in points)
    max_y = max(point[1] for point in points)
    return (min_x, min_y, max(max_x - min_x, 1.0), max(max_y - min_y, 1.0))


def _mark_auto_repair(project: ProjectState, target_id: str, tool_calls: list[str]) -> None:
    correction = next(
        (item for item in project.dimension_corrections if item.ground_truth_id == target_id),
        None,
    )
    if correction is None:
        return
    correction.source = "auto_repair"
    dimension = next(
        (item for item in project.mechanical_ir.dimensions if item.id == correction.dimension_id),
        None,
    )
    if dimension is not None:
        dimension.source = "semantic_repair_v1"
        dimension.evidence = [*dimension.evidence, *[f"agent_tool:{tool}" for tool in tool_calls]]


def _target_by_id(report: DimensionBenchmarkReport, target_id: str) -> DimensionTargetEval | None:
    return next((target for target in report.targets if target.ground_truth.id == target_id), None)


def _upsert_entity(project: ProjectState, entity: Entity) -> None:
    project.ir.entities = [item for item in project.ir.entities if item.id != entity.id]
    project.ir.entities.append(entity)


def _line_length(line: LineEntity) -> float:
    return math.hypot(line.x2 - line.x1, line.y2 - line.y1)


def _dot(first: tuple[float, float], second: tuple[float, float]) -> float:
    return first[0] * second[0] + first[1] * second[1]
