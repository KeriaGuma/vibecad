from __future__ import annotations

import math
from dataclasses import dataclass

from .cad_layers import REFERENCE_TRACE, canonical_layer_name
from .cad_ops import apply_operations
from .dimension_semantics import parse_dimension_text
from .mechanical_edit import (
    EDIT_INTENT_RE,
    _dimension_text_operation,
    _explicit_dimension_id,
    _find_dimension,
    _format_number,
    _numbers_without_explicit_ids,
    _replacement_text,
)
from .models import (
    ArcEntity,
    CircleEntity,
    DiffItem,
    Entity,
    LineEntity,
    MechanicalDimensionObject,
    MechanicalOperation,
    MechanicalTransaction,
    MechanicalValidation,
    Operation,
    PolylineEntity,
    ProjectState,
    RectangleEntity,
    TextEntity,
    new_id,
)

DRIVE_TOLERANCE_MM = 0.01
MAX_TRANSACTIONS = 20
UNDO_RE = {"撤销", "撤消", "undo", "回退", "回滚"}


class MechanicalDriveError(ValueError):
    """Raised when a semantic edit cannot be executed without risking geometry."""


@dataclass(frozen=True)
class MechanicalDriveResult:
    project: ProjectState
    operations: list[Operation]
    diffs: list[DiffItem]
    reply: str
    validation: MechanicalValidation


def is_undo_command(message: str) -> bool:
    normalized = message.strip().lower().rstrip("。.!！")
    return normalized in UNDO_RE


def plan_mechanical_drive_deterministic(
    message: str,
    project: ProjectState,
) -> MechanicalOperation | None:
    """Resolve exact value replacement commands without spending an LLM call."""

    if not EDIT_INTENT_RE.search(message) or not project.mechanical_ir.dimensions:
        return None
    explicit_id = _explicit_dimension_id(message)
    values = _numbers_without_explicit_ids(message)
    if not values:
        return None
    old_value = values[-2] if len(values) >= 2 else None
    if explicit_id is None and old_value is None:
        return None
    dimension = _find_dimension(project.mechanical_ir.dimensions, explicit_id, old_value)
    if dimension is None or not is_driveable_dimension(project, dimension):
        return None
    return MechanicalOperation(
        dimension_id=dimension.id,
        target_value=values[-1],
        planner_source="deterministic",
        confidence=1.0,
        reason=f"Exact dimension replacement resolved to {dimension.binding_id}.",
    )


def is_driveable_dimension(project: ProjectState, dimension: MechanicalDimensionObject) -> bool:
    if not dimension.export_ready or dimension.status != "complete":
        return False
    if dimension.kind == "linear":
        return (
            dimension.orientation in {"horizontal", "vertical"}
            and len(dimension.measurement_points) >= 2
            and bool(dimension.measured_geometry_ids)
        )
    if dimension.kind in {"diameter", "radius"}:
        entities = _entities_by_id(project)
        return any(
            isinstance(entities.get(entity_id), CircleEntity | ArcEntity)
            for entity_id in dimension.measured_geometry_ids
        )
    return False


def execute_mechanical_operation(
    project: ProjectState,
    plan: MechanicalOperation,
    command: str,
) -> MechanicalDriveResult:
    """Apply a dimension drive transactionally and validate geometry before commit."""

    working = project.model_copy(deep=True)
    dimension = _dimension_by_id(working, plan.dimension_id)
    if dimension is None:
        raise MechanicalDriveError(f"Mechanical dimension not found: {plan.dimension_id}")
    if not is_driveable_dimension(working, dimension):
        raise MechanicalDriveError(
            "这个尺寸还没有完整绑定尺寸线、界线和被测轮廓，当前只能修改标注文字，不能安全驱动几何。"
        )

    before_ir = working.ir.model_copy(deep=True)
    before_bindings = [item.model_copy(deep=True) for item in working.dimension_bindings]
    before_mechanical_ir = working.mechanical_ir.model_copy(deep=True)
    before_history_length = len(working.history)
    before_edit_mode = dimension.edit_mode
    before_text = dimension.text or dimension.binding_id

    if dimension.kind == "linear":
        operations, semantic = _build_linear_drive(working, dimension, plan)
    elif dimension.kind in {"diameter", "radius"}:
        operations, semantic = _build_radial_drive(working, dimension, plan)
    else:  # pragma: no cover - guarded by is_driveable_dimension
        raise MechanicalDriveError(f"Unsupported driving dimension kind: {dimension.kind}")

    _ensure_operations_are_editable(working, operations)
    next_ir, diffs = apply_operations(working.ir, operations)
    working.ir = next_ir
    _sync_dimension_semantics(working, dimension.binding_id, plan, semantic)
    validation = _validate_drive(working, dimension.binding_id, plan.target_value)
    if not validation.passed:
        raise MechanicalDriveError("机械尺寸驱动校验失败：" + "; ".join(validation.errors))

    semantic_diffs = [
        DiffItem(
            path=f"mechanical_dimensions.{dimension.id}.measured_value",
            before=semantic["old_value"],
            after=validation.measured_value,
        ),
        DiffItem(
            path=f"mechanical_dimensions.{dimension.id}.edit_mode",
            before=before_edit_mode,
            after="driving",
        ),
    ]
    diffs = [*diffs, *semantic_diffs]
    transaction = MechanicalTransaction(
        id=new_id("mechanical_tx"),
        command=command,
        planner_source=plan.planner_source,
        operations=operations,
        diffs=diffs,
        validation=validation,
        before_ir=before_ir,
        before_dimension_bindings=before_bindings,
        before_mechanical_ir=before_mechanical_ir,
        before_history_length=before_history_length,
    )
    working.mechanical_transactions = [*working.mechanical_transactions, transaction][-MAX_TRANSACTIONS:]
    working.history.extend(operations)
    working.diffs = diffs
    reply = (
        f"已完成机械尺寸驱动：{before_text} → {_format_number(plan.target_value)} {working.ir.units}。"
        f"几何实测值 {validation.measured_value:g} {working.ir.units}，校验通过；"
        f"规划来源：{'DeepSeek V4 Flash' if plan.planner_source == 'deepseek' else '本地精确解析'}。"
        "输入“撤销”可以回退这次事务。"
    )
    return MechanicalDriveResult(working, operations, diffs, reply, validation)


def undo_last_mechanical_transaction(project: ProjectState) -> tuple[ProjectState, str, list[DiffItem]]:
    if not project.mechanical_transactions:
        return project, "当前没有可撤销的机械尺寸事务。", []
    working = project.model_copy(deep=True)
    transaction = working.mechanical_transactions.pop()
    current_value = transaction.validation.measured_value
    working.ir = transaction.before_ir.model_copy(deep=True)
    working.dimension_bindings = [item.model_copy(deep=True) for item in transaction.before_dimension_bindings]
    working.mechanical_ir = transaction.before_mechanical_ir.model_copy(deep=True)
    working.mechanical_dimensions = working.mechanical_ir.dimensions
    working.history = working.history[: transaction.before_history_length]
    diffs = [
        DiffItem(
            path=f"mechanical_transactions.{transaction.id}",
            before={"measured_value": current_value, "command": transaction.command},
            after="rolled_back",
        )
    ]
    working.diffs = diffs
    return working, f"已撤销：{transaction.command}", diffs


def _build_linear_drive(project, dimension, plan):
    points = [list(map(float, point[:2])) for point in dimension.measurement_points[:2]]
    if len(points) < 2:
        raise MechanicalDriveError("线性尺寸缺少两个定义点。")
    if dimension.orientation not in {"horizontal", "vertical"}:
        raise MechanicalDriveError("当前只开放水平和垂直线性尺寸驱动，斜尺寸仍保持只读。")

    moving_index = 1 if plan.anchor == "start" else 0
    fixed_index = 1 - moving_index
    moving = points[moving_index]
    fixed = points[fixed_index]
    axis_index = 0 if dimension.orientation == "horizontal" else 1
    old_value = abs(points[1][axis_index] - points[0][axis_index])
    direction = 1.0 if moving[axis_index] >= fixed[axis_index] else -1.0
    target_point = moving.copy()
    target_point[axis_index] = fixed[axis_index] + direction * plan.target_value
    delta = [target_point[0] - moving[0], target_point[1] - moving[1]]

    operations: dict[str, Operation] = {}
    entities = _entities_by_id(project)
    geometry_updates = 0
    for entity_id in dimension.measured_geometry_ids:
        entity = entities.get(entity_id)
        changes = _linear_geometry_changes(entity, moving, fixed, delta, axis_index, old_value)
        if changes:
            _merge_modify(operations, entity_id, changes, dimension, "Resize measured geometry")
            geometry_updates += 1
    if geometry_updates == 0:
        raise MechanicalDriveError("没有找到能随该尺寸安全移动的轮廓端点。")

    if moving_index < len(dimension.extension_line_ids):
        extension_id = dimension.extension_line_ids[moving_index]
        extension = entities.get(extension_id)
        if isinstance(extension, LineEntity):
            _merge_modify(
                operations,
                extension_id,
                _translated_changes(extension, delta[0], delta[1]),
                dimension,
                "Move witness line with driven feature",
            )

    dimension_line = entities.get(dimension.dimension_line_id)
    if isinstance(dimension_line, LineEntity):
        suffix = "1" if moving_index == 0 else "2"
        line_changes = {}
        if abs(delta[0]) > 1e-12:
            line_changes[f"x{suffix}"] = getattr(dimension_line, f"x{suffix}") + delta[0]
        if abs(delta[1]) > 1e-12:
            line_changes[f"y{suffix}"] = getattr(dimension_line, f"y{suffix}") + delta[1]
        _merge_modify(
            operations,
            dimension_line.id,
            line_changes,
            dimension,
            "Extend dimension line to driven feature",
        )

    endpoint_name = "start" if moving_index == 0 else "end"
    for arrow in dimension.arrowheads:
        if arrow.endpoint != endpoint_name:
            continue
        arrow_entity = entities.get(arrow.render_entity_id)
        if arrow_entity is not None:
            _merge_modify(
                operations,
                arrow_entity.id,
                _translated_changes(arrow_entity, delta[0], delta[1]),
                dimension,
                "Move solid arrowhead with driven endpoint",
            )

    new_text = _replacement_text(dimension, _format_number(plan.target_value))
    text_entity = entities.get(dimension.text_id or "")
    if isinstance(text_entity, TextEntity):
        text_changes = {"text": new_text}
        if abs(delta[0]) > 1e-12:
            text_changes["x"] = text_entity.x + delta[0] * 0.5
        if abs(delta[1]) > 1e-12:
            text_changes["y"] = text_entity.y + delta[1] * 0.5
        _merge_modify(
            operations,
            text_entity.id,
            text_changes,
            dimension,
            "Update and recenter dimension text",
        )
        text_id = text_entity.id
    else:
        binding = next((item for item in project.dimension_bindings if item.id == dimension.binding_id), None)
        text_operation, text_id = _dimension_text_operation(project, dimension, binding, new_text)
        operations[text_id] = text_operation

    points[moving_index] = target_point
    line_point = list(dimension.dimension_line_point or [sum(point[0] for point in points) / 2, sum(point[1] for point in points) / 2])
    line_point[0] += delta[0] * 0.5
    line_point[1] += delta[1] * 0.5
    return list(operations.values()), {
        "old_value": old_value,
        "new_text": new_text,
        "measurement_points": points,
        "dimension_line_point": line_point,
        "delta": delta,
        "moving_index": moving_index,
        "text_id": text_id,
        "arrow_deltas": {
            arrow.render_entity_id: delta
            for arrow in dimension.arrowheads
            if arrow.endpoint == endpoint_name
        },
    }


def _build_radial_drive(project, dimension, plan):
    entities = _entities_by_id(project)
    radial = next(
        (
            entity
            for entity_id in dimension.measured_geometry_ids
            if isinstance((entity := entities.get(entity_id)), CircleEntity | ArcEntity)
        ),
        None,
    )
    if radial is None:
        raise MechanicalDriveError("直径/半径尺寸没有绑定圆或圆弧。")
    old_value = radial.r * 2 if dimension.kind == "diameter" else radial.r
    new_radius = plan.target_value / 2 if dimension.kind == "diameter" else plan.target_value
    operations: dict[str, Operation] = {}
    _merge_modify(operations, radial.id, {"r": new_radius}, dimension, "Resize measured circle or arc")

    old_points = [list(map(float, point[:2])) for point in dimension.measurement_points[:2]]
    center = [radial.cx, radial.cy]
    direction = _radial_direction(old_points, center)
    if dimension.kind == "diameter":
        new_points = [
            [center[0] - direction[0] * new_radius, center[1] - direction[1] * new_radius],
            [center[0] + direction[0] * new_radius, center[1] + direction[1] * new_radius],
        ]
    else:
        new_points = [center, [center[0] + direction[0] * new_radius, center[1] + direction[1] * new_radius]]

    dimension_line = entities.get(dimension.dimension_line_id)
    if isinstance(dimension_line, LineEntity):
        _merge_modify(
            operations,
            dimension_line.id,
            {"x1": new_points[0][0], "y1": new_points[0][1], "x2": new_points[1][0], "y2": new_points[1][1]},
            dimension,
            "Resize radial dimension line",
        )

    arrow_deltas: dict[str, list[float]] = {}
    for arrow in dimension.arrowheads:
        old_target = min(old_points, key=lambda point: math.dist(point, [arrow.tip_x, arrow.tip_y])) if old_points else center
        index = old_points.index(old_target) if old_target in old_points else 1
        new_target = new_points[min(index, len(new_points) - 1)]
        dx, dy = new_target[0] - old_target[0], new_target[1] - old_target[1]
        arrow_deltas[arrow.render_entity_id] = [dx, dy]
        arrow_entity = entities.get(arrow.render_entity_id)
        if arrow_entity is not None:
            _merge_modify(
                operations,
                arrow_entity.id,
                _translated_changes(arrow_entity, dx, dy),
                dimension,
                "Move radial arrowhead to resized geometry",
            )

    new_text = _replacement_text(dimension, _format_number(plan.target_value))
    text_entity = entities.get(dimension.text_id or "")
    if isinstance(text_entity, TextEntity):
        _merge_modify(operations, text_entity.id, {"text": new_text}, dimension, "Update radial dimension text")
        text_id = text_entity.id
    else:
        binding = next((item for item in project.dimension_bindings if item.id == dimension.binding_id), None)
        text_operation, text_id = _dimension_text_operation(project, dimension, binding, new_text)
        operations[text_id] = text_operation
    return list(operations.values()), {
        "old_value": old_value,
        "new_text": new_text,
        "measurement_points": new_points,
        "dimension_line_point": dimension.dimension_line_point,
        "delta": [0.0, 0.0],
        "moving_index": 1,
        "text_id": text_id,
        "arrow_deltas": arrow_deltas,
    }


def _sync_dimension_semantics(project, binding_id: str, plan, semantic) -> None:
    parsed = parse_dimension_text(semantic["new_text"])
    delta = semantic["delta"]
    moving_index = semantic["moving_index"]
    for binding in project.dimension_bindings:
        if binding.id != binding_id:
            continue
        binding.text = semantic["new_text"]
        binding.parsed = parsed
        binding.kind = parsed.kind
        binding.text_id = semantic["text_id"]
        if parsed.kind in {"diameter", "radius"}:
            first, second = semantic["measurement_points"][:2]
            binding.line_x1, binding.line_y1 = first
            binding.line_x2, binding.line_y2 = second
        elif moving_index == 0:
            binding.line_x1 += delta[0]
            binding.line_y1 += delta[1]
        else:
            binding.line_x2 += delta[0]
            binding.line_y2 += delta[1]
        if binding.text_x is not None:
            binding.text_x += delta[0] * 0.5
        if binding.text_y is not None:
            binding.text_y += delta[1] * 0.5

    seen: set[int] = set()
    for collection in (project.mechanical_dimensions, project.mechanical_ir.dimensions):
        for dimension in collection:
            if dimension.binding_id != binding_id or id(dimension) in seen:
                continue
            seen.add(id(dimension))
            dimension.text = semantic["new_text"]
            dimension.parsed = parsed
            dimension.kind = parsed.kind
            dimension.text_id = semantic["text_id"]
            dimension.measurement_points = [list(point) for point in semantic["measurement_points"]]
            dimension.dimension_line_point = (
                list(semantic["dimension_line_point"]) if semantic["dimension_line_point"] is not None else None
            )
            for arrow in dimension.arrowheads:
                arrow_delta = semantic["arrow_deltas"].get(arrow.render_entity_id)
                if arrow_delta:
                    arrow.tip_x += arrow_delta[0]
                    arrow.tip_y += arrow_delta[1]
            dimension.edit_mode = "driving"
            dimension.measured_value = plan.target_value
            dimension.last_edit_source = plan.planner_source
            dimension.validation_status = "passed"
            dimension.evidence = [
                *dimension.evidence,
                f"driving_edit:{semantic['old_value']:.6g}->{plan.target_value:.6g}",
                f"planner:{plan.planner_source}",
            ]


def _validate_drive(project: ProjectState, binding_id: str, target: float) -> MechanicalValidation:
    dimension = next((item for item in project.mechanical_ir.dimensions if item.binding_id == binding_id), None)
    errors: list[str] = []
    checks: list[str] = []
    measured: float | None = None
    if dimension is None:
        errors.append("semantic dimension disappeared")
    elif dimension.kind == "linear" and len(dimension.measurement_points) >= 2:
        first, second = dimension.measurement_points[:2]
        measured = abs(second[0] - first[0]) if dimension.orientation == "horizontal" else abs(second[1] - first[1])
        checks.append("definition points match target")
        if not _moving_geometry_touches_definition(project, dimension):
            errors.append("measured geometry does not reach the driven definition point")
        else:
            checks.append("measured geometry remains bound")
    elif dimension is not None and dimension.kind in {"diameter", "radius"}:
        entities = _entities_by_id(project)
        radial = next(
            (
                entity
                for entity_id in dimension.measured_geometry_ids
                if isinstance((entity := entities.get(entity_id)), CircleEntity | ArcEntity)
            ),
            None,
        )
        if radial is None:
            errors.append("radial geometry is missing")
        else:
            measured = radial.r * 2 if dimension.kind == "diameter" else radial.r
            checks.append("circle/arc radius matches target")
    else:
        errors.append("unsupported or incomplete semantic dimension")

    if measured is None or not math.isclose(measured, target, rel_tol=1e-7, abs_tol=DRIVE_TOLERANCE_MM):
        errors.append(f"measured value {measured!r} does not equal target {target:g}")
    else:
        checks.append("measured value equals target")
    return MechanicalValidation(
        passed=not errors,
        dimension_id=dimension.id if dimension is not None else binding_id,
        target_value=target,
        measured_value=round(measured, 6) if measured is not None else None,
        tolerance=DRIVE_TOLERANCE_MM,
        checks=checks,
        errors=errors,
    )


def _moving_geometry_touches_definition(project, dimension) -> bool:
    if len(dimension.measurement_points) < 2:
        return False
    entities = _entities_by_id(project)
    return all(
        any(_point_to_entity_distance(tuple(point[:2]), entities.get(entity_id)) <= 0.05 for entity_id in dimension.measured_geometry_ids)
        for point in dimension.measurement_points[:2]
    )


def _linear_geometry_changes(entity, moving, fixed, delta, axis_index, old_value):
    if entity is None:
        return None
    threshold = max(0.3, old_value * 0.005)
    moving_coord = moving[axis_index]
    fixed_coord = fixed[axis_index]
    if isinstance(entity, LineEntity):
        coords = [entity.x1 if axis_index == 0 else entity.y1, entity.x2 if axis_index == 0 else entity.y2]
        move_flags = [abs(value - moving_coord) <= threshold for value in coords]
        if not any(move_flags):
            return None
        changes = {}
        for index, should_move in enumerate(move_flags, start=1):
            if should_move:
                if abs(delta[0]) > 1e-12:
                    changes[f"x{index}"] = getattr(entity, f"x{index}") + delta[0]
                if abs(delta[1]) > 1e-12:
                    changes[f"y{index}"] = getattr(entity, f"y{index}") + delta[1]
        return changes
    if isinstance(entity, PolylineEntity):
        points = [list(point) for point in entity.points]
        changed = False
        for point in points:
            if abs(point[axis_index] - moving_coord) <= threshold:
                point[0] += delta[0]
                point[1] += delta[1]
                changed = True
        return {"points": points} if changed else None
    if isinstance(entity, RectangleEntity):
        start = entity.x if axis_index == 0 else entity.y
        size = entity.width if axis_index == 0 else entity.height
        end = start + size
        if abs(end - moving_coord) <= threshold and abs(start - fixed_coord) <= threshold:
            return {"width" if axis_index == 0 else "height": size + (delta[0] if axis_index == 0 else delta[1])}
        if abs(start - moving_coord) <= threshold and abs(end - fixed_coord) <= threshold:
            shift = delta[0] if axis_index == 0 else delta[1]
            return {
                "x" if axis_index == 0 else "y": start + shift,
                "width" if axis_index == 0 else "height": size - shift,
            }
    return None


def _translated_changes(entity: Entity, dx: float, dy: float) -> dict:
    if isinstance(entity, LineEntity):
        changes = {}
        if abs(dx) > 1e-12:
            changes.update({"x1": entity.x1 + dx, "x2": entity.x2 + dx})
        if abs(dy) > 1e-12:
            changes.update({"y1": entity.y1 + dy, "y2": entity.y2 + dy})
        return changes
    if isinstance(entity, PolylineEntity):
        return {"points": [[point[0] + dx, point[1] + dy] for point in entity.points]}
    if isinstance(entity, CircleEntity | ArcEntity):
        changes = {}
        if abs(dx) > 1e-12:
            changes["cx"] = entity.cx + dx
        if abs(dy) > 1e-12:
            changes["cy"] = entity.cy + dy
        return changes
    if isinstance(entity, RectangleEntity | TextEntity):
        changes = {}
        if abs(dx) > 1e-12:
            changes["x"] = entity.x + dx
        if abs(dy) > 1e-12:
            changes["y"] = entity.y + dy
        return changes
    return {}


def _merge_modify(operations, entity_id, changes, dimension, action):
    if not changes:
        return
    existing = operations.get(entity_id)
    if existing is None:
        operations[entity_id] = Operation(
            operation="modify_entity",
            entity_id=entity_id,
            changes=changes,
            reason=f"Mechanical dimension drive {dimension.binding_id}: {action}.",
        )
    else:
        existing.changes.update(changes)


def _ensure_operations_are_editable(project, operations):
    entities = _entities_by_id(project)
    layers = {layer.name: layer for layer in project.ir.layers}
    for operation in operations:
        entity = entities.get(operation.entity_id or "")
        if entity is None:
            raise MechanicalDriveError(f"Entity not found during drive: {operation.entity_id}")
        layer = layers.get(entity.layer)
        if canonical_layer_name(entity.layer) == REFERENCE_TRACE or (layer is not None and layer.locked):
            raise MechanicalDriveError(f"Refusing to modify locked reference entity: {entity.id}")


def _dimension_by_id(project, dimension_id):
    return next(
        (
            dimension
            for dimension in project.mechanical_ir.dimensions
            if dimension.id == dimension_id or dimension.binding_id == dimension_id
        ),
        None,
    )


def _entities_by_id(project):
    return {entity.id: entity for entity in project.ir.entities}


def _radial_direction(points, center):
    if points:
        target = points[-1]
        dx, dy = target[0] - center[0], target[1] - center[1]
        length = math.hypot(dx, dy)
        if length > 1e-9:
            return [dx / length, dy / length]
    return [1.0, 0.0]


def _point_to_entity_distance(point, entity):
    if entity is None:
        return math.inf
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
        return min(_point_to_segment_distance(point, corners[index], corners[(index + 1) % 4]) for index in range(4))
    return math.inf


def _point_to_segment_distance(point, start, end):
    vx, vy = end[0] - start[0], end[1] - start[1]
    length_sq = vx * vx + vy * vy
    if length_sq <= 1e-12:
        return math.dist(point, start)
    t = max(0.0, min(1.0, ((point[0] - start[0]) * vx + (point[1] - start[1]) * vy) / length_sq))
    return math.dist(point, (start[0] + t * vx, start[1] + t * vy))
