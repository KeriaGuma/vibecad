from __future__ import annotations

from copy import deepcopy
from typing import Any

from .models import (
    ArcEntity,
    CircleEntity,
    DiffItem,
    DrawingIR,
    Entity,
    Layer,
    LineEntity,
    Operation,
    PolylineEntity,
    RectangleEntity,
    TextEntity,
    new_id,
)

ENTITY_PREFIX = {
    "line": "line",
    "polyline": "pline",
    "circle": "circle",
    "arc": "arc",
    "rectangle": "rect",
    "text": "text",
}

DIFF_ID_LIMIT = 80


def ensure_layer(ir: DrawingIR, name: str, color: str = "white") -> None:
    if not any(layer.name == name for layer in ir.layers):
        ir.layers.append(Layer(name=name, color=color))


def entity_to_dict(entity: Entity) -> dict[str, Any]:
    return entity.model_dump()


def ids_diff_payload(ids: list[str]) -> list[str] | dict[str, Any]:
    if len(ids) <= DIFF_ID_LIMIT:
        return ids
    return {
        "count": len(ids),
        "sample": ids[:DIFF_ID_LIMIT],
        "truncated": len(ids) - DIFF_ID_LIMIT,
    }


def make_entity(payload: dict[str, Any]) -> Entity:
    entity_type = payload.get("type")
    if "id" not in payload or not payload["id"]:
        payload["id"] = new_id(ENTITY_PREFIX.get(entity_type, "entity"))
    if entity_type == "line":
        return LineEntity(**payload)
    if entity_type == "polyline":
        return PolylineEntity(**payload)
    if entity_type == "circle":
        return CircleEntity(**payload)
    if entity_type == "arc":
        return ArcEntity(**payload)
    if entity_type == "rectangle":
        return RectangleEntity(**payload)
    if entity_type == "text":
        return TextEntity(**payload)
    raise ValueError(f"Unsupported entity type: {entity_type}")


def apply_operation(ir: DrawingIR, operation: Operation) -> tuple[DrawingIR, list[DiffItem]]:
    next_ir = deepcopy(ir)
    diffs: list[DiffItem] = []

    if operation.operation == "create_plate":
        width = float(operation.changes.get("width", 100))
        height = float(operation.changes.get("height", 60))
        hole_diameter = float(operation.changes.get("hole_diameter", 8))
        hole_count = int(operation.changes.get("hole_count", 2))
        margin = float(operation.changes.get("margin", max(min(width, height) * 0.25, 10)))

        next_ir.entities = [
            RectangleEntity(
                id="plate_1",
                layer="outline",
                x=0,
                y=0,
                width=width,
                height=height,
                label="base plate",
            )
        ]
        if hole_count <= 1:
            hole_positions = [(width / 2, height / 2)]
        elif hole_count == 2:
            hole_positions = [(margin, height / 2), (width - margin, height / 2)]
        else:
            hole_positions = [
                (margin, margin),
                (width - margin, margin),
                (width - margin, height - margin),
                (margin, height - margin),
            ]
        for idx, (cx, cy) in enumerate(hole_positions, start=1):
            next_ir.entities.append(
                CircleEntity(
                    id=f"hole_{idx}",
                    layer="holes",
                    cx=cx,
                    cy=cy,
                    r=hole_diameter / 2,
                    label=f"hole {idx}",
                )
            )
        diffs.append(
            DiffItem(
                path="entities",
                before=ids_diff_payload([e.id for e in ir.entities]),
                after=ids_diff_payload([e.id for e in next_ir.entities]),
            )
        )
        return next_ir, diffs

    if operation.operation == "create_spur_gear_drawing":
        from .templates import spur_gear_drawing_ir

        template_ir = spur_gear_drawing_ir()
        diffs.append(
            DiffItem(
                path="drawing",
                before=ids_diff_payload([e.id for e in ir.entities]),
                after=ids_diff_payload([e.id for e in template_ir.entities]),
            )
        )
        return template_ir, diffs

    if operation.operation == "add_entity":
        if operation.entity is None:
            raise ValueError("add_entity requires entity")
        ensure_layer(next_ir, operation.entity.layer)
        next_ir.entities.append(operation.entity)
        diffs.append(DiffItem(path=f"entities.{operation.entity.id}", before=None, after=entity_to_dict(operation.entity)))
        return next_ir, diffs

    index = None
    target = None
    if operation.entity_id:
        for i, entity in enumerate(next_ir.entities):
            if entity.id == operation.entity_id:
                index = i
                target = entity
                break
    if target is None:
        raise ValueError(f"Entity not found: {operation.entity_id}")

    if operation.operation == "modify_entity":
        before = entity_to_dict(target)
        payload = {**before, **operation.changes}
        updated = make_entity(payload)
        next_ir.entities[index] = updated
        for key, after_value in operation.changes.items():
            diffs.append(DiffItem(path=f"{target.id}.{key}", before=before.get(key), after=after_value))
        return next_ir, diffs

    if operation.operation == "delete_entity":
        before = entity_to_dict(target)
        del next_ir.entities[index]
        diffs.append(DiffItem(path=f"entities.{target.id}", before=before, after=None))
        return next_ir, diffs

    if operation.operation == "move_entity":
        before = entity_to_dict(target)
        payload = deepcopy(before)
        if target.type == "line":
            payload["x1"] += operation.dx
            payload["x2"] += operation.dx
            payload["y1"] += operation.dy
            payload["y2"] += operation.dy
        elif target.type == "polyline":
            payload["points"] = [[x + operation.dx, y + operation.dy] for x, y in payload["points"]]
        elif target.type == "circle":
            payload["cx"] += operation.dx
            payload["cy"] += operation.dy
        elif target.type == "arc":
            payload["cx"] += operation.dx
            payload["cy"] += operation.dy
        elif target.type in {"rectangle", "text"}:
            payload["x"] += operation.dx
            payload["y"] += operation.dy
        updated = make_entity(payload)
        next_ir.entities[index] = updated
        diffs.append(DiffItem(path=f"{target.id}.position", before=before, after=entity_to_dict(updated)))
        return next_ir, diffs

    if operation.operation == "set_layer":
        if not operation.layer:
            raise ValueError("set_layer requires layer")
        before_layer = target.layer
        ensure_layer(next_ir, operation.layer)
        payload = entity_to_dict(target)
        payload["layer"] = operation.layer
        updated = make_entity(payload)
        next_ir.entities[index] = updated
        diffs.append(DiffItem(path=f"{target.id}.layer", before=before_layer, after=operation.layer))
        return next_ir, diffs

    raise ValueError(f"Unsupported operation: {operation.operation}")  # pragma: no cover - guarded by Operation literal


def apply_operations(ir: DrawingIR, operations: list[Operation]) -> tuple[DrawingIR, list[DiffItem]]:
    next_ir = ir
    all_diffs: list[DiffItem] = []
    for operation in operations:
        next_ir, diffs = apply_operation(next_ir, operation)
        all_diffs.extend(diffs)
    return next_ir, all_diffs
