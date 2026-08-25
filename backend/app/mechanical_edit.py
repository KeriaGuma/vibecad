from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .cad_layers import DIMENSION
from .dimension_semantics import parse_dimension_text
from .models import (
    DimensionBinding,
    Entity,
    MechanicalDimensionObject,
    Operation,
    ParsedDimensionValue,
    ProjectState,
    TextEntity,
    new_id,
)

EDIT_INTENT_RE = re.compile(r"(改成|改为|改到|变成|修改|更新|设为|设置为|\bto\b)", re.IGNORECASE)
EXPLICIT_DIMENSION_ID_RE = re.compile(r"\b(?:mechanical_dimension_)?dim_binding_\d+\b")
NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


@dataclass(frozen=True)
class MechanicalDimensionEdit:
    handled: bool
    operations: list[Operation]
    reply: str
    binding_id: str | None = None
    text_id: str | None = None
    new_text: str | None = None
    parsed: ParsedDimensionValue | None = None


def plan_mechanical_dimension_edit(message: str, project: ProjectState) -> MechanicalDimensionEdit | None:
    """Plan a chat edit against the mechanical semantic dimension layer.

    This intentionally handles a narrow but important closed loop first:
    "把 49 改成 50" resolves to a MechanicalDimensionObject, updates its bound
    CAD text entity, then syncs the semantic snapshot after operation apply.
    """

    if not EDIT_INTENT_RE.search(message):
        return None
    if not project.mechanical_dimensions:
        return None

    explicit_id = _explicit_dimension_id(message)
    number_values = _numbers_without_explicit_ids(message)
    if not number_values:
        return None

    new_number = number_values[-1]
    old_number = number_values[-2] if len(number_values) >= 2 else None
    if explicit_id is None and old_number is None:
        return None

    target = _find_dimension(project.mechanical_dimensions, explicit_id, old_number)
    if target is None:
        return MechanicalDimensionEdit(
            handled=True,
            operations=[],
            reply="我识别到这是机械尺寸编辑，但没有在机械语义层里找到匹配的尺寸。可以先点 Dims 查看可编辑尺寸，或用“把 49 改成 50”这种带原值的说法。",
        )

    binding = _binding_by_id(project.dimension_bindings, target.binding_id)
    new_text = _replacement_text(target, _format_number(new_number))
    parsed = parse_dimension_text(new_text)
    operation, text_id = _dimension_text_operation(project, target, binding, new_text)

    reply = (
        f"已通过机械语义层命中 {target.binding_id}，把尺寸文字改为 {new_text}。"
        "已同步更新文字实体、MechanicalDrawingIR 和导出的 DXF DIMENSION。"
    )
    return MechanicalDimensionEdit(
        handled=True,
        operations=[operation],
        reply=reply,
        binding_id=target.binding_id,
        text_id=text_id,
        new_text=new_text,
        parsed=parsed,
    )


def sync_mechanical_dimension_edit(project: ProjectState, edit: MechanicalDimensionEdit) -> None:
    if not edit.binding_id or edit.new_text is None or edit.parsed is None:
        return

    for binding in project.dimension_bindings:
        if binding.id != edit.binding_id:
            continue
        binding.text = edit.new_text
        binding.parsed = edit.parsed
        binding.kind = edit.parsed.kind
        if edit.text_id:
            binding.text_id = edit.text_id

    for dimension in project.mechanical_dimensions:
        if dimension.binding_id != edit.binding_id:
            continue
        dimension.text = edit.new_text
        dimension.parsed = edit.parsed
        dimension.kind = edit.parsed.kind
        if edit.text_id:
            dimension.text_id = edit.text_id
        dimension.evidence = [
            *dimension.evidence,
            f"chat_edit: dimension text updated to {edit.new_text}",
        ]

    for dimension in project.mechanical_ir.dimensions:
        if dimension.binding_id != edit.binding_id:
            continue
        dimension.text = edit.new_text
        dimension.parsed = edit.parsed
        dimension.kind = edit.parsed.kind
        if edit.text_id:
            dimension.text_id = edit.text_id
        dimension.evidence = [
            *dimension.evidence,
            f"chat_edit: dimension text updated to {edit.new_text}",
        ]


def _explicit_dimension_id(message: str) -> str | None:
    match = EXPLICIT_DIMENSION_ID_RE.search(message)
    if not match:
        return None
    value = match.group(0)
    if value.startswith("mechanical_dimension_"):
        return value.removeprefix("mechanical_dimension_")
    return value


def _numbers_without_explicit_ids(message: str) -> list[float]:
    stripped = EXPLICIT_DIMENSION_ID_RE.sub(" ", message)
    return [float(match.group(0)) for match in NUMBER_RE.finditer(stripped)]


def _find_dimension(
    dimensions: list[MechanicalDimensionObject],
    explicit_id: str | None,
    old_number: float | None,
) -> MechanicalDimensionObject | None:
    if explicit_id:
        for dimension in dimensions:
            if dimension.id == explicit_id or dimension.binding_id == explicit_id:
                return dimension

    if old_number is None:
        return None

    candidates = [
        dimension
        for dimension in dimensions
        if _dimension_matches_number(dimension, old_number)
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: (item.confidence, len(item.arrowheads)), reverse=True)[0]


def _dimension_matches_number(dimension: MechanicalDimensionObject, value: float) -> bool:
    nominal = dimension.parsed.nominal
    if nominal is not None and math.isclose(float(nominal), value, rel_tol=0.002, abs_tol=0.03):
        return True
    for raw_number in NUMBER_RE.finditer(dimension.text or dimension.parsed.raw_text):
        if math.isclose(float(raw_number.group(0)), value, rel_tol=0.002, abs_tol=0.03):
            return True
    return False


def _binding_by_id(bindings: list[DimensionBinding], binding_id: str) -> DimensionBinding | None:
    for binding in bindings:
        if binding.id == binding_id:
            return binding
    return None


def _replacement_text(dimension: MechanicalDimensionObject, replacement: str) -> str:
    raw = dimension.text or dimension.parsed.raw_text
    if raw and NUMBER_RE.search(raw):
        return NUMBER_RE.sub(replacement, raw, count=1)
    if dimension.kind == "diameter":
        return f"φ{replacement}"
    if dimension.kind == "radius":
        return f"R{replacement}"
    return replacement


def _dimension_text_operation(
    project: ProjectState,
    dimension: MechanicalDimensionObject,
    binding: DimensionBinding | None,
    new_text: str,
) -> tuple[Operation, str]:
    text_id = dimension.text_id or (binding.text_id if binding else None)
    existing = _text_entity(project.ir.entities, text_id)
    if existing is not None:
        metadata = {
            **existing.metadata,
            "mechanical_role": "dimension_text",
            "binding_id": dimension.binding_id,
            "mechanical_dimension_id": dimension.id,
        }
        tags = _merged_tags(existing.tags, ["dimensions", "dimension_text", "mechanical_semantic"])
        return (
            Operation(
                operation="modify_entity",
                entity_id=existing.id,
                changes={"text": new_text, "tags": tags, "metadata": metadata},
                reason=f"Update mechanical dimension {dimension.binding_id} text to {new_text}.",
            ),
            existing.id,
        )

    x, y = _fallback_text_position(dimension, binding)
    new_text_id = new_id("dim_text")
    entity = TextEntity(
        id=new_text_id,
        layer=DIMENSION,
        x=x,
        y=y,
        text=new_text,
        height=2.2,
        group="dimensions",
        tags=["dimensions", "dimension_text", "mechanical_semantic"],
        metadata={
            "mechanical_role": "dimension_text",
            "binding_id": dimension.binding_id,
            "mechanical_dimension_id": dimension.id,
        },
    )
    return (
        Operation(
            operation="add_entity",
            entity=entity,
            reason=f"Create editable text for mechanical dimension {dimension.binding_id}.",
        ),
        new_text_id,
    )


def _text_entity(entities: list[Entity], text_id: str | None) -> TextEntity | None:
    if not text_id:
        return None
    for entity in entities:
        if entity.id == text_id and isinstance(entity, TextEntity):
            return entity
    return None


def _fallback_text_position(
    dimension: MechanicalDimensionObject,
    binding: DimensionBinding | None,
) -> tuple[float, float]:
    if binding and binding.text_x is not None and binding.text_y is not None:
        return binding.text_x, binding.text_y
    return (
        (dimension.arrowheads[0].tip_x + dimension.arrowheads[-1].tip_x) * 0.5 if dimension.arrowheads else 0.0,
        (dimension.arrowheads[0].tip_y + dimension.arrowheads[-1].tip_y) * 0.5 + 2.0 if dimension.arrowheads else 0.0,
    )


def _format_number(value: float) -> str:
    if math.isfinite(value) and value.is_integer():
        return str(int(value))
    return f"{value:g}"


def _merged_tags(existing: list[str], extra: list[str]) -> list[str]:
    tags = list(existing)
    for tag in extra:
        if tag not in tags:
            tags.append(tag)
    return tags
