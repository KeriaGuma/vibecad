from __future__ import annotations

from dataclasses import dataclass

from .models import (
    ArcEntity,
    CircleEntity,
    DrawingIR,
    Entity,
    Layer,
    LineEntity,
    PolylineEntity,
    RectangleEntity,
    TextEntity,
)

SEMANTIC_IMPORT_TAG = "semantic_import"


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(self.width, 0.0) * max(self.height, 0.0)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) * 0.5, (self.y1 + self.y2) * 0.5)

    def overlap(self, other: "Box") -> float:
        width = max(0.0, min(self.x2, other.x2) - max(self.x1, other.x1))
        height = max(0.0, min(self.y2, other.y2) - max(self.y1, other.y1))
        return width * height

    def contains_point(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


def annotate_vector_semantics(ir: DrawingIR) -> DrawingIR:
    """Add first-pass semantic groups to raw vector-PDF geometry.

    This is deliberately geometry-first. Many CAD PDFs keep small Chinese table
    text as vector strokes instead of extractable text, so the baseline
    classifier uses frame-relative regions and line-density cues. OCR can later
    replace the text-path proxies without changing the external API.
    """
    if not ir.entities:
        return ir

    frame = _drawing_frame(ir)
    regions = _region_boxes(frame)
    for entity in ir.entities:
        box = _bbox(entity)
        if _is_sheet_frame(entity, frame, box):
            _assign(entity, "sheet", ["sheet"])
            entity.layer = "sheet"
            continue

        group = _classify_region(box, regions)
        if group is None:
            continue

        _assign(entity, group, [group])
        _set_semantic_layer(entity, group)
        _add_detail_tags(entity, group, frame)

    _ensure_semantic_layers(ir)
    if not any("Vector semantic groups" in note for note in ir.notes):
        ir.notes.append("Vector semantic groups: title_block, parameter_table, section_view, circular_view, dimensions.")
    return ir


def _bbox(entity: Entity) -> Box:
    if isinstance(entity, LineEntity):
        return Box(min(entity.x1, entity.x2), min(entity.y1, entity.y2), max(entity.x1, entity.x2), max(entity.y1, entity.y2))
    if isinstance(entity, PolylineEntity):
        xs = [point[0] for point in entity.points]
        ys = [point[1] for point in entity.points]
        return Box(min(xs), min(ys), max(xs), max(ys))
    if isinstance(entity, CircleEntity):
        return Box(entity.cx - entity.r, entity.cy - entity.r, entity.cx + entity.r, entity.cy + entity.r)
    if isinstance(entity, ArcEntity):
        return Box(entity.cx - entity.r, entity.cy - entity.r, entity.cx + entity.r, entity.cy + entity.r)
    if isinstance(entity, RectangleEntity):
        return Box(entity.x, entity.y, entity.x + entity.width, entity.y + entity.height)
    if isinstance(entity, TextEntity):
        width = max(len(entity.text) * entity.height * 0.62, entity.height)
        if abs(entity.rotation) % 180 == 90:
            return Box(entity.x - entity.height, entity.y, entity.x, entity.y + width)
        return Box(entity.x, entity.y, entity.x + width, entity.y + entity.height)
    raise TypeError(entity)


def _drawing_frame(ir: DrawingIR) -> Box:
    rectangles = [entity for entity in ir.entities if isinstance(entity, RectangleEntity)]
    if rectangles:
        largest = max(rectangles, key=lambda entity: entity.width * entity.height)
        if largest.width * largest.height > 1.0:
            return Box(largest.x, largest.y, largest.x + largest.width, largest.y + largest.height)

    boxes = [_bbox(entity) for entity in ir.entities]
    return Box(min(box.x1 for box in boxes), min(box.y1 for box in boxes), max(box.x2 for box in boxes), max(box.y2 for box in boxes))


def _region_boxes(frame: Box) -> dict[str, Box]:
    def rel(x1: float, y1: float, x2: float, y2: float) -> Box:
        return Box(
            frame.x1 + frame.width * x1,
            frame.y1 + frame.height * y1,
            frame.x1 + frame.width * x2,
            frame.y1 + frame.height * y2,
        )

    return {
        "title_block": rel(0.48, 0.00, 0.98, 0.26),
        "parameter_table": rel(0.55, 0.45, 0.98, 0.98),
        "section_view": rel(0.04, 0.16, 0.44, 0.86),
        "circular_view": rel(0.39, 0.30, 0.69, 0.80),
        "dimensions": rel(0.02, 0.12, 0.72, 0.90),
    }


def _is_sheet_frame(entity: Entity, frame: Box, box: Box) -> bool:
    if not isinstance(entity, RectangleEntity):
        return False
    return box.area >= frame.area * 0.8


def _classify_region(box: Box, regions: dict[str, Box]) -> str | None:
    cx, cy = box.center
    for name in ("title_block", "parameter_table"):
        region = regions[name]
        if region.contains_point(cx, cy) or box.overlap(region) / max(box.area, 1e-6) >= 0.45:
            return name

    scored = []
    for name in ("section_view", "circular_view"):
        region = regions[name]
        overlap_ratio = box.overlap(region) / max(box.area, 1e-6)
        center_hit = region.contains_point(cx, cy)
        score = overlap_ratio + (0.35 if center_hit else 0.0)
        scored.append((score, name))
    score, name = max(scored)
    if score >= 0.35:
        return name

    dimension_region = regions["dimensions"]
    if dimension_region.contains_point(cx, cy) and box.area < dimension_region.area * 0.08:
        return "dimensions"
    return None


def _assign(entity: Entity, group: str, tags: list[str]) -> None:
    entity.group = group
    for tag in [SEMANTIC_IMPORT_TAG, *tags]:
        if tag not in entity.tags:
            entity.tags.append(tag)


def _set_semantic_layer(entity: Entity, group: str) -> None:
    if isinstance(entity, TextEntity):
        entity.layer = "text"
    elif group in {"title_block", "parameter_table"}:
        entity.layer = "table"


def _add_detail_tags(entity: Entity, group: str, frame: Box) -> None:
    if group in {"section_view", "circular_view", "dimensions"} and "dimensions" not in entity.tags:
        entity.tags.append("dimensions")

    if not isinstance(entity, LineEntity):
        return

    dx = entity.x2 - entity.x1
    dy = entity.y2 - entity.y1
    length = (dx * dx + dy * dy) ** 0.5
    if length < 0.2:
        return

    abs_dx = abs(dx)
    abs_dy = abs(dy)
    diagonal = 0.65 <= abs_dy / max(abs_dx, 1e-6) <= 1.55
    axis_aligned = min(abs_dx, abs_dy) <= max(abs_dx, abs_dy) * 0.12

    if group == "section_view" and diagonal and 1.0 <= length <= frame.width * 0.18:
        entity.layer = "hatch"
        for tag in ("hatch", "cut_hatch"):
            if tag not in entity.tags:
                entity.tags.append(tag)
        return

    if group in {"section_view", "circular_view"} and axis_aligned and length >= frame.width * 0.05:
        entity.layer = "centerline"
        if "centerline" not in entity.tags:
            entity.tags.append("centerline")


def _ensure_semantic_layers(ir: DrawingIR) -> None:
    colors = {
        "sheet": "gray",
        "geometry": "white",
        "centerline": "white",
        "dimensions": "white",
        "hatch": "white",
        "table": "white",
        "text": "white",
    }
    existing = {layer.name for layer in ir.layers}
    for name, color in colors.items():
        if name not in existing:
            ir.layers.append(Layer(name=name, color=color))
