from __future__ import annotations

from app.models import ArcEntity, CircleEntity, Entity, LineEntity, PolylineEntity, RectangleEntity, TextEntity
from app.templates import spur_gear_drawing_ir


def _text_weight(value: str) -> float:
    return sum(1.0 if ord(char) > 127 else 0.55 for char in value) or 1.0


def _bbox(entity: Entity) -> tuple[float, float, float, float]:
    if isinstance(entity, LineEntity):
        return min(entity.x1, entity.x2), min(entity.y1, entity.y2), max(entity.x1, entity.x2), max(entity.y1, entity.y2)
    if isinstance(entity, PolylineEntity):
        xs = [point[0] for point in entity.points]
        ys = [point[1] for point in entity.points]
        return min(xs), min(ys), max(xs), max(ys)
    if isinstance(entity, CircleEntity):
        return entity.cx - entity.r, entity.cy - entity.r, entity.cx + entity.r, entity.cy + entity.r
    if isinstance(entity, ArcEntity):
        return entity.cx - entity.r, entity.cy - entity.r, entity.cx + entity.r, entity.cy + entity.r
    if isinstance(entity, RectangleEntity):
        return entity.x, entity.y, entity.x + entity.width, entity.y + entity.height
    if isinstance(entity, TextEntity):
        width = _text_weight(entity.text) * entity.height
        height = entity.height
        if abs(entity.rotation) % 180 == 90:
            return entity.x - height, entity.y, entity.x, entity.y + width
        return entity.x, entity.y, entity.x + width, entity.y + height
    raise TypeError(entity)


def _union(entities: list[Entity]) -> tuple[float, float, float, float]:
    boxes = [_bbox(entity) for entity in entities]
    return min(box[0] for box in boxes), min(box[1] for box in boxes), max(box[2] for box in boxes), max(box[3] for box in boxes)


def _overlap_area(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return width * height


def _point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    x, y = point
    for start, end in zip(polygon, polygon[1:] + polygon[:1], strict=True):
        x1, y1 = start
        x2, y2 = end
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        if abs(cross) < 1e-6 and min(x1, x2) - 1e-6 <= x <= max(x1, x2) + 1e-6 and min(y1, y2) - 1e-6 <= y <= max(y1, y2) + 1e-6:
            return True
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
        on_y_span = (y1 > y) != (y2 > y)
        if on_y_span:
            x_at_y = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < x_at_y:
                inside = not inside
        previous = current
    return inside


def test_spur_gear_template_major_modules_do_not_overlap():
    ir = spur_gear_drawing_ir()
    module_names = ["section_view", "circular_view", "parameter_table", "notes", "title_block"]
    boxes = {
        name: _union([entity for entity in ir.entities if entity.group == name])
        for name in module_names
    }

    for index, name in enumerate(module_names):
        for other in module_names[index + 1 :]:
            assert _overlap_area(boxes[name], boxes[other]) == 0.0, (name, other, boxes[name], boxes[other])


def test_spur_gear_template_table_text_stays_inside_linework():
    ir = spur_gear_drawing_ir()
    for group in ["parameter_table", "title_block"]:
        linework = [
            entity
            for entity in ir.entities
            if entity.group == group and isinstance(entity, LineEntity | RectangleEntity)
        ]
        grid_box = _union(linework)
        table_texts = [entity for entity in ir.entities if entity.group == group and isinstance(entity, TextEntity)]
        assert table_texts

        for entity in table_texts:
            box = _bbox(entity)
            assert box[0] >= grid_box[0] - 0.25, (entity.id, entity.text, box, grid_box)
            assert box[1] >= grid_box[1] - 0.25, (entity.id, entity.text, box, grid_box)
            assert box[2] <= grid_box[2] + 0.25, (entity.id, entity.text, box, grid_box)
            assert box[3] <= grid_box[3] + 0.25, (entity.id, entity.text, box, grid_box)


def test_spur_gear_template_surface_symbol_stays_above_title_block():
    ir = spur_gear_drawing_ir()
    title_box = _union([entity for entity in ir.entities if entity.group == "title_block"])
    surface_entities = [entity for entity in ir.entities if "roughness" in entity.tags and entity.id.startswith("surface")]
    assert surface_entities
    assert _overlap_area(title_box, _union(surface_entities)) == 0.0


def test_spur_gear_section_hatches_are_clipped_to_cut_material():
    ir = spur_gear_drawing_ir()
    hatch_lines = [
        entity
        for entity in ir.entities
        if isinstance(entity, LineEntity) and entity.group == "section_view" and "clipped_hatch" in entity.tags
    ]
    assert len(hatch_lines) >= 18

    upper_cut = [
        (99, 184),
        (154, 184),
        (154, 170),
        (164, 170),
        (164, 181),
        (158, 187),
        (140, 187),
        (133, 194),
        (129, 205),
        (125, 208),
        (101, 208),
        (99, 206),
    ]
    lower_cut = [
        (99, 94),
        (125, 94),
        (129, 98),
        (133, 106),
        (140, 113),
        (158, 113),
        (164, 119),
        (164, 130),
        (154, 130),
        (154, 116),
        (99, 116),
    ]

    for entity in hatch_lines:
        for point in [(entity.x1, entity.y1), (entity.x2, entity.y2)]:
            assert _point_in_polygon(point, upper_cut) or _point_in_polygon(point, lower_cut), (entity.id, point)


def test_spur_gear_section_has_explicit_detail_edges():
    ir = spur_gear_drawing_ir()
    detail_ids = {
        "section_bore_top_lip",
        "section_bore_bottom_lip",
        "section_top_land_inner",
        "section_bottom_land_inner",
        "section_left_step_vertical",
        "section_top_root_fillet",
        "section_bottom_root_fillet",
    }
    assert detail_ids.issubset({entity.id for entity in ir.entities})
