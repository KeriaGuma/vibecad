"""Edge-case coverage for operation application in ``app.cad_ops``."""
from __future__ import annotations

import pytest

from app.cad_ops import apply_operation, apply_operations, make_entity
from app.models import (
    ArcEntity,
    CircleEntity,
    DrawingIR,
    LineEntity,
    Operation,
    PolylineEntity,
    RectangleEntity,
    TextEntity,
    default_ir,
)


def _plate_op(**changes) -> Operation:
    return Operation(operation="create_plate", changes=changes)


# --- create_plate hole layouts ----------------------------------------------

def test_create_plate_one_hole_is_centered():
    ir, _ = apply_operation(DrawingIR(), _plate_op(width=100, height=60, hole_count=1))
    holes = [e for e in ir.entities if e.type == "circle"]
    assert len(holes) == 1
    assert (holes[0].cx, holes[0].cy) == (50, 30)


def test_create_plate_two_holes_horizontal():
    ir, _ = apply_operation(DrawingIR(), _plate_op(width=100, height=60, hole_count=2, margin=10))
    holes = sorted((e for e in ir.entities if e.type == "circle"), key=lambda e: e.cx)
    assert [(h.cx, h.cy) for h in holes] == [(10, 30), (90, 30)]


def test_create_plate_four_holes_at_corners():
    ir, _ = apply_operation(DrawingIR(), _plate_op(width=100, height=60, hole_count=4, margin=10))
    holes = [e for e in ir.entities if e.type == "circle"]
    assert len(holes) == 4
    corners = {(h.cx, h.cy) for h in holes}
    assert corners == {(10, 10), (90, 10), (90, 50), (10, 50)}


def test_create_plate_replaces_existing_entities():
    ir, _ = apply_operation(default_ir(), _plate_op(width=40, height=40, hole_count=1))
    # the default note_1/hole_1/hole_2 are gone; only the new plate + hole remain
    assert {e.id for e in ir.entities} == {"plate_1", "hole_1"}


# --- create_spur_gear_drawing -----------------------------------------------

def test_create_spur_gear_drawing_loads_template():
    op = Operation(operation="create_spur_gear_drawing")
    ir, diffs = apply_operation(DrawingIR(), op)
    ids = {e.id for e in ir.entities}
    assert "section_profile" in ids
    assert "keyway" in ids
    assert "title_block_border" in ids
    assert len(ir.entities) > 100
    assert diffs[0].path == "drawing"


# --- add / modify / delete ---------------------------------------------------

def test_add_entity_appends_and_creates_layer():
    op = Operation(
        operation="add_entity",
        entity=CircleEntity(id="c1", layer="custom", cx=1, cy=2, r=3),
    )
    ir, _ = apply_operation(DrawingIR(), op)
    assert ir.entities[-1].id == "c1"
    assert any(layer.name == "custom" for layer in ir.layers)


def test_add_entity_without_entity_raises():
    with pytest.raises(ValueError, match="requires entity"):
        apply_operation(DrawingIR(), Operation(operation="add_entity"))


def test_modify_entity_merges_changes():
    op = Operation(operation="modify_entity", entity_id="hole_1", changes={"r": 9})
    ir, diffs = apply_operation(default_ir(), op)
    hole = next(e for e in ir.entities if e.id == "hole_1")
    assert hole.r == 9
    assert diffs[0].before == 4 and diffs[0].after == 9


def test_modify_missing_entity_raises():
    op = Operation(operation="modify_entity", entity_id="ghost", changes={"r": 1})
    with pytest.raises(ValueError, match="Entity not found: ghost"):
        apply_operation(default_ir(), op)


def test_delete_missing_entity_raises():
    op = Operation(operation="delete_entity", entity_id="ghost")
    with pytest.raises(ValueError, match="Entity not found: ghost"):
        apply_operation(default_ir(), op)


def test_set_layer_without_layer_raises():
    op = Operation(operation="set_layer", entity_id="hole_1")
    with pytest.raises(ValueError, match="requires layer"):
        apply_operation(default_ir(), op)


# --- move applies to every geometry type ------------------------------------

def _ir_with(entity) -> DrawingIR:
    return DrawingIR(entities=[entity])


def test_move_line():
    ir = _ir_with(LineEntity(id="l", x1=0, y1=0, x2=10, y2=0))
    out, _ = apply_operation(ir, Operation(operation="move_entity", entity_id="l", dx=5, dy=2))
    line = out.entities[0]
    assert (line.x1, line.y1, line.x2, line.y2) == (5, 2, 15, 2)


def test_move_polyline():
    ir = _ir_with(PolylineEntity(id="p", points=[[0, 0], [10, 10]]))
    out, _ = apply_operation(ir, Operation(operation="move_entity", entity_id="p", dx=1, dy=-1))
    assert out.entities[0].points == [[1, -1], [11, 9]]


def test_move_arc():
    ir = _ir_with(ArcEntity(id="a", cx=0, cy=0, r=5, start_angle=0, end_angle=90))
    out, _ = apply_operation(ir, Operation(operation="move_entity", entity_id="a", dx=3, dy=4))
    assert (out.entities[0].cx, out.entities[0].cy) == (3, 4)


def test_move_circle():
    ir = _ir_with(CircleEntity(id="c", cx=5, cy=5, r=2))
    out, _ = apply_operation(ir, Operation(operation="move_entity", entity_id="c", dx=-5, dy=10))
    assert (out.entities[0].cx, out.entities[0].cy) == (0, 15)


def test_move_rectangle():
    ir = _ir_with(RectangleEntity(id="r", x=0, y=0, width=4, height=2))
    out, _ = apply_operation(ir, Operation(operation="move_entity", entity_id="r", dx=3, dy=1))
    assert (out.entities[0].x, out.entities[0].y) == (3, 1)


def test_move_text():
    ir = _ir_with(TextEntity(id="t", x=0, y=0, text="hi"))
    out, _ = apply_operation(ir, Operation(operation="move_entity", entity_id="t", dx=2, dy=3))
    assert (out.entities[0].x, out.entities[0].y) == (2, 3)


def test_set_layer_moves_entity_and_creates_layer():
    ir, diffs = apply_operation(
        default_ir(), Operation(operation="set_layer", entity_id="hole_1", layer="brand_new")
    )
    hole = next(e for e in ir.entities if e.id == "hole_1")
    assert hole.layer == "brand_new"
    assert any(layer.name == "brand_new" for layer in ir.layers)
    assert diffs[0].before == "holes" and diffs[0].after == "brand_new"


# --- immutability + chaining -------------------------------------------------

def test_apply_operation_does_not_mutate_input_ir():
    ir = default_ir()
    before_ids = [e.id for e in ir.entities]
    apply_operation(ir, Operation(operation="delete_entity", entity_id="hole_1"))
    assert [e.id for e in ir.entities] == before_ids  # original untouched


def test_apply_operations_chains_and_accumulates_diffs():
    ir = default_ir()
    ops = [
        Operation(operation="modify_entity", entity_id="hole_1", changes={"r": 5}),
        Operation(operation="delete_entity", entity_id="hole_2"),
    ]
    out, diffs = apply_operations(ir, ops)
    ids = {e.id for e in out.entities}
    assert "hole_2" not in ids
    assert next(e for e in out.entities if e.id == "hole_1").r == 5
    assert len(diffs) == 2


# --- make_entity -------------------------------------------------------------

def test_make_entity_unsupported_type_raises():
    with pytest.raises(ValueError, match="Unsupported entity type"):
        make_entity({"type": "spline"})


def test_make_entity_autogenerates_id():
    entity = make_entity({"type": "rectangle", "x": 0, "y": 0, "width": 1, "height": 1})
    assert entity.id.startswith("rect_")
