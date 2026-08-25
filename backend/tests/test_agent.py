"""Edge-case coverage for the deterministic NL parser in ``app.agent``."""
from __future__ import annotations

import pytest

from app.agent import _hole_count, plan_operations
from app.models import default_ir

# --- _hole_count: the regression guard for the "100 -> 1 hole" bug ----------

@pytest.mark.parametrize(
    "message, expected",
    [
        ("创建 100 60 8 两个孔", 2),   # the "1" in "100" must NOT leak in
        ("创建 100 60 8", 2),          # no count word -> default 2
        ("创建 80 60 8 四个孔", 4),
        ("创建 50 30 6 一个孔", 1),
        ("创建 50 30 6 三个孔", 3),
        ("create 100 60 with 4 holes", 4),
        ("create plate with 1 hole", 1),
        ("创建 200 100 10", 2),        # "1"/"2"/"0" dimensions don't leak
    ],
)
def test_hole_count(message, expected):
    assert _hole_count(message) == expected


def test_hole_count_custom_default():
    assert _hole_count("创建一块板", default=2) == 2
    assert _hole_count("创建一块板", default=0) == 0


# --- routing / operation selection ------------------------------------------

def test_create_plate_routes_with_dimensions():
    ops, reply = plan_operations("创建 120 80 10 两个孔", default_ir())
    assert len(ops) == 1
    assert ops[0].operation == "create_plate"
    assert ops[0].changes == {
        "width": 120,
        "height": 80,
        "hole_diameter": 10,
        "hole_count": 2,
    }
    assert "120" in reply


@pytest.mark.parametrize("message", ["创建齿轮零件图", "draw the gear", "LJT01.01"])
def test_gear_keyword_takes_priority(message):
    # gear routing must win even when "创建/create" is also present.
    ops, _ = plan_operations(message, default_ir())
    assert len(ops) == 1
    assert ops[0].operation == "create_spur_gear_drawing"


def test_modify_diameter_targets_left_hole():
    ir = default_ir()  # hole_1 cx=25 (left), hole_2 cx=75 (right)
    ops, _ = plan_operations("把左边孔直径改成 10", ir)
    assert ops[0].operation == "modify_entity"
    assert ops[0].entity_id == "hole_1"
    assert ops[0].changes == {"r": 5.0}  # diameter 10 -> radius 5


def test_modify_radius_targets_right_hole():
    ops, _ = plan_operations("把右边孔半径改成 7", default_ir())
    assert ops[0].entity_id == "hole_2"
    assert ops[0].changes == {"r": 7.0}


def test_modify_width_targets_plate():
    ops, _ = plan_operations("把 plate_1 宽度改成 150", default_ir())
    assert ops[0].operation == "modify_entity"
    assert ops[0].entity_id == "plate_1"
    assert ops[0].changes == {"width": 150.0}


def test_modify_height_targets_plate():
    ops, _ = plan_operations("把板的高度改成 75", default_ir())
    assert ops[0].operation == "modify_entity"
    assert ops[0].entity_id == "plate_1"   # "板" keyword -> rectangle
    assert ops[0].changes == {"height": 75.0}


def test_add_line_escape_hatch():
    ops, _ = plan_operations("add line 0 0 50 20", default_ir())
    assert ops[0].operation == "add_entity"
    assert ops[0].entity.type == "line"
    assert (ops[0].entity.x1, ops[0].entity.y1, ops[0].entity.x2, ops[0].entity.y2) == (0, 0, 50, 20)


def test_explicit_entity_id_beats_positional_keyword():
    ops, _ = plan_operations("把 hole_2 直径改成 9", default_ir())
    assert ops[0].entity_id == "hole_2"


@pytest.mark.parametrize(
    "message, dx, dy",
    [
        ("右移 20", 20, 0),
        ("左移 5", -5, 0),
        ("上移 3", 0, 3),
        ("下移 8", 0, -8),
    ],
)
def test_move_directions(message, dx, dy):
    ops, _ = plan_operations(f"hole_1 {message}", default_ir())
    assert ops[0].operation == "move_entity"
    assert (ops[0].dx, ops[0].dy) == (dx, dy)


def test_move_english_two_numbers():
    ops, _ = plan_operations("move hole_1 12 -4", default_ir())
    assert (ops[0].dx, ops[0].dy) == (12, -4)


def test_delete_routes():
    ops, _ = plan_operations("删除 hole_2", default_ir())
    assert ops[0].operation == "delete_entity"
    assert ops[0].entity_id == "hole_2"


def test_set_layer_parses_target_layer():
    ops, _ = plan_operations("把 hole_1 放到 layer: notes", default_ir())
    assert ops[0].operation == "set_layer"
    assert ops[0].layer == "notes"


def test_add_hole_with_coordinates():
    ops, _ = plan_operations("添加孔 50 30 6", default_ir())
    assert ops[0].operation == "add_entity"
    assert ops[0].entity.type == "circle"
    assert (ops[0].entity.cx, ops[0].entity.cy, ops[0].entity.r) == (50, 30, 3)


def test_add_text_extracts_quoted_value():
    ops, _ = plan_operations('加文字 "REV A" 0 75', default_ir())
    assert ops[0].operation == "add_entity"
    assert ops[0].entity.type == "text"
    assert ops[0].entity.text == "REV A"


# --- graceful failure --------------------------------------------------------

def test_unparseable_returns_no_ops_and_a_hint():
    ops, reply = plan_operations("今天天气不错", default_ir())
    assert ops == []
    assert reply  # a non-empty fallback hint, not a crash


def test_targeting_on_empty_drawing_does_not_crash():
    from app.models import DrawingIR

    ops, reply = plan_operations("把左边孔直径改成 10", DrawingIR())
    # no entities to target -> no ops, but must not raise
    assert ops == []
