from __future__ import annotations

import re

from .cad_ops import make_entity
from .models import CircleEntity, DrawingIR, Operation, TextEntity, new_id

_ENTITY_ID_RE = re.compile(r"\b([a-zA-Z]+_\d+|[a-zA-Z]+_[0-9a-f]{8})\b")


def _numbers(text: str) -> list[float]:
    # Strip entity-id tokens first so their digits (e.g. the "1" in "hole_1")
    # never leak into operands like move distance or new coordinates.
    cleaned = _ENTITY_ID_RE.sub(" ", text)
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", cleaned)]


_CN_NUM = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4}


def _hole_count(message: str, default: int = 2) -> int:
    """Parse an explicit hole count like "两个孔" / "4 holes".

    Only counts digits/words that directly qualify 孔/hole, so dimension
    numbers (e.g. the "1" in "100") no longer leak into the hole count.
    """
    match = re.search(r"(\d+|[一二两三四])\s*个?\s*(?:孔|holes?)", message, re.IGNORECASE)
    if not match:
        return default
    token = match.group(1)
    return int(token) if token.isdigit() else _CN_NUM[token]


def _find_entity_id(message: str, ir: DrawingIR) -> str | None:
    explicit = _ENTITY_ID_RE.search(message)
    if explicit:
        return explicit.group(1)

    lower = message.lower()
    if "左" in message or "left" in lower:
        holes = [e for e in ir.entities if e.type == "circle"]
        if holes:
            return min(holes, key=lambda e: e.cx).id
    if "右" in message or "right" in lower:
        holes = [e for e in ir.entities if e.type == "circle"]
        if holes:
            return max(holes, key=lambda e: e.cx).id
    if "板" in message or "矩形" in message or "plate" in lower or "rectangle" in lower:
        for entity in ir.entities:
            if entity.type == "rectangle":
                return entity.id
    if "孔" in message or "hole" in lower or "圆" in message or "circle" in lower:
        for entity in ir.entities:
            if entity.type == "circle":
                return entity.id
    return ir.entities[0].id if ir.entities else None


def plan_operations(message: str, ir: DrawingIR) -> tuple[list[Operation], str]:
    """A deterministic MVP parser.

    The production slot is intentionally obvious: replace this function with
    LLM structured output that returns the same Operation schema.
    """
    lower = message.lower()
    nums = _numbers(message)
    ops: list[Operation] = []

    if "齿轮" in message or "LJT01.01" in message or "gear" in lower:
        ops.append(
            Operation(
                operation="create_spur_gear_drawing",
                reason="Create a cylindrical spur gear mechanical drawing template.",
            )
        )
        return ops, "我会生成一张圆柱直齿轮机械零件图模板，包含剖视轮廓、圆视图、尺寸文字、参数表、技术要求和标题栏。"

    if any(token in message for token in ["新建", "创建", "生成"]) or "create" in lower:
        width = nums[0] if len(nums) >= 1 else 100
        height = nums[1] if len(nums) >= 2 else 60
        hole_diameter = nums[2] if len(nums) >= 3 else 8
        hole_count = _hole_count(message)
        ops.append(
            Operation(
                operation="create_plate",
                changes={
                    "width": width,
                    "height": height,
                    "hole_diameter": hole_diameter,
                    "hole_count": hole_count,
                },
                reason="Create a baseline plate drawing from prompt.",
            )
        )
        return ops, f"我会生成一个 {width:g} x {height:g} {ir.units} 的板件，并放置 {hole_count} 个孔。"

    target_id = _find_entity_id(message, ir)

    if any(token in message for token in ["删除", "去掉", "移除"]) or "delete" in lower or "remove" in lower:
        if target_id:
            ops.append(Operation(operation="delete_entity", entity_id=target_id, reason="Delete requested entity."))
            return ops, f"我会删除 `{target_id}`。"

    if "图层" in message or "layer" in lower:
        match = re.search(r"(?:图层|layer)\s*[:：]?\s*([A-Za-z0-9_-]+)", message)
        layer = match.group(1) if match else "notes"
        if target_id:
            ops.append(Operation(operation="set_layer", entity_id=target_id, layer=layer, reason="Move entity to layer."))
            return ops, f"我会把 `{target_id}` 放到 `{layer}` 图层。"

    if any(token in message for token in ["移动", "右移", "左移", "上移", "下移"]) or "move" in lower:
        amount = nums[0] if nums else 10
        dx = amount if ("右" in message or "right" in lower) else -amount if ("左" in message or "left" in lower) else 0
        dy = amount if ("上" in message or "up" in lower) else -amount if ("下" in message or "down" in lower) else 0
        if "move" in lower and len(nums) >= 2:
            dx, dy = nums[0], nums[1]
        if target_id:
            ops.append(Operation(operation="move_entity", entity_id=target_id, dx=dx, dy=dy, reason="Move entity."))
            return ops, f"我会把 `{target_id}` 移动 dx={dx:g}, dy={dy:g}。"

    if any(token in message for token in ["直径", "diameter"]):
        if target_id and nums:
            ops.append(Operation(operation="modify_entity", entity_id=target_id, changes={"r": nums[-1] / 2}, reason="Modify circle diameter."))
            return ops, f"我会把 `{target_id}` 的直径改成 {nums[-1]:g}。"

    if any(token in message for token in ["半径", "radius"]):
        if target_id and nums:
            ops.append(Operation(operation="modify_entity", entity_id=target_id, changes={"r": nums[-1]}, reason="Modify circle radius."))
            return ops, f"我会把 `{target_id}` 的半径改成 {nums[-1]:g}。"

    if any(token in message for token in ["宽", "width"]):
        if target_id and nums:
            ops.append(Operation(operation="modify_entity", entity_id=target_id, changes={"width": nums[-1]}, reason="Modify width."))
            return ops, f"我会把 `{target_id}` 的宽度改成 {nums[-1]:g}。"

    if any(token in message for token in ["高", "height"]):
        if target_id and nums:
            ops.append(Operation(operation="modify_entity", entity_id=target_id, changes={"height": nums[-1]}, reason="Modify height."))
            return ops, f"我会把 `{target_id}` 的高度改成 {nums[-1]:g}。"

    if any(token in message for token in ["加文字", "文字", "text"]):
        text_match = re.search(r"[\"“](.+?)[\"”]", message)
        text = text_match.group(1) if text_match else message
        x = nums[0] if len(nums) >= 1 else 0
        y = nums[1] if len(nums) >= 2 else 75
        entity = TextEntity(id=new_id("text"), layer="notes", x=x, y=y, text=text, height=4)
        ops.append(Operation(operation="add_entity", entity=entity, reason="Add text note."))
        return ops, "我会添加一段文字标注。"

    if any(token in message for token in ["加孔", "添加孔", "圆孔", "circle", "hole"]):
        cx = nums[0] if len(nums) >= 1 else 50
        cy = nums[1] if len(nums) >= 2 else 30
        diameter = nums[2] if len(nums) >= 3 else 8
        entity = CircleEntity(id=new_id("hole"), layer="holes", cx=cx, cy=cy, r=diameter / 2, label="added hole")
        ops.append(Operation(operation="add_entity", entity=entity, reason="Add circle hole."))
        return ops, f"我会在 ({cx:g}, {cy:g}) 添加直径 {diameter:g} 的圆孔。"

    # Power-user escape hatch for the demo: allow a compact entity JSON-ish body
    if "add line" in lower and len(nums) >= 4:
        entity = make_entity({"type": "line", "id": new_id("line"), "layer": "outline", "x1": nums[0], "y1": nums[1], "x2": nums[2], "y2": nums[3]})
        ops.append(Operation(operation="add_entity", entity=entity, reason="Add line."))
        return ops, "我会添加一条线。"

    return [], "我还没能把这句话稳定解析成 CAD 操作。你可以试试：`把左边孔直径改成 10`、`右移 20`、`添加孔 50 30 8`。"
