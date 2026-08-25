from __future__ import annotations

from .models import ArcEntity, CircleEntity, DrawingIR, Layer, LineEntity, PolylineEntity, RectangleEntity, TextEntity


def spur_gear_drawing_ir() -> DrawingIR:
    """A compact vector template for the supplied cylindrical spur gear drawing.

    This is not OCR reconstruction yet. It is a deterministic target drawing
    that exercises the CAD primitives needed for this class of 2D mechanical
    drawings: sheet frame, section/profile geometry, centerlines, dimensions,
    tolerance table, notes, hatching, and title block.
    """
    entities = []

    def entity_meta(group: str | None = None, tags: list[str] | None = None) -> dict[str, object]:
        tag_values = list(tags or [])
        if group and group not in tag_values:
            tag_values.append(group)
        return {"group": group, "tags": tag_values}

    def line(
        id_: str,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        layer: str = "geometry",
        group: str | None = None,
        tags: list[str] | None = None,
    ) -> None:
        entities.append(LineEntity(id=id_, layer=layer, x1=x1, y1=y1, x2=x2, y2=y2, **entity_meta(group, tags)))

    def rect(
        id_: str,
        x: float,
        y: float,
        w: float,
        h: float,
        layer: str = "geometry",
        group: str | None = None,
        tags: list[str] | None = None,
    ) -> None:
        entities.append(
            RectangleEntity(id=id_, layer=layer, x=x, y=y, width=w, height=h, **entity_meta(group, tags))
        )

    def text(
        id_: str,
        x: float,
        y: float,
        value: str,
        layer: str = "text",
        height: float = 4,
        rotation: float = 0,
        group: str | None = None,
        tags: list[str] | None = None,
    ) -> None:
        entities.append(
            TextEntity(id=id_, layer=layer, x=x, y=y, text=value, height=height, rotation=rotation, **entity_meta(group, tags))
        )

    def text_weight(value: str) -> float:
        return sum(1.0 if ord(char) > 127 else 0.55 for char in value) or 1.0

    def cell_text(
        id_: str,
        x: float,
        y: float,
        w: float,
        h: float,
        value: str,
        layer: str = "table",
        group: str | None = None,
        tags: list[str] | None = None,
        pad: float = 1.0,
        max_height: float = 3.2,
    ) -> None:
        usable_width = max(w - pad * 2, 1.0)
        height = min(max_height, h * 0.58, usable_width / text_weight(value))
        height = max(height, 1.35)
        baseline_y = y + max((h - height) * 0.5, 0.2)
        text(id_, x + pad, baseline_y, value, layer, height, group=group, tags=tags)

    def poly(
        id_: str,
        points: list[tuple[float, float]],
        closed: bool = False,
        layer: str = "geometry",
        group: str | None = None,
        tags: list[str] | None = None,
    ) -> None:
        entities.append(
            PolylineEntity(
                id=id_,
                layer=layer,
                points=[[x, y] for x, y in points],
                closed=closed,
                **entity_meta(group, tags),
            )
        )

    def circle(
        id_: str,
        cx: float,
        cy: float,
        r: float,
        layer: str = "geometry",
        group: str | None = None,
        tags: list[str] | None = None,
    ) -> None:
        entities.append(CircleEntity(id=id_, layer=layer, cx=cx, cy=cy, r=r, **entity_meta(group, tags)))

    def arc(
        id_: str,
        cx: float,
        cy: float,
        r: float,
        start: float,
        end: float,
        layer: str = "geometry",
        group: str | None = None,
        tags: list[str] | None = None,
    ) -> None:
        entities.append(
            ArcEntity(
                id=id_,
                layer=layer,
                cx=cx,
                cy=cy,
                r=r,
                start_angle=start,
                end_angle=end,
                **entity_meta(group, tags),
            )
        )

    def grid(
        id_prefix: str,
        x: float,
        y: float,
        w: float,
        h: float,
        cols: int,
        rows: int,
        layer: str = "table",
        group: str | None = None,
        tags: list[str] | None = None,
    ) -> None:
        rect(f"{id_prefix}_border", x, y, w, h, layer, group, tags)
        for c in range(1, cols):
            xx = x + w * c / cols
            line(f"{id_prefix}_v{c}", xx, y, xx, y + h, layer, group, tags)
        for r in range(1, rows):
            yy = y + h * r / rows
            line(f"{id_prefix}_h{r}", x, yy, x + w, yy, layer, group, tags)

    def custom_grid(
        id_prefix: str,
        x: float,
        y: float,
        w: float,
        h: float,
        x_fracs: list[float],
        y_fracs: list[float],
        layer: str = "table",
        group: str | None = None,
        tags: list[str] | None = None,
    ) -> None:
        rect(f"{id_prefix}_border", x, y, w, h, layer, group, tags)
        for idx, frac in enumerate(x_fracs, start=1):
            xx = x + w * frac
            line(f"{id_prefix}_v{idx}", xx, y, xx, y + h, layer, group, tags)
        for idx, frac in enumerate(y_fracs, start=1):
            yy = y + h * frac
            line(f"{id_prefix}_h{idx}", x, yy, x + w, yy, layer, group, tags)

    def hatch_region(id_prefix: str, polygon: list[tuple[float, float]], spacing: float = 7.0) -> None:
        c_values = [point[1] - point[0] for point in polygon]
        c = min(c_values) - spacing
        idx = 0
        while c <= max(c_values) + spacing:
            intersections: list[tuple[float, float]] = []
            for start, end in zip(polygon, polygon[1:] + polygon[:1], strict=True):
                x1, y1 = start
                x2, y2 = end
                f1 = y1 - x1 - c
                f2 = y2 - x2 - c
                if abs(f1) < 1e-9 and abs(f2) < 1e-9:
                    continue
                if f1 * f2 > 0:
                    continue
                denom = f1 - f2
                if abs(denom) < 1e-9:
                    continue
                t = f1 / denom
                if -1e-9 <= t <= 1 + 1e-9:
                    x = x1 + (x2 - x1) * t
                    y = y1 + (y2 - y1) * t
                    if not any(abs(x - px) < 1e-6 and abs(y - py) < 1e-6 for px, py in intersections):
                        intersections.append((x, y))

            intersections.sort()
            for pair_idx in range(0, len(intersections) - 1, 2):
                x1, y1 = intersections[pair_idx]
                x2, y2 = intersections[pair_idx + 1]
                dx = x2 - x1
                dy = y2 - y1
                length = (dx * dx + dy * dy) ** 0.5
                if length < 2.0:
                    continue
                trim = min(0.55, length * 0.2)
                ux = dx / length
                uy = dy / length
                line(
                    f"{id_prefix}_{idx}",
                    x1 + ux * trim,
                    y1 + uy * trim,
                    x2 - ux * trim,
                    y2 - uy * trim,
                    "hatch",
                    group="section_view",
                    tags=["hatch", "clipped_hatch"],
                )
                idx += 1
            c += spacing

    layers = [
        Layer(name="sheet", color="gray"),
        Layer(name="geometry", color="white"),
        Layer(name="centerline", color="white"),
        Layer(name="dimensions", color="white"),
        Layer(name="hatch", color="white"),
        Layer(name="table", color="white"),
        Layer(name="text", color="white"),
        Layer(name="notes", color="white"),
    ]

    # Sheet and drawing frame.
    rect("sheet_border", 0, 0, 420, 297, "sheet", group="sheet")
    rect("drawing_frame", 32, 25, 355, 238, "sheet", group="sheet")
    text("title", 168, 283, "第 1 章   机械零件图", "text", 9, group="heading")
    text("chapter", 4, 267, "1.1  齿轮类零件图", "text", 7, group="heading")
    text("part_title", 18, 256, "1. 圆柱直齿轮（LJT01.01）", "text", 5.5, group="heading")

    # Main sectional view, simplified from the supplied image.
    profile = [
        (98, 76),
        (98, 88),
        (101, 92),
        (125, 92),
        (129, 96),
        (129, 110),
        (134, 116),
        (154, 116),
        (160, 122),
        (160, 130),
        (154, 130),
        (154, 170),
        (160, 170),
        (160, 178),
        (154, 184),
        (134, 184),
        (129, 190),
        (129, 204),
        (125, 208),
        (101, 208),
        (98, 212),
        (98, 224),
        (124, 224),
        (129, 219),
        (129, 205),
        (133, 194),
        (140, 187),
        (158, 187),
        (164, 181),
        (164, 170),
        (154, 170),
        (154, 130),
        (164, 130),
        (164, 119),
        (158, 113),
        (140, 113),
        (133, 106),
        (129, 95),
        (129, 88),
        (124, 76),
    ]
    poly("section_profile", profile, True, "geometry", group="section_view")
    rect("section_bore", 98, 116, 56, 68, "geometry", group="section_view")
    line("section_bore_top_lip", 101, 184, 154, 184, "geometry", group="section_view", tags=["section_detail"])
    line("section_bore_bottom_lip", 101, 116, 154, 116, "geometry", group="section_view", tags=["section_detail"])
    line("section_top_land_outer", 98, 208, 125, 208, "geometry", group="section_view", tags=["section_detail"])
    line("section_top_land_inner", 98, 199, 129, 199, "geometry", group="section_view", tags=["section_detail"])
    line("section_bottom_land_outer", 98, 92, 125, 92, "geometry", group="section_view", tags=["section_detail"])
    line("section_bottom_land_inner", 98, 101, 129, 101, "geometry", group="section_view", tags=["section_detail"])
    line("section_right_shoulder_top", 134, 184, 154, 184, "geometry", group="section_view", tags=["section_detail"])
    line("section_right_shoulder_bottom", 134, 116, 154, 116, "geometry", group="section_view", tags=["section_detail"])
    line("section_left_step_top", 92, 178, 98, 184, "geometry", group="section_view", tags=["section_detail"])
    line("section_left_step_bottom", 92, 122, 98, 116, "geometry", group="section_view", tags=["section_detail"])
    line("section_left_step_vertical", 92, 122, 92, 178, "geometry", group="section_view", tags=["section_detail"])
    arc("section_top_root_fillet", 142, 205, 18, 180, 250, "geometry", group="section_view", tags=["section_detail", "fillet"])
    arc("section_bottom_root_fillet", 142, 95, 18, 110, 180, "geometry", group="section_view", tags=["section_detail", "fillet"])
    line("section_axis_h", 72, 150, 170, 150, "centerline", group="section_view", tags=["centerline"])
    line("section_axis_v", 108, 64, 108, 235, "centerline", group="section_view", tags=["centerline"])

    # Hatching in the cut material.
    hatch_region(
        "hatch_upper",
        [
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
        ],
    )
    hatch_region(
        "hatch_lower",
        [
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
        ],
    )

    # Section dimensions and roughness notes.
    line("dim_left_outer", 54, 78, 54, 222, "dimensions", group="section_view", tags=["dimensions"])
    line("dim_left_inner", 64, 93, 64, 207, "dimensions", group="section_view", tags=["dimensions"])
    line("dim_right_outer", 190, 115, 190, 185, "dimensions", group="section_view", tags=["dimensions"])
    line("dim_right_inner", 174, 124, 174, 176, "dimensions", group="section_view", tags=["dimensions"])
    line("dim_bottom", 86, 68, 126, 68, "dimensions", group="section_view", tags=["dimensions"])
    line("dim_top", 92, 238, 126, 238, "dimensions", group="section_view", tags=["dimensions"])
    text("dim_phi62", 47, 145, "φ62  -0.2", "dimensions", 4, 90, group="section_view", tags=["dimensions"])
    text("dim_phi58", 60, 146, "φ58", "dimensions", 4, 90, group="section_view", tags=["dimensions"])
    text("dim_phi42", 194, 146, "φ42", "dimensions", 4, 90, group="section_view", tags=["dimensions"])
    text("dim_phi25", 176, 146, "φ25 +0.021", "dimensions", 4, 90, group="section_view", tags=["dimensions"])
    text("dim_25", 101, 63, "25", "dimensions", 4, group="section_view", tags=["dimensions"])
    text("dim_15", 104, 241, "15", "dimensions", 4, group="section_view", tags=["dimensions"])
    text("ra32_a", 54, 216, "Ra3.2", "dimensions", 4, group="section_view", tags=["dimensions", "roughness"])
    text("ra16_a", 132, 216, "Ra1.6", "dimensions", 4, group="section_view", tags=["dimensions", "roughness"])
    text("ra16_b", 112, 137, "Ra1.6", "dimensions", 4, group="section_view", tags=["dimensions", "roughness"])
    text("r3", 139, 92, "R3", "dimensions", 4, group="section_view", tags=["dimensions"])
    rect("datum_a", 171, 101, 10, 9, "dimensions", group="section_view", tags=["dimensions", "datum"])
    text("datum_a_text", 173, 103, "A", "dimensions", 4, group="section_view", tags=["dimensions", "datum"])

    # Right circular view with keyway.
    circle("side_outer_circle", 230, 150, 33, "geometry", group="circular_view")
    circle("side_inner_circle", 230, 150, 29, "geometry", group="circular_view")
    poly("keyway", [(223, 181), (223, 203), (237, 203), (237, 181)], False, "geometry", group="circular_view")
    line("side_axis_h", 196, 150, 266, 150, "centerline", group="circular_view", tags=["centerline"])
    line("side_axis_v", 230, 117, 230, 207, "centerline", group="circular_view", tags=["centerline"])
    line("side_dim_width", 223, 214, 237, 214, "dimensions", group="circular_view", tags=["dimensions"])
    line("side_dim_height", 268, 118, 268, 203, "dimensions", group="circular_view", tags=["dimensions"])
    text("side_dim_8", 221, 219, "8±0.018", "dimensions", 3.5, group="circular_view", tags=["dimensions"])
    text("side_dim_283", 271, 145, "28.3 +0.02", "dimensions", 3.4, 90, group="circular_view", tags=["dimensions"])
    text("side_ra32", 255, 174, "Ra3.2", "dimensions", 3.4, group="circular_view", tags=["dimensions", "roughness"])
    rect("tol_box", 254, 198, 26, 10, "dimensions", group="circular_view", tags=["dimensions", "tolerance"])
    text("tol_box_text", 257, 201, "0.01  A", "dimensions", 3.0, group="circular_view", tags=["dimensions", "tolerance"])

    # Parameter and tolerance table.
    param_x = 282
    param_y = 166
    param_w = 95
    param_h = 80
    param_cols = [0.0, 0.24, 0.38, 0.55, 0.70, 0.84, 1.0]
    param_rows = 10
    custom_grid(
        "param_table",
        param_x,
        param_y,
        param_w,
        param_h,
        [0.24, 0.38, 0.55, 0.70, 0.84],
        [i / param_rows for i in range(1, param_rows)],
        "table",
        group="parameter_table",
    )

    def param_cell(col: int, row_from_top: int, value: str, col_span: int = 1, max_height: float = 3.1) -> None:
        x0 = param_x + param_w * param_cols[col]
        x1 = param_x + param_w * param_cols[col + col_span]
        y0 = param_y + param_h * (param_rows - row_from_top - 1) / param_rows
        cell_text(
            f"param_{col}_{row_from_top}_{len(entities)}",
            x0,
            y0,
            x1 - x0,
            param_h / param_rows,
            value,
            "table",
            group="parameter_table",
            max_height=max_height,
        )

    param_entries = [
        (0, 0, "齿廓", 1),
        (2, 0, "渐开线", 1),
        (0, 1, "齿数 z", 1),
        (2, 1, "29", 1),
        (0, 2, "模数 m", 1),
        (2, 2, "2", 1),
        (0, 3, "螺旋角 β", 1),
        (2, 3, "0°", 1),
        (0, 4, "压力角 α", 1),
        (2, 4, "20°", 1),
        (0, 5, "配对齿轮", 1),
        (2, 5, "齿数 z", 1),
        (5, 5, "58", 1),
        (0, 6, "公法线长度", 2),
        (2, 6, "21.48", 2),
        (5, 6, "k 3", 1),
        (0, 7, "跨球尺寸", 2),
        (2, 7, "M", 1),
        (5, 7, "DM", 1),
        (0, 8, "精度等级", 2),
        (2, 8, "7", 1),
        (3, 8, "GB/T 10095", 3),
    ]
    for col, row, value, span in param_entries:
        param_cell(col, row, value, span)

    check_x = param_x
    check_y = 124
    check_w = param_w
    check_h = param_y - check_y
    check_rows = 7
    check_cols = [0.0, 0.48, 0.62, 0.78, 1.0]
    custom_grid(
        "check_table",
        check_x,
        check_y,
        check_w,
        check_h,
        check_cols[1:-1],
        [i / check_rows for i in range(1, check_rows)],
        "table",
        group="parameter_table",
        tags=["tolerance_table"],
    )

    def check_cell(col: int, row_from_top: int, value: str, col_span: int = 1) -> None:
        x0 = check_x + check_w * check_cols[col]
        x1 = check_x + check_w * check_cols[col + col_span]
        y0 = check_y + check_h * (check_rows - row_from_top - 1) / check_rows
        cell_text(
            f"check_{col}_{row_from_top}_{len(entities)}",
            x0,
            y0,
            x1 - x0,
            check_h / check_rows,
            value,
            "table",
            group="parameter_table",
            tags=["tolerance_table"],
            max_height=2.55,
        )

    check_entries = [
        (0, 0, "检测项目", 1),
        (1, 0, "代号", 1),
        (3, 0, "允许值", 1),
        (0, 1, "单个齿距偏差", 1),
        (1, 1, "Fpt", 1),
        (3, 1, "0.011", 1),
        (0, 2, "齿距累积偏差", 1),
        (1, 2, "Fp", 1),
        (3, 2, "0.037", 1),
        (0, 3, "齿廓总偏差", 1),
        (1, 3, "Fa", 1),
        (3, 3, "0.012", 1),
        (0, 4, "齿廓形状偏差", 1),
        (1, 4, "Ffa", 1),
        (3, 4, "0.009", 1),
        (0, 5, "轮廓倾斜偏差", 1),
        (1, 5, "FHa", 1),
        (3, 5, "0.0075", 1),
        (0, 6, "径向跳动公差", 1),
        (1, 6, "Fr", 1),
        (3, 6, "0.029", 1),
    ]
    for col, row, value, span in check_entries:
        check_cell(col, row, value, span)

    # Technical requirements.
    text("tech_title", 304, 116, "技术要求", "notes", 4.2, group="notes")
    notes = [
        "1. 热处理后齿面硬度为241~286HBW。",
        "2. 未注倒角为C2。",
        "3. 齿轮内在质量按MQ级执行。",
        "4. 齿根圆滑过渡，棱角倒钝。",
        "5. 未注尺寸公差按GB/T 1804-m。",
        "6. 未注几何公差按GB/T 1184-K。",
    ]
    for idx, value in enumerate(notes):
        text(f"note_{idx}", 286, 105 - idx * 4.5, value, "notes", 2.2, group="notes")
    text("surface_ra", 340, 90, "Ra12.5", "dimensions", 3.2, group="dimensions", tags=["roughness"])
    poly("surface_symbol", [(331, 86), (336, 79), (341, 96)], False, "dimensions", group="dimensions", tags=["roughness"])

    # Title block.
    custom_grid(
        "title_block",
        230,
        25,
        147,
        52,
        [0.08, 0.20, 0.38, 0.58, 0.73, 0.86],
        [0.18, 0.34, 0.52, 0.72],
        "table",
        group="title_block",
    )
    line("title_block_split", 300, 25, 300, 77, "table", group="title_block")
    title_x = 230
    title_y = 25
    title_w = 147
    title_h = 52
    title_cols = [0.0, 0.08, 0.20, 0.38, 0.58, 0.73, 0.86, 1.0]
    title_rows = [0.0, 0.18, 0.34, 0.52, 0.72, 1.0]

    def title_cell(col: int, row: int, value: str, col_span: int = 1, row_span: int = 1, max_height: float = 3.6) -> None:
        x0 = title_x + title_w * title_cols[col]
        x1 = title_x + title_w * title_cols[col + col_span]
        y0 = title_y + title_h * title_rows[row]
        y1 = title_y + title_h * title_rows[row + row_span]
        cell_text(
            f"title_{col}_{row}_{len(entities)}",
            x0,
            y0,
            x1 - x0,
            y1 - y0,
            value,
            "table",
            group="title_block",
            max_height=max_height,
        )

    title_cell(5, 4, "合肥工业大学", 2, 1, 3.7)
    title_cell(5, 2, "圆柱直齿轮", 2, 2, 3.5)
    title_cell(5, 1, "LJT01.01", 2, 1, 3.0)
    title_cell(4, 3, "4:5", 1, 2, 4.6)
    title_cell(0, 0, "工艺", 1, 1, 2.7)
    title_cell(0, 1, "审核", 1, 1, 2.7)
    title_cell(0, 2, "制图", 1, 1, 2.7)
    title_cell(0, 3, "设计", 1, 1, 2.7)
    title_cell(2, 0, "批准", 2, 1, 2.7)
    title_cell(3, 1, "阶段标记", 1, 1, 2.4)
    title_cell(4, 1, "量", 1, 1, 2.4)

    return DrawingIR(
        units="mm",
        layers=layers,
        entities=entities,
        notes=[
            "Template target: cylindrical spur gear mechanical part drawing.",
            "This deterministic template is a stepping stone before PDF/OCR reconstruction.",
        ],
    )
