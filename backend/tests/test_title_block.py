from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from app.models import DrawingIR, LineEntity, RectangleEntity, TableCellOcr, TextEntity, TitleBlockCell
from app.title_block import (
    TITLE_BLOCK_RENDER_TAG,
    CurrentGridTitleBlockProvider,
    _cv_grid_rectangles,
    _cv_title_block_cells,
    _detect_cv_title_grid,
    _merge_grid_segments,
    _paddle_structure_results_to_cells,
    _pp_structure_results_to_cells,
    _title_block_quality_issue,
    render_title_block_cells_into_ir,
)


def test_current_grid_title_block_provider_converts_title_cells():
    cells = [
        TableCellOcr(
            target="title_block",
            row=0,
            col=1,
            text="圆柱直齿轮",
            confidence=0.93,
            x=0.60,
            y=0.82,
            width=0.12,
            height=0.04,
            source="mock_table",
        ),
        TableCellOcr(
            target="parameter_table",
            row=0,
            col=1,
            text="29",
            confidence=0.90,
            x=0.60,
            y=0.10,
            width=0.06,
            height=0.04,
        ),
    ]

    result = CurrentGridTitleBlockProvider(cells).extract(None, None)  # type: ignore[arg-type]

    assert result.provider == "current_grid"
    assert len(result.cells) == 1
    assert result.cells[0].text == "圆柱直齿轮"
    assert result.cells[0].provider == "current_grid"
    assert result.cells[0].source == "mock_table"


def test_render_title_block_cells_replaces_title_block_group_without_table_text_logic():
    ir = DrawingIR(
        entities=[
            RectangleEntity(
                id="scan_sheet_border",
                layer="sheet",
                x=0,
                y=0,
                width=420,
                height=210,
                group="sheet",
                tags=["sheet"],
            ),
            LineEntity(
                id="old_title_grid",
                layer="table",
                x1=1,
                y1=1,
                x2=2,
                y2=1,
                group="title_block",
                tags=["title_block", "grid"],
            ),
        ]
    )
    cells = [
        TitleBlockCell(
            row=0,
            col=0,
            text="合肥工业大学",
            confidence=0.94,
            x=0.66,
            y=0.72,
            width=0.18,
            height=0.05,
            provider="current_grid",
        ),
        TitleBlockCell(
            row=0,
            col=1,
            text="I",
            confidence=0.96,
            x=0.84,
            y=0.72,
            width=0.02,
            height=0.05,
            provider="current_grid",
        ),
    ]

    result = render_title_block_cells_into_ir(ir, cells)

    assert not any(entity.id == "old_title_grid" for entity in result.ir.entities)
    grid_lines = [
        entity for entity in result.ir.entities if isinstance(entity, LineEntity) and TITLE_BLOCK_RENDER_TAG in entity.tags
    ]
    texts = [entity for entity in result.ir.entities if isinstance(entity, TextEntity)]
    assert result.grid_count == len(grid_lines)
    assert result.grid_count > 0
    assert result.text_count == 1
    assert texts[0].text == "合肥工业大学"
    assert texts[0].group == "title_block"
    assert TITLE_BLOCK_RENDER_TAG in texts[0].tags
    assert result.warnings


def test_render_title_block_cells_preserves_ir_when_no_cells():
    ir = DrawingIR(entities=[LineEntity(id="keep", layer="table", x1=0, y1=0, x2=1, y2=1, group="title_block")])

    result = render_title_block_cells_into_ir(ir, [])

    assert result.ir is ir
    assert result.grid_count == 0
    assert result.text_count == 0
    assert result.warnings


def test_pp_structure_result_parser_outputs_title_block_cells():
    raw_results = [
        {
            "cell_box_list": [[0, 0, 10, 10], [10, 0, 20, 10], [0, 10, 10, 20]],
            "cell_texts": ["设计", "日期", "审核"],
            "scores": [0.91, 0.82, 0.77],
        }
    ]

    cells = _pp_structure_results_to_cells(raw_results, (100, 200, 200, 300), 1000, 1000)

    assert [cell.text for cell in cells] == ["设计", "日期", "审核"]
    assert [(cell.row, cell.col) for cell in cells] == [(0, 0), (0, 1), (1, 0)]
    assert cells[0].provider == "pp_structure"
    assert cells[0].x == 0.1
    assert cells[0].y == 0.2


def test_paddle_structure_cells_run_cell_ocr(monkeypatch):
    crop = Image.new("RGB", (120, 40), "white")
    warnings: list[str] = []

    def fake_ocr(_crop, bbox, _warnings):
        return f"cell-{bbox[0]}-{bbox[1]}", 0.88

    monkeypatch.setattr("app.title_block._ocr_title_block_cell", fake_ocr)

    cells = _paddle_structure_results_to_cells(
        [{"bbox": [[0, 0, 20, 20], [20, 0, 40, 20], [0, 0, 120, 40]]}],
        crop,
        (100, 200, 220, 240),
        1000,
        1000,
        warnings,
    )

    assert [cell.text for cell in cells] == ["cell-0-0", "cell-20-0"]
    assert [(cell.row, cell.col) for cell in cells] == [(0, 0), (0, 1)]
    assert {cell.provider for cell in cells} == {"paddlex_table"}
    assert {cell.source for cell in cells} == {"slanext_wired_title_block"}
    assert cells[0].confidence == 0.88
    assert cells[0].x == 0.1
    assert cells[0].y == 0.2


def test_cv_title_grid_ignores_short_text_strokes():
    image = Image.new("L", (420, 180), "white")
    draw = ImageDraw.Draw(image)
    for y in [10, 50, 90, 130, 170]:
        draw.line((10, y, 410, y), fill="black", width=2)
    for x in [10, 110, 210, 310, 410]:
        draw.line((x, 10, x, 170), fill="black", width=2)
    for x in [44, 52, 61]:
        draw.line((x, 62, x, 82), fill="black", width=2)

    rows, columns, _horizontal_mask, _vertical_mask = _detect_cv_title_grid(image.convert("RGB"))

    assert len(rows) == 5
    assert len(columns) == 5
    assert not any(40 <= col <= 65 for col in columns)


def test_cv_grid_rectangles_merge_cells_where_divider_is_absent():
    rows = [10, 50, 90, 130]
    columns = [10, 110, 210, 310]
    horizontal_mask = Image.new("L", (320, 140), "black")
    vertical_mask = Image.new("L", (320, 140), "black")
    hdraw = ImageDraw.Draw(horizontal_mask)
    vdraw = ImageDraw.Draw(vertical_mask)
    for y in rows:
        hdraw.line((10, y, 310, y), fill="white", width=3)
    for x in [10, 110, 310]:
        vdraw.line((x, 10, x, 130), fill="white", width=3)
    vdraw.line((210, 90, 210, 130), fill="white", width=3)

    rectangles = _cv_grid_rectangles(
        rows,
        columns,
        np.asarray(horizontal_mask),
        np.asarray(vertical_mask),
    )

    assert (0, 1, 1, 3) in rectangles
    assert (1, 1, 2, 3) in rectangles
    assert (2, 1, 3, 2) in rectangles
    assert (2, 2, 3, 3) in rectangles


def test_cv_title_block_cells_use_grid_provider_and_cell_ocr(monkeypatch):
    image = Image.new("L", (420, 180), "white")
    draw = ImageDraw.Draw(image)
    for y in [10, 50, 90, 130, 170]:
        draw.line((10, y, 410, y), fill="black", width=2)
    for x in [10, 110, 210, 310, 410]:
        draw.line((x, 10, x, 170), fill="black", width=2)

    def fake_ocr(_crop, bbox, _warnings):
        return f"{bbox[0]}-{bbox[1]}", 0.9

    monkeypatch.setattr("app.title_block._ocr_title_block_cell", fake_ocr)
    cells = _cv_title_block_cells(image.convert("RGB"), (100, 200, 520, 380), (1000, 1000), [])

    assert len(cells) == 16
    assert {cell.provider for cell in cells} == {"cv_title_block"}
    assert {cell.source for cell in cells} == {"cv_grid_title_block"}
    assert cells[0].text
    assert cells[0].x > 0.10


def test_title_block_quality_rejects_bad_two_row_paddle_cells():
    cells = [
        TitleBlockCell(
            row=idx // 20,
            col=idx % 20,
            text="",
            confidence=0.5,
            x=0.52 + idx * 0.002,
            y=0.74 if idx < 20 else 0.82,
            width=0.012,
            height=0.08,
            provider="paddlex_table",
        )
        for idx in range(40)
    ]

    assert _title_block_quality_issue(cells, "paddlex_table") == "too few row bands (2)"


def test_title_block_quality_accepts_multiband_provider_cells():
    cells = [
        TitleBlockCell(
            row=row,
            col=col,
            text="",
            confidence=0.8,
            x=0.52 + col * 0.02,
            y=0.74 + row * 0.02,
            width=0.018,
            height=0.015,
            provider="paddlex_table",
        )
        for row in range(5)
        for col in range(6)
    ]

    assert _title_block_quality_issue(cells, "paddlex_table") == ""


def test_title_block_grid_segments_merge_adjacent_ranges():
    segments = [
        ("h", 10.0, 0.0, 5.0),
        ("h", 10.0, 5.02, 9.0),
        ("h", 10.0, 12.0, 13.0),
        ("v", 2.0, 0.0, 1.0),
        ("v", 2.0, 1.01, 2.0),
    ]

    merged = _merge_grid_segments(segments)

    assert ("h", 10.0, 0.0, 9.0) in merged
    assert ("h", 10.0, 12.0, 13.0) in merged
    assert ("v", 2.0, 0.0, 2.0) in merged
