from __future__ import annotations

from app.models import DrawingIR, RectangleEntity, TableCellOcr, TextEntity
from app.table_text import TABLE_OCR_TEXT_TAG, render_table_ocr_cells_into_ir


def test_render_table_ocr_cells_into_ir_adds_text_inside_cells():
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
            )
        ]
    )
    cells = [
        TableCellOcr(
            target="title_block",
            row=0,
            col=0,
            text="圆柱直齿轮",
            confidence=0.93,
            x=0.70,
            y=0.80,
            width=0.12,
            height=0.04,
        ),
        TableCellOcr(
            target="title_block",
            row=0,
            col=1,
            text="I",
            confidence=0.98,
            x=0.10,
            y=0.10,
            width=0.02,
            height=0.02,
        ),
    ]

    result = render_table_ocr_cells_into_ir(ir, cells)

    texts = [entity for entity in result.ir.entities if isinstance(entity, TextEntity)]
    assert result.text_count == 1
    assert texts[0].text == "圆柱直齿轮"
    assert texts[0].group == "title_block"
    assert TABLE_OCR_TEXT_TAG in texts[0].tags
    assert 294 < texts[0].x < 345
    assert 35 < texts[0].y < 45
    assert result.warnings


def test_render_table_ocr_cells_into_ir_replaces_previous_ocr_texts():
    ir = DrawingIR(
        entities=[
            TextEntity(
                id="table_ocr_text_old",
                layer="text",
                x=1,
                y=1,
                text="old",
                tags=[TABLE_OCR_TEXT_TAG],
            )
        ]
    )
    cells = [
        TableCellOcr(
            target="title_block",
            row=1,
            col=1,
            text="LJT01.01",
            confidence=0.9,
            x=0.6,
            y=0.8,
            width=0.08,
            height=0.04,
        )
    ]

    result = render_table_ocr_cells_into_ir(ir, cells)

    texts = [entity for entity in result.ir.entities if isinstance(entity, TextEntity)]
    assert len(texts) == 1
    assert texts[0].text == "LJT01.01"
    assert texts[0].id != "table_ocr_text_old"
