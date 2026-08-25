from __future__ import annotations

import re
from dataclasses import dataclass

from .models import DrawingIR, Entity, Layer, LineEntity, RectangleEntity, TableCellOcr, TextEntity

TABLE_OCR_TEXT_TAG = "table_ocr_text"
TABLE_TEXT_LAYER = "text"
MIN_CELL_TEXT_CONFIDENCE = 0.58
MIN_SINGLE_CJK_CONFIDENCE = 0.90


@dataclass(frozen=True)
class TableTextRender:
    ir: DrawingIR
    text_count: int
    warnings: list[str]


def render_table_ocr_cells_into_ir(ir: DrawingIR, cells: list[TableCellOcr]) -> TableTextRender:
    """Render recognized table/title-block cells back into editable CAD text."""
    next_ir = ir.model_copy(deep=True)
    next_ir.entities = [
        entity
        for entity in next_ir.entities
        if TABLE_OCR_TEXT_TAG not in entity.tags and not entity.id.startswith("table_ocr_text_")
    ]
    _ensure_text_layer(next_ir)
    sheet = _sheet_bbox(next_ir.entities)

    text_entities: list[TextEntity] = []
    skipped_noise = 0
    for cell in cells:
        text = _clean_cell_text(cell.text)
        if not _useful_cell_text(text, cell.confidence):
            skipped_noise += 1
            continue
        entity = _cell_to_text_entity(cell, text, sheet, len(text_entities))
        if entity is None:
            skipped_noise += 1
            continue
        text_entities.append(entity)

    next_ir.entities.extend(text_entities)
    if text_entities:
        next_ir.notes = [
            *next_ir.notes,
            f"Rendered {len(text_entities)} OCR table/title-block cell texts into editable CAD text.",
        ]
    warnings = []
    if skipped_noise:
        warnings.append(f"Skipped {skipped_noise} low-confidence/noisy table OCR cells.")
    return TableTextRender(ir=next_ir, text_count=len(text_entities), warnings=warnings)


def _cell_to_text_entity(
    cell: TableCellOcr,
    text: str,
    sheet: tuple[float, float, float, float],
    index: int,
) -> TextEntity | None:
    sx, sy, sw, sh = sheet
    x1 = sx + cell.x * sw
    x2 = sx + (cell.x + cell.width) * sw
    y_top = sy + (1.0 - cell.y) * sh
    y_bottom = sy + (1.0 - cell.y - cell.height) * sh
    width = max(0.0, x2 - x1)
    height = max(0.0, y_top - y_bottom)
    if width < 1.2 or height < 0.8:
        return None

    text_len = max(len(text), 1)
    font_height = min(height * 0.58, width / (text_len * 0.62) * 0.92, 2.6)
    if font_height < 0.72:
        return None
    font_height = round(font_height, 3)
    return TextEntity(
        id=f"table_ocr_text_{cell.target}_{cell.row:03d}_{cell.col:03d}_{index:04d}",
        layer=TABLE_TEXT_LAYER,
        x=round(x1 + width * 0.08, 4),
        y=round(y_bottom + height * 0.58, 4),
        text=text,
        height=font_height,
        group=cell.target,
        tags=[TABLE_OCR_TEXT_TAG, cell.target, "ocr_text", cell.engine, cell.source],
    )


def _clean_cell_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text.strip())
    return cleaned.replace("|", "I").strip()


def _useful_cell_text(text: str, confidence: float) -> bool:
    if not text or confidence < MIN_CELL_TEXT_CONFIDENCE:
        return False
    compact = text.replace(" ", "")
    if compact in {"I", "J", "l", "1", "-", "—", "–", "_", ".", "·"}:
        return False
    if len(compact) == 1 and _contains_cjk(compact):
        return confidence >= MIN_SINGLE_CJK_CONFIDENCE
    if len(compact) == 1 and compact.isascii() and compact.isalpha():
        return False
    if not re.search(r"[\w\u3400-\u9fff]", compact):
        return False
    return True


def _contains_cjk(text: str) -> bool:
    return any("\u3400" <= char <= "\u9fff" for char in text)


def _sheet_bbox(entities: list[Entity]) -> tuple[float, float, float, float]:
    rectangles = [entity for entity in entities if isinstance(entity, RectangleEntity)]
    preferred = [
        entity
        for entity in rectangles
        if entity.id in {"scan_sheet_border", "sheet_border", "recon_sheet_border"}
        or entity.group == "sheet"
        or "sheet" in entity.tags
    ]
    if preferred:
        rect = max(preferred, key=lambda item: item.width * item.height)
        return float(rect.x), float(rect.y), float(rect.width), float(rect.height)
    if rectangles:
        rect = max(rectangles, key=lambda item: item.width * item.height)
        return float(rect.x), float(rect.y), float(rect.width), float(rect.height)

    xs: list[float] = []
    ys: list[float] = []
    for entity in entities:
        if isinstance(entity, LineEntity):
            xs.extend([entity.x1, entity.x2])
            ys.extend([entity.y1, entity.y2])
        elif isinstance(entity, TextEntity):
            xs.append(entity.x)
            ys.append(entity.y)
    if xs and ys:
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        return float(min_x), float(min_y), float(max(max_x - min_x, 1.0)), float(max(max_y - min_y, 1.0))
    return 0.0, 0.0, 420.0, 297.0


def _ensure_text_layer(ir: DrawingIR) -> None:
    for layer in ir.layers:
        if layer.name == TABLE_TEXT_LAYER:
            layer.color = "white"
            return
    ir.layers.append(Layer(name=TABLE_TEXT_LAYER, color="white"))
