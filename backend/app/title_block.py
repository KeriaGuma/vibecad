from __future__ import annotations

import importlib.util
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
from PIL import Image as PILImage

from .models import DrawingIR, Entity, Layer, LineEntity, ProjectState, TableCellOcr, TextEntity, TitleBlockCell
from .reconstruct import PixelGrid, _detect_table_grids
from .reference import _detect_inner_frame, _preprocess_reference_image, _upload_url_to_path

TITLE_BLOCK_RENDER_TAG = "title_block_provider_render"
TITLE_BLOCK_LAYER = "table"
TITLE_BLOCK_TEXT_LAYER = "text"
MIN_TITLE_TEXT_CONFIDENCE = 0.58
MIN_SINGLE_CJK_CONFIDENCE = 0.90
GRID_MERGE_TOLERANCE_MM = 0.08
TITLE_BLOCK_GRID_STROKE_MM = 0.18
CV_TITLE_BLOCK_PROVIDER = "cv_title_block"
PADDLE_TABLE_STRUCTURE_MODEL = "SLANeXt_wired"
PADDLE_TABLE_DOWNLOAD_ENV = "VIBECAD_ALLOW_PADDLE_TABLE_MODEL_DOWNLOAD"
PADDLE_PARALLEL_CELL_KEYS = {
    "cell_box_list",
    "cell_bbox_list",
    "cell_boxes",
    "cell_bboxes",
    "bbox",
    "boxes",
    "cell_texts",
    "texts",
    "rec_texts",
    "contents",
    "cell_scores",
    "scores",
    "rec_scores",
    "confidences",
}


@dataclass(frozen=True)
class TitleBlockExtraction:
    cells: list[TitleBlockCell]
    provider: str
    warnings: list[str]


@dataclass(frozen=True)
class TitleBlockRender:
    ir: DrawingIR
    grid_count: int
    text_count: int
    warnings: list[str]


class TitleBlockProvider(Protocol):
    name: str

    def extract(self, project: ProjectState, uploads_dir: Path) -> TitleBlockExtraction:
        """Extract title-block cells in normalized full-page image coordinates."""


class Img2TableTitleBlockProvider:
    name = "img2table"

    def extract(self, project: ProjectState, uploads_dir: Path) -> TitleBlockExtraction:
        if importlib.util.find_spec("img2table") is None:
            raise RuntimeError("img2table is not installed")
        if not project.source_image:
            raise ValueError("Upload a PDF or image before extracting the title block.")

        image_path = _upload_url_to_path(project.source_image, uploads_dir)
        if not image_path.exists():
            raise FileNotFoundError("Reference image not found")

        image, grid = _load_title_block_crop(image_path)
        crop, crop_bbox = _crop_grid(image, grid)
        with tempfile.TemporaryDirectory(prefix="vibecad_img2table_title_") as tmp:
            crop_path = Path(tmp) / "title_block.png"
            crop.save(crop_path)
            extracted_tables = _run_img2table(crop_path)

        image_width, image_height = image.size
        cells = _img2table_tables_to_cells(extracted_tables, crop_bbox, image_width, image_height)
        if not cells:
            raise ValueError("img2table did not return parseable title-block cells")
        return TitleBlockExtraction(cells=cells, provider=self.name, warnings=[])


class CurrentGridTitleBlockProvider:
    name = "current_grid"

    def __init__(self, table_cells: list[TableCellOcr] | None = None):
        self.table_cells = table_cells or []

    def extract(self, project: ProjectState, uploads_dir: Path) -> TitleBlockExtraction:
        del project, uploads_dir
        cells = [
            TitleBlockCell(
                row=cell.row,
                col=cell.col,
                text=cell.text,
                confidence=cell.confidence,
                x=cell.x,
                y=cell.y,
                width=cell.width,
                height=cell.height,
                provider=self.name,
                source=cell.source,
            )
            for cell in self.table_cells
            if cell.target == "title_block"
        ]
        warnings = [] if cells else ["current_grid title-block provider found no title-block cells."]
        return TitleBlockExtraction(cells=cells, provider=self.name, warnings=warnings)


class CvTitleBlockProvider:
    name = CV_TITLE_BLOCK_PROVIDER

    def extract(self, project: ProjectState, uploads_dir: Path) -> TitleBlockExtraction:
        if not project.source_image:
            raise ValueError("Upload a PDF or image before extracting the title block.")

        image_path = _upload_url_to_path(project.source_image, uploads_dir)
        if not image_path.exists():
            raise FileNotFoundError("Reference image not found")

        image, grid = _load_title_block_crop(image_path)
        crop, crop_bbox = _crop_grid(image, grid)
        warnings: list[str] = []
        cells = _cv_title_block_cells(crop, crop_bbox, image.size, warnings)
        if not cells:
            raise ValueError("CV title-block grid detector did not return cells")
        return TitleBlockExtraction(cells=cells, provider=self.name, warnings=warnings)


class PaddleTableTitleBlockProvider:
    name = "paddlex_table"

    def extract(self, project: ProjectState, uploads_dir: Path) -> TitleBlockExtraction:
        if importlib.util.find_spec("paddleocr") is None:
            raise RuntimeError("PaddleOCR/PaddleX table recognition is not installed")
        if not _paddle_table_model_ready() and not _paddle_model_download_allowed():
            raise RuntimeError(
                f"{PADDLE_TABLE_STRUCTURE_MODEL} model is not available locally; preload it or set "
                f"{PADDLE_TABLE_DOWNLOAD_ENV}=1 to allow first-run download"
            )
        if not project.source_image:
            raise ValueError("Upload a PDF or image before extracting the title block.")

        image_path = _upload_url_to_path(project.source_image, uploads_dir)
        if not image_path.exists():
            raise FileNotFoundError("Reference image not found")

        image, grid = _load_title_block_crop(image_path)
        crop, crop_bbox = _crop_grid(image, grid)
        with tempfile.TemporaryDirectory(prefix="vibecad_paddlex_title_") as tmp:
            crop_path = Path(tmp) / "title_block.png"
            crop.save(crop_path)
            raw_results = _run_paddle_table_structure(crop_path)

        image_width, image_height = image.size
        warnings: list[str] = []
        cells = _paddle_structure_results_to_cells(raw_results, crop, crop_bbox, image_width, image_height, warnings)
        if not cells:
            raise ValueError("Paddle table structure did not return parseable title-block cells")
        return TitleBlockExtraction(cells=cells, provider=self.name, warnings=warnings)


class PPStructureTitleBlockProvider:
    name = "pp_structure"

    def extract(self, project: ProjectState, uploads_dir: Path) -> TitleBlockExtraction:
        if importlib.util.find_spec("paddleocr") is None:
            raise RuntimeError("PaddleOCR/PP-Structure is not installed")
        if not project.source_image:
            raise ValueError("Upload a PDF or image before extracting the title block.")

        image_path = _upload_url_to_path(project.source_image, uploads_dir)
        if not image_path.exists():
            raise FileNotFoundError("Reference image not found")

        image, grid = _load_title_block_crop(image_path)
        crop, crop_bbox = _crop_grid(image, grid)
        with tempfile.TemporaryDirectory(prefix="vibecad_pp_structure_title_") as tmp:
            crop_path = Path(tmp) / "title_block.png"
            crop.save(crop_path)
            raw_results = _run_pp_structure(crop_path)

        image_width, image_height = image.size
        cells = _pp_structure_results_to_cells(raw_results, crop_bbox, image_width, image_height)
        if not cells:
            raise ValueError("PP-Structure did not return parseable title-block cells")
        return TitleBlockExtraction(cells=cells, provider=self.name, warnings=[])


def extract_title_block_cells(
    project: ProjectState,
    uploads_dir: Path,
    table_cells: list[TableCellOcr] | None = None,
) -> TitleBlockExtraction:
    warnings: list[str] = []
    providers: list[TitleBlockProvider] = [
        CvTitleBlockProvider(),
        PaddleTableTitleBlockProvider(),
        Img2TableTitleBlockProvider(),
    ]
    if _pp_structure_enabled():
        providers.append(PPStructureTitleBlockProvider())
    providers.append(CurrentGridTitleBlockProvider(table_cells))
    for provider in providers:
        try:
            result = provider.extract(project, uploads_dir)
        except (FileNotFoundError, NotImplementedError, RuntimeError, ValueError) as exc:
            warnings.append(f"Title block provider {provider.name} skipped: {exc}.")
            continue
        if result.cells:
            quality_issue = _title_block_quality_issue(result.cells, provider.name)
            if quality_issue:
                warnings.append(f"Title block provider {provider.name} rejected: {quality_issue}.")
                warnings.extend(result.warnings)
                continue
            return TitleBlockExtraction(
                cells=result.cells,
                provider=result.provider,
                warnings=[*warnings, *result.warnings],
            )
        warnings.extend(result.warnings)
    return TitleBlockExtraction(cells=[], provider="none", warnings=warnings)


def _pp_structure_enabled() -> bool:
    value = os.environ.get("VIBECAD_ENABLE_PP_STRUCTURE_TITLE_BLOCK", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _paddle_model_download_allowed() -> bool:
    value = os.environ.get(PADDLE_TABLE_DOWNLOAD_ENV, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _paddle_table_model_ready() -> bool:
    model_dir = Path.home() / ".paddlex" / "official_models" / PADDLE_TABLE_STRUCTURE_MODEL
    if not model_dir.exists():
        return False
    return any(model_dir.glob("*.json")) and any(model_dir.glob("*.pdmodel")) or any(model_dir.glob("*.yml"))


def _title_block_quality_issue(cells: list[TitleBlockCell], provider_name: str) -> str:
    if provider_name == "current_grid":
        return ""
    if len(cells) < 12:
        return f"too few cells ({len(cells)})"

    unique_rows = len({cell.row for cell in cells})
    unique_cols = len({cell.col for cell in cells})
    if unique_rows < 4:
        return f"too few row bands ({unique_rows})"
    if unique_cols < 4:
        return f"too few column bands ({unique_cols})"

    min_y = min(cell.y for cell in cells)
    max_y = max(cell.y + cell.height for cell in cells)
    block_height = max(0.001, max_y - min_y)
    oversized = [cell for cell in cells if cell.height > block_height * 0.32]
    if len(oversized) / len(cells) > 0.30:
        return f"too many row-spanning cells ({len(oversized)}/{len(cells)})"
    return ""


def render_title_block_cells_into_ir(ir: DrawingIR, cells: list[TitleBlockCell]) -> TitleBlockRender:
    if not cells:
        return TitleBlockRender(
            ir=ir,
            grid_count=0,
            text_count=0,
            warnings=["Skipped title-block redraw because no title-block cells were available."],
        )

    next_ir = ir.model_copy(deep=True)
    next_ir.entities = [entity for entity in next_ir.entities if not _is_previous_title_block_entity(entity)]
    _ensure_layer(next_ir, TITLE_BLOCK_LAYER, "white")
    _ensure_layer(next_ir, TITLE_BLOCK_TEXT_LAYER, "cyan")
    sheet = _sheet_bbox(next_ir.entities)

    cell_boxes = [(cell, _cell_cad_box(cell, sheet)) for cell in cells]
    grid_entities = _cell_grid_entities(cell_boxes)
    text_entities: list[TextEntity] = []
    skipped_noise = 0
    for cell, box in cell_boxes:
        text = _clean_cell_text(cell.text)
        if not _useful_title_text(text, cell.confidence):
            skipped_noise += 1
            continue
        text_entity = _cell_to_text_entity(cell, box, text, len(text_entities))
        if text_entity is None:
            skipped_noise += 1
            continue
        text_entities.append(text_entity)

    next_ir.entities.extend(grid_entities)
    next_ir.entities.extend(text_entities)
    next_ir.notes = [
        *next_ir.notes,
        f"Redrew title block from {len(cells)} {cells[0].provider} cells.",
    ]
    warnings = []
    if skipped_noise:
        warnings.append(f"Skipped {skipped_noise} low-confidence/noisy title-block cells.")
    return TitleBlockRender(ir=next_ir, grid_count=len(grid_entities), text_count=len(text_entities), warnings=warnings)


def _load_title_block_crop(image_path: Path) -> tuple[PILImage.Image, PixelGrid]:
    processed = _preprocess_reference_image(image_path)
    frame = _detect_inner_frame(processed.dark)
    if frame is None:
        raise ValueError("Could not detect drawing frame.")
    grids = _detect_table_grids(processed.dark, frame)
    title_grids = [grid for grid in grids if grid.target == "title_block"]
    if not title_grids:
        raise ValueError("Could not detect title-block grid.")
    grid = max(title_grids, key=lambda item: (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1]))
    return processed.image.convert("RGB"), grid


def _crop_grid(image: PILImage.Image, grid: PixelGrid) -> tuple[PILImage.Image, tuple[int, int, int, int]]:
    image_width, image_height = image.size
    x1, y1, x2, y2 = grid.bbox
    pad_x = max(4, round((x2 - x1) * 0.01))
    pad_y = max(4, round((y2 - y1) * 0.02))
    bbox = (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(image_width, x2 + pad_x),
        min(image_height, y2 + pad_y),
    )
    return image.crop(bbox), bbox


def _run_img2table(crop_path: Path) -> list[Any]:
    from img2table.document import Image as Img2TableImage

    try:
        from img2table.ocr import TesseractOCR
    except ImportError:
        ocr = None
    else:
        ocr = TesseractOCR(n_threads=1, lang="chi_sim+eng")

    document = Img2TableImage(str(crop_path))
    return document.extract_tables(
        ocr=ocr,
        implicit_rows=True,
        implicit_columns=True,
        borderless_tables=False,
    )


def _run_pp_structure(crop_path: Path) -> list[Any]:
    from paddleocr import TableRecognitionPipelineV2

    pipeline = TableRecognitionPipelineV2(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_layout_detection=False,
        use_ocr_model=True,
    )
    return pipeline.predict(
        str(crop_path),
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_layout_detection=False,
        use_ocr_model=True,
        use_wired_table_cells_trans_to_html=False,
        use_wireless_table_cells_trans_to_html=False,
        use_table_orientation_classify=True,
        use_ocr_results_with_table_cells=True,
    )


def _run_paddle_table_structure(crop_path: Path) -> list[Any]:
    from paddleocr import TableStructureRecognition

    model = TableStructureRecognition(model_name=PADDLE_TABLE_STRUCTURE_MODEL)
    result = model.predict(str(crop_path))
    return result if isinstance(result, list) else list(result)


def _cv_title_block_cells(
    crop_image: PILImage.Image,
    crop_bbox: tuple[int, int, int, int],
    image_size: tuple[int, int],
    warnings: list[str],
) -> list[TitleBlockCell]:
    rows, columns, horizontal_mask, vertical_mask = _detect_cv_title_grid(crop_image)
    if len(rows) < 4 or len(columns) < 4:
        warnings.append(
            f"CV title-block grid rejected: detected {len(rows)} row lines and {len(columns)} column lines."
        )
        return []

    rectangles = _cv_grid_rectangles(rows, columns, horizontal_mask, vertical_mask)
    if len(rectangles) < 12:
        warnings.append(f"CV title-block grid rejected: detected only {len(rectangles)} cells.")
        return []

    crop_x, crop_y, _, _ = crop_bbox
    image_width, image_height = image_size
    cells: list[TitleBlockCell] = []
    for row1, col1, row2, col2 in rectangles:
        x1, x2 = columns[col1], columns[col2]
        y1, y2 = rows[row1], rows[row2]
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        text = ""
        confidence = 0.0
        ocr_bbox = _cv_inner_cell_bbox(x1, y1, x2, y2)
        if ocr_bbox is not None:
            text, confidence = _ocr_title_block_cell(crop_image, ocr_bbox, warnings)
        cells.append(
            TitleBlockCell(
                row=row1,
                col=col1,
                row_span=max(1, row2 - row1),
                col_span=max(1, col2 - col1),
                text=_clean_cell_text(text),
                confidence=confidence,
                x=_ratio(crop_x + x1, image_width),
                y=_ratio(crop_y + y1, image_height),
                width=max(0.001, _ratio(x2 - x1, image_width)),
                height=max(0.001, _ratio(y2 - y1, image_height)),
                provider=CV_TITLE_BLOCK_PROVIDER,
                source="cv_grid_title_block",
            )
        )
    if cells:
        warnings.append(
            f"CV title-block grid: {len(rows)} row lines, {len(columns)} column lines, {len(cells)} cells."
        )
    return _dedupe_cells(cells)


def _detect_cv_title_grid(crop_image: PILImage.Image) -> tuple[list[int], list[int], np.ndarray, np.ndarray]:
    gray = np.asarray(crop_image.convert("L"))
    if gray.size == 0:
        empty = np.zeros((0, 0), dtype=np.uint8)
        return [], [], empty, empty

    denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    horizontal_mask, vertical_mask = _cv_table_line_masks(binary)
    rows = _cv_horizontal_line_positions(horizontal_mask)
    columns = _cv_vertical_line_positions(vertical_mask, rows)
    return rows, columns, horizontal_mask, vertical_mask


def _cv_table_line_masks(binary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = binary.shape
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(18, width // 35), 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(12, height // 18)))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
    horizontal = cv2.dilate(horizontal, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)), iterations=1)
    vertical = cv2.dilate(vertical, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5)), iterations=1)
    return horizontal, vertical


def _cv_horizontal_line_positions(horizontal_mask: np.ndarray) -> list[int]:
    if horizontal_mask.size == 0:
        return []
    height, width = horizontal_mask.shape
    positions: list[int] = []
    for x, y, w, h, _area, cx, cy in _cv_components(horizontal_mask):
        del x, cx
        if w >= width * 0.22 and h <= max(20, height * 0.07):
            positions.append(round(cy))

    positions = _merge_close_pixel_positions(positions, tolerance=max(3, round(height * 0.01)))
    if len(positions) >= 2:
        return positions

    projection = (horizontal_mask > 0).sum(axis=1)
    threshold = max(width * 0.22, _positive_percentile(projection, 75) * 0.50)
    return _projection_positions(projection, threshold, tolerance=max(3, round(height * 0.01)))


def _cv_vertical_line_positions(vertical_mask: np.ndarray, rows: list[int]) -> list[int]:
    if vertical_mask.size == 0:
        return []
    height, width = vertical_mask.shape
    positions: list[int] = []
    min_height = height * 0.07
    long_height = height * 0.14
    max_width = max(18, width * 0.018)
    for x, y, w, h, _area, cx, _cy in _cv_components(vertical_mask):
        del x, y
        if h < min_height or w > max_width:
            continue
        intersections = _cv_row_intersections(vertical_mask, round(cx), rows)
        if h >= long_height or intersections >= 3:
            positions.append(round(cx))

    positions = _merge_close_pixel_positions(positions, tolerance=max(4, round(width * 0.004)))
    if len(positions) >= 2:
        return positions

    projection = (vertical_mask > 0).sum(axis=0)
    threshold = max(height * 0.18, _positive_percentile(projection, 75) * 0.55)
    return _projection_positions(projection, threshold, tolerance=max(4, round(width * 0.004)))


def _cv_components(mask: np.ndarray) -> list[tuple[int, int, int, int, int, float, float]]:
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    components: list[tuple[int, int, int, int, int, float, float]] = []
    for idx in range(1, count):
        x, y, width, height, area = [int(value) for value in stats[idx]]
        if area < 20:
            continue
        cx, cy = centroids[idx]
        components.append((x, y, width, height, area, float(cx), float(cy)))
    return components


def _cv_row_intersections(vertical_mask: np.ndarray, x: int, rows: list[int]) -> int:
    height, width = vertical_mask.shape
    count = 0
    for y in rows:
        patch = vertical_mask[max(0, y - 4) : min(height, y + 5), max(0, x - 4) : min(width, x + 5)]
        if np.any(patch > 0):
            count += 1
    return count


def _projection_positions(projection: np.ndarray, threshold: float, tolerance: int) -> list[int]:
    indices = np.flatnonzero(projection >= threshold)
    if indices.size == 0:
        return []
    clusters: list[list[int]] = [[int(indices[0])]]
    for raw_idx in indices[1:]:
        idx = int(raw_idx)
        if idx <= clusters[-1][-1] + tolerance:
            clusters[-1].append(idx)
        else:
            clusters.append([idx])
    return [round(sum(cluster) / len(cluster)) for cluster in clusters]


def _positive_percentile(values: np.ndarray, percentile: float) -> float:
    positive = values[values > 0]
    if positive.size == 0:
        return 0.0
    return float(np.percentile(positive, percentile))


def _merge_close_pixel_positions(values: list[int], tolerance: int) -> list[int]:
    if not values:
        return []
    groups: list[list[int]] = []
    for value in sorted(values):
        if not groups or value - groups[-1][-1] > tolerance:
            groups.append([value])
        else:
            groups[-1].append(value)
    return [round(sum(group) / len(group)) for group in groups]


def _cv_grid_rectangles(
    rows: list[int],
    columns: list[int],
    horizontal_mask: np.ndarray,
    vertical_mask: np.ndarray,
) -> list[tuple[int, int, int, int]]:
    row_count = len(rows) - 1
    col_count = len(columns) - 1
    if row_count <= 0 or col_count <= 0 or row_count * col_count > 800:
        return []

    parent = list(range(row_count * col_count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for row in range(row_count):
        y1, y2 = rows[row], rows[row + 1]
        for col in range(col_count):
            x1, x2 = columns[col], columns[col + 1]
            cell_idx = row * col_count + col
            if col + 1 < col_count and not _cv_vertical_boundary_exists(vertical_mask, columns[col + 1], y1, y2):
                union(cell_idx, cell_idx + 1)
            if row + 1 < row_count and not _cv_horizontal_boundary_exists(horizontal_mask, rows[row + 1], x1, x2):
                union(cell_idx, cell_idx + col_count)

    by_component: dict[int, list[tuple[int, int]]] = {}
    for row in range(row_count):
        for col in range(col_count):
            by_component.setdefault(find(row * col_count + col), []).append((row, col))

    rectangles: list[tuple[int, int, int, int]] = []
    for atomic_cells in by_component.values():
        row_values = [row for row, _col in atomic_cells]
        col_values = [col for _row, col in atomic_cells]
        row1, row2 = min(row_values), max(row_values) + 1
        col1, col2 = min(col_values), max(col_values) + 1
        rectangles.append((row1, col1, row2, col2))
    return sorted(set(rectangles))


def _cv_vertical_boundary_exists(mask: np.ndarray, x: int, y1: int, y2: int) -> bool:
    height, width = mask.shape
    if y2 - y1 < 4:
        return False
    band = mask[max(0, y1 + 1) : min(height, y2 - 1), max(0, x - 3) : min(width, x + 4)] > 0
    if band.size == 0:
        return False
    row_has_line = band.any(axis=1)
    edge = max(2, min(8, (y2 - y1) // 4))
    return bool(row_has_line.mean() >= 0.45 and row_has_line[:edge].any() and row_has_line[-edge:].any())


def _cv_horizontal_boundary_exists(mask: np.ndarray, y: int, x1: int, x2: int) -> bool:
    height, width = mask.shape
    if x2 - x1 < 4:
        return False
    band = mask[max(0, y - 3) : min(height, y + 4), max(0, x1 + 1) : min(width, x2 - 1)] > 0
    if band.size == 0:
        return False
    col_has_line = band.any(axis=0)
    edge = max(2, min(10, (x2 - x1) // 4))
    return bool(col_has_line.mean() >= 0.45 and col_has_line[:edge].any() and col_has_line[-edge:].any())


def _cv_inner_cell_bbox(x1: int, y1: int, x2: int, y2: int) -> tuple[int, int, int, int] | None:
    width = x2 - x1
    height = y2 - y1
    if width < 10 or height < 10:
        return None
    pad_x = max(1, min(6, round(width * 0.05)))
    pad_y = max(1, min(5, round(height * 0.10)))
    inner = (x1 + pad_x, y1 + pad_y, x2 - pad_x, y2 - pad_y)
    if inner[2] <= inner[0] or inner[3] <= inner[1]:
        return None
    return inner


def _paddle_structure_results_to_cells(
    results: list[Any],
    crop_image: PILImage.Image,
    crop_bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    warnings: list[str],
) -> list[TitleBlockCell]:
    records: list[dict[str, Any]] = []
    for result in results:
        records.extend(_collect_cell_records(_result_to_plain_data(result)))
    records = _filter_structural_cell_records(records, crop_image.size)
    if not records:
        return []

    assigned = _assign_record_rows_cols(records)
    crop_x, crop_y, _, _ = crop_bbox
    cells: list[TitleBlockCell] = []
    for row, col, record in assigned:
        x1, y1, x2, y2 = _clamp_bbox(record["bbox"], crop_image.size)
        if x2 <= x1 or y2 <= y1:
            continue
        text = str(record.get("text", "")).strip()
        confidence = _extract_confidence(record)
        if not text:
            text, confidence = _ocr_title_block_cell(crop_image, (x1, y1, x2, y2), warnings)
        cells.append(
            TitleBlockCell(
                row=row,
                col=col,
                text=_clean_cell_text(text),
                confidence=confidence,
                x=_ratio(crop_x + x1, image_width),
                y=_ratio(crop_y + y1, image_height),
                width=max(0.001, _ratio(x2 - x1, image_width)),
                height=max(0.001, _ratio(y2 - y1, image_height)),
                provider="paddlex_table",
                source="slanext_wired_title_block",
            )
        )
    return _dedupe_cells(cells)


def _ocr_title_block_cell(
    crop_image: PILImage.Image,
    bbox: tuple[int, int, int, int],
    warnings: list[str],
) -> tuple[str, float]:
    from .ocr import _paddleocr_available
    from .table_ocr import _ocr_cell

    engine = "paddle" if _paddleocr_available() else "tesseract"
    tesseract = shutil.which("tesseract")
    if engine == "tesseract" and tesseract is None:
        warnings.append("Title-block cell OCR skipped: no PaddleOCR runtime or Tesseract binary is available.")
        return "", 0.0
    text, confidence, _used_engine, _used_language, _source = _ocr_cell(
        crop_image,
        bbox,
        "zh",
        engine,
        tesseract,
        warnings,
    )
    return text, confidence


def _pp_structure_results_to_cells(
    results: list[Any],
    crop_bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> list[TitleBlockCell]:
    records: list[dict[str, Any]] = []
    for result in results:
        records.extend(_collect_cell_records(_result_to_plain_data(result)))
    if not records:
        return []

    assigned = _assign_record_rows_cols(records)
    crop_x, crop_y, _, _ = crop_bbox
    cells: list[TitleBlockCell] = []
    for row, col, record in assigned:
        bbox = record["bbox"]
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            continue
        cells.append(
            TitleBlockCell(
                row=row,
                col=col,
                text=_clean_cell_text(str(record.get("text", ""))),
                confidence=_extract_confidence(record),
                x=_ratio(crop_x + x1, image_width),
                y=_ratio(crop_y + y1, image_height),
                width=max(0.001, _ratio(x2 - x1, image_width)),
                height=max(0.001, _ratio(y2 - y1, image_height)),
                provider="pp_structure",
                source="pp_structure_title_block",
            )
        )
    return _dedupe_cells(cells)


def _img2table_tables_to_cells(
    tables: list[Any],
    crop_bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> list[TitleBlockCell]:
    crop_x, crop_y, _, _ = crop_bbox
    cells: list[TitleBlockCell] = []
    for table in tables:
        for row, col, raw_cell in _iter_img2table_cells(table):
            bbox = _extract_bbox(raw_cell)
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            if x2 <= x1 or y2 <= y1:
                continue
            text = _extract_cell_value(raw_cell)
            cells.append(
                TitleBlockCell(
                    row=row,
                    col=col,
                    text=_clean_cell_text(text),
                    confidence=_extract_confidence(raw_cell),
                    x=_ratio(crop_x + x1, image_width),
                    y=_ratio(crop_y + y1, image_height),
                    width=max(0.001, _ratio(x2 - x1, image_width)),
                    height=max(0.001, _ratio(y2 - y1, image_height)),
                    provider="img2table",
                    source="img2table_title_block",
                )
            )
    return _dedupe_cells(cells)


def _result_to_plain_data(value: Any) -> Any:
    if isinstance(value, dict):
        return value
    json_method = getattr(value, "json", None)
    if callable(json_method):
        try:
            json_value = json_method()
            if isinstance(json_value, dict):
                return json_value.get("res", json_value)
        except Exception:  # noqa: BLE001 - third-party result wrappers vary by version
            pass
    if hasattr(value, "items"):
        try:
            return dict(value.items())
        except Exception:  # noqa: BLE001 - keep provider best-effort
            return value
    return value


def _collect_cell_records(value: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if isinstance(value, dict):
        parallel_records = _records_from_parallel_lists(value)
        records.extend(parallel_records)
        bbox = _bbox_from_mapping(value)
        if bbox is not None and _looks_like_cell_mapping(value):
            records.append(
                {
                    "bbox": bbox,
                    "text": _text_from_mapping(value),
                    "confidence": _extract_confidence(value),
                }
            )
        for key, child in value.items():
            if parallel_records and str(key) in PADDLE_PARALLEL_CELL_KEYS:
                continue
            records.extend(_collect_cell_records(child))
    elif isinstance(value, list):
        if _looks_like_bbox(value):
            return [{"bbox": _normalize_bbox(value), "text": "", "confidence": 0.75}]
        for child in value:
            records.extend(_collect_cell_records(child))
    return [record for record in records if record.get("bbox") is not None]


def _records_from_parallel_lists(value: dict[str, Any]) -> list[dict[str, Any]]:
    bbox_list = None
    for key in ("cell_box_list", "cell_bbox_list", "cell_boxes", "cell_bboxes", "bbox", "boxes"):
        candidate = value.get(key)
        if isinstance(candidate, list) and candidate and all(_looks_like_bbox(item) for item in candidate):
            bbox_list = candidate
            break
    if not bbox_list:
        return []

    texts = _parallel_values(value, ("cell_texts", "texts", "rec_texts", "contents"))
    scores = _parallel_values(value, ("cell_scores", "scores", "rec_scores", "confidences"))
    records: list[dict[str, Any]] = []
    for idx, bbox in enumerate(bbox_list):
        records.append(
            {
                "bbox": _normalize_bbox(bbox),
                "text": str(texts[idx]) if idx < len(texts) else "",
                "confidence": scores[idx] if idx < len(scores) else 0.75,
            }
        )
    return records


def _parallel_values(value: dict[str, Any], keys: tuple[str, ...]) -> list[Any]:
    for key in keys:
        candidate = value.get(key)
        if isinstance(candidate, list):
            return candidate
    return []


def _assign_record_rows_cols(records: list[dict[str, Any]]) -> list[tuple[int, int, dict[str, Any]]]:
    sorted_records = sorted(records, key=lambda item: (_bbox_center_y(item["bbox"]), item["bbox"][0]))
    heights = [max(1, record["bbox"][3] - record["bbox"][1]) for record in sorted_records]
    tolerance = max(6.0, (sum(heights) / len(heights)) * 0.55)

    rows: list[list[dict[str, Any]]] = []
    row_centers: list[float] = []
    for record in sorted_records:
        center_y = _bbox_center_y(record["bbox"])
        target_idx = None
        for idx, row_center in enumerate(row_centers):
            if abs(center_y - row_center) <= tolerance:
                target_idx = idx
                break
        if target_idx is None:
            rows.append([record])
            row_centers.append(center_y)
            continue
        rows[target_idx].append(record)
        row_centers[target_idx] = sum(_bbox_center_y(item["bbox"]) for item in rows[target_idx]) / len(rows[target_idx])

    assigned: list[tuple[int, int, dict[str, Any]]] = []
    for row_idx, row_records in enumerate(rows):
        for col_idx, record in enumerate(sorted(row_records, key=lambda item: item["bbox"][0])):
            assigned.append((row_idx, col_idx, record))
    return assigned


def _filter_structural_cell_records(records: list[dict[str, Any]], crop_size: tuple[int, int]) -> list[dict[str, Any]]:
    crop_width, crop_height = crop_size
    crop_area = max(1, crop_width * crop_height)
    filtered: list[dict[str, Any]] = []
    for record in records:
        bbox = _clamp_bbox(record["bbox"], crop_size)
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        area = width * height
        if width < 4 or height < 4:
            continue
        if area >= crop_area * 0.82:
            continue
        if width >= crop_width * 0.96 and height >= crop_height * 0.60:
            continue
        filtered.append({**record, "bbox": bbox})
    return _dedupe_record_bboxes(filtered)


def _dedupe_record_bboxes(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[int, int, int, int]] = set()
    result: list[dict[str, Any]] = []
    for record in records:
        key = tuple(record["bbox"])
        if key in seen:
            continue
        seen.add(key)
        result.append(record)
    return result


def _clamp_bbox(bbox: tuple[int, int, int, int], image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = image_size
    x1, y1, x2, y2 = bbox
    return (
        max(0, min(width - 1, x1)),
        max(0, min(height - 1, y1)),
        max(1, min(width, x2)),
        max(1, min(height, y2)),
    )


def _bbox_from_mapping(value: dict[str, Any]) -> tuple[int, int, int, int] | None:
    for key in ("bbox", "box", "cell_bbox", "cell_box", "coordinate", "poly"):
        candidate = value.get(key)
        if _looks_like_bbox(candidate):
            return _normalize_bbox(candidate)
    return None


def _looks_like_cell_mapping(value: dict[str, Any]) -> bool:
    keys = {str(key).lower() for key in value}
    return bool(keys & {"cell", "text", "value", "content", "rec_text"}) or any("cell" in key for key in keys)


def _text_from_mapping(value: dict[str, Any]) -> str:
    for key in ("text", "value", "content", "rec_text"):
        text = value.get(key)
        if text is not None:
            return str(text)
    return ""


def _looks_like_bbox(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) not in {4, 8}:
        return False
    try:
        [float(item) for item in value]
    except (TypeError, ValueError):
        return False
    return True


def _normalize_bbox(value: Any) -> tuple[int, int, int, int]:
    coords = [float(item) for item in value]
    if len(coords) == 4:
        x1, y1, x2, y2 = coords
        return _ordered_bbox(x1, y1, x2, y2)
    xs = coords[0::2]
    ys = coords[1::2]
    return _ordered_bbox(min(xs), min(ys), max(xs), max(ys))


def _ordered_bbox(x1: float, y1: float, x2: float, y2: float) -> tuple[int, int, int, int]:
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    return round(left), round(top), round(right), round(bottom)


def _bbox_center_y(bbox: tuple[int, int, int, int]) -> float:
    return (bbox[1] + bbox[3]) / 2


def _iter_img2table_cells(table: Any) -> list[tuple[int, int, Any]]:
    content = getattr(table, "content", None)
    if content is None and isinstance(table, dict):
        content = table.get("content")
    if content is None:
        return []

    out: list[tuple[int, int, Any]] = []
    if isinstance(content, dict):
        for row_key, row_value in content.items():
            row_idx = _safe_int(row_key)
            out.extend((row_idx, col, cell) for col, cell in _iter_row_cells(row_value))
    elif isinstance(content, list):
        for row_idx, row_value in enumerate(content):
            out.extend((row_idx, col, cell) for col, cell in _iter_row_cells(row_value))
    return out


def _iter_row_cells(row_value: Any) -> list[tuple[int, Any]]:
    if isinstance(row_value, dict):
        return [(_safe_int(col), cell) for col, cell in row_value.items()]
    if isinstance(row_value, list):
        return list(enumerate(row_value))
    return []


def _extract_bbox(cell: Any) -> tuple[int, int, int, int] | None:
    bbox = getattr(cell, "bbox", None)
    if bbox is None and isinstance(cell, dict):
        bbox = cell.get("bbox")
    if bbox is None:
        return None
    coords: list[int | None] = []
    for names in (("x1", "left", "xmin"), ("y1", "top", "ymin"), ("x2", "right", "xmax"), ("y2", "bottom", "ymax")):
        coords.append(_first_attr_or_key(bbox, names))
    if any(value is None for value in coords):
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            coords = [_safe_int(value) for value in bbox[:4]]
        else:
            return None
    x1, y1, x2, y2 = [int(value or 0) for value in coords]
    return x1, y1, x2, y2


def _extract_cell_value(cell: Any) -> str:
    value = getattr(cell, "value", None)
    if value is None:
        value = getattr(cell, "text", None)
    if value is None and isinstance(cell, dict):
        value = cell.get("value", cell.get("text", ""))
    return str(value or "")


def _extract_confidence(cell: Any) -> float:
    for name in ("confidence", "conf", "score"):
        value = getattr(cell, name, None)
        if value is None and isinstance(cell, dict):
            value = cell.get(name)
        if value is not None:
            try:
                return max(0.0, min(1.0, float(value)))
            except (TypeError, ValueError):
                return 0.0
    return 0.75


def _dedupe_cells(cells: list[TitleBlockCell]) -> list[TitleBlockCell]:
    seen: set[tuple[int, int, float, float, float, float]] = set()
    result: list[TitleBlockCell] = []
    for cell in cells:
        key = (cell.row, cell.col, round(cell.x, 4), round(cell.y, 4), round(cell.width, 4), round(cell.height, 4))
        if key in seen:
            continue
        seen.add(key)
        result.append(cell)
    return result


def _cell_grid_entities(cell_boxes: list[tuple[TitleBlockCell, tuple[float, float, float, float]]]) -> list[LineEntity]:
    segments: set[tuple[str, float, float, float]] = set()
    for _, (x1, y1, x2, y2) in cell_boxes:
        segments.add(("v", round(x1, 3), round(y1, 3), round(y2, 3)))
        segments.add(("v", round(x2, 3), round(y1, 3), round(y2, 3)))
        segments.add(("h", round(y1, 3), round(x1, 3), round(x2, 3)))
        segments.add(("h", round(y2, 3), round(x1, 3), round(x2, 3)))

    merged_segments = _merge_grid_segments(sorted(segments))
    entities: list[LineEntity] = []
    for idx, segment in enumerate(merged_segments):
        orientation, primary, start, end = segment
        if orientation == "v":
            entity = LineEntity(
                id=f"title_block_grid_v_{idx:04d}",
                layer=TITLE_BLOCK_LAYER,
                x1=primary,
                y1=start,
                x2=primary,
                y2=end,
                group="title_block",
                tags=["title_block", TITLE_BLOCK_RENDER_TAG, "grid"],
                stroke_width=TITLE_BLOCK_GRID_STROKE_MM,
            )
        else:
            entity = LineEntity(
                id=f"title_block_grid_h_{idx:04d}",
                layer=TITLE_BLOCK_LAYER,
                x1=start,
                y1=primary,
                x2=end,
                y2=primary,
                group="title_block",
                tags=["title_block", TITLE_BLOCK_RENDER_TAG, "grid"],
                stroke_width=TITLE_BLOCK_GRID_STROKE_MM,
            )
        entities.append(entity)
    return entities


def _merge_grid_segments(
    segments: list[tuple[str, float, float, float]],
) -> list[tuple[str, float, float, float]]:
    merged: list[tuple[str, float, float, float]] = []
    by_axis: dict[tuple[str, float], list[tuple[float, float]]] = {}
    for orientation, primary, start, end in segments:
        start, end = sorted((start, end))
        by_axis.setdefault((orientation, primary), []).append((start, end))

    for (orientation, primary), ranges in sorted(by_axis.items()):
        current_start: float | None = None
        current_end: float | None = None
        for start, end in sorted(ranges):
            if current_start is None or current_end is None:
                current_start, current_end = start, end
                continue
            if start <= current_end + GRID_MERGE_TOLERANCE_MM:
                current_end = max(current_end, end)
                continue
            merged.append((orientation, primary, round(current_start, 3), round(current_end, 3)))
            current_start, current_end = start, end
        if current_start is not None and current_end is not None:
            merged.append((orientation, primary, round(current_start, 3), round(current_end, 3)))
    return merged


def _cell_to_text_entity(
    cell: TitleBlockCell,
    box: tuple[float, float, float, float],
    text: str,
    index: int,
) -> TextEntity | None:
    x1, y1, x2, y2 = box
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    if width < 1.2 or height < 0.8:
        return None

    text_len = max(len(text.replace(" ", "")), 1)
    font_height = min(height * 0.56, width / (text_len * 0.62) * 0.90, 2.4)
    if font_height < 0.68:
        return None
    return TextEntity(
        id=f"title_block_text_{cell.row:03d}_{cell.col:03d}_{index:04d}",
        layer=TITLE_BLOCK_TEXT_LAYER,
        x=round(x1 + width * 0.08, 4),
        y=round(y1 + height * 0.58, 4),
        text=text,
        height=round(font_height, 3),
        group="title_block",
        tags=["title_block", TITLE_BLOCK_RENDER_TAG, "ocr_text", cell.provider, cell.source],
    )


def _is_previous_title_block_entity(entity: Entity) -> bool:
    tags = set(entity.tags)
    if TITLE_BLOCK_RENDER_TAG in tags:
        return True
    return entity.group == "title_block"


def _cell_cad_box(
    cell: TitleBlockCell,
    sheet: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    sx, sy, sw, sh = sheet
    x1 = sx + cell.x * sw
    x2 = sx + min(1.0, cell.x + cell.width) * sw
    y_top = sy + (1.0 - cell.y) * sh
    y_bottom = sy + (1.0 - min(1.0, cell.y + cell.height)) * sh
    return round(x1, 4), round(y_bottom, 4), round(x2, 4), round(y_top, 4)


def _clean_cell_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).replace("|", "I").strip()


def _useful_title_text(text: str, confidence: float) -> bool:
    if not text or confidence < MIN_TITLE_TEXT_CONFIDENCE:
        return False
    compact = text.replace(" ", "")
    if compact in {"I", "J", "l", "1", "-", "—", "–", "_", ".", "·", "/", "\\"}:
        return False
    if len(compact) == 1 and _contains_cjk(compact):
        return confidence >= MIN_SINGLE_CJK_CONFIDENCE
    if len(compact) == 1 and compact.isascii() and compact.isalpha():
        return False
    return bool(re.search(r"[\w\u3400-\u9fff]", compact))


def _contains_cjk(text: str) -> bool:
    return any("\u3400" <= char <= "\u9fff" for char in text)


def _sheet_bbox(entities: list[Entity]) -> tuple[float, float, float, float]:
    rectangles = [entity for entity in entities if getattr(entity, "type", "") == "rectangle"]
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
        return min(xs), min(ys), max(max(xs) - min(xs), 1.0), max(max(ys) - min(ys), 1.0)
    return 0.0, 0.0, 420.0, 297.0


def _ensure_layer(ir: DrawingIR, name: str, color: str) -> None:
    if not any(layer.name == name for layer in ir.layers):
        ir.layers.append(Layer(name=name, color=color))


def _first_attr_or_key(value: Any, names: tuple[str, ...]) -> int | None:
    for name in names:
        item = getattr(value, name, None)
        if item is None and isinstance(value, dict):
            item = value.get(name)
        if item is not None:
            return _safe_int(item)
    return None


def _safe_int(value: Any) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return 0


def _ratio(value: int | float, total: int) -> float:
    return round(max(0.0, min(1.0, float(value) / max(total, 1))), 4)
