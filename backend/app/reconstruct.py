from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .models import DrawingIR, Layer, LineEntity, ProjectState, RectangleEntity, TextEntity
from .reference import _cluster_mid, _clusters, _detect_inner_frame, _preprocess_reference_image, _upload_url_to_path

CANVAS_WIDTH_MM = 420.0


@dataclass(frozen=True)
class PixelGrid:
    target: str
    label: str
    bbox: tuple[int, int, int, int]
    columns: list[int]
    rows: list[int]


@dataclass(frozen=True)
class LayoutRegion:
    target: str
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class TableReconstruction:
    ir: DrawingIR
    layout_passed: bool
    warnings: list[str]
    regions: list[LayoutRegion]


def reconstruct_tables_from_reference(project: ProjectState, uploads_dir: Path) -> TableReconstruction:
    if not project.source_image:
        raise ValueError("Upload a PDF or image before reconstructing tables.")

    image_path = _upload_url_to_path(project.source_image, uploads_dir)
    if not image_path.exists():
        raise FileNotFoundError("Reference image not found")

    processed = _preprocess_reference_image(image_path)
    image_height, image_width = processed.dark.shape
    canvas_height = CANVAS_WIDTH_MM * image_height / image_width
    dark = processed.dark
    frame = _detect_inner_frame(dark)
    if frame is None:
        raise ValueError("Could not detect drawing frame.")

    grids = _detect_table_grids(dark, frame)
    if not grids:
        raise ValueError("Could not detect parameter/title table grids.")

    entities = [
        RectangleEntity(
            id="recon_sheet_border",
            layer="sheet",
            x=0,
            y=0,
            width=CANVAS_WIDTH_MM,
            height=canvas_height,
            group="sheet",
            tags=["sheet", "reconstructed"],
        )
    ]
    regions: list[LayoutRegion] = []
    for grid in grids:
        regions.append(LayoutRegion(target=grid.target, bbox=_px_bbox_to_cad(grid.bbox, image_width, image_height, canvas_height)))
        entities.extend(_grid_to_entities(grid, image_width, image_height, canvas_height))
        entities.extend(_table_stub_text(grid, image_width, image_height, canvas_height))

    warnings = _layout_warnings(regions)
    ir = DrawingIR(
        units="mm",
        layers=[
            Layer(name="sheet", color="gray"),
            Layer(name="table", color="white"),
            Layer(name="text", color="white"),
        ],
        entities=entities,
        notes=[
            "Reconstructed tables from reference image grid lines.",
            "OCR is not enabled yet; text is seeded from the known sample template.",
        ],
    )
    return TableReconstruction(ir=ir, layout_passed=not warnings, warnings=warnings, regions=regions)


def _detect_table_grids(dark: np.ndarray, frame: tuple[int, int, int, int]) -> list[PixelGrid]:
    fx1, fy1, fx2, fy2 = frame
    fw = max(fx2 - fx1, 1)
    fh = max(fy2 - fy1, 1)
    candidates = [
        (
            "parameter_table",
            "参数表",
            (fx1 + fw * 0.65, fy1 - fh * 0.02, fx2 + fw * 0.01, fy1 + fh * 0.50),
            0.30,
            0.30,
        ),
        (
            "title_block",
            "标题栏",
            (fx1 + fw * 0.45, fy1 + fh * 0.73, fx2 + fw * 0.01, fy2 + fh * 0.02),
            0.25,
            0.25,
        ),
    ]
    grids: list[PixelGrid] = []
    for target, label, roi, v_threshold, h_threshold in candidates:
        grid = _grid_lines_in_roi(dark, roi, target, label, v_threshold, h_threshold)
        if grid:
            grids.append(grid)
    return grids


def _grid_lines_in_roi(
    dark: np.ndarray,
    roi: tuple[float, float, float, float],
    target: str,
    label: str,
    vertical_threshold: float,
    horizontal_threshold: float,
) -> PixelGrid | None:
    image_height, image_width = dark.shape
    x1 = max(0, min(image_width - 1, int(round(roi[0]))))
    y1 = max(0, min(image_height - 1, int(round(roi[1]))))
    x2 = max(x1 + 1, min(image_width, int(round(roi[2]))))
    y2 = max(y1 + 1, min(image_height, int(round(roi[3]))))
    sub = dark[y1:y2, x1:x2]
    roi_height, roi_width = sub.shape
    col_clusters = _clusters(np.flatnonzero(sub.sum(axis=0) > roi_height * vertical_threshold), gap=3)
    row_clusters = _clusters(np.flatnonzero(sub.sum(axis=1) > roi_width * horizontal_threshold), gap=3)
    if len(col_clusters) < 2 or len(row_clusters) < 2:
        return None

    columns = [x1 + _cluster_mid(cluster) for cluster in col_clusters]
    rows = [y1 + _cluster_mid(cluster) for cluster in row_clusters]
    left, right = columns[0], columns[-1]
    top, bottom = rows[0], rows[-1]
    if right <= left or bottom <= top:
        return None
    return PixelGrid(target=target, label=label, bbox=(left, top, right, bottom), columns=columns, rows=rows)


def _grid_to_entities(
    grid: PixelGrid,
    image_width: int,
    image_height: int,
    canvas_height: float,
) -> list[LineEntity]:
    x1, y1, x2, y2 = grid.bbox
    entities: list[LineEntity] = []
    for idx, x in enumerate(_dedupe_sorted(grid.columns)):
        cx = _px_x_to_cad(x, image_width)
        entities.append(
            LineEntity(
                id=f"recon_{grid.target}_v{idx}",
                layer="table",
                x1=cx,
                y1=_px_y_to_cad(y1, image_height, canvas_height),
                x2=cx,
                y2=_px_y_to_cad(y2, image_height, canvas_height),
                group=grid.target,
                tags=[grid.target, "reconstructed", "grid"],
            )
        )
    for idx, y in enumerate(_dedupe_sorted(grid.rows)):
        cy = _px_y_to_cad(y, image_height, canvas_height)
        entities.append(
            LineEntity(
                id=f"recon_{grid.target}_h{idx}",
                layer="table",
                x1=_px_x_to_cad(x1, image_width),
                y1=cy,
                x2=_px_x_to_cad(x2, image_width),
                y2=cy,
                group=grid.target,
                tags=[grid.target, "reconstructed", "grid"],
            )
        )
    return entities


def _table_stub_text(
    grid: PixelGrid,
    image_width: int,
    image_height: int,
    canvas_height: float,
) -> list[TextEntity]:
    values = _parameter_texts() if grid.target == "parameter_table" else _title_block_texts()
    x1, y1, x2, y2 = grid.bbox
    table_width = _px_x_to_cad(x2, image_width) - _px_x_to_cad(x1, image_width)
    table_height = _px_y_to_cad(y1, image_height, canvas_height) - _px_y_to_cad(y2, image_height, canvas_height)
    height = max(1.4, min(2.6, min(table_width, table_height) * 0.045))
    entities: list[TextEntity] = []
    for idx, (x_frac, y_frac, value) in enumerate(values):
        if not value:
            continue
        entities.append(
            TextEntity(
                id=f"recon_{grid.target}_text_{idx}",
                layer="text",
                x=_px_x_to_cad(x1 + (x2 - x1) * x_frac, image_width),
                y=_px_y_to_cad(y1 + (y2 - y1) * y_frac, image_height, canvas_height),
                text=value,
                height=height,
                group=grid.target,
                tags=[grid.target, "reconstructed", "text_stub"],
            )
        )
    return entities


def _parameter_texts() -> list[tuple[float, float, str]]:
    rows = ["齿廓", "齿数 z", "模数 m", "螺旋角 β", "压力角 α", "齿数 z", "精度等级"]
    vals = ["渐开线", "29", "2", "0°", "20°", "58", "7"]
    out = []
    for idx, (left, right) in enumerate(zip(rows, vals, strict=True)):
        y = 0.05 + idx * 0.095
        out.append((0.05, y, left))
        out.append((0.56, y, right))
    return out


def _title_block_texts() -> list[tuple[float, float, str]]:
    return [
        (0.73, 0.20, "合肥工业大学"),
        (0.73, 0.47, "圆柱直齿轮"),
        (0.74, 0.73, "LJT01.01"),
        (0.55, 0.36, "4:5"),
        (0.03, 0.20, "设计"),
        (0.03, 0.38, "制图"),
        (0.03, 0.56, "审核"),
        (0.03, 0.75, "工艺"),
    ]


def _layout_warnings(regions: list[LayoutRegion]) -> list[str]:
    warnings: list[str] = []
    for idx, left in enumerate(regions):
        for right in regions[idx + 1 :]:
            overlap = _bbox_intersection_area(left.bbox, right.bbox)
            if overlap > 1e-6:
                warnings.append(f"{left.target} overlaps {right.target}: area={overlap:.3f}")
    return warnings


def _bbox_intersection_area(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0.0, min(ay2, by2) - max(ay1, by1))
    return width * height


def _px_bbox_to_cad(
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    canvas_height: float,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    return (
        _px_x_to_cad(x1, image_width),
        _px_y_to_cad(y2, image_height, canvas_height),
        _px_x_to_cad(x2, image_width),
        _px_y_to_cad(y1, image_height, canvas_height),
    )


def _px_x_to_cad(x: int | float, image_width: int) -> float:
    return round(float(x) / image_width * CANVAS_WIDTH_MM, 4)


def _px_y_to_cad(y: int | float, image_height: int, canvas_height: float) -> float:
    return round((1.0 - float(y) / image_height) * canvas_height, 4)


def _dedupe_sorted(values: list[int], tolerance: int = 2) -> list[int]:
    if not values:
        return []
    result = [int(values[0])]
    for value in values[1:]:
        if abs(value - result[-1]) > tolerance:
            result.append(int(value))
    return result
