from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image

from .models import ProjectState, TableCellOcr
from .ocr import (
    _paddleocr_available,
    _parse_tesseract_tsv,
    _preprocess_crop,
    _preprocess_paddle_crop,
    _run_paddle_ocr,
    _select_languages,
)
from .reconstruct import PixelGrid, _dedupe_sorted, _detect_table_grids
from .reference import _detect_inner_frame, _preprocess_reference_image, _upload_url_to_path

MAX_TABLE_CELLS = 180


@dataclass(frozen=True)
class TableOcrRun:
    cells: list[TableCellOcr]
    warnings: list[str]


def extract_table_ocr_from_reference(
    project: ProjectState,
    uploads_dir: Path,
    language_hint: Literal["auto", "zh", "en"] = "auto",
    engine_hint: Literal["auto", "edocr2", "paddle", "tesseract"] = "auto",
) -> TableOcrRun:
    if not project.source_image:
        raise ValueError("Upload a PDF or image before running table OCR.")

    image_path = _upload_url_to_path(project.source_image, uploads_dir)
    if not image_path.exists():
        raise FileNotFoundError("Reference image not found")

    processed = _preprocess_reference_image(image_path)
    frame = _detect_inner_frame(processed.dark)
    if frame is None:
        raise ValueError("Could not detect drawing frame.")

    grids = _detect_table_grids(processed.dark, frame)
    if not grids:
        raise ValueError("Could not detect table grids.")

    language = "en" if language_hint == "en" else "zh"
    engine = _select_table_engine(language, engine_hint)
    warnings: list[str] = []
    tesseract = shutil.which("tesseract")
    if engine == "paddle" and not _paddleocr_available():
        warnings.append("PaddleOCR runtime is not installed; falling back to Tesseract.")
        engine = "tesseract"
    if engine == "tesseract" and tesseract is None:
        raise ValueError("No table OCR engine is available; install PaddleOCR or Tesseract.")

    rgb = processed.image.convert("RGB")
    image_width, image_height = rgb.size
    cells = _ocr_grids(rgb, image_width, image_height, grids, language, engine, tesseract, warnings)
    return TableOcrRun(cells=cells, warnings=warnings)


def _select_table_engine(language: str, engine_hint: str) -> Literal["paddle", "tesseract"]:
    if engine_hint == "paddle":
        return "paddle"
    if engine_hint == "tesseract":
        return "tesseract"
    return "paddle" if language == "zh" and _paddleocr_available() else "tesseract"


def _ocr_grids(
    image: Image.Image,
    image_width: int,
    image_height: int,
    grids: list[PixelGrid],
    language: Literal["zh", "en"],
    engine: Literal["paddle", "tesseract"],
    tesseract: str | None,
    warnings: list[str],
) -> list[TableCellOcr]:
    cells: list[TableCellOcr] = []
    remaining = MAX_TABLE_CELLS
    for grid in grids:
        columns = _dedupe_sorted(grid.columns)
        rows = _dedupe_sorted(grid.rows)
        if len(columns) < 2 or len(rows) < 2:
            continue
        for row_idx, (y1, y2) in enumerate(zip(rows, rows[1:], strict=False)):
            for col_idx, (x1, x2) in enumerate(zip(columns, columns[1:], strict=False)):
                if remaining <= 0:
                    warnings.append(f"Table OCR stopped after {MAX_TABLE_CELLS} cells.")
                    return cells
                remaining -= 1
                text, confidence, used_engine, used_language, source = _ocr_cell(
                    image,
                    (x1, y1, x2, y2),
                    language,
                    engine,
                    tesseract,
                    warnings,
                )
                cells.append(
                    TableCellOcr(
                        target=grid.target,
                        row=row_idx,
                        col=col_idx,
                        text=text,
                        confidence=confidence,
                        x=_ratio(x1, image_width),
                        y=_ratio(y1, image_height),
                        width=max(0.001, _ratio(x2 - x1, image_width)),
                        height=max(0.001, _ratio(y2 - y1, image_height)),
                        engine=used_engine,
                        language=used_language,
                        source=source,
                    )
                )
    return cells


def _ocr_cell(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    language: Literal["zh", "en"],
    engine: Literal["paddle", "tesseract"],
    tesseract: str | None,
    warnings: list[str],
) -> tuple[str, float, str, str, str]:
    crop = _inner_cell_crop(image, bbox)
    if engine == "paddle":
        try:
            text, confidence = _run_paddle_ocr(_preprocess_paddle_crop(crop), language)
            return text, confidence, "paddleocr", language, "table_cell_ocr_v0"
        except Exception as exc:  # noqa: BLE001 - preserve useful fallback behavior
            warnings.append(f"PaddleOCR failed for a table cell: {exc}.")

    if not tesseract:
        return "", 0.0, "none", language, "table_cell_ocr_unavailable"

    languages, lang_warnings = _select_languages(tesseract)
    warnings.extend(lang_warnings)
    text, confidence = _run_tesseract_cell(tesseract, crop, languages, warnings)
    return text, confidence, "tesseract", languages, "table_cell_ocr_tesseract"


def _inner_cell_crop(image: Image.Image, bbox: tuple[int, int, int, int]) -> Image.Image:
    x1, y1, x2, y2 = bbox
    pad_x = max(2, round((x2 - x1) * 0.08))
    pad_y = max(2, round((y2 - y1) * 0.12))
    return image.crop((min(x2 - 1, x1 + pad_x), min(y2 - 1, y1 + pad_y), max(x1 + 1, x2 - pad_x), max(y1 + 1, y2 - pad_y)))


def _run_tesseract_cell(tesseract: str, crop: Image.Image, languages: str, warnings: list[str]) -> tuple[str, float]:
    prepared = _preprocess_crop(crop.convert("L"))
    with tempfile.TemporaryDirectory(prefix="vibecad_table_ocr_") as tmp:
        crop_path = Path(tmp) / "cell.png"
        prepared.save(crop_path)
        try:
            result = subprocess.run(
                [tesseract, str(crop_path), "stdout", "-l", languages, "--psm", "6", "tsv"],
                check=False,
                capture_output=True,
                text=True,
                timeout=20,
            )
        except subprocess.TimeoutExpired:
            warnings.append("Tesseract timed out for a table cell.")
            return "", 0.0
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
        warnings.append(f"Tesseract failed for a table cell: {detail}")
        return "", 0.0
    return _parse_tesseract_tsv(result.stdout)


def _ratio(value: int | float, total: int) -> float:
    return round(max(0.0, min(1.0, float(value) / max(total, 1))), 4)
