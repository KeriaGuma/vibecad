from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .models import DrawingIR, Entity, Layer, LineEntity, PolylineEntity, ProjectState, RectangleEntity
from .reconstruct import CANVAS_WIDTH_MM, _px_x_to_cad, _px_y_to_cad, reconstruct_tables_from_reference
from .reference import _detect_inner_frame, _preprocess_reference_image, _upload_url_to_path
from .section_cv import reconstruct_section_from_reference
from .svg_dxf import svg_geometry_to_polylines

MAX_HOUGH_FALLBACK_LINES = 1200
MAX_REFERENCE_TRACE_ENTITIES = 1800
MAX_EDITABLE_LINEWORK_ENTITIES = 2600
MIN_CONFIDENT_SECTION_ENTITIES = 8
TABLE_TRACE_MASK_PADDING_MM = 1.5
SHEET_BORDER_STROKE_MM = 0.35
DRAWING_FRAME_STROKE_MM = 0.25
REFERENCE_TRACE_STROKE_MM = 0.10
EDITABLE_LINEWORK_STROKE_MM = 0.28
HOUGH_FALLBACK_STROKE_MM = 0.22
TABLE_GRID_STROKE_MM = 0.18


@dataclass(frozen=True)
class ScanCadReconstruction:
    ir: DrawingIR
    entity_count: int
    trace_count: int
    structured_counts: dict[str, int]
    warnings: list[str]


def reconstruct_scan_cad_from_reference(
    project: ProjectState,
    uploads_dir: Path,
    output_dir: Path | None = None,
) -> ScanCadReconstruction:
    if not project.source_image:
        raise ValueError("Upload a PDF or image before generating CAD from the scan.")
    if project.source_kind == "vector_pdf":
        raise ValueError("Use vector extraction for vector PDFs.")

    image_path = _upload_url_to_path(project.source_image, uploads_dir)
    if not image_path.exists():
        raise FileNotFoundError("Reference image not found")

    processed = _preprocess_reference_image(image_path)
    image_height, image_width = processed.dark.shape
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Reference image is empty.")
    canvas_height = CANVAS_WIDTH_MM * image_height / image_width
    frame = _detect_inner_frame(processed.dark)
    warnings: list[str] = []
    if frame is None:
        warnings.append("Could not detect drawing frame; tracing full page only.")

    ir = _base_scan_ir(image_width, image_height, canvas_height, frame)
    vectorizer_dir = output_dir or image_path.parent
    trace_entities = _external_vectorizer_entities(processed.image, vectorizer_dir, warnings)
    if not any(entity.group == "editable_linework" for entity in trace_entities):
        fallback = _hough_fallback_entities(processed.image.convert("L"), image_width, image_height, canvas_height)
        trace_entities.extend(fallback)
        warnings.append(f"Editable centerline fallback used: {len(fallback)} Hough line entities.")
    ir.entities.extend(trace_entities)
    structured_counts: dict[str, int] = {
        "reference_trace": sum(1 for entity in trace_entities if entity.group == "reference_trace"),
        "editable_linework": sum(1 for entity in trace_entities if entity.group == "editable_linework"),
    }

    tables_project = project.model_copy(update={"ir": ir}, deep=True)
    try:
        tables = reconstruct_tables_from_reference(tables_project, uploads_dir)
        table_entities = _scan_table_entities(
            [entity for entity in tables.ir.entities if entity.group in {"title_block", "parameter_table"}]
        )
        masked_trace_entities = _mask_trace_entities_in_regions(trace_entities, [region.bbox for region in tables.regions])
        if len(masked_trace_entities) != len(trace_entities):
            removed = len(trace_entities) - len(masked_trace_entities)
            trace_entities = masked_trace_entities
            trace_ids = {trace.id for trace in trace_entities}
            ir.entities = [
                entity
                for entity in ir.entities
                if entity.group not in {"reference_trace", "editable_linework"} or entity.id in trace_ids
            ]
            structured_counts["reference_trace"] = sum(1 for entity in trace_entities if entity.group == "reference_trace")
            structured_counts["editable_linework"] = sum(1 for entity in trace_entities if entity.group == "editable_linework")
            warnings.append(f"Masked {removed} trace entities inside reconstructed table regions.")
        ir.entities.extend(table_entities)
        structured_counts["tables"] = len(table_entities)
        if tables.warnings:
            warnings.extend(tables.warnings)
    except ValueError as exc:
        warnings.append(f"Table reconstruction skipped: {exc}")

    section_project = project.model_copy(update={"ir": ir}, deep=True)
    try:
        section = reconstruct_section_from_reference(section_project, uploads_dir)
        section_entities = [entity for entity in section.ir.entities if entity.group == "section_view"]
        if _accept_section_reconstruction(len(section_entities), section.warnings):
            ir = section.ir
            structured_counts["section_view"] = len(section_entities)
            if section.warnings:
                warnings.extend(section.warnings)
        else:
            structured_counts["section_view"] = 0
            warnings.append(
                f"Section reconstruction skipped: low-confidence CV result ({len(section_entities)} entities)."
            )
            warnings.extend(f"Skipped section detail: {warning}" for warning in section.warnings)
    except ValueError as exc:
        structured_counts["section_view"] = 0
        warnings.append(f"Section reconstruction skipped: {exc}")

    ir.notes = [
        *ir.notes,
        "Generated from scanned reference: vtracer reference layer plus autotrace centerline layer when available.",
    ]
    return ScanCadReconstruction(
        ir=ir,
        entity_count=len(ir.entities),
        trace_count=len(trace_entities),
        structured_counts=structured_counts,
        warnings=warnings,
    )


def _base_scan_ir(
    image_width: int,
    image_height: int,
    canvas_height: float,
    frame: tuple[int, int, int, int] | None,
) -> DrawingIR:
    entities = [
        RectangleEntity(
            id="scan_sheet_border",
            layer="sheet",
            x=0,
            y=0,
            width=CANVAS_WIDTH_MM,
            height=canvas_height,
            group="sheet",
            tags=["sheet", "scan_cad"],
            stroke_width=SHEET_BORDER_STROKE_MM,
        )
    ]
    if frame is not None:
        fx1, fy1, fx2, fy2 = frame
        x1 = _px_x_to_cad(fx1, image_width)
        y1 = _px_y_to_cad(fy2, image_height, canvas_height)
        x2 = _px_x_to_cad(fx2, image_width)
        y2 = _px_y_to_cad(fy1, image_height, canvas_height)
        entities.append(
            RectangleEntity(
                id="scan_drawing_frame",
                layer="sheet",
                x=x1,
                y=y1,
                width=round(max(0.001, x2 - x1), 4),
                height=round(max(0.001, y2 - y1), 4),
                group="sheet",
                tags=["drawing_frame", "scan_cad"],
                stroke_width=DRAWING_FRAME_STROKE_MM,
            )
        )
    return DrawingIR(
        units="mm",
        layers=[
            Layer(name="sheet", color="gray"),
            Layer(name="reference_trace", color="gray"),
            Layer(name="editable_linework", color="white"),
            Layer(name="geometry", color="white"),
            Layer(name="hatch", color="white"),
            Layer(name="table", color="white"),
            Layer(name="text", color="white"),
            Layer(name="dimensions", color="white"),
        ],
        entities=entities,
        notes=["Scan CAD baseline: generated from external raster vectorizers."],
    )


def _external_vectorizer_entities(
    image: Image.Image,
    output_dir: Path,
    warnings: list[str],
) -> list[PolylineEntity]:
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_png = output_dir / "scan_normalized_binary.png"
    normalized_pbm = output_dir / "scan_normalized_binary.pbm"
    image.save(normalized_png)
    image.convert("1").save(normalized_pbm)
    entities: list[PolylineEntity] = []
    entities.extend(_run_vtracer_entities(normalized_png, output_dir, warnings))
    entities.extend(_run_autotrace_entities(normalized_png, output_dir, warnings))
    return entities


def _accept_section_reconstruction(section_entity_count: int, warnings: list[str]) -> bool:
    if section_entity_count < MIN_CONFIDENT_SECTION_ENTITIES:
        return False
    warning_text = "\n".join(warnings)
    return "Low section primitive count" not in warning_text and "Low hatch line count" not in warning_text


def _scan_table_entities(entities: list[Entity]) -> list[Entity]:
    clean: list[Entity] = []
    for entity in entities:
        if "text_stub" in entity.tags:
            continue
        if "grid" in entity.tags:
            clean.append(entity.model_copy(update={"stroke_width": TABLE_GRID_STROKE_MM}))
            continue
        clean.append(entity)
    return clean


def _mask_trace_entities_in_regions(
    entities: list[Entity],
    regions: list[tuple[float, float, float, float]],
) -> list[Entity]:
    if not regions:
        return entities
    masks = [_pad_bbox(region, TABLE_TRACE_MASK_PADDING_MM) for region in regions]
    kept: list[Entity] = []
    for entity in entities:
        if entity.group not in {"reference_trace", "editable_linework"}:
            kept.append(entity)
            continue
        bbox = _entity_bbox(entity)
        if bbox is None or not _bbox_hits_any_mask(bbox, masks):
            kept.append(entity)
    return kept


def _bbox_hits_any_mask(
    bbox: tuple[float, float, float, float],
    masks: list[tuple[float, float, float, float]],
) -> bool:
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    for mask in masks:
        if mask[0] <= cx <= mask[2] and mask[1] <= cy <= mask[3]:
            return True
        if _bbox_overlap_ratio(bbox, mask) >= 0.35:
            return True
    return False


def _entity_bbox(entity: Entity) -> tuple[float, float, float, float] | None:
    if isinstance(entity, LineEntity):
        return (
            min(entity.x1, entity.x2),
            min(entity.y1, entity.y2),
            max(entity.x1, entity.x2),
            max(entity.y1, entity.y2),
        )
    if isinstance(entity, PolylineEntity):
        if not entity.points:
            return None
        xs = [point[0] for point in entity.points]
        ys = [point[1] for point in entity.points]
        return min(xs), min(ys), max(xs), max(ys)
    return None


def _pad_bbox(
    bbox: tuple[float, float, float, float],
    padding: float,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    return x1 - padding, y1 - padding, x2 + padding, y2 + padding


def _bbox_overlap_ratio(
    bbox: tuple[float, float, float, float],
    mask: tuple[float, float, float, float],
) -> float:
    width = max(0.0, min(bbox[2], mask[2]) - max(bbox[0], mask[0]))
    height = max(0.0, min(bbox[3], mask[3]) - max(bbox[1], mask[1]))
    area = max((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), 1e-6)
    return width * height / area


def _run_vtracer_entities(image_path: Path, output_dir: Path, warnings: list[str]) -> list[PolylineEntity]:
    try:
        import vtracer
    except ImportError:
        warnings.append("vtracer not installed; skipped reference_trace layer.")
        return []

    svg_path = output_dir / "reference_trace.vtracer.svg"
    try:
        vtracer.convert_image_to_svg_py(
            str(image_path),
            str(svg_path),
            colormode="binary",
            hierarchical="stacked",
            mode="spline",
            filter_speckle=8,
            color_precision=6,
            layer_difference=16,
            corner_threshold=60,
            length_threshold=4.0,
            max_iterations=10,
            splice_threshold=45,
            path_precision=3,
        )
        return svg_geometry_to_polylines(
            svg_path,
            layer="reference_trace",
            group="reference_trace",
            id_prefix="ref_trace",
            tags=["vtracer", "reference"],
            stroke_width=REFERENCE_TRACE_STROKE_MM,
            max_entities=MAX_REFERENCE_TRACE_ENTITIES,
            target_width=CANVAS_WIDTH_MM,
        )
    except Exception as exc:  # noqa: BLE001 - keep scan CAD generation usable
        warnings.append(f"vtracer reference_trace failed: {exc}")
        return []


def _run_autotrace_entities(image_path: Path, output_dir: Path, warnings: list[str]) -> list[PolylineEntity]:
    autotrace = shutil.which("autotrace")
    if not autotrace:
        warnings.append("autotrace not installed; skipped editable_linework centerline layer.")
        return []

    svg_path = output_dir / "editable_linework.autotrace.svg"
    command = [
        autotrace,
        "--centerline",
        "--output-format=svg",
        f"--output-file={svg_path}",
        str(image_path),
    ]
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=90)
    except subprocess.TimeoutExpired:
        warnings.append("autotrace centerline timed out; skipped editable_linework layer.")
        return []

    if result.returncode != 0 or not svg_path.exists():
        detail = (result.stderr or result.stdout or "unknown error").strip()
        warnings.append(f"autotrace centerline failed: {detail}")
        return []

    try:
        return svg_geometry_to_polylines(
            svg_path,
            layer="editable_linework",
            group="editable_linework",
            id_prefix="editable_line",
            tags=["autotrace", "centerline"],
            stroke_width=EDITABLE_LINEWORK_STROKE_MM,
            max_entities=MAX_EDITABLE_LINEWORK_ENTITIES,
            target_width=CANVAS_WIDTH_MM,
        )
    except Exception as exc:  # noqa: BLE001 - keep scan CAD generation usable
        warnings.append(f"autotrace SVG import failed: {exc}")
        return []


def _hough_fallback_entities(
    image,
    image_width: int,
    image_height: int,
    canvas_height: float,
) -> list[LineEntity]:
    binary = np.asarray(image)
    if binary.size == 0:
        return []

    ink = 255 - binary
    edges = cv2.Canny(ink, 50, 150, apertureSize=3)
    min_length = max(18, min(image_width, image_height) // 70)
    threshold = max(28, min(image_width, image_height) // 90)
    raw_lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=threshold,
        minLineLength=min_length,
        maxLineGap=max(6, min(image_width, image_height) // 220),
    )
    if raw_lines is None:
        return []

    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    for x1, y1, x2, y2 in raw_lines[:, 0, :]:
        length = float(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
        if length < min_length:
            continue
        candidates.append((length, (int(x1), int(y1), int(x2), int(y2))))
    candidates.sort(reverse=True, key=lambda item: item[0])

    entities: list[LineEntity] = []
    seen: set[tuple[int, int, int, int]] = set()
    for idx, (_, (x1, y1, x2, y2)) in enumerate(candidates[:MAX_HOUGH_FALLBACK_LINES]):
        key = _quantized_line_key(x1, y1, x2, y2)
        if key in seen:
            continue
        seen.add(key)
        entities.append(
            LineEntity(
                id=f"editable_hough_{idx:05d}",
                layer="editable_linework",
                x1=_px_x_to_cad(x1, image_width),
                y1=_px_y_to_cad(y1, image_height, canvas_height),
                x2=_px_x_to_cad(x2, image_width),
                y2=_px_y_to_cad(y2, image_height, canvas_height),
                group="editable_linework",
                tags=["editable_linework", "hough_fallback"],
                stroke_width=HOUGH_FALLBACK_STROKE_MM,
            )
        )
    return entities


def _quantized_line_key(x1: int, y1: int, x2: int, y2: int) -> tuple[int, int, int, int]:
    a = (round(x1 / 4), round(y1 / 4))
    b = (round(x2 / 4), round(y2 / 4))
    if b < a:
        a, b = b, a
    return a[0], a[1], b[0], b[1]
