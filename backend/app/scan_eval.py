from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from .cad_layers import OUTLINE, REFERENCE_TRACE, TITLE_BLOCK, canonical_layer_name
from .models import ArcEntity, CircleEntity, DrawingIR, Entity, LineEntity, PolylineEntity, RectangleEntity
from .structure_eval import StructureEvalReport, TargetEval

SCAN_TRACE_GROUPS = {"reference_trace", "editable_linework"}
TABLE_GROUPS = {"title_block", "parameter_table"}
PROMOTED_GROUP = "promoted_geometry"
QUALITY_RASTER_MAX_SIDE = 1200
SHORT_FRAGMENT_MM = 1.2
MICRO_FRAGMENT_MM = 0.35
DISPLAY_MIN_STROKE_BY_LAYER = {
    "sheet": 0.35,
    "reference_trace": 0.16,
    "editable_linework": 0.38,
    "table": 0.18,
    "promoted_geometry": 0.42,
    REFERENCE_TRACE: 0.13,
    OUTLINE: 0.50,
    TITLE_BLOCK: 0.25,
}


def evaluate_scan_structure(ir: DrawingIR, source_image_path: Path | None = None) -> StructureEvalReport:
    """Evaluate scan-to-CAD output with quality-oriented raster/vectorizer goals."""
    targets = [
        _evaluate_scan_trace(ir),
        _evaluate_scan_visual_match(ir, source_image_path),
        _evaluate_scan_primitive_quality(ir),
        _evaluate_scan_noise(ir),
        _evaluate_scan_tables(ir),
        _evaluate_scan_lineweights(ir),
    ]
    overall_score = round(sum(target.score for target in targets) / len(targets), 3)
    return StructureEvalReport(
        overall_score=overall_score,
        passed=all(target.passed for target in targets),
        targets=targets,
    )


def _target(name: str, checks: dict[str, bool], evidence: dict[str, int | float | str | list[str]]) -> TargetEval:
    missing = [label for label, passed in checks.items() if not passed]
    score = round(sum(1 for passed in checks.values() if passed) / max(len(checks), 1), 3)
    return TargetEval(
        name=name,
        score=score,
        passed=not missing,
        checks=checks,
        missing=missing,
        evidence=evidence,
    )


def _by_group(ir: DrawingIR, group: str) -> list[Entity]:
    return [entity for entity in ir.entities if entity.group == group or group in entity.tags]


def _by_layer(ir: DrawingIR, layer: str) -> list[Entity]:
    canonical = canonical_layer_name(layer)
    return [entity for entity in ir.entities if canonical_layer_name(entity.layer) == canonical]


def _trace_entities(ir: DrawingIR) -> list[Entity]:
    return [entity for entity in ir.entities if entity.group in SCAN_TRACE_GROUPS or entity.layer in SCAN_TRACE_GROUPS]


def _table_entities(ir: DrawingIR) -> list[Entity]:
    return [entity for entity in ir.entities if entity.group in TABLE_GROUPS or entity.layer == "table"]


def _table_grid_entities(ir: DrawingIR) -> list[Entity]:
    return [
        entity
        for entity in _table_entities(ir)
        if isinstance(entity, LineEntity | RectangleEntity) or "grid" in entity.tags
    ]


def _effective_stroke(entity: Entity) -> float:
    stroke = entity.stroke_width if entity.stroke_width is not None else 0.0
    if entity.group == PROMOTED_GROUP:
        return max(stroke, DISPLAY_MIN_STROKE_BY_LAYER[PROMOTED_GROUP])
    return max(stroke, DISPLAY_MIN_STROKE_BY_LAYER.get(entity.layer, stroke))


def _stroke_values(entities: list[Entity]) -> list[float]:
    return [_effective_stroke(entity) for entity in entities]


def _avg(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _max(values: list[float]) -> float:
    return round(max(values), 4) if values else 0.0


def _count_type(entities: list[Entity], entity_type: type[Entity]) -> int:
    return sum(1 for entity in entities if isinstance(entity, entity_type))


def _entity_length(entity: Entity) -> float:
    if isinstance(entity, LineEntity):
        return math.hypot(entity.x2 - entity.x1, entity.y2 - entity.y1)
    if isinstance(entity, PolylineEntity):
        if len(entity.points) < 2:
            return 0.0
        points = np.asarray(entity.points, dtype=float)
        length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        if entity.closed:
            length += float(np.linalg.norm(points[0] - points[-1]))
        return length
    if isinstance(entity, CircleEntity):
        return 2 * math.pi * entity.r
    if isinstance(entity, ArcEntity):
        angle = (entity.end_angle - entity.start_angle) % 360
        return 2 * math.pi * entity.r * angle / 360
    if isinstance(entity, RectangleEntity):
        return 2 * (entity.width + entity.height)
    return 0.0


def _evaluate_scan_trace(ir: DrawingIR) -> TargetEval:
    reference = _by_group(ir, "reference_trace")
    editable = _by_group(ir, "editable_linework")
    sheet = _by_group(ir, "sheet")
    layer_names = {layer.name for layer in ir.layers}
    entity_layer_names = {entity.layer for entity in ir.entities}
    dangling_layers = sorted(entity_layer_names - layer_names)
    trace_count = len(reference) + len(editable)
    checks = {
        "uses millimeter units": ir.units == "mm",
        "has sheet border": len(sheet) >= 1,
        "has drawing frame": len(sheet) >= 2 or any(entity.id == "scan_sheet_border" for entity in sheet),
        "has reference trace layer": len(reference) >= 20,
        "has editable linework layer": len(editable) >= 20,
        "has substantial linework": trace_count >= 100,
        "entity count is bounded": 10 <= len(ir.entities) <= 20000,
        "declares used layers": not dangling_layers,
    }
    return _target(
        "scan_trace",
        checks,
        {
            "sheet_count": len(sheet),
            "reference_trace_count": len(reference),
            "editable_linework_count": len(editable),
            "trace_count": trace_count,
            "entity_count": len(ir.entities),
            "dangling_layers": dangling_layers[:8],
        },
    )


def _evaluate_scan_visual_match(ir: DrawingIR, source_image_path: Path | None) -> TargetEval:
    if source_image_path is None or not source_image_path.exists():
        return _target(
            "scan_visual_match",
            {
                "has source image": False,
                "has source line mask": False,
                "has cad line mask": False,
                "edge f-score is acceptable": False,
                "edge precision is acceptable": False,
                "edge recall is acceptable": False,
                "ink density is comparable": False,
            },
            {"source": "missing"},
        )

    image = cv2.imread(str(source_image_path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.size == 0:
        return _target(
            "scan_visual_match",
            {
                "has source image": False,
                "has source line mask": False,
                "has cad line mask": False,
                "edge f-score is acceptable": False,
                "edge precision is acceptable": False,
                "edge recall is acceptable": False,
                "ink density is comparable": False,
            },
            {"source": "unreadable"},
        )

    image = _resize_for_quality(image)
    source_mask = _source_line_mask(image)
    cad_mask = _rasterize_ir_to_mask(ir, image.shape[1], image.shape[0])
    metrics = _mask_similarity(source_mask, cad_mask)
    checks = {
        "has source image": True,
        "has source line mask": metrics["source_pixels"] > 100,
        "has cad line mask": metrics["cad_pixels"] > 100,
        "edge f-score is acceptable": metrics["edge_fscore"] >= 0.42,
        "edge precision is acceptable": metrics["edge_precision"] >= 0.38,
        "edge recall is acceptable": metrics["edge_recall"] >= 0.38,
        "ink density is comparable": 0.35 <= metrics["ink_density_ratio"] <= 3.5,
    }
    return _target(
        "scan_visual_match",
        checks,
        {
            "edge_fscore": metrics["edge_fscore"],
            "edge_precision": metrics["edge_precision"],
            "edge_recall": metrics["edge_recall"],
            "ink_density_ratio": metrics["ink_density_ratio"],
            "source_pixels": metrics["source_pixels"],
            "cad_pixels": metrics["cad_pixels"],
        },
    )


def _evaluate_scan_primitive_quality(ir: DrawingIR) -> TargetEval:
    editable = _by_group(ir, "editable_linework")
    reference = _by_group(ir, "reference_trace")
    promoted = _by_group(ir, PROMOTED_GROUP)
    promoted_layer = [entity for entity in promoted if canonical_layer_name(entity.layer) == OUTLINE]
    promoted_lines = _count_type(promoted, LineEntity)
    promoted_circles = _count_type(promoted, CircleEntity)
    promoted_arrows = sum(1 for entity in promoted if "dimension_arrow" in entity.tags or "arrowhead" in entity.tags)
    promoted_length = sum(_entity_length(entity) for entity in promoted)
    editable_length = sum(_entity_length(entity) for entity in editable)
    promoted_count_ratio = len(promoted) / max(len(editable), 1)
    promoted_length_ratio = promoted_length / max(editable_length, 1e-6)
    checks = {
        "has editable source": len(editable) > 0,
        "has promoted primitive layer": len(promoted) > 0,
        "promoted straight segments": promoted_lines >= 10,
        "promoted circular features": promoted_circles > 0,
        "promoted dimension arrows": promoted_arrows >= 2,
        "primitive coverage is useful": promoted_count_ratio >= 0.03 or promoted_length_ratio >= 0.08,
        "promotion keeps raw trace": len(reference) > 0 and len(editable) > 0,
        "promoted layer is isolated": len(promoted) == len(promoted_layer),
    }
    return _target(
        "scan_primitive_quality",
        checks,
        {
            "editable_source_count": len(editable),
            "promoted_count": len(promoted),
            "promoted_line_count": promoted_lines,
            "promoted_circle_count": promoted_circles,
            "promoted_arrow_count": promoted_arrows,
            "promoted_count_ratio": round(promoted_count_ratio, 4),
            "promoted_length_ratio": round(promoted_length_ratio, 4),
        },
    )


def _evaluate_scan_noise(ir: DrawingIR) -> TargetEval:
    trace = [entity for entity in _trace_entities(ir) if isinstance(entity, PolylineEntity | LineEntity)]
    lengths = [_entity_length(entity) for entity in trace]
    short_count = sum(1 for length in lengths if length < SHORT_FRAGMENT_MM)
    micro_count = sum(1 for length in lengths if length < MICRO_FRAGMENT_MM)
    zero_count = sum(1 for length in lengths if length <= 1e-6)
    total_length = sum(lengths)
    short_ratio = short_count / max(len(lengths), 1)
    micro_ratio = micro_count / max(len(lengths), 1)
    avg_length = total_length / max(len(lengths), 1)
    checks = {
        "has measurable trace": bool(trace),
        "fragment count is bounded": len(trace) <= 6000,
        "short fragment ratio is controlled": short_ratio <= 0.45,
        "micro fragment ratio is controlled": micro_ratio <= 0.22,
        "average trace length is useful": avg_length >= 1.2,
        "no zero-length fragments": zero_count == 0,
    }
    return _target(
        "scan_noise",
        checks,
        {
            "trace_fragment_count": len(trace),
            "short_fragment_count": short_count,
            "micro_fragment_count": micro_count,
            "zero_length_count": zero_count,
            "short_fragment_ratio": round(short_ratio, 4),
            "micro_fragment_ratio": round(micro_ratio, 4),
            "average_trace_length_mm": round(avg_length, 4),
        },
    )


def _evaluate_scan_tables(ir: DrawingIR) -> TargetEval:
    table_entities = _table_entities(ir)
    title_count = len(_by_group(ir, "title_block"))
    parameter_count = len(_by_group(ir, "parameter_table"))
    grid_count = sum(
        1
        for entity in table_entities
        if isinstance(entity, LineEntity | RectangleEntity) or "grid" in entity.tags
    )
    text_stub_count = sum(1 for entity in table_entities if "text_stub" in entity.tags)
    checks = {
        "has table/title regions": title_count > 0 or parameter_count > 0,
        "has reconstructed grid": grid_count >= 8,
        "removed text-stub clutter": text_stub_count == 0,
        "table layer is not empty": len(table_entities) >= 8,
    }
    return _target(
        "scan_tables",
        checks,
        {
            "title_block_count": title_count,
            "parameter_table_count": parameter_count,
            "table_entity_count": len(table_entities),
            "grid_entity_count": grid_count,
            "text_stub_count": text_stub_count,
        },
    )


def _evaluate_scan_lineweights(ir: DrawingIR) -> TargetEval:
    reference = _by_group(ir, "reference_trace")
    editable = _by_group(ir, "editable_linework")
    sheet = _by_group(ir, "sheet")
    tables = _table_grid_entities(ir)
    promoted = _by_group(ir, PROMOTED_GROUP)
    reference_avg = _avg(_stroke_values(reference))
    editable_avg = _avg(_stroke_values(editable))
    table_avg = _avg(_stroke_values(tables))
    promoted_avg = _avg(_stroke_values(promoted))
    sheet_max = _max(_stroke_values(sheet))
    checks = {
        "reference trace is thin": bool(reference) and reference_avg <= 0.2,
        "editable linework is emphasized": bool(editable) and editable_avg >= 0.28 and editable_avg > reference_avg,
        "sheet/frame is heavier": bool(sheet) and sheet_max >= 0.25,
        "table grid is readable": not tables or 0.14 <= table_avg <= 0.24,
        "promoted geometry is prominent": not promoted or promoted_avg >= editable_avg,
    }
    return _target(
        "scan_lineweights",
        checks,
        {
            "reference_avg_mm": reference_avg,
            "editable_avg_mm": editable_avg,
            "table_avg_mm": table_avg,
            "promoted_avg_mm": promoted_avg,
            "sheet_max_mm": sheet_max,
        },
    )


def _resize_for_quality(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    max_side = max(height, width)
    if max_side <= QUALITY_RASTER_MAX_SIDE:
        return image
    scale = QUALITY_RASTER_MAX_SIDE / max_side
    return cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)


def _source_line_mask(image: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    kernel = np.ones((2, 2), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def _mask_similarity(source_mask: np.ndarray, cad_mask: np.ndarray) -> dict[str, float | int]:
    source = source_mask > 0
    cad = cad_mask > 0
    source_pixels = int(source.sum())
    cad_pixels = int(cad.sum())
    if source_pixels == 0 or cad_pixels == 0:
        return {
            "edge_fscore": 0.0,
            "edge_precision": 0.0,
            "edge_recall": 0.0,
            "ink_density_ratio": 0.0,
            "source_pixels": source_pixels,
            "cad_pixels": cad_pixels,
        }

    tolerance = max(2, round(max(source_mask.shape) * 0.003))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tolerance * 2 + 1, tolerance * 2 + 1))
    source_dilated = cv2.dilate(source.astype(np.uint8), kernel) > 0
    cad_dilated = cv2.dilate(cad.astype(np.uint8), kernel) > 0
    precision = float(np.logical_and(cad, source_dilated).sum() / cad_pixels)
    recall = float(np.logical_and(source, cad_dilated).sum() / source_pixels)
    fscore = 2 * precision * recall / max(precision + recall, 1e-9)
    density_ratio = cad_pixels / max(source_pixels, 1)
    return {
        "edge_fscore": round(fscore, 4),
        "edge_precision": round(precision, 4),
        "edge_recall": round(recall, 4),
        "ink_density_ratio": round(density_ratio, 4),
        "source_pixels": source_pixels,
        "cad_pixels": cad_pixels,
    }


def _rasterize_ir_to_mask(ir: DrawingIR, width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    min_x, min_y, max_x, max_y = _canvas_bounds(ir)
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)
    px_per_mm = (width / span_x + height / span_y) / 2

    def point(x: float, y: float) -> tuple[int, int]:
        px = round((x - min_x) / span_x * (width - 1))
        py = round((max_y - y) / span_y * (height - 1))
        return int(np.clip(px, 0, width - 1)), int(np.clip(py, 0, height - 1))

    for entity in ir.entities:
        thickness = max(1, min(8, round(_effective_stroke(entity) * px_per_mm)))
        if isinstance(entity, LineEntity):
            cv2.line(mask, point(entity.x1, entity.y1), point(entity.x2, entity.y2), 255, thickness, cv2.LINE_AA)
        elif isinstance(entity, PolylineEntity) and entity.points:
            points = np.asarray([point(x, y) for x, y in entity.points], dtype=np.int32)
            cv2.polylines(mask, [points], entity.closed, 255, thickness, cv2.LINE_AA)
        elif isinstance(entity, CircleEntity):
            radius = max(1, round(entity.r * px_per_mm))
            cv2.circle(mask, point(entity.cx, entity.cy), radius, 255, thickness, cv2.LINE_AA)
        elif isinstance(entity, ArcEntity):
            radius = max(1, round(entity.r * px_per_mm))
            center = point(entity.cx, entity.cy)
            cv2.ellipse(mask, center, (radius, radius), 0, -entity.end_angle, -entity.start_angle, 255, thickness, cv2.LINE_AA)
        elif isinstance(entity, RectangleEntity):
            cv2.rectangle(mask, point(entity.x, entity.y + entity.height), point(entity.x + entity.width, entity.y), 255, thickness)
    return mask


def _canvas_bounds(ir: DrawingIR) -> tuple[float, float, float, float]:
    sheet_rectangles = [
        entity
        for entity in ir.entities
        if isinstance(entity, RectangleEntity) and (entity.id == "scan_sheet_border" or entity.group == "sheet")
    ]
    if sheet_rectangles:
        sheet = max(sheet_rectangles, key=lambda entity: entity.width * entity.height)
        return sheet.x, sheet.y, sheet.x + sheet.width, sheet.y + sheet.height

    xs: list[float] = []
    ys: list[float] = []
    for entity in ir.entities:
        if isinstance(entity, LineEntity):
            xs.extend([entity.x1, entity.x2])
            ys.extend([entity.y1, entity.y2])
        elif isinstance(entity, PolylineEntity):
            xs.extend(point[0] for point in entity.points)
            ys.extend(point[1] for point in entity.points)
        elif isinstance(entity, CircleEntity | ArcEntity):
            xs.extend([entity.cx - entity.r, entity.cx + entity.r])
            ys.extend([entity.cy - entity.r, entity.cy + entity.r])
        elif isinstance(entity, RectangleEntity):
            xs.extend([entity.x, entity.x + entity.width])
            ys.extend([entity.y, entity.y + entity.height])
    if not xs or not ys:
        return 0, 0, 100, 100
    return min(xs), min(ys), max(xs), max(ys)
