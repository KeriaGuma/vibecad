from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .cad_layers import DIMENSION
from .models import DrawingIR, Layer, LineEntity, ProjectState
from .promote import PROMOTED_GROUP
from .reconstruct import CANVAS_WIDTH_MM, _px_x_to_cad, _px_y_to_cad
from .reference import _preprocess_reference_image, _upload_url_to_path

TEMPLATE_ARROW_TAG = "template_arrow"
TEMPLATE_ARROW_LAYER = DIMENSION
TEMPLATE_ARROW_STROKE_MM = 0.25
MATCH_THRESHOLD = 0.76
MAX_TEMPLATE_ARROWHEADS = 80
DETECTION_MAX_SIDE_PX = 2200
TEMPLATE_SIZES = (11, 15, 21)
TEMPLATE_ANGLE_STEP_DEG = 15
ARROW_WING_ANGLE_DEG = 28


@dataclass(frozen=True)
class ArrowTemplateHit:
    tip_x: float
    tip_y: float
    angle_deg: float
    size_px: float
    score: float


@dataclass(frozen=True)
class ArrowTemplateDetection:
    ir: DrawingIR
    detected_count: int
    warnings: list[str]


def detect_arrowheads_from_reference(project: ProjectState, uploads_dir: Path) -> ArrowTemplateDetection:
    if not project.source_image:
        raise ValueError("Upload a PDF or image before detecting arrowheads.")
    image_path = _upload_url_to_path(project.source_image, uploads_dir)
    if not image_path.exists():
        raise FileNotFoundError("Reference image not found")

    processed = _preprocess_reference_image(image_path)
    dark = processed.dark
    if dark.size == 0:
        return ArrowTemplateDetection(ir=project.ir, detected_count=0, warnings=["Reference image is empty."])

    hits = _detect_arrow_template_hits(dark)
    next_ir = project.ir.model_copy(deep=True)
    _ensure_dimension_arrow_layer(next_ir)
    next_ir.entities = [
        entity
        for entity in next_ir.entities
        if TEMPLATE_ARROW_TAG not in entity.tags and not entity.id.startswith("template_arrow_")
    ]

    image_height, image_width = dark.shape
    canvas_height = CANVAS_WIDTH_MM * image_height / image_width
    entities: list[LineEntity] = []
    for idx, hit in enumerate(hits):
        entities.extend(_hit_to_arrow_lines(hit, idx, image_width, image_height, canvas_height))
    next_ir.entities.extend(entities)
    if hits:
        next_ir.notes = [
            *next_ir.notes,
            f"Detected {len(hits)} raster arrowheads with OpenCV template matching.",
        ]

    warnings: list[str] = []
    if not hits:
        warnings.append("No raster arrowheads matched the OpenCV templates above threshold.")
    return ArrowTemplateDetection(ir=next_ir, detected_count=len(hits), warnings=warnings)


def _detect_arrow_template_hits(dark: np.ndarray) -> list[ArrowTemplateHit]:
    resized_dark, scale = _resize_for_detection(dark)
    search = (resized_dark.astype(np.uint8) * 255)
    search = cv2.dilate(search, np.ones((2, 2), dtype=np.uint8), iterations=1)

    raw_hits: list[ArrowTemplateHit] = []
    for size in TEMPLATE_SIZES:
        for angle in range(0, 360, TEMPLATE_ANGLE_STEP_DEG):
            template, mask, tip = _arrow_template(size, angle)
            result = cv2.matchTemplate(search, template, cv2.TM_CCORR_NORMED, mask=mask)
            result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
            ys, xs = np.where(result >= MATCH_THRESHOLD)
            if len(xs) > 40:
                scores = result[ys, xs]
                top = np.argpartition(scores, -40)[-40:]
                xs = xs[top]
                ys = ys[top]
            for x, y in zip(xs, ys, strict=False):
                raw_hits.append(
                    ArrowTemplateHit(
                        tip_x=(float(x) + tip[0]) / scale,
                        tip_y=(float(y) + tip[1]) / scale,
                        angle_deg=float(angle),
                        size_px=size / scale,
                        score=float(result[y, x]),
                    )
                )
    return _nms_hits(raw_hits, MAX_TEMPLATE_ARROWHEADS)


def _resize_for_detection(dark: np.ndarray) -> tuple[np.ndarray, float]:
    height, width = dark.shape
    max_side = max(height, width)
    if max_side <= DETECTION_MAX_SIDE_PX:
        return dark, 1.0
    scale = DETECTION_MAX_SIDE_PX / max_side
    resized = cv2.resize(
        dark.astype(np.uint8),
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized.astype(bool), scale


def _arrow_template(size: int, angle_deg: float) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    canvas = int(size * 3.2) | 1
    template = np.zeros((canvas, canvas), dtype=np.uint8)
    tip = np.asarray([canvas / 2, canvas / 2], dtype=float)
    direction = _unit(angle_deg)
    normal = np.asarray([-direction[1], direction[0]], dtype=float)
    wing = math.radians(ARROW_WING_ANGLE_DEG)
    tail_center = tip - direction * size
    tail_offset = math.tan(wing) * size * 0.72
    tails = (tail_center + normal * tail_offset, tail_center - normal * tail_offset)
    for tail in tails:
        cv2.line(
            template,
            tuple(np.round(tip).astype(int)),
            tuple(np.round(tail).astype(int)),
            255,
            max(1, int(round(size * 0.18))),
            lineType=cv2.LINE_AA,
        )
    mask = cv2.dilate(template, np.ones((3, 3), dtype=np.uint8), iterations=1)
    return template, mask, (float(tip[0]), float(tip[1]))


def _nms_hits(hits: list[ArrowTemplateHit], limit: int) -> list[ArrowTemplateHit]:
    kept: list[ArrowTemplateHit] = []
    for hit in sorted(hits, key=lambda item: item.score, reverse=True):
        min_distance = max(hit.size_px * 0.9, 7.0)
        if any(math.hypot(hit.tip_x - old.tip_x, hit.tip_y - old.tip_y) <= min_distance for old in kept):
            continue
        kept.append(hit)
        if len(kept) >= limit:
            break
    return kept


def _hit_to_arrow_lines(
    hit: ArrowTemplateHit,
    index: int,
    image_width: int,
    image_height: int,
    canvas_height: float,
) -> list[LineEntity]:
    candidate_id = f"template_arrow_{index:05d}"
    direction = _unit(hit.angle_deg)
    normal = np.asarray([-direction[1], direction[0]], dtype=float)
    tip = np.asarray([hit.tip_x, hit.tip_y], dtype=float)
    length = max(hit.size_px, 4.0)
    tail_center = tip - direction * length
    tail_offset = math.tan(math.radians(ARROW_WING_ANGLE_DEG)) * length * 0.72
    tails = (tail_center + normal * tail_offset, tail_center - normal * tail_offset)
    tip_cad = np.asarray(
        [
            _px_x_to_cad(float(tip[0]), image_width),
            _px_y_to_cad(float(tip[1]), image_height, canvas_height),
        ],
        dtype=float,
    )
    tail_center_cad = np.asarray(
        [
            _px_x_to_cad(float(tail_center[0]), image_width),
            _px_y_to_cad(float(tail_center[1]), image_height, canvas_height),
        ],
        dtype=float,
    )
    direction_cad = tip_cad - tail_center_cad
    direction_length = float(np.linalg.norm(direction_cad))
    if direction_length > 1e-9:
        direction_cad = direction_cad / direction_length
    else:
        direction_cad = np.asarray([1.0, 0.0], dtype=float)
    cad_points = [
        tip_cad,
        *[
            np.asarray(
                [
                    _px_x_to_cad(float(tail[0]), image_width),
                    _px_y_to_cad(float(tail[1]), image_height, canvas_height),
                ],
                dtype=float,
            )
            for tail in tails
        ],
    ]
    min_x = min(float(point[0]) for point in cad_points)
    max_x = max(float(point[0]) for point in cad_points)
    min_y = min(float(point[1]) for point in cad_points)
    max_y = max(float(point[1]) for point in cad_points)
    metadata = {
        "arrow_candidate_id": candidate_id,
        "arrow_source": "opencv_template",
        "tip_x": round(float(tip_cad[0]), 4),
        "tip_y": round(float(tip_cad[1]), 4),
        "direction_x": round(float(direction_cad[0]), 6),
        "direction_y": round(float(direction_cad[1]), 6),
        "score": round(float(hit.score), 4),
        "size_mm": round(direction_length, 4),
        "bbox_x": round(min_x, 4),
        "bbox_y": round(min_y, 4),
        "bbox_width": round(max(max_x - min_x, 0.001), 4),
        "bbox_height": round(max(max_y - min_y, 0.001), 4),
    }

    entities: list[LineEntity] = []
    for suffix, tail in zip(("a", "b"), tails, strict=True):
        entities.append(
            LineEntity(
                id=f"{candidate_id}_{suffix}",
                layer=TEMPLATE_ARROW_LAYER,
                x1=_px_x_to_cad(float(tip[0]), image_width),
                y1=_px_y_to_cad(float(tip[1]), image_height, canvas_height),
                x2=_px_x_to_cad(float(tail[0]), image_width),
                y2=_px_y_to_cad(float(tail[1]), image_height, canvas_height),
                group=PROMOTED_GROUP,
                tags=[
                    "promoted_geometry",
                    "dimension_arrow",
                    "arrowhead",
                    TEMPLATE_ARROW_TAG,
                    "template_match",
                    f"score_{hit.score:.3f}",
                ],
                stroke_width=TEMPLATE_ARROW_STROKE_MM,
                metadata=metadata,
            )
        )
    return entities


def _unit(angle_deg: float) -> np.ndarray:
    radians = math.radians(angle_deg)
    return np.asarray([math.cos(radians), math.sin(radians)], dtype=float)


def _ensure_dimension_arrow_layer(ir: DrawingIR) -> None:
    for layer in ir.layers:
        if layer.name == TEMPLATE_ARROW_LAYER:
            layer.color = "white"
            return
    ir.layers.append(Layer(name=TEMPLATE_ARROW_LAYER, color="white"))
