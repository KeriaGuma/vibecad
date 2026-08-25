from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cad_layers import DIMENSION
from .models import (
    DimensionBinding,
    DrawingIR,
    Layer,
    LineEntity,
    MechanicalArrowhead,
    MechanicalDimensionObject,
    PolylineEntity,
)

DIMENSION_ARROW_RENDER_TAG = "dimension_arrow_render"
DIMENSION_ARROW_STROKE_MM = 0.22
DIMENSION_ARROW_LENGTH_MM = 2.6
DIMENSION_ARROW_ANGLE_DEG = 28.0
DIMENSION_ARROW_WIDTH_RATIO = 0.42
DIMENSION_ARROW_LAYER = DIMENSION
DIMENSION_ARROW_GROUP = "dimensions"
MAX_RENDERED_BINDINGS = 80
MIN_VERIFIED_ARROW_SCORE = 0.70
VERIFIED_ARROW_TIP_ENDPOINT_TOLERANCE_MM = 6.0
MAX_ARROWHEADS_PER_BINDING = 2


@dataclass(frozen=True)
class DimensionArrowRender:
    ir: DrawingIR
    arrow_line_count: int
    mechanical_dimensions: list[MechanicalDimensionObject]
    warnings: list[str]


@dataclass(frozen=True)
class ArrowheadRef:
    candidate_id: str
    source_entity_id: str
    tip: np.ndarray
    direction: np.ndarray
    size_mm: float
    score: float
    endpoint: str
    endpoint_distance: float


def render_dimension_binding_arrowheads(ir: DrawingIR, bindings: list[DimensionBinding]) -> DimensionArrowRender:
    if not bindings:
        return DimensionArrowRender(
            ir=ir,
            arrow_line_count=0,
            mechanical_dimensions=[],
            warnings=["No dimension bindings available."],
        )

    next_ir = ir.model_copy(deep=True)
    next_ir.entities = [entity for entity in next_ir.entities if DIMENSION_ARROW_RENDER_TAG not in entity.tags]
    _ensure_dimensions_layer(next_ir)
    line_entities = {entity.id: entity for entity in next_ir.entities if isinstance(entity, LineEntity)}

    arrows: list[PolylineEntity] = []
    mechanical_dimensions: list[MechanicalDimensionObject] = []
    skipped = 0
    skipped_unverified = 0
    rejected_far = 0
    for binding in bindings[:MAX_RENDERED_BINDINGS]:
        candidates, rejected_count = _binding_arrowhead_candidates(binding, line_entities)
        rejected_far += rejected_count
        selected = _select_arrowheads_for_binding(binding, candidates)
        if not selected:
            skipped_unverified += 1
            continue

        binding_arrowheads: list[MechanicalArrowhead] = []
        for candidate in selected:
            rendered = _render_candidate_arrow(binding, candidate, len(arrows))
            if rendered:
                arrows.append(rendered)
                binding_arrowheads.append(_mechanical_arrowhead(candidate, rendered.id))
            else:
                skipped += 1
        if binding_arrowheads:
            mechanical_dimensions.append(_mechanical_dimension(binding, binding_arrowheads))

    next_ir.entities.extend(arrows)
    if arrows:
        next_ir.notes = [
            *next_ir.notes,
            f"Rendered {len(arrows)} verified solid dimension arrowheads from {min(len(bindings), MAX_RENDERED_BINDINGS)} bindings.",
        ]

    warnings: list[str] = []
    if skipped_unverified:
        warnings.append(f"Skipped {skipped_unverified} dimension bindings without verified arrowhead candidates.")
    if rejected_far:
        warnings.append(f"Rejected {rejected_far} arrowhead candidates too far from bound dimension-line endpoints.")
    if skipped:
        warnings.append(f"Skipped {skipped} degenerate dimension arrow endpoints.")
    if len(bindings) > MAX_RENDERED_BINDINGS:
        warnings.append(f"Dimension arrow rendering capped at {MAX_RENDERED_BINDINGS} bindings.")
    return DimensionArrowRender(
        ir=next_ir,
        arrow_line_count=len(arrows),
        mechanical_dimensions=mechanical_dimensions,
        warnings=warnings,
    )


def _ensure_dimensions_layer(ir: DrawingIR) -> None:
    for layer in ir.layers:
        if layer.name == DIMENSION_ARROW_LAYER:
            layer.color = "white"
            return
    ir.layers.append(Layer(name=DIMENSION_ARROW_LAYER, color="white"))


def _binding_arrowhead_candidates(
    binding: DimensionBinding,
    line_entities: dict[str, LineEntity],
) -> tuple[list[ArrowheadRef], int]:
    by_candidate_id: dict[str, ArrowheadRef] = {}
    rejected_far = 0
    for arrow_id in binding.arrow_ids:
        arrow = line_entities.get(arrow_id)
        if arrow is None:
            continue
        candidate = _arrowhead_ref_from_entity(binding, arrow)
        if candidate is None:
            continue
        if candidate.endpoint_distance > VERIFIED_ARROW_TIP_ENDPOINT_TOLERANCE_MM:
            rejected_far += 1
            continue
        old = by_candidate_id.get(candidate.candidate_id)
        if old is None or (candidate.score, -candidate.endpoint_distance) > (old.score, -old.endpoint_distance):
            by_candidate_id[candidate.candidate_id] = candidate
    return sorted(by_candidate_id.values(), key=lambda item: (item.endpoint, item.endpoint_distance, -item.score)), rejected_far


def _arrowhead_ref_from_entity(binding: DimensionBinding, arrow: LineEntity) -> ArrowheadRef | None:
    metadata = arrow.metadata or {}
    candidate_id = metadata.get("arrow_candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        return None
    score = _metadata_float(metadata, "score")
    if score is None or score < MIN_VERIFIED_ARROW_SCORE:
        return None
    tip_x = _metadata_float(metadata, "tip_x")
    tip_y = _metadata_float(metadata, "tip_y")
    direction_x = _metadata_float(metadata, "direction_x")
    direction_y = _metadata_float(metadata, "direction_y")
    if tip_x is None or tip_y is None or direction_x is None or direction_y is None:
        return None
    direction = np.asarray([direction_x, direction_y], dtype=float)
    direction_length = float(np.linalg.norm(direction))
    if direction_length <= 1e-9:
        return None
    direction = direction / direction_length
    size_mm = _metadata_float(metadata, "size_mm") or DIMENSION_ARROW_LENGTH_MM
    tip = np.asarray([tip_x, tip_y], dtype=float)
    endpoint, endpoint_distance = _nearest_binding_endpoint(binding, tip)
    return ArrowheadRef(
        candidate_id=candidate_id,
        source_entity_id=arrow.id,
        tip=tip,
        direction=direction,
        size_mm=max(0.8, size_mm),
        score=score,
        endpoint=endpoint,
        endpoint_distance=endpoint_distance,
    )


def _metadata_float(metadata: dict[str, object], key: str) -> float | None:
    value = metadata.get(key)
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _nearest_binding_endpoint(binding: DimensionBinding, point: np.ndarray) -> tuple[str, float]:
    start = np.asarray([binding.line_x1, binding.line_y1], dtype=float)
    end = np.asarray([binding.line_x2, binding.line_y2], dtype=float)
    start_distance = float(np.linalg.norm(point - start))
    end_distance = float(np.linalg.norm(point - end))
    if start_distance <= end_distance:
        return "start", start_distance
    return "end", end_distance


def _select_arrowheads_for_binding(binding: DimensionBinding, candidates: list[ArrowheadRef]) -> list[ArrowheadRef]:
    if not candidates:
        return []
    if binding.kind not in {"linear", "diameter"}:
        return [min(candidates, key=lambda item: (item.endpoint_distance, -item.score))]

    by_endpoint: dict[str, ArrowheadRef] = {}
    for candidate in candidates:
        old = by_endpoint.get(candidate.endpoint)
        if old is None or (candidate.score, -candidate.endpoint_distance) > (old.score, -old.endpoint_distance):
            by_endpoint[candidate.endpoint] = candidate

    ordered = [by_endpoint[endpoint] for endpoint in ("start", "end") if endpoint in by_endpoint]
    if len(ordered) < MAX_ARROWHEADS_PER_BINDING:
        extras = [
            candidate
            for candidate in sorted(candidates, key=lambda item: (item.endpoint_distance, -item.score))
            if candidate.candidate_id not in {item.candidate_id for item in ordered}
        ]
        ordered.extend(extras[: MAX_ARROWHEADS_PER_BINDING - len(ordered)])
    return ordered[:MAX_ARROWHEADS_PER_BINDING]


def _render_candidate_arrow(
    binding: DimensionBinding,
    candidate: ArrowheadRef,
    index: int,
) -> PolylineEntity | None:
    direction = candidate.direction
    normal = np.asarray([-direction[1], direction[0]], dtype=float)
    tip = candidate.tip
    arrow_length = min(DIMENSION_ARROW_LENGTH_MM, max(1.2, candidate.size_mm))
    tail_center = tip - direction * arrow_length
    tail_offset = arrow_length * DIMENSION_ARROW_WIDTH_RATIO * 0.5
    tails = (tail_center + normal * tail_offset, tail_center - normal * tail_offset)
    points = [
        [round(float(tip[0]), 4), round(float(tip[1]), 4)],
        [round(float(tails[0][0]), 4), round(float(tails[0][1]), 4)],
        [round(float(tails[1][0]), 4), round(float(tails[1][1]), 4)],
    ]
    if len({tuple(point) for point in points}) < 3:
        return None

    entity_id = f"dimension_arrow_render_{index:04d}_{candidate.endpoint}"
    return PolylineEntity(
        id=entity_id,
        layer=DIMENSION_ARROW_LAYER,
        points=points,
        closed=True,
        group=DIMENSION_ARROW_GROUP,
        tags=[
            "dimensions",
            "dimension_arrow",
            "arrowhead",
            "solid_fill",
            "mechanical_semantic",
            DIMENSION_ARROW_RENDER_TAG,
            binding.id,
            candidate.candidate_id,
        ],
        stroke_width=DIMENSION_ARROW_STROKE_MM,
        metadata={
            "fill": True,
            "mechanical_role": "dimension_arrowhead",
            "binding_id": binding.id,
            "arrow_candidate_id": candidate.candidate_id,
            "source_entity_id": candidate.source_entity_id,
            "tip_x": round(float(tip[0]), 4),
            "tip_y": round(float(tip[1]), 4),
            "direction_x": round(float(direction[0]), 6),
            "direction_y": round(float(direction[1]), 6),
            "score": round(candidate.score, 4),
            "endpoint": candidate.endpoint,
            "endpoint_distance": round(candidate.endpoint_distance, 4),
        },
    )


def _mechanical_arrowhead(candidate: ArrowheadRef, render_entity_id: str) -> MechanicalArrowhead:
    return MechanicalArrowhead(
        candidate_id=candidate.candidate_id,
        source_entity_id=candidate.source_entity_id,
        render_entity_id=render_entity_id,
        tip_x=round(float(candidate.tip[0]), 4),
        tip_y=round(float(candidate.tip[1]), 4),
        direction_x=round(float(candidate.direction[0]), 6),
        direction_y=round(float(candidate.direction[1]), 6),
        score=round(candidate.score, 4),
        endpoint=candidate.endpoint,
        endpoint_distance=round(candidate.endpoint_distance, 4),
    )


def _mechanical_dimension(
    binding: DimensionBinding,
    arrowheads: list[MechanicalArrowhead],
) -> MechanicalDimensionObject:
    return MechanicalDimensionObject(
        id=f"mechanical_dimension_{binding.id}",
        binding_id=binding.id,
        kind=binding.kind,
        text=binding.text,
        parsed=binding.parsed,
        confidence=binding.confidence,
        dimension_line_id=binding.dimension_line_id,
        text_id=binding.text_id,
        arrowheads=arrowheads,
        evidence=[
            binding.binding_method,
            *binding.graph_path[:6],
            *[f"arrow:{arrow.candidate_id}" for arrow in arrowheads],
        ],
    )
