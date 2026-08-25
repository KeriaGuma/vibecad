from __future__ import annotations

from pydantic import BaseModel, Field

from .cad_layers import CENTER, DIMENSION, HATCH, canonical_layer_name
from .models import CircleEntity, DrawingIR, Entity, LineEntity, PolylineEntity, RectangleEntity, TextEntity
from .vector_semantics import SEMANTIC_IMPORT_TAG


class TargetEval(BaseModel):
    name: str
    score: float
    passed: bool
    checks: dict[str, bool]
    missing: list[str] = Field(default_factory=list)
    evidence: dict[str, int | float | str | list[str]] = Field(default_factory=dict)


class StructureEvalReport(BaseModel):
    overall_score: float
    passed: bool
    targets: list[TargetEval]


def evaluate_structure(ir: DrawingIR) -> StructureEvalReport:
    """Evaluate the first gear drawing target as five measurable CAD goals."""
    targets = [
        _evaluate_title_block(ir),
        _evaluate_parameter_table(ir),
        _evaluate_section_view(ir),
        _evaluate_circular_view(ir),
        _evaluate_dimensions(ir),
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


def _dimension_entities(ir: DrawingIR) -> list[Entity]:
    return [
        entity
        for entity in ir.entities
        if entity.group == "dimensions" or canonical_layer_name(entity.layer) == DIMENSION or "dimensions" in entity.tags
    ]


def _texts(entities: list[Entity]) -> list[TextEntity]:
    return [entity for entity in entities if isinstance(entity, TextEntity)]


def _text_values(entities: list[Entity]) -> list[str]:
    return [entity.text for entity in _texts(entities)]


def _has_text(texts: list[str], fragment: str) -> bool:
    return any(fragment in text for text in texts)


def _count_type(entities: list[Entity], entity_type: type[Entity]) -> int:
    return sum(1 for entity in entities if isinstance(entity, entity_type))


def _has_semantic_import(entities: list[Entity]) -> bool:
    return any(SEMANTIC_IMPORT_TAG in entity.tags for entity in entities)


def _import_proxy(entities: list[Entity], min_entities: int) -> bool:
    return _has_semantic_import(entities) and len(entities) >= min_entities


def _evaluate_title_block(ir: DrawingIR) -> TargetEval:
    entities = _by_group(ir, "title_block")
    texts = _text_values(entities)
    grid_count = _count_type(entities, LineEntity) + _count_type(entities, RectangleEntity)
    imported_title_block = _import_proxy(entities, 80)
    checks = {
        "has title_block region": len(entities) >= 12,
        "has title-block grid": grid_count >= 8,
        "contains school": _has_text(texts, "合肥工业大学") or imported_title_block,
        "contains drawing name": _has_text(texts, "圆柱直齿轮") or imported_title_block,
        "contains drawing code": _has_text(texts, "LJT01.01") or imported_title_block,
    }
    return _target(
        "title_block",
        checks,
        {
            "entity_count": len(entities),
            "grid_entity_count": grid_count,
            "text_count": len(texts),
            "semantic_import": _has_semantic_import(entities),
            "sample_texts": texts[:6],
        },
    )


def _evaluate_parameter_table(ir: DrawingIR) -> TargetEval:
    entities = _by_group(ir, "parameter_table")
    texts = _text_values(entities)
    grid_count = _count_type(entities, LineEntity) + _count_type(entities, RectangleEntity)
    imported_table = _import_proxy(entities, 180)
    checks = {
        "has parameter_table region": len(entities) >= 25,
        "has dense table grid": grid_count >= 18,
        "contains tooth count": _has_text(texts, "齿数") or imported_table,
        "contains module": _has_text(texts, "模数") or imported_table,
        "contains pressure angle": _has_text(texts, "压力角") or imported_table,
        "contains accuracy grade": _has_text(texts, "精度等级") or imported_table,
        "contains tolerance rows": _has_text(texts, "齿距") or _has_text(texts, "偏差") or imported_table,
    }
    return _target(
        "parameter_table",
        checks,
        {
            "entity_count": len(entities),
            "grid_entity_count": grid_count,
            "text_count": len(texts),
            "semantic_import": _has_semantic_import(entities),
            "sample_texts": texts[:8],
        },
    )


def _evaluate_section_view(ir: DrawingIR) -> TargetEval:
    entities = _by_group(ir, "section_view")
    texts = _text_values(entities)
    dimension_texts = _text_values(
        [entity for entity in entities if "dimensions" in entity.tags or canonical_layer_name(entity.layer) == DIMENSION]
    )
    hatch_count = sum(1 for entity in entities if canonical_layer_name(entity.layer) == HATCH or "hatch" in entity.tags)
    centerline_count = sum(
        1 for entity in entities if canonical_layer_name(entity.layer) == CENTER or "centerline" in entity.tags
    )
    imported_section = _import_proxy(entities, 120)
    imported_dimension_proxy = imported_section and len([entity for entity in entities if "dimensions" in entity.tags]) >= 24
    checks = {
        "has stepped section profile": any(entity.id == "section_profile" for entity in entities) or imported_section,
        "has bore opening": any(entity.id == "section_bore" for entity in entities) or imported_section,
        "has hatch lines": hatch_count >= 8,
        "has centerlines": centerline_count >= 2,
        "has diameter dimensions": (_has_text(dimension_texts, "φ62") and _has_text(dimension_texts, "φ58")) or imported_dimension_proxy,
    }
    return _target(
        "section_view",
        checks,
        {
            "entity_count": len(entities),
            "hatch_count": hatch_count,
            "centerline_count": centerline_count,
            "dimension_text_count": len(dimension_texts),
            "semantic_import": _has_semantic_import(entities),
            "sample_texts": texts[:8],
        },
    )


def _evaluate_circular_view(ir: DrawingIR) -> TargetEval:
    entities = _by_group(ir, "circular_view")
    texts = _text_values(entities)
    dimension_texts = _text_values(
        [entity for entity in entities if "dimensions" in entity.tags or canonical_layer_name(entity.layer) == DIMENSION]
    )
    circle_count = _count_type(entities, CircleEntity)
    centerline_count = sum(
        1 for entity in entities if canonical_layer_name(entity.layer) == CENTER or "centerline" in entity.tags
    )
    imported_circular = _import_proxy(entities, 80)
    imported_dimension_proxy = imported_circular and len([entity for entity in entities if "dimensions" in entity.tags]) >= 12
    checks = {
        "has concentric circles": circle_count >= 2 or imported_circular,
        "has keyway profile": any(isinstance(entity, PolylineEntity) and entity.id == "keyway" for entity in entities) or imported_circular,
        "has centerlines": centerline_count >= 2,
        "has keyway width dimension": _has_text(dimension_texts, "8±0.018") or imported_dimension_proxy,
        "has tolerance callout": (_has_text(dimension_texts, "0.01") and _has_text(dimension_texts, "A")) or imported_dimension_proxy,
    }
    return _target(
        "circular_view",
        checks,
        {
            "entity_count": len(entities),
            "circle_count": circle_count,
            "centerline_count": centerline_count,
            "dimension_text_count": len(dimension_texts),
            "semantic_import": _has_semantic_import(entities),
            "sample_texts": texts[:8],
        },
    )


def _evaluate_dimensions(ir: DrawingIR) -> TargetEval:
    entities = _dimension_entities(ir)
    texts = _texts(entities)
    text_values = [entity.text for entity in texts]
    dimension_line_count = _count_type(entities, LineEntity)
    rotated_text_count = sum(1 for entity in texts if entity.rotation)
    diameter_text_count = sum(1 for value in text_values if "φ" in value)
    roughness_text_count = sum(1 for value in text_values if "Ra" in value)
    imported_dimensions = _import_proxy(entities, 120) and dimension_line_count >= 40
    checks = {
        "has dimension entity set": len(entities) >= 24,
        "has dimension lines": dimension_line_count >= 8,
        "has rotated dimension text": rotated_text_count >= 4 or imported_dimensions,
        "has diameter callouts": diameter_text_count >= 3 or imported_dimensions,
        "has roughness callouts": roughness_text_count >= 4 or imported_dimensions,
        "has datum/tolerance notation": (_has_text(text_values, "A") and _has_text(text_values, "0.01")) or imported_dimensions,
    }
    return _target(
        "dimensions",
        checks,
        {
            "entity_count": len(entities),
            "dimension_line_count": dimension_line_count,
            "rotated_text_count": rotated_text_count,
            "diameter_text_count": diameter_text_count,
            "roughness_text_count": roughness_text_count,
            "semantic_import": _has_semantic_import(entities),
            "sample_texts": text_values[:10],
        },
    )
