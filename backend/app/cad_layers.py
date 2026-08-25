from __future__ import annotations

from dataclasses import dataclass

from .models import BaseEntity, DrawingIR, Layer, RectangleEntity, TextEntity

REFERENCE_TRACE = "REFERENCE_TRACE"
OUTLINE = "OUTLINE"
DIMENSION = "DIMENSION"
CENTER = "CENTER"
HATCH = "HATCH"
TEXT = "TEXT"
TITLE_BLOCK = "TITLE_BLOCK"


@dataclass(frozen=True)
class CadLayerSpec:
    name: str
    color: str
    lineweight: float
    linetype: str = "CONTINUOUS"
    locked: bool = False
    editable: bool = True


CAD_LAYER_SPECS = (
    CadLayerSpec(REFERENCE_TRACE, "gray", 0.13, locked=True, editable=False),
    CadLayerSpec(OUTLINE, "white", 0.50),
    CadLayerSpec(DIMENSION, "white", 0.25),
    CadLayerSpec(CENTER, "white", 0.18, linetype="CENTER2"),
    CadLayerSpec(HATCH, "white", 0.18),
    CadLayerSpec(TEXT, "white", 0.18),
    CadLayerSpec(TITLE_BLOCK, "white", 0.25),
)

CAD_LAYER_BY_NAME = {spec.name: spec for spec in CAD_LAYER_SPECS}

LEGACY_LAYER_ALIASES = {
    "reference_trace": REFERENCE_TRACE,
    "editable_linework": OUTLINE,
    "promoted_geometry": OUTLINE,
    "geometry": OUTLINE,
    "outline": OUTLINE,
    "holes": OUTLINE,
    "0": OUTLINE,
    "dimensions": DIMENSION,
    "dimension": DIMENSION,
    "centerline": CENTER,
    "center": CENTER,
    "hatch": HATCH,
    "text": TEXT,
    "notes": TEXT,
    "table": TITLE_BLOCK,
    "title_block": TITLE_BLOCK,
    "sheet": TITLE_BLOCK,
}


def canonical_layer_name(layer_name: str) -> str:
    if layer_name in CAD_LAYER_BY_NAME:
        return layer_name
    return LEGACY_LAYER_ALIASES.get(layer_name.strip().lower(), OUTLINE)


def entity_cad_layer(entity: BaseEntity) -> str:
    tags = {tag.lower() for tag in entity.tags}
    group = (entity.group or "").lower()
    layer = entity.layer.strip().lower()

    if group == "reference_trace" or "reference_trace" in tags or layer == "reference_trace":
        return REFERENCE_TRACE
    if (
        group == "dimensions"
        or layer in {"dimension", "dimensions"}
        or tags.intersection({"dimensions", "dimension_text", "dimension_arrow", "dimension_arrow_render"})
    ):
        return DIMENSION
    if layer in {"center", "centerline"} or "centerline" in tags:
        return CENTER
    if layer == "hatch" or tags.intersection({"hatch", "cut_hatch", "clipped_hatch"}):
        return HATCH
    if (
        group in {"title_block", "parameter_table", "sheet"}
        or layer in {"table", "title_block", "sheet"}
        or tags.intersection({"title_block", "parameter_table", "sheet_border", "drawing_frame"})
    ):
        return TITLE_BLOCK
    if isinstance(entity, TextEntity) or layer in {"text", "notes"}:
        return TEXT
    return canonical_layer_name(entity.layer)


def normalize_cad_layers(ir: DrawingIR) -> DrawingIR:
    """Migrate an IR in place to the seven public CAD layers.

    Legacy groups and tags stay intact because the CV/semantic pipeline uses
    them as provenance. The public layer is the rendering and DXF contract.
    """

    for entity in ir.entities:
        old_layer = entity.layer
        canonical = entity_cad_layer(entity)
        spec = CAD_LAYER_BY_NAME[canonical]
        entity.layer = canonical
        entity.stroke_width = _entity_lineweight(entity, spec)
        entity.metadata = {
            **entity.metadata,
            **({"legacy_layer": old_layer} if old_layer != canonical and "legacy_layer" not in entity.metadata else {}),
            "cad_role": canonical,
            "locked": spec.locked,
            "editable": spec.editable,
        }

    ir.layers = [
        Layer(
            name=spec.name,
            color=spec.color,
            lineweight=spec.lineweight,
            linetype=spec.linetype,
            locked=spec.locked,
            editable=spec.editable,
        )
        for spec in CAD_LAYER_SPECS
    ]
    return ir


def normalized_cad_ir(ir: DrawingIR) -> DrawingIR:
    return normalize_cad_layers(ir.model_copy(deep=True))


def _entity_lineweight(entity: BaseEntity, spec: CadLayerSpec) -> float:
    if spec.name == TITLE_BLOCK and _is_border_entity(entity):
        return 0.50
    return spec.lineweight


def _is_border_entity(entity: BaseEntity) -> bool:
    entity_id = entity.id.lower()
    tags = {tag.lower() for tag in entity.tags}
    return (
        isinstance(entity, RectangleEntity)
        or entity.group == "sheet"
        or "border" in entity_id
        or "frame" in entity_id
        or bool(tags.intersection({"sheet_border", "drawing_frame"}))
    )
