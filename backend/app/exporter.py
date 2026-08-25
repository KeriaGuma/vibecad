from __future__ import annotations

import html
import math
from pathlib import Path

import ezdxf

from .cad_layers import (
    CENTER,
    DIMENSION,
    HATCH,
    OUTLINE,
    REFERENCE_TRACE,
    TEXT,
    TITLE_BLOCK,
    normalized_cad_ir,
)
from .models import (
    ArcEntity,
    BaseEntity,
    CircleEntity,
    DrawingIR,
    LineEntity,
    MechanicalDimensionObject,
    MechanicalDrawingIR,
    PolylineEntity,
    RectangleEntity,
    TextEntity,
)

VIBECAD_APP_ID = "VIBECAD"
DIMENSION_LAYER = DIMENSION

DXF_COLOR_INDEX = {
    "red": 1,
    "yellow": 2,
    "green": 3,
    "cyan": 4,
    "blue": 5,
    "magenta": 6,
    "white": 7,
    "gray": 8,
}


SVG_COLOR = {
    "red": "#ef4444",
    "yellow": "#eab308",
    "green": "#22c55e",
    "cyan": "#06b6d4",
    "blue": "#3b82f6",
    "magenta": "#d946ef",
    "white": "#111827",
    "gray": "#6b7280",
}


SVG_LAYER_STROKE_WIDTH = {
    REFERENCE_TRACE: 0.13,
    OUTLINE: 0.50,
    DIMENSION: 0.25,
    CENTER: 0.18,
    HATCH: 0.18,
    TEXT: 0.18,
    TITLE_BLOCK: 0.25,
}


def layer_color(ir: DrawingIR, layer_name: str) -> str:
    for layer in ir.layers:
        if layer.name == layer_name:
            return layer.color
    return "white"


def layer_stroke_width(layer_name: str) -> float:
    return SVG_LAYER_STROKE_WIDTH.get(layer_name, 1.0)


def entity_stroke_width(entity: BaseEntity) -> float:
    layer_width = layer_stroke_width(entity.layer)
    if entity.stroke_width is None:
        return layer_width
    return max(entity.stroke_width, layer_width)


def entity_dxf_attrs(entity: BaseEntity) -> dict:
    return {
        "layer": entity.layer,
        "lineweight": max(0, min(211, round(entity_stroke_width(entity) * 100))),
    }


def entity_has_solid_fill(entity: BaseEntity) -> bool:
    return bool(entity.metadata.get("fill")) or "solid_fill" in entity.tags


def export_dxf(ir: DrawingIR, path: Path, mechanical_ir: MechanicalDrawingIR | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ir = normalized_cad_ir(ir)
    doc = ezdxf.new("R2010")
    doc.units = ezdxf.units.MM if ir.units == "mm" else ezdxf.units.IN
    if "CENTER2" not in doc.linetypes:
        doc.linetypes.add(
            "CENTER2",
            pattern=[2.0, 1.25, -0.25, 0.25, -0.25],
            description="Center line",
        )
    for layer in ir.layers:
        if layer.name not in doc.layers:
            dxf_layer = doc.layers.add(
                layer.name,
                color=DXF_COLOR_INDEX.get(layer.color, 7),
                linetype=layer.linetype,
                lineweight=max(0, min(211, round(layer.lineweight * 100))),
            )
            if layer.locked:
                dxf_layer.lock()
    if DIMENSION_LAYER not in doc.layers:
        doc.layers.add(DIMENSION_LAYER, color=7)
    if VIBECAD_APP_ID not in doc.appids:
        doc.appids.add(VIBECAD_APP_ID)

    msp = doc.modelspace()
    for entity in ir.entities:
        attrs = entity_dxf_attrs(entity)
        if isinstance(entity, LineEntity):
            msp.add_line((entity.x1, entity.y1), (entity.x2, entity.y2), dxfattribs=attrs)
        elif isinstance(entity, PolylineEntity):
            points = [tuple(point) for point in entity.points]
            if entity.closed and entity_has_solid_fill(entity) and len(points) >= 3:
                solid_points = points[:3]
                if len(solid_points) == 3:
                    solid_points.append(solid_points[2])
                msp.add_solid(solid_points, dxfattribs=attrs)
            else:
                msp.add_lwpolyline(points, close=entity.closed, dxfattribs=attrs)
        elif isinstance(entity, CircleEntity):
            msp.add_circle((entity.cx, entity.cy), entity.r, dxfattribs=attrs)
        elif isinstance(entity, ArcEntity):
            msp.add_arc((entity.cx, entity.cy), entity.r, entity.start_angle, entity.end_angle, dxfattribs=attrs)
        elif isinstance(entity, RectangleEntity):
            points = [
                (entity.x, entity.y),
                (entity.x + entity.width, entity.y),
                (entity.x + entity.width, entity.y + entity.height),
                (entity.x, entity.y + entity.height),
                (entity.x, entity.y),
            ]
            msp.add_lwpolyline(points, dxfattribs=attrs)
        elif isinstance(entity, TextEntity):
            text = msp.add_text(entity.text, height=entity.height, rotation=entity.rotation, dxfattribs=attrs)
            text.set_placement((entity.x, entity.y))
    if mechanical_ir is not None:
        _export_mechanical_dimensions(msp, mechanical_ir)
    doc.saveas(path)


def _export_mechanical_dimensions(msp, mechanical_ir: MechanicalDrawingIR) -> int:
    exported = 0
    for dimension in mechanical_ir.dimensions:
        if not dimension.export_ready or not dimension.dxf_dimension_type:
            continue
        override = _add_native_dimension(msp, dimension)
        if override is None:
            continue
        native_dimension = override.dimension
        native_dimension.set_xdata(
            VIBECAD_APP_ID,
            [
                (1000, dimension.id),
                (1000, dimension.binding_id),
                (1000, dimension.kind),
                *[(1000, entity_id) for entity_id in dimension.measured_geometry_ids],
            ],
        )
        override.render()
        exported += 1
    return exported


def _add_native_dimension(msp, dimension: MechanicalDimensionObject):
    points = [tuple(point[:2]) for point in dimension.measurement_points if len(point) >= 2]
    if len(points) < 2:
        return None
    base = (
        tuple(dimension.dimension_line_point[:2])
        if dimension.dimension_line_point and len(dimension.dimension_line_point) >= 2
        else _midpoint(points[0], points[1])
    )
    text = dimension.text or "<>"
    attrs = {"layer": DIMENSION_LAYER}
    style = {
        "dimasz": 2.5,
        "dimtxt": 2.5,
        "dimexe": 1.25,
        "dimexo": 0.625,
        "dimclrd": 7,
        "dimclre": 7,
        "dimclrt": 7,
    }

    try:
        if dimension.dxf_dimension_type == "radius":
            return msp.add_radius_dim(
                center=points[0],
                mpoint=points[1],
                location=base,
                text=text,
                override=style,
                dxfattribs=attrs,
            )
        if dimension.dxf_dimension_type == "diameter":
            center = _midpoint(points[0], points[1])
            return msp.add_diameter_dim(
                center=center,
                mpoint=points[1],
                location=base,
                text=text,
                override=style,
                dxfattribs=attrs,
            )
        if dimension.dxf_dimension_type == "aligned":
            distance = _signed_distance_from_line(base, points[0], points[1])
            return msp.add_aligned_dim(
                p1=points[0],
                p2=points[1],
                distance=distance,
                text=text,
                override=style,
                dxfattribs=attrs,
            )
        angle = 90.0 if dimension.orientation == "vertical" else 0.0
        return msp.add_linear_dim(
            base=base,
            p1=points[0],
            p2=points[1],
            angle=angle,
            text=text,
            override=style,
            dxfattribs=attrs,
        )
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _midpoint(first: tuple[float, float], second: tuple[float, float]) -> tuple[float, float]:
    return ((first[0] + second[0]) * 0.5, (first[1] + second[1]) * 0.5)


def _signed_distance_from_line(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    if length <= 1e-9:
        return 0.0
    return ((point[0] - start[0]) * -dy + (point[1] - start[1]) * dx) / length


def _bounds(ir: DrawingIR) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for entity in ir.entities:
        if isinstance(entity, LineEntity):
            xs.extend([entity.x1, entity.x2])
            ys.extend([entity.y1, entity.y2])
        elif isinstance(entity, PolylineEntity):
            xs.extend([point[0] for point in entity.points])
            ys.extend([point[1] for point in entity.points])
        elif isinstance(entity, CircleEntity):
            xs.extend([entity.cx - entity.r, entity.cx + entity.r])
            ys.extend([entity.cy - entity.r, entity.cy + entity.r])
        elif isinstance(entity, ArcEntity):
            xs.extend([entity.cx - entity.r, entity.cx + entity.r])
            ys.extend([entity.cy - entity.r, entity.cy + entity.r])
        elif isinstance(entity, RectangleEntity):
            xs.extend([entity.x, entity.x + entity.width])
            ys.extend([entity.y, entity.y + entity.height])
        elif isinstance(entity, TextEntity):
            xs.append(entity.x)
            ys.append(entity.y)
    if not xs or not ys:
        return 0, 0, 100, 100
    return min(xs), min(ys), max(xs), max(ys)


def export_svg(ir: DrawingIR, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ir = normalized_cad_ir(ir)
    min_x, min_y, max_x, max_y = _bounds(ir)
    margin = max((max_x - min_x), (max_y - min_y), 1) * 0.08
    min_x -= margin
    min_y -= margin
    max_x += margin
    max_y += margin
    width = max_x - min_x
    height = max_y - min_y

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{min_x} {-max_y} {width} {height}" width="100%" height="100%">',
        '<rect x="-100000" y="-100000" width="200000" height="200000" fill="#f8fafc"/>',
        '<g transform="scale(1,-1)" fill="none" stroke-linecap="round" stroke-linejoin="round">',
    ]
    for entity in ir.entities:
        color = SVG_COLOR.get(layer_color(ir, entity.layer), "#111827")
        stroke_width = entity_stroke_width(entity)
        group_attr = f' data-group="{html.escape(entity.group)}"' if entity.group else ""
        tags_attr = f' data-tags="{html.escape(",".join(entity.tags))}"' if entity.tags else ""
        dash_attr = 'stroke-dasharray="2.5 0.8 0.4 0.8" ' if entity.layer == CENTER else ""
        opacity_attr = 'stroke-opacity="0.48" ' if entity.layer == REFERENCE_TRACE else ""
        common = (
            f'stroke="{color}" stroke-width="{stroke_width}" '
            f"{dash_attr}{opacity_attr}"
            f'data-id="{html.escape(entity.id)}"{group_attr}{tags_attr}'
        )
        if isinstance(entity, LineEntity):
            parts.append(f'<line x1="{entity.x1}" y1="{entity.y1}" x2="{entity.x2}" y2="{entity.y2}" {common}/>')
        elif isinstance(entity, PolylineEntity):
            points = " ".join(f"{x},{y}" for x, y in entity.points)
            if entity.closed:
                fill_attr = f'fill="{color}" ' if entity_has_solid_fill(entity) else ""
                parts.append(f'<polygon points="{points}" {fill_attr}{common}/>')
            else:
                parts.append(f'<polyline points="{points}" {common}/>')
        elif isinstance(entity, CircleEntity):
            parts.append(f'<circle cx="{entity.cx}" cy="{entity.cy}" r="{entity.r}" {common}/>')
        elif isinstance(entity, ArcEntity):
            start = math.radians(entity.start_angle)
            end = math.radians(entity.end_angle)
            x1 = entity.cx + math.cos(start) * entity.r
            y1 = entity.cy + math.sin(start) * entity.r
            x2 = entity.cx + math.cos(end) * entity.r
            y2 = entity.cy + math.sin(end) * entity.r
            delta = (entity.end_angle - entity.start_angle) % 360
            large_arc = 1 if delta > 180 else 0
            parts.append(f'<path d="M {x1} {y1} A {entity.r} {entity.r} 0 {large_arc} 1 {x2} {y2}" {common}/>')
        elif isinstance(entity, RectangleEntity):
            parts.append(f'<rect x="{entity.x}" y="{entity.y}" width="{entity.width}" height="{entity.height}" {common}/>')
        elif isinstance(entity, TextEntity):
            escaped = html.escape(entity.text)
            parts.append("</g>")
            transform = ""
            if entity.rotation:
                transform = f' transform="rotate({-entity.rotation} {entity.x} {-entity.y})"'
            parts.append(
                f'<text x="{entity.x}" y="{-entity.y}" fill="{color}" '
                f'font-family="Songti SC, STSong, SimSun, Noto Serif CJK SC, serif" '
                f'font-size="{entity.height}" data-id="{html.escape(entity.id)}"{transform}>{escaped}</text>'
            )
            parts.append('<g transform="scale(1,-1)" fill="none" stroke-linecap="round" stroke-linejoin="round">')
    parts.append("</g></svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
