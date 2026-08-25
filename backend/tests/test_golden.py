"""Golden (snapshot) tests for the spur-gear template exports.

These freeze the *exported geometry* of ``spur_gear_drawing_ir()`` so that any
future change to the template or the exporters that alters the drawing fails
loudly instead of silently.

The SVG is byte-for-byte deterministic (fixed ids, pure coordinate formatting),
so it is compared verbatim. A raw DXF embeds timestamps/handles, so we compare a
normalized geometry summary instead of the raw bytes.

Regenerate the goldens after an intentional change with:

    GOLDEN_UPDATE=1 ./.venv/bin/python -m pytest tests/test_golden.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import ezdxf

from app.exporter import export_dxf, export_svg
from app.templates import spur_gear_drawing_ir

GOLDEN_DIR = Path(__file__).parent / "golden"
UPDATE = os.environ.get("GOLDEN_UPDATE") == "1"


def _round(value: float, ndigits: int = 4) -> float:
    return round(float(value), ndigits)


def _dxf_summary(path: Path) -> list[dict]:
    """A stable, timestamp-free representation of a DXF modelspace."""
    doc = ezdxf.readfile(path)
    rows: list[dict] = []
    for entity in doc.modelspace():
        dxftype = entity.dxftype()
        row: dict = {"type": dxftype, "layer": entity.dxf.layer}
        if dxftype == "LINE":
            start, end = entity.dxf.start, entity.dxf.end
            row["geo"] = [_round(start.x), _round(start.y), _round(end.x), _round(end.y)]
        elif dxftype == "CIRCLE":
            center = entity.dxf.center
            row["geo"] = [_round(center.x), _round(center.y), _round(entity.dxf.radius)]
        elif dxftype == "ARC":
            center = entity.dxf.center
            row["geo"] = [
                _round(center.x),
                _round(center.y),
                _round(entity.dxf.radius),
                _round(entity.dxf.start_angle),
                _round(entity.dxf.end_angle),
            ]
        elif dxftype == "LWPOLYLINE":
            row["geo"] = [[_round(x), _round(y)] for x, y, *_ in entity.get_points()]
            row["closed"] = bool(entity.closed)
        elif dxftype == "TEXT":
            insert = entity.dxf.insert
            row["geo"] = [_round(insert.x), _round(insert.y)]
            row["text"] = entity.dxf.text
        rows.append(row)
    return rows


def _read_or_write(path: Path, content: str) -> str:
    if UPDATE:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return path.read_text(encoding="utf-8")


def test_gear_svg_matches_golden(tmp_path):
    out = tmp_path / "gear.svg"
    export_svg(spur_gear_drawing_ir(), out)
    actual = out.read_text(encoding="utf-8")
    expected = _read_or_write(GOLDEN_DIR / "gear.svg", actual)
    assert actual == expected, "Gear SVG changed; rerun with GOLDEN_UPDATE=1 if intentional."


def test_gear_dxf_geometry_matches_golden(tmp_path):
    out = tmp_path / "gear.dxf"
    export_dxf(spur_gear_drawing_ir(), out)
    actual = json.dumps(_dxf_summary(out), ensure_ascii=False, indent=2, sort_keys=True)
    expected = _read_or_write(GOLDEN_DIR / "gear_dxf.json", actual)
    assert actual == expected, "Gear DXF geometry changed; rerun with GOLDEN_UPDATE=1 if intentional."
