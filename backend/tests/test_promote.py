from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np

from app import promote as promote_module
from app.models import DrawingIR, Layer, LineEntity, PolylineEntity, ProjectState
from app.promote import (
    PROMOTED_LAYER,
    LineCandidate,
    _angle_coverage,
    _arrow_pair,
    _candidate_fits_cluster,
    _is_closed_polyline,
    _point_line_distances,
    _promote_circle,
    _promote_line,
    _single_arrow_stroke,
    _unit_vector,
    promote_scan_primitives,
)


def test_promote_scan_primitives_extracts_lines_and_circles_idempotently():
    now = datetime.now(timezone.utc)
    ir = DrawingIR(
        units="mm",
        layers=[
            Layer(name="editable_linework", color="white"),
            Layer(name=PROMOTED_LAYER, color="white"),
        ],
        entities=[
            PolylineEntity(
                id="editable_line",
                layer="editable_linework",
                points=[[0, 0], [10, 0.02], [24, 0]],
                group="editable_linework",
            ),
            PolylineEntity(
                id="editable_circle",
                layer="editable_linework",
                points=[
                    [50 + math.cos(index / 32 * math.tau) * 8, 40 + math.sin(index / 32 * math.tau) * 8]
                    for index in range(33)
                ],
                closed=True,
                group="editable_linework",
            ),
            LineEntity(
                id="old_promoted",
                layer=PROMOTED_LAYER,
                x1=0,
                y1=0,
                x2=1,
                y2=1,
                group="promoted_geometry",
            ),
        ],
    )
    project = ProjectState(project_id="pid", name="demo", created_at=now, updated_at=now, ir=ir)

    result = promote_scan_primitives(project)

    assert result.source_count == 2
    assert result.promoted_counts == {"line": 1, "circle": 1, "arrow": 0}
    promoted = [entity for entity in result.ir.entities if entity.group == "promoted_geometry"]
    assert {entity.type for entity in promoted} == {"line", "circle"}
    assert all(entity.layer == PROMOTED_LAYER for entity in promoted)
    assert not any(entity.id == "old_promoted" for entity in result.ir.entities)


def test_promote_merges_collinear_segments_and_keeps_dimension_arrows():
    now = datetime.now(timezone.utc)
    ir = DrawingIR(
        units="mm",
        layers=[Layer(name="editable_linework", color="white")],
        entities=[
            PolylineEntity(
                id="editable_dim_a",
                layer="editable_linework",
                points=[[0, 0], [9, 0.01]],
                group="editable_linework",
            ),
            PolylineEntity(
                id="editable_dim_b",
                layer="editable_linework",
                points=[[10.2, 0.02], [20, 0]],
                group="editable_linework",
            ),
            PolylineEntity(
                id="editable_arrow_a",
                layer="editable_linework",
                points=[[20, 0], [17, 1.2]],
                group="editable_linework",
            ),
            PolylineEntity(
                id="editable_arrow_b",
                layer="editable_linework",
                points=[[20, 0], [17, -1.2]],
                group="editable_linework",
            ),
        ],
    )
    project = ProjectState(project_id="pid", name="demo", created_at=now, updated_at=now, ir=ir)

    result = promote_scan_primitives(project)

    assert result.promoted_counts == {"line": 1, "circle": 0, "arrow": 2}
    promoted = [entity for entity in result.ir.entities if entity.group == "promoted_geometry"]
    merged = next(entity for entity in promoted if "merged_collinear" in entity.tags)
    arrows = [entity for entity in promoted if "dimension_arrow" in entity.tags]
    assert math.hypot(merged.x2 - merged.x1, merged.y2 - merged.y1) >= 19.5
    assert len(arrows) == 2
    assert all(entity.stroke_width == 0.25 for entity in arrows)


def test_promote_scan_primitives_requires_editable_linework():
    now = datetime.now(timezone.utc)
    project = ProjectState(
        project_id="pid",
        name="demo",
        created_at=now,
        updated_at=now,
        ir=DrawingIR(units="mm", layers=[Layer(name="sheet")], entities=[]),
    )

    try:
        promote_scan_primitives(project)
    except ValueError as exc:
        assert "editable_linework" in str(exc)
    else:
        raise AssertionError("expected missing editable_linework error")


def test_promote_scan_primitives_reports_low_confidence_skips():
    now = datetime.now(timezone.utc)
    ir = DrawingIR(
        units="mm",
        layers=[Layer(name="editable_linework", color="white")],
        entities=[
            PolylineEntity(
                id="short_line",
                layer="editable_linework",
                points=[[0, 0], [1, 0]],
                group="editable_linework",
            ),
            PolylineEntity(
                id="noisy_line",
                layer="editable_linework",
                points=[[0, 0], [10, 5], [25, 0]],
                group="editable_linework",
            ),
            PolylineEntity(
                id="bad_circle",
                layer="editable_linework",
                points=[[0, 0], [20, 0], [20, 2], [0, 2], [0, 0]] * 3,
                closed=True,
                group="editable_linework",
            ),
        ],
    )
    project = ProjectState(project_id="pid", name="demo", created_at=now, updated_at=now, ir=ir)

    result = promote_scan_primitives(project)

    assert result.promoted_counts == {"line": 0, "circle": 0, "arrow": 0}
    assert any("No high-confidence" in warning for warning in result.warnings)
    assert any("Skipped 3" in warning for warning in result.warnings)


def test_promote_scan_primitives_reports_caps(monkeypatch):
    now = datetime.now(timezone.utc)
    monkeypatch.setattr(promote_module, "MAX_PROMOTED_LINES", 1)
    monkeypatch.setattr(promote_module, "MAX_PROMOTED_CIRCLES", 1)
    entities = [
        PolylineEntity(
            id=f"line_{idx}",
            layer="editable_linework",
            points=[[0, idx], [20, idx]],
            group="editable_linework",
        )
        for idx in range(2)
    ]
    entities.extend(
        [
            PolylineEntity(
                id=f"circle_{idx}",
                layer="editable_linework",
                points=[
                    [50 + idx * 30 + math.cos(step / 32 * math.tau) * 8, 40 + math.sin(step / 32 * math.tau) * 8]
                    for step in range(33)
                ],
                closed=True,
                group="editable_linework",
            )
            for idx in range(2)
        ]
    )
    project = ProjectState(
        project_id="pid",
        name="demo",
        created_at=now,
        updated_at=now,
        ir=DrawingIR(units="mm", layers=[Layer(name="editable_linework")], entities=entities),
    )

    result = promote_scan_primitives(project)

    assert result.promoted_counts == {"line": 1, "circle": 1, "arrow": 0}
    assert any("Line promotion capped" in warning for warning in result.warnings)
    assert any("Circle promotion capped" in warning for warning in result.warnings)


def test_promote_scan_primitives_reports_arrow_cap(monkeypatch):
    now = datetime.now(timezone.utc)
    monkeypatch.setattr(promote_module, "MAX_PROMOTED_ARROWS", 1)
    ir = DrawingIR(
        units="mm",
        layers=[Layer(name="editable_linework", color="white")],
        entities=[
            PolylineEntity(
                id="editable_dim",
                layer="editable_linework",
                points=[[0, 0], [20, 0]],
                group="editable_linework",
            ),
            PolylineEntity(
                id="editable_arrow_a",
                layer="editable_linework",
                points=[[20, 0], [17, 1.2]],
                group="editable_linework",
            ),
            PolylineEntity(
                id="editable_arrow_b",
                layer="editable_linework",
                points=[[20, 0], [17, -1.2]],
                group="editable_linework",
            ),
        ],
    )
    project = ProjectState(project_id="pid", name="demo", created_at=now, updated_at=now, ir=ir)

    result = promote_scan_primitives(project)

    assert result.promoted_counts == {"line": 1, "circle": 0, "arrow": 1}
    assert any("Dimension arrow promotion capped" in warning for warning in result.warnings)


def test_promote_helper_rejection_edges(monkeypatch):
    open_circle = PolylineEntity(
        id="open",
        layer="editable_linework",
        points=[[math.cos(step / 32 * math.tau), math.sin(step / 32 * math.tau)] for step in range(20)],
        group="editable_linework",
    )
    assert _promote_circle(open_circle, 0) is None

    tiny_circle = PolylineEntity(
        id="tiny",
        layer="editable_linework",
        points=[[math.cos(step / 32 * math.tau) * 0.2, math.sin(step / 32 * math.tau) * 0.2] for step in range(33)],
        closed=True,
        group="editable_linework",
    )
    assert _promote_circle(tiny_circle, 0) is None

    partial_circle = PolylineEntity(
        id="partial",
        layer="editable_linework",
        points=[[math.cos(step / 32 * math.pi) * 8, math.sin(step / 32 * math.pi) * 8] for step in range(33)],
        closed=True,
        group="editable_linework",
    )
    assert _promote_circle(partial_circle, 0) is None

    monkeypatch.setattr(promote_module, "_fit_circle", lambda points: (_ for _ in ()).throw(np.linalg.LinAlgError()))
    assert _promote_circle(partial_circle, 0) is None

    closed_line = PolylineEntity(
        id="closed_line",
        layer="editable_linework",
        points=[[0, 0], [20, 0], [0, 0]],
        closed=True,
        group="editable_linework",
    )
    assert _promote_line(closed_line, 0) is None
    assert _is_closed_polyline(closed_line, np.asarray([])) is True
    assert _is_closed_polyline(open_circle, np.asarray([[0.0, 0.0]])) is False
    assert np.isinf(_point_line_distances(np.asarray([[0.0, 0.0]]), np.asarray([0.0, 0.0]), np.asarray([0.0, 0.0]))[0])
    assert _angle_coverage(np.asarray([[0.0, 0.0]]), 0, 0) == 0.0


def test_promote_v2_helper_rejection_edges():
    base = LineCandidate(
        source_id="a",
        start=np.asarray([0.0, 0.0]),
        end=np.asarray([10.0, 0.0]),
        length=10.0,
        angle=0.0,
    )
    angled = LineCandidate(
        source_id="b",
        start=np.asarray([0.0, 0.0]),
        end=np.asarray([10.0, 2.0]),
        length=10.2,
        angle=math.atan2(2, 10),
    )
    assert _candidate_fits_cluster(angled, [base]) is False

    zero = LineCandidate(
        source_id="zero",
        start=np.asarray([1.0, 1.0]),
        end=np.asarray([1.0, 1.0]),
        length=0.0,
        angle=0.0,
    )
    assert np.allclose(_unit_vector(zero), np.asarray([1.0, 0.0]))

    arrow = LineCandidate(
        source_id="arrow",
        start=np.asarray([20.0, 0.0]),
        end=np.asarray([17.0, 1.2]),
        length=3.23,
        angle=math.atan2(1.2, -3) % math.pi,
    )
    assert _single_arrow_stroke(arrow, [(np.asarray([100.0, 0.0]), 0.0)]) is False
    assert _single_arrow_stroke(base, [(np.asarray([10.0, 0.0]), 0.0)]) is False
    assert _arrow_pair(base, angled, [np.asarray([100.0, 100.0])]) is None
    assert _arrow_pair(zero, zero, [np.asarray([1.0, 1.0])]) is None
