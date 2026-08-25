"""Coverage for the gear-drawing structure scorer in ``app.structure_eval``."""
from __future__ import annotations

from app.models import DrawingIR, default_ir
from app.structure_eval import evaluate_structure
from app.templates import spur_gear_drawing_ir

TARGET_NAMES = {"title_block", "parameter_table", "section_view", "circular_view", "dimensions"}


def test_template_scores_full_marks():
    report = evaluate_structure(spur_gear_drawing_ir())
    assert report.passed is True
    assert report.overall_score == 1.0
    assert {t.name for t in report.targets} == TARGET_NAMES
    assert all(t.passed and not t.missing for t in report.targets)


def test_empty_drawing_fails_every_target():
    report = evaluate_structure(DrawingIR())
    assert report.passed is False
    assert report.overall_score == 0.0
    # every target should report what's missing rather than crash
    assert all(t.missing for t in report.targets)


def test_default_plate_is_not_a_gear_drawing():
    report = evaluate_structure(default_ir())
    assert report.passed is False
    assert report.overall_score == 0.0


def test_score_is_monotonic_in_satisfied_checks():
    """Removing the title-block entities should only lower that target's score."""
    ir = spur_gear_drawing_ir()
    full = evaluate_structure(ir)
    ir.entities = [e for e in ir.entities if e.group != "title_block"]
    degraded = evaluate_structure(ir)

    full_tb = next(t for t in full.targets if t.name == "title_block")
    degraded_tb = next(t for t in degraded.targets if t.name == "title_block")
    assert degraded_tb.score < full_tb.score
    assert degraded.overall_score < full.overall_score
    assert degraded.passed is False
