from __future__ import annotations

from datetime import datetime, timezone

from PIL import Image, ImageDraw

from app.arrow_cv import TEMPLATE_ARROW_TAG, detect_arrowheads_from_reference
from app.models import DrawingIR, LineEntity, ProjectState


def _project() -> ProjectState:
    now = datetime.now(timezone.utc)
    return ProjectState(
        project_id="pid",
        name="demo",
        created_at=now,
        updated_at=now,
        source_image="/api/uploads/pid_reference.png",
        ir=DrawingIR(
            entities=[
                LineEntity(
                    id="dim_line",
                    layer="promoted_geometry",
                    x1=40,
                    y1=80,
                    x2=120,
                    y2=80,
                    group="promoted_geometry",
                    tags=["promoted_geometry", "line_fit"],
                )
            ]
        ),
    )


def _write_arrow_reference(path) -> None:
    image = Image.new("L", (420, 210), 255)
    draw = ImageDraw.Draw(image)
    draw.line((80, 105, 340, 105), fill=0, width=2)
    # Right-pointing V arrowhead with tip at x=260.
    draw.line((260, 105, 242, 94), fill=0, width=4)
    draw.line((260, 105, 242, 116), fill=0, width=4)
    image.save(path)


def test_detect_arrowheads_from_reference_adds_template_arrow_lines(tmp_path):
    _write_arrow_reference(tmp_path / "pid_reference.png")

    result = detect_arrowheads_from_reference(_project(), tmp_path)

    arrows = [
        entity
        for entity in result.ir.entities
        if isinstance(entity, LineEntity) and TEMPLATE_ARROW_TAG in entity.tags
    ]
    assert result.detected_count >= 1
    assert len(arrows) >= 2
    assert all("dimension_arrow" in entity.tags and "arrowhead" in entity.tags for entity in arrows)
    assert all(entity.layer == "DIMENSION" for entity in arrows)
    assert all(entity.group == "promoted_geometry" for entity in arrows)
    assert all(entity.metadata.get("arrow_candidate_id") for entity in arrows)
    assert all(isinstance(entity.metadata.get("tip_x"), float) for entity in arrows)
    assert all(isinstance(entity.metadata.get("tip_y"), float) for entity in arrows)
    assert all(isinstance(entity.metadata.get("direction_x"), float) for entity in arrows)
    assert all(isinstance(entity.metadata.get("direction_y"), float) for entity in arrows)


def test_detect_arrowheads_from_reference_is_idempotent(tmp_path):
    _write_arrow_reference(tmp_path / "pid_reference.png")
    first = detect_arrowheads_from_reference(_project(), tmp_path)
    project = _project().model_copy(update={"ir": first.ir})
    second = detect_arrowheads_from_reference(project, tmp_path)

    first_count = sum(1 for entity in first.ir.entities if TEMPLATE_ARROW_TAG in entity.tags)
    second_count = sum(1 for entity in second.ir.entities if TEMPLATE_ARROW_TAG in entity.tags)
    assert first_count == second_count
