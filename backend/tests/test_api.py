"""HTTP-level coverage of the FastAPI endpoints (storage redirected to tmp)."""
from __future__ import annotations

import struct
from io import BytesIO

from PIL import Image, ImageDraw


def png_bytes(width: int = 320, height: int = 200) -> bytes:
    return b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + struct.pack(">II", width, height) + b"\x08\x02\x00\x00\x00"


def table_page_png_bytes() -> bytes:
    img = Image.new("L", (1000, 700), 255)
    draw = ImageDraw.Draw(img)
    draw.rectangle((40, 40, 960, 680), outline=0, width=2)
    draw.rectangle((120, 120, 900, 650), outline=0, width=2)
    draw.rectangle((690, 130, 890, 350), outline=0, width=2)
    for x in range(730, 890, 40):
        draw.line((x, 130, x, 350), fill=0, width=1)
    for y in range(160, 350, 30):
        draw.line((690, y, 890, y), fill=0, width=1)
    draw.rectangle((520, 545, 900, 650), outline=0, width=2)
    for x in range(570, 900, 70):
        draw.line((x, 545, x, 650), fill=0, width=1)
    for y in range(575, 650, 25):
        draw.line((520, y, 900, y), fill=0, width=1)
    out = BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def section_page_png_bytes() -> bytes:
    img = Image.new("L", (1000, 700), 255)
    draw = ImageDraw.Draw(img)
    draw.rectangle((40, 40, 960, 680), outline=0, width=2)
    draw.rectangle((120, 120, 900, 650), outline=0, width=2)
    body = [(250, 245), (310, 245), (325, 270), (365, 270), (365, 455), (325, 455), (310, 485), (250, 485)]
    draw.line(body + [body[0]], fill=0, width=4)
    draw.rectangle((270, 315, 348, 405), outline=0, width=4)
    draw.line((235, 365, 380, 365), fill=0, width=2)
    draw.line((300, 225, 300, 505), fill=0, width=2)
    for offset in range(0, 90, 16):
        draw.line((255 + offset, 300, 305 + offset, 250), fill=0, width=2)
        draw.line((255 + offset, 480, 305 + offset, 430), fill=0, width=2)
    out = BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def scan_cad_page_png_bytes() -> bytes:
    img = Image.new("L", (1000, 700), 255)
    draw = ImageDraw.Draw(img)
    draw.rectangle((40, 40, 960, 680), outline=0, width=2)
    draw.rectangle((120, 120, 900, 650), outline=0, width=2)
    draw.rectangle((690, 130, 890, 350), outline=0, width=2)
    for x in range(730, 890, 40):
        draw.line((x, 130, x, 350), fill=0, width=1)
    for y in range(160, 350, 30):
        draw.line((690, y, 890, y), fill=0, width=1)
    draw.rectangle((520, 545, 900, 650), outline=0, width=2)
    for x in range(570, 900, 70):
        draw.line((x, 545, x, 650), fill=0, width=1)
    for y in range(575, 650, 25):
        draw.line((520, y, 900, y), fill=0, width=1)
    body = [(250, 245), (310, 245), (325, 270), (365, 270), (365, 455), (325, 455), (310, 485), (250, 485)]
    draw.line(body + [body[0]], fill=0, width=4)
    draw.rectangle((270, 315, 348, 405), outline=0, width=4)
    draw.line((235, 365, 380, 365), fill=0, width=2)
    draw.line((300, 225, 300, 505), fill=0, width=2)
    for offset in range(0, 90, 16):
        draw.line((255 + offset, 300, 305 + offset, 250), fill=0, width=2)
        draw.line((255 + offset, 480, 305 + offset, 430), fill=0, width=2)
    out = BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def test_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_create_project_with_prompt_runs_the_operation(client):
    res = client.post("/api/projects", json={"name": "t", "prompt": "创建 100 60 8 两个孔"})
    assert res.status_code == 200
    body = res.json()
    holes = [e for e in body["ir"]["entities"] if e["type"] == "circle"]
    assert len(holes) == 2  # regression: "100" must not collapse this to 1 hole


def test_chat_full_loop(client):
    pid = client.post("/api/projects", json={"name": "t", "prompt": ""}).json()["project_id"]
    res = client.post(f"/api/projects/{pid}/chat", json={"message": "把左边孔直径改成 10"})
    assert res.status_code == 200
    body = res.json()
    assert body["operations"][0]["operation"] == "modify_entity"
    assert body["diffs"][0]["path"] == "hole_1.r"


def test_chat_uses_llm_planner_when_available(client, monkeypatch):
    from app import main
    from app.models import Operation

    def fake_llm(message, ir):
        return [Operation(operation="move_entity", entity_id="hole_2", dx=12, reason="llm")], "已右移右孔"

    monkeypatch.setattr(main, "plan_operations_llm", fake_llm)
    pid = client.post("/api/projects", json={"name": "t", "prompt": ""}).json()["project_id"]
    res = client.post(f"/api/projects/{pid}/chat", json={"message": "把右边孔右移 12（自由表达）"})
    assert res.status_code == 200
    body = res.json()
    assert body["reply"] == "已右移右孔"
    assert body["operations"][0]["operation"] == "move_entity"


def test_chat_falls_back_to_deterministic_when_llm_unavailable(client, monkeypatch):
    from app import main
    from app.llm_agent import LlmUnavailable

    def unavailable(message, ir):
        raise LlmUnavailable("no key in test env")

    monkeypatch.setattr(main, "plan_operations_llm", unavailable)
    pid = client.post("/api/projects", json={"name": "t", "prompt": ""}).json()["project_id"]
    res = client.post(f"/api/projects/{pid}/chat", json={"message": "把左边孔直径改成 10"})
    assert res.status_code == 200
    assert res.json()["operations"][0]["operation"] == "modify_entity"


def test_chat_can_edit_mechanical_dimension_semantic_text(client):
    from app.models import DimensionBinding, MechanicalDimensionObject, ParsedDimensionValue, TextEntity
    from app.storage import load_project, save_project

    pid = client.post("/api/projects", json={"name": "t", "prompt": ""}).json()["project_id"]
    project = load_project(pid)
    project.ir.entities.append(
        TextEntity(id="dim_text_49", layer="dimensions", x=10, y=10, text="φ49", height=2.4)
    )
    parsed = ParsedDimensionValue(kind="diameter", raw_text="φ49", nominal=49)
    project.dimension_bindings = [
        DimensionBinding(
            id="dim_binding_00000",
            dimension_line_id="dim_line_1",
            text_id="dim_text_49",
            text="φ49",
            parsed=parsed,
            confidence=0.9,
            kind="diameter",
            line_x1=0,
            line_y1=0,
            line_x2=20,
            line_y2=0,
        )
    ]
    project.mechanical_dimensions = [
        MechanicalDimensionObject(
            id="mechanical_dimension_dim_binding_00000",
            binding_id="dim_binding_00000",
            text="φ49",
            parsed=parsed,
            confidence=0.9,
            kind="diameter",
            dimension_line_id="dim_line_1",
            text_id="dim_text_49",
        )
    ]
    save_project(project)

    res = client.post(f"/api/projects/{pid}/chat", json={"message": "把49改成50"})

    assert res.status_code == 200
    body = res.json()
    assert body["operations"][0]["operation"] == "modify_entity"
    assert body["diffs"][0]["path"] == "dim_text_49.text"
    assert body["project"]["mechanical_dimensions"][0]["text"] == "φ50"
    assert body["project"]["mechanical_dimensions"][0]["parsed"]["nominal"] == 50


def test_chat_drives_complete_dimension_and_undoes_transaction(client):
    from app.models import (
        DimensionBinding,
        LineEntity,
        MechanicalDimensionObject,
        MechanicalDrawingIR,
        ParsedDimensionValue,
        TextEntity,
    )
    from app.storage import load_project, save_project

    pid = client.post("/api/projects", json={"name": "drive", "prompt": ""}).json()["project_id"]
    project = load_project(pid)
    project.ir.entities = [
        LineEntity(id="outline", layer="OUTLINE", x1=0, y1=0, x2=244, y2=0),
        LineEntity(id="ext_start", layer="DIMENSION", x1=0, y1=0, x2=0, y2=20),
        LineEntity(id="ext_end", layer="DIMENSION", x1=244, y1=0, x2=244, y2=20),
        LineEntity(id="dim_line", layer="DIMENSION", x1=0, y1=20, x2=244, y2=20),
        TextEntity(id="dim_text", layer="DIMENSION", x=120, y=23, text="244"),
    ]
    parsed = ParsedDimensionValue(kind="linear", raw_text="244", nominal=244)
    binding = DimensionBinding(
        id="dim_binding_244",
        dimension_line_id="dim_line",
        text_id="dim_text",
        text="244",
        parsed=parsed,
        confidence=0.98,
        kind="linear",
        line_x1=0,
        line_y1=20,
        line_x2=244,
        line_y2=20,
        text_x=120,
        text_y=23,
    )
    dimension = MechanicalDimensionObject(
        id="mechanical_dimension_dim_binding_244",
        binding_id=binding.id,
        kind="linear",
        text="244",
        parsed=parsed,
        confidence=0.98,
        dimension_line_id="dim_line",
        text_id="dim_text",
        extension_line_ids=["ext_start", "ext_end"],
        measured_geometry_ids=["outline"],
        measurement_points=[[0, 0], [244, 0]],
        dimension_line_point=[122, 20],
        orientation="horizontal",
        dxf_dimension_type="linear",
        export_ready=True,
        status="complete",
    )
    project.dimension_bindings = [binding]
    project.mechanical_ir = MechanicalDrawingIR(dimensions=[dimension])
    project.mechanical_dimensions = [dimension.model_copy(deep=True)]
    save_project(project)

    response = client.post(f"/api/projects/{pid}/chat", json={"message": "把244改成250"})

    assert response.status_code == 200
    body = response.json()
    outline = next(entity for entity in body["project"]["ir"]["entities"] if entity["id"] == "outline")
    assert outline["x2"] == 250
    assert body["project"]["mechanical_ir"]["dimensions"][0]["measured_value"] == 250
    assert body["project"]["mechanical_transactions"][0]["validation"]["passed"] is True
    assert "244 → 250" in body["reply"]
    assert "本地精确解析" in body["reply"]

    undo = client.post(f"/api/projects/{pid}/chat", json={"message": "撤销"})
    assert undo.status_code == 200
    restored = undo.json()["project"]
    outline = next(entity for entity in restored["ir"]["entities"] if entity["id"] == "outline")
    assert outline["x2"] == 244
    assert restored["mechanical_transactions"] == []


def test_dimension_semantics_endpoint_returns_unified_mechanical_ir(client):
    from app.models import LineEntity, TextEntity
    from app.storage import load_project, save_project

    pid = client.post("/api/projects", json={"name": "semantic ir", "prompt": ""}).json()["project_id"]
    project = load_project(pid)
    project.ir.entities = [
        LineEntity(
            id="dim_line",
            layer="dimensions",
            x1=0,
            y1=10,
            x2=20,
            y2=10,
            tags=["dimensions"],
        ),
        LineEntity(
            id="arrow_left",
            layer="dimensions",
            x1=0,
            y1=10,
            x2=2,
            y2=11,
            tags=["dimension_arrow", "arrowhead"],
            metadata={
                "arrow_candidate_id": "left",
                "score": 0.94,
                "tip_x": 0,
                "tip_y": 10,
                "direction_x": 1,
                "direction_y": 0,
                "size_mm": 2.5,
            },
        ),
        LineEntity(
            id="arrow_right",
            layer="dimensions",
            x1=20,
            y1=10,
            x2=18,
            y2=11,
            tags=["dimension_arrow", "arrowhead"],
            metadata={
                "arrow_candidate_id": "right",
                "score": 0.94,
                "tip_x": 20,
                "tip_y": 10,
                "direction_x": -1,
                "direction_y": 0,
                "size_mm": 2.5,
            },
        ),
        TextEntity(id="dim_text", layer="dimensions", x=9, y=12, text="20"),
        LineEntity(id="extension_left", layer="dimensions", x1=0, y1=0, x2=0, y2=10),
        LineEntity(id="extension_right", layer="dimensions", x1=20, y1=0, x2=20, y2=10),
        LineEntity(id="measured_outline", layer="outline", x1=0, y1=0, x2=20, y2=0),
    ]
    save_project(project)

    response = client.post(f"/api/projects/{pid}/semantics/dimensions")

    assert response.status_code == 200
    body = response.json()["project"]
    assert body["mechanical_ir"]["schema_version"] == "1.0"
    dimensions = body["mechanical_ir"]["dimensions"]
    dimension = next(item for item in dimensions if item["dimension_line_id"] == "dim_line")
    assert dimension["status"] == "complete"
    assert set(dimension["extension_line_ids"]) == {"extension_left", "extension_right"}
    assert dimension["measured_geometry_ids"] == ["measured_outline"]
    assert dimension["measurement_points"] == [[0.0, 0.0], [20.0, 0.0]]
    assert dimension["dxf_dimension_type"] == "linear"
    assert dimension["export_ready"] is True
    assert len(dimension["arrowheads"]) == 2


def test_chat_unparseable_is_a_clean_noop(client):
    pid = client.post("/api/projects", json={"name": "t", "prompt": ""}).json()["project_id"]
    res = client.post(f"/api/projects/{pid}/chat", json={"message": "今天天气不错"})
    assert res.status_code == 200
    body = res.json()
    assert body["operations"] == []
    assert body["diffs"] == []
    assert body["reply"]


def test_chat_on_missing_entity_returns_400_not_500(client):
    """Regression: operating on a non-existent entity must be a clean 400."""
    pid = client.post("/api/projects", json={"name": "t", "prompt": ""}).json()["project_id"]
    res = client.post(f"/api/projects/{pid}/chat", json={"message": "删除 hole_99"})
    assert res.status_code == 400
    assert "hole_99" in res.json()["detail"]


def test_chat_on_missing_project_returns_404(client):
    res = client.post("/api/projects/does-not-exist/chat", json={"message": "删除 hole_1"})
    assert res.status_code == 404


def test_gear_drawing_then_eval_passes(client):
    pid = client.post("/api/projects", json={"name": "t", "prompt": ""}).json()["project_id"]
    client.post(f"/api/projects/{pid}/chat", json={"message": "创建齿轮零件图"})
    res = client.get(f"/api/projects/{pid}/eval")
    assert res.status_code == 200
    report = res.json()
    assert report["passed"] is True
    assert report["overall_score"] == 1.0


def test_eval_on_missing_project_returns_404(client):
    res = client.get("/api/projects/nope/eval")
    assert res.status_code == 404


def test_scan_eval_endpoint_returns_scan_targets(client):
    pid = client.post("/api/projects", json={"name": "t", "prompt": ""}).json()["project_id"]

    res = client.get(f"/api/projects/{pid}/eval/scan")

    assert res.status_code == 200
    report = res.json()
    assert {target["name"] for target in report["targets"]} == {
        "scan_trace",
        "scan_visual_match",
        "scan_tables",
        "scan_lineweights",
        "scan_primitive_quality",
        "scan_noise",
    }


def test_exports_and_files_are_served(client):
    pid = client.post("/api/projects", json={"name": "t", "prompt": ""}).json()["project_id"]
    paths = client.get(f"/api/projects/{pid}/exports").json()
    assert paths["dxf_url"].endswith("output.dxf")
    assert client.get(f"/api/projects/{pid}/files/preview.svg").status_code == 200
    assert client.get(f"/api/projects/{pid}/files/output.dxf").status_code == 200


def test_missing_file_returns_404(client):
    pid = client.post("/api/projects", json={"name": "t", "prompt": ""}).json()["project_id"]
    assert client.get(f"/api/projects/{pid}/files/nope.dxf").status_code == 404


def test_list_and_get_project(client):
    pid = client.post("/api/projects", json={"name": "alpha", "prompt": ""}).json()["project_id"]
    listed = client.get("/api/projects").json()
    assert pid in {p["project_id"] for p in listed}

    fetched = client.get(f"/api/projects/{pid}")
    assert fetched.status_code == 200
    assert fetched.json()["project_id"] == pid


def test_get_missing_project_returns_404(client):
    assert client.get("/api/projects/ghost").status_code == 404


def test_create_with_bad_prompt_op_returns_400(client):
    # An explicit op against a non-existent entity at creation time -> clean 400.
    res = client.post("/api/projects", json={"name": "t", "prompt": "删除 hole_99"})
    assert res.status_code == 400


def test_upload_attaches_source_image(client):
    pid = client.post("/api/projects", json={"name": "t", "prompt": ""}).json()["project_id"]
    files = {"file": ("ref.png", png_bytes(), "image/png")}
    res = client.post(f"/api/projects/{pid}/upload", files=files)
    assert res.status_code == 200
    body = res.json()
    assert body["source_file"].endswith("_source.png")
    source = body["source_image"]
    assert source and source.endswith("_reference.png")

    served = client.get(source)  # the uploaded file is served back
    assert served.status_code == 200


def test_upload_image_then_analyze_returns_candidate_boxes(client):
    pid = client.post("/api/projects", json={"name": "t", "prompt": ""}).json()["project_id"]
    files = {"file": ("ref.png", table_page_png_bytes(), "image/png")}
    client.post(f"/api/projects/{pid}/upload", files=files)

    res = client.get(f"/api/projects/{pid}/analyze")
    assert res.status_code == 200
    body = res.json()
    assert body["image_width"] == 1000
    assert body["image_height"] == 700
    assert {box["target"] for box in body["boxes"]} == {
        "title_block",
        "parameter_table",
        "section_view",
        "circular_view",
        "dimensions",
    }
    assert body["overlay_image"].endswith("/analysis_overlay.png")
    assert body["preprocessed_image"].endswith("/analysis_preprocessed.png")
    assert isinstance(body["deskew_angle"], float)
    assert len(body["frame"]) == 4
    assert client.get(body["overlay_image"]).headers["content-type"].startswith("image/png")


def test_reconstruct_tables_updates_project_ir_and_checks_layout(client):
    pid = client.post("/api/projects", json={"name": "t", "prompt": ""}).json()["project_id"]
    files = {"file": ("ref.png", table_page_png_bytes(), "image/png")}
    client.post(f"/api/projects/{pid}/upload", files=files)

    res = client.post(f"/api/projects/{pid}/reconstruct/tables")
    assert res.status_code == 200
    body = res.json()
    assert body["layout_passed"] is True
    assert body["warnings"] == []
    assert {region["target"] for region in body["regions"]} == {"parameter_table", "title_block"}
    entities = body["project"]["ir"]["entities"]
    assert any(entity["group"] == "parameter_table" for entity in entities)
    assert any(entity["group"] == "title_block" for entity in entities)


def test_reconstruct_section_updates_project_ir_with_cv_lines(client):
    pid = client.post("/api/projects", json={"name": "t", "prompt": "创建齿轮零件图"}).json()["project_id"]
    files = {"file": ("ref.png", section_page_png_bytes(), "image/png")}
    client.post(f"/api/projects/{pid}/upload", files=files)

    res = client.post(f"/api/projects/{pid}/reconstruct/section")
    assert res.status_code == 200
    body = res.json()
    assert body["line_count"] >= 12
    assert body["hatch_count"] >= 4
    assert body["region"]["target"] == "section_view"
    section_entities = [entity for entity in body["project"]["ir"]["entities"] if entity.get("group") == "section_view"]
    assert section_entities
    assert all(entity["id"].startswith("cv_section_") for entity in section_entities)
    assert any(entity["layer"] == "HATCH" for entity in section_entities)


def test_reconstruct_scan_generates_cad_exports(client, monkeypatch):
    from app.models import PolylineEntity

    monkeypatch.setattr(
        "app.scan_cad._run_vtracer_entities",
        lambda image_path, output_dir, warnings: [
            PolylineEntity(
                id="ref_trace_00000",
                layer="reference_trace",
                points=[[0, 0], [10, 0]],
                group="reference_trace",
                tags=["vtracer"],
            )
        ],
    )
    monkeypatch.setattr(
        "app.scan_cad._run_autotrace_entities",
        lambda image_path, output_dir, warnings: [
            PolylineEntity(
                id="editable_line_00000",
                layer="editable_linework",
                points=[[0, 0], [0, 10]],
                group="editable_linework",
                tags=["autotrace"],
            )
        ],
    )
    pid = client.post("/api/projects", json={"name": "t", "prompt": ""}).json()["project_id"]
    files = {"file": ("scan.png", scan_cad_page_png_bytes(), "image/png")}
    client.post(f"/api/projects/{pid}/upload", files=files)

    res = client.post(f"/api/projects/{pid}/reconstruct/scan")

    assert res.status_code == 200
    body = res.json()
    assert body["entity_count"] == len(body["project"]["ir"]["entities"])
    assert body["trace_count"] == 2
    assert body["structured_counts"]["reference_trace"] == 1
    assert body["structured_counts"]["editable_linework"] == 1
    assert body["structured_counts"]["tables"] > 0
    assert body["structured_counts"]["section_view"] > 0
    assert any(entity.get("group") == "reference_trace" for entity in body["project"]["ir"]["entities"])
    assert any(entity.get("group") == "editable_linework" for entity in body["project"]["ir"]["entities"])
    assert client.get(f"/api/projects/{pid}/files/preview.svg").status_code == 200
    assert client.get(f"/api/projects/{pid}/files/output.dxf").status_code == 200


def test_promote_scan_generates_editable_primitives(client, monkeypatch):
    import math

    from app.models import PolylineEntity

    monkeypatch.setattr("app.scan_cad._run_vtracer_entities", lambda image_path, output_dir, warnings: [])
    monkeypatch.setattr(
        "app.scan_cad._run_autotrace_entities",
        lambda image_path, output_dir, warnings: [
            PolylineEntity(
                id="editable_line_00000",
                layer="editable_linework",
                points=[[20, 20], [34, 20.01], [48, 20]],
                group="editable_linework",
                tags=["autotrace"],
            ),
            PolylineEntity(
                id="editable_circle_00000",
                layer="editable_linework",
                points=[
                    [70 + math.cos(index / 32 * math.tau) * 8, 55 + math.sin(index / 32 * math.tau) * 8]
                    for index in range(33)
                ],
                closed=True,
                group="editable_linework",
                tags=["autotrace"],
            ),
        ],
    )
    pid = client.post("/api/projects", json={"name": "t", "prompt": ""}).json()["project_id"]
    files = {"file": ("scan.png", scan_cad_page_png_bytes(), "image/png")}
    client.post(f"/api/projects/{pid}/upload", files=files)
    client.post(f"/api/projects/{pid}/reconstruct/scan")

    res = client.post(f"/api/projects/{pid}/promote/scan")

    assert res.status_code == 200
    body = res.json()
    assert body["source_count"] >= 2
    assert body["promoted_counts"]["line"] >= 1
    assert body["promoted_counts"]["circle"] >= 1
    promoted = [entity for entity in body["project"]["ir"]["entities"] if entity.get("group") == "promoted_geometry"]
    assert {entity["type"] for entity in promoted} >= {"line", "circle"}
    assert all(entity["layer"] == "OUTLINE" for entity in promoted)
    assert client.get(f"/api/projects/{pid}/files/preview.svg").status_code == 200
    assert client.get(f"/api/projects/{pid}/files/output.dxf").status_code == 200


def test_dimension_semantics_endpoint_returns_bindings_field(client):
    pid = client.post("/api/projects", json={"name": "t", "prompt": ""}).json()["project_id"]

    res = client.post(f"/api/projects/{pid}/semantics/dimensions")

    assert res.status_code == 200
    body = res.json()
    assert body["bindings"] == []
    assert body["project"]["dimension_bindings"] == []
    assert body["warnings"]


def test_cad_pipeline_runs_reconstruct_promote_ocr_and_dimension_binding(client, monkeypatch):
    from types import SimpleNamespace

    from app.arrow_cv import ArrowTemplateDetection
    from app.models import DrawingIR, LineEntity, OcrRegion, TextEntity
    from app.ocr import OcrRun
    from app.promote import ScanPromotion
    from app.scan_cad import ScanCadReconstruction
    from app.table_ocr import TableOcrRun

    pid = client.post("/api/projects", json={"name": "t", "prompt": ""}).json()["project_id"]
    files = {"file": ("scan.png", scan_cad_page_png_bytes(), "image/png")}
    client.post(f"/api/projects/{pid}/upload", files=files)

    promoted_ir = DrawingIR(
        entities=[
            LineEntity(
                id="dim_line",
                layer="promoted_geometry",
                x1=0,
                y1=0,
                x2=20,
                y2=0,
                group="promoted_geometry",
            ),
            LineEntity(
                id="arrow_left",
                layer="promoted_geometry",
                x1=0,
                y1=0,
                x2=2,
                y2=1,
                group="promoted_geometry",
                tags=["dimension_arrow", "arrowhead"],
                metadata={
                    "arrow_candidate_id": "arrow_left",
                    "tip_x": 0,
                    "tip_y": 0,
                    "direction_x": -1,
                    "direction_y": 0,
                    "score": 0.95,
                    "size_mm": 2.0,
                },
            ),
            LineEntity(
                id="arrow_right",
                layer="promoted_geometry",
                x1=20,
                y1=0,
                x2=18,
                y2=1,
                group="promoted_geometry",
                tags=["dimension_arrow", "arrowhead"],
                metadata={
                    "arrow_candidate_id": "arrow_right",
                    "tip_x": 20,
                    "tip_y": 0,
                    "direction_x": 1,
                    "direction_y": 0,
                    "score": 0.95,
                    "size_mm": 2.0,
                },
            ),
            TextEntity(id="dim_text", layer="text", x=9, y=2, text="φ25 +0.021"),
        ]
    )

    monkeypatch.setattr("app.main.analyze_reference", lambda project, uploads_dir, output_dir=None: SimpleNamespace(boxes=[]))
    monkeypatch.setattr(
        "app.main.reconstruct_scan_cad_from_reference",
        lambda project, uploads_dir, output_dir=None: ScanCadReconstruction(
            ir=DrawingIR(entities=[]),
            entity_count=0,
            trace_count=0,
            structured_counts={},
            warnings=[],
        ),
    )
    monkeypatch.setattr(
        "app.main.promote_scan_primitives",
        lambda project: ScanPromotion(
            ir=promoted_ir,
            promoted_counts={"line": 1, "circle": 0, "arrow": 2},
            source_count=3,
            warnings=[],
        ),
    )
    monkeypatch.setattr(
        "app.main.detect_arrowheads_from_reference",
        lambda project, uploads_dir: ArrowTemplateDetection(ir=project.ir, detected_count=2, warnings=[]),
    )
    monkeypatch.setattr(
        "app.main.run_project_ocr",
        lambda project, uploads_dir, language_hint="auto", engine_hint="auto": OcrRun(
            regions=[
                OcrRegion(
                    target="dimensions",
                    label="尺寸标注",
                    text="φ25 +0.021",
                    confidence=0.9,
                    x=0.1,
                    y=0.1,
                    width=0.2,
                    height=0.1,
                )
            ],
            warnings=[],
        ),
    )
    monkeypatch.setattr(
        "app.main.extract_table_ocr_from_reference",
        lambda project, uploads_dir, language_hint="auto", engine_hint="auto": TableOcrRun(cells=[], warnings=[]),
    )

    res = client.post(f"/api/projects/{pid}/pipeline/cad", json={"language": "zh"})

    assert res.status_code == 200
    body = res.json()
    step_names = [step["name"] for step in body["steps"]]
    assert step_names == [
        "analyze",
        "scan_reconstruct",
        "promote",
        "arrow_template",
        "ocr",
        "table_ocr",
        "title_block",
        "title_block_render",
        "table_text",
        "dimension_semantics",
        "dimension_arrow_render",
    ]
    assert body["project"]["dimension_bindings"][0]["kind"] == "diameter"
    assert body["project"]["dimension_bindings"][0]["text_id"] == "dim_text"
    assert body["project"]["dimension_bindings"][0]["binding_method"] == "graph_text_arrow_line"
    assert body["project"]["dimension_bindings"][0]["graph_path"][0] == "text:dim_text"
    assert body["project"]["dimension_bindings"][0]["graph_path"][-1] == "line:dim_line"
    rendered_arrows = [
        entity
        for entity in body["project"]["ir"]["entities"]
        if "dimension_arrow_render" in entity.get("tags", [])
    ]
    assert len(rendered_arrows) == 2
    assert {entity["layer"] for entity in rendered_arrows} == {"DIMENSION"}
    assert {entity["type"] for entity in rendered_arrows} == {"polyline"}
    assert all(entity["closed"] is True for entity in rendered_arrows)
    assert all("solid_fill" in entity["tags"] for entity in rendered_arrows)
    assert len(body["project"]["mechanical_dimensions"]) == 1
    assert body["project"]["mechanical_dimensions"][0]["binding_id"] == body["project"]["dimension_bindings"][0]["id"]
    assert len(body["project"]["mechanical_dimensions"][0]["arrowheads"]) == 2


def test_cad_pipeline_without_upload_returns_400(client):
    pid = client.post("/api/projects", json={"name": "t", "prompt": ""}).json()["project_id"]

    res = client.post(f"/api/projects/{pid}/pipeline/cad")

    assert res.status_code == 400
    assert "Upload" in res.json()["detail"]


def test_vectorizer_benchmark_endpoint_returns_open_source_baselines(client, monkeypatch):
    from app.vectorizer_benchmark import VectorizerBenchmark, VectorizerResult

    pid = client.post("/api/projects", json={"name": "t", "prompt": ""}).json()["project_id"]
    files = {"file": ("scan.png", scan_cad_page_png_bytes(), "image/png")}
    client.post(f"/api/projects/{pid}/upload", files=files)

    def fake_benchmark(project, uploads_dir):
        return VectorizerBenchmark(
            project_id=project.project_id,
            prepared_image_url=f"/api/projects/{project.project_id}/files/benchmark_normalized_binary.png",
            results=[
                VectorizerResult(
                    name="vtracer",
                    status="ok",
                    elapsed_sec=0.1,
                    svg_url=f"/api/projects/{project.project_id}/files/benchmark_vtracer.svg",
                    dxf_entity_count=12,
                    svg_path_count=10,
                ),
                VectorizerResult(name="potrace", status="skipped", detail="not installed"),
            ],
        )

    monkeypatch.setattr("app.main.run_project_vectorizer_benchmark", fake_benchmark)

    res = client.post(f"/api/projects/{pid}/benchmark/vectorizers")

    assert res.status_code == 200
    body = res.json()
    assert body["project_id"] == pid
    assert body["results"][0]["name"] == "vtracer"
    assert body["results"][0]["dxf_entity_count"] == 12
    assert body["results"][1]["status"] == "skipped"


def test_upload_pdf_renders_first_page_preview(client, monkeypatch):
    from app import reference
    from app.ingest import SourceClassification, SourceKind

    def fake_render(source_path, preview_path):
        assert source_path.name.endswith("_source.pdf")
        preview_path.write_bytes(png_bytes(1200, 900))

    monkeypatch.setattr(reference, "_render_pdf_first_page", fake_render)
    monkeypatch.setattr(
        reference,
        "classify_source",
        lambda path: SourceClassification(SourceKind.VECTOR_PDF, 999, 0, 0.0, "stub"),
    )
    pid = client.post("/api/projects", json={"name": "t", "prompt": ""}).json()["project_id"]
    files = {"file": ("ref.pdf", b"%PDF fake", "application/pdf")}

    res = client.post(f"/api/projects/{pid}/upload", files=files)
    assert res.status_code == 200
    body = res.json()
    assert body["source_file"].endswith("_source.pdf")
    assert body["source_image"].endswith("_reference.png")
    assert body["source_kind"] == "vector_pdf"


def _vector_pdf_bytes() -> bytes:
    import pymupdf

    document = pymupdf.open()
    page = document.new_page(width=300, height=200)
    for i in range(40):
        y = 10 + i * 4
        page.draw_line(pymupdf.Point(10, y), pymupdf.Point(290, y))
    page.insert_text((20, 180), "LJT01.01")
    data = document.tobytes()
    document.close()
    return data


def test_upload_vector_pdf_sets_source_kind_and_extracts(client):
    pid = client.post("/api/projects", json={"name": "t", "prompt": ""}).json()["project_id"]
    files = {"file": ("drawing.pdf", _vector_pdf_bytes(), "application/pdf")}

    upload = client.post(f"/api/projects/{pid}/upload", files=files).json()
    assert upload["source_kind"] == "vector_pdf"

    res = client.post(f"/api/projects/{pid}/reconstruct/vector")
    assert res.status_code == 200
    body = res.json()
    assert body["entity_count"] > 40
    types = {e["type"] for e in body["project"]["ir"]["entities"]}
    assert "line" in types and "text" in types


def test_chat_template_create_does_not_overwrite_vector_pdf_import(client):
    from app import main

    pid = client.post("/api/projects", json={"name": "t", "prompt": ""}).json()["project_id"]
    files = {"file": ("drawing.pdf", _vector_pdf_bytes(), "application/pdf")}
    client.post(f"/api/projects/{pid}/upload", files=files)
    vectorized = client.post(f"/api/projects/{pid}/reconstruct/vector")
    assert vectorized.status_code == 200

    project_path = main.PROJECTS_DIR / pid
    preview_before = (project_path / "preview.svg").read_bytes()
    dxf_before = (project_path / "output.dxf").read_bytes()

    res = client.post(f"/api/projects/{pid}/chat", json={"message": "画圆柱直齿轮图"})
    assert res.status_code == 200
    body = res.json()
    assert body["operations"] == []
    assert body["diffs"] == []
    assert "覆盖" in body["reply"]
    assert len(body["project"]["ir"]["entities"]) == vectorized.json()["entity_count"]
    assert (project_path / "preview.svg").read_bytes() == preview_before
    assert (project_path / "output.dxf").read_bytes() == dxf_before


def test_reconstruct_vector_rejects_non_vector_source(client):
    pid = client.post("/api/projects", json={"name": "t", "prompt": ""}).json()["project_id"]
    files = {"file": ("ref.png", png_bytes(1000, 800), "image/png")}
    client.post(f"/api/projects/{pid}/upload", files=files)

    res = client.post(f"/api/projects/{pid}/reconstruct/vector")
    assert res.status_code == 400
    assert "vector" in res.json()["detail"].lower()


def test_analyze_without_upload_returns_400(client):
    pid = client.post("/api/projects", json={"name": "t", "prompt": ""}).json()["project_id"]
    res = client.get(f"/api/projects/{pid}/analyze")
    assert res.status_code == 400


def test_ocr_endpoint_attaches_regions_without_rewriting_exports(client, monkeypatch):
    from app import main
    from app.models import OcrRegion
    from app.ocr import OcrRun

    pid = client.post("/api/projects", json={"name": "t", "prompt": ""}).json()["project_id"]
    files = {"file": ("ref.png", table_page_png_bytes(), "image/png")}
    client.post(f"/api/projects/{pid}/upload", files=files)
    project_path = main.PROJECTS_DIR / pid
    preview_before = (project_path / "preview.svg").read_bytes()
    dxf_before = (project_path / "output.dxf").read_bytes()

    def fake_ocr(project, uploads_dir, language_hint="auto", engine_hint="auto"):
        return OcrRun(
            regions=[
                OcrRegion(
                    target="title_block",
                    label="标题栏",
                    text="LJT01.01",
                    confidence=0.91,
                    x=0.5,
                    y=0.8,
                    width=0.3,
                    height=0.1,
                    language="chi_sim+eng",
                )
            ],
            warnings=["mock warning"],
        )

    monkeypatch.setattr(main, "run_project_ocr", fake_ocr)

    res = client.post(f"/api/projects/{pid}/ocr")

    assert res.status_code == 200
    body = res.json()
    assert body["regions"][0]["text"] == "LJT01.01"
    assert body["warnings"] == ["mock warning"]
    assert body["project"]["ocr_regions"][0]["confidence"] == 0.91
    assert (project_path / "preview.svg").read_bytes() == preview_before
    assert (project_path / "output.dxf").read_bytes() == dxf_before


def test_table_ocr_endpoint_attaches_cells_without_rewriting_exports(client, monkeypatch):
    from app import main
    from app.models import TableCellOcr
    from app.table_ocr import TableOcrRun

    pid = client.post("/api/projects", json={"name": "t", "prompt": ""}).json()["project_id"]
    files = {"file": ("ref.png", table_page_png_bytes(), "image/png")}
    client.post(f"/api/projects/{pid}/upload", files=files)
    project_path = main.PROJECTS_DIR / pid
    preview_before = (project_path / "preview.svg").read_bytes()
    dxf_before = (project_path / "output.dxf").read_bytes()

    def fake_table_ocr(project, uploads_dir, language_hint="auto", engine_hint="auto"):
        return TableOcrRun(
            cells=[
                TableCellOcr(
                    target="parameter_table",
                    row=0,
                    col=1,
                    text="29",
                    confidence=0.93,
                    x=0.7,
                    y=0.2,
                    width=0.05,
                    height=0.04,
                    engine="paddleocr",
                    language="zh",
                )
            ],
            warnings=["table mock warning"],
        )

    monkeypatch.setattr(main, "extract_table_ocr_from_reference", fake_table_ocr)

    res = client.post(f"/api/projects/{pid}/ocr/tables", json={"language": "zh", "engine": "auto"})

    assert res.status_code == 200
    body = res.json()
    assert body["cells"][0]["text"] == "29"
    assert body["warnings"] == ["table mock warning"]
    assert body["project"]["table_ocr_cells"][0]["confidence"] == 0.93
    assert (project_path / "preview.svg").read_bytes() == preview_before
    assert (project_path / "output.dxf").read_bytes() == dxf_before


def test_ocr_without_upload_returns_400(client):
    pid = client.post("/api/projects", json={"name": "t", "prompt": ""}).json()["project_id"]
    res = client.post(f"/api/projects/{pid}/ocr")
    assert res.status_code == 400


def test_analyze_missing_project_returns_404(client):
    assert client.get("/api/projects/ghost/analyze").status_code == 404


def test_analyze_missing_reference_file_returns_404(client, monkeypatch):
    from app import main

    def fail(*args, **kwargs):
        raise FileNotFoundError("Reference image not found")

    monkeypatch.setattr(main, "analyze_reference", fail)
    pid = client.post("/api/projects", json={"name": "t", "prompt": ""}).json()["project_id"]
    res = client.get(f"/api/projects/{pid}/analyze")
    assert res.status_code == 404


def test_upload_unsupported_reference_type_returns_400(client):
    pid = client.post("/api/projects", json={"name": "t", "prompt": ""}).json()["project_id"]
    files = {"file": ("notes.txt", b"hello", "text/plain")}
    res = client.post(f"/api/projects/{pid}/upload", files=files)
    assert res.status_code == 400


def test_upload_to_missing_project_returns_404(client):
    files = {"file": ("ref.png", b"data", "image/png")}
    assert client.post("/api/projects/ghost/upload", files=files).status_code == 404


def test_exports_on_missing_project_returns_404(client):
    assert client.get("/api/projects/ghost/exports").status_code == 404


def test_missing_upload_file_returns_404(client):
    assert client.get("/api/uploads/nope.png").status_code == 404
