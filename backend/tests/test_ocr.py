from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from PIL import Image, ImageDraw

from app.models import ProjectState, TextEntity, default_ir
from app.ocr import (
    _ocr_region_with_tesseract,
    _parse_paddle_result,
    _parse_tesseract_tsv,
    _resolve_language,
    _select_languages,
    _select_ocr_provider,
    run_project_ocr,
)


def _write_reference(path) -> None:
    img = Image.new("L", (1000, 700), 255)
    draw = ImageDraw.Draw(img)
    draw.rectangle((40, 40, 960, 680), outline=0, width=2)
    draw.rectangle((120, 120, 900, 650), outline=0, width=2)
    draw.rectangle((690, 130, 890, 350), outline=0, width=2)
    draw.rectangle((520, 545, 900, 650), outline=0, width=2)
    draw.text((720, 180), "z 29", fill=0)
    draw.text((740, 585), "LJT01.01", fill=0)
    img.save(path)


def _project() -> ProjectState:
    now = datetime.now(timezone.utc)
    return ProjectState(
        project_id="pid",
        name="demo",
        created_at=now,
        updated_at=now,
        source_image="/api/uploads/pid_reference.png",
        ir=default_ir(),
    )


def test_parse_tesseract_tsv_combines_words_and_confidence():
    payload = "\n".join(
        [
            "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext",
            "5\t1\t1\t1\t1\t1\t0\t0\t10\t10\t90\tLJT01.01",
            "5\t1\t1\t1\t1\t2\t12\t0\t10\t10\t70\t29",
            "5\t1\t1\t1\t1\t3\t20\t0\t10\t10\t-1\t",
        ]
    )

    text, confidence = _parse_tesseract_tsv(payload)

    assert text == "LJT01.01 29"
    assert confidence == 0.8


def test_parse_tesseract_tsv_ignores_invalid_confidence():
    payload = "\n".join(
        [
            "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext",
            "5\t1\t1\t1\t1\t1\t0\t0\t10\t10\tbad\tLJT01.01",
        ]
    )

    text, confidence = _parse_tesseract_tsv(payload)

    assert text == "LJT01.01"
    assert confidence == 0.0


def test_parse_paddle_result_combines_texts_and_scores():
    text, confidence = _parse_paddle_result(
        [
            {
                "rec_texts": ["圆柱直齿轮", "LJT01.01"],
                "rec_scores": [0.91, 0.83],
            }
        ]
    )

    assert text == "圆柱直齿轮 LJT01.01"
    assert confidence == 0.87


def test_parse_paddle_result_handles_legacy_shape_and_bad_scores():
    text, confidence = _parse_paddle_result(
        [
            object(),
            [
                ["box", ("", "bad")],
                ["box", ("legacy", "bad")],
                ["box", ("ok", 0.6)],
                ["bad"],
            ],
        ]
    )

    assert text == "legacy ok"
    assert confidence == 0.6


def test_auto_language_prefers_semantic_chinese_text():
    project = _project()
    project.ir.entities.append(
        TextEntity(
            id="imported_title",
            x=0,
            y=0,
            text="第 1 章 机械零件图",
            tags=["semantic_import"],
        )
    )

    assert _resolve_language(project, "auto") == "zh"
    assert _resolve_language(project, "en") == "en"


def test_auto_language_handles_semantic_english_and_unknown(monkeypatch):
    project = _project()
    project.ir.entities.append(
        TextEntity(
            id="imported_title",
            x=0,
            y=0,
            text="Spur gear drawing",
            tags=["semantic_import"],
        )
    )
    assert _resolve_language(project, "auto") == "en"

    project.ir.entities.clear()
    monkeypatch.setattr("app.ocr._paddleocr_available", lambda: True)
    assert _resolve_language(project, "auto") == "zh"
    monkeypatch.setattr("app.ocr._paddleocr_available", lambda: False)
    assert _resolve_language(project, "auto") == "en"


def test_select_languages_prefers_chinese_pack(monkeypatch):
    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout="List of available languages\neng\nchi_sim\n", stderr="")

    monkeypatch.setattr("app.ocr.subprocess.run", fake_run)

    language, warnings = _select_languages("/usr/bin/tesseract")

    assert language == "chi_sim+eng"
    assert warnings == []


def test_select_languages_warns_without_chinese_pack(monkeypatch):
    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout="List of available languages\neng\nsnum\n", stderr="")

    monkeypatch.setattr("app.ocr.subprocess.run", fake_run)

    language, warnings = _select_languages("/usr/bin/tesseract")

    assert language == "eng+snum"
    assert "chi_sim" in warnings[0]


def test_select_languages_handles_probe_error(monkeypatch):
    def fake_run(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("app.ocr.subprocess.run", fake_run)

    language, warnings = _select_languages("/usr/bin/tesseract")

    assert language == "eng"
    assert "Could not inspect" in warnings[0]


def test_select_languages_falls_back_to_first_available_language(monkeypatch):
    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout="List of available languages\ndeu\n", stderr="")

    monkeypatch.setattr("app.ocr.subprocess.run", fake_run)

    language, warnings = _select_languages("/usr/bin/tesseract")

    assert language == "deu"
    assert "using deu" in warnings[0]


def test_select_ocr_provider_reserves_edocr2_and_falls_back(monkeypatch):
    monkeypatch.setattr("app.ocr._edocr2_available", lambda: False)
    monkeypatch.setattr("app.ocr._paddleocr_available", lambda: False)
    monkeypatch.setattr("app.ocr.shutil.which", lambda name: "/usr/bin/tesseract")
    monkeypatch.setattr("app.ocr._select_languages", lambda tesseract: ("eng", []))
    warnings: list[str] = []

    provider = _select_ocr_provider("en", "edocr2", warnings)

    assert provider is not None
    assert provider.name == "tesseract"
    assert "eDOCr2 runtime is not installed" in warnings[0]


def test_run_project_ocr_missing_reference_raises(tmp_path):
    project = _project()

    try:
        run_project_ocr(project, tmp_path)
    except FileNotFoundError as exc:
        assert "Reference image not found" in str(exc)
    else:
        raise AssertionError("expected missing reference error")


def test_run_project_ocr_without_tesseract_returns_empty_regions(tmp_path, monkeypatch):
    _write_reference(tmp_path / "pid_reference.png")
    monkeypatch.setattr("app.ocr.shutil.which", lambda name: None)
    monkeypatch.setattr("app.ocr._paddleocr_available", lambda: False)

    result = run_project_ocr(_project(), tmp_path, language_hint="en", engine_hint="tesseract")

    assert result.warnings
    assert {region.target for region in result.regions} >= {"title_block", "parameter_table", "section_view", "circular_view", "dimensions"}
    assert all(region.engine == "none" for region in result.regions)


def test_tesseract_region_records_timeout_and_errors(monkeypatch):
    img = Image.new("L", (100, 100), 255)
    box = SimpleNamespace(target="title_block", label="标题栏", x=0.1, y=0.1, width=0.5, height=0.3)

    def timeout_run(*args, **kwargs):
        raise TimeoutError("slow")

    warnings: list[str] = []
    monkeypatch.setattr("app.ocr.subprocess.run", timeout_run)
    region = _ocr_region_with_tesseract("/usr/bin/tesseract", img, 100, 100, box, "eng", warnings)
    assert region.text == ""
    assert "OCR failed" in warnings[0]

    def failed_run(*args, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="bad image")

    warnings = []
    monkeypatch.setattr("app.ocr.subprocess.run", failed_run)
    region = _ocr_region_with_tesseract("/usr/bin/tesseract", img, 100, 100, box, "eng", warnings)
    assert region.text == ""
    assert "bad image" in warnings[0]


def test_run_project_ocr_with_fake_tesseract_reads_each_region(tmp_path, monkeypatch):
    _write_reference(tmp_path / "pid_reference.png")
    monkeypatch.setattr("app.ocr.shutil.which", lambda name: "/usr/bin/tesseract")

    def fake_run(args, **kwargs):
        if "--list-langs" in args:
            return SimpleNamespace(returncode=0, stdout="List of available languages\neng\n", stderr="")
        return SimpleNamespace(
            returncode=0,
            stdout="\n".join(
                [
                    "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext",
                    "5\t1\t1\t1\t1\t1\t0\t0\t10\t10\t88\tLJT01.01",
                ]
            ),
            stderr="",
        )

    monkeypatch.setattr("app.ocr.subprocess.run", fake_run)

    result = run_project_ocr(_project(), tmp_path, language_hint="en", engine_hint="tesseract")

    assert result.warnings and "chi_sim" in result.warnings[0]
    assert result.regions
    assert all(region.text == "LJT01.01" for region in result.regions)
    assert all(region.confidence == 0.88 for region in result.regions)


def test_run_project_ocr_with_fake_paddle_for_chinese(tmp_path, monkeypatch):
    _write_reference(tmp_path / "pid_reference.png")
    monkeypatch.setattr("app.ocr.shutil.which", lambda name: None)
    monkeypatch.setattr("app.ocr._paddleocr_available", lambda: True)

    class FakePaddle:
        def predict(self, path):
            return [{"rec_texts": ["圆柱直齿轮", "LJT01.01"], "rec_scores": [0.92, 0.84]}]

    monkeypatch.setattr("app.ocr._get_paddle_ocr", lambda language: FakePaddle())

    result = run_project_ocr(_project(), tmp_path, language_hint="zh")

    assert result.regions
    assert all(region.engine == "paddleocr" for region in result.regions)
    assert all(region.language == "zh" for region in result.regions)
    assert all(region.text == "圆柱直齿轮 LJT01.01" for region in result.regions)
    assert all(region.confidence == 0.88 for region in result.regions)


def test_paddle_failure_falls_back_to_tesseract(tmp_path, monkeypatch):
    _write_reference(tmp_path / "pid_reference.png")
    monkeypatch.setattr("app.ocr.shutil.which", lambda name: "/usr/bin/tesseract")
    monkeypatch.setattr("app.ocr._paddleocr_available", lambda: True)

    class BrokenPaddle:
        def predict(self, path):
            raise RuntimeError("model unavailable")

    def fake_run(args, **kwargs):
        if "--list-langs" in args:
            return SimpleNamespace(returncode=0, stdout="List of available languages\neng\nchi_sim\n", stderr="")
        return SimpleNamespace(
            returncode=0,
            stdout="\n".join(
                [
                    "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext",
                    "5\t1\t1\t1\t1\t1\t0\t0\t10\t10\t90\tfallback",
                ]
            ),
            stderr="",
        )

    monkeypatch.setattr("app.ocr._get_paddle_ocr", lambda language: BrokenPaddle())
    monkeypatch.setattr("app.ocr.subprocess.run", fake_run)

    result = run_project_ocr(_project(), tmp_path, language_hint="zh", engine_hint="paddle")

    assert any("PaddleOCR failed" in warning for warning in result.warnings)
    assert result.regions
    assert all(region.engine == "tesseract" for region in result.regions)
    assert all(region.source == "paddleocr_fallback_tesseract" for region in result.regions)
    assert all(region.text == "fallback" for region in result.regions)
