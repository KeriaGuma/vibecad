from __future__ import annotations

import sys
import types
from datetime import datetime, timezone
from pathlib import Path

import ezdxf
from PIL import Image

from app.models import DrawingIR, ProjectState
from app.vectorizer_benchmark import (
    VectorizerResult,
    _complete_svg_result,
    _count_dxf_entities,
    _count_svg_paths,
    _run_autotrace,
    _run_command,
    _run_potrace,
    _run_vtracer,
    _timed_svg_runner,
    run_project_vectorizer_benchmark,
)


def _project(source_image: str) -> ProjectState:
    now = datetime.now(timezone.utc)
    return ProjectState(
        project_id="pid",
        name="bench",
        created_at=now,
        updated_at=now,
        source_image=source_image,
        ir=DrawingIR(),
    )


def test_run_project_vectorizer_benchmark_prepares_input_and_runs_tools(tmp_path, monkeypatch):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    source = uploads / "scan.png"
    Image.new("RGB", (64, 48), "white").save(source)
    project_output = tmp_path / "projects" / "pid"
    project_output.mkdir(parents=True)

    seen: list[Path] = []

    def fake_runner(name: str):
        def run(prepared, output_dir, project_id):
            assert prepared.exists()
            assert prepared.name == "benchmark_normalized_binary.png"
            assert (output_dir / "benchmark_normalized_binary.pbm").exists()
            assert project_id == "pid"
            seen.append(prepared)
            return VectorizerResult(name=name, status="ok", svg_path_count=1, dxf_entity_count=2)

        return run

    monkeypatch.setattr("app.vectorizer_benchmark.project_dir", lambda project_id: project_output)
    monkeypatch.setattr("app.vectorizer_benchmark._run_vtracer", fake_runner("vtracer"))
    monkeypatch.setattr("app.vectorizer_benchmark._run_potrace", fake_runner("potrace"))
    monkeypatch.setattr("app.vectorizer_benchmark._run_autotrace", fake_runner("autotrace"))

    result = run_project_vectorizer_benchmark(_project("/api/uploads/scan.png"), uploads)

    assert len(seen) == 3
    assert result.prepared_image_url == "/api/projects/pid/files/benchmark_normalized_binary.png"
    assert [item.name for item in result.results] == ["vtracer", "potrace", "autotrace"]


def test_run_project_vectorizer_benchmark_rejects_missing_source(tmp_path):
    try:
        run_project_vectorizer_benchmark(_project("/api/uploads/missing.png"), tmp_path)
    except FileNotFoundError as exc:
        assert "Reference image" in str(exc)
    else:
        raise AssertionError("expected missing source error")

    no_upload = _project("")
    no_upload.source_image = None
    try:
        run_project_vectorizer_benchmark(no_upload, tmp_path)
    except ValueError as exc:
        assert "Upload" in str(exc)
    else:
        raise AssertionError("expected missing upload error")


def test_vectorizer_cli_runners_skip_and_attach_commands(tmp_path, monkeypatch):
    prepared = tmp_path / "benchmark_normalized_binary.png"
    prepared.write_bytes(b"png")
    prepared.with_suffix(".pbm").write_bytes(b"pbm")

    monkeypatch.setattr("app.vectorizer_benchmark.shutil.which", lambda name: None)
    assert _run_potrace(prepared, tmp_path, "pid").status == "skipped"
    assert _run_autotrace(prepared, tmp_path, "pid").status == "skipped"

    def fake_timed(name, runner, svg_path, output_dir, project_id):
        runner()
        return VectorizerResult(name=name, status="ok")

    commands: list[list[str]] = []
    monkeypatch.setattr("app.vectorizer_benchmark.shutil.which", lambda name: f"/bin/{name}")
    monkeypatch.setattr("app.vectorizer_benchmark._timed_svg_runner", fake_timed)
    monkeypatch.setattr("app.vectorizer_benchmark._run_command", lambda command, timeout: commands.append(command))

    potrace = _run_potrace(prepared, tmp_path, "pid")
    autotrace = _run_autotrace(prepared, tmp_path, "pid")

    assert potrace.command and potrace.command[0] == "/bin/potrace"
    assert autotrace.command and autotrace.command[0] == "/bin/autotrace"
    assert len(commands) == 2


def test_vtracer_runner_uses_python_package(tmp_path, monkeypatch):
    prepared = tmp_path / "benchmark_normalized_binary.png"
    prepared.write_bytes(b"png")
    calls: list[tuple[str, str]] = []

    fake_vtracer = types.SimpleNamespace(
        convert_image_to_svg_py=lambda source, target, **kwargs: calls.append((source, target))
    )
    monkeypatch.setitem(sys.modules, "vtracer", fake_vtracer)
    monkeypatch.setattr(
        "app.vectorizer_benchmark._timed_svg_runner",
        lambda name, runner, svg_path, output_dir, project_id: (runner() or VectorizerResult(name=name, status="ok")),
    )

    result = _run_vtracer(prepared, tmp_path, "pid")

    assert result.status == "ok"
    assert calls == [(str(prepared), str(tmp_path / "benchmark_vtracer.svg"))]


def test_timed_svg_runner_success_empty_and_failure(tmp_path, monkeypatch):
    monkeypatch.setattr("app.vectorizer_benchmark._count_svg_paths", lambda path: 3)
    monkeypatch.setattr("app.vectorizer_benchmark._count_dxf_entities", lambda path: 5)
    monkeypatch.setattr("app.vectorizer_benchmark.export_svg_geometry_to_dxf", lambda svg, dxf: dxf.write_text("dxf"))
    monkeypatch.setattr("app.vectorizer_benchmark.render_dxf_to_svg", lambda dxf, preview: preview.write_text("<svg/>"))

    svg_path = tmp_path / "benchmark_tool.svg"

    def write_svg():
        svg_path.write_text("<svg><path d='M 0 0 L 1 1'/></svg>")

    ok = _timed_svg_runner("tool", write_svg, svg_path, tmp_path, "pid")
    assert ok.status == "ok"
    assert ok.svg_url == "/api/projects/pid/files/benchmark_tool.svg"
    assert ok.dxf_entity_count == 5

    empty_path = tmp_path / "empty.svg"
    empty = _timed_svg_runner("empty", lambda: empty_path.write_text(""), empty_path, tmp_path, "pid")
    assert empty.status == "failed"
    assert "empty" in empty.detail

    failed = _timed_svg_runner("failed", lambda: (_ for _ in ()).throw(RuntimeError("boom")), tmp_path / "bad.svg", tmp_path, "pid")
    assert failed.status == "failed"
    assert failed.detail == "boom"


def test_complete_svg_result_keeps_partial_svg_when_dxf_conversion_fails(tmp_path, monkeypatch):
    svg_path = tmp_path / "benchmark_tool.svg"
    svg_path.write_text("<svg><path d='M 0 0 L 1 1'/></svg>")
    monkeypatch.setattr("app.vectorizer_benchmark._count_svg_paths", lambda path: 1)
    monkeypatch.setattr(
        "app.vectorizer_benchmark.export_svg_geometry_to_dxf",
        lambda svg, dxf: (_ for _ in ()).throw(RuntimeError("convert failed")),
    )

    result = _complete_svg_result("tool", svg_path, tmp_path, "pid", 0.01)

    assert result.status == "partial"
    assert result.svg_path_count == 1
    assert "convert failed" in result.detail


def test_run_command_and_count_helpers(tmp_path):
    try:
        _run_command([sys.executable, "-c", "import sys; sys.stderr.write('bad'); sys.exit(2)"], timeout=10)
    except RuntimeError as exc:
        assert "bad" in str(exc)
    else:
        raise AssertionError("expected command failure")

    svg = tmp_path / "one.svg"
    svg.write_text("<svg xmlns='http://www.w3.org/2000/svg'><path d='M 0 0 L 1 1'/></svg>")
    assert _count_svg_paths(svg) == 1

    dxf = tmp_path / "one.dxf"
    doc = ezdxf.new("R2010")
    doc.modelspace().add_line((0, 0), (1, 1))
    doc.saveas(dxf)
    assert _count_dxf_entities(dxf) == 1
