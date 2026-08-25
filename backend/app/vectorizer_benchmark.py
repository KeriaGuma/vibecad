from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import ezdxf
from PIL import Image
from svgelements import SVG

from .models import ProjectState
from .reference import _preprocess_reference_image, _upload_url_to_path
from .storage import project_dir
from .svg_dxf import export_svg_geometry_to_dxf, render_dxf_to_svg


@dataclass(frozen=True)
class VectorizerResult:
    name: str
    status: str
    elapsed_sec: float = 0.0
    svg_url: str | None = None
    dxf_url: str | None = None
    preview_url: str | None = None
    svg_path_count: int = 0
    dxf_entity_count: int = 0
    output_bytes: int = 0
    command: list[str] = field(default_factory=list)
    detail: str = ""


@dataclass(frozen=True)
class VectorizerBenchmark:
    project_id: str
    prepared_image_url: str
    results: list[VectorizerResult]


def run_project_vectorizer_benchmark(project: ProjectState, uploads_dir: Path) -> VectorizerBenchmark:
    if not project.source_image:
        raise ValueError("Upload a PDF or image before running vectorizer benchmark.")
    source_image = _upload_url_to_path(project.source_image, uploads_dir)
    if not source_image.exists():
        raise FileNotFoundError("Reference image not found")

    output_dir = project_dir(project.project_id)
    prepared = _prepare_input(source_image, output_dir)
    results = [
        _run_vtracer(prepared, output_dir, project.project_id),
        _run_potrace(prepared, output_dir, project.project_id),
        _run_autotrace(prepared, output_dir, project.project_id),
    ]
    return VectorizerBenchmark(
        project_id=project.project_id,
        prepared_image_url=f"/api/projects/{project.project_id}/files/{prepared.name}",
        results=results,
    )


def _prepare_input(source_image: Path, output_dir: Path) -> Path:
    raster = output_dir / "benchmark_source_first_page.png"
    Image.open(source_image).convert("RGB").save(raster)

    processed = _preprocess_reference_image(raster)
    normalized = output_dir / "benchmark_normalized_binary.png"
    processed.image.save(normalized)

    binary = output_dir / "benchmark_normalized_binary.pbm"
    processed.image.convert("1").save(binary)
    return normalized


def _run_vtracer(prepared_png: Path, output_dir: Path, project_id: str) -> VectorizerResult:
    try:
        import vtracer
    except ImportError:
        return VectorizerResult(name="vtracer", status="skipped", detail="Python package vtracer is not installed.")

    svg_path = output_dir / "benchmark_vtracer.svg"

    def run() -> None:
        vtracer.convert_image_to_svg_py(
            str(prepared_png),
            str(svg_path),
            colormode="binary",
            hierarchical="stacked",
            mode="spline",
            filter_speckle=8,
            color_precision=6,
            layer_difference=16,
            corner_threshold=60,
            length_threshold=4.0,
            max_iterations=10,
            splice_threshold=45,
            path_precision=3,
        )

    return _timed_svg_runner("vtracer", run, svg_path, output_dir, project_id)


def _run_potrace(prepared_png: Path, output_dir: Path, project_id: str) -> VectorizerResult:
    potrace = shutil.which("potrace")
    if not potrace:
        return VectorizerResult(
            name="potrace",
            status="skipped",
            detail="potrace is not installed. Install with: brew install potrace",
        )
    pbm_path = prepared_png.with_suffix(".pbm")
    svg_path = output_dir / "benchmark_potrace.svg"
    command = [potrace, str(pbm_path), "-b", "svg", "-o", str(svg_path)]

    def run() -> None:
        _run_command(command, timeout=120)

    result = _timed_svg_runner("potrace", run, svg_path, output_dir, project_id)
    result.command.extend(command)
    return result


def _run_autotrace(prepared_png: Path, output_dir: Path, project_id: str) -> VectorizerResult:
    autotrace = shutil.which("autotrace")
    if not autotrace:
        return VectorizerResult(
            name="autotrace",
            status="skipped",
            detail="autotrace is not installed. Install with: brew install autotrace",
        )
    svg_path = output_dir / "benchmark_autotrace.svg"
    command = [
        autotrace,
        "--centerline",
        "--output-format=svg",
        f"--output-file={svg_path}",
        str(prepared_png),
    ]

    def run() -> None:
        _run_command(command, timeout=120)

    result = _timed_svg_runner("autotrace", run, svg_path, output_dir, project_id)
    result.command.extend(command)
    return result


def _timed_svg_runner(
    name: str,
    runner: Callable[[], None],
    svg_path: Path,
    output_dir: Path,
    project_id: str,
) -> VectorizerResult:
    started = time.perf_counter()
    try:
        runner()
        elapsed = time.perf_counter() - started
        if not svg_path.exists() or svg_path.stat().st_size <= 0:
            return VectorizerResult(name=name, status="failed", elapsed_sec=round(elapsed, 3), detail="SVG output is empty.")
        return _complete_svg_result(name, svg_path, output_dir, project_id, elapsed)
    except Exception as exc:  # noqa: BLE001 - benchmark should keep comparing other tools
        return VectorizerResult(
            name=name,
            status="failed",
            elapsed_sec=round(time.perf_counter() - started, 3),
            detail=str(exc),
        )


def _complete_svg_result(
    name: str,
    svg_path: Path,
    output_dir: Path,
    project_id: str,
    elapsed: float,
) -> VectorizerResult:
    dxf_path = output_dir / f"benchmark_{name}.dxf"
    preview_path = output_dir / f"benchmark_{name}_preview.svg"
    svg_count = _count_svg_paths(svg_path)
    dxf_entity_count = 0
    detail = ""
    try:
        export_svg_geometry_to_dxf(svg_path, dxf_path)
        dxf_entity_count = _count_dxf_entities(dxf_path)
        render_dxf_to_svg(dxf_path, preview_path)
    except Exception as exc:  # noqa: BLE001 - keep SVG result even if DXF conversion fails
        detail = f"SVG produced, but SVG->DXF/preview failed: {exc}"
    output_bytes = sum(path.stat().st_size for path in [svg_path, dxf_path, preview_path] if path.exists())
    return VectorizerResult(
        name=name,
        status="ok" if dxf_entity_count > 0 else "partial",
        elapsed_sec=round(elapsed, 3),
        svg_url=f"/api/projects/{project_id}/files/{svg_path.name}",
        dxf_url=f"/api/projects/{project_id}/files/{dxf_path.name}" if dxf_path.exists() else None,
        preview_url=f"/api/projects/{project_id}/files/{preview_path.name}" if preview_path.exists() else None,
        svg_path_count=svg_count,
        dxf_entity_count=dxf_entity_count,
        output_bytes=output_bytes,
        detail=detail,
    )


def _run_command(command: list[str], timeout: int) -> None:
    result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or f"Command failed: {command}").strip()
        raise RuntimeError(detail)


def _count_svg_paths(svg_path: Path) -> int:
    svg = SVG.parse(svg_path, reify=False, ppi=72)
    return sum(1 for element in svg.elements() if element.__class__.__name__ == "Path")


def _count_dxf_entities(dxf_path: Path) -> int:
    doc = ezdxf.readfile(dxf_path)
    return len(list(doc.modelspace()))
