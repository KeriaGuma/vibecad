from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import ezdxf
from PIL import Image
from svgelements import SVG

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.reference import _preprocess_reference_image, _render_pdf_first_page  # noqa: E402
from app.svg_dxf import export_svg_geometry_to_dxf, render_dxf_to_svg  # noqa: E402


@dataclass
class VectorizerResult:
    name: str
    status: str
    elapsed_sec: float = 0.0
    svg_path: str | None = None
    dxf_path: str | None = None
    preview_path: str | None = None
    svg_path_count: int = 0
    dxf_entity_count: int = 0
    output_bytes: int = 0
    command: list[str] = field(default_factory=list)
    detail: str = ""


def main() -> int:
    args = _parse_args()
    source = args.input.expanduser().resolve()
    if not source.exists():
        raise SystemExit(f"Input not found: {source}")

    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = _prepare_input(source, output_dir)
    results = [
        _run_vtracer(prepared, output_dir),
        _run_potrace(prepared, output_dir),
        _run_autotrace(prepared, output_dir),
    ]
    _write_report(source, prepared, output_dir, results)
    print(f"Benchmark written to: {output_dir}")
    print(f"Report: {output_dir / 'report.md'}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark external raster-to-vector tools on one CAD reference.")
    parser.add_argument("input", type=Path, help="Input PDF or image.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/vectorizer_benchmarks/latest"),
        help="Output directory for benchmark artifacts.",
    )
    return parser.parse_args()


def _prepare_input(source: Path, output_dir: Path) -> Path:
    raster = output_dir / "source_first_page.png"
    if source.suffix.lower() == ".pdf":
        _render_pdf_first_page(source, raster)
    else:
        Image.open(source).convert("RGB").save(raster)

    processed = _preprocess_reference_image(raster)
    normalized = output_dir / "normalized_binary.png"
    processed.image.save(normalized)

    binary = output_dir / "normalized_binary.pbm"
    processed.image.convert("1").save(binary)
    return normalized


def _run_vtracer(prepared_png: Path, output_dir: Path) -> VectorizerResult:
    try:
        import vtracer
    except ImportError:
        return VectorizerResult(name="vtracer", status="skipped", detail="Python package vtracer is not installed.")

    svg_path = output_dir / "vtracer.svg"

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

    return _timed_svg_runner("vtracer", run, svg_path, output_dir)


def _run_potrace(prepared_png: Path, output_dir: Path) -> VectorizerResult:
    potrace = shutil.which("potrace")
    if not potrace:
        return VectorizerResult(
            name="potrace",
            status="skipped",
            detail="potrace is not installed. Install with: brew install potrace",
        )
    pbm_path = prepared_png.with_suffix(".pbm")
    svg_path = output_dir / "potrace.svg"
    command = [potrace, str(pbm_path), "-b", "svg", "-o", str(svg_path)]

    def run() -> None:
        _run_command(command, timeout=120)

    result = _timed_svg_runner("potrace", run, svg_path, output_dir)
    result.command = command
    return result


def _run_autotrace(prepared_png: Path, output_dir: Path) -> VectorizerResult:
    autotrace = shutil.which("autotrace")
    if not autotrace:
        return VectorizerResult(
            name="autotrace",
            status="skipped",
            detail="autotrace is not installed. Install with: brew install autotrace",
        )
    svg_path = output_dir / "autotrace.svg"
    command = [
        autotrace,
        "--centerline",
        "--output-format=svg",
        f"--output-file={svg_path}",
        str(prepared_png),
    ]

    def run() -> None:
        _run_command(command, timeout=120)

    result = _timed_svg_runner("autotrace", run, svg_path, output_dir)
    result.command = command
    return result


def _timed_svg_runner(
    name: str,
    runner: Callable[[], None],
    svg_path: Path,
    output_dir: Path,
) -> VectorizerResult:
    started = time.perf_counter()
    try:
        runner()
        elapsed = time.perf_counter() - started
        if not svg_path.exists() or svg_path.stat().st_size <= 0:
            return VectorizerResult(name=name, status="failed", elapsed_sec=elapsed, detail="SVG output is empty.")
        return _complete_svg_result(name, svg_path, output_dir, elapsed)
    except Exception as exc:  # noqa: BLE001 - benchmark should keep comparing other tools
        return VectorizerResult(
            name=name,
            status="failed",
            elapsed_sec=time.perf_counter() - started,
            detail=str(exc),
        )


def _complete_svg_result(name: str, svg_path: Path, output_dir: Path, elapsed: float) -> VectorizerResult:
    dxf_path = output_dir / f"{name}.dxf"
    preview_path = output_dir / f"{name}_preview.svg"
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
        svg_path=str(svg_path),
        dxf_path=str(dxf_path) if dxf_path.exists() else None,
        preview_path=str(preview_path) if preview_path.exists() else None,
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


def _write_report(source: Path, prepared: Path, output_dir: Path, results: list[VectorizerResult]) -> None:
    payload = {
        "source": str(source),
        "prepared": str(prepared),
        "results": [asdict(result) for result in results],
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# External Vectorizer Benchmark",
        "",
        f"- source: `{source}`",
        f"- prepared image: `{prepared}`",
        "",
        "| tool | status | sec | svg paths | dxf entities | output | detail |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for result in results:
        lines.append(
            "| "
            + " | ".join(
                [
                    result.name,
                    result.status,
                    f"{result.elapsed_sec:.3f}",
                    str(result.svg_path_count),
                    str(result.dxf_entity_count),
                    f"{result.output_bytes / 1024:.1f} KB",
                    result.detail.replace("|", "\\|") or "",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
        ]
    )
    for result in results:
        if result.svg_path:
            lines.append(f"- {result.name} SVG: `{result.svg_path}`")
        if result.dxf_path:
            lines.append(f"- {result.name} DXF: `{result.dxf_path}`")
        if result.preview_path:
            lines.append(f"- {result.name} preview: `{result.preview_path}`")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
