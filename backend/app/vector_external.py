from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import ezdxf
import pymupdf

from .cad_layers import REFERENCE_TRACE
from .svg_dxf import export_svg_geometry_to_dxf, render_dxf_to_svg


@dataclass(frozen=True)
class VectorAssetExport:
    preview_source: str
    dxf_source: str
    warnings: list[str]


def export_vector_pdf_assets(pdf_path: Path, output_dir: Path, page_number: int = 0) -> VectorAssetExport:
    """Export vector-PDF assets with mature external/native converters.

    The app's editable IR is still useful for inspection and future agent edits,
    but visual fidelity should not depend on our small PDF parser. MuPDF renders
    the SVG preview from the original PDF content; Inkscape is preferred for DXF
    export, with pstoedit as a legacy fallback. If neither produces a non-empty
    DXF, the existing IR-generated DXF remains in place.
    """
    warnings: list[str] = []
    source_svg = output_dir / "source_preview.svg"
    preview_svg = output_dir / "preview.svg"
    output_dxf = output_dir / "output.dxf"
    _write_mupdf_svg_preview(pdf_path, source_svg, page_number)
    dxf_source = _try_inkscape_dxf(source_svg, output_dxf, warnings)
    if dxf_source == "ir_fallback":
        dxf_source = _try_svg_dxf(source_svg, output_dxf, warnings)
    if dxf_source == "ir_fallback":
        dxf_source = _try_pstoedit_dxf(pdf_path, output_dxf, warnings)
    if dxf_source != "ir_fallback":
        _normalize_external_reference_layer(output_dxf, warnings)
    preview_source = _write_dxf_preview_or_fallback(output_dxf, preview_svg, source_svg, warnings)
    return VectorAssetExport(
        preview_source=preview_source,
        dxf_source=dxf_source,
        warnings=warnings,
    )


def _write_mupdf_svg_preview(pdf_path: Path, svg_path: Path, page_number: int) -> None:
    try:
        document = pymupdf.open(pdf_path)
    except Exception as exc:  # noqa: BLE001 - surface PyMuPDF errors to the API caller
        raise ValueError(f"Could not open PDF for MuPDF SVG export: {exc}") from exc

    try:
        if not 0 <= page_number < document.page_count:
            raise ValueError(f"Page {page_number} is out of range for this PDF.")
        svg = document.load_page(page_number).get_svg_image(text_as_path=1)
    finally:
        document.close()

    svg_path.write_text(svg, encoding="utf-8")


def _try_inkscape_dxf(svg_path: Path, dxf_path: Path, warnings: list[str]) -> str:
    inkscape = _find_inkscape()
    if not inkscape:
        warnings.append("Inkscape CLI not installed; skipped mature SVG-to-DXF export.")
        return "ir_fallback"

    temp_dxf = dxf_path.with_name(f"{dxf_path.stem}.inkscape{dxf_path.suffix}")
    result = subprocess.run(
        [inkscape, str(svg_path), f"--export-filename={temp_dxf}", "--export-type=dxf"],
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
    )
    detail = (result.stderr or result.stdout or "Inkscape DXF export failed").strip()
    if result.returncode != 0 or not temp_dxf.exists():
        warnings.append(f"Inkscape DXF export failed; trying fallback converter. Detail: {detail}")
        return "ir_fallback"
    if not _dxf_has_entities(temp_dxf):
        temp_dxf.unlink(missing_ok=True)
        warnings.append(f"Inkscape produced an empty DXF; trying fallback converter. Detail: {detail}")
        return "ir_fallback"
    temp_dxf.replace(dxf_path)
    return "inkscape"


def _find_inkscape() -> str | None:
    if found := shutil.which("inkscape"):
        return found
    for path in (
        "/Applications/Inkscape.app/Contents/MacOS/inkscape",
        "/opt/homebrew/bin/inkscape",
        "/usr/local/bin/inkscape",
    ):
        if Path(path).exists():
            return path
    return None


def _write_dxf_preview_or_fallback(
    dxf_path: Path,
    preview_svg: Path,
    source_svg: Path,
    warnings: list[str],
) -> str:
    try:
        result = render_dxf_to_svg(dxf_path, preview_svg)
    except Exception as exc:  # noqa: BLE001 - keep the UI usable if DXF rendering fails
        shutil.copyfile(source_svg, preview_svg)
        warnings.append(f"DXF preview render failed; showing MuPDF source SVG instead. Detail: {exc}")
        return "mupdf_svg_fallback"
    warnings.append(
        f"DXF preview rendered from output.dxf: {result.entity_count} entities, "
        f"{result.width_mm:.3f} x {result.height_mm:.3f} mm."
    )
    return "dxf_render_svg"


def _try_svg_dxf(svg_path: Path, dxf_path: Path, warnings: list[str]) -> str:
    temp_dxf = dxf_path.with_name(f"{dxf_path.stem}.svgdxf{dxf_path.suffix}")
    try:
        result = export_svg_geometry_to_dxf(svg_path, temp_dxf)
    except Exception as exc:  # noqa: BLE001 - report converter errors without losing fallback DXF
        warnings.append(f"svgelements DXF export failed; trying fallback converter. Detail: {exc}")
        return "ir_fallback"
    if result.entity_count <= 0 or not _dxf_has_entities(temp_dxf):
        temp_dxf.unlink(missing_ok=True)
        warnings.append("svgelements produced an empty DXF; trying fallback converter.")
        return "ir_fallback"
    temp_dxf.replace(dxf_path)
    warnings.append(
        f"DXF generated via svgelements+ezdxf fallback: {result.entity_count} polylines "
        f"from {result.source_path_count} SVG paths."
    )
    return "svg_dxf"


def _try_pstoedit_dxf(pdf_path: Path, dxf_path: Path, warnings: list[str]) -> str:
    pstoedit = shutil.which("pstoedit")
    ghostscript = shutil.which("gs")
    if not pstoedit or not ghostscript:
        warnings.append(
            "pstoedit/ghostscript not installed; DXF is still generated by the fallback IR exporter. "
            "Install with: brew install pstoedit ghostscript"
        )
        return "ir_fallback"

    temp_dxf = dxf_path.with_name(f"{dxf_path.stem}.pstoedit{dxf_path.suffix}")
    result = subprocess.run(
        [pstoedit, "-f", "dxf_14", str(pdf_path), str(temp_dxf)],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    detail = (result.stderr or result.stdout or "pstoedit failed").strip()
    if result.returncode != 0 or not temp_dxf.exists():
        warnings.append(f"pstoedit DXF export failed; kept fallback IR DXF. Detail: {detail}")
        return "ir_fallback"
    if not _dxf_has_entities(temp_dxf):
        temp_dxf.unlink(missing_ok=True)
        warnings.append(f"pstoedit produced an empty DXF; kept fallback IR DXF. Detail: {detail}")
        return "ir_fallback"
    temp_dxf.replace(dxf_path)
    return "pstoedit"


def _dxf_has_entities(path: Path) -> bool:
    """Return true when the DXF ENTITIES section contains drawable records."""
    entity_types = {
        "ARC",
        "CIRCLE",
        "ELLIPSE",
        "INSERT",
        "LINE",
        "LWPOLYLINE",
        "MTEXT",
        "POINT",
        "POLYLINE",
        "SPLINE",
        "TEXT",
    }
    try:
        return _entities_section_has_any(path.read_text(encoding="utf-8", errors="ignore").splitlines(), entity_types)
    except OSError:
        return False


def _normalize_external_reference_layer(path: Path, warnings: list[str]) -> None:
    """Isolate mature-converter output as a locked, non-editable reference."""

    try:
        doc = ezdxf.readfile(path)
        if REFERENCE_TRACE not in doc.layers:
            layer = doc.layers.add(REFERENCE_TRACE, color=8, lineweight=13)
        else:
            layer = doc.layers.get(REFERENCE_TRACE)
            layer.dxf.color = 8
            layer.dxf.lineweight = 13
        layer.lock()
        for entity in doc.modelspace():
            entity.dxf.layer = REFERENCE_TRACE
            entity.dxf.color = 256
            entity.dxf.lineweight = 13
        doc.saveas(path)
    except (OSError, ezdxf.DXFError, ValueError) as exc:
        warnings.append(f"Could not isolate external DXF on REFERENCE_TRACE: {exc}")
        return
    warnings.append("External DXF source isolated on locked REFERENCE_TRACE layer.")


def _entities_section_has_any(lines: Iterable[str], entity_types: set[str]) -> bool:
    in_entities = False
    pending_section_name = False
    iterator = iter(lines)
    for raw_code in iterator:
        raw_value = next(iterator, "")
        code = raw_code.strip()
        value = raw_value.strip().upper()
        if pending_section_name:
            in_entities = value == "ENTITIES"
            pending_section_name = False
            continue
        if code != "0":
            continue
        if value == "SECTION":
            pending_section_name = True
        elif value == "ENDSEC":
            in_entities = False
        elif in_entities and value in entity_types:
            return True
    return False
