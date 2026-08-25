from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pymupdf
from PIL import Image, ImageDraw

from .ingest import IMAGE_SUFFIXES, classify_source
from .models import AnalyzeResponse, ProjectState, RegionBox

UPLOAD_PREFIX = "/api/uploads/"
ANALYSIS_SOURCE = "scan_preprocess_layout_v3"


@dataclass(frozen=True)
class PreprocessResult:
    image: Image.Image
    dark: np.ndarray
    angle: float
    orientation_angle: float = 0.0


@dataclass(frozen=True)
class LayoutDetection:
    boxes: list[RegionBox]
    frame: tuple[int, int, int, int] | None
    processed: PreprocessResult | None


@dataclass(frozen=True)
class ReferenceUpload:
    source_file: str
    source_image: str
    image_width: int | None
    image_height: int | None
    source_kind: str


def save_reference_upload(project_id: str, filename: str, payload: bytes, uploads_dir: Path) -> ReferenceUpload:
    suffix = _safe_suffix(filename)
    uploads_dir.mkdir(parents=True, exist_ok=True)
    source_path = uploads_dir / f"{project_id}_source{suffix}"
    source_path.write_bytes(payload)

    if suffix == ".pdf":
        classification = classify_source(source_path)
        preview_path = uploads_dir / f"{project_id}_reference.png"
        _render_pdf_first_page(source_path, preview_path)
    elif suffix in IMAGE_SUFFIXES:
        preview_path = uploads_dir / f"{project_id}_reference{suffix}"
        if preview_path != source_path:
            preview_path.write_bytes(payload)
        classification = classify_source(source_path)

    width, height = image_size(preview_path)
    return ReferenceUpload(
        source_file=f"{UPLOAD_PREFIX}{source_path.name}",
        source_image=f"{UPLOAD_PREFIX}{preview_path.name}",
        image_width=width,
        image_height=height,
        source_kind=classification.kind.value,
    )


def analyze_reference(project: ProjectState, uploads_dir: Path, output_dir: Path | None = None) -> AnalyzeResponse:
    if not project.source_image:
        raise ValueError("Upload a PDF or image before running analysis.")

    image_path = _upload_url_to_path(project.source_image, uploads_dir)
    if not image_path.exists():
        raise FileNotFoundError("Reference image not found")

    source_width, source_height = image_size(image_path)
    detection = _detect_layout(image_path, source_width, source_height)
    if detection.processed is not None:
        width, height = detection.processed.image.size
    else:
        width, height = source_width, source_height
    overlay_image = preprocessed_image = None
    if output_dir is not None and detection.processed is not None:
        overlay_image, preprocessed_image = _write_analysis_debug_images(
            project.project_id,
            image_path,
            output_dir,
            detection,
        )

    return AnalyzeResponse(
        project_id=project.project_id,
        source_image=project.source_image,
        image_width=width,
        image_height=height,
        overlay_image=overlay_image,
        preprocessed_image=preprocessed_image,
        frame=_normalized_frame(detection.frame, width, height),
        deskew_angle=round(detection.processed.angle + detection.processed.orientation_angle, 3) if detection.processed else 0.0,
        boxes=detection.boxes,
    )


def image_size(path: Path) -> tuple[int | None, int | None]:
    image_type = _image_kind(path)
    if image_type == "png":
        return _png_size(path)
    if image_type == "gif":
        return _gif_size(path)
    if image_type == "jpeg":
        return _jpeg_size(path)
    return None, None


def _safe_suffix(filename: str) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix == ".pdf" or suffix in IMAGE_SUFFIXES:
        return suffix
    raise ValueError("Unsupported reference file type. Upload a PDF or image.")


def _render_pdf_first_page(source_path: Path, preview_path: Path, dpi: int = 300) -> None:
    try:
        document = pymupdf.open(source_path)
    except Exception as exc:  # noqa: BLE001 - surface any PyMuPDF open failure to the caller
        raise ValueError(f"Could not render PDF first page: {exc}") from exc

    try:
        if document.page_count < 1:
            raise ValueError("Could not render PDF first page: the document has no pages.")
        page = document.load_page(0)
        pixmap = page.get_pixmap(dpi=dpi)
        pixmap.save(preview_path)
    except ValueError:
        raise
    except Exception as exc:  # noqa: BLE001 - surface any PyMuPDF render failure to the caller
        raise ValueError(f"Could not render PDF first page: {exc}") from exc
    finally:
        document.close()

    if not preview_path.exists():
        raise ValueError("Could not render PDF first page.")


def _upload_url_to_path(url: str, uploads_dir: Path) -> Path:
    if not url.startswith(UPLOAD_PREFIX):
        raise ValueError("Invalid reference image URL.")
    filename = Path(url.removeprefix(UPLOAD_PREFIX)).name
    return uploads_dir / filename


def _image_kind(path: Path) -> str | None:
    with path.open("rb") as fh:
        header = fh.read(16)
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if header[:6] in {b"GIF87a", b"GIF89a"}:
        return "gif"
    if header.startswith(b"\xff\xd8"):
        return "jpeg"
    return None


def _baseline_region_boxes() -> list[RegionBox]:
    return [
        RegionBox(
            target="title_block",
            label="标题栏",
            x=0.54,
            y=0.80,
            width=0.36,
            height=0.16,
            confidence=0.62,
        ),
        RegionBox(
            target="parameter_table",
            label="参数表",
            x=0.66,
            y=0.16,
            width=0.26,
            height=0.39,
            confidence=0.58,
        ),
        RegionBox(
            target="section_view",
            label="剖视图",
            x=0.08,
            y=0.28,
            width=0.32,
            height=0.58,
            confidence=0.55,
        ),
        RegionBox(
            target="circular_view",
            label="圆视图",
            x=0.41,
            y=0.36,
            width=0.20,
            height=0.30,
            confidence=0.54,
        ),
        RegionBox(
            target="dimensions",
            label="尺寸标注",
            x=0.07,
            y=0.25,
            width=0.59,
            height=0.64,
            confidence=0.50,
        ),
    ]


def _detect_region_boxes(path: Path, width: int | None, height: int | None) -> list[RegionBox]:
    return _detect_layout(path, width, height).boxes


def _detect_layout(path: Path, width: int | None, height: int | None) -> LayoutDetection:
    try:
        processed = _preprocess_reference_image(path)
    except Exception:
        return LayoutDetection(boxes=_baseline_region_boxes(), frame=None, processed=None)

    dark = processed.dark
    if dark.size == 0:
        return LayoutDetection(boxes=_baseline_region_boxes(), frame=None, processed=processed)

    image_height, image_width = dark.shape
    frame = _detect_inner_frame(dark)
    if frame is None:
        return LayoutDetection(boxes=_baseline_region_boxes(), frame=None, processed=processed)

    return LayoutDetection(
        boxes=_detect_region_boxes_from_dark(dark, frame, image_width, image_height),
        frame=frame,
        processed=processed,
    )


def _preprocess_reference_image(path: Path) -> PreprocessResult:
    try:
        gray = np.asarray(Image.open(path).convert("L"))
    except Exception:
        raise

    if gray.size == 0:
        return PreprocessResult(image=Image.fromarray(gray), dark=np.zeros_like(gray, dtype=bool), angle=0.0)

    denoised = cv2.medianBlur(gray, 3)
    block_size = max(15, min(61, (min(gray.shape) // 12) | 1))
    binary = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        11,
    )
    angle = _estimate_skew_angle(binary)
    if abs(angle) >= 0.15:
        binary = _rotate_same_size(binary, -angle, fill=255)

    orientation_angle = 0.0
    if _needs_landscape_rotation(binary):
        binary = cv2.rotate(binary, cv2.ROTATE_90_COUNTERCLOCKWISE)
        orientation_angle = -90.0

    dark = binary < 128
    return PreprocessResult(image=Image.fromarray(binary), dark=dark, angle=angle, orientation_angle=orientation_angle)


def _needs_landscape_rotation(binary: np.ndarray) -> bool:
    height, width = binary.shape
    if height <= width * 1.08:
        return False
    dark = binary < 128
    frame = _detect_inner_frame(dark)
    if frame is None:
        return True
    x1, y1, x2, y2 = frame
    return (y2 - y1) > (x2 - x1) * 1.08


def _estimate_skew_angle(binary: np.ndarray) -> float:
    height, width = binary.shape
    if height < 40 or width < 40:
        return 0.0

    edges = cv2.Canny(255 - binary, 50, 150, apertureSize=3)
    min_length = max(30, min(width, height) // 8)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=min_length, maxLineGap=8)
    if lines is None:
        return 0.0

    deviations: list[float] = []
    for x1, y1, x2, y2 in lines[:, 0, :]:
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = (dx * dx + dy * dy) ** 0.5
        if length < min_length:
            continue
        angle = float(np.degrees(np.arctan2(dy, dx)))
        if angle >= 90:
            angle -= 180
        if angle < -90:
            angle += 180
        candidates = [angle, angle - 90, angle + 90]
        deviation = min(candidates, key=abs)
        if abs(deviation) <= 4.0:
            deviations.append(deviation)

    if len(deviations) < 4:
        return 0.0
    return float(np.median(deviations))


def _rotate_same_size(image: np.ndarray, angle: float, fill: int) -> np.ndarray:
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=fill,
    )


def _detect_region_boxes_from_dark(
    dark: np.ndarray,
    frame: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> list[RegionBox]:
    fx1, fy1, fx2, fy2 = frame
    fw = max(fx2 - fx1, 1)
    fh = max(fy2 - fy1, 1)
    parameter_table = _grid_bbox_in_roi(
        dark,
        (fx1 + fw * 0.65, fy1 - fh * 0.02, fx2 + fw * 0.01, fy1 + fh * 0.50),
        vertical_threshold=0.30,
        horizontal_threshold=0.30,
    )
    title_block = _grid_bbox_in_roi(
        dark,
        (fx1 + fw * 0.45, fy1 + fh * 0.73, fx2 + fw * 0.01, fy2 + fh * 0.02),
        vertical_threshold=0.25,
        horizontal_threshold=0.25,
    ) or _bbox_in_roi(
        dark,
        (fx1 + fw * 0.48, fy1 + fh * 0.78, fx2 + fw * 0.01, fy2 + fh * 0.02),
        pad=8,
    )
    detected = {
        "parameter_table": parameter_table,
        "title_block": title_block,
        "section_view": _bbox_in_roi(
            dark,
            (fx1 + fw * 0.06, fy1 + fh * 0.18, fx1 + fw * 0.40, fy1 + fh * 0.84),
            pad=8,
        ),
        "circular_view": _bbox_in_roi(
            dark,
            (fx1 + fw * 0.38, fy1 + fh * 0.25, fx1 + fw * 0.67, fy1 + fh * 0.65),
            pad=8,
        ),
    }
    dimension_candidates = [
        (
            "尺寸标注-左侧",
            _bbox_in_roi(
                dark,
                (fx1 + fw * 0.02, fy1 + fh * 0.22, fx1 + fw * 0.11, fy1 + fh * 0.72),
                pad=5,
            ),
        ),
        (
            "尺寸标注-剖视上方",
            _bbox_in_roi(
                dark,
                (fx1 + fw * 0.02, fy1 + fh * 0.13, fx1 + fw * 0.39, fy1 + fh * 0.28),
                pad=5,
            ),
        ),
        (
            "尺寸标注-剖视右侧",
            _bbox_in_roi(
                dark,
                (fx1 + fw * 0.31, fy1 + fh * 0.31, fx1 + fw * 0.40, fy1 + fh * 0.66),
                pad=5,
            ),
        ),
        (
            "尺寸标注-圆视图",
            _bbox_in_roi(
                dark,
                (fx1 + fw * 0.43, fy1 + fh * 0.27, fx1 + fw * 0.69, fy1 + fh * 0.71),
                pad=5,
            ),
        ),
    ]
    dimension_boxes = [
        _region_box(
            target="dimensions",
            label=label,
            bbox=bbox,
            image_width=image_width,
            image_height=image_height,
            confidence=0.58,
        )
        for label, bbox in dimension_candidates
        if bbox is not None
    ]
    if not dimension_boxes:
        dimensions = _bbox_in_roi(
            dark,
            (fx1 - fw * 0.01, fy1 + fh * 0.09, fx1 + fw * 0.70, fy1 + fh * 0.87),
            pad=8,
        )
        if dimensions is not None:
            dimension_boxes.append(
                _region_box(
                    target="dimensions",
                    label="尺寸标注",
                    bbox=dimensions,
                    image_width=image_width,
                    image_height=image_height,
                    confidence=0.52,
                )
            )

    labels = {
        "title_block": "标题栏",
        "parameter_table": "参数表",
        "section_view": "剖视图",
        "circular_view": "圆视图",
        "dimensions": "尺寸标注",
    }
    confidences = {
        "title_block": 0.84,
        "parameter_table": 0.84,
        "section_view": 0.72,
        "circular_view": 0.70,
    }
    fallback = {box.target: box for box in _baseline_region_boxes()}
    boxes: list[RegionBox] = []
    for target in ["title_block", "parameter_table", "section_view", "circular_view"]:
        bbox = detected[target]
        if bbox is None:
            if target == "parameter_table":
                continue
            base = fallback[target]
            boxes.append(base.model_copy(update={"confidence": min(base.confidence, 0.45)}))
            continue
        boxes.append(
            _region_box(
                target=target,
                label=labels[target],
                bbox=bbox,
                image_width=image_width,
                image_height=image_height,
                confidence=confidences[target],
            )
        )
    boxes.extend(dimension_boxes or [fallback["dimensions"].model_copy(update={"confidence": 0.45})])
    return boxes


def _write_analysis_debug_images(
    project_id: str,
    image_path: Path,
    output_dir: Path,
    detection: LayoutDetection,
) -> tuple[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = output_dir / "analysis_overlay.png"
    processed_path = output_dir / "analysis_preprocessed.png"

    base = detection.processed.image.convert("RGB") if detection.processed else Image.open(image_path).convert("RGB")
    processed = detection.processed.image if detection.processed else base.convert("L")
    processed.save(processed_path)

    draw = ImageDraw.Draw(base)
    if detection.frame is not None:
        draw.rectangle(detection.frame, outline=(100, 116, 139), width=4)
        draw.text((detection.frame[0] + 8, max(0, detection.frame[1] - 18)), "drawing_frame", fill=(100, 116, 139))

    colors = {
        "title_block": (37, 99, 235),
        "parameter_table": (5, 150, 105),
        "section_view": (220, 38, 38),
        "circular_view": (147, 51, 234),
        "dimensions": (202, 138, 4),
    }
    width, height = base.size
    for box in detection.boxes:
        x1 = int(round(box.x * width))
        y1 = int(round(box.y * height))
        x2 = int(round((box.x + box.width) * width))
        y2 = int(round((box.y + box.height) * height))
        color = colors.get(box.target, (15, 23, 42))
        draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
        label = f"{box.target} {int(round(box.confidence * 100))}%"
        label_y = max(0, y1 - 18)
        draw.rectangle((x1, label_y, min(width, x1 + max(120, len(label) * 7)), label_y + 16), fill=(255, 255, 255))
        draw.text((x1 + 3, label_y + 1), label, fill=color)

    base.save(overlay_path)
    return (
        f"/api/projects/{project_id}/files/{overlay_path.name}",
        f"/api/projects/{project_id}/files/{processed_path.name}",
    )


def _normalized_frame(
    frame: tuple[int, int, int, int] | None,
    image_width: int | None,
    image_height: int | None,
) -> list[float] | None:
    if frame is None or image_width is None or image_height is None:
        return None
    x1, y1, x2, y2 = frame
    return [
        _round_ratio(x1, image_width),
        _round_ratio(y1, image_height),
        _round_ratio(x2 - x1, image_width),
        _round_ratio(y2 - y1, image_height),
    ]


def _detect_inner_frame(dark: np.ndarray) -> tuple[int, int, int, int] | None:
    image_height, image_width = dark.shape
    row_counts = dark.sum(axis=1)
    col_counts = dark.sum(axis=0)
    row_clusters = _clusters(np.flatnonzero(row_counts > image_width * 0.50), gap=4)
    col_clusters = _clusters(np.flatnonzero(col_counts > image_height * 0.50), gap=4)

    if len(col_clusters) >= 4:
        left = _cluster_mid(col_clusters[1])
        right = _cluster_mid(col_clusters[-2])
    elif len(col_clusters) >= 2:
        left = _cluster_mid(col_clusters[0])
        right = _cluster_mid(col_clusters[-1])
    else:
        left = right = None

    if len(row_clusters) >= 4:
        top = _cluster_mid(row_clusters[1])
        bottom = _cluster_mid(row_clusters[-2])
    elif len(row_clusters) >= 3:
        top = _cluster_mid(row_clusters[1])
        bottom = _cluster_mid(row_clusters[-1])
    elif len(row_clusters) >= 2:
        top = _cluster_mid(row_clusters[0])
        bottom = _cluster_mid(row_clusters[-1])
    else:
        top = bottom = None

    if left is None or right is None or top is None or bottom is None or right <= left or bottom <= top:
        return _ink_bbox(dark, pad=0)
    return left, top, right, bottom


def _clusters(indices: np.ndarray, gap: int = 1) -> list[tuple[int, int]]:
    if indices.size == 0:
        return []
    clusters: list[tuple[int, int]] = []
    start = int(indices[0])
    prev = int(indices[0])
    for raw in indices[1:]:
        value = int(raw)
        if value - prev <= gap + 1:
            prev = value
            continue
        clusters.append((start, prev))
        start = prev = value
    clusters.append((start, prev))
    return clusters


def _cluster_mid(cluster: tuple[int, int]) -> int:
    return int(round((cluster[0] + cluster[1]) / 2))


def _bbox_in_roi(dark: np.ndarray, roi: tuple[float, float, float, float], pad: int = 0) -> tuple[int, int, int, int] | None:
    image_height, image_width = dark.shape
    x1 = max(0, min(image_width - 1, int(round(roi[0]))))
    y1 = max(0, min(image_height - 1, int(round(roi[1]))))
    x2 = max(x1 + 1, min(image_width, int(round(roi[2]))))
    y2 = max(y1 + 1, min(image_height, int(round(roi[3]))))
    bbox = _ink_bbox(dark[y1:y2, x1:x2], pad=pad)
    if bbox is None:
        return None
    bx1, by1, bx2, by2 = bbox
    return (
        max(0, x1 + bx1),
        max(0, y1 + by1),
        min(image_width, x1 + bx2),
        min(image_height, y1 + by2),
    )


def _grid_bbox_in_roi(
    dark: np.ndarray,
    roi: tuple[float, float, float, float],
    vertical_threshold: float,
    horizontal_threshold: float,
) -> tuple[int, int, int, int] | None:
    image_height, image_width = dark.shape
    x1 = max(0, min(image_width - 1, int(round(roi[0]))))
    y1 = max(0, min(image_height - 1, int(round(roi[1]))))
    x2 = max(x1 + 1, min(image_width, int(round(roi[2]))))
    y2 = max(y1 + 1, min(image_height, int(round(roi[3]))))
    sub = dark[y1:y2, x1:x2]
    roi_height, roi_width = sub.shape
    col_clusters = _clusters(np.flatnonzero(sub.sum(axis=0) > roi_height * vertical_threshold), gap=3)
    row_clusters = _clusters(np.flatnonzero(sub.sum(axis=1) > roi_width * horizontal_threshold), gap=3)
    if len(col_clusters) < 2 or len(row_clusters) < 2:
        return None
    left = x1 + _cluster_mid(col_clusters[0])
    right = x1 + _cluster_mid(col_clusters[-1])
    top = y1 + _cluster_mid(row_clusters[0])
    bottom = y1 + _cluster_mid(row_clusters[-1])
    if right - left < image_width * 0.05 or bottom - top < image_height * 0.05:
        return None
    return left, top, right, bottom


def _ink_bbox(dark: np.ndarray, pad: int = 0) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(dark)
    if xs.size < 20:
        return None
    height, width = dark.shape
    return (
        max(0, int(xs.min()) - pad),
        max(0, int(ys.min()) - pad),
        min(width, int(xs.max()) + 1 + pad),
        min(height, int(ys.max()) + 1 + pad),
    )


def _region_box(
    target: str,
    label: str,
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    confidence: float,
) -> RegionBox:
    x1, y1, x2, y2 = bbox
    return RegionBox(
        target=target,
        label=label,
        x=_round_ratio(x1, image_width),
        y=_round_ratio(y1, image_height),
        width=max(0.001, _round_ratio(x2 - x1, image_width)),
        height=max(0.001, _round_ratio(y2 - y1, image_height)),
        confidence=confidence,
        source=ANALYSIS_SOURCE,
    )


def _round_ratio(value: int | float, total: int) -> float:
    return round(max(0.0, min(1.0, float(value) / max(total, 1))), 4)


def _png_size(path: Path) -> tuple[int | None, int | None]:
    with path.open("rb") as fh:
        header = fh.read(24)
    if len(header) >= 24 and header.startswith(b"\x89PNG\r\n\x1a\n"):
        return struct.unpack(">II", header[16:24])
    return None, None


def _gif_size(path: Path) -> tuple[int | None, int | None]:
    with path.open("rb") as fh:
        header = fh.read(10)
    if len(header) >= 10 and header[:6] in {b"GIF87a", b"GIF89a"}:
        return struct.unpack("<HH", header[6:10])
    return None, None


def _jpeg_size(path: Path) -> tuple[int | None, int | None]:
    data = path.read_bytes()
    idx = 2
    while idx + 9 < len(data):
        if data[idx] != 0xFF:
            idx += 1
            continue
        marker = data[idx + 1]
        idx += 2
        if marker in {0xD8, 0xD9}:
            continue
        if idx + 2 > len(data):
            break
        segment_length = int.from_bytes(data[idx : idx + 2], "big")
        if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
            if idx + 7 <= len(data):
                height = int.from_bytes(data[idx + 3 : idx + 5], "big")
                width = int.from_bytes(data[idx + 5 : idx + 7], "big")
                return width, height
        idx += max(segment_length, 2)
    return None, None
