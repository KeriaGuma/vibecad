from __future__ import annotations

import csv
import importlib.util
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal, Protocol

from PIL import Image, ImageOps

from .models import OcrRegion, ProjectState, TextEntity
from .reference import _upload_url_to_path, analyze_reference

OcrLanguage = Literal["auto", "zh", "en"]
OcrEngine = Literal["auto", "edocr2", "paddle", "tesseract"]


@dataclass(frozen=True)
class OcrRun:
    regions: list[OcrRegion]
    warnings: list[str]


class OCRProvider(Protocol):
    """Mechanical drawing OCR provider contract.

    eDOCr/eDOCr2 can be added behind this interface without changing the
    pipeline or downstream dimension-binding code.
    """

    name: str
    language: str

    def run_regions(
        self,
        image: Image.Image,
        image_width: int,
        image_height: int,
        boxes,
        warnings: list[str],
    ) -> list[OcrRegion]:
        ...


@dataclass(frozen=True)
class TesseractOCRProvider:
    tesseract: str
    languages: str
    name: str = "tesseract"

    @property
    def language(self) -> str:
        return self.languages

    def run_regions(
        self,
        image: Image.Image,
        image_width: int,
        image_height: int,
        boxes,
        warnings: list[str],
    ) -> list[OcrRegion]:
        gray = image.convert("L")
        return [
            _ocr_region_with_tesseract(self.tesseract, gray, image_width, image_height, box, self.languages, warnings)
            for box in boxes
        ]


@dataclass(frozen=True)
class PaddleOCRProvider:
    language: Literal["zh", "en"]
    tesseract: str | None = None
    name: str = "paddleocr"

    def run_regions(
        self,
        image: Image.Image,
        image_width: int,
        image_height: int,
        boxes,
        warnings: list[str],
    ) -> list[OcrRegion]:
        rgb = image.convert("RGB")
        return [
            _ocr_region_with_paddle(
                rgb,
                image_width,
                image_height,
                box,
                self.language,
                self.tesseract,
                warnings,
            )
            for box in boxes
        ]


@dataclass(frozen=True)
class Edocr2OCRProvider:
    language: Literal["zh", "en"]
    name: str = "edocr2"

    def run_regions(
        self,
        image: Image.Image,
        image_width: int,
        image_height: int,
        boxes,
        warnings: list[str],
    ) -> list[OcrRegion]:
        raise RuntimeError("eDOCr2 provider is registered but no local adapter is configured yet.")


def run_project_ocr(
    project: ProjectState,
    uploads_dir: Path,
    language_hint: OcrLanguage = "auto",
    engine_hint: OcrEngine = "auto",
) -> OcrRun:
    if not project.source_image:
        raise ValueError("Upload a PDF or image before running OCR.")

    image_path = _upload_url_to_path(project.source_image, uploads_dir)
    if not image_path.exists():
        raise FileNotFoundError("Reference image not found")

    boxes = analyze_reference(project, uploads_dir).boxes
    language = _resolve_language(project, language_hint)
    warnings: list[str] = []
    provider = _select_ocr_provider(language, engine_hint, warnings)
    if provider is None:
        return OcrRun(
            regions=_empty_regions(boxes),
            warnings=[*warnings, "No OCR engine is available; install eDOCr2, PaddleOCR or Tesseract to enable OCR."],
        )

    with Image.open(image_path) as image:
        image_width, image_height = image.size
        try:
            regions = provider.run_regions(image, image_width, image_height, boxes, warnings)
        except Exception as exc:  # noqa: BLE001 - keep the pipeline alive and downgrade cleanly
            warnings.append(f"{provider.name} OCR failed: {exc}")
            fallback = _select_ocr_provider(language, "paddle" if provider.name != "paddleocr" else "tesseract", warnings)
            if fallback is None or fallback.name == provider.name:
                regions = _empty_regions(boxes)
            else:
                regions = fallback.run_regions(image, image_width, image_height, boxes, warnings)
    return OcrRun(regions=regions, warnings=warnings)


def _empty_regions(boxes) -> list[OcrRegion]:
    return [
        OcrRegion(
            target=box.target,
            label=box.label,
            x=box.x,
            y=box.y,
            width=box.width,
            height=box.height,
            engine="none",
            source="ocr_unavailable",
        )
        for box in boxes
    ]


def _resolve_language(project: ProjectState, language_hint: OcrLanguage) -> Literal["zh", "en"]:
    if language_hint in {"zh", "en"}:
        return language_hint

    semantic_text = " ".join(
        entity.text
        for entity in project.ir.entities
        if isinstance(entity, TextEntity) and ("semantic_import" in entity.tags or entity.group)
    )
    if _contains_cjk(semantic_text):
        return "zh"
    if semantic_text.strip():
        return "en"

    # The product is currently optimized for Chinese mechanical drawings. When
    # the drawing language is unknown, prefer the Chinese-capable OCR branch if
    # PaddleOCR is available; the UI can still force EN for English-only sheets.
    return "zh" if _paddleocr_available() else "en"


def _contains_cjk(text: str) -> bool:
    return any("\u3400" <= char <= "\u9fff" for char in text)


def _select_engine(language: Literal["zh", "en"], engine_hint: OcrEngine) -> Literal["paddle", "tesseract"]:
    if engine_hint == "paddle":
        return "paddle"
    if engine_hint == "tesseract":
        return "tesseract"
    return "paddle" if language == "zh" and _paddleocr_available() else "tesseract"


def _select_ocr_provider(
    language: Literal["zh", "en"],
    engine_hint: OcrEngine,
    warnings: list[str],
) -> OCRProvider | None:
    tesseract = shutil.which("tesseract")

    if engine_hint == "edocr2":
        if _edocr2_available():
            warnings.append("eDOCr2 runtime was detected, but the local adapter is not configured yet; using fallback OCR.")
        else:
            warnings.append("eDOCr2 runtime is not installed; using fallback OCR.")

    if engine_hint == "paddle" or (engine_hint in {"auto", "edocr2"} and language == "zh"):
        if _paddleocr_available():
            return PaddleOCRProvider(language=language, tesseract=tesseract)
        if engine_hint == "paddle":
            warnings.append("PaddleOCR runtime is not installed; falling back to Tesseract.")

    if engine_hint == "tesseract" or engine_hint in {"auto", "edocr2", "paddle"}:
        if tesseract:
            languages, lang_warnings = _select_languages(tesseract)
            if language == "zh" and "chi_sim" not in languages:
                warnings.append("Chinese OCR was requested, but Tesseract chi_sim is not installed.")
            warnings.extend(lang_warnings)
            return TesseractOCRProvider(tesseract=tesseract, languages=languages)
        if engine_hint == "tesseract":
            warnings.append("Tesseract executable was not found.")

    if engine_hint in {"auto", "tesseract", "edocr2"} and _paddleocr_available():
        warnings.append("Tesseract executable was not found; falling back to PaddleOCR.")
        return PaddleOCRProvider(language=language, tesseract=None)

    return None


def _paddleocr_available() -> bool:
    return importlib.util.find_spec("paddleocr") is not None and importlib.util.find_spec("paddle") is not None


def _edocr2_available() -> bool:
    return importlib.util.find_spec("edocr2") is not None or importlib.util.find_spec("edocr") is not None


def _select_languages(tesseract: str) -> tuple[str, list[str]]:
    warnings: list[str] = []
    try:
        result = subprocess.run(
            [tesseract, "--list-langs"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:  # noqa: BLE001 - make OCR setup failures user-visible
        return "eng", [f"Could not inspect Tesseract languages: {exc}. Falling back to eng."]

    langs = {
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip() and not line.lower().startswith("list of available languages")
    }
    if "chi_sim" in langs:
        return "chi_sim+eng", warnings
    if "eng" in langs and "snum" in langs:
        warnings.append("Tesseract chi_sim language pack is not installed; OCR is limited to English/numbers.")
        return "eng+snum", warnings
    if "eng" in langs:
        warnings.append("Tesseract chi_sim language pack is not installed; OCR is limited to English.")
        return "eng", warnings
    language = sorted(langs)[0] if langs else "eng"
    warnings.append(f"Tesseract chi_sim/eng language packs are not installed; using {language}.")
    return language, warnings


def _ocr_region_with_tesseract(
    tesseract: str,
    image: Image.Image,
    image_width: int,
    image_height: int,
    box,
    languages: str,
    warnings: list[str],
) -> OcrRegion:
    x1 = max(0, min(image_width - 1, int(round(box.x * image_width))))
    y1 = max(0, min(image_height - 1, int(round(box.y * image_height))))
    x2 = max(x1 + 1, min(image_width, int(round((box.x + box.width) * image_width))))
    y2 = max(y1 + 1, min(image_height, int(round((box.y + box.height) * image_height))))
    crop = _preprocess_crop(image.crop((x1, y1, x2, y2)))
    text = ""
    confidence = 0.0
    with tempfile.TemporaryDirectory(prefix="vibecad_ocr_") as tmp:
        crop_path = Path(tmp) / "crop.png"
        crop.save(crop_path)
        try:
            result = subprocess.run(
                [tesseract, str(crop_path), "stdout", "-l", languages, "--psm", "6", "tsv"],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            warnings.append(f"OCR timed out for {box.label}.")
            result = None
        except Exception as exc:  # noqa: BLE001 - keep the endpoint non-fatal per region
            warnings.append(f"OCR failed for {box.label}: {exc}.")
            result = None

    if result is not None:
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
            warnings.append(f"OCR failed for {box.label}: {detail}")
        else:
            text, confidence = _parse_tesseract_tsv(result.stdout)

    return OcrRegion(
        target=box.target,
        label=box.label,
        text=text,
        confidence=confidence,
        x=box.x,
        y=box.y,
        width=box.width,
        height=box.height,
        engine="tesseract",
        language=languages,
    )


def _ocr_region_with_paddle(
    image: Image.Image,
    image_width: int,
    image_height: int,
    box,
    language: Literal["zh", "en"],
    tesseract: str | None,
    warnings: list[str],
) -> OcrRegion:
    x1 = max(0, min(image_width - 1, int(round(box.x * image_width))))
    y1 = max(0, min(image_height - 1, int(round(box.y * image_height))))
    x2 = max(x1 + 1, min(image_width, int(round((box.x + box.width) * image_width))))
    y2 = max(y1 + 1, min(image_height, int(round((box.y + box.height) * image_height))))
    crop = _preprocess_paddle_crop(image.crop((x1, y1, x2, y2)))

    try:
        text, confidence = _run_paddle_ocr(crop, language)
        return OcrRegion(
            target=box.target,
            label=box.label,
            text=text,
            confidence=confidence,
            x=box.x,
            y=box.y,
            width=box.width,
            height=box.height,
            engine="paddleocr",
            language=language,
            source="paddleocr",
        )
    except Exception as exc:  # noqa: BLE001 - fallback keeps OCR endpoint useful
        warnings.append(f"PaddleOCR failed for {box.label}: {exc}.")

    if tesseract:
        tesseract_languages, lang_warnings = _select_languages(tesseract)
        warnings.extend(lang_warnings)
        fallback = _ocr_region_with_tesseract(
            tesseract,
            image.convert("L"),
            image_width,
            image_height,
            box,
            tesseract_languages,
            warnings,
        )
        fallback.source = "paddleocr_fallback_tesseract"
        return fallback

    return OcrRegion(
        target=box.target,
        label=box.label,
        x=box.x,
        y=box.y,
        width=box.width,
        height=box.height,
        engine="none",
        language=language,
        source="paddleocr_failed",
    )


def _preprocess_paddle_crop(crop: Image.Image) -> Image.Image:
    crop = ImageOps.expand(crop.convert("RGB"), border=max(8, round(max(crop.size) * 0.025)), fill="white")
    scale = 3 if max(crop.size) < 900 else 2
    crop = crop.resize((crop.width * scale, crop.height * scale), Image.Resampling.LANCZOS)
    return ImageOps.autocontrast(crop)


def _run_paddle_ocr(crop: Image.Image, language: Literal["zh", "en"]) -> tuple[str, float]:
    with tempfile.TemporaryDirectory(prefix="vibecad_paddle_ocr_") as tmp:
        crop_path = Path(tmp) / "crop.png"
        crop.save(crop_path)
        result = _get_paddle_ocr(language).predict(str(crop_path))
    return _parse_paddle_result(result)


@lru_cache(maxsize=2)
def _get_paddle_ocr(language: Literal["zh", "en"]):
    from paddleocr import PaddleOCR

    lang = "ch" if language == "zh" else "en"
    return PaddleOCR(
        lang=lang,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


def _parse_paddle_result(result) -> tuple[str, float]:
    texts: list[str] = []
    scores: list[float] = []

    def add_text(text, score=None) -> None:
        value = str(text).strip()
        if not value:
            return
        texts.append(value)
        try:
            numeric_score = float(score)
        except (TypeError, ValueError):
            numeric_score = -1
        if numeric_score >= 0:
            scores.append(numeric_score)

    def mapping_value(item, key: str):
        if isinstance(item, dict):
            return item.get(key)
        try:
            return item[key]
        except Exception:  # noqa: BLE001 - Paddle result objects are dict-like
            return None

    pages = result if isinstance(result, list) else [result]
    for page in pages:
        rec_texts = mapping_value(page, "rec_texts")
        rec_scores = mapping_value(page, "rec_scores")
        if rec_texts is not None:
            for idx, text in enumerate(rec_texts):
                score = rec_scores[idx] if isinstance(rec_scores, list) and idx < len(rec_scores) else None
                add_text(text, score)
            continue

        if isinstance(page, list):
            for item in page:
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    continue
                payload = item[1]
                if isinstance(payload, (list, tuple)) and payload:
                    add_text(payload[0], payload[1] if len(payload) > 1 else None)

    confidence = round(sum(scores) / len(scores), 3) if scores else 0.0
    return " ".join(texts), confidence


def _preprocess_crop(crop: Image.Image) -> Image.Image:
    crop = ImageOps.expand(crop, border=max(6, round(max(crop.size) * 0.02)), fill=255)
    scale = 3 if max(crop.size) < 900 else 2
    crop = crop.resize((crop.width * scale, crop.height * scale), Image.Resampling.LANCZOS)
    crop = ImageOps.autocontrast(crop)
    return crop.point(lambda value: 0 if value < 185 else 255)


def _parse_tesseract_tsv(payload: str) -> tuple[str, float]:
    rows = csv.DictReader(payload.splitlines(), delimiter="\t")
    words: list[str] = []
    confidences: list[float] = []
    for row in rows:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        words.append(text)
        try:
            confidence = float(row.get("conf") or -1)
        except ValueError:
            confidence = -1
        if confidence >= 0:
            confidences.append(confidence)
    text = " ".join(words)
    confidence = round(sum(confidences) / len(confidences) / 100, 3) if confidences else 0.0
    return text, confidence
