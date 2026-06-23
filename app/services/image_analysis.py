import base64
import io
import statistics
from pathlib import Path
from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from PIL import Image, ImageStat, UnidentifiedImageError

from app.core.config import Settings
from app.rag.langchain_providers import build_chat_model, normalize_provider


ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}
ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_IMAGE_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
FORMAT_BY_SUFFIX = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".webp": "WEBP"}
FORMAT_BY_CONTENT_TYPE = {"image/jpeg": "JPEG", "image/png": "PNG", "image/webp": "WEBP"}


class ImageValidationError(ValueError):
    pass


def validate_image_upload(
    filename: str,
    data: bytes,
    content_type: Optional[str],
    max_bytes: int,
    max_pixels: int,
) -> Dict[str, Any]:
    if not data:
        raise ImageValidationError("The uploaded image is empty.")
    if len(data) > max_bytes:
        limit_mb = max_bytes / (1024 * 1024)
        raise ImageValidationError(f"Image must be no larger than {limit_mb:g} MB.")

    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_IMAGE_SUFFIXES:
        raise ImageValidationError("Supported image formats are PNG, JPEG, and WebP.")
    if content_type and content_type.lower() not in ALLOWED_IMAGE_CONTENT_TYPES:
        raise ImageValidationError("The uploaded file does not have a supported image MIME type.")

    try:
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
            image_format = (image.format or "").upper()
            if image_format not in ALLOWED_IMAGE_FORMATS:
                raise ImageValidationError("Supported image formats are PNG, JPEG, and WebP.")
            if FORMAT_BY_SUFFIX[suffix] != image_format:
                raise ImageValidationError("The image content does not match its filename extension.")
            if content_type and FORMAT_BY_CONTENT_TYPE[content_type.lower()] != image_format:
                raise ImageValidationError("The image content does not match its MIME type.")
            if width <= 0 or height <= 0:
                raise ImageValidationError("The uploaded image has invalid dimensions.")
            if width * height > max_pixels:
                raise ImageValidationError(
                    f"Image resolution is too large. Maximum allowed pixel count is {max_pixels:,}."
                )
            image.verify()
    except ImageValidationError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageValidationError("The uploaded file is not a valid readable image.") from exc

    return {
        "filename": filename,
        "width": width,
        "height": height,
        "format": image_format,
        "content_type": content_type or _media_type(filename),
        "size_bytes": len(data),
    }


def build_basic_image_summary(filename: str, data: bytes, modality: str = "unknown") -> tuple[str, Dict[str, Any]]:
    with Image.open(io.BytesIO(data)) as image:
        width, height = image.size
        mode = image.mode
        fmt = image.format or Path(filename).suffix.replace(".", "").upper()

        metadata: Dict[str, Any] = {
            "filename": filename,
            "width": width,
            "height": height,
            "mode": mode,
            "format": fmt,
        }

        summary_parts = [
            f"Image file '{filename}' was uploaded as modality '{modality}'.",
            f"Dimensions: {width}x{height}; mode: {mode}; format: {fmt}.",
        ]

        gray = image.convert("L")
        stat = ImageStat.Stat(gray)
        mean_intensity = stat.mean[0]
        extrema = gray.getextrema()
        metadata.update(
            {
                "mean_intensity": round(mean_intensity, 2),
                "min_intensity": int(extrema[0]),
                "max_intensity": int(extrema[1]),
            }
        )

        if modality.lower() == "thermal":
            summary_parts.append(
                "Basic thermal proxy stats from pixel intensity: "
                f"mean={mean_intensity:.2f}, min={extrema[0]}, max={extrema[1]}. "
                "These are not calibrated temperature values unless the image source provides radiometric metadata."
            )
        elif modality.lower() == "ultrasound":
            summary_parts.append(
                "Basic ultrasound proxy stats were extracted from grayscale intensity. "
                "Clinical interpretation requires acquisition metadata and expert validation."
            )
        else:
            summary_parts.append("Only basic image metadata was extracted locally.")

        try:
            metadata["intensity_stdev"] = round(statistics.pstdev(gray.getdata()), 2)
        except statistics.StatisticsError:
            metadata["intensity_stdev"] = 0.0

    return " ".join(summary_parts), metadata


class ImageAnalyzer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.provider = normalize_provider(settings.llm_provider)
        self.chat_model = build_chat_model(settings)

    def summarize(
        self,
        filename: str,
        data: bytes,
        modality: str = "unknown",
        question: Optional[str] = None,
    ) -> tuple[str, Dict[str, Any]]:
        basic_summary, metadata = build_basic_image_summary(filename, data, modality)
        if not self.chat_model or not self._vision_enabled():
            return basic_summary, metadata

        media_type = _media_type(filename)
        encoded = base64.b64encode(data).decode("ascii")
        prompt = (
            "You are helping build a research RAG system for ultrasound and thermal imaging. "
            "Analyse only observable visual features that are relevant to the user's question. "
            "Do not diagnose, recommend treatment, identify a patient, or infer facts that are not visible. "
            "Do not treat grayscale or colour intensity as a calibrated temperature unless radiometric "
            "metadata is explicitly visible. Return concise plain text with exactly these headings: "
            "Image type, Observable features, Regions relevant to the question, Image quality limitations, "
            "Uncertainty, Retrieval terms. Retrieval terms should contain short scientific phrases that can "
            "help search a literature knowledge base."
        )
        user_question = question.strip() if question and question.strip() else "Provide a general research description."
        try:
            response = self.chat_model.invoke(
                [
                    SystemMessage(content=prompt),
                    HumanMessage(
                        content=[
                            {
                                "type": "text",
                                "text": f"Image modality: {modality}.\nUser question: {user_question}",
                            },
                            {
                                "type": "image_url",
                                "image_url": f"data:{media_type};base64,{encoded}",
                            },
                        ]
                    ),
                ]
            )
        except Exception as exc:
            metadata["vision_error"] = str(exc)
            return basic_summary, metadata

        vision_summary = _response_text(response.content)
        if not vision_summary:
            metadata["vision_error"] = "Vision model returned no text content."
            return basic_summary, metadata
        metadata["vision_provider"] = self.provider
        metadata["question_aware_summary"] = bool(question and question.strip())
        return f"{basic_summary}\n\nVision model summary:\n{vision_summary}", metadata

    def _vision_enabled(self) -> bool:
        if self.provider == "gemini":
            return self.settings.enable_gemini_vision
        return self.settings.enable_llm_generation


def _media_type(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    if suffix == ".gif":
        return "image/gif"
    return "image/png"


def _response_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(part.strip() for part in parts if part.strip()).strip()
    return str(content or "").strip()
