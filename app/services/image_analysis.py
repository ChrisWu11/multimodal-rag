import io
import statistics
from pathlib import Path
from typing import Any, Dict, Optional

from google import genai
from google.genai import types
from PIL import Image, ImageStat

from app.core.config import Settings


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
        self.client: Optional[genai.Client] = None
        if settings.gemini_api_key:
            self.client = genai.Client(api_key=settings.gemini_api_key)

    def summarize(self, filename: str, data: bytes, modality: str = "unknown") -> tuple[str, Dict[str, Any]]:
        basic_summary, metadata = build_basic_image_summary(filename, data, modality)
        if not self.client or not self.settings.enable_gemini_vision:
            return basic_summary, metadata

        media_type = _media_type(filename)
        prompt = (
            "You are helping build a research RAG system for ultrasound and thermal imaging. "
            "Describe observable visual features only. Do not diagnose. "
            "Return a concise structured summary with acquisition-quality notes, visible patterns, "
            "uncertainties, and metadata that would be useful for retrieval."
        )
        response = self.client.models.generate_content(
            model=self.settings.gemini_model,
            contents=[
                types.Part.from_bytes(data=data, mime_type=media_type),
                f"{prompt}\nModality: {modality}.",
            ],
        )
        gemini_summary = (response.text or "").strip()
        metadata["gemini_vision_used"] = True
        return f"{basic_summary}\n\nVision model summary:\n{gemini_summary}", metadata


def _media_type(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    if suffix == ".gif":
        return "image/gif"
    return "image/png"
