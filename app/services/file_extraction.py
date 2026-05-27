import json
from pathlib import Path
from typing import Tuple

from pypdf import PdfReader

TEXT_SUFFIXES = {".txt", ".md", ".csv", ".json"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
PDF_SUFFIXES = {".pdf"}


class ExtractionError(ValueError):
    pass


def source_type_for_filename(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return "text"
    if suffix in PDF_SUFFIXES:
        return "pdf"
    if suffix in IMAGE_SUFFIXES:
        return "image"
    return "unknown"


def extract_text(filename: str, data: bytes) -> Tuple[str, str]:
    source_type = source_type_for_filename(filename)
    if source_type == "text":
        return _decode_text(data), source_type
    if source_type == "pdf":
        return _extract_pdf_text(data), source_type
    raise ExtractionError(f"Unsupported text extraction for file type: {Path(filename).suffix}")


def _decode_text(data: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ExtractionError("Could not decode file as text.")


def _extract_pdf_text(data: bytes) -> str:
    import io

    reader = PdfReader(io.BytesIO(data))
    pages = []
    for index, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append(f"[Page {index + 1}]\n{text}")
    return "\n\n".join(pages).strip()


def parse_metadata_json(metadata_json: str | None) -> dict:
    if not metadata_json:
        return {}
    try:
        parsed = json.loads(metadata_json)
    except json.JSONDecodeError as exc:
        raise ExtractionError(f"metadata_json must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ExtractionError("metadata_json must decode to an object.")
    return parsed
