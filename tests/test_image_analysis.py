import io

import pytest
from PIL import Image

from app.services.image_analysis import ImageValidationError, _response_text, validate_image_upload


def png_bytes(width: int = 12, height: int = 8) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (width, height), color=(120, 40, 30)).save(output, format="PNG")
    return output.getvalue()


def test_validate_image_upload_accepts_png() -> None:
    metadata = validate_image_upload(
        filename="thermal.png",
        data=png_bytes(),
        content_type="image/png",
        max_bytes=1024 * 1024,
        max_pixels=1000,
    )

    assert metadata["format"] == "PNG"
    assert metadata["width"] == 12
    assert metadata["height"] == 8


def test_validate_image_upload_rejects_invalid_content() -> None:
    with pytest.raises(ImageValidationError, match="valid readable image"):
        validate_image_upload(
            filename="thermal.png",
            data=b"not an image",
            content_type="image/png",
            max_bytes=1024 * 1024,
            max_pixels=1000,
        )


def test_validate_image_upload_rejects_excessive_resolution() -> None:
    with pytest.raises(ImageValidationError, match="resolution is too large"):
        validate_image_upload(
            filename="thermal.png",
            data=png_bytes(width=20, height=20),
            content_type="image/png",
            max_bytes=1024 * 1024,
            max_pixels=100,
        )


def test_validate_image_upload_rejects_extension_mismatch() -> None:
    with pytest.raises(ImageValidationError, match="filename extension"):
        validate_image_upload(
            filename="thermal.jpg",
            data=png_bytes(),
            content_type="image/jpeg",
            max_bytes=1024 * 1024,
            max_pixels=1000,
        )


def test_response_text_extracts_multimodal_content_blocks() -> None:
    content = [
        {"type": "text", "text": "Image type: ultrasound"},
        {"type": "text", "text": "Observable features: focal bright region"},
    ]

    assert _response_text(content) == (
        "Image type: ultrasound\nObservable features: focal bright region"
    )
