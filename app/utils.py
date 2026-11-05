"""Image processing, validation, and annotation utilities."""

import base64
import hashlib
import io
import logging
from typing import Dict, List, Tuple
from fastapi import HTTPException, UploadFile, status
from PIL import Image, ImageDraw, ImageFont

from app.config import (
    ALLOWED_EXTENSIONS,
    ALLOWED_MIME_TYPES,
    MAX_FILE_SIZE_BYTES,
)
from models.base import DetectionResult

logger = logging.getLogger(__name__)

# Distinct vibrant color palette for visual clarity
PALETTE = [
    (59, 130, 246),   # Blue
    (16, 185, 129),   # Emerald
    (245, 158, 11),   # Amber
    (239, 68, 68),    # Red
    (139, 92, 246),   # Purple
    (236, 72, 153),   # Pink
    (20, 184, 166),   # Teal
    (249, 115, 22),   # Orange
    (99, 102, 241),   # Indigo
    (168, 85, 247),   # Violet
    (34, 197, 94),    # Green
    (234, 179, 8),    # Yellow
]


def get_color_for_label(label: str) -> Tuple[int, int, int]:
    """Deterministically maps a class label to a color in the palette."""
    hash_val = int(hashlib.md5(label.encode("utf-8")).hexdigest(), 16)
    return PALETTE[hash_val % len(PALETTE)]


async def validate_and_read_image(file: UploadFile) -> Image.Image:
    """Validate uploaded file size, extension, MIME type, and image integrity.

    Args:
        file: UploadFile from FastAPI request.

    Returns:
        Verified PIL Image in RGB mode.

    Raises:
        HTTPException: If the file is invalid, corrupt, oversized, or empty.
    """
    filename = file.filename or "unknown"
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    # Validate extension
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension '{ext}'. Allowed extensions are: {', '.join(sorted(ALLOWED_EXTENSIONS))}.",
        )

    # Validate MIME type if available
    content_type = file.content_type
    if content_type and content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid MIME type '{content_type}'. Allowed formats are JPG, PNG, and WEBP.",
        )

    # Read contents into memory
    content = await file.read()

    # Check for empty file
    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded image file is empty (0 bytes).",
        )

    # Check file size limit (10 MB)
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=getattr(status, "HTTP_413_CONTENT_TOO_LARGE", 413),
            detail=f"File size exceeds the 10 MB limit (received {len(content) / (1024 * 1024):.2f} MB).",
        )

    # Verify PIL image readability and integrity
    try:
        raw_stream = io.BytesIO(content)
        img_verify = Image.open(raw_stream)
        img_verify.verify()
    except Exception as exc:
        logger.warning("Corrupt image upload rejected: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Corrupted or invalid image file. Could not decode image format.",
        ) from exc

    # Re-open verified image for processing and convert to RGB
    try:
        image = Image.open(io.BytesIO(content))
        image = image.convert("RGB")
        return image
    except Exception as exc:
        logger.warning("Failed to convert image to RGB: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to decode and convert image to RGB format.",
        ) from exc


def annotate_image(image: Image.Image, detections: List[DetectionResult]) -> Image.Image:
    """Draw bounding boxes and class labels onto a copy of the image.

    Args:
        image: Original PIL Image.
        detections: List of DetectionResult objects.

    Returns:
        Annotated copy of the PIL Image.
    """
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    w, h = image.size

    # Scale stroke width and font size adaptively based on image resolution
    base_dim = max(w, h)
    stroke_width = max(2, int(base_dim / 300))
    font_size = max(12, int(base_dim / 45))

    try:
        font = ImageFont.load_default(size=font_size)
    except TypeError:
        # Fallback for older PIL versions without size param
        font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2 = det.box
        color = get_color_for_label(det.label)

        # Draw bounding box
        draw.rectangle(
            [(x1, y1), (x2, y2)],
            outline=color,
            width=stroke_width,
        )

        # Label text with percentage
        score_pct = int(round(det.confidence * 100))
        label_text = f"{det.label} {score_pct}%"

        # Calculate text bounding box
        text_bbox = draw.textbbox((x1, y1), label_text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        padding = 4

        # Position label above box if space permits, else inside
        if y1 - text_h - (padding * 2) >= 0:
            bg_top = y1 - text_h - (padding * 2)
            bg_bottom = y1
        else:
            bg_top = y1
            bg_bottom = y1 + text_h + (padding * 2)

        bg_left = x1
        bg_right = min(x1 + text_w + (padding * 2), float(w))

        # Draw label background pill
        draw.rectangle(
            [(bg_left, bg_top), (bg_right, bg_bottom)],
            fill=color,
        )

        # Draw text inside pill (white text with slight contrast)
        draw.text(
            (bg_left + padding, bg_top + padding),
            label_text,
            fill=(255, 255, 255),
            font=font,
        )

    return annotated


def image_to_base64(image: Image.Image, format: str = "JPEG") -> str:
    """Encode a PIL image to a Base64 data URL string."""
    buffer = io.BytesIO()
    image.save(buffer, format=format, quality=90, optimize=True)
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{b64_str}"
