"""Application configuration settings."""

from typing import Set

APP_NAME = "Object Detector & Counter"
VERSION = "1.0.0"

# Maximum file size: 10 MB in bytes
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024

# Allowed image file extensions and MIME types
ALLOWED_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_MIME_TYPES: Set[str] = {
    "image/jpeg",
    "image/png",
    "image/webp",
}

# Detection defaults
DEFAULT_CONFIDENCE_THRESHOLD = 0.50
DEFAULT_NMS_THRESHOLD = 0.50
MIN_THRESHOLD = 0.01
MAX_THRESHOLD = 1.00

