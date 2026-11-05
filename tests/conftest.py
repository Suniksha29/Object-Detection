"""Pytest fixtures and test helpers for Object Detector & Counter."""

import io
from typing import Generator
from fastapi.testclient import TestClient
from PIL import Image
import pytest

from app.main import app


@pytest.fixture(scope="session")
def client() -> Generator[TestClient, None, None]:
    """Provide a session-scoped TestClient with initialized lifespan models."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_jpeg_bytes() -> bytes:
    """Generate in-memory valid JPEG image bytes."""
    img = Image.new("RGB", (320, 240), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def valid_png_bytes() -> bytes:
    """Generate in-memory valid PNG image bytes."""
    img = Image.new("RGB", (200, 200), color=(50, 180, 80))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def valid_webp_bytes() -> bytes:
    """Generate in-memory valid WEBP image bytes."""
    img = Image.new("RGB", (250, 250), color=(220, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="WEBP")
    return buf.getvalue()


@pytest.fixture
def corrupt_image_bytes() -> bytes:
    """Generate invalid corrupted image bytes with a .jpg filename."""
    return b"NOT_A_VALID_IMAGE_HEADER_DATA_12345"


@pytest.fixture
def oversized_file_bytes() -> bytes:
    """Generate a dummy byte array strictly exceeding 10 MB limit."""
    # 10 MB + 1024 bytes
    return b"0" * ((10 * 1024 * 1024) + 1024)

