"""Unit and integration tests for Object Detector & Counter API endpoints."""

import os
from fastapi.testclient import TestClient
import pytest


def test_health_endpoint(client: TestClient):
    """GET /health returns 200 with status ok and model information."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "device" in data
    assert "models" in data
    assert "version" in data


def test_root_endpoint(client: TestClient):
    """GET / returns 200 with HTML content."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "Object Detector &amp; Counter" in response.text or "Object Detector & Counter" in response.text


def test_detect_valid_jpeg(client: TestClient, valid_jpeg_bytes: bytes):
    """POST /detect with valid JPEG returns 200 and required fields."""
    response = client.post(
        "/detect",
        files={"file": ("test.jpg", valid_jpeg_bytes, "image/jpeg")},
        data={"threshold": "0.5", "nms_threshold": "0.5", "mode": "standard"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "counts" in data
    assert "detections" in data
    assert "annotated_image" in data
    assert isinstance(data["total"], int)
    assert isinstance(data["counts"], dict)
    assert isinstance(data["detections"], list)
    assert data["annotated_image"].startswith("data:image/jpeg;base64,")


def test_detect_valid_png(client: TestClient, valid_png_bytes: bytes):
    """POST /detect with valid PNG returns 200."""
    response = client.post(
        "/detect",
        files={"file": ("test.png", valid_png_bytes, "image/png")},
        data={"threshold": "0.4"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert isinstance(data["total"], int)


def test_detect_valid_webp(client: TestClient, valid_webp_bytes: bytes):
    """POST /detect with valid WEBP returns 200."""
    response = client.post(
        "/detect",
        files={"file": ("test.webp", valid_webp_bytes, "image/webp")},
        data={"threshold": "0.5"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "total" in data


def test_detect_invalid_file_type(client: TestClient):
    """POST /detect rejects unsupported file extensions with 400."""
    fake_txt = b"This is a text file, not an image."
    response = client.post(
        "/detect",
        files={"file": ("document.txt", fake_txt, "text/plain")},
    )
    assert response.status_code == 400
    detail = response.json().get("detail", "")
    assert "Unsupported file extension" in detail or "Invalid MIME type" in detail


def test_detect_empty_file(client: TestClient):
    """POST /detect rejects 0-byte file with 400."""
    response = client.post(
        "/detect",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert response.status_code == 400
    assert "empty" in response.json().get("detail", "").lower()


def test_detect_oversized_file(client: TestClient, oversized_file_bytes: bytes):
    """POST /detect rejects files strictly greater than 10 MB with 413."""
    response = client.post(
        "/detect",
        files={"file": ("large_image.jpg", oversized_file_bytes, "image/jpeg")},
    )
    assert response.status_code == 413
    assert "10 MB" in response.json().get("detail", "")


def test_detect_corrupt_file(client: TestClient, corrupt_image_bytes: bytes):
    """POST /detect rejects corrupted image contents with 400."""
    response = client.post(
        "/detect",
        files={"file": ("corrupt.jpg", corrupt_image_bytes, "image/jpeg")},
    )
    assert response.status_code == 400
    assert "corrupt" in response.json().get("detail", "").lower()


def test_detect_threshold_validation(client: TestClient, valid_jpeg_bytes: bytes):
    """POST /detect rejects thresholds outside the allowed [0.01, 1.00] range."""
    # Too high
    res_high = client.post(
        "/detect",
        files={"file": ("test.jpg", valid_jpeg_bytes, "image/jpeg")},
        data={"threshold": "1.5"},
    )
    assert res_high.status_code == 400

    # Too low
    res_low = client.post(
        "/detect",
        files={"file": ("test.jpg", valid_jpeg_bytes, "image/jpeg")},
        data={"threshold": "-0.1"},
    )
    assert res_low.status_code == 400


def test_detect_open_vocabulary_mode(client: TestClient):
    """POST /detect works with open-vocabulary mode and custom labels."""
    sample_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "static",
        "sample_images",
        "street.jpg",
    )
    if not os.path.exists(sample_path):
        pytest.skip("Sample image street.jpg not present.")

    with open(sample_path, "rb") as f:
        img_bytes = f.read()

    response = client.post(
        "/detect",
        files={"file": ("street.jpg", img_bytes, "image/jpeg")},
        data={
            "threshold": "0.15",
            "nms_threshold": "0.5",
            "mode": "open_vocab",
            "custom_labels": "car, vehicle, wheel, house",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "counts" in data
    assert "detections" in data


def test_detect_sample_image_detection(client: TestClient):
    """POST /detect correctly identifies objects in standard mode on sample image."""
    sample_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "static",
        "sample_images",
        "street.jpg",
    )
    if not os.path.exists(sample_path):
        pytest.skip("Sample image street.jpg not present.")

    with open(sample_path, "rb") as f:
        img_bytes = f.read()

    response = client.post(
        "/detect",
        files={"file": ("street.jpg", img_bytes, "image/jpeg")},
        data={"threshold": "0.35", "nms_threshold": "0.5", "mode": "standard"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total"] >= 1  # Should detect at least 1 object (e.g. car)
    assert len(data["counts"]) >= 1
    # Check structure of detections
    for det in data["detections"]:
        assert "label" in det
        assert "confidence" in det
        assert "box" in det
        assert len(det["box"]) == 4
        assert det["confidence"] >= 0.35

