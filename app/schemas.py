"""Pydantic data models for API requests and responses."""

from typing import Dict, List
from pydantic import BaseModel, Field


class DetectionItem(BaseModel):
    """Single detected object bounding box and confidence."""
    label: str = Field(..., description="Detected object class name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    box: List[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Bounding box coordinates [x1, y1, x2, y2] in pixels",
    )


class DetectResponse(BaseModel):
    """Response payload for POST /detect."""
    total: int = Field(..., description="Total count of all detected objects")
    counts: Dict[str, int] = Field(
        ...,
        description="Per-class counts breakdown, e.g. {'person': 2, 'chair': 3}",
    )
    detections: List[DetectionItem] = Field(
        ...,
        description="List of all detected objects with labels, scores, and coordinates",
    )
    annotated_image: str = Field(
        ...,
        description="Base64-encoded annotated image with drawn bounding boxes and labels",
    )


class HealthResponse(BaseModel):
    """Response payload for GET /health."""
    status: str
    device: str
    models: Dict[str, str]
    version: str

