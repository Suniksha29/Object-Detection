"""FastAPI API route definitions for object detection and health checks."""

from collections import Counter
import logging
from typing import Optional
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status

from app.config import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_NMS_THRESHOLD,
    MAX_THRESHOLD,
    MIN_THRESHOLD,
    VERSION,
)
from app.schemas import DetectResponse, HealthResponse
from app.utils import annotate_image, image_to_base64, validate_and_read_image

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check(request: Request) -> HealthResponse:
    """System health check and loaded model status."""
    manager = getattr(request.app.state, "detector_manager", None)
    if manager is None:
        return HealthResponse(
            status="initializing",
            device="unknown",
            models={},
            version=VERSION,
        )

    model_status = manager.get_status()
    return HealthResponse(
        status="ok",
        device=model_status["device"],
        models={
            "standard_detector": model_status["standard_model"],
            "open_vocabulary_detector": model_status["open_vocab_model"],
        },
        version=VERSION,
    )


@router.post("/detect", response_model=DetectResponse, tags=["Inference"])
async def detect_objects(
    request: Request,
    file: UploadFile = File(..., description="Image file (JPG, PNG, WEBP, max 10MB)"),
    threshold: float = Form(
        DEFAULT_CONFIDENCE_THRESHOLD,
        description="Confidence threshold cutoff (0.01 - 1.00)",
    ),
    nms_threshold: float = Form(
        DEFAULT_NMS_THRESHOLD,
        description="Non-Maximum Suppression IoU cutoff (0.01 - 1.00)",
    ),
    custom_labels: Optional[str] = Form(
        None,
        description="Optional comma-separated list of target classes for open-vocabulary detection",
    ),
    mode: str = Form(
        "standard",
        description="Detection mode: 'standard' (COCO 80 classes) or 'open_vocab' (custom classes)",
    ),
    enable_slicing: bool = Form(
        True,
        description="Enable High-Resolution Slicing (SAHI) to detect tiny, distant, or occluded objects",
    ),
) -> DetectResponse:
    """Accept an image file, detect real-world objects, and return labeled detections and counts."""
    # Validate threshold range
    if not (MIN_THRESHOLD <= threshold <= MAX_THRESHOLD):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Confidence threshold must be between {MIN_THRESHOLD} and {MAX_THRESHOLD}.",
        )

    if not (MIN_THRESHOLD <= nms_threshold <= MAX_THRESHOLD):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"NMS threshold must be between {MIN_THRESHOLD} and {MAX_THRESHOLD}.",
        )

    # Validate and load image
    image = await validate_and_read_image(file)

    # Access preloaded detector manager
    manager = getattr(request.app.state, "detector_manager", None)
    if manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Detection model service is not yet loaded.",
        )

    # Run inference
    try:
        detections = manager.detect(
            image=image,
            threshold=threshold,
            nms_threshold=nms_threshold,
            custom_labels=custom_labels,
            mode=mode,
            enable_slicing=enable_slicing,
        )
    except Exception as exc:
        logger.exception("Inference failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model inference failed: {str(exc)}",
        ) from exc

    # Count objects by class (sorted by count descending, then class name)
    class_counts = Counter(d.label for d in detections)
    sorted_counts = dict(sorted(class_counts.items(), key=lambda item: (-item[1], item[0])))

    # Generate annotated image
    annotated = annotate_image(image, detections)
    annotated_b64 = image_to_base64(annotated, format="JPEG")

    # Serialize detections
    detection_items = [d.to_dict() for d in detections]

    return DetectResponse(
        total=len(detections),
        counts=sorted_counts,
        detections=detection_items,
        annotated_image=annotated_b64,
    )

