"""Abstract base detector interface and detection data structures."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
from PIL import Image
import torch


@dataclass
class DetectionResult:
    """Represents a single detected bounding box and classification."""
    label: str
    confidence: float
    box: List[float]  # [x1, y1, x2, y2] in pixel coordinates

    def to_dict(self) -> dict:
        """Convert detection result to serializable dict."""
        return {
            "label": self.label,
            "confidence": round(float(self.confidence), 4),
            "box": [round(float(c), 2) for c in self.box],
        }


class BaseDetector(ABC):
    """Abstract base class for object detection models."""

    def __init__(self, model_id: str, device: Optional[str] = None):
        self.model_id = model_id
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.is_loaded: bool = False

    @abstractmethod
    def load(self) -> None:
        """Load model weights and processor into memory/device."""
        pass

    @abstractmethod
    def detect(
        self,
        image: Image.Image,
        threshold: float = 0.5,
        nms_threshold: float = 0.5,
        **kwargs,
    ) -> List[DetectionResult]:
        """Run object detection on an image.

        Args:
            image: PIL Image in RGB format.
            threshold: Confidence score cutoff (0.0 - 1.0).
            nms_threshold: Intersection-over-Union threshold for NMS.
            **kwargs: Extra detector-specific options (e.g. custom_labels).

        Returns:
            List of DetectionResult instances.
        """
        pass

