"""Detector Manager for managing model lifecycles and inference dispatch."""

import logging
from typing import Dict, List, Optional
from PIL import Image
import torch

from models.base import BaseDetector, DetectionResult
from models.rtdetr_detector import RTDetrDetector
from models.owlv2_detector import Owlv2Detector

logger = logging.getLogger(__name__)


class DetectorManager:
    """Manages object detector instances, device selection, and inference dispatch."""

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.standard_detector = RTDetrDetector(device=self.device)
        self.open_vocab_detector = Owlv2Detector(device=self.device)

    def load_models(self) -> None:
        """Load detector models into memory at application startup."""
        logger.info("Initializing detectors on device: %s", self.device)
        self.standard_detector.load()
        # Pre-load OWLv2 for fast zero-shot inference
        self.open_vocab_detector.load()
        logger.info("All detector models loaded successfully.")

    def detect(
        self,
        image: Image.Image,
        threshold: float = 0.5,
        nms_threshold: float = 0.5,
        custom_labels: Optional[str] = None,
        mode: str = "standard",
        enable_slicing: bool = True,
    ) -> List[DetectionResult]:
        """Dispatch detection request to appropriate model.

        Args:
            image: PIL Image object.
            threshold: Confidence threshold (0.0 to 1.0).
            nms_threshold: IoU NMS threshold (0.0 to 1.0).
            custom_labels: Optional custom label string (e.g. 'helmet, laptop').
            mode: Detection mode ('standard' or 'open_vocab').
            enable_slicing: Enable multi-scale tiled slicing for tiny objects.

        Returns:
            List of DetectionResult instances.
        """
        # If custom labels are provided or mode is open_vocab, use OWLv2
        if (custom_labels and custom_labels.strip()) or mode.lower() in ("open_vocab", "open_vocabulary", "owlv2"):
            logger.info("Using OWLv2 open-vocabulary detector with custom labels: %s", custom_labels)
            return self.open_vocab_detector.detect(
                image=image,
                threshold=threshold,
                nms_threshold=nms_threshold,
                custom_labels=custom_labels,
                enable_slicing=enable_slicing,
            )

        # Default standard mode uses RT-DETR
        return self.standard_detector.detect(
            image=image,
            threshold=threshold,
            nms_threshold=nms_threshold,
            enable_slicing=enable_slicing,
        )

    def get_status(self) -> Dict[str, str]:
        """Return status information about loaded models."""
        return {
            "device": str(self.device),
            "standard_model": self.standard_detector.model_id,
            "standard_loaded": str(self.standard_detector.is_loaded),
            "open_vocab_model": self.open_vocab_detector.model_id,
            "open_vocab_loaded": str(self.open_vocab_detector.is_loaded),
        }
