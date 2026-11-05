"""YOLOS Object Detector implementation with High-Resolution Slicing (SAHI).

Uses hustvl/yolos-tiny (Apache-2.0 license) trained on COCO dataset (80 classes).
Ultra-lightweight architecture (~28MB weights, <80MB RAM peak) perfectly suited
for memory-constrained cloud environments (Render Free Tier 512MB RAM) and rapid inference.
"""

import gc
import logging
from typing import List, Optional, Tuple
from PIL import Image
import torch
from transformers import YolosForObjectDetection

try:
    from transformers.models.yolos.image_processing_pil_yolos import (
        YolosImageProcessorPil as YolosImageProcessor,
    )
except ImportError:
    from transformers import YolosImageProcessor

from models.base import BaseDetector, DetectionResult
from models.nms import batched_nms

logger = logging.getLogger(__name__)


def _generate_slices(
    img_w: int,
    img_h: int,
    slice_w_ratio: float = 0.60,
    slice_h_ratio: float = 0.60,
) -> List[Tuple[int, int, int, int]]:
    """Generate 2x2 overlapping crop coordinates [x1, y1, x2, y2]."""
    sw = int(img_w * slice_w_ratio)
    sh = int(img_h * slice_h_ratio)
    slices = [
        (0, 0, sw, sh),                               # Top-left
        (img_w - sw, 0, img_w, sh),                   # Top-right
        (0, img_h - sh, sw, img_h),                   # Bottom-left
        (img_w - sw, img_h - sh, img_w, img_h),       # Bottom-right
    ]
    return slices


class YolosDetector(BaseDetector):
    """Detector using pretrained YOLOS-tiny with optional sliced inference."""

    def __init__(
        self,
        model_id: str = "hustvl/yolos-tiny",
        device: Optional[str] = None,
    ):
        super().__init__(model_id=model_id, device=device)
        self.model: Optional[YolosForObjectDetection] = None
        self.processor: Optional[YolosImageProcessor] = None

    def load(self) -> None:
        """Load YOLOS model and image processor onto the target device."""
        if self.is_loaded:
            logger.info("YOLOS-tiny already loaded.")
            return

        logger.info("Loading YOLOS processor from %s...", self.model_id)
        self.processor = YolosImageProcessor.from_pretrained(self.model_id)

        logger.info("Loading YOLOS model weights onto %s...", self.device)
        self.model = YolosForObjectDetection.from_pretrained(self.model_id)
        self.model.to(self.device)
        self.model.eval()

        gc.collect()
        self.is_loaded = True
        logger.info("YOLOS-tiny loaded successfully.")

    def _infer_single(
        self,
        img: Image.Image,
        threshold: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run single image forward pass and return raw boxes, scores, and labels."""
        w, h = img.size
        inputs = self.processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([[h, w]], device=self.device)
        results = self.processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=threshold,
        )[0]

        return (
            results["boxes"].detach().cpu(),
            results["scores"].detach().cpu(),
            results["labels"].detach().cpu(),
        )

    def detect(
        self,
        image: Image.Image,
        threshold: float = 0.5,
        nms_threshold: float = 0.5,
        enable_slicing: bool = False,
        **kwargs,
    ) -> List[DetectionResult]:
        """Detect objects in a PIL Image using YOLOS-tiny.

        Args:
            image: PIL Image in RGB mode.
            threshold: Minimum score threshold (default: 0.5).
            nms_threshold: Non-Maximum Suppression IoU threshold (default: 0.5).
            enable_slicing: Enable multi-scale high-res tiled slicing for small objects.

        Returns:
            List of filtered and deduplicated DetectionResult objects.
        """
        if not self.is_loaded or self.model is None or self.processor is None:
            raise RuntimeError("YOLOS model is not loaded. Call load() first.")

        if image.mode != "RGB":
            image = image.convert("RGB")

        orig_w, orig_h = image.size

        # 1. Full image inference
        all_boxes, all_scores, all_labels = self._infer_single(image, threshold)

        # 2. Tiled high-resolution slicing (if enabled and image is large enough)
        if enable_slicing and min(orig_w, orig_h) >= 300:
            slice_coords = _generate_slices(orig_w, orig_h)
            slice_boxes_list = []
            slice_scores_list = []
            slice_labels_list = []

            for x1, y1, x2, y2 in slice_coords:
                crop = image.crop((x1, y1, x2, y2))
                s_boxes, s_scores, s_labels = self._infer_single(crop, threshold)

                if s_boxes.numel() > 0:
                    # Translate coordinates back to full image space
                    s_boxes[:, [0, 2]] += x1
                    s_boxes[:, [1, 3]] += y1
                    slice_boxes_list.append(s_boxes)
                    slice_scores_list.append(s_scores)
                    slice_labels_list.append(s_labels)

            if slice_boxes_list:
                all_boxes = torch.cat([all_boxes] + slice_boxes_list, dim=0)
                all_scores = torch.cat([all_scores] + slice_scores_list, dim=0)
                all_labels = torch.cat([all_labels] + slice_labels_list, dim=0)

        if all_scores.numel() == 0:
            return []

        # Apply batched Non-Maximum Suppression across all full and sliced detections
        keep_indices = batched_nms(
            boxes=all_boxes,
            scores=all_scores,
            labels=all_labels,
            iou_threshold=nms_threshold,
        )

        detections: List[DetectionResult] = []
        for idx in keep_indices:
            idx_int = idx.item()
            lbl_id = all_labels[idx_int].item()
            score = float(all_scores[idx_int].item())
            box_tensor = all_boxes[idx_int].tolist()

            x1 = max(0.0, min(float(box_tensor[0]), float(orig_w)))
            y1 = max(0.0, min(float(box_tensor[1]), float(orig_h)))
            x2 = max(0.0, min(float(box_tensor[2]), float(orig_w)))
            y2 = max(0.0, min(float(box_tensor[3]), float(orig_h)))

            label_name = self.model.config.id2label.get(lbl_id, f"class_{lbl_id}")

            detections.append(
                DetectionResult(
                    label=label_name,
                    confidence=score,
                    box=[x1, y1, x2, y2],
                )
            )

        return detections
