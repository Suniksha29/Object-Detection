"""OWLv2 Open-Vocabulary Object Detector implementation with High-Resolution Slicing.

Uses google/owlv2-base-patch16-ensemble (Apache-2.0 license) for open-world,
text-promptable zero-shot object detection with multi-scale tiled crops.
"""

import logging
from typing import List, Optional, Tuple, Union
from PIL import Image
import torch
from transformers import Owlv2ForObjectDetection, Owlv2Processor

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


class Owlv2Detector(BaseDetector):
    """Zero-shot open-vocabulary detector using OWLv2 with optional tiled slicing."""

    def __init__(
        self,
        model_id: str = "google/owlv2-base-patch16-ensemble",
        device: Optional[str] = None,
    ):
        super().__init__(model_id=model_id, device=device)
        self.model: Optional[Owlv2ForObjectDetection] = None
        self.processor: Optional[Owlv2Processor] = None

    def load(self) -> None:
        """Load OWLv2 processor and model weights."""
        if self.is_loaded:
            logger.info("OWLv2 already loaded.")
            return

        logger.info("Loading OWLv2 processor from %s...", self.model_id)
        self.processor = Owlv2Processor.from_pretrained(self.model_id)

        logger.info("Loading OWLv2 model weights onto %s...", self.device)
        self.model = Owlv2ForObjectDetection.from_pretrained(self.model_id)
        self.model.to(self.device)
        self.model.eval()

        self.is_loaded = True
        logger.info("OWLv2 loaded successfully.")

    def _infer_single(
        self,
        img: Image.Image,
        text_queries: List[List[str]],
        threshold: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        """Run single image forward pass for OWLv2."""
        w, h = img.size
        inputs = self.processor(
            text=text_queries,
            images=img,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([[h, w]], device=self.device)
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=threshold,
            text_labels=text_queries,
        )[0]

        return (
            results["boxes"].detach().cpu(),
            results["scores"].detach().cpu(),
            results["labels"].detach().cpu(),
            results.get("text_labels", []),
        )

    def detect(
        self,
        image: Image.Image,
        threshold: float = 0.15,
        nms_threshold: float = 0.45,
        custom_labels: Optional[Union[List[str], str]] = None,
        enable_slicing: bool = True,
        **kwargs,
    ) -> List[DetectionResult]:
        """Detect open-vocabulary objects in an image using custom text queries.

        Args:
            image: PIL Image in RGB format.
            threshold: Confidence score cutoff (default: 0.15 for zero-shot).
            nms_threshold: IoU cutoff for Non-Maximum Suppression.
            custom_labels: List of target query strings or comma-separated string.
            enable_slicing: Whether to run tiled high-res slices for tiny objects.

        Returns:
            List of DetectionResult instances.
        """
        if not self.is_loaded or self.model is None or self.processor is None:
            raise RuntimeError("OWLv2 model is not loaded. Call load() first.")

        if image.mode != "RGB":
            image = image.convert("RGB")

        orig_w, orig_h = image.size

        # Parse custom labels
        if isinstance(custom_labels, str):
            labels_list = [c.strip() for c in custom_labels.split(",") if c.strip()]
        elif isinstance(custom_labels, list):
            labels_list = [c.strip() for c in custom_labels if c.strip()]
        else:
            labels_list = []

        if not labels_list:
            labels_list = [
                "mobile phone",
                "laptop charger",
                "power adapter",
                "pen",
                "laptop",
                "person",
                "backpack",
                "book",
                "keyboard",
                "mouse",
                "chair",
            ]

        text_queries = [labels_list]

        # 1. Full image inference
        all_boxes, all_scores, all_labels, text_labels = self._infer_single(
            image, text_queries, threshold
        )

        # 2. Tiled high-resolution slicing (if enabled and image is large enough)
        if enable_slicing and min(orig_w, orig_h) >= 300:
            slice_coords = _generate_slices(orig_w, orig_h)
            slice_boxes_list = []
            slice_scores_list = []
            slice_labels_list = []

            for x1, y1, x2, y2 in slice_coords:
                crop = image.crop((x1, y1, x2, y2))
                s_boxes, s_scores, s_labels, _ = self._infer_single(
                    crop, text_queries, threshold
                )

                if s_boxes.numel() > 0:
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

        # Apply per-class NMS across all detections
        keep_indices = batched_nms(
            boxes=all_boxes,
            scores=all_scores,
            labels=all_labels,
            iou_threshold=nms_threshold,
        )

        detections: List[DetectionResult] = []
        for idx in keep_indices:
            idx_int = idx.item()
            score = float(all_scores[idx_int].item())
            box_tensor = all_boxes[idx_int].tolist()

            c_id = all_labels[idx_int].item()
            label_name = labels_list[c_id] if c_id < len(labels_list) else f"object_{c_id}"

            x1 = max(0.0, min(float(box_tensor[0]), float(orig_w)))
            y1 = max(0.0, min(float(box_tensor[1]), float(orig_h)))
            x2 = max(0.0, min(float(box_tensor[2]), float(orig_w)))
            y2 = max(0.0, min(float(box_tensor[3]), float(orig_h)))

            detections.append(
                DetectionResult(
                    label=label_name,
                    confidence=score,
                    box=[x1, y1, x2, y2],
                )
            )

        return detections
