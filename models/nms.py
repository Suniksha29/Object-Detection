"""Non-Maximum Suppression (NMS) module.

Provides a fast, pure PyTorch implementation of standard greedy NMS
without requiring external binary dependencies such as torchvision.
"""

from typing import List, Tuple
import torch


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute pairwise Intersection over Union (IoU) between two sets of boxes.

    Args:
        boxes1: Tensor of shape (N, 4) in [x1, y1, x2, y2] format.
        boxes2: Tensor of shape (M, 4) in [x1, y1, x2, y2] format.

    Returns:
        Tensor of shape (N, M) containing IoU values in [0, 1].
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=torch.float32, device=boxes1.device)

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    # Intersections
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # (N, M, 2)

    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    inter = wh[:, :, 0] * wh[:, :, 1]  # (N, M)

    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union.clamp(min=1e-6)
    return iou


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5) -> torch.Tensor:
    """Standard Greedy Non-Maximum Suppression (NMS).

    Iteratively selects the detection with the highest confidence and removes
    all remaining overlapping detections whose IoU exceeds `iou_threshold`.

    Args:
        boxes: Tensor of shape (N, 4) in [x1, y1, x2, y2] format.
        scores: Tensor of shape (N,) with confidence scores.
        iou_threshold: Overlap threshold for suppressing redundant boxes.

    Returns:
        1D Tensor of indices of kept boxes, sorted in descending order of score.
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    order = scores.argsort(descending=True)
    keep: List[int] = []

    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)

        if order.numel() == 1:
            break

        xx1 = torch.maximum(x1[i], x1[order[1:]])
        yy1 = torch.maximum(y1[i], y1[order[1:]])
        xx2 = torch.minimum(x2[i], x2[order[1:]])
        yy2 = torch.minimum(y2[i], y2[order[1:]])

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h

        union = areas[i] + areas[order[1:]] - inter
        iou = inter / union.clamp(min=1e-6)

        remaining_indices = torch.where(iou <= iou_threshold)[0]
        order = order[remaining_indices + 1]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def batched_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float = 0.5,
) -> torch.Tensor:
    """Performs per-class Non-Maximum Suppression.

    Displaces boxes by an offset proportional to their class ID so boxes of
    different classes do not suppress one another.

    Args:
        boxes: Tensor of shape (N, 4) in [x1, y1, x2, y2] format.
        scores: Tensor of shape (N,) with confidence scores.
        labels: Tensor of shape (N,) with integer class labels.
        iou_threshold: IoU overlap threshold.

    Returns:
        1D Tensor of indices of kept detections.
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    # Strategy: add a large coordinate offset based on label
    max_coordinate = boxes.max() if boxes.numel() > 0 else 0.0
    offsets = labels.to(boxes.dtype) * (max_coordinate + 1.0)
    boxes_for_nms = boxes + offsets[:, None]
    return nms(boxes_for_nms, scores, iou_threshold)

