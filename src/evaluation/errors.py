"""Disagreement analysis between predictions and pseudo-labels, by (heuristic) cause.

The causes are automatic proxies, checked by eye on drawings, not ground truth:
  can               the box lies on a can (a Grounding DINO "can." detection with IoU >= 0.5). Cans are out of scope.
  fragment          the box is a small piece inside a larger box (>= 80% of its area inside, and under 60% of its area)
  partial overlap   the box overlaps a box on the other side with IoU between 0.1 and 0.5: same object, different
                    extent (typical of occluded or overlapping bottles, and of loose vs tight boxes)
  small bottle      shorter than 3% of the image height
  other             nothing near it on the other side
A pseudo-label that is a can or a fragment is a pseudo-label ERROR. A prediction that matches such a pseudo-label
is an imitated error (the model copied it).
"""
import numpy as np

from src.evaluation.detection import greedy_match
from src.prelabel.filters import box_iou

CAUSES = ["can", "fragment", "partial overlap", "small bottle", "other"]


def inside_fraction(boxes: np.ndarray, containers: np.ndarray) -> np.ndarray:
    """(n, m): share of the area of boxes[i] that lies inside containers[j]."""
    if len(boxes) == 0 or len(containers) == 0:
        return np.zeros((len(boxes), len(containers)))
    lt = np.maximum(boxes[:, None, :2], containers[None, :, :2])
    rb = np.minimum(boxes[:, None, 2:], containers[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area[:, None] + 1e-12)


def _is_fragment(box_area, inside_row, container_areas, frac=0.8, max_ratio=0.6) -> bool:
    return bool(np.any((inside_row >= frac) & (box_area < max_ratio * container_areas)))


def _area(b):
    return (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])


def categorize(gt: np.ndarray, boxes: np.ndarray, scores: np.ndarray, cans: np.ndarray, image_height: float,
               iou_thr: float = 0.5, small_fraction: float = 0.03) -> dict:
    """For one photo: causes of the extra predictions, of the missed pseudo-labels, and of the imitated errors.
    Returns {"extra": [cause per unmatched prediction], "missed": [...], "tp": n, "imitated": {"can": n, "fragment": n}}."""
    gt, boxes, cans = (np.asarray(a, float).reshape(-1, 4) for a in (gt, boxes, cans))
    gt_hit, pred_hit, iou = greedy_match(gt, boxes, np.asarray(scores, float), iou_thr)
    can_iou_pred = box_iou(boxes, cans) if len(boxes) and len(cans) else np.zeros((len(boxes), len(cans)))
    can_iou_gt = box_iou(gt, cans) if len(gt) and len(cans) else np.zeros((len(gt), len(cans)))
    pred_in_gt = inside_fraction(boxes, gt)
    pred_in_pred = inside_fraction(boxes, boxes)
    np.fill_diagonal(pred_in_pred, 0.0)
    gt_in_gt = inside_fraction(gt, gt)
    np.fill_diagonal(gt_in_gt, 0.0)
    gt_area, pred_area = _area(gt), _area(boxes)

    extra = []
    for i in np.where(~pred_hit)[0]:
        if len(cans) and can_iou_pred[i].max() >= iou_thr:
            cause = "can"
        elif _is_fragment(pred_area[i], pred_in_gt[i], gt_area) or _is_fragment(pred_area[i], pred_in_pred[i], pred_area):
            cause = "fragment"
        elif len(gt) and iou[i].max() >= 0.1:
            cause = "partial overlap"
        elif boxes[i, 3] - boxes[i, 1] < small_fraction * image_height:
            cause = "small bottle"
        else:
            cause = "other"
        extra.append(cause)
    missed, imitated = [], {"can": 0, "fragment": 0}
    gt_is_can = np.array([len(cans) > 0 and can_iou_gt[j].max() >= iou_thr for j in range(len(gt))], bool)
    gt_is_fragment = np.array([_is_fragment(gt_area[j], gt_in_gt[j], gt_area) for j in range(len(gt))], bool)
    for j in range(len(gt)):
        if gt_hit[j]:
            if gt_is_can[j]:
                imitated["can"] += 1
            elif gt_is_fragment[j]:
                imitated["fragment"] += 1
            continue
        if gt_is_can[j]:
            cause = "can"
        elif gt_is_fragment[j]:
            cause = "fragment"
        elif len(boxes) and iou[:, j].max() >= 0.1:
            cause = "partial overlap"
        elif gt[j, 3] - gt[j, 1] < small_fraction * image_height:
            cause = "small bottle"
        else:
            cause = "other"
        missed.append(cause)
    return {"extra": extra, "missed": missed, "tp": int(pred_hit.sum()), "imitated": imitated,
            "extra_idx": np.where(~pred_hit)[0].tolist(), "missed_idx": np.where(~gt_hit)[0].tolist(),
            "gt_hit": gt_hit, "pred_hit": pred_hit}
