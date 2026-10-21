"""Merge the boxes of two pre-labelers."""
import numpy as np

from src.prelabel.filters import box_iou, nms


def merge_prefer_first(boxes_a, scores_a, boxes_b, scores_b, iou: float):
    """Keep every box of A. Add the boxes of B that do not overlap a kept box above `iou`
    (B boxes overlapping each other are reduced with NMS first). Returns boxes, scores, source
    (0 = A, 1 = B). Scores of the two models are not comparable, so A simply wins on overlaps."""
    if len(boxes_b):
        keep_b = nms(boxes_b, scores_b, iou)
        boxes_b, scores_b = boxes_b[keep_b], scores_b[keep_b]
    if len(boxes_a) and len(boxes_b):
        free = box_iou(boxes_b, boxes_a).max(axis=1) <= iou
        boxes_b, scores_b = boxes_b[free], scores_b[free]
    boxes = np.concatenate([boxes_a.reshape(-1, 4), boxes_b.reshape(-1, 4)])
    scores = np.concatenate([scores_a, scores_b])
    source = np.concatenate([np.zeros(len(boxes_a), int), np.ones(len(boxes_b), int)])
    return boxes, scores, source
