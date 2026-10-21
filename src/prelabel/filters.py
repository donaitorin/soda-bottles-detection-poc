"""Filters for pre-labeler boxes. Boxes are xyxy in original-image pixels (numpy, float).

Every filter returns what it dropped by reason, so nothing disappears silently.
"""
from collections import Counter

import numpy as np
from scipy import ndimage


def wedge_mask(image: np.ndarray, value: int = 128, tolerance: int = 2,
               min_component_fraction: float = 0.002) -> np.ndarray:
    """Boolean mask (H, W) of the grey padding the photos have at their sides.

    The padding is a flat RGB (128, 128, 128). A pixel is a candidate if all three channels are within
    `tolerance` of `value` (JPEG compression can move them by a level or two). Only candidate regions that
    TOUCH THE IMAGE BORDER and are not tiny count, so a grey object in the middle of a shelf is not masked.
    """
    candidate = np.all(np.abs(image.astype(int) - value) <= tolerance, axis=2)
    labels, n = ndimage.label(candidate)
    if n == 0:
        return np.zeros(candidate.shape, dtype=bool)
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    sizes = ndimage.sum(candidate, labels, index=np.arange(1, n + 1))
    keep = [i for i in border if i != 0 and sizes[i - 1] >= min_component_fraction * candidate.size]
    return np.isin(labels, keep)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix (len(a), len(b)) for xyxy boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-12)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Indices kept by greedy non-maximum suppression (highest score first)."""
    order = np.argsort(-scores)
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        ious = box_iou(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_threshold]
    return np.array(keep, dtype=int)


def below_height_floor(boxes: np.ndarray, image_height: float, min_height_fraction: float) -> np.ndarray:
    """Boolean array: True where the box is shorter than `min_height_fraction` of the image height."""
    return (boxes[:, 3] - boxes[:, 1]) < min_height_fraction * image_height


def filter_boxes(boxes: np.ndarray, scores: np.ndarray, image_size: tuple[int, int], mask: np.ndarray | None,
                 cfg: dict) -> tuple[np.ndarray, np.ndarray, Counter]:
    """Apply, in order: oversized, grey wedge (drop or clip), tiny, duplicates.
    image_size is (width, height). Returns (boxes, scores, drops by reason)."""
    width, height = image_size
    drops: Counter = Counter()
    boxes, scores = boxes.copy(), scores.copy()

    # 1. Whole-shelf boxes: far too big or far too wide to be one bottle.
    w, h = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    big = (w * h > cfg["max_area_fraction"] * width * height) | (w > cfg["max_width_fraction"] * width)
    drops["oversized"] += int(big.sum())
    boxes, scores = boxes[~big], scores[~big]

    # 2. Grey wedges: drop a box mostly inside, clip the others to their non-grey part.
    if mask is not None and len(boxes):
        keep, clipped = [], boxes.copy()
        for k, (x1, y1, x2, y2) in enumerate(boxes):
            xi1, yi1 = int(np.floor(max(x1, 0))), int(np.floor(max(y1, 0)))
            xi2, yi2 = int(np.ceil(min(x2, width))), int(np.ceil(min(y2, height)))
            crop = mask[yi1:yi2, xi1:xi2]
            if crop.size == 0 or crop.mean() > cfg["wedge"]["drop_if_overlap_over"]:
                drops["wedge_dropped"] += 1
                continue
            if crop.any():  # clip to the bounding rectangle of the non-grey pixels
                rows, cols = np.where(~crop)
                clipped[k] = [xi1 + cols.min(), yi1 + rows.min(), xi1 + cols.max() + 1, yi1 + rows.max() + 1]
                drops["wedge_clipped"] += 1  # counted, but the box is kept
            keep.append(k)
        boxes, scores = clipped[keep], scores[keep]

    # 3. Tiny boxes: the owner's rule is a RELATIVE floor, a share of the image height, so a photo of any
    # resolution is treated the same (the labeling guide, the Stage 3 export and the Stage 4/5 evaluation
    # use the same function).
    tiny = below_height_floor(boxes, height, cfg["min_height_fraction"])
    drops["tiny"] += int(tiny.sum())
    boxes, scores = boxes[~tiny], scores[~tiny]

    # 4. Duplicates: NMS, then boxes mostly contained in a larger, higher-score box.
    if len(boxes):
        keep = nms(boxes, scores, cfg["nms_iou"])
        drops["duplicate_nms"] += len(boxes) - len(keep)
        boxes, scores = boxes[keep], scores[keep]
    if len(boxes) > 1:
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        lt = np.maximum(boxes[:, None, :2], boxes[None, :, :2])
        rb = np.minimum(boxes[:, None, 2:], boxes[None, :, 2:])
        inter = np.clip(rb - lt, 0, None).prod(-1)
        inside = inter / (area[None, :] + 1e-12)  # inside[i, j] = share of box j covered by box i
        drop = np.zeros(len(boxes), dtype=bool)
        for j in range(len(boxes)):
            for i in range(len(boxes)):
                if i != j and area[i] > area[j] and scores[i] >= scores[j] and inside[i, j] >= cfg["contained_fraction"]:
                    drop[j] = True
                    break
        drops["duplicate_contained"] += int(drop.sum())
        boxes, scores = boxes[~drop], scores[~drop]
    return boxes, scores, drops
