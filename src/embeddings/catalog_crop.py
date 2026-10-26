"""Choose the bottle box in a catalog product photo (each photo holds ONE bottle on a plain background)."""
import numpy as np


def box_area(b) -> float:
    return float((b[2] - b[0]) * (b[3] - b[1]))


def enclosing_box(boxes: np.ndarray) -> np.ndarray:
    return np.array([boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max()], float)


def choose_box(boxes: np.ndarray, image_size: tuple[int, int], several: str = "largest", override: str | None = None) -> tuple[np.ndarray, str]:
    """Returns (box xyxy, rule). Rules: `single` (one box), `largest` (several boxes: almost surely fragments of the same
    bottle, the largest is used), `enclosing` (the box around all of them; only by an explicit override after looking at the
    photo) and `whole_photo` (no box). The caller lists every photo whose rule is not `single`."""
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    width, height = image_size
    if len(boxes) == 0:
        return np.array([0, 0, width, height], float), "whole_photo"
    if len(boxes) == 1:
        return boxes[0], "single"
    rule = override or several
    if rule == "enclosing":
        return enclosing_box(boxes), "enclosing"
    return boxes[int(np.argmax([box_area(b) for b in boxes]))], "largest"
