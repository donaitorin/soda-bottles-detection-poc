import numpy as np

from src.prelabel.filters import below_height_floor, box_iou, filter_boxes, nms, wedge_mask
from src.prelabel.merge import merge_prefer_first

CFG = {"max_area_fraction": 0.20, "max_width_fraction": 0.60, "nms_iou": 0.7, "contained_fraction": 0.90,
       "min_height_fraction": 0.015,
       "wedge": {"value": 128, "tolerance": 2, "min_component_fraction": 0.002, "drop_if_overlap_over": 0.5}}
W, H = 1000, 800


def _boxes(*rows):
    return np.array(rows, dtype=float)


def _filter(boxes, scores=None, mask=None):
    boxes = _boxes(*boxes)
    scores = np.array(scores if scores is not None else [0.9] * len(boxes))
    return filter_boxes(boxes, scores, (W, H), mask, CFG)


def test_normal_bottles_pass_untouched():
    b, s, drops = _filter([[100, 100, 160, 300], [200, 100, 260, 300]])
    assert len(b) == 2 and sum(drops.values()) == 0


def test_oversized_and_wide_boxes_are_dropped():
    b, _, drops = _filter([[0, 0, 900, 700], [100, 100, 760, 160], [100, 100, 160, 300]])
    assert len(b) == 1 and drops["oversized"] == 2


def test_height_floor_is_relative_to_the_image_height():
    # H = 800 -> floor 12 px; the same 15 px box is below the floor on a 1200 px tall image (18 px).
    boxes = _boxes([100, 100, 130, 111], [200, 100, 260, 300])  # 11 px and 200 px tall
    b, _, drops = _filter(boxes)
    assert len(b) == 1 and drops["tiny"] == 1
    assert below_height_floor(_boxes([0, 0, 30, 15]), 1200, 0.015).tolist() == [True]
    assert below_height_floor(_boxes([0, 0, 30, 15]), 800, 0.015).tolist() == [False]
    assert below_height_floor(_boxes([0, 0, 30, 12]), 800, 0.015).tolist() == [False]  # exactly at the floor is kept


def test_duplicates_are_reduced_by_nms_and_containment():
    boxes = [[100, 100, 160, 300], [102, 101, 162, 301], [110, 110, 150, 290]]
    b, s, drops = _filter(boxes, [0.9, 0.8, 0.7])
    assert len(b) == 1 and drops["duplicate_nms"] == 1 and drops["duplicate_contained"] == 1
    assert s[0] == 0.9  # the highest score survives


def test_nms_keeps_the_best_and_separate_boxes():
    boxes = _boxes([0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60])
    assert list(nms(boxes, np.array([0.5, 0.9, 0.7]), 0.5)) == [1, 2]


def _image_with_wedges():
    img = np.full((H, W, 3), 60, dtype=np.uint8)
    img[:, :100] = 128   # left grey band touching the border
    img[:, -100:] = 128  # right grey band
    img[300:340, 500:540] = 128  # a grey object in the middle: must NOT be masked
    return img


def test_wedge_mask_only_marks_border_regions():
    mask = wedge_mask(_image_with_wedges(), 128, 2, 0.002)
    assert mask[:, :100].all() and mask[:, -100:].all()
    assert not mask[300:340, 500:540].any()
    assert 0.19 < mask.mean() < 0.21


def test_wedge_boxes_are_dropped_or_clipped():
    mask = wedge_mask(_image_with_wedges(), 128, 2, 0.002)
    b, _, drops = _filter([[10, 100, 80, 300],      # fully inside the left wedge -> dropped
                           [60, 100, 160, 300],     # 40% inside the wedge -> clipped to x >= 100
                           [400, 100, 460, 300]],   # clear of the wedge -> untouched
                          mask=mask)
    assert drops["wedge_dropped"] == 1 and drops["wedge_clipped"] == 1 and len(b) == 2
    assert b[0][0] == 100 and b[0][2] == 160
    assert list(b[1]) == [400, 100, 460, 300]


def test_merge_prefers_the_first_model_on_overlaps():
    a = _boxes([100, 100, 160, 300])
    b = _boxes([102, 100, 162, 300], [400, 100, 460, 300])
    boxes, scores, source = merge_prefer_first(a, np.array([0.9]), b, np.array([0.8, 0.6]), 0.55)
    assert len(boxes) == 2 and list(source) == [0, 1] and list(boxes[0]) == [100, 100, 160, 300]


def test_iou_of_identical_boxes_is_one():
    b = _boxes([0, 0, 10, 10])
    assert abs(box_iou(b, b)[0, 0] - 1) < 1e-9
