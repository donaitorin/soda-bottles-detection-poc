import numpy as np

from src.evaluation.errors import categorize

H = 1000  # synthetic photo height


def _run(gt, boxes, cans=()):
    return categorize(np.array(gt, float).reshape(-1, 4), np.array(boxes, float).reshape(-1, 4),
                      np.ones(len(boxes)), np.array(cans, float).reshape(-1, 4), H)


BIG = [100, 100, 200, 400]


def test_matched_boxes_have_no_disagreement():
    r = _run([BIG], [BIG])
    assert r["extra"] == [] and r["missed"] == [] and r["tp"] == 1


def test_extra_box_on_a_can_is_a_can():
    r = _run([], [[500, 100, 560, 250]], cans=[[502, 102, 558, 248]])
    assert r["extra"] == ["can"]


def test_small_piece_inside_a_pseudo_label_is_a_fragment():
    r = _run([BIG], [BIG, [110, 110, 150, 200]])
    assert r["extra"] == ["fragment"]


def test_same_object_with_a_different_extent_is_partial_overlap():
    r = _run([BIG], [[150, 100, 290, 400]])   # IoU about 0.2 with the label: extra and missed at once
    assert r["extra"] == ["partial overlap"] and r["missed"] == ["partial overlap"]


def test_small_and_isolated_boxes_are_small_or_other():
    assert _run([], [[700, 100, 720, 120]]).get("extra") == ["small bottle"]   # 20 px < 3% of 1000
    assert _run([], [[700, 100, 760, 300]]).get("extra") == ["other"]


def test_missed_pseudo_labels_are_classified_as_label_errors_or_misses():
    on_can = _run([[500, 100, 560, 250]], [], cans=[[500, 100, 560, 250]])
    assert on_can["missed"] == ["can"]
    frag = _run([BIG, [110, 110, 150, 200]], [BIG])
    assert frag["missed"] == ["fragment"]
    assert _run([BIG], [])["missed"] == ["other"]


def test_imitated_errors_are_counted_when_the_model_copies_them():
    frag = [110, 110, 150, 200]
    r = _run([BIG, frag], [BIG, frag])
    assert r["imitated"] == {"can": 0, "fragment": 1} and r["extra"] == [] and r["missed"] == []
