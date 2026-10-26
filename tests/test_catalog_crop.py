import numpy as np

from src.embeddings.catalog_crop import choose_box

SIZE = (400, 800)


def test_no_box_uses_the_whole_photo():
    box, rule = choose_box(np.zeros((0, 4)), SIZE)
    assert rule == "whole_photo" and box.tolist() == [0, 0, 400, 800]


def test_a_single_box_is_used_as_it_is():
    box, rule = choose_box(np.array([[50, 40, 350, 760]]), SIZE)
    assert rule == "single" and box.tolist() == [50, 40, 350, 760]


def test_several_boxes_default_to_the_largest():
    boxes = np.array([[50, 40, 350, 760], [100, 40, 200, 100], [120, 300, 300, 500]])
    box, rule = choose_box(boxes, SIZE)
    assert rule == "largest" and box.tolist() == [50, 40, 350, 760]


def test_an_explicit_override_uses_the_enclosing_box():
    boxes = np.array([[100, 200, 300, 760], [120, 40, 280, 210]])   # body and cap area
    box, rule = choose_box(boxes, SIZE, override="enclosing")
    assert rule == "enclosing" and box.tolist() == [100, 40, 300, 760]
