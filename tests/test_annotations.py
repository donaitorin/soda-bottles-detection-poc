import numpy as np
import pytest

from src.annotations.export import (OWNER, PRELABEL, build_coco, clamp_xyxy, pick_annotation,
                                    read_owner_annotations, xyxy_to_coco_bbox)
from src.annotations.sanity import check_coco
from src.prelabel.labelstudio import xyxy_to_ls_result

W, H = 1000, 800  # synthetic photo; ids below are synthetic too
PRED = {"s01_aaaaaaaa": {"width": W, "height": H, "boxes": [[100, 100, 160, 300], [200, 100, 260, 300], [300, 100, 330, 108]],
                         "scores": [0.9, 0.8, 0.7]},   # third box is 8 px tall: under 1.5% of 800 (12 px)
        "s01_bbbbbbbb": {"width": W, "height": H, "boxes": [[10, 10, 70, 210]], "scores": [0.6]}}
FILES = {k: f"{k}.jpg" for k in PRED}
STORES = {k: "s01" for k in PRED}


def _coco(owner=None):
    return build_coco(sorted(PRED), PRED, owner or {}, FILES, STORES, "train", 0.015, "grounding_dino@x+cfgy")


def test_coco_bbox_is_x_y_width_height_in_pixels():
    assert xyxy_to_coco_bbox([100, 50, 160, 300]) == [100, 50, 60, 250]


def test_clamp_keeps_boxes_inside_the_image():
    assert clamp_xyxy([-3, -2, W + 4, H + 1], W, H) == [0, 0, W, H]


def test_prelabels_are_marked_unverified_and_the_floor_is_applied_and_counted():
    coco, stats = _coco()
    assert {im["label_source"] for im in coco["images"]} == {PRELABEL}
    assert len(coco["annotations"]) == 3 and stats["dropped_below_height_floor_prelabel_unverified"] == 1
    assert "agreement with Grounding DINO" in coco["info"]["disclaimer"]
    assert all(im["file_name"] == im["photo_id"] + ".jpg" for im in coco["images"])


def test_owner_annotation_wins_over_the_prelabel():
    owner = {"s01_aaaaaaaa": {"boxes": [[0, 0, 50, 50]], "products": ["sprite"], "lead_time": 42.0, "updated_at": "t"}}
    coco, stats = _coco(owner)
    first = coco["images"][0]
    assert first["label_source"] == OWNER and first["products"] == ["sprite"] and first["lead_time_s"] == 42.0
    assert [a["bbox"] for a in coco["annotations"] if a["image_id"] == first["id"]] == [[0.0, 0.0, 50.0, 50.0]]
    assert coco["images"][1]["label_source"] == PRELABEL and stats["photos_owner_corrected"] == 1


def test_owner_annotation_with_zero_boxes_means_no_bottles():
    owner = {"s01_aaaaaaaa": {"boxes": [], "products": [], "lead_time": 1.0, "updated_at": "t"}}
    coco, _ = _coco(owner)
    assert not [a for a in coco["annotations"] if a["image_id"] == 1]


def _task(pid, boxes, choices, **ann):
    result = [xyxy_to_ls_result(b, W, H, 1.0, "bottle", "image", "bottle") for b in boxes]
    result.append({"from_name": "products", "to_name": "image", "type": "choices", "value": {"choices": choices}})
    return {"data": {"photo_id": pid}, "annotations": [{"id": 1, "result": result, "lead_time": 12.5,
                                                         "updated_at": "2026-01-02T00:00:00Z", **ann}]}


def test_read_owner_annotations_converts_percent_back_to_pixels():
    sizes = {k: (W, H) for k in PRED}
    owner, notes = read_owner_annotations([_task("s01_aaaaaaaa", [[100, 100, 160, 300]], ["sprite"]),
                                           {"data": {"photo_id": "s01_bbbbbbbb"}, "annotations": []},
                                           {"data": {"photo_id": "s99_zzzzzzzz"}, "annotations": []}],
                                          "bottle", "products", sizes)
    assert list(owner) == ["s01_aaaaaaaa"] and owner["s01_aaaaaaaa"]["products"] == ["sprite"]
    assert np.allclose(owner["s01_aaaaaaaa"]["boxes"][0], [100, 100, 160, 300])
    assert notes["task_without_submitted_annotation"] == 1 and notes["task_with_unknown_photo_id"] == 1


def test_skipped_annotations_are_ignored_and_the_latest_is_used():
    task = {"annotations": [{"id": 1, "was_cancelled": True, "updated_at": "z"},
                            {"id": 2, "was_cancelled": False, "updated_at": "a"},
                            {"id": 3, "was_cancelled": False, "updated_at": "b"}]}
    assert pick_annotation(task)["id"] == 3
    assert pick_annotation({"annotations": [{"id": 1, "was_cancelled": True}]}) is None


def test_annotation_with_a_different_image_size_is_rejected():
    with pytest.raises(ValueError):
        read_owner_annotations([_task("s01_aaaaaaaa", [[1, 1, 50, 50]], [])], "bottle", "products", {"s01_aaaaaaaa": (W + 1, H)})


ARGS = dict(min_height_fraction=0.015, duplicate_iou=0.9, boxes_per_photo=(1, 150), id_regex=r"^s\d{2}_[0-9a-f]{8}(\.[A-Za-z]+)?$")


def test_sanity_passes_on_a_clean_export():
    coco, _ = _coco()
    errors, warnings = check_coco(coco, sorted(PRED), **ARGS)
    assert errors == [] and warnings == []


def test_sanity_catches_the_listed_problems():
    coco, _ = _coco()
    bad = {**coco, "annotations": [dict(a) for a in coco["annotations"]]}
    bad["annotations"][0]["bbox"] = [950, 100, 100, 100]                       # outside the image
    bad["annotations"][1]["bbox"] = [200, 100, 0, 50]; bad["annotations"][1]["area"] = 0   # area zero
    bad["annotations"][2]["bbox"] = [300, 100, 30, 8]                          # under the height floor
    bad["annotations"].append({**bad["annotations"][0], "id": 99, "bbox": [950, 100, 100, 100]})  # duplicate
    errors, _ = check_coco(bad, sorted(PRED), **ARGS)
    text = " | ".join(errors)
    for expected in ("outside the image", "area zero", "height floor", "duplicate box pair"):
        assert expected in text


def test_sanity_catches_wrong_photo_list_and_real_looking_file_names():
    coco, _ = _coco()
    errors, _ = check_coco(coco, ["s01_aaaaaaaa"], **ARGS)
    assert any("split list" in e for e in errors)
    coco["images"][0]["file_name"] = "some real name.jpg"
    errors, _ = check_coco(coco, sorted(PRED), **ARGS)
    assert any("not a photo id" in e for e in errors)


def test_sanity_warns_about_photos_without_boxes():
    coco, _ = _coco()
    coco["annotations"] = [a for a in coco["annotations"] if a["image_id"] != 2]
    _, warnings = check_coco(coco, sorted(PRED), **ARGS)
    assert len(warnings) == 1
