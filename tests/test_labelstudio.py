import copy
import json
from pathlib import Path

import numpy as np
import yaml

from src.prelabel.labelstudio import (build_task, config_hash, ls_result_to_xyxy, model_version,
                                      xyxy_to_ls_result)

CFG = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/prelabel.yaml").read_text())
ARGS = ("bottle", "image", "bottle")


def convert(box, w, h, score=0.9):
    return xyxy_to_ls_result(box, w, h, score, *ARGS)


def test_known_box_on_a_portrait_photo():
    r = convert((118.4, 288.0, 236.8, 864.0), 1184, 1440)  # 1184 wide, 1440 tall
    v = r["value"]
    assert (v["x"], v["y"], v["width"], v["height"]) == (10.0, 20.0, 10.0, 40.0)  # percent of width / height
    assert r["original_width"] == 1184 and r["original_height"] == 1440
    assert r["type"] == "rectanglelabels" and v["rectanglelabels"] == ["bottle"] and v["rotation"] == 0


def test_landscape_photo_does_not_mix_width_and_height():
    # A box twice as wide as tall, on a landscape photo: if width/height were swapped this would fail.
    v = convert((100, 50, 500, 250), 1000, 500)["value"]
    assert (v["x"], v["y"], v["width"], v["height"]) == (10.0, 10.0, 40.0, 40.0)


def test_round_trip_on_small_and_large_photos():
    rng = np.random.default_rng(0)
    for w, h in ((1184, 1440), (2467, 3000), (4096, 3368), (1579, 1080)):
        for _ in range(50):
            x1, y1 = rng.uniform(0, w - 30), rng.uniform(0, h - 30)
            box = (x1, y1, x1 + rng.uniform(5, w - x1), y1 + rng.uniform(5, h - y1))
            back = ls_result_to_xyxy(convert(box, w, h))
            assert np.allclose(back, box, atol=1e-6)


def test_boxes_are_clamped_to_the_image():
    v = convert((-4, -2, 1010, 505), 1000, 500)["value"]
    assert v["x"] == 0 and v["y"] == 0 and v["width"] == 100 and v["height"] == 100


def test_width_and_height_are_never_negative_and_stay_inside():
    v = convert((990, 495, 1030, 520), 1000, 500)["value"]
    assert v["width"] >= 0 and v["height"] >= 0 and v["x"] + v["width"] <= 100 + 1e-9 and v["y"] + v["height"] <= 100 + 1e-9


def test_config_hash_follows_the_frozen_settings_only():
    base = config_hash(CFG)
    assert base == config_hash(copy.deepcopy(CFG))
    changed = copy.deepcopy(CFG)
    changed["dino"]["box_threshold"] = 0.15
    assert config_hash(changed) != base
    unrelated = copy.deepcopy(CFG)
    unrelated["compare"]["panel_height_px"] = 1
    unrelated["frcnn"]["box_score_thresh"] = 0.9   # Faster R-CNN settings are not part of the frozen Grounding DINO run
    assert config_hash(unrelated) == base
    assert model_version(CFG).startswith("grounding_dino@") and "+cfg" + base in model_version(CFG)


def test_task_has_prediction_with_ids_only():
    boxes = np.array([[10.0, 20.0, 110.0, 220.0], [300.0, 40.0, 360.0, 400.0]])
    task = build_task("s01_abcdef12", "train", "s01", "/data/local-files/?d=labelstudio/images/s01_abcdef12.jpg",
                      boxes, np.array([0.8, 0.4]), 1184, 1440, CFG)
    assert task["data"] == {"image": "/data/local-files/?d=labelstudio/images/s01_abcdef12.jpg",
                            "photo_id": "s01_abcdef12", "split": "train", "store_code": "s01"}
    pred = task["predictions"][0]
    assert pred["model_version"] == model_version(CFG) and len(pred["result"]) == 2
    assert pred["result"][0]["id"] == "s01_abcdef12_0" and abs(pred["score"] - 0.6) < 1e-9
    json.dumps(task)  # serializable


def test_task_without_boxes_is_valid():
    task = build_task("s01_abcdef12", "val", "s01", "/x.jpg", np.zeros((0, 4)), np.zeros(0), 100, 100, CFG)
    assert task["predictions"][0]["result"] == []
