import json

import numpy as np
import pytest
from PIL import Image

from src.embeddings.demo import build_json, check_not_test, content_index, decision_colour, draw_result, file_sha256, panel_entry

COLOURS = {"automatic": [0, 170, 0], "review": [235, 180, 0], "unknown": [128, 128, 128], "candidates": [40, 110, 230]}
NAMES = {"coca": "Coca-Cola", "coca_zero": "zero", "coca_classic": "classic"}
PANEL = {"column_width_px": 400, "max_photo_height_px": 600, "font_px": 20}
TOP3 = [("coca_zero", 0.83), ("coca_classic", 0.80), ("sprite", 0.52)]


def test_decisions_map_to_the_three_colours():
    assert decision_colour("automatic", COLOURS) == (0, 170, 0) and decision_colour("review", COLOURS) == (235, 180, 0)
    assert decision_colour("unknown", COLOURS) == (128, 128, 128)


def test_panel_text_follows_the_decision():
    assert panel_entry(3, "automatic", TOP3) == ["#3  coca_zero  0.83"]
    r = panel_entry(4, "review", TOP3)
    assert r[0] == "#4  review" and len(r) == 4 and "coca_classic  0.80" in r[2]
    u = panel_entry(5, "unknown", TOP3)
    assert u == ["#5  unknown", "   rejected: coca_zero  0.83"]


def test_drawing_colours_the_boxes_and_adds_a_panel():
    img = Image.new("RGB", (400, 800), (255, 255, 255))
    boxes = np.array([[20, 100, 100, 300], [150, 100, 230, 300], [270, 100, 350, 300]], float)
    out = draw_result(img, boxes, ["automatic", "review", "unknown"], [TOP3] * 3, COLOURS, PANEL, "demo")
    assert out.height >= 600 and out.width > 450                         # the photo was scaled to 600 px tall, plus the panel
    scale = 600 / 800
    px = lambda x, y: out.getpixel((int(x * scale), int(y * scale)))      # noqa: E731
    assert px(20 + 1, 200) == (0, 170, 0) or px(21, 200)[1] > 150          # the left edge of box 1 is green
    assert px(150 + 1, 200)[0] > 200 and px(150 + 1, 200)[2] < 100          # box 2: yellow
    assert abs(px(270 + 1, 200)[0] - 128) < 30                              # box 3: grey


def test_a_long_panel_is_split_into_columns():
    img = Image.new("RGB", (300, 300), (255, 255, 255))
    boxes = np.array([[10 + i, 10, 30 + i, 60] for i in range(40)], float)
    out = draw_result(img, boxes, ["review"] * 40, [TOP3] * 40, COLOURS, PANEL, "demo")
    assert out.width > 300 + 400                                           # more than one column


def test_json_has_every_box_with_pixels_score_decision_and_top3():
    j = build_json("s05_c3a58ca5", (400, 800), np.array([[1.04, 2, 3, 4.0]]), np.array([0.912345]), ["review"], [TOP3], {"detector": "x"})
    assert j["image"] == "s05_c3a58ca5" and j["boxes"][0]["box_xyxy"] == [1.0, 2.0, 3.0, 4.0] and j["boxes"][0]["detector_score"] == 0.9123
    assert j["boxes"][0]["decision"] == "review" and j["boxes"][0]["top3"][0] == {"product": "coca_zero", "similarity": 0.83}
    json.dumps(j)


def _raw(tmp_path):
    names = {"s01_aaaaaaaa": "real store name_a.jpg", "s02_bbbbbbbb": "real store name_b.jpg"}          # synthetic
    for pid, n in names.items():
        (tmp_path / n).write_bytes(pid.encode() * 50)
    return names


def test_a_test_photo_is_refused_without_printing_any_file_name(tmp_path):
    names = _raw(tmp_path)
    index = content_index(tmp_path, names, tmp_path / "cache" / "hashes.csv")
    sha = file_sha256(tmp_path / "real store name_b.jpg")
    with pytest.raises(SystemExit) as e:
        check_not_test(sha, index, {"s02_bbbbbbbb"})
    msg = str(e.value)
    assert "s02_bbbbbbbb" in msg and "real store name" not in msg and str(tmp_path) not in msg


def test_train_val_and_unknown_files_are_allowed_and_a_copy_with_another_name_is_still_caught(tmp_path):
    names = _raw(tmp_path)
    index = content_index(tmp_path, names, tmp_path / "hashes.csv")
    assert check_not_test(file_sha256(tmp_path / "real store name_a.jpg"), index, {"s02_bbbbbbbb"}) == "s01_aaaaaaaa"   # allowed, returns its anonymous id
    (tmp_path / "copy.jpg").write_bytes(b"s02_bbbbbbbb" * 50)                                                      # same bytes, other name
    with pytest.raises(SystemExit):
        check_not_test(file_sha256(tmp_path / "copy.jpg"), index, {"s02_bbbbbbbb"})
    (tmp_path / "other.jpg").write_bytes(b"something else")
    assert check_not_test(file_sha256(tmp_path / "other.jpg"), index, {"s02_bbbbbbbb"}) is None


def test_the_hash_cache_is_reused_and_rebuilt_when_the_photo_count_changes(tmp_path):
    names = _raw(tmp_path)
    cache = tmp_path / "h.csv"
    a = content_index(tmp_path, names, cache)
    assert cache.exists() and content_index(tmp_path, names, cache) == a
    names["s03_cccccccc"] = "real store name_c.jpg"
    (tmp_path / "real store name_c.jpg").write_bytes(b"c" * 30)
    assert len(content_index(tmp_path, names, cache)) == 3


def test_candidates_have_their_own_colour_and_panel_text():
    assert decision_colour("candidates", COLOURS) == (40, 110, 230)
    brand = {"kind": "brand", "brand": "coca", "items": [("coca_zero", 0.83), ("coca_classic", 0.80)]}
    assert panel_entry(6, "candidates", TOP3, brand, NAMES) == ["#6  Coca-Cola: which one?", "   zero  0.83", "   classic  0.80"]
    lst = {"kind": "list", "brand": "coca", "items": [("coca_zero", 0.83), ("sprite", 0.80)]}
    assert panel_entry(7, "candidates", TOP3, lst, NAMES) == ["#7  one of:", "   zero  0.83", "   sprite  0.80"]


def test_drawing_and_json_with_candidates():
    img = Image.new("RGB", (400, 800), (255, 255, 255))
    boxes = np.array([[20, 100, 100, 300], [150, 100, 230, 300]], float)
    c = {"kind": "brand", "brand": "coca", "items": [("coca_zero", 0.83), ("coca_classic", 0.80)]}
    out = draw_result(img, boxes, ["candidates", "review"], [TOP3] * 2, COLOURS, PANEL, "demo", [c, None], NAMES)
    px = out.getpixel((int(21 * 0.75), int(200 * 0.75)))
    assert px[2] > 180 and px[0] < 100                                   # box 1 is blue
    j = build_json("x", (400, 800), boxes, np.array([0.9, 0.8]), ["candidates", "review"], [TOP3] * 2, {}, [c, None])
    assert j["boxes"][0]["candidates"] == {"kind": "brand", "brand": "coca", "items": [{"product": "coca_zero", "similarity": 0.83}, {"product": "coca_classic", "similarity": 0.8}]}
    assert "candidates" not in j["boxes"][1]
    json.dumps(j)
