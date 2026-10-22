import numpy as np

from src.evaluation.detection import (bootstrap_by_store, gt_counts, ignore_below_floor, operating_point,
                                      per_photo_results, summarize)

W, H = 1000, 800  # synthetic photos and ids


def _gt(boxes_by_photo, stores):
    images, anns = [], []
    for i, (pid, boxes) in enumerate(boxes_by_photo.items(), 1):
        images.append({"id": i, "photo_id": pid, "width": W, "height": H, "store_code": stores[pid]})
        for x1, y1, x2, y2 in boxes:
            anns.append({"id": len(anns) + 1, "image_id": i, "category_id": 1, "bbox": [x1, y1, x2 - x1, y2 - y1],
                         "area": (x2 - x1) * (y2 - y1), "iscrowd": 0})
    return {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "bottle"}]}


A = [[100, 100, 160, 300], [200, 100, 260, 300]]
B = [[100, 100, 160, 300]]
STORES = {"s01_aaaaaaaa": "s01", "s02_bbbbbbbb": "s02"}


def _eval(preds):
    gt = _gt({"s01_aaaaaaaa": A, "s02_bbbbbbbb": B}, STORES)
    results, ref = per_photo_results(gt, preds)
    return gt, results, ref


def test_perfect_predictions_give_ap_one():
    _, res, _ = _eval({"s01_aaaaaaaa": (np.array(A, float), np.array([.9, .8])), "s02_bbbbbbbb": (np.array(B, float), np.array([.7]))})
    m = summarize(list(res.values()))
    assert m["AP"] == 1.0 and m["recall50"] == 1.0
    assert operating_point(list(res.values()))["fp"] == 0


def test_a_missed_box_and_an_extra_box_are_counted():
    preds = {"s01_aaaaaaaa": (np.array([A[0], [500, 100, 560, 300]], float), np.array([.9, .8])),
             "s02_bbbbbbbb": (np.array(B, float), np.array([.7]))}
    _, res, _ = _eval(preds)
    op = operating_point(list(res.values()))
    assert (op["tp"], op["fp"], op["fn"]) == (2, 1, 1)
    assert abs(op["recall"] - 2 / 3) < 1e-9 and abs(op["precision"] - 2 / 3) < 1e-9


def test_no_predictions_give_zero():
    _, res, _ = _eval({"s01_aaaaaaaa": (np.zeros((0, 4)), np.zeros(0)), "s02_bbbbbbbb": (np.zeros((0, 4)), np.zeros(0))})
    assert summarize(list(res.values()))["AP"] == 0.0


def test_predictions_under_the_height_floor_are_ignored_not_false_positives():
    preds = {"s01_aaaaaaaa": (np.array(A + [[500, 100, 520, 108]], float), np.array([.9, .8, .6])),  # 8 px < 12 px floor
             "s02_bbbbbbbb": (np.array(B, float), np.array([.7]))}
    kept, removed = ignore_below_floor(preds, {k: (W, H) for k in preds}, 0.015)
    assert removed == 1 and len(kept["s01_aaaaaaaa"][0]) == 2
    _, res, _ = _eval(kept)
    assert operating_point(list(res.values()))["fp"] == 0


def test_own_accumulation_matches_cocoeval_on_noisy_predictions():
    rng = np.random.default_rng(0)
    boxes = {f"s0{k}_{k:08x}": [[x, 100, x + 60, 300] for x in range(10, 900, 90)] for k in range(1, 5)}
    stores = {p: p[:3] for p in boxes}
    gt = _gt(boxes, stores)
    preds = {}
    for p, bs in boxes.items():
        b = np.array(bs, float) + rng.normal(0, 8, (len(bs), 4))
        keep = rng.random(len(b)) > 0.15
        extra = np.array([[rng.uniform(0, 900), 50, 0, 250]], float)
        extra[:, 2] = extra[:, 0] + 50
        preds[p] = (np.vstack([b[keep], extra]), rng.uniform(0.1, 1, keep.sum() + 1))
    res, ref = per_photo_results(gt, preds)
    m = summarize(list(res.values()))
    assert abs(m["AP"] - ref[0]) < 1e-9 and abs(m["AP50"] - ref[1]) < 1e-9 and 0 < m["AP"] < 1


def test_bootstrap_resamples_whole_stores_and_is_reproducible():
    seen = []

    def metric(photos):
        seen.append(sorted(photos))
        return {"x": float(len(photos))}

    store_of = {"p1": "s01", "p2": "s01", "p3": "s02"}
    a = bootstrap_by_store(store_of, metric, 50, seed=1)
    b = bootstrap_by_store(store_of, metric, 50, seed=1)
    assert a == b
    assert all(s.count("p1") == s.count("p2") for s in seen)   # the two photos of a store always travel together


def test_gt_counts():
    assert gt_counts(_gt({"s01_aaaaaaaa": A, "s02_bbbbbbbb": B}, STORES)) == {"s01_aaaaaaaa": 2, "s02_bbbbbbbb": 1}
