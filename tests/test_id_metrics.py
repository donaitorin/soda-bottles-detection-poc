import numpy as np

from src.embeddings.id_metrics import confusion, outcome_counts, summarize

PRODUCTS = ["coca_zero", "fanta"]
BRAND = {"coca_zero": "coca", "fanta": "fanta"}
W = {"auto_correct": 1.0, "auto_wrong": -5.0, "review": -0.1, "unknown_for_known": -1.0, "unknown_for_unknown": 1.0, "review_for_unknown": 0.5}
# crops: 0 known ok automatic, 1 known wrong automatic, 2 known review, 3 unknown rejected, 4 unknown false known, 5 not_a_bottle automatic
TRUTH = np.array(["coca_zero", "fanta", "fanta", "unknown", "unknown", "not_a_bottle"], dtype=object)
OUT = np.array(["automatic", "automatic", "review", "unknown", "automatic", "automatic"], dtype=object)
TOP1 = np.array(["coca_zero", "coca_zero", "fanta", "fanta", "fanta", "fanta"], dtype=object)
TOP3 = [["coca_zero", "fanta"], ["coca_zero", "fanta"], ["fanta", "coca_zero"], ["fanta"], ["fanta"], ["fanta"]]


def test_summary_counts_known_and_unknown_separately_and_ignores_not_a_bottle():
    m = summarize(range(6), TRUTH, OUT, TOP1, TOP3, BRAND, PRODUCTS, W)
    assert m["n_known"] == 3 and m["n_unknown"] == 2
    assert abs(m["top1_acc"] - 2 / 3) < 1e-9 and m["top3_acc"] == 1.0 and abs(m["brand_acc"] - 2 / 3) < 1e-9
    assert abs(m["auto_correct_rate"] - 1 / 3) < 1e-9 and abs(m["known_review_rate"] - 1 / 3) < 1e-9
    assert m["unknown_rejection"] == 0.5 and m["false_known"] == 0.5 and m["unknown_strict_rejection"] == 0.5
    assert abs(m["confident_wrong_rate"] - 2 / 3) < 1e-9       # 3 automatic labeled answers, 2 wrong (known wrong, false known)


def test_repeated_indices_count_repeatedly_like_a_bootstrap_sample():
    a = summarize([0, 0, 1], TRUTH, OUT, TOP1, TOP3, BRAND, PRODUCTS, W)
    assert a["n_known"] == 3 and abs(a["top1_acc"] - 2 / 3) < 1e-9


def test_empty_groups_give_nan_not_a_crash():
    m = summarize([5], TRUTH, OUT, TOP1, TOP3, BRAND, PRODUCTS, W)
    assert np.isnan(m["top1_acc"]) and np.isnan(m["unknown_rejection"])


def test_outcome_counts_for_not_a_bottle():
    assert outcome_counts(range(6), TRUTH, OUT, "not_a_bottle") == {"automatic": 1, "review": 0, "unknown": 0, "n": 1}


def test_confusion_columns_are_products_review_unknown():
    rows, cols, m = confusion(range(6), TRUTH, OUT, TOP1, PRODUCTS)
    assert rows == ["coca_zero", "fanta", "unknown"] and cols == ["coca_zero", "fanta", "review", "unknown"]
    assert m[0, 0] == 1 and m[1, 0] == 1 and m[1, 2] == 1 and m[2, 3] == 1 and m[2, 1] == 1 and m.sum() == 5
