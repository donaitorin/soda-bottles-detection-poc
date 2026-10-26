import numpy as np

from src.embeddings.identify import AUTO, UNKNOWN, decide, features, leave_one_store_out, tune, utility

PRODUCTS = ["coca_zero", "fanta", "sprite"]
CFG = {"grid": {"top1_similarity": {"start": 0.2, "stop": 0.9, "step": 0.1}, "margin": [0.0, 0.05], "vote": [0.0, 0.6]},
       "utility": {"auto_correct": 1.0, "auto_wrong": -5.0, "review": -0.1, "unknown_for_known": -1.0, "unknown_for_unknown": 1.0, "review_for_unknown": 0.5}}


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


# a toy catalog in 3 dimensions: each product has two source photos, one original and one augmented entry each
CAT = _unit([[1, 0.1, 0], [1, 0.0, 0.1], [0.1, 1, 0], [0, 1, 0.1], [0, 0.1, 1], [0.1, 0, 1]])
ENTRY_PRODUCT = np.array(["coca_zero", "coca_zero", "fanta", "fanta", "sprite", "sprite"])
ENTRY_PHOTO = np.array(["z/1", "z/2", "f/1", "f/2", "s/1", "s/2"])


def test_the_nearest_product_wins_and_votes_count_source_photos():
    f = features(_unit([[1, 0.05, 0.0], [0, 1, 0.05]]), CAT, ENTRY_PRODUCT, ENTRY_PHOTO, PRODUCTS, k=2)
    assert [PRODUCTS[i] for i in f["top1"]] == ["coca_zero", "fanta"]
    assert np.all(f["top1_sim"] > 0.95) and np.all(f["margin"] > 0.5) and np.all(f["vote"] == 1.0)


def test_an_augmented_copy_never_counts_twice_one_photo_one_vote():
    # photo z/1 has an original and an augmented entry: both identical to the query, but the vote is over PHOTOS
    cat = np.vstack([CAT, CAT[0:1]])
    f = features(_unit([[1, 0.1, 0]]), cat, np.append(ENTRY_PRODUCT, "coca_zero"), np.append(ENTRY_PHOTO, "z/1"), PRODUCTS, k=3)
    assert abs(f["vote"][0] - 2 / 3) < 1e-9          # 2 coca_zero photos among the 3 nearest photos, although 3 entries point at coca_zero


def test_entry_mask_restricts_the_catalog():
    mask = np.array([True, False, True, False, True, False])
    f = features(_unit([[1, 0, 0]]), CAT, ENTRY_PRODUCT, ENTRY_PHOTO, PRODUCTS, k=3, entry_mask=mask)
    assert PRODUCTS[f["top1"][0]] == "coca_zero"


def test_rules_unknown_review_automatic():
    f = features(_unit([[1, 0.05, 0.0], [1, 1, 1], [-1, -1, -1]]), CAT, ENTRY_PRODUCT, ENTRY_PHOTO, PRODUCTS, k=2)
    out = decide(f, t_sim=0.9, t_margin=0.2, t_vote=0.5)
    assert out[0] == AUTO and out[1] == UNKNOWN
    assert out[2] == UNKNOWN                                                  # far from everything


def test_utility_rewards_correct_automatic_and_punishes_confident_errors():
    w = CFG["utility"]
    truth = np.array(["fanta", "unknown", "sprite"])
    assert utility(np.array([AUTO, UNKNOWN, AUTO]), np.array(["fanta", "x", "sprite"]), truth, w) == 1.0
    assert utility(np.array([AUTO, AUTO, AUTO]), np.array(["sprite", "fanta", "sprite"]), truth, w) < 0          # wrong and false known


def test_tuning_and_leave_one_store_out_run_and_are_reproducible():
    rng = np.random.default_rng(0)
    q = np.vstack([_unit(CAT[i % 6] + rng.normal(0, 0.05, 3)) for i in range(24)])
    truth = np.array([ENTRY_PRODUCT[i % 6] for i in range(24)])
    stores = np.array(["s1", "s2", "s3"] * 8)
    f = features(q, CAT, ENTRY_PRODUCT, ENTRY_PHOTO, PRODUCTS, k=2)
    th = tune(f, truth, CFG)
    assert th["utility"] > 0.5
    out1, used1 = leave_one_store_out(f, truth, stores, CFG)
    out2, _ = leave_one_store_out(f, truth, stores, CFG)
    assert (out1 == out2).all() and len(used1) == 3 and {u["store"] for u in used1} == {"s1", "s2", "s3"}
