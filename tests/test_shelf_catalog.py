import numpy as np

from src.embeddings.identify import decide, features_from_sims, tune
from src.embeddings.shelf_catalog import assemble, capped_subset

PRODUCTS = ["a", "b", "c"]


def _unit(rows):
    v = np.array(rows, float)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_cap_keeps_small_labels_whole_and_is_seeded_and_order_independent():
    labels = np.array(["a"] * 10 + ["b"] * 3 + ["unknown"] * 8)
    ids = np.array([f"c{i:02d}" for i in range(21)])
    k1 = capped_subset(labels, ids, 4, 5, seed=0)
    assert (labels[k1] == "a").sum() == 4 and (labels[k1] == "b").sum() == 3 and (labels[k1] == "unknown").sum() == 5
    assert np.array_equal(k1, capped_subset(labels, ids, 4, 5, seed=0))
    perm = np.random.default_rng(1).permutation(21)
    assert set(ids[perm][capped_subset(labels[perm], ids[perm], 4, 5, seed=0)]) == set(ids[k1])
    assert set(ids[capped_subset(labels, ids, 4, 5, seed=1)]) != set(ids[k1])


def test_assemble_modes():
    orig = {"vec": _unit([[1, 0, 0]] * 3), "product": np.array(PRODUCTS), "photo": np.array(["a/1", "b/1", "c/1"])}
    shelf = {"vec": _unit([[1, 1, 0]] * 4), "label": np.array(["a", "a", "unknown", "b"], dtype=object), "photo": np.array(["x1", "x2", "x3", "x4"])}
    n = lambda m: len(assemble(orig, shelf, m, 10, 10, 0)["product"])  # noqa: E731
    assert (n("originals"), n("originals+shelf"), n("originals+shelf+negatives"), n("shelf")) == (3, 6, 7, 3)
    assert "unknown" not in assemble(orig, shelf, "shelf", 10, 10, 0)["product"]


def test_empty_product_never_wins_and_negatives_give_a_gap():
    entries = _unit([[1, 0, 0], [0, 1, 0], [0.8, 0.6, 0]])
    prod = np.array(["a", "b", "unknown"])
    q = _unit([[0.7, 0.7, 0]])
    f = features_from_sims(q @ entries.T, prod, np.array(["p1", "p2", "p3"]), PRODUCTS, 2)
    assert f["prod_sim"][0, 2] == -1.0 and PRODUCTS[f["top1"][0]] in ("a", "b")
    assert f["neg_gap"][0] > 0                                   # the negative is nearer than either product
    assert decide(f, 0.0, 0.0, 0.0, t_neg=0.0)[0] == "unknown" and decide(f, 0.0, 0.0, 0.0)[0] != "unknown"


def test_no_negatives_means_no_gap_and_no_tuning_of_the_rule():
    f = features_from_sims(_unit([[1, 0.1, 0]]) @ _unit([[1, 0, 0], [0, 1, 0]]).T, np.array(["a", "b"]), np.array(["p1", "p2"]), PRODUCTS, 2)
    assert np.isneginf(f["neg_gap"]).all()
    cfg = {"grid": {"top1_similarity": {"start": 0.3, "stop": 0.9, "step": 0.3}, "margin": [0.0], "vote": [0.0], "neg_gap": [9.0, 0.0]},
           "utility": {"auto_correct": 1, "auto_wrong": -5, "review": -0.1, "unknown_for_known": -1, "unknown_for_unknown": 1, "review_for_unknown": 0.5}}
    assert tune(f, np.array(["a"], dtype=object), cfg)["t_neg"] is None


def test_balanced_vote_does_not_favour_the_product_with_more_entries():
    # 4 entries of "a" slightly nearer than 1 entry of "b"; k = 5: the plain vote for the top-1 "a" is 0.8, the balanced one is lower
    vecs = _unit([[1, 0.05, 0], [1, 0.06, 0], [1, 0.07, 0], [1, 0.08, 0], [1, 0.3, 0]])
    prod = np.array(["a"] * 4 + ["b"])
    photo = np.array([f"p{i}" for i in range(5)])
    sims = _unit([[1, 0, 0]]) @ vecs.T
    plain = features_from_sims(sims, prod, photo, PRODUCTS, 5)["vote"][0]
    bal = features_from_sims(sims, prod, photo, PRODUCTS, 5, balance=True)["vote"][0]
    assert abs(plain - 0.8) < 1e-9 and abs(bal - 0.5) < 1e-9
