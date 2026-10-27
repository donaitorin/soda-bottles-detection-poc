import numpy as np

from src.embeddings.answers import (candidates_info, to_candidates, calibrate_bm, calibrate_by_store, candidate_sets, conformal_delta, list_metrics, nonconformity,
                                    two_level)

PRODUCTS = ["c1", "c2", "sp", "fa"]
BRANDS = {"c1": "coca", "c2": "coca", "sp": "sprite", "fa": "fanta"}


def test_nonconformity_is_zero_when_the_true_product_is_nearest():
    s = np.array([[0.9, 0.5, 0.1, 0.0], [0.5, 0.9, 0.1, 0.0]])
    assert np.allclose(nonconformity(s, np.array([0, 0])), [0.0, 0.4])


def test_conformal_delta_uses_the_finite_sample_rank():
    scores = np.arange(10) / 10                                  # 0.0 .. 0.9
    assert conformal_delta(scores, 0.9) == 0.9                  # ceil(11 * 0.9) = 10 -> the largest
    assert conformal_delta(scores, 0.5) == 0.5                  # ceil(5.5) = 6 -> sixth smallest
    assert conformal_delta(np.array([]), 0.9) == float("inf")


def test_candidate_sets_contain_the_top1_and_everything_within_delta():
    s = np.array([[0.9, 0.85, 0.5, 0.1]])
    assert candidate_sets(s, 0.0).tolist() == [[True, False, False, False]]
    assert candidate_sets(s, 0.06).tolist() == [[True, True, False, False]]


def test_calibrated_delta_reaches_the_target_coverage_on_new_crops():
    rng = np.random.default_rng(0)
    def draw(n):
        truth = rng.integers(0, 4, n)
        s = rng.random((n, 4)) * 0.5
        s[np.arange(n), truth] += rng.random(n) * 0.6              # the true product is usually, not always, nearest
        return s, truth
    s_cal, t_cal = draw(2000)
    delta = conformal_delta(nonconformity(s_cal, t_cal), 0.9)
    s_new, t_new = draw(2000)
    cov = candidate_sets(s_new, delta)[np.arange(2000), t_new].mean()
    assert 0.87 < cov < 0.93


def test_list_metrics_sizes_coverage_and_brand():
    sets = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 0]], bool)
    m = list_metrics(sets, np.array([0, 1, 1, 0]), PRODUCTS, BRANDS)
    assert m["coverage"] == 0.75 and m["size1"] == 0.25 and m["size2"] == 0.5 and m["size3plus"] == 0.25 and m["n_multi"] == 3
    assert abs(m["multi_one_brand"] - 1 / 3) < 1e-9             # only [c1, c2] stays inside one brand


def test_two_level_needs_a_clear_brand_and_a_brand_with_several_variants():
    s = np.array([[0.80, 0.78, 0.40, 0.30],                     # Coca-Cola clear, variants unclear
                  [0.80, 0.78, 0.79, 0.30],                     # brand not clear (sprite is close)
                  [0.40, 0.30, 0.80, 0.30]])                    # Sprite clear but a single-variant brand
    sets = candidate_sets(s, 0.05)
    applies, variants = two_level(s, sets, PRODUCTS, BRANDS, t_bm=0.1)
    assert applies.tolist() == [True, False, False] and variants[0].tolist() == [True, True, False, False]


def test_brand_margin_calibration_picks_the_smallest_margin_that_is_precise_enough():
    margin = np.array([0.0, 0.01, 0.02, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13])
    correct = margin >= 0.05
    assert calibrate_bm(margin, correct, [0.0, 0.02, 0.05, 0.1], 0.95, 5) == 0.05
    assert calibrate_bm(margin, correct, [0.0, 0.02], 0.95, 5) == float("inf")


def test_calibration_by_store_never_uses_the_held_out_store():
    rng = np.random.default_rng(1)
    n = 120
    truth = np.array(rng.choice(PRODUCTS, n), dtype=object)
    stores = np.array(["s1", "s2", "s3"])[rng.integers(0, 3, n)]
    s = rng.random((n, 4)) * 0.4
    s[np.arange(n), [PRODUCTS.index(t) for t in truth]] += 0.3
    cfg = {"coverage": 0.9, "brand_margin_grid": [0.0, 0.05], "brand_precision_target": 0.9, "min_answered": 5}
    a = calibrate_by_store(s, truth, stores, PRODUCTS, BRANDS, cfg)
    s2 = s.copy()
    s2[stores == "s1", 1] += 3.0                                  # change s1 only: the calibration FOR s1 must not change
    b = calibrate_by_store(s2, truth, stores, PRODUCTS, BRANDS, cfg)
    assert a["s1"] == b["s1"] and a["s2"] != b["s2"]


def test_candidates_replace_review_only_and_info_lists_most_similar_first():
    s = np.array([[0.80, 0.78, 0.40, 0.30], [0.50, 0.48, 0.47, 0.46], [0.80, 0.78, 0.40, 0.30]])
    sets = candidate_sets(s, 0.05)
    applies, variants = two_level(s, sets, PRODUCTS, BRANDS, t_bm=0.1)
    outcome = np.array(["review", "review", "automatic"], dtype=object)
    final, kind = to_candidates(outcome, sets, applies, max_list=3)
    assert final.tolist() == ["candidates", "review", "automatic"] and kind.tolist() == ["brand", "", ""]   # row 2: list of 4 is too long
    info = candidates_info(s[0], variants[0], True, PRODUCTS, BRANDS)
    assert info == {"kind": "brand", "brand": "coca", "items": [("c1", 0.80), ("c2", 0.78)]}
