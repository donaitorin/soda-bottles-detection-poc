from collections import Counter

from src.embeddings.crops import allocate_quota, crop_box_with_margin, crop_id, sample_crops


def test_crop_id_is_photo_id_plus_box_index():
    assert crop_id("s03_aaaaaaaa", 7) == "s03_aaaaaaaa_007"


def test_margin_grows_the_box_and_clamps_to_the_photo():
    assert crop_box_with_margin([100, 200, 200, 400], 1000, 1000, 0.10) == (90, 180, 210, 420)
    assert crop_box_with_margin([0, 5, 100, 200], 150, 205, 0.10) == (0, 0, 110, 205)   # clamped at the photo edges


def test_margin_never_gives_an_empty_crop():
    x1, y1, x2, y2 = crop_box_with_margin([10, 10, 10.2, 10.2], 100, 100, 0.0)
    assert x2 > x1 and y2 > y1


AVAILABLE = {"s01": (200, 32), "s03": (860, 104), "s06": (117, 56), "s11": (300, 64), "s15": (590, 72), "s16": (86, 16)}


def test_allocation_reaches_the_total_and_respects_minimum_and_caps():
    q = allocate_quota(AVAILABLE, 180, 15)
    assert sum(q.values()) == 180
    assert all(q[s] >= 15 for s in q) and all(q[s] <= AVAILABLE[s][1] for s in q)
    assert q["s03"] >= q["s01"] >= q["s16"]          # bigger stores give more


def test_allocation_is_capped_by_what_exists():
    q = allocate_quota({"a": (5, 5), "b": (500, 8)}, 100, 2)
    assert q == {"a": 5, "b": 8}


def test_sampling_is_reproducible_spread_over_photos_and_capped_per_photo():
    cands = [(f"s01_p{p}", k, "s01") for p in range(4) for k in range(30)] + [("s02_q", k, "s02") for k in range(3)]
    a = sample_crops(cands, {"s01": 12, "s02": 5}, 4, seed=0)
    b = sample_crops(cands, {"s01": 12, "s02": 5}, 4, seed=0)
    assert a == b and sample_crops(cands, {"s01": 12, "s02": 5}, 4, seed=1) != a
    per = Counter(p for p, _, s in a if s == "s01")
    assert len(a) == 12 + 3 and set(per.values()) == {3} and len(per) == 4     # 12 over 4 photos, evenly
    assert len([1 for _, _, s in a if s == "s02"]) == 3                         # only 3 exist


def test_the_per_photo_cap_limits_a_store_with_few_photos():
    cands = [("s01_p0", k, "s01") for k in range(50)]
    assert len(sample_crops(cands, {"s01": 20}, 8, seed=0)) == 8


def test_targeted_pick_uses_retrieval_rank_caps_and_does_not_repeat_crops():
    import numpy as np
    from src.embeddings.crops import pick_targeted
    products = ["a", "b", "c"]
    cands = [(f"s01_p{p}", k, "s01") for p in range(3) for k in range(4)] + [("s02_q", k, "s02") for k in range(4)]
    rng = np.random.default_rng(0)
    sim = rng.random((len(cands), 3))
    taken = {("s01_p0", 0)}
    got = pick_targeted(cands, sim, products, ["a", "b"], per_product=5, top_rank=2, max_per_photo=2, max_per_store=6, taken=taken)
    assert len({(p, k) for p, k, _, _ in got}) == len(got) and ("s01_p0", 0) not in {(p, k) for p, k, _, _ in got}
    assert Counter(p for p, _, _, _ in got).most_common(1)[0][1] <= 2          # photo cap (s01_p0 already holds one)
    assert Counter(s for _, _, s, _ in got)["s01"] <= 6                        # store cap
    for pid, k, _, name in got:                                               # the product is among the crop's top 2
        i = cands.index((pid, k, _))
        assert products.index(name) in np.argsort(-sim[i])[:2]
    assert got == pick_targeted(cands, sim, products, ["a", "b"], 5, 2, 2, 6, {("s01_p0", 0)})   # deterministic
