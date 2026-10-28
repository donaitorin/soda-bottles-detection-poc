import numpy as np

from src.ocr.combine import combine, contradicts
from src.ocr.variant_b import status, weight_grid

P = ["coca_classic", "coca_classic_returnable", "coca_zero", "coca_light", "sprite", "fanta"]
EV = {"p_ctx": 0.05, "p_ctx_coca": 0.30}
W = {"w_brand": 2, "w_variant": 2, "w_unknown": 2}
OTHER = ["BILZ", "PEPSI"]
WORDS_OF = {"coca_classic": {"COCA-COLA"}, "coca_classic_returnable": {"COCA-COLA", "RETORNABLE"}, "coca_zero": {"COCA-COLA", "ZERO"},
            "coca_light": {"COCA-COLA", "LIGHT"}, "sprite": {"SPRITE"}, "fanta": {"FANTA"}}


def row(**kw):
    p = np.full(7, 0.01)
    for k, v in kw.items():
        p[(P + ["unknown"]).index(k)] = v
    return p / p.sum()


def run(p, ev, w=W):
    out, info = combine(np.array([p]), P, [ev], w, EV, OTHER)
    return out[0], info[0]


def test_no_evidence_keeps_the_row_exactly():
    p = row(sprite=0.3, unknown=0.5)
    out, info = run(p, {})
    assert (out == p).all() and info["rules"] == []


def test_brand_word_boosts_only_with_classifier_context():
    p = row(fanta=0.2, unknown=0.6)
    out, info = run(p, {"FANTA": "exact"})
    assert out[5] > p[5] and out.argmax() == 5 and info["rules"] == ["boost_fanta"]
    low = row(fanta=0.01, unknown=0.9)                      # below p_ctx: stray text does not rescue
    assert (run(low, {"FANTA": "exact"})[0] == low).all()


def test_variant_word_needs_coca_context_and_zero_weight_is_variant_a():
    p = row(coca_light=0.4, coca_zero=0.35)
    out, _ = run(p, {"ZERO": "exact"})
    assert out.argmax() == 2
    sprite = row(sprite=0.8)                                  # no Coca context: ignored
    assert (run(sprite, {"ZERO": "exact"})[0] == sprite).all()
    assert (run(p, {"ZERO": "exact"}, {"w_brand": 0, "w_variant": 0, "w_unknown": 0})[0] == p).all()


def test_not_a_target_combinations_and_other_brands_push_to_unknown_without_boost():
    p = row(sprite=0.4, unknown=0.3)
    out, info = run(p, {"SPRITE": "exact", "ZERO": "exact"})
    assert out[7 - 1] > p[6] and out[4] < p[4] and "not_a_target" in info["rules"]       # sprite zero: not sprite, more unknown
    q = row(coca_classic=0.6)
    out, info = run(q, {"PEPSI": "exact"})
    assert out[6] > q[6] and info["rules"] == ["other_brand"]
    out, info = run(row(coca_classic_returnable=0.5, coca_zero=0.3), {"ZERO": "exact", "RETORNABLE": "exact"})
    assert "not_a_target" in info["rules"]


def test_two_brands_get_no_boost():
    p = row(coca_classic=0.5, fanta=0.2)
    out, info = run(p, {"FANTA": "exact", "COCA-COLA": "exact"})
    assert (out == p).all() and info["rules"] == ["two_brands"]


def test_contradiction_flags_words_of_other_products():
    ev = [{"LIGHT": "exact"}, {"ZERO": "exact"}, {}, {"COCA-COLA": "exact", "ZERO": "exact"}]
    top = np.array(["coca_zero", "coca_zero", "sprite", "coca_zero"], dtype=object)
    assert contradicts(top, ev, WORDS_OF).tolist() == [True, False, False, False]


def test_grid_starts_with_all_zeros_and_status_rules():
    g = weight_grid({"w_brand": [0, 1], "w_variant": [0, 1], "w_unknown": [0, 2], "contradiction": [0, 1]})
    assert g[0] == {"w_brand": 0, "w_variant": 0, "w_unknown": 0, "contradiction": 0} and len(g) == 16
    s = status(np.array(["sprite", "sprite", "fanta", "unknown", "unknown", "not_a_bottle"], dtype=object),
               np.array(["automatic", "unknown", "review", "unknown", "automatic", "automatic"], dtype=object),
               np.array(["sprite", "sprite", "fanta", "x", "x", "x"], dtype=object), P)
    assert s.tolist() == ["right", "wrong", "neutral", "right", "wrong", "-"]
