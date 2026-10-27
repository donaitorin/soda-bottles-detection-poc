import numpy as np
from PIL import Image

from src.embeddings.experiments import (COCA, balanced_accuracy, coca_confusion, coca_variant_top1, colour_descriptor, combine_sims,
                                        decide_prob, loso, prob_features, tune_prob)

CENTRE = {"x": [0.25, 0.75], "y": [0.30, 0.70]}
BINS = {"h": 12, "s": 3, "v": 3}


def test_colour_descriptor_is_unit_length_and_separates_red_from_black_and_silver():
    red, black, silver, red2 = (Image.new("RGB", (60, 120), c) for c in [(200, 20, 20), (15, 15, 15), (190, 190, 195), (210, 25, 15)])
    d = {k: colour_descriptor(im, CENTRE, BINS) for k, im in dict(red=red, black=black, silver=silver, red2=red2).items()}
    assert abs(float(np.linalg.norm(d["red"])) - 1) < 1e-5
    assert d["red"] @ d["red2"] > 0.9 > d["red"] @ d["black"] and d["red"] @ d["silver"] < 0.2


def test_colour_descriptor_looks_only_at_the_centre():
    im = Image.new("RGB", (100, 100), (0, 0, 0))
    im.paste((255, 0, 0), (25, 30, 75, 70))                      # exactly the central part is red
    assert colour_descriptor(im, CENTRE, BINS) @ colour_descriptor(Image.new("RGB", (100, 100), (255, 0, 0)), CENTRE, BINS) > 0.99


def test_combined_similarity_stays_on_the_cosine_scale():
    a, b = np.array([[0.8]]), np.array([[0.4]])
    assert combine_sims(a, b, 0.0)[0, 0] == 0.8 and abs(combine_sims(a, b, 1.0)[0, 0] - 0.6) < 1e-9


def test_probability_rules():
    proba = np.array([[0.9, 0.05, 0.05], [0.5, 0.3, 0.2], [0.2, 0.1, 0.7], [0.3, 0.2, 0.5]])
    f = prob_features(proba, ["a", "b"])
    assert decide_prob(f, 0.4, 0.8).tolist() == ["automatic", "review", "unknown", "unknown"]
    assert decide_prob(f, 0.6, 0.8).tolist() == ["automatic", "unknown", "unknown", "unknown"]
    grid = {"start": 0.3, "stop": 0.9, "step": 0.3}
    w = {"auto_correct": 1, "auto_wrong": -5, "review": -0.1, "unknown_for_known": -1, "unknown_for_unknown": 1, "review_for_unknown": 0.5}
    th = tune_prob(f, np.array(["a", "a", "unknown", "unknown"], dtype=object), grid, w)
    assert th["t_unk"] <= th["t_auto"]


def test_loso_never_tunes_on_the_held_out_store():
    products = ["a", "b"]
    truth = np.array(["a", "b", "a", "b", "a", "b"], dtype=object)
    stores = np.array(["s1", "s1", "s2", "s2", "s3", "s3"])
    feat = {"products": products, "top1": np.array([0, 1, 0, 1, 0, 1])}
    seen = []

    def tune_fn(f, t):
        seen.append(len(t))
        return {"x": 0}
    out, used = loso({"only": feat}, truth, stores, products, tune_fn, lambda f, th: np.array(["automatic"] * len(f["top1"]), dtype=object))
    assert seen == [4, 4, 4] and len(out) == 6 and [u["store"] for u in used] == ["s1", "s2", "s3"]


def test_coca_metrics_and_confusion_ignore_other_products():
    truth = np.array(["coca_zero", "coca_zero", "coca_light", "sprite", "coca_classic"], dtype=object)
    top1 = np.array(["coca_zero", "coca_classic_returnable", "sprite", "sprite", "coca_classic"], dtype=object)
    assert coca_variant_top1(range(5), truth, top1) == 0.5
    cols, m = coca_confusion(truth, top1, ["coca_zero"])
    assert cols[-1] == "sprite_or_fanta" and m.sum() == 4 and m[COCA.index("coca_light"), 4] == 1 and m[COCA.index("coca_zero"), COCA.index("coca_zero")] == 1
