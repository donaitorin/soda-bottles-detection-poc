"""Pieces of the Stage 6d experiments on separating the Coca-Cola variants: colour descriptor, probability rules,
leave-one-store-out with a choice of option, and the variant metrics. All development numbers (agreement with the owner's labels).
"""
import numpy as np
from PIL import Image

from src.embeddings.id_metrics import summarize
from src.embeddings.identify import AUTO, REVIEW, UNKNOWN, subset, utility

COCA = ["coca_classic", "coca_classic_returnable", "coca_zero", "coca_light"]


# ---------------------------------------------------------------- colour descriptor
def colour_descriptor(image: Image.Image, centre: dict, bins: dict) -> np.ndarray:
    """Hellinger-normalised joint HSV histogram of the central part of a crop (where the label is): sqrt of the L1-normalised
    histogram, so its L2 norm is 1 and a dot product between two descriptors is a cosine similarity. Hue is circular (the bins wrap
    naturally because PIL's hue is 0 to 255 around the circle)."""
    w, h = image.size
    x0, x1 = (int(w * v) for v in centre["x"])
    y0, y1 = (int(h * v) for v in centre["y"])
    hsv = np.asarray(image.convert("RGB").crop((x0, y0, max(x1, x0 + 1), max(y1, y0 + 1))).convert("HSV"), dtype=np.int64).reshape(-1, 3)
    idx = (hsv[:, 0] * bins["h"] // 256) * bins["s"] * bins["v"] + (hsv[:, 1] * bins["s"] // 256) * bins["v"] + (hsv[:, 2] * bins["v"] // 256)
    hist = np.bincount(idx, minlength=bins["h"] * bins["s"] * bins["v"]).astype(np.float64)
    return np.sqrt(hist / hist.sum()).astype(np.float32)


def combine_sims(cls_sims: np.ndarray, colour_sims: np.ndarray, w: float) -> np.ndarray:
    """(cls + w * colour) / (1 + w): stays on the cosine scale, so the similarity thresholds keep their meaning."""
    return (cls_sims + w * colour_sims) / (1.0 + w)


# ---------------------------------------------------------------- logistic regression helpers (the model itself: src/embeddings/classifier.py)
def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean([(y_pred[y_true == c] == c).mean() for c in sorted(set(y_true))]))


# ---------------------------------------------------------------- probability rules for the classifier
def prob_features(proba: np.ndarray, products: list[str]) -> dict:
    """Features from class probabilities over (products + unknown): top product among the products, its probability, the unknown
    class probability and whether `unknown` is the most probable class. `prod_sim` holds the product probabilities."""
    p = proba[:, :len(products)]
    order = np.argsort(-p, axis=1)
    top1 = order[:, 0]
    return {"products": products, "prod_sim": p, "top1": top1, "top1_sim": p[np.arange(len(p)), top1], "top3": order[:, :3],
            "unknown_wins": proba.argmax(axis=1) == len(products)}


def decide_prob(feat: dict, t_unk: float, t_auto: float) -> np.ndarray:
    """unknown if the unknown class is the most probable or the top product's probability is below t_unk; else automatic if it is
    at least t_auto, else review."""
    out = np.full(len(feat["top1"]), REVIEW, dtype=object)
    out[feat["top1_sim"] >= t_auto] = AUTO
    out[feat["unknown_wins"] | (feat["top1_sim"] < t_unk)] = UNKNOWN
    return out


def tune_prob(feat: dict, truth: np.ndarray, grid: dict, w: dict) -> dict:
    ts = np.round(np.arange(grid["start"], grid["stop"] + 1e-9, grid["step"]), 4)
    products = np.array(feat["products"])[feat["top1"]]
    best, best_u = None, -1e9
    for t_unk in ts:
        for t_auto in ts:
            if t_auto < t_unk:
                continue
            u = utility(decide_prob(feat, t_unk, t_auto), products, truth, w)
            if u > best_u + 1e-12:
                best, best_u = {"t_unk": float(t_unk), "t_auto": float(t_auto)}, u
    return {**best, "utility": best_u}


# ---------------------------------------------------------------- leave one store out with a choice of option
def loso(options: dict, truth: np.ndarray, stores: np.ndarray, products: list[str], tune_fn, decide_fn, choose_fn=None):
    """For each held-out store: pick the option with `choose_fn(options, idx)` on the OTHER stores' labeled crops (default: the first
    option), tune the thresholds for it on those crops with `tune_fn(feat, truth)`, and apply them to the held-out store with
    `decide_fn(feat, th)`. Returns (outcome pooled over stores, [{store, option, ...thresholds}])."""
    labeled = np.isin(truth, products + ["unknown"])
    outcome = np.full(len(truth), UNKNOWN, dtype=object)
    used = []
    for s in sorted(set(stores)):
        tune_idx = np.where(labeled & (stores != s))[0]
        name = choose_fn(options, tune_idx) if choose_fn else next(iter(options))
        feat = options[name]
        th = tune_fn(subset(feat, tune_idx), truth[tune_idx])
        test_idx = np.where(stores == s)[0]
        outcome[test_idx] = decide_fn(subset(feat, test_idx), th)
        used.append({"store": s, "option": name, **th})
    return outcome, used


def top1_on_known(feat: dict, truth: np.ndarray, idx: np.ndarray) -> float:
    prods = np.array(feat["products"], dtype=object)
    known = [i for i in idx if truth[i] in feat["products"]]
    return float(np.mean([prods[feat["top1"][i]] == truth[i] for i in known])) if known else 0.0


# ---------------------------------------------------------------- metrics
def coca_variant_top1(idx, truth: np.ndarray, top1_product: np.ndarray) -> float:
    idx = np.asarray(idx, int)
    sel = idx[np.isin(truth[idx], COCA)]
    return float((top1_product[sel] == truth[sel]).mean()) if len(sel) else float("nan")


def full_metrics(idx, truth, outcome, top1_product, top3_products, brand_of, products, w) -> dict:
    return {**summarize(idx, truth, outcome, top1_product, top3_products, brand_of, products, w), "coca_top1": coca_variant_top1(idx, truth, top1_product)}


def coca_confusion(truth: np.ndarray, top1_product: np.ndarray, products: list[str]) -> tuple[list[str], np.ndarray]:
    """Rows: the 4 Coca-Cola truths. Columns: the 4 Coca variants, then `sprite or fanta`. Nearest product, no rejection rules."""
    cols = COCA + ["sprite_or_fanta"]
    m = np.zeros((len(COCA), len(cols)), int)
    for t, p in zip(truth, top1_product):
        if t in COCA:
            m[COCA.index(t), COCA.index(p) if p in COCA else 4] += 1
    return cols, m
