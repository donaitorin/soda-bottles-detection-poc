"""Identification by catalog similarity (pipeline variant A, embeddings only) with the unknown rules.

For one query crop: cosine similarity to every catalog entry (dot product of unit vectors); one score per SOURCE PHOTO (the best
over the photo's original and augmented entries, so augmented copies never count twice); one score per product (its best photo).
Features: top-1 product and its similarity, margin to the best DIFFERENT product, and the k-NN vote (share of the K nearest
photos that belong to the top-1 product). Rules (three outcomes):
  top-1 similarity below t_sim            -> unknown
  else vote share below t_vote            -> review
  else margin below t_margin              -> review
  else                                    -> automatic (the top-1 product)
Thresholds are tuned by maximising a utility over labeled crops (see configs/stage6.yaml `identify.utility`).
"""
import itertools

import numpy as np

AUTO, REVIEW, UNKNOWN = "automatic", "review", "unknown"


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Dot products of unit vectors (cosine similarity), checked to be finite. NumPy on macOS (Accelerate) can print spurious
    "divide by zero / overflow / invalid value in matmul" warnings for perfectly finite results (checked against float64 here:
    differences around 1e-6), so the floating-point flags are silenced for this one product and the result is asserted instead."""
    with np.errstate(all="ignore"):
        out = a @ b.T
    assert np.isfinite(out).all(), "non-finite similarity: check the embeddings"
    return out


def features(queries: np.ndarray, catalog: np.ndarray, entry_product: np.ndarray, entry_photo: np.ndarray, products: list[str], k: int,
             entry_mask: np.ndarray | None = None) -> dict:
    """Features for (Q, D) unit-length queries against an (E, D) catalog. `entry_mask` (E,) keeps only some entries (for example the
    originals only, for the plain-catalog experiment)."""
    if entry_mask is not None:
        catalog, entry_product, entry_photo = catalog[entry_mask], entry_product[entry_mask], entry_photo[entry_mask]
    sims = cosine_matrix(queries, catalog)                       # cosine similarity (Q, E)
    photos = sorted(set(entry_photo))
    photo_sim = np.stack([sims[:, entry_photo == p].max(axis=1) for p in photos], axis=1)          # (Q, P)
    photo_product = np.array([entry_product[entry_photo == p][0] for p in photos])
    prod_sim = np.stack([photo_sim[:, photo_product == p].max(axis=1) for p in products], axis=1)  # (Q, n products)
    order = np.argsort(-prod_sim, axis=1)
    top1 = order[:, 0]
    top1_sim = prod_sim[np.arange(len(top1)), top1]
    second = prod_sim[np.arange(len(top1)), order[:, 1]]
    nearest = np.argsort(-photo_sim, axis=1)[:, :k]
    vote = np.array([(photo_product[nearest[i]] == products[top1[i]]).mean() for i in range(len(top1))])
    return {"products": products, "prod_sim": prod_sim, "top1": top1, "top1_sim": top1_sim, "margin": top1_sim - second, "vote": vote,
            "top3": order[:, :3]}


def decide(feat: dict, t_sim: float, t_margin: float, t_vote: float) -> np.ndarray:
    """Outcome per query: automatic, review or unknown."""
    out = np.full(len(feat["top1"]), AUTO, dtype=object)
    out[(feat["vote"] < t_vote) | (feat["margin"] < t_margin)] = REVIEW
    out[feat["top1_sim"] < t_sim] = UNKNOWN
    return out


def utility(outcome: np.ndarray, top1_product: np.ndarray, truth: np.ndarray, w: dict) -> float:
    """Mean utility over labeled crops. truth is a product id or `unknown`."""
    total = 0.0
    for o, p, t in zip(outcome, top1_product, truth):
        if t == "unknown":
            total += {AUTO: w["auto_wrong"], REVIEW: w["review_for_unknown"], UNKNOWN: w["unknown_for_unknown"]}[o]
        else:
            total += {AUTO: w["auto_correct"] if p == t else w["auto_wrong"], REVIEW: w["review"], UNKNOWN: w["unknown_for_known"]}[o]
    return total / max(1, len(truth))


def grid_values(cfg_grid: dict) -> tuple[list, list, list]:
    g = cfg_grid["top1_similarity"]
    sims = list(np.round(np.arange(g["start"], g["stop"] + 1e-9, g["step"]), 4))
    return sims, list(cfg_grid["margin"]), list(cfg_grid["vote"])


def tune(feat: dict, truth: np.ndarray, cfg: dict) -> dict:
    """Thresholds with the best mean utility on the given crops (ties: the first in grid order, i.e. the lowest thresholds)."""
    sims, margins, votes = grid_values(cfg["grid"])
    products = np.array(feat["products"])[feat["top1"]]
    best, best_u = None, -1e9
    for ts, tm, tv in itertools.product(sims, margins, votes):
        u = utility(decide(feat, ts, tm, tv), products, truth, cfg["utility"])
        if u > best_u + 1e-12:
            best, best_u = {"t_sim": float(ts), "t_margin": float(tm), "t_vote": float(tv)}, u
    return {**best, "utility": best_u}


def subset(feat: dict, idx: np.ndarray) -> dict:
    return {k: (v[idx] if isinstance(v, np.ndarray) else v) for k, v in feat.items()}


def leave_one_store_out(feat: dict, truth: np.ndarray, stores: np.ndarray, cfg: dict, tune_exclude_stores: tuple = ()) -> tuple[np.ndarray, list[dict]]:
    """For each store: tune the thresholds on the labeled crops of the OTHER stores (minus `tune_exclude_stores`), apply them to the
    held-out store. Returns the outcome per crop (pooled over stores) and the thresholds used per store. Crops whose truth is
    not a product or `unknown` are not used for tuning but still get an outcome."""
    labeled = np.isin(truth, feat["products"] + ["unknown"])
    outcome = np.full(len(truth), UNKNOWN, dtype=object)
    used = []
    for s in sorted(set(stores)):
        tune_idx = np.where(labeled & (stores != s) & ~np.isin(stores, list(tune_exclude_stores)))[0]
        th = tune(subset(feat, tune_idx), truth[tune_idx], cfg)
        test_idx = np.where(stores == s)[0]
        outcome[test_idx] = decide(subset(feat, test_idx), th["t_sim"], th["t_margin"], th["t_vote"])
        used.append({"store": s, **th})
    return outcome, used
