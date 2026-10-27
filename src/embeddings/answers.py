"""Answer types beyond automatic / review / unknown: a calibrated candidate list and the two-level (brand + variants) answer.

Both work on the per-product similarity matrix `prod_sim` (Q, P) of `identify.features`.

Candidate list (conformal-style prediction set): for a crop, every product whose similarity is within `delta` of the top-1
similarity. `delta` is calibrated on labeled KNOWN crops so that the list contains the correct product for about `coverage`
(90%) of them: the nonconformity score of a crop is (top-1 similarity - similarity of its true product), and delta is the
ceil((n+1)*coverage)/n quantile of these scores (split conformal prediction: the finite-sample rule makes the coverage hold on
average for new crops from the same distribution; with 6 stores it is only a rough guarantee).

Two-level answer: when the best BRAND is clear (its best product beats the best product of every other brand by `t_bm`) but its
variants are not, answer the brand plus the variants of that brand inside the candidate list ("Coca-Cola: classic or zero").
`t_bm` is the smallest margin at which the brand is right at least `precision_target` of the time on the calibration crops.
"""
import math

import numpy as np

CANDIDATES = "candidates"


def nonconformity(prod_sim: np.ndarray, true_idx: np.ndarray) -> np.ndarray:
    """top-1 similarity minus the similarity of the true product: 0 when the true product is the nearest, larger when it is further."""
    return prod_sim.max(axis=1) - prod_sim[np.arange(len(true_idx)), true_idx]


def conformal_delta(scores: np.ndarray, coverage: float) -> float:
    """Split-conformal quantile: the ceil((n+1)*coverage)-th smallest score (the largest score if n is too small for that rank)."""
    s = np.sort(np.asarray(scores, float))
    if len(s) == 0:
        return float("inf")
    return float(s[min(len(s), math.ceil((len(s) + 1) * coverage)) - 1])


def candidate_sets(prod_sim: np.ndarray, delta: float) -> np.ndarray:
    """(Q, P) bool: products whose similarity is within `delta` of the top-1 (the top-1 is always in)."""
    return prod_sim >= prod_sim.max(axis=1, keepdims=True) - delta - 1e-9


def brand_info(prod_sim: np.ndarray, products: list[str], brand_of: dict) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Index of the best brand and its margin over the best product of any OTHER brand. Returns (best brand index, margin, brands)."""
    brands = sorted(set(brand_of[p] for p in products))
    member = np.array([[brand_of[p] == b for p in products] for b in brands])           # (B, P)
    brand_sim = np.stack([prod_sim[:, m].max(axis=1) for m in member], axis=1)          # (Q, B)
    order = np.argsort(-brand_sim, axis=1)
    rows = np.arange(len(prod_sim))
    return order[:, 0], brand_sim[rows, order[:, 0]] - brand_sim[rows, order[:, 1]], brands


def calibrate_bm(margin: np.ndarray, correct: np.ndarray, grid: list[float], precision_target: float, min_answered: int) -> float:
    """Smallest brand margin whose answered crops have a right brand at least `precision_target` of the time (needs `min_answered`
    crops); infinity (never answer a brand) when no margin qualifies."""
    for t in sorted(grid):
        sel = margin >= t
        if sel.sum() >= min_answered and correct[sel].mean() >= precision_target:
            return float(t)
    return float("inf")


def two_level(prod_sim: np.ndarray, sets: np.ndarray, products: list[str], brand_of: dict, t_bm: float) -> tuple[np.ndarray, np.ndarray]:
    """(applies (Q,) bool, variants (Q, P) bool). It applies when the brand is clear (margin >= t_bm) and the brand has several
    products; the variants are that brand's products inside the candidate list (the top-1 product is always one of them)."""
    best, margin, brands = brand_info(prod_sim, products, brand_of)
    in_brand = np.array([[brand_of[p] == brands[b] for p in products] for b in best])    # (Q, P) products of each crop's best brand
    several = in_brand.sum(axis=1) > 1
    return (margin >= t_bm) & several, in_brand & sets


def calibrate_by_store(prod_sim: np.ndarray, truth: np.ndarray, stores: np.ndarray, products: list[str], brand_of: dict, cfg: dict) -> dict:
    """For each store: delta and t_bm calibrated on the KNOWN crops of the OTHER stores. {store: {delta, t_bm}}."""
    known = np.isin(truth, products)
    true_idx = np.array([products.index(t) if t in products else 0 for t in truth])
    brand_idx, margin, brands = brand_info(prod_sim, products, brand_of)
    true_brand = np.array([brands.index(brand_of[t]) if t in products else -1 for t in truth])
    out = {}
    for s in sorted(set(stores)):
        cal = known & (stores != s)
        delta = conformal_delta(nonconformity(prod_sim[cal], true_idx[cal]), cfg["coverage"])
        t_bm = calibrate_bm(margin[cal], (brand_idx == true_brand)[cal], cfg["brand_margin_grid"], cfg["brand_precision_target"], cfg["min_answered"])
        out[s] = {"delta": delta, "t_bm": t_bm}
    return out


def list_metrics(sets: np.ndarray, true_idx: np.ndarray, products: list[str], brand_of: dict) -> dict:
    """Coverage (the correct product is in the list), list size distribution and how often a list of 2 or more stays inside one brand."""
    n = len(sets)
    size = sets.sum(axis=1)
    covered = sets[np.arange(n), true_idx]
    multi = np.where(size >= 2)[0]
    one_brand = [len({brand_of[p] for p, m in zip(products, sets[i]) if m}) == 1 for i in multi]
    return {"n": n, "coverage": float(covered.mean()) if n else float("nan"), "mean_size": float(size.mean()) if n else float("nan"),
            "size1": float((size == 1).mean()), "size2": float((size == 2).mean()), "size3plus": float((size >= 3).mean()),
            "multi_one_brand": float(np.mean(one_brand)) if one_brand else float("nan"), "n_multi": len(multi)}


def candidates_info(prod_sim_row: np.ndarray, set_row: np.ndarray, applies: bool, products: list[str], brand_of: dict) -> dict:
    """What the panel and the JSON show for one `candidates` crop: kind `brand` (two-level answer: the brand and its variants) or `list`
    (a short list of products), with (product, similarity) for each candidate, most similar first."""
    items = sorted(((p, float(prod_sim_row[j])) for j, p in enumerate(products) if set_row[j]), key=lambda t: -t[1])
    return {"kind": "brand" if applies else "list", "brand": brand_of[items[0][0]], "items": items}


def to_candidates(outcome: np.ndarray, sets: np.ndarray, applies: np.ndarray, max_list: int) -> tuple[np.ndarray, np.ndarray]:
    """Replace `review` by `candidates` where there is something useful to say: the two-level answer applies, or the list has at most
    `max_list` products. Returns (final decisions, kind per crop: `brand`, `list` or empty)."""
    final = np.array(outcome, dtype=object)
    kind = np.full(len(final), "", dtype=object)
    for i in np.where(outcome == "review")[0]:
        if applies[i]:
            final[i], kind[i] = CANDIDATES, "brand"
        elif sets[i].sum() <= max_list:
            final[i], kind[i] = CANDIDATES, "list"
    return final, kind
