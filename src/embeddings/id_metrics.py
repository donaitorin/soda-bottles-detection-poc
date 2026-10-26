"""Identification metrics on labeled crops (development numbers: a small sample from a few stores, not accuracy).

Truth is a product id, `unknown`, `not_a_bottle` or `cannot_tell`. Accuracy-style numbers use only crops labeled with a
product (known) or `unknown`; `not_a_bottle` and `cannot_tell` are reported separately (what the system answers for them).
Every function accepts an index list WITH repeats (a bootstrap sample of stores).
"""
import numpy as np

from src.embeddings.identify import AUTO, REVIEW, UNKNOWN, utility


def _rate(k: float, n: float) -> float:
    return k / n if n else float("nan")


def summarize(idx, truth, outcome, top1_product, top3_products, brand_of: dict, products: list[str], w: dict) -> dict:
    idx = np.asarray(idx, int)
    t, o, p1 = truth[idx], outcome[idx], top1_product[idx]
    known = np.isin(t, products)
    unk = t == "unknown"
    labeled = known | unk
    auto = o == AUTO
    wrong_auto = auto & labeled & (unk | (p1 != t))
    top3_ok = np.array([tt in top3_products[i] for tt, i in zip(t[known], idx[known])]) if known.any() else np.array([], bool)
    return {
        "n_known": int(known.sum()), "n_unknown": int(unk.sum()),
        "top1_acc": _rate((p1[known] == t[known]).sum(), known.sum()),
        "top3_acc": _rate(top3_ok.sum(), known.sum()),
        "brand_acc": _rate(sum(brand_of[a] == brand_of[b] for a, b in zip(p1[known], t[known])), known.sum()),
        "auto_correct_rate": _rate((auto & known & (p1 == t)).sum(), known.sum()),
        "known_review_rate": _rate((o[known] == REVIEW).sum(), known.sum()),
        "known_rejected_rate": _rate((o[known] == UNKNOWN).sum(), known.sum()),
        "unknown_rejection": _rate(((o == UNKNOWN) | (o == REVIEW))[unk].sum(), unk.sum()),
        "unknown_strict_rejection": _rate((o == UNKNOWN)[unk].sum(), unk.sum()),
        "false_known": _rate(auto[unk].sum(), unk.sum()),
        "confident_wrong_rate": _rate(wrong_auto.sum(), (auto & labeled).sum()),
        "share_automatic": _rate(auto[labeled].sum(), labeled.sum()),
        "share_review": _rate((o == REVIEW)[labeled].sum(), labeled.sum()),
        "share_unknown": _rate((o == UNKNOWN)[labeled].sum(), labeled.sum()),
        "utility": utility(o[labeled], p1[labeled], t[labeled], w) if labeled.any() else float("nan"),
    }


def outcome_counts(idx, truth, outcome, label: str) -> dict:
    """How many crops with a given truth label got each outcome (for not_a_bottle and cannot_tell)."""
    idx = np.asarray(idx, int)
    sel = truth[idx] == label
    return {k: int((outcome[idx][sel] == k).sum()) for k in (AUTO, REVIEW, UNKNOWN)} | {"n": int(sel.sum())}


def confusion(idx, truth, outcome, top1_product, products: list[str]) -> tuple[list[str], list[str], np.ndarray]:
    """Rows: truth (6 products + unknown). Columns: the automatic product answer (6), `review`, `unknown`."""
    idx = np.asarray(idx, int)
    rows, cols = products + ["unknown"], products + [REVIEW, UNKNOWN]
    m = np.zeros((len(rows), len(cols)), int)
    for t, o, p in zip(truth[idx], outcome[idx], top1_product[idx]):
        if t not in rows:
            continue
        col = cols.index(p) if o == AUTO else cols.index(o)
        m[rows.index(t), col] += 1
    return rows, cols, m
