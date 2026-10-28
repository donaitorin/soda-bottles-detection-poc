"""Variant B = variant A + OCR evidence: options per weight setting, tuning and leave-one-store-out (development numbers).

Each option is a weight setting (w_brand, w_variant, w_unknown, contradiction switch). The thresholds t_unk / t_auto are tuned for each option on the
OTHER stores, the option with the best tuning utility is chosen (strict improvement only: ties keep the earlier option, and the grid starts
with all zeros = variant A), and both are applied to the held-out store.
"""
import itertools

import numpy as np

from src.embeddings.experiments import decide_prob, prob_features
from src.embeddings.identify import AUTO, REVIEW, UNKNOWN, utility
from src.ocr.combine import combine, contradicts


def weight_grid(grid: dict) -> list[dict]:
    keys = ["w_brand", "w_variant", "w_unknown", "contradiction"]
    return [dict(zip(keys, v)) for v in itertools.product(*(grid[k] for k in keys))]       # the first is all zeros: variant A


def build_option(proba, products, evidence, w, ev_cfg, other_words, words_of) -> dict:
    p, infos = combine(proba, products, evidence, w, ev_cfg, other_words)
    feat = prob_features(p, products)
    top1 = np.array(products, dtype=object)[feat["top1"]]
    return {"w": w, "proba": p, "feat": feat, "infos": infos, "top1": top1, "contra": contradicts(top1, evidence, words_of)}


def decide_b(opt: dict, idx, t_unk: float, t_auto: float) -> np.ndarray:
    feat = {k: (v[idx] if isinstance(v, np.ndarray) else v) for k, v in opt["feat"].items()}
    out = decide_prob(feat, t_unk, t_auto)
    if opt["w"]["contradiction"]:
        out[(out == AUTO) & opt["contra"][idx]] = REVIEW
    return out


def tune_b(opt: dict, idx, truth, t_grid: dict, wutil: dict) -> dict:
    ts = np.round(np.arange(t_grid["start"], t_grid["stop"] + 1e-9, t_grid["step"]), 4)
    best, best_u = None, -1e9
    for t_unk in ts:
        for t_auto in ts:
            if t_auto < t_unk:
                continue
            u = utility(decide_b(opt, idx, t_unk, t_auto), opt["top1"][idx], truth[idx], wutil)
            if u > best_u + 1e-12:
                best, best_u = {"t_unk": float(t_unk), "t_auto": float(t_auto)}, u
    return {**best, "utility": best_u}


def choose_and_tune(options: list[dict], idx, truth, t_grid, wutil) -> tuple[dict, dict]:
    best, best_th = None, None
    for opt in options:
        th = tune_b(opt, idx, truth, t_grid, wutil)
        if best is None or th["utility"] > best_th["utility"] + 1e-12:
            best, best_th = opt, th
    return best, best_th


def loso_b(options, truth, stores, products, t_grid, wutil) -> dict:
    """Pooled held-out results: outcome, top1, top3 per crop, the rules/words that fired, and what was used per store."""
    n = len(truth)
    labeled = np.isin(truth, products + ["unknown"])
    outcome = np.full(n, UNKNOWN, dtype=object)
    top1 = np.empty(n, dtype=object)
    top3 = [None] * n
    infos = [None] * n
    used = []
    for s in sorted(set(stores)):
        tune_idx = np.where(labeled & (stores != s))[0]
        opt, th = choose_and_tune(options, tune_idx, truth, t_grid, wutil)
        test = np.where(stores == s)[0]
        outcome[test] = decide_b(opt, test, th["t_unk"], th["t_auto"])
        for i in test:
            top1[i] = opt["top1"][i]
            top3[i] = [products[j] for j in opt["feat"]["top3"][i]]
            infos[i] = opt["infos"][i]
        used.append({"store": s, **opt["w"], **th})
    return {"outcome": outcome, "top1": top1, "top3": top3, "infos": infos, "used": used}


def status(truth, outcome, top1, products) -> np.ndarray:
    """Per crop: "right", "wrong" or "neutral" (review) for labeled crops, "-" for the others (see the keep-rule in configs/stage7.yaml)."""
    out = np.full(len(truth), "-", dtype=object)
    for i, (t, o, p) in enumerate(zip(truth, outcome, top1)):
        if t in products:
            out[i] = "neutral" if o == REVIEW else ("right" if o == AUTO and p == t else "wrong")
        elif t == "unknown":
            out[i] = "neutral" if o == REVIEW else ("right" if o == UNKNOWN else "wrong")
    return out
