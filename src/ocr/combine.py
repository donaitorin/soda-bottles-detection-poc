"""How OCR evidence changes the classifier's probabilities (pipeline variant B). Rules fixed in reports/stage_07_proposal.md BEFORE any number.

Input per crop: the classifier probabilities over (products + unknown) and the keywords read (a dict word -> "exact" | "fuzzy"; empty when the
crop was not read or nothing was found). Crops without evidence keep their row EXACTLY (variant A).
 1. SPRITE / FANTA read: that product's logit += w_brand, if the classifier already gives it >= p_ctx.
 2. LIGHT / ZERO / RETORNABLE read: the logit of coca_light / coca_zero / coca_classic_returnable += w_variant, only with Coca-Cola context
    (total Coca probability >= p_ctx_coca, or COCA-COLA read).
 3. (hard rule, `contradicts`) an AUTOMATIC answer becomes review when a word of a different product is read.
 4. Not-a-target combinations (SPRITE or FANTA with ZERO or RETORNABLE; ZERO with RETORNABLE on a Coca-Cola crop) and the other-brand words
    BILZ / PEPSI: the `unknown` logit += w_unknown and rules 1 and 2 do not fire for that crop.
 5. Two different brand words (SPRITE, FANTA, COCA-COLA) on one crop: no boost; rule 3 then sends an automatic answer to review.
"""
import numpy as np

COCA = ["coca_classic", "coca_classic_returnable", "coca_zero", "coca_light"]
VARIANT_PRODUCT = {"LIGHT": "coca_light", "ZERO": "coca_zero", "RETORNABLE": "coca_classic_returnable"}
BRAND_PRODUCT = {"SPRITE": "sprite", "FANTA": "fanta"}


def combine(proba: np.ndarray, products: list[str], evidence: list[dict], w: dict, ev_cfg: dict, other_words: list[str]) -> tuple[np.ndarray, list[dict]]:
    """(new probabilities, one info dict per crop: {"rules": [...], "words": [fired words], "kinds": [their kinds]})."""
    out = proba.copy()
    infos = []
    coca_cols = [products.index(p) for p in COCA]
    unk = len(products)
    for i, ev in enumerate(evidence):
        info = {"rules": [], "words": [], "kinds": []}
        infos.append(info)
        if not ev:
            continue
        kws = set(ev)
        others = kws & set(other_words)
        coca_ctx = proba[i, coca_cols].sum() >= ev_cfg["p_ctx_coca"] or "COCA-COLA" in kws
        non_target = bool(kws & {"SPRITE", "FANTA"}) and bool(kws & {"ZERO", "RETORNABLE"}) or (coca_ctx and {"ZERO", "RETORNABLE"} <= kws)
        logit = np.log(np.clip(proba[i], 1e-9, 1.0))
        changed = False

        def fire(word, rule):
            info["rules"].append(rule)
            info["words"].append(word)
            info["kinds"].append(ev[word])
        if non_target or others:
            for word in sorted(others) or sorted(kws & {"SPRITE", "FANTA", "ZERO", "RETORNABLE"}):
                if w["w_unknown"] > 0:
                    fire(word, "not_a_target" if non_target else "other_brand")
            if w["w_unknown"] > 0:
                logit[unk] += w["w_unknown"]
                changed = True
        elif len(kws & {"SPRITE", "FANTA", "COCA-COLA"}) >= 2:
            info["rules"].append("two_brands")
        else:
            for word, product in BRAND_PRODUCT.items():
                if word in kws and proba[i, products.index(product)] >= ev_cfg["p_ctx"] and w["w_brand"] > 0:
                    logit[products.index(product)] += w["w_brand"]
                    fire(word, f"boost_{product}")
                    changed = True
            if coca_ctx and w["w_variant"] > 0:
                for word, product in VARIANT_PRODUCT.items():
                    if word in kws:
                        logit[products.index(product)] += w["w_variant"]
                        fire(word, f"boost_{product}")
                        changed = True
        if changed:
            e = np.exp(logit - logit.max())
            out[i] = e / e.sum()
    return out, infos


def contradicts(top1_products: np.ndarray, evidence: list[dict], words_of: dict) -> np.ndarray:
    """True where a keyword is read that the top-1 product does not allow (`words_of`: product -> words a correct bottle may show)."""
    return np.array([bool(set(ev) - set(words_of[p])) if ev else False for p, ev in zip(top1_products, evidence)])


def evidence_from_readings(readings: dict, ids, min_height: int, min_ratio: float, keywords_detail) -> list[dict]:
    """Evidence per crop id: only crops at least `min_height` px tall are read (shorter ones have no evidence)."""
    out = []
    for cid in ids:
        r = readings[str(cid)]
        out.append(keywords_detail(" ".join(t for t, _ in r["pieces"]), min_ratio) if r["height"] >= min_height else {})
    return out
