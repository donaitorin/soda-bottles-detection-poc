"""Stage 7 tasks 3 and 4: what does the OCR read on the owner-labeled TRAIN shelf entries, by crop height, and how fast.

TRAIN only: the minimum reading height and the word lists are chosen here, never on val. Rules for the choices are declared in
configs/stage7.yaml `reading_rate` BEFORE the numbers. Readings (text only, by crop id) are cached in data/ocr/stage7/ (gitignored).

Usage: python scripts/ocr_reading_rate.py [--reread]
"""
import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.common.device import get_device  # noqa: E402
from src.embeddings.crop_records import build_records, render  # noqa: E402
from src.embeddings.eval_data import load_shelf  # noqa: E402
from src.ocr.evidence import KEYWORDS, keywords_read, tokens  # noqa: E402
from src.ocr.reader import OcrReader, OcrUnavailable  # noqa: E402

CONFUSING = {"SPRITE", "FANTA", "LIGHT", "ZERO", "RETORNABLE"}        # words that decide a variant/brand (COCA-COLA never contradicts)


def read_all(cfg6, s1, c7) -> dict:
    shelf = load_shelf(cfg6, REPO_ROOT)
    ids = [str(c) for c in shelf["photo"]]
    records = build_records(cfg6, s1, REPO_ROOT, [], ids, [])["shelf"]
    images = render(records, "margin10", c7["ocr"]["margin"])
    device = get_device()
    reader = OcrReader(c7["ocr"], REPO_ROOT, device)
    reader.read(images[0])                                           # warm-up (first call on MPS includes setup)
    out, times = {}, []
    for cid, label, im in zip(ids, shelf["label"], images):
        t = time.perf_counter()
        pieces = reader.read(im)
        times.append(time.perf_counter() - t)
        out[cid] = {"label": str(label), "height": im.height, "width": im.width, "pieces": [[t_, round(c, 3)] for t_, c in pieces]}
    print(f"read {len(out)} crops on {device}: {np.mean(times):.3f} s per crop (median {np.median(times):.3f})")
    return out


def analyse(readings: dict, c7: dict) -> dict:
    rr, ratio = c7["reading_rate"], c7["ocr"]["fuzzy_min_ratio"]
    rows = []
    for cid, r in readings.items():
        text = " ".join(t for t, _ in r["pieces"])
        rows.append({"id": cid, "label": r["label"], "h": r["height"], "text": text, "kw": keywords_read(text, ratio), "toks": set(tokens(text))})
    expect = {k: set(v) for k, v in c7["expect"].items()}

    def wrong(row):                                                  # a decisive word the label does not allow (never for unknown labels)
        return row["label"] in expect and bool((row["kw"] & CONFUSING) - expect[row["label"]])
    edges = rr["height_buckets_px"]
    cumulative = []
    for h in edges:
        sub = [r for r in rows if r["h"] >= h]
        with_kw = [r for r in sub if r["kw"]]
        known_kw = [r for r in with_kw if r["label"] in expect]
        bad = [r for r in known_kw if wrong(r)]
        cumulative.append({"min_h": h, "n": len(sub), "any_text": sum(bool(r["text"].strip()) for r in sub), "any_kw": len(with_kw), "known_with_kw": len(known_kw),
                           "wrong": len(bad), "consistent": (1 - len(bad) / len(known_kw)) if known_kw else None,
                           "unknown_with_kw": sum(r["label"] == "unknown" for r in with_kw), "unknown_n": sum(r["label"] == "unknown" for r in sub)})
    chosen = next((c["min_h"] for c in cumulative if c["min_h"] > 0 and c["known_with_kw"] >= rr["min_reading_crops"]
                   and c["consistent"] is not None and c["consistent"] >= rr["min_consistent_share"]), None)
    return {"rows": rows, "cumulative": cumulative, "chosen": chosen, "expect": expect}


def target_word_rates(rows, expect, chosen):
    """Per product (crops at or above the chosen minimum): how many crops, in how many its decisive word was read."""
    target = {"coca_light": "LIGHT", "coca_zero": "ZERO", "coca_classic_returnable": "RETORNABLE", "sprite": "SPRITE", "fanta": "FANTA"}
    out = []
    for p in ["coca_classic", "coca_classic_returnable", "coca_zero", "coca_light", "sprite", "fanta", "unknown"]:
        sub = [r for r in rows if r["label"] == p and r["h"] >= chosen]
        out.append({"product": p, "n_all": sum(r["label"] == p for r in rows), "n_tall": len(sub), "target": target.get(p),
                    "target_read": sum(target[p] in r["kw"] for r in sub) if p in target else None,
                    "any_kw": sum(bool(r["kw"]) for r in sub),
                    "contradict": sum(bool((r["kw"] & CONFUSING) - expect[p]) for r in sub) if p in expect else None,
                    "any_brand_or_variant": sum(bool(r["kw"] & CONFUSING) for r in sub) if p == "unknown" else None})
    return out


def word_table(rows, chosen, c7):
    rr = c7["reading_rate"]
    sub = [r for r in rows if r["h"] >= chosen]
    labels = sorted({r["label"] for r in sub})
    per = {l: Counter(w for r in sub if r["label"] == l for w in r["toks"]) for l in labels}      # words counted once per crop
    top = {l: per[l].most_common(12) for l in labels}
    watch = {w: {l: per[l][w] for l in labels if per[l][w]} for w in rr["watch_words"]}
    proposals = []
    for l in labels:
        for w, n in per[l].items():
            if len(w) >= 4 and w.isalpha() and n >= rr["word_min_crops"] and all(per[o][w] == 0 for o in labels if o != l):
                proposals.append((l, w, n))
    return labels, top, watch, sorted(proposals, key=lambda x: (-x[2], x[0], x[1])), {l: sum(r["label"] == l for r in sub) for l in labels}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage7.yaml")
    parser.add_argument("--reread", action="store_true", help="read the crops again even if a cached reading exists")
    args = parser.parse_args()
    c7 = yaml.safe_load((REPO_ROOT / args.config).read_text())
    cfg6 = yaml.safe_load((REPO_ROOT / c7["config_stage6"]).read_text())
    s1 = yaml.safe_load((REPO_ROOT / "configs/stage1.yaml").read_text())
    out_dir = REPO_ROOT / c7["reading_rate"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "train_shelf_readings.json"
    try:
        if args.reread or not cache.exists():
            cache.write_text(json.dumps(read_all(cfg6, s1, c7)))
    except OcrUnavailable as error:
        raise SystemExit(str(error))
    readings = json.loads(cache.read_text())
    a = analyse(readings, c7)
    print(f"\nTRAIN shelf entries read: {len(a['rows'])} (labels: " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(r['label'] for r in a['rows']).items())) + ")")
    print("\nReading rate by minimum crop height (cumulative: crops at least this tall); consistent = share of keyword-reading known crops with no decisive word the label does not allow")
    print(f"{'min h':>6} {'crops':>6} {'any text':>9} {'any kw':>7} {'known+kw':>9} {'wrong':>6} {'consistent':>11} {'unknown+kw':>11}")
    for c in a["cumulative"]:
        cons = "-" if c["consistent"] is None else f"{c['consistent']:.0%}"
        print(f"{c['min_h']:>6} {c['n']:>6} {c['any_text']:>9} {c['any_kw']:>7} {c['known_with_kw']:>9} {c['wrong']:>6} {cons:>11} {c['unknown_with_kw']:>7}/{c['unknown_n']}")
    print(f"\nMINIMUM READING HEIGHT by the declared rule: {a['chosen']} px" if a["chosen"] else "\nNo height satisfies the declared rule: OCR is not usable on this data.")
    if a["chosen"] is None:
        return 0
    ch = a["chosen"]
    print(f"\nPer product, crops at least {ch} px tall (count of crops, never percentages on small groups):")
    print(f"{'product':>24} {'all':>4} {'tall':>5} {'decisive word':>14} {'read':>5} {'any kw':>7} {'contradicts':>12}")
    for t in target_word_rates(a["rows"], a["expect"], ch):
        extra = f"  any brand/variant word: {t['any_brand_or_variant']}" if t["any_brand_or_variant"] is not None else ""
        print(f"{t['product']:>24} {t['n_all']:>4} {t['n_tall']:>5} {str(t['target'] or '-'):>14} {str(t['target_read'] if t['target_read'] is not None else '-'):>5} {t['any_kw']:>7} {str(t['contradict'] if t['contradict'] is not None else '-'):>12}{extra}")
    print("\nDecisive word read, by minimum crop height (read / crops of the product at least that tall):")
    target = {"coca_light": "LIGHT", "coca_zero": "ZERO", "coca_classic_returnable": "RETORNABLE", "sprite": "SPRITE", "fanta": "FANTA"}
    hs = [0, 250, 300, 400, 500]
    print(f"{'product':>24} {'word':>11} " + " ".join(f"{'>=' + str(h):>9}" for h in hs))
    for p, w in target.items():
        cells = []
        for h in hs:
            sub = [r for r in a["rows"] if r["label"] == p and r["h"] >= h]
            cells.append(f"{sum(w in r['kw'] for r in sub)}/{len(sub)}")
        print(f"{p:>24} {w:>11} " + " ".join(f"{c:>9}" for c in cells))
    labels, top, watch, proposals, n_by = word_table(a["rows"], ch, c7)
    print(f"\nMost frequent normalised words per label (crops >= {ch} px; count = crops with the word):")
    for l in labels:
        print(f"  {l} ({n_by[l]} crops): " + ", ".join(f"{w} {n}" for w, n in top[l]))
    print("\nWatch words (crops per label):")
    for w, d in watch.items():
        print(f"  {w}: " + (", ".join(f"{l} {n}" for l, n in d.items()) if d else "not read"))
    print(f"\nExtra words by the declared rule (>= {c7['reading_rate']['word_min_crops']} crops of ONE label, 0 of any other):")
    print("  " + ("; ".join(f"{w} on {l} ({n})" for l, w, n in proposals) if proposals else "none"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
