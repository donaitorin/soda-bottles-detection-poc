"""Second step: does adding owner-labeled TRAIN shelf crops to the catalog help? Three paired comparisons on the same 180 val crops.

DEVELOPMENT numbers (agreement with the owner's labels on 180 val crops from 6 stores), not accuracy; the test split is not touched.
Thresholds are tuned LEAVING ONE STORE OUT (as in Stage 6b); intervals are paired bootstraps that resample whole val STORES.
Catalogs (src/embeddings/shelf_catalog.py): originals (30 studio photos) | originals+shelf | originals+shelf+negatives | shelf only.
  (a) originals            vs originals+shelf
  (b) originals+shelf      vs originals+shelf+negatives (crops the owner labeled unknown, as negative entries); KEPT only if it improves
      strict unknown rejection by at least 3 points without hurting known products (point estimates of top-1, brand and
      automatic-correct not below -2 points, and no paired interval entirely below zero for them). Pre-declared here, before any number is seen.
  (c) originals+shelf      vs shelf only (what the studio photos still contribute)
Class balance: shelf entries are capped per product and the k-NN vote is weighted by 1/entries of the product (configs/stage6.yaml
`shelf_round.catalog`); a sensitivity table shows the same comparison (a) without the cap and without the weighting.

Usage: python scripts/evaluate_shelf_catalog.py
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.paths import check_output_dir  # noqa: E402
from src.embeddings.eval_data import load_originals, load_shelf, load_val_queries  # noqa: E402
from src.embeddings.id_metrics import summarize  # noqa: E402
from src.embeddings.identify import features, leave_one_store_out  # noqa: E402
from src.embeddings.shelf_catalog import assemble  # noqa: E402
from src.evaluation.detection import bootstrap_by_store, difference_metric  # noqa: E402

NOTE = "DEVELOPMENT numbers on 180 val crops from 6 stores (leave-one-store-out), agreement with the owner's labels, not accuracy"
KEYS = [("top1_acc", "Top-1 (known crops)"), ("top3_acc", "Top-3 (known crops)"), ("brand_acc", "Brand accuracy (known crops)"),
        ("auto_correct_rate", "Known crops answered automatically and correctly"), ("unknown_rejection", "Unknown rejection (unknown or review)"),
        ("unknown_strict_rejection", "Unknown rejection (strictly unknown)"), ("false_known", "False known rate"),
        ("confident_wrong_rate", "Confident wrong rate (among automatic)"), ("share_automatic", "Share automatic"), ("utility", "Mean utility")]
MODES = {"originals": "originals", "a": "originals+shelf", "b": "originals+shelf+negatives", "c": "shelf"}


def pct(v):
    return "n/a" if v != v else f"{100 * v:.1f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage6.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load((REPO_ROOT / args.config).read_text())
    s1 = yaml.safe_load((REPO_ROOT / "configs/stage1.yaml").read_text())
    protected = [REPO_ROOT / d for d in s1["protected_dirs"]]
    products, idc, brand_of, sc, bs = cfg["products"], cfg["identify"], cfg["brands"], cfg["shelf_round"]["catalog"], cfg["bootstrap"]
    print(f"ALL NUMBERS BELOW ARE {NOTE.upper()}.\n")
    val, orig, shelf = load_val_queries(cfg, REPO_ROOT), load_originals(cfg, REPO_ROOT), load_shelf(cfg, REPO_ROOT)
    truth, stores, ids = val["truth"], val["stores"], list(val["crop_ids"])
    store_of, index_of = dict(zip(ids, stores)), {c: i for i, c in enumerate(ids)}
    lab = Counter(shelf["label"])
    print(f"{len(ids)} labeled val crops. Shelf entries labeled from TRAIN photos ({len(shelf['label'])}, from stores {sorted(map(str, set(shelf['store'])))}): "
          + ", ".join(f"{k} {lab[k]}" for k in products + ["unknown"]))

    def run(mode: str, cap: int, cap_unknown: int, balance: bool):
        cat = assemble(orig, shelf, mode, cap, cap_unknown, cfg["seed"])
        feat = features(val["vec"], cat["vec"], cat["product"], cat["photo"], products, idc["top_k_photos"], balance=balance)
        outcome, used = leave_one_store_out(feat, truth, stores, idc)
        top1 = np.array(products, dtype=object)[feat["top1"]]
        top3 = [[products[j] for j in row] for row in feat["top3"]]
        fn = lambda sel: summarize([index_of[c] for c in sel], truth, outcome, top1, top3, brand_of, products, idc["utility"])  # noqa: E731
        return {"feat": feat, "outcome": outcome, "top1": top1, "fn": fn, "n_entries": len(cat["product"]), "per_product": Counter(cat["product"])}

    cap, capu, bal = sc["cap_per_product"], sc["cap_unknown"], sc["balance_vote"]
    runs = {k: run(m, cap, capu, bal) for k, m in MODES.items()}
    pts = {k: r["fn"](ids) for k, r in runs.items()}
    print("\nCatalog sizes (entries per product after the cap of %d per product, %d unknown): " % (cap, capu))
    for k, r in runs.items():
        pp = r["per_product"]
        extra = f", unknown {pp['unknown']}" if pp.get("unknown") else ""
        print(f"- {MODES[k]}: {r['n_entries']} entries (" + ", ".join(f"{p} {pp.get(p, 0)}" for p in products) + extra + ")")

    print("\n### Does the system favour the products with more entries?")
    print("Share of ALL labeled val crops whose nearest product is each product, against the share of the owner's labels (known crops; unknown crops go to some product by force):")
    print("| Product | Entries in originals+shelf | Share of known crops (labels) | Share of nearest-product answers on known crops | Same without cap and balance |")
    print("|---|---:|---:|---:|---:|")
    raw = run("originals+shelf", 10 ** 6, 10 ** 6, False)
    known = np.isin(truth, products)
    for p in products:
        print(f"| {p} | {runs['a']['per_product'].get(p, 0)} | {pct((truth[known] == p).mean())}% | {pct((runs['a']['top1'][known] == p).mean())}% | {pct((raw['top1'][known] == p).mean())}% |")

    print("\nTop-1 correct per product (COUNTS of known val crops; a percentage only from 10 crops), and where the wrong ones go with originals + shelf:")
    print("| Product | Crops | originals | originals+shelf | wrong ones go to (originals+shelf) |\n|---|---:|---:|---:|---|")
    for p in products:
        sel = np.where(truth == p)[0]
        wrong = Counter(runs["a"]["top1"][i] for i in sel if runs["a"]["top1"][i] != p)
        print(f"| {p} | {len(sel)} | {int((runs['originals']['top1'][sel] == p).sum())} | {int((runs['a']['top1'][sel] == p).sum())} | " + (", ".join(f"{k} {v}" for k, v in wrong.most_common()) or "-") + " |")
    nb = ((runs["a"]["outcome"] == "automatic") & np.isin(truth, products + ["unknown"])).sum()
    ok = ((runs["a"]["outcome"] == "automatic") & (runs["a"]["top1"] == truth)).sum()
    print(f"Automatic answers with originals + shelf: {int(nb)} labeled crops, {int(ok)} correct.")

    def compare(title, base, other):
        diff = difference_metric(runs[other]["fn"], runs[base]["fn"])
        dpoint, dci = diff(ids), bootstrap_by_store(store_of, diff, bs["n"], cfg["seed"], bs["level"])
        print(f"\n### {title}\n| Metric (%) | {MODES[base]} | {MODES[other]} | difference [95% CI by store] | Excludes zero |\n|---|---:|---:|---|---|")
        out = {}
        for k, name in KEYS:
            lo, hi = dci[k]
            scale, f = (1, lambda v: f"{v:.3f}") if k == "utility" else (100, pct)
            print(f"| {name} | {f(pts[base][k])} | {f(pts[other][k])} | {scale * dpoint[k]:+.1f} [{scale * lo:+.1f}, {scale * hi:+.1f}] | {'yes' if lo > 0 or hi < 0 else 'no'} |")
            out[k] = {"diff": dpoint[k], "ci": dci[k]}
        return out

    ca = compare("(a) originals only vs originals + shelf examples", "originals", "a")
    cb = compare("(b) originals + shelf vs the same with the crops labeled unknown added as negative entries", "a", "b")
    cc_ = compare("(c) originals + shelf vs shelf examples only (what the studio photos still contribute)", "a", "c")
    gain = 100 * cb["unknown_strict_rejection"]["diff"]
    hurt = [k for k in ("top1_acc", "brand_acc", "auto_correct_rate") if 100 * cb[k]["diff"] < -2 or cb[k]["ci"][1] < 0]
    keep_b = gain >= 3 and not hurt
    print(f"\nVerdict (b), by the rule declared above: strict unknown rejection {gain:+.1f} points; known metrics hurt: {hurt or 'none'} -> negatives {'KEPT' if keep_b else 'NOT kept'}.")
    print("\n### Sensitivity of (a) to the class-balance choices (point estimates)")
    print("| Setting | Top-1 | Brand | Strict unknown rejection | False known |\n|---|---:|---:|---:|---:|")
    for name, r in (("cap + balanced vote (chosen)", runs["a"]), ("no cap, plain vote", raw), ("cap only", run("originals+shelf", cap, capu, False)), ("balanced vote only", run("originals+shelf", 10 ** 6, 10 ** 6, True))):
        m = r["fn"](ids)
        print(f"| {name} | {pct(m['top1_acc'])} | {pct(m['brand_acc'])} | {pct(m['unknown_strict_rejection'])} | {pct(m['false_known'])} |")

    out = REPO_ROOT / "data/results/stage6b_shelf/results.json"
    check_output_dir(out.parent, protected)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"note": NOTE, "points": pts, "a": ca, "b": cb, "c": cc_, "keep_negatives": bool(keep_b)}, indent=1, default=float))
    print(f"\nresults written to {out.relative_to(REPO_ROOT)} (gitignored)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
