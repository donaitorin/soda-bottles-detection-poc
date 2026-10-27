"""Tune the thresholds of the demo tool ONCE on all 180 labeled val crops (no leave-one-store-out) and print them.

These are DEMO thresholds, not an evaluation result: they are tuned on the same crops they would be measured on, so any number computed
with them on val is optimistic. Same utility and grid as the Stage 6b rules; catalog (b) when `catalog.use_shelf` (originals + shelf entries + unknown negatives, as in the database), CLS embeddings.
The values are then copied by hand into configs/stage6.yaml (`demo.thresholds`).

Usage: python scripts/tune_demo_thresholds.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.embeddings.answers import brand_info, calibrate_bm, conformal_delta, nonconformity  # noqa: E402
from src.embeddings.eval_data import load_originals, load_shelf, load_val_queries  # noqa: E402
from src.embeddings.shelf_catalog import assemble  # noqa: E402
from src.embeddings.identify import AUTO, REVIEW, UNKNOWN, decide, features, tune  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage6.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load((REPO_ROOT / args.config).read_text())
    products, idc = cfg["products"], cfg["identify"]
    val, orig = load_val_queries(cfg, REPO_ROOT), load_originals(cfg, REPO_ROOT)
    truth, q = val["truth"], val["vec"]
    sc = cfg["shelf_round"]["catalog"]
    mode = "originals+shelf+negatives" if cfg["catalog"].get("use_shelf") else "originals"
    cat = assemble(orig, load_shelf(cfg, REPO_ROOT) if mode != "originals" else None, mode, sc["cap_per_product"], sc["cap_unknown"], cfg["seed"])
    feat = features(q, cat["vec"], cat["product"], cat["photo"], products, idc["top_k_photos"], balance=sc["balance_vote"] and mode != "originals")
    labeled = np.where(np.isin(truth, products + ["unknown"]))[0]
    sub = {k: (v[labeled] if isinstance(v, np.ndarray) else v) for k, v in feat.items()}
    th = tune(sub, truth[labeled], idc)
    out = decide(feat, th["t_sim"], th["t_margin"], th["t_vote"], th["t_neg"])
    print(f"demo thresholds tuned on all {len(labeled)} labeled crops (products and unknown): {th}")
    known = np.where(np.isin(truth, products))[0]
    true_idx = np.array([products.index(truth[i]) for i in known])
    delta = conformal_delta(nonconformity(feat["prod_sim"][known], true_idx), cfg["answers"]["coverage"])
    brand_idx, margin, brands = brand_info(feat["prod_sim"][known], products, cfg["brands"])
    true_brand = np.array([brands.index(cfg["brands"][truth[i]]) for i in known])
    ac = cfg["answers"]
    t_bm = calibrate_bm(margin, brand_idx == true_brand, ac["brand_margin_grid"], ac["brand_precision_target"], ac["min_answered"])
    print(f"demo candidates calibrated on the {len(known)} known crops (optimistic): delta {delta:.4f}, t_bm {t_bm}  -> configs/stage6.yaml demo.candidates {{delta, t_bm, max_list}}")
    print("outcomes on ALL 180 crops with them (optimistic, tuned on the same crops): " + ", ".join(f"{k} {int((out == k).sum())}" for k in (AUTO, REVIEW, UNKNOWN)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
