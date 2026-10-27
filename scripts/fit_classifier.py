"""Fit the adopted scorer ONCE on all owner-labeled TRAIN shelf entries and save it with a model card.

Model: logistic regression (scikit-learn: L2, multinomial, class_weight balanced, standardised features) on the DINOv2 CLS embedding of the
10% margin crops; classes = the 6 products + `unknown`. `C` is chosen by leaving one TRAIN store out on the shelf entries (pooled
accuracy; ties go to the smaller C). The demo thresholds (unknown / automatic probability thresholds and the candidate-list distance) are tuned ONCE
on all 180 labeled val crops, so they are OPTIMISTIC demo values, not an evaluation (the honest numbers are the leave-one-store-out ones of
scripts/evaluate_stage6_gate.py). Output (ignored): models/stage6/logreg_cls.npz and logreg_cls_card.json. The values printed at the end are copied by
hand into configs/stage6.yaml (`demo.classifier`).

Usage: python scripts/fit_classifier.py
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.paths import check_output_dir  # noqa: E402
from src.embeddings.answers import conformal_delta, nonconformity  # noqa: E402
from src.embeddings.classifier import LinearClassifier  # noqa: E402
from src.embeddings.eval_data import load_shelf, load_val_queries  # noqa: E402
from src.embeddings.experiments import decide_prob, prob_features, tune_prob  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage6.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load((REPO_ROOT / args.config).read_text())
    s1 = yaml.safe_load((REPO_ROOT / "configs/stage1.yaml").read_text())
    products, lc, cc = cfg["products"], cfg["experiments"]["logreg"], cfg["classifier"]
    classes = products + ["unknown"]
    shelf, val = load_shelf(cfg, REPO_ROOT), load_val_queries(cfg, REPO_ROOT)
    y = np.array([classes.index(l) for l in shelf["label"]])
    s_store = shelf["store"]

    acc = {}
    for c in lc["c_grid"]:
        folds = []
        for s in sorted(set(s_store)):
            tr, te = s_store != s, s_store == s
            folds.append((LinearClassifier.fit(shelf["vec"][tr], y[tr], classes, c).predict_proba(shelf["vec"][te]).argmax(axis=1) == y[te]).mean())
        acc[c] = float(np.mean(folds))
    best_c = max(acc, key=lambda c: (round(acc[c], 6), -c))
    print("leave-one-TRAIN-store-out accuracy on the shelf entries by C: " + ", ".join(f"{c}: {100 * a:.1f}%" for c, a in acc.items()) + f" -> C = {best_c}")
    model = LinearClassifier.fit(shelf["vec"], y, classes, best_c)

    proba = model.predict_proba(val["vec"])
    feat = prob_features(proba, products)
    labeled = np.where(np.isin(val["truth"], products + ["unknown"]))[0]
    sub = {k: (v[labeled] if isinstance(v, np.ndarray) else v) for k, v in feat.items()}
    th = tune_prob(sub, val["truth"][labeled], lc["t_grid"], cfg["identify"]["utility"])
    known = np.where(np.isin(val["truth"], products))[0]
    delta = conformal_delta(nonconformity(feat["prod_sim"][known], np.array([products.index(val["truth"][i]) for i in known])), cfg["answers"]["coverage"])
    out = decide_prob(feat, th["t_unk"], th["t_auto"])
    print(f"demo thresholds tuned on all {len(labeled)} labeled val crops (OPTIMISTIC): t_unk {th['t_unk']}, t_auto {th['t_auto']}; probability-list delta {delta:.4f} ({len(known)} known crops)")
    print("outcomes on ALL 180 val crops with them: " + ", ".join(f"{k} {int((out == k).sum())}" for k in ("automatic", "review", "unknown")))

    model_dir = REPO_ROOT / cc["model_dir"]
    check_output_dir(model_dir, [REPO_ROOT / d for d in s1["protected_dirs"]])
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_dir / f"{cc['name']}.npz")
    labels_file = REPO_ROOT / cfg["labels_export"]["output_dir"] / f"{cfg['shelf_round']['name']}_labels.csv"
    card = {"model": "LogisticRegression on DINOv2 CLS embeddings (10% margin crops), classes = 6 products + unknown", "date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "training_data": {"source": "owner-labeled TRAIN shelf crops (project soda-bottles-crops-train), not_a_bottle and cannot_tell left out", "n_crops": int(len(y)),
                              "per_label": {k: int(v) for k, v in sorted(Counter(shelf["label"]).items())}, "stores": sorted(map(str, set(s_store))),
                              "labels_file_sha256": hashlib.sha256(labels_file.read_bytes()).hexdigest()},
            "settings": {"C": best_c, "class_weight": "balanced", "penalty": "l2", "solver": "lbfgs", "standardised_features": True, "seed": cfg["seed"], "randomness": "none (deterministic solver)",
                         "c_chosen_by": "leave one TRAIN store out, pooled accuracy", "c_grid_accuracy": {str(c): round(a, 4) for c, a in acc.items()}},
            "embedding": {"model_id": cfg["embeddings"]["model_id"], "revision": cfg["embeddings"]["revision"], "pooling": "cls", "crop_margin": cfg["crops"]["margin"]},
            "thresholds": {"t_unk": th["t_unk"], "t_auto": th["t_auto"], "candidate_list_delta": round(delta, 6), "max_list": cc.get("max_list", 3),
                           "note": "DEMO thresholds tuned once on all 180 labeled val crops: optimistic, not an evaluation"},
            "versions": {"scikit-learn": sklearn.__version__, "numpy": np.__version__}}
    (model_dir / f"{cc['name']}_card.json").write_text(json.dumps(card, indent=1))
    print(f"saved {cc['model_dir']}/{cc['name']}.npz and {cc['name']}_card.json (gitignored)")
    print(f"copy into configs/stage6.yaml demo.classifier: thresholds {{t_unk: {th['t_unk']}, t_auto: {th['t_auto']}}}, candidates {{delta: {delta:.4f}}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
