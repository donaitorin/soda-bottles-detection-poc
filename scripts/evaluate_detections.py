"""Evaluate saved predictions against a COCO ground-truth file (reusable by later stages).

For every run in configs/stage4.yaml: predictions under the 1.5% height floor are ignored (removed, counted), then
AP@0.5:0.95, AP@0.5 and recall at IoU 0.5 (all scores) plus precision / recall / F1 at the operating score
threshold, each with a 95% bootstrap interval resampled by STORE, and a breakdown per store and per box-count
bin. It also checks the own accumulation against pycocotools' COCOeval on the real data.

THE GROUND TRUTH IS GROUNDING DINO PSEUDO-LABELS: every number is agreement with Grounding DINO, not accuracy.

Usage: python scripts/evaluate_detections.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.paths import check_output_dir  # noqa: E402
from src.evaluation.detection import (bootstrap_by_store, difference_metric, gt_counts, ignore_below_floor, operating_point,  # noqa: E402
                                      per_photo_results, summarize)

NOTE = "agreement with Grounding DINO pseudo-labels, NOT accuracy"


def fmt(value, interval=None, pct=True):
    scale = 100 if pct else 1
    text = f"{value * scale:.1f}" if value == value else "n/a"
    return f"{text} [{interval[0] * scale:.1f}, {interval[1] * scale:.1f}]" if interval else text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage4.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load((REPO_ROOT / args.config).read_text())
    pcfg = yaml.safe_load((REPO_ROOT / cfg["prelabel_config"]).read_text())
    s1 = yaml.safe_load((REPO_ROOT / "configs/stage1.yaml").read_text())
    floor, max_dets = pcfg["filters"]["min_height_fraction"], cfg["max_dets"]
    pred_dir = REPO_ROOT / cfg["predictions_dir"]
    check_output_dir(pred_dir, [REPO_ROOT / d for d in s1["protected_dirs"]])
    gt = json.loads((REPO_ROOT / cfg["gt"]).read_text())
    sizes = {im["photo_id"]: (im["width"], im["height"]) for im in gt["images"]}
    store_of = {im["photo_id"]: im["store_code"] for im in gt["images"]}
    counts = gt_counts(gt)
    order = sorted(counts, key=lambda p: (counts[p], p))
    bins = [order[i * len(order) // cfg["density_bins"]:(i + 1) * len(order) // cfg["density_bins"]] for i in range(cfg["density_bins"])]
    print(f"Ground truth: {cfg['gt']} ({len(sizes)} photos, {len(gt['annotations'])} boxes, stores {sorted(set(store_of.values()))})")
    print(f"ALL NUMBERS BELOW ARE {NOTE.upper()}.\n")

    all_results, metric_fns = {}, {}
    for run in cfg["runs"]:
        if not (pred_dir / run["file"]).exists():
            print(f"### {run['label']}: SKIPPED, {run['file']} not found (run scripts/predict_val.py first)\n")
            continue
        raw = json.loads((pred_dir / run["file"]).read_text())["variants"][run["variant"]]
        preds = {pid: (np.array(raw[pid]["boxes"], float), np.array(raw[pid]["scores"], float)) for pid in sizes}
        preds, n_floor = ignore_below_floor(preds, sizes, floor)
        n_total = sum(len(v[0]) for v in preds.values())
        op_thr = run["operating_threshold"]
        at_op = {pid: (b[s >= op_thr], s[s >= op_thr]) for pid, (b, s) in preds.items()}
        res_all, reference = per_photo_results(gt, preds, max_dets)
        res_op, _ = per_photo_results(gt, at_op, max_dets)

        def metrics(photos, res_all=res_all, res_op=res_op):
            m = summarize([res_all[p] for p in photos], max_dets)
            op = operating_point([res_op[p] for p in photos])
            return {**m, "precision": op["precision"], "recall": op["recall"], "f1": op["f1"]}

        metric_fns[run["name"]] = metrics
        photos = sorted(sizes)
        point = metrics(photos)
        check = max(abs(point["AP"] - reference[0]), abs(point["AP50"] - reference[1]))
        ci = bootstrap_by_store(store_of, metrics, cfg["bootstrap"]["n"], cfg["seed"], cfg["bootstrap"]["level"])
        op_counts = operating_point([res_op[p] for p in photos])
        print(f"### {run['label']}  ({NOTE})")
        print(f"predictions: {n_total} kept after ignoring {n_floor} under the 1.5% height floor; operating score threshold {op_thr}")
        print(f"check against pycocotools COCOeval: largest difference in AP / AP50 = {check:.2e}")
        print("| Metric (agreement, %) | Value [95% CI by store] |\n|---|---|")
        for key, name in (("AP", "AP@0.5:0.95"), ("AP50", "AP@0.5"), ("recall50", "Recall at IoU 0.5 (all scores)"),
                          ("precision", f"Precision at score >= {op_thr}, IoU 0.5"), ("recall", f"Recall at score >= {op_thr}, IoU 0.5"),
                          ("f1", f"F1 at score >= {op_thr}, IoU 0.5")):
            print(f"| {name} | {fmt(point[key], ci[key])} |")
        print(f"operating point counts: TP {op_counts['tp']}, FP {op_counts['fp']}, FN {op_counts['fn']}\n")
        stores, bin_rows = {}, []
        for s in sorted(set(store_of.values())):
            ph = [p for p in photos if store_of[p] == s]
            stores[s] = {"photos": len(ph), "gt_boxes": sum(counts[p] for p in ph), **metrics(ph)}
        for i, ph in enumerate(bins, 1):
            bin_rows.append({"bin": f"{i} ({counts[ph[0]]}-{counts[ph[-1]]} boxes)", "photos": len(ph), **metrics(ph)})
        all_results[run["name"]] = {"label": run["label"], "baseline": run["baseline"], "ignored_below_floor": n_floor,
                                    "kept": n_total, "point": point, "ci": ci, "counts": op_counts, "check_vs_cocoeval": check,
                                    "per_store": stores, "per_density": bin_rows}
        if run.get("breakdown", run["baseline"]):
            print(f"Per store ({NOTE}):\n| Store | Photos | GT boxes | AP@0.5 | Precision | Recall |\n|---|---:|---:|---:|---:|---:|")
            for s, v in stores.items():
                print(f"| {s} | {v['photos']} | {v['gt_boxes']} | {fmt(v['AP50'])} | {fmt(v['precision'])} | {fmt(v['recall'])} |")
            print(f"\nPer box-count bin ({NOTE}):\n| Bin | Photos | AP@0.5 | Precision | Recall |\n|---|---:|---:|---:|---:|")
            for v in bin_rows:
                print(f"| {v['bin']} | {v['photos']} | {fmt(v['AP50'])} | {fmt(v['precision'])} | {fmt(v['recall'])} |")
            print()
    paired_results = []
    for pair in cfg.get("paired", []):
        if pair["a"] not in metric_fns or pair["b"] not in metric_fns:
            continue
        diff = difference_metric(metric_fns[pair["a"]], metric_fns[pair["b"]])
        point = diff(sorted(sizes))
        ci = bootstrap_by_store(store_of, diff, cfg["bootstrap"]["n"], cfg["seed"], cfg["bootstrap"]["level"])
        names = {r["name"]: r["label"] for r in cfg["runs"]}
        print(f"### Paired difference by store: {pair['a']} minus {pair['b']}  ({NOTE})")
        print(f"a = {names[pair['a']]}; b = {names[pair['b']]}; the same resampled stores are used for both models")
        print("| Metric (percentage points) | Difference a - b [95% CI by store] | Interval excludes zero |\n|---|---|---|")
        for key, name in (("AP", "AP@0.5:0.95"), ("AP50", "AP@0.5"), ("recall50", "Recall at IoU 0.5 (all scores)"),
                          ("precision", "Precision, score >= 0.3"), ("recall", "Recall, score >= 0.3"), ("f1", "F1, score >= 0.3")):
            lo, hi = ci[key]
            print(f"| {name} | {fmt(point[key], ci[key])} | {'yes' if lo > 0 or hi < 0 else 'no'} |")
        print()
        paired_results.append({"a": pair["a"], "b": pair["b"], "point": point, "ci": ci})
    (pred_dir / "results.json").write_text(json.dumps({"note": NOTE, "runs": all_results, "paired": paired_results}, indent=1))
    print(f"results written to {(pred_dir / 'results.json').relative_to(REPO_ROOT)} (gitignored)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
