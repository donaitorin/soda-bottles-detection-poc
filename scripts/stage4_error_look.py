"""Stage 4 error look: where does the Faster R-CNN baseline disagree with the Grounding DINO pseudo-labels on val?

Draws seeded random val photos plus the ones listed in the config (the can-only photo) into the gitignored
data/review/stage4_errors/: green = agrees with a pseudo-label (IoU >= 0.5), red = Faster R-CNN box with no pseudo-label
("extra"), blue = pseudo-label the Faster R-CNN missed. Also prints a breakdown of the disagreements by type.
Remember: the pseudo-labels have errors of their own, so an "extra" or a "miss" is not necessarily the Faster R-CNN's fault.

Usage: python scripts/stage4_error_look.py
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.images import load_image  # noqa: E402
from src.data.paths import check_output_dir  # noqa: E402
from src.evaluation.detection import ignore_below_floor  # noqa: E402
from src.prelabel.filters import box_iou  # noqa: E402


def match(gt: np.ndarray, boxes: np.ndarray, scores: np.ndarray, iou_thr: float):
    """Greedy matching by descending score (same rule as COCOeval). Returns (matched gt mask, matched pred mask, iou matrix)."""
    iou = box_iou(boxes, gt) if len(boxes) and len(gt) else np.zeros((len(boxes), len(gt)))
    gt_hit, pred_hit = np.zeros(len(gt), bool), np.zeros(len(boxes), bool)
    for i in np.argsort(-scores):
        free = np.where(~gt_hit & (iou[i] >= iou_thr))[0] if len(gt) else []
        if len(free):
            j = free[np.argmax(iou[i, free])]
            gt_hit[j], pred_hit[i] = True, True
    return gt_hit, pred_hit, iou


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage4.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load((REPO_ROOT / args.config).read_text())
    pcfg = yaml.safe_load((REPO_ROOT / cfg["prelabel_config"]).read_text())
    s1 = yaml.safe_load((REPO_ROOT / "configs/stage1.yaml").read_text())
    el = cfg["error_look"]
    run = next(r for r in cfg["runs"] if r["name"] == el["run"])
    gt = json.loads((REPO_ROOT / cfg["gt"]).read_text())
    sizes = {im["photo_id"]: (im["width"], im["height"]) for im in gt["images"]}
    pid_of = {im["id"]: im["photo_id"] for im in gt["images"]}
    gt_boxes = {p: [] for p in sizes}
    for a in gt["annotations"]:
        x, y, w, h = a["bbox"]
        gt_boxes[pid_of[a["image_id"]]].append([x, y, x + w, y + h])
    raw = json.loads((REPO_ROOT / cfg["predictions_dir"] / run["file"]).read_text())["variants"][run["variant"]]
    preds = {p: (np.array(raw[p]["boxes"], float), np.array(raw[p]["scores"], float)) for p in sizes}
    preds, _ = ignore_below_floor(preds, sizes, pcfg["filters"]["min_height_fraction"])
    thr = run["operating_threshold"]
    iou_thr = cfg["iou_for_precision_recall"]

    tally = {"tp": 0, "extra_no_overlap": 0, "extra_poor_overlap": 0, "miss_no_prediction": 0, "miss_poor_overlap": 0,
             "miss_small_under_3pct": 0, "miss_not_small": 0}
    per_photo, drawn = {}, {}
    for pid in sorted(sizes):
        boxes, scores = preds[pid]
        keep = scores >= thr
        boxes, scores = boxes[keep], scores[keep]
        g = np.array(gt_boxes[pid], float).reshape(-1, 4)
        gt_hit, pred_hit, iou = match(g, boxes, scores, iou_thr)
        tally["tp"] += int(pred_hit.sum())
        for i in np.where(~pred_hit)[0]:
            best = iou[i].max() if len(g) else 0.0
            tally["extra_poor_overlap" if best >= 0.1 else "extra_no_overlap"] += 1
        for j in np.where(~gt_hit)[0]:
            best = iou[:, j].max() if len(boxes) else 0.0
            tally["miss_poor_overlap" if best >= 0.1 else "miss_no_prediction"] += 1
            small = (g[j, 3] - g[j, 1]) < 0.03 * sizes[pid][1]
            tally["miss_small_under_3pct" if small else "miss_not_small"] += 1
        per_photo[pid] = (int((~pred_hit).sum()), int((~gt_hit).sum()), len(g))
        drawn[pid] = (g, boxes, gt_hit, pred_hit)

    print("AGREEMENT with Grounding DINO pseudo-labels (not accuracy), Faster R-CNN at score >= %.2f, IoU >= %.1f, val" % (thr, iou_thr))
    for k, v in tally.items():
        print(f"  {k}: {v}")
    print("photos with the most disagreements (extra, missed, pseudo-label boxes):")
    for pid, (fp, fn, n) in sorted(per_photo.items(), key=lambda kv: -(kv[1][0] + kv[1][1]))[:6]:
        print(f"  {pid}: {fp} extra, {fn} missed, {n} pseudo-label boxes")

    rng = random.Random(cfg["seed"])
    chosen = list(dict.fromkeys(el["include"] + rng.sample(sorted(sizes), el["n_photos"])))
    names = {r["photo_id"]: r["file_name"] for r in csv.DictReader((REPO_ROOT / s1["id_maps_dir"] / "photo_ids.csv").open())}
    out = REPO_ROOT / el["output_dir"]
    check_output_dir(out, [REPO_ROOT / d for d in s1["protected_dirs"]])
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.jpg"):
        old.unlink()
    for pid in chosen:
        g, boxes, gt_hit, pred_hit = drawn[pid]
        image = load_image(REPO_ROOT / s1["raw_dir"] / names[pid])
        d = ImageDraw.Draw(image)
        w = max(2, image.height // 300)
        for box, hit in zip(g, gt_hit):
            if not hit:
                d.rectangle(list(box), outline=(0, 90, 255), width=w + 1)       # blue: missed by Faster R-CNN
        for box, hit in zip(boxes, pred_hit):
            d.rectangle(list(box), outline=(0, 200, 0) if hit else (255, 0, 0), width=w if not hit else max(1, w // 2))
        image.thumbnail((el["max_side_px"],) * 2)
        image.save(out / f"{pid}.jpg", quality=85)
    print(f"drew {len(chosen)} photos in {out.relative_to(REPO_ROOT)}/ (gitignored): {', '.join(chosen)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
