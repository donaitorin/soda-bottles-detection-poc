"""Stage 5 error analysis on val: where does the fine-tuned detector disagree with the pseudo-labels, and why?

Causes are automatic proxies (see src/evaluation/errors.py): can (found with Grounding DINO and the prompt "can."),
fragment, partial overlap (occlusion or box extent), small bottle, other. Draws photos into the gitignored
data/review/stage5_errors/: green = agrees, red = extra prediction, blue = missed pseudo-label; the letter is the
cause (C can, F fragment, P partial overlap, S small, O other).
Everything is AGREEMENT with Grounding DINO pseudo-labels, not accuracy; a pseudo-label that is a can or a fragment
is a label ERROR, and a model that matches it has imitated the error.

Usage: python scripts/stage5_error_analysis.py
"""
import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.common.device import get_device  # noqa: E402
from src.data.images import load_image  # noqa: E402
from src.data.paths import check_output_dir  # noqa: E402
from src.evaluation.detection import ignore_below_floor  # noqa: E402
from src.evaluation.errors import CAUSES, categorize  # noqa: E402
from src.prelabel import grounding_dino  # noqa: E402

LETTER = {"can": "C", "fragment": "F", "partial overlap": "P", "small bottle": "S", "other": "O"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage5.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load((REPO_ROOT / args.config).read_text())
    ea = cfg["error_analysis"]
    pcfg = yaml.safe_load((REPO_ROOT / cfg["prelabel_config"]).read_text())
    s1 = yaml.safe_load((REPO_ROOT / "configs/stage1.yaml").read_text())
    protected = [REPO_ROOT / d for d in s1["protected_dirs"]]
    gt = json.loads((REPO_ROOT / cfg["data"]["val_coco"]).read_text())
    sizes = {im["photo_id"]: (im["width"], im["height"]) for im in gt["images"]}
    pid_of = {im["id"]: im["photo_id"] for im in gt["images"]}
    gt_boxes = {p: [] for p in sizes}
    for a in gt["annotations"]:
        x, y, w, h = a["bbox"]
        gt_boxes[pid_of[a["image_id"]]].append([x, y, x + w, y + h])
    names = {r["photo_id"]: r["file_name"] for r in csv.DictReader((REPO_ROOT / s1["id_maps_dir"] / "photo_ids.csv").open())}

    # Cans: Grounding DINO with the prompt "can." on the val photos (cached; photo ids only).
    cache = REPO_ROOT / ea["cache"]
    check_output_dir(cache.parent, protected)
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        cans = {p: np.array(v, float).reshape(-1, 4) for p, v in json.loads(cache.read_text()).items()}
    else:
        device = get_device() if pcfg["device"] == "auto" else torch.device(pcfg["device"])
        model, processor = grounding_dino.load_model(pcfg["dino"], device, REPO_ROOT)
        cans = {}
        for pid in sorted(sizes):
            image = load_image(REPO_ROOT / s1["raw_dir"] / names[pid])
            boxes, _ = grounding_dino.detect(model, processor, image, ea["can_prompt"], device, ea["can_box_threshold"], pcfg["dino"]["text_threshold"])
            cans[pid] = boxes
        cache.write_text(json.dumps({p: b.round(2).tolist() for p, b in cans.items()}))

    raw = json.loads((REPO_ROOT / ea["predictions"]).read_text())["variants"][ea["variant"]]
    preds = {p: (np.array(raw[p]["boxes"], float), np.array(raw[p]["scores"], float)) for p in sizes}
    preds, _ = ignore_below_floor(preds, sizes, pcfg["filters"]["min_height_fraction"])

    extra, missed, imitated, tp = Counter(), Counter(), Counter(), 0
    per_photo, results = {}, {}
    for pid in sorted(sizes):
        boxes, scores = preds[pid]
        keep = scores >= ea["score_threshold"]
        boxes, scores = boxes[keep], scores[keep]
        r = categorize(np.array(gt_boxes[pid]), boxes, scores, cans[pid], sizes[pid][1])
        extra.update(r["extra"]); missed.update(r["missed"]); imitated.update(r["imitated"]); tp += r["tp"]
        per_photo[pid] = (len(r["extra"]), len(r["missed"]), len(gt_boxes[pid]), len(cans[pid]))
        results[pid] = (np.array(gt_boxes[pid]).reshape(-1, 4), boxes, r)

    print(f"AGREEMENT with Grounding DINO pseudo-labels (not accuracy). Fine-tuned model ({Path(ea['predictions']).name}), {ea['variant']} output, score >= {ea['score_threshold']}, IoU >= 0.5, val")
    print(f"matched (TP): {tp}; extra predictions: {sum(extra.values())}; missed pseudo-labels: {sum(missed.values())}")
    print("| Cause | Extra predictions | Missed pseudo-labels |\n|---|---:|---:|")
    for c in CAUSES:
        print(f"| {c} | {extra[c]} | {missed[c]} |")
    n_gt = len(gt["annotations"])
    n_err_gt = missed["can"] + missed["fragment"] + imitated["can"] + imitated["fragment"]
    print(f"pseudo-labels that look like label errors (on a can, or a fragment inside a larger box): {n_err_gt} of {n_gt} "
          f"({100 * n_err_gt / n_gt:.1f}%); the model copied (matched) {imitated['can'] + imitated['fragment']} of them "
          f"(cans {imitated['can']}, fragments {imitated['fragment']}) and refused {missed['can'] + missed['fragment']}")
    print("photos with the most disagreements (extra, missed, pseudo-labels, DINO can boxes):")
    for pid, v in sorted(per_photo.items(), key=lambda kv: -(kv[1][0] + kv[1][1]))[:6]:
        print(f"  {pid}: {v[0]} extra, {v[1]} missed, {v[2]} pseudo-labels, {v[3]} can boxes")

    rng = random.Random(cfg["seed"])
    chosen = list(dict.fromkeys(ea["include"] + rng.sample(sorted(sizes), ea["n_photos"])))
    out = REPO_ROOT / ea["output_dir"]
    check_output_dir(out, protected)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.jpg"):
        old.unlink()
    for pid in chosen:
        g, boxes, r = results[pid]
        image = load_image(REPO_ROOT / s1["raw_dir"] / names[pid])
        d = ImageDraw.Draw(image)
        w = max(2, image.height // 300)
        font = ImageFont.load_default(size=max(14, image.height // 45))
        for i, box in enumerate(boxes):
            if r["pred_hit"][i]:
                d.rectangle(list(box), outline=(0, 200, 0), width=max(1, w // 2))
        for i, cause in zip(r["extra_idx"], r["extra"]):
            d.rectangle(list(boxes[i]), outline=(255, 0, 0), width=w)
            d.text((boxes[i][0] + 2, boxes[i][1] + 2), LETTER[cause], fill=(255, 0, 0), font=font)
        for j, cause in zip(r["missed_idx"], r["missed"]):
            d.rectangle(list(g[j]), outline=(0, 90, 255), width=w + 1)
            d.text((g[j][0] + 2, g[j][3] - 2 - font.size), LETTER[cause], fill=(0, 90, 255), font=font)
        image.thumbnail((ea["max_side_px"],) * 2)
        image.save(out / f"{pid}.jpg", quality=85)
    print(f"drew {len(chosen)} photos in {out.relative_to(REPO_ROOT)}/ (gitignored): {', '.join(chosen)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
