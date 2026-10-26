"""Crop each catalog product photo to its bottle with the FROZEN Grounding DINO, and draw a contact sheet for the owner.

Each photo holds one bottle on a plain background, so the crop removes the background and any text around it. Same prompt
and thresholds as the shelf pre-labels, but NO Stage 2 filters (a single bottle fills much of the photo and the "oversized"
filter would drop it). Same 10% margin as the shelf crops. More than one box: almost surely fragments of the same bottle, the
largest is used (an explicit `overrides` entry in the config can choose the enclosing box after looking at the photo);
no box: the whole photo. Every photo whose rule is not `single` is listed.
Outputs (gitignored): data/catalog/crops/<product id>/<same file name>, data/catalog/crops/crops.csv,
data/review/catalog_crops/contact_sheet.jpg (all crops) and review_<product>_<NN>.jpg (original with the boxes, for listed photos).

Usage: python scripts/crop_catalog_photos.py
"""
import argparse
import csv
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.common.device import get_device  # noqa: E402
from src.data.images import load_image  # noqa: E402
from src.data.paths import check_output_dir  # noqa: E402
from src.embeddings.catalog_crop import choose_box  # noqa: E402
from src.embeddings.crops import crop_box_with_margin  # noqa: E402
from src.prelabel import grounding_dino  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage6.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load((REPO_ROOT / args.config).read_text())
    cc = cfg["catalog"]["crop"]
    pcfg = yaml.safe_load((REPO_ROOT / cc["prompt_config"]).read_text())
    s1 = yaml.safe_load((REPO_ROOT / "configs/stage1.yaml").read_text())
    protected = [REPO_ROOT / d for d in s1["protected_dirs"]]
    photos_dir, crops_dir, review_dir = REPO_ROOT / cfg["catalog"]["photos_dir"], REPO_ROOT / cc["crops_dir"], REPO_ROOT / cc["review_dir"]
    for d in (crops_dir, review_dir):
        check_output_dir(d, protected)
        d.mkdir(parents=True, exist_ok=True)
    for old in review_dir.glob("*.jpg"):
        old.unlink()

    device = get_device() if pcfg["device"] == "auto" else torch.device(pcfg["device"])
    model, processor = grounding_dino.load_model(pcfg["dino"], device, REPO_ROOT)
    d = pcfg["dino"]
    rows, crops, listed = [], [], []
    overrides = cc.get("overrides") or {}
    for product in cfg["products"]:
        for path in sorted((photos_dir / product).iterdir()):
            if path.name.startswith(".") or path.suffix.lower() not in cfg["catalog"]["image_extensions"]:
                continue
            rel = f"{product}/{path.name}"
            image = load_image(path)
            boxes, scores = grounding_dino.detect(model, processor, image, d["prompt"], device, d["box_threshold"], d["text_threshold"])
            box, rule = choose_box(boxes, image.size, cc["several_boxes"], overrides.get(rel))
            region = crop_box_with_margin(box, image.width, image.height, cc["margin"]) if rule != "whole_photo" else (0, 0, image.width, image.height)
            out = crops_dir / product / path.name
            out.parent.mkdir(parents=True, exist_ok=True)
            image.crop(region).save(out, quality=95)
            crops.append((rel, image.crop(region)))
            rows.append({"file": rel, "photo_w": image.width, "photo_h": image.height, "n_boxes": len(boxes), "rule": rule,
                         "scores": ";".join(f"{s:.2f}" for s in scores), "box": ";".join(f"{v:.0f}" for v in box),
                         "crop": ";".join(str(v) for v in region),
                         "box_share_of_photo": f"{(box[2] - box[0]) * (box[3] - box[1]) / (image.width * image.height):.2f}"})
            if rule != "single":
                listed.append((rel, len(boxes), rule))
                ctx = image.copy()
                dr = ImageDraw.Draw(ctx)
                for b in boxes:
                    dr.rectangle(list(b), outline=(255, 0, 0), width=max(3, image.height // 200))
                dr.rectangle(list(region), outline=(0, 200, 0), width=max(3, image.height // 200))
                ctx.thumbnail((700, 700))
                ctx.save(review_dir / f"review_{product}_{path.stem.split('_')[-1]}.jpg", quality=85)
            print(f"{rel}: {len(boxes)} box(es), rule {rule}, box {rows[-1]['box']}", flush=True)
    with (crops_dir / "crops.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    cell, per_row = cc["contact_sheet"]["cell_px"], max(sum(1 for r, _ in crops if r.startswith(p + "/")) for p in cfg["products"])
    sheet = Image.new("RGB", (per_row * cell, len(cfg["products"]) * (cell + 22)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=14)
    row_of = {p: i for i, p in enumerate(cfg["products"])}
    col = {p: 0 for p in cfg["products"]}
    for rel, im in crops:
        p = rel.split("/")[0]
        thumb = im.copy()
        thumb.thumbnail((cell - 8, cell - 8))
        x, y = col[p] * cell, row_of[p] * (cell + 22)
        sheet.paste(thumb, (x + (cell - thumb.width) // 2, y + 22 + (cell - thumb.height) // 2))
        draw.text((x + 4, y + 2), rel.split("/")[1], fill=(0, 0, 0), font=font)
        col[p] += 1
    sheet.save(review_dir / "contact_sheet.jpg", quality=88)
    print(f"\n{len(rows)} photos cropped; photos with zero or several boxes ({len(listed)}):")
    for rel, n, rule in listed:
        print(f"  {rel}: {n} box(es) -> {rule}")
    print(f"contact sheet: {(review_dir / 'contact_sheet.jpg').relative_to(REPO_ROOT)} (gitignored); crops in {crops_dir.relative_to(REPO_ROOT)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
