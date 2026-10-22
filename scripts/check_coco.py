"""Stage 3 gate: sanity checks on the COCO files and drawings of 20 random photos.

Checks per split: the photos are exactly those of data/splits/<split>.txt, no photo in two splits, file names are
photo ids, boxes inside the image, area above zero, no duplicate boxes (IoU above the configured value), every
box at least the minimum height, boxes per photo (warning outside the range). Draws 20 seeded random photos into the
gitignored data/review/coco_check/ (green = owner_corrected, red = prelabel_unverified).

Usage: python scripts/check_coco.py
"""
import argparse
import csv
import json
import random
import statistics
import sys
from pathlib import Path

import yaml
from PIL import ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.annotations.sanity import check_coco, counts_per_photo  # noqa: E402
from src.data.images import load_image  # noqa: E402
from src.data.paths import check_output_dir  # noqa: E402


def read_ids(path: Path) -> list[str]:
    return [l.strip() for l in path.read_text().split("\n") if l.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage3.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load((REPO_ROOT / args.config).read_text())
    pcfg = yaml.safe_load((REPO_ROOT / cfg["prelabel_config"]).read_text())
    s1 = yaml.safe_load((REPO_ROOT / "configs/stage1.yaml").read_text())
    sn = cfg["sanity"]
    out_dir = REPO_ROOT / cfg["export"]["output_dir"]
    names = {r["photo_id"]: r["file_name"] for r in csv.DictReader((REPO_ROOT / s1["id_maps_dir"] / "photo_ids.csv").open())}

    failed, all_images = False, {}
    for split in ("train", "val", "test"):
        coco = json.loads((out_dir / f"{split}.json").read_text())
        errors, warnings = check_coco(coco, read_ids(REPO_ROOT / s1["splits"]["dir"] / f"{split}.txt"),
                                      pcfg["filters"]["min_height_fraction"], sn["duplicate_iou"],
                                      tuple(sn["boxes_per_photo"]), sn["id_regex"])
        counts = counts_per_photo(coco)
        sources = {}
        for im in coco["images"]:
            sources[im["label_source"]] = sources.get(im["label_source"], 0) + 1
        print(f"{split:5} {len(coco['images']):3} photos {len(coco['annotations']):6} boxes | per photo min/median/max "
              f"{counts[0]}/{int(statistics.median(counts))}/{counts[-1]} | sources {sources} | "
              f"{len(errors)} error(s), {len(warnings)} warning(s)")
        for msg in errors[:10]:
            print("   ERROR", msg)
        for msg in warnings[:10]:
            print("   warning", msg)
        failed |= bool(errors)
        boxes_of = {}
        for a in coco["annotations"]:
            boxes_of.setdefault(a["image_id"], []).append(a["bbox"])
        for im in coco["images"]:
            if im["photo_id"] in all_images:
                print(f"   ERROR {im['photo_id']} is in two splits")
                failed = True
            all_images[im["photo_id"]] = (im, boxes_of.get(im["id"], []))

    rng = random.Random(cfg["seed"])
    chosen = rng.sample(sorted(all_images), sn["n_drawn_photos"])
    draw_dir = REPO_ROOT / sn["draw_dir"]
    check_output_dir(draw_dir, [REPO_ROOT / d for d in s1["protected_dirs"]])
    draw_dir.mkdir(parents=True, exist_ok=True)
    for old in draw_dir.glob("*.jpg"):
        old.unlink()
    for pid in chosen:
        im, boxes = all_images[pid]
        image = load_image(REPO_ROOT / s1["raw_dir"] / names[pid])
        assert image.size == (im["width"], im["height"]), f"{pid}: size differs from the COCO file"
        draw = ImageDraw.Draw(image)
        color = (0, 200, 0) if im["label_source"] == "owner_corrected" else (255, 0, 0)
        for x, y, w, h in boxes:
            draw.rectangle([x, y, x + w, y + h], outline=color, width=max(2, image.height // 300))
        image.thumbnail((sn["draw_max_side_px"],) * 2)
        image.save(draw_dir / f"{pid}.jpg", quality=85)
    print(f"drew {len(chosen)} random photos in {draw_dir.relative_to(REPO_ROOT)}/ (gitignored)")
    print("FAILED" if failed else "OK: no sanity errors. Reminder: pseudo-labels, so metrics measure agreement with Grounding DINO.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
