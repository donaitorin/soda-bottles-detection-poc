"""Build the val crop sample that the owner labels by hand (the Stage 6 yardstick), plus the Label Studio import file.

Crops are cut from the Grounding DINO pseudo-label boxes of val (10% margin), stratified by store (proportional,
at least `min_per_store`, at most `max_per_photo` per photo), seeded. NO suggestion is attached to a task: the owner
labels without seeing a model's opinion. Each task shows the crop and a context image (the photo with the box drawn).
Crops are upscaled for DISPLAY only; embeddings later use the original pixels.

Outputs (all gitignored, ids only): data/labelstudio/crops/<crop id>.jpg, .../context/<crop id>.jpg,
data/crops/<name>/mapping.csv and data/crops/<name>/tasks.json. An existing sample is never overwritten (labels would
no longer match): use --name for a second batch (for example targeted unknowns) and --exclude to skip crops already sampled.

Usage: python scripts/build_crop_sample.py [--name NAME] [--n N] [--exclude NAME ...] [--only-photos FILE]
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.images import load_image  # noqa: E402
from src.data.paths import check_output_dir  # noqa: E402
from src.embeddings.crops import allocate_quota, crop_box_with_margin, crop_id, sample_crops  # noqa: E402

FIELDS = ["crop_id", "photo_id", "store_code", "box_index", "x1", "y1", "x2", "y2", "crop_x1", "crop_y1", "crop_x2", "crop_y2",
          "photo_width", "photo_height", "label_source", "score"]


def resize_for_display(image: Image.Image, min_long: int, max_long: int) -> Image.Image:
    long_side = max(image.size)
    scale = min_long / long_side if long_side < min_long else (max_long / long_side if long_side > max_long else 1.0)
    if scale == 1.0:
        return image
    return image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.LANCZOS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage6.yaml")
    parser.add_argument("--name", help="sample name (default: crops.sample.name)")
    parser.add_argument("--n", type=int, help="number of crops (default: crops.sample.n)")
    parser.add_argument("--exclude", nargs="*", default=[], help="names of earlier samples whose crops are skipped")
    parser.add_argument("--only-photos", help="text file with photo ids (one per line): sample only from these photos")
    args = parser.parse_args()
    cfg = yaml.safe_load((REPO_ROOT / args.config).read_text())
    s1 = yaml.safe_load((REPO_ROOT / "configs/stage1.yaml").read_text())
    cc, ls = cfg["crops"], cfg["label_studio"]
    name, n = args.name or cc["sample"]["name"], args.n or cc["sample"]["n"]
    protected = [REPO_ROOT / d for d in s1["protected_dirs"]]
    out_dir = REPO_ROOT / cc["mapping_dir"] / name
    images_dir, context_dir = REPO_ROOT / cc["images_dir"], REPO_ROOT / cc["context_dir"]
    for d in (out_dir, images_dir, context_dir):
        check_output_dir(d, protected)
    if (out_dir / "mapping.csv").exists():
        raise SystemExit(f"{out_dir.relative_to(REPO_ROOT)}/mapping.csv already exists: refusing to overwrite a sample that may be labeled. Use --name for another batch.")

    coco = json.loads((REPO_ROOT / cc["coco"]).read_text())
    names = {r["photo_id"]: r["file_name"] for r in csv.DictReader((REPO_ROOT / s1["id_maps_dir"] / "photo_ids.csv").open())}
    only = set(l.strip() for l in Path(args.only_photos).read_text().split("\n") if l.strip()) if args.only_photos else None
    excluded = set()
    for other in args.exclude:
        with (REPO_ROOT / cc["mapping_dir"] / other / "mapping.csv").open() as f:
            excluded |= {r["crop_id"] for r in csv.DictReader(f)}

    boxes_of = defaultdict(list)
    for a in coco["annotations"]:
        boxes_of[a["image_id"]].append(a)
    info, candidates = {}, []
    for im in coco["images"]:
        if only is not None and im["photo_id"] not in only:
            continue
        info[im["photo_id"]] = im
        for k, a in enumerate(boxes_of[im["id"]]):
            if crop_id(im["photo_id"], k) not in excluded:
                candidates.append((im["photo_id"], k, im["store_code"]))
    n_boxes, n_photos = Counter(s for _, _, s in candidates), Counter()
    for pid in {p for p, _, _ in candidates}:
        n_photos[info[pid]["store_code"]] += 1
    available = {s: (n_boxes[s], cc["sample"]["max_per_photo"] * n_photos[s]) for s in n_boxes}
    quota = allocate_quota(available, n, cc["sample"]["min_per_store"])
    chosen = sample_crops(candidates, quota, cc["sample"]["max_per_photo"], cfg["seed"])
    chosen.sort(key=lambda c: (c[0], boxes_of[info[c[0]]["id"]][c[1]]["bbox"][0]))   # by photo, left to right

    rows, tasks = [], []
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    context_dir.mkdir(parents=True, exist_ok=True)
    by_photo = defaultdict(list)
    for pid, k, store in chosen:
        by_photo[pid].append((k, store))
    for pid, items in by_photo.items():
        im = info[pid]
        photo = load_image(REPO_ROOT / s1["raw_dir"] / names[pid])
        assert photo.size == (im["width"], im["height"]), f"{pid}: size differs from the COCO file"
        for k, store in items:
            a = boxes_of[im["id"]][k]
            x, y, w, h = a["bbox"]
            box = (x, y, x + w, y + h)
            cid = crop_id(pid, k)
            region = crop_box_with_margin(box, im["width"], im["height"], cc["margin"])
            resize_for_display(photo.crop(region), cc["display_min_long_side_px"], cc["display_max_long_side_px"]).save(images_dir / f"{cid}.jpg", quality=92)
            ctx = photo.copy()
            ImageDraw.Draw(ctx).rectangle(box, outline=(255, 220, 0), width=max(3, im["height"] // 250))
            ctx.thumbnail((cc["context_long_side_px"],) * 2)
            ctx.save(context_dir / f"{cid}.jpg", quality=85)
            rows.append({"crop_id": cid, "photo_id": pid, "store_code": store, "box_index": k, **dict(zip(("x1", "y1", "x2", "y2"), (round(v, 2) for v in box))),
                         **dict(zip(("crop_x1", "crop_y1", "crop_x2", "crop_y2"), region)), "photo_width": im["width"], "photo_height": im["height"],
                         "label_source": im["label_source"], "score": a.get("score", "")})
            tasks.append({"data": {"image": f"{ls['url_prefix']}{cid}.jpg", "context": f"{ls['context_prefix']}{cid}.jpg", "crop_id": cid}})
    order = {r["crop_id"]: i for i, r in enumerate(sorted(rows, key=lambda r: (r["photo_id"], float(r["x1"]))))}
    tasks.sort(key=lambda t: order[t["data"]["crop_id"]])
    with (out_dir / "mapping.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: order[r["crop_id"]]))
    (out_dir / "tasks.json").write_text(json.dumps(tasks))
    per_store = Counter(r["store_code"] for r in rows)
    print(f"sample '{name}': {len(rows)} crops from {len(by_photo)} photos | quota {dict(sorted(quota.items()))} | per store {dict(sorted(per_store.items()))}")
    print(f"crops in {images_dir.relative_to(REPO_ROOT)}/, context images in {context_dir.relative_to(REPO_ROOT)}/, mapping and tasks in {out_dir.relative_to(REPO_ROOT)}/ (all gitignored)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
