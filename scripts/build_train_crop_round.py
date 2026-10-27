"""Second step: build the TRAIN crop round the owner labels (shelf examples for the catalog), plus the Label Studio import file.

Crops come from the Grounding DINO pseudo-label boxes of TRAIN photos only (never val or test: every sampled photo must be in
data/splits/train.txt). Two parts, mixed in one project so the labeler cannot tell them apart and gets no suggestion:
  random    n_random crops, stratified by train store (proportional, at least min_per_store, at most max_per_photo per photo), seeded
  targeted  n_targeted crops picked by retrieval for the rare products: for each target product, the crops whose top-3 products
            (similarity to the 30 catalog originals) include it, most similar first, with per-photo and per-store caps.
            Retrieval is only a way to find candidates; the owner's label decides what each crop is.
The CLS embeddings of EVERY train box are cached (needed for the pick and for the later measurements). An existing sample is
never overwritten. Same 10% margin crops, same display upscaling and same context image as the val crop sample.

Outputs (gitignored, ids only): data/labelstudio/crops/<crop id>.jpg (+ context/), data/crops/<name>/{mapping.csv,tasks.json,candidate_embeddings.npz}.

Usage: python scripts/build_train_crop_round.py [--name NAME] [--n-random N] [--n-targeted N]
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.common.device import get_device  # noqa: E402
from src.data.images import load_image  # noqa: E402
from src.data.paths import check_output_dir  # noqa: E402
from src.embeddings.crops import (allocate_quota, crop_box_with_margin, crop_id, pick_targeted, resize_for_display,  # noqa: E402
                                  sample_crops)
from src.embeddings.embed import embed, load_dinov2  # noqa: E402
from src.embeddings.identify import features  # noqa: E402

FIELDS = ["crop_id", "photo_id", "store_code", "box_index", "x1", "y1", "x2", "y2", "crop_x1", "crop_y1", "crop_x2", "crop_y2",
          "photo_width", "photo_height", "label_source", "score", "batch", "targeted_for"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage6.yaml")
    parser.add_argument("--name", help="sample name (default: shelf_round.name)")
    parser.add_argument("--n-random", type=int)
    parser.add_argument("--n-targeted", type=int)
    args = parser.parse_args()
    cfg = yaml.safe_load((REPO_ROOT / args.config).read_text())
    s1 = yaml.safe_load((REPO_ROOT / "configs/stage1.yaml").read_text())
    cc, ls, sr, ec, products = cfg["crops"], cfg["label_studio"], cfg["shelf_round"], cfg["embeddings"], cfg["products"]
    name = args.name or sr["name"]
    n_random, n_targeted = args.n_random if args.n_random is not None else sr["n_random"], args.n_targeted if args.n_targeted is not None else sr["n_targeted"]
    protected = [REPO_ROOT / d for d in s1["protected_dirs"]]
    out_dir, images_dir, context_dir = REPO_ROOT / cc["mapping_dir"] / name, REPO_ROOT / cc["images_dir"], REPO_ROOT / cc["context_dir"]
    for d in (out_dir, images_dir, context_dir):
        check_output_dir(d, protected)
    if (out_dir / "mapping.csv").exists():
        raise SystemExit(f"{out_dir.relative_to(REPO_ROOT)}/mapping.csv already exists: refusing to overwrite a round that may be labeled. Use --name for another one.")

    coco = json.loads((REPO_ROOT / sr["coco"]).read_text())
    train_ids = {l.strip() for l in (REPO_ROOT / sr["split_file"]).read_text().split("\n") if l.strip()}
    outside = {im["photo_id"] for im in coco["images"]} - train_ids
    assert not outside, f"{len(outside)} photos of {sr['coco']} are not in the train split: refusing to sample from them"
    names = {r["photo_id"]: r["file_name"] for r in csv.DictReader((REPO_ROOT / s1["id_maps_dir"] / "photo_ids.csv").open())}
    boxes_of = defaultdict(list)
    for a in coco["annotations"]:
        boxes_of[a["image_id"]].append(a)
    info = {im["photo_id"]: im for im in coco["images"]}
    candidates = [(im["photo_id"], k, im["store_code"]) for im in coco["images"] for k, _ in enumerate(boxes_of[im["id"]])]
    cand_ids = [crop_id(p, k) for p, k, _ in candidates]
    print(f"{len(candidates)} train pseudo-label boxes in {len(info)} photos, {len(set(s for _, _, s in candidates))} stores")

    # ---- CLS embeddings of every train box (cached), searched against the 30 originals for the targeted pick
    emb_path = out_dir / "candidate_embeddings.npz"
    device = get_device()
    model = load_dinov2(ec, device, REPO_ROOT)
    vecs = np.zeros((len(candidates), 768), np.float32)
    by_photo = defaultdict(list)
    for i, (pid, k, _) in enumerate(candidates):
        by_photo[pid].append(i)
    for n_done, (pid, idxs) in enumerate(sorted(by_photo.items()), 1):
        photo = load_image(REPO_ROOT / s1["raw_dir"] / names[pid])
        im = info[pid]
        assert photo.size == (im["width"], im["height"]), f"{pid}: size differs from the COCO file"
        crops = []
        for i in idxs:
            x, y, w, h = boxes_of[im["id"]][candidates[i][1]]["bbox"]
            crops.append(photo.crop(crop_box_with_margin((x, y, x + w, y + h), im["width"], im["height"], cc["margin"])))
        vecs[idxs] = embed(model, crops, device, {**ec, "pooling": "cls"})
        if n_done % 40 == 0:
            print(f"  embedded {n_done} of {len(by_photo)} photos")
    cat = np.load(REPO_ROOT / cfg["catalog_files"]["embeddings_npz"])
    orig = ~cat["is_augmented"]
    feat = features(vecs, cat["cls"], cat["product"], cat["source_photo"], products, cfg["identify"]["top_k_photos"], orig)

    # ---- random part, stratified by store
    n_boxes, n_photos = Counter(s for _, _, s in candidates), Counter()
    for pid in {p for p, _, _ in candidates}:
        n_photos[info[pid]["store_code"]] += 1
    available = {s: (n_boxes[s], sr["max_per_photo"] * n_photos[s]) for s in n_boxes}
    quota = allocate_quota(available, n_random, sr["min_per_store"])
    chosen = [(p, k, s, "random", "") for p, k, s in sample_crops(candidates, quota, sr["max_per_photo"], cfg["seed"])]
    # ---- targeted part: rare products by retrieval, on the crops not already chosen
    t = sr["targeted"]
    taken = {(p, k) for p, k, _, _, _ in chosen}
    per_product = -(-n_targeted // len(t["products"]))
    picked = pick_targeted(candidates, feat["prod_sim"], products, t["products"], per_product, t["top_rank"], t["max_per_photo"],
                           int(t["max_per_store_share"] * n_targeted), taken)[:n_targeted]
    chosen += [(p, k, s, "targeted", prod) for p, k, s, prod in picked]
    assert len({(p, k) for p, k, *_ in chosen}) == len(chosen), "a crop was chosen twice"

    # ---- write images, context images, mapping and tasks
    for d in (out_dir, images_dir, context_dir):
        d.mkdir(parents=True, exist_ok=True)
    np.savez(emb_path, crop_ids=np.array(cand_ids), cls=vecs, store=np.array([s for _, _, s in candidates]))
    rows, tasks = [], []
    chosen_by_photo = defaultdict(list)
    for item in chosen:
        chosen_by_photo[item[0]].append(item)
    for pid, items in chosen_by_photo.items():
        im = info[pid]
        photo = load_image(REPO_ROOT / s1["raw_dir"] / names[pid])
        for _, k, store, batch, target in items:
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
                         "label_source": im["label_source"], "score": a.get("score", ""), "batch": batch, "targeted_for": target})
            tasks.append({"data": {"image": f"{ls['url_prefix']}{cid}.jpg", "context": f"{ls['context_prefix']}{cid}.jpg", "crop_id": cid}})
    order = {r["crop_id"]: i for i, r in enumerate(sorted(rows, key=lambda r: (r["photo_id"], float(r["x1"]))))}   # by photo, left to right
    tasks.sort(key=lambda t_: order[t_["data"]["crop_id"]])
    with (out_dir / "mapping.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: order[r["crop_id"]]))
    (out_dir / "tasks.json").write_text(json.dumps(tasks))
    print(f"round '{name}': {len(rows)} crops from {len(chosen_by_photo)} photos (random {sum(r['batch'] == 'random' for r in rows)}, targeted {sum(r['batch'] == 'targeted' for r in rows)})")
    print("per store: " + ", ".join(f"{s} {c}" for s, c in sorted(Counter(r["store_code"] for r in rows).items())))
    print("targeted per product: " + ", ".join(f"{p} {c}" for p, c in Counter(r["targeted_for"] for r in rows if r["batch"] == "targeted").items()))
    print(f"retrieval top-1 of ALL chosen crops (not labels, only what the 30 originals suggest): " + ", ".join(
        f"{p} {c}" for p, c in sorted(Counter(products[feat['top1'][cand_ids.index(r['crop_id'])]] for r in rows).items())))
    print(f"files: {images_dir.relative_to(REPO_ROOT)}/ (+ context/), {out_dir.relative_to(REPO_ROOT)}/{{mapping.csv,tasks.json,candidate_embeddings.npz}} (all gitignored)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
