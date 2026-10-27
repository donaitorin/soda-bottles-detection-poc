"""Sanity check of the mean-pooling result of Stage 6b (top-1 12.8%, below the 17% chance level of 6 products).

Hypothesis (owner): the white padding patches of tall crops dominate the mean of the patch tokens, and catalog photos
and shelf crops have different amounts of padding. This embeds the labeled val crops and the 30 catalog crops with
three poolings (cls, mean over all patches, mean over the patches ON THE CROP only) and compares them with the same
protocol as evaluate_identification.py (plain catalog, thresholds tuned leaving one store out). It also prints how much
of the 224 px square is padding for catalog crops and shelf crops, and how often the nearest catalog entry is the same
one for every crop (a collapsed embedding). Development numbers on 180 crops, not accuracy. Nothing is written to the database.

Usage: python scripts/check_pooling.py
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.common.device import get_device  # noqa: E402
from src.data.images import load_image  # noqa: E402
from src.embeddings.embed import content_patch_mask, embed, load_dinov2  # noqa: E402
from src.embeddings.id_metrics import summarize  # noqa: E402
from src.embeddings.identify import cosine_matrix, features, leave_one_store_out  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage6.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load((REPO_ROOT / args.config).read_text())
    s1 = yaml.safe_load((REPO_ROOT / "configs/stage1.yaml").read_text())
    products, ec, idc, brand_of = cfg["products"], cfg["embeddings"], cfg["identify"], cfg["brands"]
    labels = list(csv.DictReader((REPO_ROOT / cfg["labels_export"]["output_dir"] / "val_sample_labels.csv").open()))
    mapping = {r["crop_id"]: r for r in csv.DictReader((REPO_ROOT / cfg["crops"]["mapping_dir"] / cfg["crops"]["sample"]["name"] / "mapping.csv").open())}
    names = {r["photo_id"]: r["file_name"] for r in csv.DictReader((REPO_ROOT / s1["id_maps_dir"] / "photo_ids.csv").open())}
    crop_ids = [r["crop_id"] for r in labels]
    truth = np.array([r["label"] for r in labels], dtype=object)
    stores = np.array([r["store_code"] for r in labels])
    shelf = [load_image(REPO_ROOT / s1["raw_dir"] / names[mapping[c]["photo_id"]]).crop(tuple(int(mapping[c][k]) for k in ("crop_x1", "crop_y1", "crop_x2", "crop_y2"))) for c in crop_ids]
    cat_paths = [(p, sorted((REPO_ROOT / cfg["catalog"]["crop"]["crops_dir"] / p).glob("*.*"))) for p in products]
    entry_product = np.array([p for p, fs in cat_paths for _ in fs])
    entry_photo = np.array([f"{p}/{f.name}" for p, fs in cat_paths for f in fs])
    catalog = [load_image(f) for _, fs in cat_paths for f in fs]

    def padding_share(images):
        m = content_patch_mask([im.size for im in images], ec["image_size"])
        return 1 - m.float().mean(dim=1).numpy()
    for name, ims in (("catalog crops", catalog), ("shelf crops", shelf)):
        s = padding_share(ims)
        print(f"padding share of the 224 px square, {name}: median {np.median(s):.2f} (quartiles {np.percentile(s, 25):.2f} to {np.percentile(s, 75):.2f})")

    device = get_device()
    model = load_dinov2(ec, device, REPO_ROOT)
    index_of = {i: i for i in range(len(crop_ids))}
    print("\nALL NUMBERS ARE DEVELOPMENT NUMBERS on 180 val crops (153 with a product or unknown label), plain catalog of 30 originals, thresholds tuned leaving one store out; agreement with the owner's labels, not accuracy\n")
    print("| Pooling | Top-1 (known) | Top-3 (known) | Brand | Unknown crops: strictly unknown | False known | Most common nearest entry (share of crops) |\n|---|---:|---:|---:|---:|---:|---|")
    out = {}
    for pool in ("cls", "mean", "mean_content"):
        c = {**ec, "pooling": pool}
        q, e = embed(model, shelf, device, c), embed(model, catalog, device, c)
        feat = features(q, e, entry_product, entry_photo, products, idc["top_k_photos"])
        outcome, _ = leave_one_store_out(feat, truth, stores, idc)
        top1 = np.array(products, dtype=object)[feat["top1"]]
        top3 = [[products[j] for j in row] for row in feat["top3"]]
        m = summarize(list(range(len(truth))), truth, outcome, top1, top3, brand_of, products, idc["utility"])
        nearest = np.argmax(cosine_matrix(q, e), axis=1)
        top_entry, top_n = Counter(entry_photo[nearest]).most_common(1)[0]
        s = cosine_matrix(q, e)
        out[pool] = (m, s)
        print(f"| {pool} | {100 * m['top1_acc']:.1f} | {100 * m['top3_acc']:.1f} | {100 * m['brand_acc']:.1f} | {100 * m['unknown_strict_rejection']:.1f} | {100 * m['false_known']:.1f} | {top_entry} ({100 * top_n / len(nearest):.0f}%) |")
    for pool, (m, s) in out.items():
        print(f"- {pool}: similarity of shelf crops to catalog entries: median {np.median(s):.3f}, spread (std of the best-entry similarity over crops) {s.max(axis=1).std():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
