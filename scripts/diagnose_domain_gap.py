"""Diagnostic for the domain gap between clean catalog photos and small, blurry shelf crops (Stage 6b, decision input only).

Nothing here goes into the database. (1) Plain-catalog top-1 by the ORIGINAL pixel height of the shelf crop: do small crops do worse?
(2) What if the catalog also contained its own photos degraded to shelf resolution (downscaled to a few heights, then
enlarged back)? Evaluated with the same leave-one-store-out protocol on the 180 labeled val crops; paired bootstrap by store. These
are DEVELOPMENT numbers (a diagnostic on the val crops), not accuracy.

Usage: python scripts/diagnose_domain_gap.py
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.common.device import get_device  # noqa: E402
from src.data.images import load_image  # noqa: E402
from src.embeddings.embed import embed, load_dinov2  # noqa: E402
from src.embeddings.id_metrics import summarize  # noqa: E402
from src.embeddings.identify import features, leave_one_store_out  # noqa: E402
from src.evaluation.detection import bootstrap_by_store, difference_metric  # noqa: E402

HEIGHTS = [60, 100, 150, 220]   # shelf-resolution heights to degrade the catalog crops to


def degrade(img: Image.Image, h: int) -> Image.Image:
    w = max(8, round(img.width * h / img.height))
    return img.resize((w, h), Image.BILINEAR).resize(img.size, Image.BICUBIC)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage6.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load((REPO_ROOT / args.config).read_text())
    products, ec, idc = cfg["products"], cfg["embeddings"], cfg["identify"]
    labels = list(csv.DictReader((REPO_ROOT / cfg["labels_export"]["output_dir"] / "val_sample_labels.csv").open()))
    mapping = {r["crop_id"]: r for r in csv.DictReader((REPO_ROOT / cfg["crops"]["mapping_dir"] / cfg["crops"]["sample"]["name"] / "mapping.csv").open())}
    crop_ids = np.array([r["crop_id"] for r in labels])
    truth = np.array([r["label"] for r in labels], dtype=object)
    stores = np.array([r["store_code"] for r in labels])
    z = np.load(REPO_ROOT / cfg["crops"]["mapping_dir"] / cfg["crops"]["sample"]["name"] / "embeddings.npz")
    order = {c: i for i, c in enumerate(z["crop_ids"])}
    q = z["cls"][[order[c] for c in crop_ids]]
    cat = np.load(REPO_ROOT / cfg["catalog_files"]["embeddings_npz"])
    orig = ~cat["is_augmented"]
    height = np.array([int(mapping[c]["crop_y2"]) - int(mapping[c]["crop_y1"]) for c in crop_ids])

    feat = features(q, cat["cls"], cat["product"], cat["source_photo"], products, idc["top_k_photos"], orig)
    top1 = np.array(products, dtype=object)[feat["top1"]]
    known = np.isin(truth, products)
    print("DEVELOPMENT numbers (diagnostic on the 180 labeled val crops), not accuracy.\n")
    print("(1) Plain catalog, top-1 on crops labeled with a product, by the shelf crop's original height in pixels:")
    print("| Crop height | Crops | Top-1 correct |\n|---|---:|---:|")
    for lo, hi in ((0, 150), (150, 300), (300, 10_000)):
        sel = known & (height >= lo) & (height < hi)
        print(f"| {lo} to {hi if hi < 10_000 else 'more'} px | {int(sel.sum())} | {int((top1[sel] == truth[sel]).sum())} of {int(sel.sum())} |")

    device = get_device()
    model = load_dinov2(ec, device, REPO_ROOT)
    crops_dir = REPO_ROOT / cfg["catalog"]["crop"]["crops_dir"]
    prods, photos, imgs = [], [], []
    for product in products:
        for path in sorted(p for p in (crops_dir / product).glob("*.*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")):
            base = load_image(path)
            for h in HEIGHTS:
                prods.append(product); photos.append(f"{product}/{path.name}"); imgs.append(degrade(base, h))
    deg = embed(model, imgs, device, ec)
    prods, photos = np.array(prods), np.array(photos)
    store_of = dict(zip(crop_ids, stores))
    index_of = {c: i for i, c in enumerate(crop_ids)}
    brand_of = cfg["brands"]

    def evaluate(cat_vecs, cat_prod, cat_photo):
        f = features(q, cat_vecs, cat_prod, cat_photo, products, idc["top_k_photos"])
        out, _ = leave_one_store_out(f, truth, stores, idc)
        t1 = np.array(products, dtype=object)[f["top1"]]
        t3 = [[products[j] for j in row] for row in f["top3"]]
        return lambda ids: summarize([index_of[c] for c in ids], truth, out, t1, t3, brand_of, products, idc["utility"])

    plain = evaluate(cat["cls"][orig], cat["product"][orig], cat["source_photo"][orig])
    with_deg = evaluate(np.vstack([cat["cls"][orig], deg]), np.concatenate([cat["product"][orig], prods]), np.concatenate([cat["source_photo"][orig], photos]))
    only_deg = evaluate(deg, prods, photos)
    all_ids = list(crop_ids)
    print(f"\n(2) Catalog = originals plus the same photos degraded to shelf resolution (heights {HEIGHTS} px, enlarged back), in memory only:")
    print("| Metric (%) | Plain originals | Originals + degraded | Degraded only |\n|---|---:|---:|---:|")
    for k, name in (("top1_acc", "Top-1 (known)"), ("top3_acc", "Top-3 (known)"), ("brand_acc", "Brand accuracy"), ("auto_correct_rate", "Known answered automatically and correctly"),
                    ("unknown_rejection", "Unknown rejection (unknown or review)"), ("false_known", "False known rate")):
        print(f"| {name} | {100 * plain(all_ids)[k]:.1f} | {100 * with_deg(all_ids)[k]:.1f} | {100 * only_deg(all_ids)[k]:.1f} |")
    d = difference_metric(with_deg, plain)
    ci = bootstrap_by_store(store_of, d, cfg["bootstrap"]["n"], cfg["seed"], cfg["bootstrap"]["level"])
    pt = d(all_ids)
    print("\nPaired difference (originals + degraded minus plain, points) [95% CI by store]: " + ", ".join(
        f"{n} {100 * pt[k]:+.1f} [{100 * ci[k][0]:+.1f}, {100 * ci[k][1]:+.1f}]" for k, n in (("top1_acc", "top-1"), ("brand_acc", "brand"), ("auto_correct_rate", "auto-correct"))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
