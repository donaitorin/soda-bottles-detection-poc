"""Build the reference catalog: embed the catalog crops and their augmented variants and load them into Postgres + pgvector.

Entries: each catalog photo's bottle crop (original) and its 6 augmented variants (same seeds as the preview), each its own
entry tagged `augmented` (in the database only if `catalog.use_augmented` is true: it is false, because the augmented catalog identified worse) with its source photo and parameters, and with the source URL and license from sources.csv (unknown
while the owner has not filled them in; the final-report gate is `check_catalog_photos.py --final`).
Writes the ignored data/catalog/embeddings.npz and entries.csv (for the offline experiments), and rebuilds the table
`catalog_entries` in the OWN database `soda_catalog` of the running Postgres container (idempotent). Never the Label Studio data.

Usage: python scripts/build_catalog.py [--no-db]
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.common.device import get_device  # noqa: E402
from src.data.images import load_image  # noqa: E402
from src.data.paths import check_output_dir  # noqa: E402
from src.embeddings import catalog_db  # noqa: E402
from src.embeddings.augment import augment, seed_for  # noqa: E402
from src.embeddings.embed import embed, load_dinov2  # noqa: E402
from src.embeddings.eval_data import load_shelf  # noqa: E402
from src.embeddings.shelf_catalog import shelf_db_entries  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage6.yaml")
    parser.add_argument("--no-db", action="store_true", help="only write the npz and csv, do not touch Postgres")
    args = parser.parse_args()
    cfg = yaml.safe_load((REPO_ROOT / args.config).read_text())
    s1 = yaml.safe_load((REPO_ROOT / "configs/stage1.yaml").read_text())
    protected = [REPO_ROOT / d for d in s1["protected_dirs"]]
    cat, ac, ec = cfg["catalog"], cfg["catalog"]["augment"], cfg["embeddings"]
    sources = {r["file"]: r for r in csv.DictReader((REPO_ROOT / cat["sources_csv"]).open(newline=""))}
    crops_dir = REPO_ROOT / cat["crop"]["crops_dir"]

    entries, images = [], []
    for product in cfg["products"]:
        for path in sorted(p for p in (crops_dir / product).glob("*.*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")):
            rel = f"{product}/{path.name}"
            src = sources[rel]
            base = {"product_id": product, "brand": cfg["brands"][product], "size": src.get("size"), "material": src.get("material"),
                    "source_photo": rel, "source_url": src["source_url"], "license": src["license"],
                    "packaging_checked": src["uruguay_packaging_checked"], "crop_method": "dino_box_margin10"}
            original = load_image(path)
            entries.append({**base, "is_augmented": False, "aug_index": None, "aug_params": None})
            images.append(original)
            for k, (im, p) in enumerate(augment(original, ac, seed_for(cfg["seed"], rel)), 1):
                entries.append({**base, "is_augmented": True, "aug_index": k, "aug_params": p})
                images.append(im)
    device = get_device()
    model = load_dinov2(ec, device, REPO_ROOT)
    cls_vecs = embed(model, images, device, {**ec, "pooling": "cls"})
    mean_vecs = embed(model, images, device, {**ec, "pooling": "mean"})
    print(f"{len(entries)} catalog entries ({sum(not e['is_augmented'] for e in entries)} originals, {sum(e['is_augmented'] for e in entries)} augmented), "
          f"embedding {cls_vecs.shape}, device {device}")

    out_npz, out_csv = REPO_ROOT / cfg["catalog_files"]["embeddings_npz"], REPO_ROOT / cfg["catalog_files"]["entries_csv"]
    check_output_dir(out_npz.parent, protected)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, cls=cls_vecs, mean=mean_vecs, product=np.array([e["product_id"] for e in entries]),
             source_photo=np.array([e["source_photo"] for e in entries]), is_augmented=np.array([e["is_augmented"] for e in entries]))
    with out_csv.open("w", newline="") as f:
        fields = ["product_id", "brand", "source_photo", "is_augmented", "aug_index", "source_url", "license", "packaging_checked", "crop_method"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(entries)
    per = Counter((e["product_id"], e["is_augmented"]) for e in entries)
    print("entries per product (originals + augmented): " + ", ".join(f"{p} {per[(p, False)]}+{per[(p, True)]}" for p in cfg["products"]))
    unknown = sum(1 for e in entries if not e["is_augmented"] and (e["source_url"].lower() == "unknown" or e["license"].lower() == "unknown"))
    if unknown:
        print(f"NOTE: {unknown} catalog photos still have an unknown source or license in sources.csv (fine for development; a FINAL report needs check_catalog_photos.py --final to pass)")
    if args.no_db:
        return 0
    db = cfg["database"]
    created = catalog_db.create_database(db, REPO_ROOT, db["name"])
    conn = catalog_db.connect(db, REPO_ROOT, db["name"])
    try:
        catalog_db.init_schema(conn, cls_vecs.shape[1], db["hnsw"])
        keep = [i for i, e in enumerate(entries) if cat["use_augmented"] or not e["is_augmented"]]
        if not cat["use_augmented"]:
            print("catalog.use_augmented is false: only the original crops go into the database (the augmented entries stay in embeddings.npz for the experiments)")
        db_entries, db_vecs = [entries[i] for i in keep], cls_vecs[keep]
        if cat.get("use_shelf"):
            sc = cfg["shelf_round"]["catalog"]
            shelf_entries, shelf_vecs = shelf_db_entries(load_shelf(cfg, REPO_ROOT), cfg["brands"], sc["cap_per_product"], sc["cap_unknown"], cfg["seed"])
            db_entries, db_vecs = db_entries + shelf_entries, np.concatenate([db_vecs, shelf_vecs])
            print(f"catalog.use_shelf is true: + {len(shelf_entries)} shelf entries from owner-labeled TRAIN crops (caps {sc['cap_per_product']} per product, {sc['cap_unknown']} unknown negatives)")
        n = catalog_db.reload_entries(conn, db_entries, db_vecs)
        print(f"database {db['name']} ({'created' if created else 'existed'}): {n} rows loaded, cosine HNSW index; per product (originals, augmented): {catalog_db.counts(conn)}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
