"""Check the catalog product photos folder and its sources CSV (see docs/catalog_sources_template.csv).

Layout: data/catalog/product_photos/<product id>/<product id>_NN.<ext> plus data/catalog/product_photos/sources.csv.
Errors (exit 1): wrong folder or file names, a photo without a sources row, empty source_url or license, a bad
uruguay_packaging_checked value, unreadable or duplicate images, fewer than the minimum photos for a product.
Warnings: small images, packaging not confirmed, fewer photos than the target.

Usage: python scripts/check_catalog_photos.py
"""
import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.embeddings.catalog_files import check_catalog_photos  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage6.yaml")
    parser.add_argument("--init", action="store_true", help="create the empty folders (one per product id) and a header-only sources.csv, then exit")
    args = parser.parse_args()
    cfg = yaml.safe_load((REPO_ROOT / args.config).read_text())
    c = cfg["catalog"]
    if args.init:
        root = REPO_ROOT / c["photos_dir"]
        for product in cfg["products"]:
            (root / product).mkdir(parents=True, exist_ok=True)
        sources = REPO_ROOT / c["sources_csv"]
        if not sources.exists():
            sources.write_text(",".join(c["required_columns"] + ["material", "size", "notes"]) + "\n")
        print(f"created {len(cfg['products'])} folders in {c['photos_dir']}/ and {Path(c['sources_csv']).name} (header only)")
        return 0
    errors, warnings, counts = check_catalog_photos(REPO_ROOT / c["photos_dir"], REPO_ROOT / c["sources_csv"], cfg["products"], c["image_extensions"],
                                                    c["required_columns"], c["min_short_side_px"], c["min_per_product"], c["target_per_product"])
    print("photos per product: " + ", ".join(f"{p} {counts[p]}" for p in cfg["products"]))
    for w in warnings:
        print("warning:", w)
    for e in errors:
        print("ERROR:", e)
    print(f"catalog photos: {sum(counts.values())} photos, {len(errors)} error(s), {len(warnings)} warning(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
