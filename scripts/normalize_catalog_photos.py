"""Put the catalog photos in the layout the check expects (one-time helper; dry run by default).

1. Removes the EMPTY folders of products that are not in the target list (a folder with files is an error, never deleted).
2. Renames each photo to <product id>_NN.<ext>: sorted by original name, NN from 01, lowercase extension. Nothing is
   overwritten (two-phase rename through temporary names) and the old to new names are printed.
3. Adds one row per renamed photo to sources.csv: source_url and license `unknown`, uruguay_packaging_checked `no`, the original
   name in `notes`. Update those rows by hand when the sources are known (see the handoff).

Usage: python scripts/normalize_catalog_photos.py [--apply]
"""
import argparse
import csv
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.embeddings.catalog_files import plan_renames  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage6.yaml")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load((REPO_ROOT / args.config).read_text())
    c = cfg["catalog"]
    root = REPO_ROOT / c["photos_dir"]
    exts = set(c["image_extensions"])

    for folder in sorted(p for p in root.iterdir() if p.is_dir() and p.name not in cfg["products"]):
        files = [f for f in folder.iterdir() if not f.name.startswith(".")]
        if files:
            raise SystemExit(f"{folder.name}: not a target product and not empty ({len(files)} files): refusing to delete it")
        print(f"remove empty folder {folder.name}/")
        if args.apply:
            for hidden in folder.glob(".*"):
                hidden.unlink()
            folder.rmdir()

    plans, rows = {}, []
    for product in cfg["products"]:
        folder = root / product
        names = [f.name for f in folder.iterdir() if f.is_file() and not f.name.startswith(".")] if folder.is_dir() else []
        bad = [n for n in names if Path(n).suffix.lower() not in exts]
        if bad:
            raise SystemExit(f"{product}: files with an unexpected extension: {bad}")
        plans[product] = plan_renames(product, names)
    print(f"{'old name':40} -> new name")
    for product, plan in plans.items():
        for old, new in plan:
            print(f"{product}/{old:40} -> {product}/{new}")
            rows.append({"file": f"{product}/{new}", "product_id": product, "source_url": "unknown", "license": "unknown",
                         "uruguay_packaging_checked": "no", "material": "", "size": "", "notes": f"original name: {old}"})
    if not args.apply:
        print("dry run: nothing changed (use --apply)")
        return 0
    for product, plan in plans.items():  # phase 1: temporary names, so a new name can never overwrite an old one
        for i, (old, _) in enumerate(plan):
            (root / product / old).rename(root / product / f".tmp_{i:02d}")
    for product, plan in plans.items():
        for i, (_, new) in enumerate(plan):
            target = root / product / new
            if target.exists():
                raise SystemExit(f"{target.name} already exists: refusing to overwrite")
            (root / product / f".tmp_{i:02d}").rename(target)
    sources = REPO_ROOT / c["sources_csv"]
    existing = list(csv.DictReader(sources.open(newline=""))) if sources.exists() else []
    known = {r["file"] for r in existing}
    fields = c["required_columns"] + ["material", "size", "notes"]
    with sources.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(existing + [r for r in rows if r["file"] not in known])
    print(f"renamed {len(rows)} photos and wrote {len(rows)} rows to {sources.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
