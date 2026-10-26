"""Check the folder of catalog product photos and its sources CSV (data/catalog/product_photos, ignored by git).

Layout:  <photos_dir>/<product id>/<product id>_<NN>.<ext>   and   <photos_dir>/sources.csv
Every photo needs a row in sources.csv (file, product_id, source_url, license, uruguay_packaging_checked, ...).
"""
import csv
import hashlib
import re
from collections import Counter
from pathlib import Path

from PIL import Image


def check_catalog_photos(photos_dir: Path, sources_csv: Path, products: list[str], extensions: list[str], required_columns: list[str],
                         min_short_side: int, min_per_product: int, target_per_product: int) -> tuple[list[str], list[str], Counter]:
    """Returns (errors, warnings, photos per product). Errors must be fixed; warnings are reported."""
    errors, warnings, counts = [], [], Counter()
    if not photos_dir.is_dir():
        return ["the catalog photos folder does not exist (python scripts/check_catalog_photos.py --init creates it)"], [], counts
    rows = {}
    if not sources_csv.exists():
        errors.append("sources.csv is missing")
    else:
        with sources_csv.open(newline="") as f:
            reader = csv.DictReader(f)
            missing = [c for c in required_columns if c not in (reader.fieldnames or [])]
            if missing:
                errors.append(f"sources.csv lacks the columns: {', '.join(missing)}")
            for r in reader:
                rows[r.get("file", "")] = r
    name_re = {p: re.compile(rf"^{re.escape(p)}_\d{{2}}({'|'.join(re.escape(e) for e in extensions)})$") for p in products}
    seen = {}
    for folder in sorted(p for p in photos_dir.iterdir() if p.is_dir()):
        if folder.name not in products:
            errors.append(f"folder {folder.name}: not one of the 10 product ids")
            continue
        for path in sorted(folder.iterdir()):
            if path.name.startswith("."):
                continue
            rel = f"{folder.name}/{path.name}"
            if not name_re[folder.name].match(path.name):
                errors.append(f"{rel}: name must be {folder.name}_NN with one of {extensions}")
                continue
            counts[folder.name] += 1
            try:
                with Image.open(path) as im:
                    short = min(im.size)
                    im.verify()
            except Exception:  # noqa: BLE001
                errors.append(f"{rel}: cannot be opened as an image")
                continue
            if short < min_short_side:
                warnings.append(f"{rel}: short side {short} px (under {min_short_side})")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in seen:
                errors.append(f"{rel}: identical file to {seen[digest]}")
            seen[digest] = rel
            row = rows.get(rel)
            if row is None:
                errors.append(f"{rel}: no row in sources.csv")
                continue
            if row.get("product_id") != folder.name:
                errors.append(f"{rel}: product_id in sources.csv is '{row.get('product_id')}'")
            if not (row.get("source_url") or "").strip():
                errors.append(f"{rel}: source_url is empty")
            if not (row.get("license") or "").strip():
                errors.append(f"{rel}: license is empty (write 'unknown' if it is not known)")
            if (row.get("uruguay_packaging_checked") or "").strip().lower() not in ("yes", "no"):
                errors.append(f"{rel}: uruguay_packaging_checked must be yes or no")
            elif row["uruguay_packaging_checked"].strip().lower() == "no":
                warnings.append(f"{rel}: Uruguayan packaging not confirmed")
    for rel in rows:
        if rel and not (photos_dir / rel).exists():
            errors.append(f"sources.csv lists {rel}, which does not exist")
    for p in products:
        if counts[p] < min_per_product:
            errors.append(f"{p}: {counts[p]} photos (at least {min_per_product} needed)")
        elif counts[p] < target_per_product:
            warnings.append(f"{p}: {counts[p]} photos (target {target_per_product})")
    return errors, warnings, counts
