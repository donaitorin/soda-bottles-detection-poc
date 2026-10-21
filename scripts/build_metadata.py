"""Build data/metadata.csv (gitignored), keyed by photo id. Read-only on data/raw.

Columns: photo_id, store_code, ext, width, height, long_side, visit_date,
screenshot, low_res, exact_duplicate, near_duplicate_candidate.
The two duplicate flags are filled by scripts/find_duplicates.py and are kept
when this script is rerun. Prints aggregates only (no file names).

Usage: python scripts/build_metadata.py --config configs/stage1.yaml
"""
import argparse
import csv
import statistics
import sys
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.ids import parse_name  # noqa: E402

FIELDS = ["photo_id", "store_code", "ext", "width", "height", "long_side", "visit_date",
          "screenshot", "low_res", "exact_duplicate", "near_duplicate_candidate"]
EXIF_ORIENTATION = 274
SWAPPING_ORIENTATIONS = {5, 6, 7, 8}  # these rotate by 90 degrees, so width and height swap


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage1.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load((REPO_ROOT / args.config).read_text())
    maps_dir = REPO_ROOT / cfg["id_maps_dir"]
    out_path = REPO_ROOT / cfg["metadata_csv"]

    previous = {}
    if out_path.exists():
        previous = {r["photo_id"]: r for r in csv.DictReader(out_path.open(newline=""))}

    rows = []
    for item in csv.DictReader((maps_dir / "photo_ids.csv").open(newline="")):
        path = REPO_ROOT / cfg["raw_dir"] / item["file_name"]
        with Image.open(path) as im:  # reads the header only
            width, height = im.size
            if im.getexif().get(EXIF_ORIENTATION, 1) in SWAPPING_ORIENTATIONS:
                width, height = height, width  # size as the photo is displayed
        info = parse_name(item["file_name"], cfg["name_regex"])
        old = previous.get(item["photo_id"], {})
        rows.append({
            "photo_id": item["photo_id"], "store_code": item["store_code"], "ext": path.suffix.lower(),
            "width": width, "height": height, "long_side": max(width, height),
            "visit_date": info["date"] if info else "",
            "screenshot": item["store_code"] == cfg["no_store_code"],
            "low_res": max(width, height) < cfg["low_res_long_side_px"],
            "exact_duplicate": old.get("exact_duplicate", ""),
            "near_duplicate_candidate": old.get("near_duplicate_candidate", ""),
        })

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    long_side = [r["long_side"] for r in rows]
    print(f"photos: {len(rows)} | screenshots (store code {cfg['no_store_code']}): {sum(r['screenshot'] for r in rows)}"
          f" | low-res (<{cfg['low_res_long_side_px']} px): {sum(r['low_res'] for r in rows)}")
    print(f"long side px: min {min(long_side)}, median {int(statistics.median(long_side))}, max {max(long_side)}")
    per_store = Counter(r["store_code"] for r in rows)
    print("photos per store code:", dict(sorted(per_store.items())))
    print(f"csv: {out_path.relative_to(REPO_ROOT)} (gitignored)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
