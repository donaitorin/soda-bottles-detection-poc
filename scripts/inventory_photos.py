"""Read-only inventory of data/raw: formats, resolutions, EXIF orientation,
and whether file names allow grouping photos by store / visit.

Never modifies data/raw. Privacy: prints only counts, never GPS values or EXIF
personal data (it only reports whether GPS tags exist).

Usage: python scripts/inventory_photos.py --config configs/photo_inventory.yaml
"""
import argparse
import csv
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
EXIF_ORIENTATION, EXIF_GPS_IFD, EXIF_DATETIME_ORIGINAL, EXIF_EXIF_IFD = 274, 0x8825, 36867, 0x8769


def inspect(path: Path) -> dict:
    row = {"file": path.name, "ext": path.suffix, "width": "", "height": "",
           "orientation": "", "has_gps": "", "exif_date": ""}
    try:
        with Image.open(path) as im:  # lazy: reads the header, not the pixels
            row["width"], row["height"] = im.size
            exif = im.getexif()
            row["orientation"] = exif.get(EXIF_ORIENTATION, 1)
            row["has_gps"] = bool(exif.get_ifd(EXIF_GPS_IFD))
            row["exif_date"] = str(exif.get_ifd(EXIF_EXIF_IFD).get(EXIF_DATETIME_ORIGINAL, ""))[:10]
    except Exception as err:  # HEIC and corrupt files end up here
        row["error"] = type(err).__name__
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/photo_inventory.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load((REPO_ROOT / args.config).read_text())
    files = sorted(p for p in (REPO_ROOT / cfg["raw_dir"]).iterdir()
                   if p.is_file() and p.name != ".gitkeep" and not p.name.startswith(".DS_"))
    rows = [inspect(p) for p in files]
    print(f"files: {len(rows)}")
    print("extensions (case-sensitive):", dict(Counter(r["ext"] for r in rows)))
    errors = [r for r in rows if "error" in r]
    print(f"could not open with Pillow: {len(errors)}", dict(Counter(r['ext'] for r in errors)))

    ok = [r for r in rows if "error" not in r]
    if ok:
        long_side = [max(r["width"], r["height"]) for r in ok]
        print(f"long side px: min {min(long_side)}, median {int(statistics.median(long_side))}, max {max(long_side)}")
        print("most common WxH:", Counter(f"{r['width']}x{r['height']}" for r in ok).most_common(cfg["top_resolutions"]))
        print("portrait / landscape / square:",
              sum(r["height"] > r["width"] for r in ok), sum(r["width"] > r["height"] for r in ok),
              sum(r["width"] == r["height"] for r in ok))
        print("EXIF orientation values (1 = normal):", dict(Counter(r["orientation"] for r in ok)))
        print(f"photos with GPS tags: {sum(r['has_gps'] for r in ok)} (values not read or printed)")
        print(f"photos with EXIF capture date: {sum(bool(r['exif_date']) for r in ok)}")

    # Grouping by store / visit from the file name
    store_re, date_re = re.compile(cfg["store_regex"]), re.compile(cfg["date_regex"])
    for r in rows:
        m, d = store_re.search(r["file"]), date_re.search(r["file"])
        r["store"], r["visit_date"] = (m["store"] if m else ""), (d["date"] if d else "")
    print(f"names with a store id: {sum(bool(r['store']) for r in rows)}/{len(rows)}; "
          f"with a date: {sum(bool(r['visit_date']) for r in rows)}/{len(rows)}")
    named = [r for r in rows if r["store"]]  # photos whose name does not match have no store id
    per_store = Counter(r["store"] for r in named)
    visits = Counter((r["store"], r["visit_date"]) for r in named)
    print("names without a store id:", [r["file"][:30] for r in rows if not r["store"]])
    print(f"distinct stores: {len(per_store)}; photos per store: min {min(per_store.values())}, "
          f"median {int(statistics.median(per_store.values()))}, max {max(per_store.values())}")
    print(f"distinct (store, date) visits: {len(visits)}; stores with more than one visit date: "
          f"{sum(1 for s in per_store if len({v for (st, v) in visits if st == s}) > 1)}")
    print("photos per store, largest 5:", per_store.most_common(5))

    out = REPO_ROOT / cfg["output_csv"]
    fields = ["file", "ext", "width", "height", "orientation", "has_gps", "exif_date", "store", "visit_date", "error"]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"csv: {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
