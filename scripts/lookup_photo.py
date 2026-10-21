"""Print the path under data/raw of one or more photo ids.

Usage: python scripts/lookup_photo.py s07_3fa2c9d1 [more ids ...]
"""
import argparse
import csv
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("photo_ids", nargs="+")
    parser.add_argument("--config", default="configs/stage1.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load((REPO_ROOT / args.config).read_text())
    with (REPO_ROOT / cfg["id_maps_dir"] / "photo_ids.csv").open(newline="") as f:
        names = {r["photo_id"]: r["file_name"] for r in csv.DictReader(f)}
    status = 0
    for photo_id in args.photo_ids:
        if photo_id in names:
            print(f"{photo_id}\t{Path(cfg['raw_dir']) / names[photo_id]}")
        else:
            print(f"{photo_id}\tNOT FOUND", file=sys.stderr)
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
