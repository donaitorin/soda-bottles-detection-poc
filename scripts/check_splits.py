"""Check the Stage 1 gate on data/splits and print the per-split summary.

Fails (exit 1) if a store or photo is in two splits, an id is malformed or unknown, a screenshot
(sx_) id is in a split, a non-excluded photo is in no split, or the shares are outside the
tolerance. Prints only counts and store codes, never file names.

Usage: python scripts/check_splits.py
"""
import argparse
import csv
import statistics
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.splits import SPLITS, validate_splits  # noqa: E402


def read_ids(path: Path) -> list[str]:
    return [l.strip() for l in path.read_text().split("\n") if l.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage1.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load((REPO_ROOT / args.config).read_text())
    sp = cfg["splits"]

    meta = {r["photo_id"]: r for r in csv.DictReader((REPO_ROOT / cfg["metadata_csv"]).open(newline=""))}
    photo_store = {i: r["store_code"] for i, r in meta.items()}
    splits = {s: read_ids(REPO_ROOT / sp["dir"] / f"{s}.txt") for s in SPLITS}
    excluded = read_ids(REPO_ROOT / cfg["exclusions"]["path"])

    errors = validate_splits(splits, excluded, photo_store, sp["id_regex"], sp["excluded_id_regex"],
                             cfg["no_store_code"], sp["fractions"], sp["tolerance_points"])

    total = sum(len(v) for v in splits.values())
    print(f"{'split':6} {'photos':>6} {'share':>6} {'stores':>6} {'visits':>6}  long side px (min / median / max)")
    for name in SPLITS:
        rows = [meta[i] for i in splits[name] if i in meta]
        stores = {r["store_code"] for r in rows}
        visits = {(r["store_code"], r["visit_date"]) for r in rows}
        sides = [int(r["long_side"]) for r in rows] or [0]
        print(f"{name:6} {len(rows):>6} {100 * len(rows) / max(total, 1):>5.1f}% {len(stores):>6} {len(visits):>6}  "
              f"{min(sides)} / {int(statistics.median(sides))} / {max(sides)}")
    print(f"excluded: {len(excluded)} | photos in splits: {total} | photos in mapping: {len(meta)}")
    if errors:
        print("\nFAILED:")
        for e in errors:
            print(" -", e)
        return 1
    print("OK: no store or photo is in two splits; ids are well formed; nothing but ids in the files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
