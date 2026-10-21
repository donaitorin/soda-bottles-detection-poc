"""Write data/splits/excluded.txt: photo ids that never enter a split.

Always excluded: the screenshots (store code sx). Duplicate drops from
data/review/duplicates/decisions.csv are applied ONLY with --apply-duplicate-decisions,
i.e. after the owner has filled in and confirmed that file. Nothing in data/raw is
deleted or moved; the list only says which photos the splits skip.

Usage: python scripts/make_exclusions.py [--apply-duplicate-decisions]
"""
import argparse
import csv
import re
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage1.yaml")
    parser.add_argument("--apply-duplicate-decisions", action="store_true",
                        help="also exclude the photos the owner marked drop_a / drop_b")
    args = parser.parse_args()
    cfg = yaml.safe_load((REPO_ROOT / args.config).read_text())

    meta = list(csv.DictReader((REPO_ROOT / cfg["metadata_csv"]).open(newline="")))
    excluded = {r["photo_id"] for r in meta if r["screenshot"] == "True"}
    print(f"screenshots: {len(excluded)}")

    decisions_path = REPO_ROOT / cfg["exclusions"]["duplicate_decisions"]
    rows = list(csv.DictReader(decisions_path.open(newline=""))) if decisions_path.exists() else []
    drops = [(r["id_b"] if r["decision"] == "drop_b" else r["id_a"]) for r in rows
             if r["decision"] in ("drop_a", "drop_b")]
    undecided = [r for r in rows if r["decision"] == "" and r["same_store"] == "False"]
    print(f"duplicate pairs in decisions.csv: {len(rows)} | drops marked: {len(drops)} | "
          f"undecided cross-store pairs: {len(undecided)}")
    if args.apply_duplicate_decisions:
        if undecided:
            print("error: cross-store pairs without a decision; the owner must decide them first")
            return 1
        excluded |= set(drops)
    elif drops:
        print("note: drops are NOT applied (run with --apply-duplicate-decisions after confirming)")

    pattern = re.compile(cfg["splits"]["excluded_id_regex"])
    assert all(pattern.fullmatch(i) for i in excluded), "malformed id in the exclusion list"
    out = REPO_ROOT / cfg["exclusions"]["path"]
    out.write_text("".join(f"{i}\n" for i in sorted(excluded)))
    print(f"excluded: {len(excluded)} photo ids -> {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
