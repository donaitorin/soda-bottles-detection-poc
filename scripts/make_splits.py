"""Build data/splits/{train,val,test}.txt: photo ids, split strictly by store.

Photos of one store always land in the same split, because two photos of the same shelf
(also on two different days) look almost identical and would leak between train and test.
Excluded photos (data/splits/excluded.txt) are skipped. The search is deterministic for a
given seed (configs/stage1.yaml).

Usage: python scripts/make_splits.py [--config configs/stage1.yaml]
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.splits import SPLITS, search_assignment  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage1.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load((REPO_ROOT / args.config).read_text())
    sp = cfg["splits"]

    excluded = {l.strip() for l in (REPO_ROOT / cfg["exclusions"]["path"]).read_text().split("\n") if l.strip()}
    meta = [r for r in csv.DictReader((REPO_ROOT / cfg["metadata_csv"]).open(newline=""))
            if r["photo_id"] not in excluded]
    photos_of = defaultdict(list)
    visits_of = defaultdict(set)
    for r in meta:
        photos_of[r["store_code"]].append(r["photo_id"])
        visits_of[r["store_code"]].add(r["visit_date"])
    stores = {s: {"photos": len(p), "visits": len(visits_of[s])} for s, p in photos_of.items()}

    force_train = ()
    if sp["largest_store_to_train"]:
        force_train = (max(stores, key=lambda s: (stores[s]["photos"], s)),)
    result = search_assignment(stores, sp["fractions"], sp["tolerance_points"], sp["n_shuffles"],
                               sp["seed"], force_train)

    out_dir = REPO_ROOT / sp["dir"]
    total = len(meta)
    for split in SPLITS:
        ids = sorted(i for s, name in result["assignment"].items() if name == split for i in photos_of[s])
        (out_dir / f"{split}.txt").write_text("".join(f"{i}\n" for i in ids))
        stores_in = sorted(s for s, name in result["assignment"].items() if name == split)
        print(f"{split}: {len(ids)} photos ({100 * len(ids) / total:.1f}%) | {len(stores_in)} stores | "
              f"{sum(stores[s]['visits'] for s in stores_in)} visits | stores: {' '.join(stores_in)}")
    print(f"seed {sp['seed']} | forced to train: {force_train} | valid assignments found: "
          f"{result['n_valid_assignments_found']} of {sp['n_shuffles']} shuffles | "
          f"max deviation {result['deviation']:.2f} points")
    print(f"written to {out_dir.relative_to(REPO_ROOT)}/ (photo ids only)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
