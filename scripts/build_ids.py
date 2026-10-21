"""Build the anonymous id mappings for data/raw (read-only on the photos).

Writes, in the gitignored data/id_maps/:
  store_codes.csv  store_code, store_id
  photo_ids.csv    photo_id, file_name, store_code
Existing mappings are authoritative: known stores and photos keep their ids.
Prints only counts, never file names.

Usage: python scripts/build_ids.py --config configs/stage1.yaml
"""
import argparse
import csv
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.ids import assign_store_codes, make_photo_id, parse_name  # noqa: E402


def read_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open(newline=""))) if path.exists() else []


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage1.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load((REPO_ROOT / args.config).read_text())
    maps_dir = REPO_ROOT / cfg["id_maps_dir"]
    maps_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(p.name for p in (REPO_ROOT / cfg["raw_dir"]).iterdir()
                   if p.is_file() and p.name != ".gitkeep" and not p.name.startswith(".DS_"))
    parsed = {name: parse_name(name, cfg["name_regex"]) for name in files}

    existing_codes = {r["store_id"]: r["store_code"] for r in read_csv(maps_dir / "store_codes.csv")}
    store_ids = {p["store"] for p in parsed.values() if p}
    codes = assign_store_codes(existing_codes, store_ids, cfg["store_code_prefix"], cfg["store_code_width"])

    existing_photos = {r["file_name"]: r["photo_id"] for r in read_csv(maps_dir / "photo_ids.csv")}
    rows, seen = [], {}
    for name in files:
        info = parsed[name]
        store_code = codes[info["store"]] if info else cfg["no_store_code"]
        photo_id = make_photo_id(store_code, name, cfg["id_hash_length"])
        if name in existing_photos and existing_photos[name] != photo_id:
            raise SystemExit("an existing photo id would change; the mapping is authoritative, aborting")
        if photo_id in seen:
            raise SystemExit(f"hash collision between two different files (id {photo_id}); raise id_hash_length")
        seen[photo_id] = name
        rows.append({"photo_id": photo_id, "file_name": name, "store_code": store_code})

    write_csv(maps_dir / "store_codes.csv", ["store_code", "store_id"],
              [{"store_code": c, "store_id": s} for s, c in sorted(codes.items(), key=lambda kv: kv[1])])
    write_csv(maps_dir / "photo_ids.csv", ["photo_id", "file_name", "store_code"], rows)
    n_new = sum(1 for r in rows if r["file_name"] not in existing_photos)
    print(f"photos: {len(rows)} ({n_new} new) | stores: {len(codes)} | "
          f"without store ({cfg['no_store_code']}): {sum(r['store_code'] == cfg['no_store_code'] for r in rows)}")
    print(f"mappings written to {maps_dir.relative_to(REPO_ROOT)}/ (gitignored)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
