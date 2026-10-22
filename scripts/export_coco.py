"""Export COCO files (train/val/test) from the pre-labels, merging the owner's Label Studio annotations if given.

Without --ls-export every photo is `prelabel_unverified` (Grounding DINO pseudo-labels). With a Label Studio JSON
export (Export -> JSON, NOT the built-in COCO export, which drops the per-photo product choices and lead_time),
a photo with a submitted annotation becomes `owner_corrected` and its boxes replace the pre-label.
The export is also copied, untouched, to data/annotations/ls_backups/ with a UTC timestamp (the backup routine).
Boxes under the minimum height (configs/prelabel.yaml, filters.min_height_fraction) are dropped and counted.
Outputs go to the gitignored data/annotations/ and hold photo ids only.

Usage: python scripts/export_coco.py [--ls-export PATH]
"""
import argparse
import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.annotations.export import DISCLAIMER, build_coco, read_owner_annotations  # noqa: E402
from src.data.paths import check_output_dir  # noqa: E402
from src.prelabel.labelstudio import model_version  # noqa: E402


def read_ids(path: Path) -> list[str]:
    return [l.strip() for l in path.read_text().split("\n") if l.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage3.yaml")
    parser.add_argument("--ls-export", help="Label Studio JSON export (full JSON format)")
    args = parser.parse_args()
    cfg = yaml.safe_load((REPO_ROOT / args.config).read_text())
    pcfg = yaml.safe_load((REPO_ROOT / cfg["prelabel_config"]).read_text())
    s1 = yaml.safe_load((REPO_ROOT / "configs/stage1.yaml").read_text())
    ex = cfg["export"]
    protected = [REPO_ROOT / d for d in s1["protected_dirs"]]
    out_dir, backup_dir = REPO_ROOT / ex["output_dir"], REPO_ROOT / ex["backup_dir"]
    for d in (out_dir, backup_dir):
        check_output_dir(d, protected)
        d.mkdir(parents=True, exist_ok=True)

    data = json.loads((REPO_ROOT / cfg["predictions"]).read_text())
    assert data["model_version"] == model_version(pcfg), "predictions.json was made with different frozen settings"
    predictions = data["photos"]
    store_of = {r["photo_id"]: r["store_code"] for r in csv.DictReader((REPO_ROOT / s1["id_maps_dir"] / "photo_ids.csv").open())}
    images_dir = REPO_ROOT / pcfg["label_studio"]["images_dir"]
    link_of = {p.stem: p.name for p in images_dir.iterdir() if p.suffix.lower() in ex["image_extensions"]}  # <id>.<ext>

    owner, notes = {}, {}
    if args.ls_export:
        src = Path(args.ls_export)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        shutil.copyfile(src, backup_dir / f"ls_export_{stamp}.json")
        sizes = {pid: (p["width"], p["height"]) for pid, p in predictions.items()}
        owner, notes = read_owner_annotations(json.loads(src.read_text()), pcfg["label_studio"]["from_name"],
                                              ex["choices_name"], sizes)
        print(f"Label Studio export: {len(owner)} photos with a submitted annotation; notes {dict(notes)}; "
              f"backup copy in {backup_dir.relative_to(REPO_ROOT)}/")

    summary = {"model_version": data["model_version"], "disclaimer": DISCLAIMER, "notes": dict(notes), "splits": {}}
    for split in ("train", "val", "test"):
        ids = read_ids(REPO_ROOT / s1["splits"]["dir"] / f"{split}.txt")
        coco, stats = build_coco(ids, predictions, owner, link_of, store_of, split,
                                 pcfg["filters"]["min_height_fraction"], data["model_version"])
        (out_dir / f"{split}.json").write_text(json.dumps(coco))
        products = {}
        for im in coco["images"]:
            for p in im["products"] or []:
                products[p] = products.get(p, 0) + 1
        summary["splits"][split] = {"photos": len(coco["images"]), "boxes": len(coco["annotations"]),
                                    **dict(sorted(stats.items())), "owner_photos_per_product": products}
        print(f"{split:5} {len(coco['images']):3} photos {len(coco['annotations']):6} boxes | "
              + ", ".join(f"{k} {v}" for k, v in sorted(stats.items()) if v))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"written to {out_dir.relative_to(REPO_ROOT)}/ (gitignored). {DISCLAIMER}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
