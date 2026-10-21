"""Build the Label Studio import file and the photo symlinks. Read-only on data/raw.

- data/labelstudio/images/<photo id>.<ext>: RELATIVE symlinks to the photos in data/raw, so Label Studio
  (which only sees the read-only `data/` mount) shows and stores photo ids, never file names.
- data/labelstudio/tasks.json: one task per photo (train first, then val, then test) with the pre-labels
  as `predictions` in percent coordinates. Import it through the Label Studio UI.
Both are gitignored.

Usage: python scripts/build_ls_tasks.py [--config configs/prelabel.yaml]
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.paths import check_output_dir  # noqa: E402
from src.prelabel.labelstudio import build_task, model_version  # noqa: E402


def read_ids(path: Path) -> list[str]:
    return [l.strip() for l in path.read_text().split("\n") if l.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/prelabel.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load((REPO_ROOT / args.config).read_text())
    s1 = yaml.safe_load((REPO_ROOT / "configs/stage1.yaml").read_text())
    ls = cfg["label_studio"]
    images_dir, tasks_path = REPO_ROOT / ls["images_dir"], REPO_ROOT / ls["tasks_json"]
    protected = [REPO_ROOT / d for d in s1["protected_dirs"]]
    check_output_dir(images_dir, protected)  # never create links inside the photo folders
    check_output_dir(tasks_path.parent, protected)
    images_dir.mkdir(parents=True, exist_ok=True)

    names = {r["photo_id"]: r["file_name"] for r in csv.DictReader((REPO_ROOT / s1["id_maps_dir"] / "photo_ids.csv").open())}
    store_of = {r["photo_id"]: r["store_code"] for r in csv.DictReader((REPO_ROOT / s1["id_maps_dir"] / "photo_ids.csv").open())}
    data = json.loads((REPO_ROOT / cfg["run"]["output_dir"] / "predictions.json").read_text())
    assert data["model_version"] == model_version(cfg), "predictions.json was made with different frozen settings"

    tasks, n_boxes = [], 0
    for split in ("train", "val", "test"):
        for pid in read_ids(REPO_ROOT / s1["splits"]["dir"] / f"{split}.txt"):
            source = REPO_ROOT / s1["raw_dir"] / names[pid]
            link = images_dir / f"{pid}{source.suffix.lower()}"
            target = os.path.relpath(source, images_dir)   # relative, so it also resolves inside the container mount
            if link.is_symlink():
                if os.readlink(link) != target:
                    link.unlink()
            elif link.exists():
                raise SystemExit(f"{link.name} exists and is not a symlink; refusing to overwrite")
            if not link.is_symlink():
                link.symlink_to(target)
            assert link.exists(), "symlink does not resolve"
            pred = data["photos"][pid]
            tasks.append(build_task(pid, split, store_of[pid], f"{ls['url_prefix']}{link.name}",
                                    pred["boxes"], pred["scores"], pred["width"], pred["height"], cfg))
            n_boxes += len(pred["boxes"])

    tasks_path.write_text(json.dumps(tasks))
    print(f"{len(tasks)} tasks ({n_boxes} predicted boxes), model_version {model_version(cfg)}")
    print(f"symlinks in {images_dir.relative_to(REPO_ROOT)}/ and tasks in {tasks_path.relative_to(REPO_ROOT)} (both gitignored)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
