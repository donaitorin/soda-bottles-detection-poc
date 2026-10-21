"""Pre-label all non-excluded photos (train, val and test) with the FROZEN Grounding DINO settings.

Writes, in the gitignored data/prelabels/ (photo ids only):
  predictions.json   model_version + per photo: size, boxes (xyxy pixels), scores
  run_summary.csv    per photo: split, raw boxes, kept boxes, drops by reason
  box_heights.csv    per box before the tiny-box filter: photo id, split, image size, box height
Labeling val and test is labeling, not tuning: nothing here is used to change a setting. The statistics that
inform the minimum-height rule are computed from TRAIN photos only.

Usage: python scripts/prelabel_run.py [--config configs/prelabel.yaml]
"""
import argparse
import csv
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.common.device import get_device  # noqa: E402
from src.data.images import load_image  # noqa: E402
from src.data.paths import check_output_dir  # noqa: E402
from src.prelabel import grounding_dino  # noqa: E402
from src.prelabel.filters import filter_boxes, wedge_mask  # noqa: E402
from src.prelabel.labelstudio import model_version  # noqa: E402

DROP_REASONS = ["oversized", "wedge_dropped", "wedge_clipped", "tiny", "duplicate_nms", "duplicate_contained"]


def read_ids(path: Path) -> list[str]:
    return [l.strip() for l in path.read_text().split("\n") if l.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/prelabel.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load((REPO_ROOT / args.config).read_text())
    s1 = yaml.safe_load((REPO_ROOT / "configs/stage1.yaml").read_text())
    assert cfg["prelabeler"] == "grounding_dino"
    out_dir = REPO_ROOT / cfg["run"]["output_dir"]
    check_output_dir(out_dir, [REPO_ROOT / d for d in s1["protected_dirs"]])
    out_dir.mkdir(parents=True, exist_ok=True)

    split_of = {}
    for split in ("train", "val", "test"):
        for pid in read_ids(REPO_ROOT / s1["splits"]["dir"] / f"{split}.txt"):
            split_of[pid] = split
    names = {r["photo_id"]: r["file_name"] for r in csv.DictReader((REPO_ROOT / s1["id_maps_dir"] / "photo_ids.csv").open())}
    device = get_device() if cfg["device"] == "auto" else torch.device(cfg["device"])
    dcfg, fcfg = cfg["dino"], cfg["filters"]
    version = model_version(cfg)
    print(f"{version} | device {device} | photos {len(split_of)}")
    model, processor = grounding_dino.load_model(dcfg, device, REPO_ROOT)

    wedge_args = {k: fcfg["wedge"][k] for k in ("value", "tolerance", "min_component_fraction")}
    predictions, summary, heights = {}, [], []
    start = time.perf_counter()
    for k, pid in enumerate(sorted(split_of), 1):
        image = load_image(REPO_ROOT / s1["raw_dir"] / names[pid])
        mask = wedge_mask(np.asarray(image), **wedge_args)
        raw_boxes, raw_scores = grounding_dino.detect(model, processor, image, dcfg["prompt"], device,
                                                      dcfg["box_threshold"], dcfg["text_threshold"])
        boxes, scores, drops = filter_boxes(raw_boxes, raw_scores, image.size, mask, fcfg)
        # Box heights before the tiny-box floor, to study the floor (same filters, floor at 0).
        no_floor, _, _ = filter_boxes(raw_boxes, raw_scores, image.size, mask, dict(fcfg, min_height_fraction=0.0))
        heights += [(pid, split_of[pid], image.width, image.height, float(b[3] - b[1])) for b in no_floor]
        predictions[pid] = {"width": image.width, "height": image.height,
                            "boxes": [[round(float(v), 2) for v in b] for b in boxes],
                            "scores": [round(float(s), 4) for s in scores]}
        summary.append({"photo_id": pid, "split": split_of[pid], "width": image.width, "height": image.height,
                        "raw_boxes": len(raw_boxes), "kept_boxes": len(boxes), **{f"drop_{r}": drops[r] for r in DROP_REASONS}})
        if k % 25 == 0:
            print(f"  {k}/{len(split_of)} photos, {time.perf_counter() - start:.0f} s", flush=True)

    (out_dir / "predictions.json").write_text(json.dumps({"model_version": version, "photos": predictions}))
    for name, rows, fields in (("run_summary.csv", summary, list(summary[0].keys())),
                               ("box_heights.csv", [dict(zip(("photo_id", "split", "width", "height", "box_height_px"), h)) for h in heights],
                                ["photo_id", "split", "width", "height", "box_height_px"])):
        with (out_dir / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    total = Counter()
    for r in summary:
        total["raw"] += r["raw_boxes"]
        total["kept"] += r["kept_boxes"]
        for reason in DROP_REASONS:
            total[reason] += r[f"drop_{reason}"]
    per_split = {s: sum(r["kept_boxes"] for r in summary if r["split"] == s) for s in ("train", "val", "test")}
    print(f"done in {time.perf_counter() - start:.0f} s | raw boxes {total['raw']} -> kept {total['kept']} {per_split}")
    print("drops:", {r: total[r] for r in DROP_REASONS},
          f"| containment share of raw boxes: {100 * total['duplicate_contained'] / total['raw']:.1f}%")
    print(f"boxes per photo: median {statistics.median(r['kept_boxes'] for r in summary)}, "
          f"max {max(r['kept_boxes'] for r in summary)} | photos with 0 boxes: {sum(r['kept_boxes'] == 0 for r in summary)}")
    print(f"written to {out_dir.relative_to(REPO_ROOT)}/ (gitignored)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
