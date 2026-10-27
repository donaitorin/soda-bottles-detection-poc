"""Speed of SAM 2 (box prompts) on the real device, before anything is built on it (experiment 4b).

Takes a few labeled val photos (anonymous ids), prompts each with the pseudo-label boxes of the val crop sample and prints seconds
per photo and per box on MPS (and CPU for one photo). Nothing is written.

Usage: python scripts/sam2_timing.py
"""
import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.common.device import get_device  # noqa: E402
from src.data.images import load_image  # noqa: E402
from src.embeddings.sam_masks import load_sam2, masks_for_boxes  # noqa: E402


def sync(d):
    if d.type == "mps":
        torch.mps.synchronize()


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    cfg = yaml.safe_load((REPO_ROOT / "configs/stage6.yaml").read_text())
    s1 = yaml.safe_load((REPO_ROOT / "configs/stage1.yaml").read_text())
    names = {r["photo_id"]: r["file_name"] for r in csv.DictReader((REPO_ROOT / s1["id_maps_dir"] / "photo_ids.csv").open())}
    rows = list(csv.DictReader((REPO_ROOT / cfg["crops"]["mapping_dir"] / cfg["crops"]["sample"]["name"] / "mapping.csv").open()))
    by_photo = defaultdict(list)
    for r in rows:
        by_photo[r["photo_id"]].append([float(r[k]) for k in ("x1", "y1", "x2", "y2")])
    photos = sorted(by_photo, key=lambda p: -len(by_photo[p]))[:3]
    for dev in (get_device(), torch.device("cpu")):
        t0 = time.perf_counter()
        model, proc = load_sam2(cfg["experiments"]["sam2"], dev, REPO_ROOT)
        masks_for_boxes(model, proc, load_image(REPO_ROOT / s1["raw_dir"] / names[photos[0]]), np.array(by_photo[photos[0]][:2]), dev)   # warm-up
        sync(dev)
        print(f"{dev}: load + warm-up {time.perf_counter() - t0:.1f} s")
        for pid in photos[: 3 if dev.type != "cpu" else 1]:
            img = load_image(REPO_ROOT / s1["raw_dir"] / names[pid])
            t = time.perf_counter()
            m = masks_for_boxes(model, proc, img, np.array(by_photo[pid]), dev)
            sync(dev)
            dt = time.perf_counter() - t
            area = np.mean([mm.sum() / max(1, (b[2] - b[0]) * (b[3] - b[1])) for mm, b in zip(m, by_photo[pid])])
            print(f"  {pid}: {len(by_photo[pid])} boxes, {img.width}x{img.height}: {dt:.2f} s ({1000 * dt / len(by_photo[pid]):.0f} ms per box); mask area / box area {area:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
