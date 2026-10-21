"""Time a pre-labeler on one train photo, MPS vs CPU, and check both devices agree.

Method as in Stage 0: warm-up runs, then the median of several timed runs.
Usage: python scripts/prelabel_timing.py --model dino|frcnn [--photo-id s07_cfed3ea4]
"""
import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.common.device import get_device  # noqa: E402,F401  (sets the MPS fallback before torch loads ops)
from src.data.images import load_image  # noqa: E402
from src.prelabel import frcnn, grounding_dino  # noqa: E402
from src.prelabel.filters import box_iou  # noqa: E402


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["dino", "frcnn"], required=True)
    parser.add_argument("--photo-id")
    parser.add_argument("--config", default="configs/prelabel.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load((REPO_ROOT / args.config).read_text())
    s1 = yaml.safe_load((REPO_ROOT / "configs/stage1.yaml").read_text())
    photo_id = args.photo_id or cfg["timing"]["photo_id"]
    train = {l.strip() for l in (REPO_ROOT / s1["splits"]["dir"] / "train.txt").read_text().split("\n")}
    assert photo_id in train, "timing and comparison use train photos only"  # val and test are never used here
    names = {r["photo_id"]: r["file_name"] for r in csv.DictReader((REPO_ROOT / s1["id_maps_dir"] / "photo_ids.csv").open())}
    image = load_image(REPO_ROOT / s1["raw_dir"] / names[photo_id])
    print(f"photo {photo_id} ({image.width}x{image.height}), model {args.model}")

    results = {}
    for device in (torch.device("mps"), torch.device("cpu")):
        if args.model == "dino":
            model, processor = grounding_dino.load_model(cfg["dino"], device, REPO_ROOT)
            run = lambda: grounding_dino.detect(model, processor, image, cfg["dino"]["prompt"], device,  # noqa: E731
                                                cfg["dino"]["box_threshold"], cfg["dino"]["text_threshold"])
        else:
            model, target_id = frcnn.load_model(cfg["frcnn"], device, REPO_ROOT)
            run = lambda: frcnn.detect(model, target_id, image, device)  # noqa: E731
        try:
            for _ in range(cfg["timing"]["warmup_runs"]):
                run()
            times = []
            for _ in range(cfg["timing"]["timed_runs"]):
                sync(device)
                start = time.perf_counter()
                out = run()
                sync(device)
                times.append(time.perf_counter() - start)
        except Exception as err:  # an unsupported MPS operation ends up here
            print(f"{device}: FAILED with {type(err).__name__}: {str(err)[:200]}")
            continue
        results[device.type] = out
        print(f"{device}: median {statistics.median(times):.2f} s over {len(times)} runs "
              f"(min {min(times):.2f}, max {max(times):.2f}) | {len(out[0])} boxes")
        del model
    if "mps" in results and "cpu" in results:
        (bm, sm), (bc, sc) = results["mps"], results["cpu"]
        iou = box_iou(bm, bc)
        matched = int((iou.max(axis=1) >= 0.9).sum()) if iou.size else 0
        print(f"agreement: {matched}/{len(bm)} MPS boxes have a CPU box with IoU >= 0.9 (CPU has {len(bc)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
