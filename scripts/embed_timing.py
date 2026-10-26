"""Stage 6b: speed of DINOv2 on the real device (MPS), before anything is built on it.

Embeds real crops (catalog crops and shelf crops) at several batch sizes and prints seconds per crop and memory. If MPS is too
slow or fails, this says so; nothing is worked around.

Usage: python scripts/embed_timing.py
"""
import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import torch
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.common.device import get_device  # noqa: E402  (sets the MPS fallback before torch loads)
from src.data.images import load_image  # noqa: E402
from src.embeddings.embed import embed, load_dinov2  # noqa: E402


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage6.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load((REPO_ROOT / args.config).read_text())
    ec = cfg["embeddings"]
    s1 = yaml.safe_load((REPO_ROOT / "configs/stage1.yaml").read_text())
    device = get_device()
    names = {r["photo_id"]: r["file_name"] for r in csv.DictReader((REPO_ROOT / s1["id_maps_dir"] / "photo_ids.csv").open())}
    images = [load_image(p) for p in sorted((REPO_ROOT / cfg["catalog"]["crop"]["crops_dir"]).glob("*/*.*")) if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    rows = list(csv.DictReader((REPO_ROOT / cfg["crops"]["mapping_dir"] / cfg["crops"]["sample"]["name"] / "mapping.csv").open()))[:40]
    for r in rows:
        images.append(load_image(REPO_ROOT / s1["raw_dir"] / names[r["photo_id"]]).crop(tuple(int(r[k]) for k in ("crop_x1", "crop_y1", "crop_x2", "crop_y2"))))
    print(f"{len(images)} real crops | torch {torch.__version__} | device {device}")
    t0 = time.perf_counter()
    model = load_dinov2(ec, device, REPO_ROOT)
    sync(device)
    print(f"model load {time.perf_counter() - t0:.1f} s | parameters {sum(p.numel() for p in model.parameters()) / 1e6:.0f} M")
    t = cfg["embeddings"]["timing"]
    for bs in t["batch_sizes"]:
        times = []
        batch = (images * 4)[:bs]
        for k in range(t["warmup_batches"] + t["timed_batches"]):
            t1 = time.perf_counter()
            embed(model, batch, device, {**ec, "batch_size": bs})
            sync(device)
            if k >= t["warmup_batches"]:
                times.append(time.perf_counter() - t1)
        med = statistics.median(times)
        mem = f", {torch.mps.driver_allocated_memory() / 2**20:.0f} MB MPS driver memory" if device.type == "mps" else ""
        print(f"batch {bs:2d}: {med:.3f} s per batch = {1000 * med / bs:.1f} ms per crop{mem}")
    t1 = time.perf_counter()
    v = embed(model, images, device, ec)
    sync(device)
    print(f"all {len(images)} crops in {time.perf_counter() - t1:.1f} s | embedding shape {v.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
