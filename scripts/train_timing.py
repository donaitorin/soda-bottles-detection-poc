"""Stage 5 timing and memory check on the real device, before any long training run.

For each variant in configs/stage5.yaml (`timing.variants`): seconds per training iteration (median of the timed
iterations after warm-ups), images per second, memory in use after the iterations, and the data loading speed
(decoding the big photos and augmenting them on the CPU). Then an estimate of the time of one epoch (174 train
photos) and of the full run. An out-of-memory error or a failing operation is reported, not hidden.

Usage: python scripts/train_timing.py
"""
import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.common.device import get_device  # noqa: E402  (sets the MPS fallback before torch loads)
from src.detection.dataset import BottleDataset, build_transforms, collate  # noqa: E402
from src.detection.engine import sync  # noqa: E402
from src.detection.model import build_model  # noqa: E402


def memory_mb(device) -> str:
    if device.type == "mps":
        return f"{torch.mps.current_allocated_memory() / 2**20:.0f} MB allocated, {torch.mps.driver_allocated_memory() / 2**20:.0f} MB driver"
    return "n/a"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage5.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load((REPO_ROOT / args.config).read_text())
    s1 = yaml.safe_load((REPO_ROOT / "configs/stage1.yaml").read_text())
    torch.manual_seed(cfg["seed"])
    device = get_device() if cfg["device"] == "auto" else torch.device(cfg["device"])
    names = {r["photo_id"]: r["file_name"] for r in csv.DictReader((REPO_ROOT / s1["id_maps_dir"] / "photo_ids.csv").open())}
    paths = {pid: REPO_ROOT / s1["raw_dir"] / n for pid, n in names.items()}
    tcfg = cfg["timing"]
    n_iter = tcfg["warmup_iterations"] + tcfg["timed_iterations"]
    print(f"torch {torch.__version__} | device {device}")
    n_train = None
    for v in tcfg["variants"]:
        ds = BottleDataset(REPO_ROOT / cfg["data"]["train_coco"], paths, build_transforms(True, cfg["augment"]))
        n_train = len(ds)
        g = torch.Generator().manual_seed(cfg["seed"])
        loader = torch.utils.data.DataLoader(ds, batch_size=v["batch_size"], shuffle=True, generator=g, collate_fn=collate,
                                             num_workers=cfg["data"]["num_workers"], persistent_workers=cfg["data"]["num_workers"] > 0)
        model = build_model({**cfg["model"], "min_size": v["min_size"], "max_size": int(v["min_size"] * 1333 / 800)}, REPO_ROOT).to(device)
        model.train()
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.001, momentum=0.9, weight_decay=1e-4)
        times, load_times, label = [], [], f"min_size {v['min_size']}, batch {v['batch_size']}, AMP {v['amp']}"
        try:
            it = iter(loader)
            for k in range(n_iter):
                t0 = time.perf_counter()
                images, targets = next(it)
                load_times.append(time.perf_counter() - t0)
                images = [i.to(device) for i in images]
                targets = [{"boxes": t["boxes"].to(device), "labels": t["labels"].to(device)} for t in targets]
                sync(device)
                t1 = time.perf_counter()
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=v["amp"]):
                    loss = sum(model(images, targets).values())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                sync(device)
                if k >= tcfg["warmup_iterations"]:
                    times.append(time.perf_counter() - t1)
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite loss")
            med = statistics.median(times)
            per_epoch = n_train / v["batch_size"] * med
            print(f"{label}: {med:.2f} s/iteration ({v['batch_size'] / med:.2f} img/s) | epoch of {n_train} photos about {per_epoch / 60:.1f} min "
                  f"| 30 epochs about {30 * per_epoch / 3600:.1f} h | data wait per batch {statistics.median(load_times[tcfg['warmup_iterations']:]):.2f} s | {memory_mb(device)} | loss {loss.item():.2f}")
        except RuntimeError as error:
            print(f"{label}: FAILED ({str(error)[:200]})")
        del model, optimizer, loader
        if device.type == "mps":
            torch.mps.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
