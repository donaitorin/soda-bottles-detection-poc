"""Stage 5: fine-tune Faster R-CNN (COCO weights, 2 classes: background + bottle) on the train pseudo-labels.

Modes
  --overfit   sanity check: train on N seeded random train photos without augmentation and check that the model
              reproduces their labels (if it cannot, the dataset or the loss has a bug).
  (default)   full run: SGD with linear warmup and cosine decay, a checkpoint EVERY epoch, the loss per epoch and
              the val agreement of each checkpoint in a CSV. `--resume` continues from the last checkpoint.

THE LABELS ARE GROUNDING DINO PSEUDO-LABELS: the model learns to imitate Grounding DINO, errors included, and the
val numbers are AGREEMENT with Grounding DINO, not accuracy. Test is never read here.

Long runs on the Mac (keeps the Mac awake; resume with the same command plus --resume):
  caffeinate -i conda run --no-capture-output -n soda-poc python scripts/train_detector.py [--resume]

Usage: python scripts/train_detector.py [--overfit] [--resume] [--epochs N] [--limit-photos N] [--run-name NAME]
"""
import argparse
import csv
import json
import random
import re
import sys
import time
from pathlib import Path

import torch
import yaml
from PIL import ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.common.device import get_device  # noqa: E402  (sets the MPS fallback before torch loads)
from src.data.paths import check_output_dir  # noqa: E402
from src.detection.dataset import BottleDataset, build_transforms, collate  # noqa: E402
from src.detection.engine import (agreement, load_checkpoint, lr_at, predict, save_checkpoint,  # noqa: E402
                                  train_one_epoch)
from src.detection.model import build_model  # noqa: E402

LOG_FIELDS = ["epoch", "lr_end", "train_loss", "epoch_seconds", "val_AP", "val_AP50", "val_recall50", "val_precision",
              "val_recall", "val_f1", "val_ignored_below_floor", "is_best"]
NOTE = "agreement with Grounding DINO pseudo-labels, NOT accuracy"


def subset_coco(coco: dict, photo_ids: list[str]) -> dict:
    ids = {im["id"] for im in coco["images"] if im["photo_id"] in set(photo_ids)}
    return {**coco, "images": [im for im in coco["images"] if im["id"] in ids],
            "annotations": [a for a in coco["annotations"] if a["image_id"] in ids]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage5.yaml")
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs", type=int, help="override train.epochs (smoke tests)")
    parser.add_argument("--limit-photos", type=int, help="use only the first N train photos (smoke tests)")
    parser.add_argument("--run-name", help="override run.name")
    args = parser.parse_args()
    cfg = yaml.safe_load((REPO_ROOT / args.config).read_text())
    pcfg = yaml.safe_load((REPO_ROOT / cfg["prelabel_config"]).read_text())
    s1 = yaml.safe_load((REPO_ROOT / "configs/stage1.yaml").read_text())
    tr = cfg["train"]
    floor = pcfg["filters"]["min_height_fraction"]
    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    device = get_device() if cfg["device"] == "auto" else torch.device(cfg["device"])
    protected = [REPO_ROOT / d for d in s1["protected_dirs"]]

    names = {r["photo_id"]: r["file_name"] for r in csv.DictReader((REPO_ROOT / s1["id_maps_dir"] / "photo_ids.csv").open())}
    paths = {pid: REPO_ROOT / s1["raw_dir"] / n for pid, n in names.items()}
    train_coco = json.loads((REPO_ROOT / cfg["data"]["train_coco"]).read_text())
    val_coco = json.loads((REPO_ROOT / cfg["data"]["val_coco"]).read_text())
    train_ids = sorted(im["photo_id"] for im in train_coco["images"])
    model = build_model(cfg["model"], REPO_ROOT).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"device {device} | torch {torch.__version__} | labels: {NOTE}")

    if args.overfit:
        return overfit(cfg, args, model, params, device, train_ids, train_coco, paths, floor, protected)

    run = args.run_name or cfg["run"]["name"]
    ckpt_dir, log_dir = REPO_ROOT / cfg["run"]["checkpoint_dir"] / run, REPO_ROOT / cfg["run"]["log_dir"] / run
    for d in (ckpt_dir, log_dir):
        check_output_dir(d, protected)
        d.mkdir(parents=True, exist_ok=True)
    if args.limit_photos:
        train_ids = train_ids[:args.limit_photos]
    epochs = args.epochs or tr["epochs"]
    train_ds = BottleDataset(REPO_ROOT / cfg["data"]["train_coco"], paths, build_transforms(True, cfg["augment"]), train_ids)
    val_ds = BottleDataset(REPO_ROOT / cfg["data"]["val_coco"], paths, build_transforms(False))
    gen = torch.Generator()
    loader = torch.utils.data.DataLoader(train_ds, batch_size=tr["batch_size"], shuffle=True, generator=gen, collate_fn=collate,
                                         num_workers=cfg["data"]["num_workers"], persistent_workers=cfg["data"]["num_workers"] > 0)
    optimizer = torch.optim.SGD(params, lr=tr["optimizer"]["lr"], momentum=tr["optimizer"]["momentum"],
                                weight_decay=tr["optimizer"]["weight_decay"])
    steps_per_epoch = len(loader)
    total_steps, warmup_steps = epochs * steps_per_epoch, tr["warmup"]["epochs"] * steps_per_epoch
    schedule = lambda s: lr_at(s, total_steps, tr["optimizer"]["lr"], warmup_steps, tr["warmup"]["factor"], tr["final_lr_factor"])  # noqa: E731

    start_epoch, step, rows, best = 0, 0, [], -1.0
    log_path = log_dir / "train_log.csv"
    if args.resume:
        done = sorted(int(m.group(1)) for p in ckpt_dir.glob("epoch_*.pt") if (m := re.match(r"epoch_(\d+)\.pt", p.name)))
        if not done:
            raise SystemExit(f"--resume: no checkpoint in {ckpt_dir.relative_to(REPO_ROOT)}")
        state = load_checkpoint(ckpt_dir / f"epoch_{done[-1]:02d}.pt", model, optimizer, device)
        start_epoch, step = state["epoch"], state["step"]
        rows = [r for r in csv.DictReader(log_path.open())] if log_path.exists() else []
        rows = [r for r in rows if int(r["epoch"]) <= start_epoch]
        best = max([float(r["val_AP50"]) for r in rows] or [-1.0])
        print(f"resumed from epoch {start_epoch} (step {step}); best val AP50 so far {best:.4f}")
    elif list(ckpt_dir.glob("epoch_*.pt")):
        raise SystemExit(f"{ckpt_dir.relative_to(REPO_ROOT)} already has checkpoints: use --resume or another --run-name")

    print(f"run {run}: {len(train_ds)} train photos, {steps_per_epoch} steps/epoch, {epochs} epochs, batch {tr['batch_size']}, "
          f"lr {tr['optimizer']['lr']} (warmup {tr['warmup']['epochs']} epoch(s), cosine to x{tr['final_lr_factor']})")
    for epoch in range(start_epoch + 1, epochs + 1):
        gen.manual_seed(cfg["seed"] + epoch)  # same shuffle and augmentation after a resume
        loss, step, seconds = train_one_epoch(model, loader, optimizer, device, schedule, step, tr["grad_clip_norm"], tr["amp"])
        preds = predict(model, val_ds, device)
        m = agreement(val_coco, preds, floor, tr["eval_operating_threshold"], tr["max_dets"])
        is_best = m["AP50"] > best
        if is_best:
            best = m["AP50"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_AP50": best}, ckpt_dir / "best.pt")
        save_checkpoint(ckpt_dir / f"epoch_{epoch:02d}.pt", model, optimizer, epoch, step)
        rows.append({"epoch": epoch, "lr_end": f"{schedule(step - 1):.6f}", "train_loss": f"{loss:.4f}", "epoch_seconds": f"{seconds:.0f}",
                     "val_AP": f"{m['AP']:.4f}", "val_AP50": f"{m['AP50']:.4f}", "val_recall50": f"{m['recall50']:.4f}",
                     "val_precision": f"{m['precision']:.4f}", "val_recall": f"{m['recall']:.4f}", "val_f1": f"{m['f1']:.4f}",
                     "val_ignored_below_floor": m["ignored_below_floor"], "is_best": int(is_best)})
        with log_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"epoch {epoch:02d}/{epochs} loss {loss:.3f} | val ({NOTE}): AP {m['AP']:.3f} AP50 {m['AP50']:.3f} "
              f"recall {m['recall50']:.3f} | {seconds:.0f} s{' | best' if is_best else ''}", flush=True)
    print(f"done. log: {log_path.relative_to(REPO_ROOT)}; checkpoints: {ckpt_dir.relative_to(REPO_ROOT)}/ (gitignored)")
    return 0


def overfit(cfg, args, model, params, device, train_ids, train_coco, paths, floor, protected) -> int:
    o, tr = cfg["overfit"], cfg["train"]
    rng = random.Random(cfg["seed"])
    ids = sorted(rng.sample(train_ids, o["n_photos"]))
    ds = BottleDataset(REPO_ROOT / cfg["data"]["train_coco"], paths, build_transforms(False), ids)  # NO augmentation
    gen = torch.Generator().manual_seed(cfg["seed"])
    loader = torch.utils.data.DataLoader(ds, batch_size=tr["batch_size"], shuffle=True, generator=gen, collate_fn=collate, num_workers=0)
    optimizer = torch.optim.SGD(params, lr=o["lr"], momentum=tr["optimizer"]["momentum"], weight_decay=tr["optimizer"]["weight_decay"])
    sub = subset_coco(train_coco, ids)
    n_boxes = len(sub["annotations"])
    n_epochs = -(-o["iterations"] // len(loader))
    total = n_epochs * len(loader)
    schedule = lambda s: lr_at(s, total, o["lr"], 10, 0.1, 1.0)  # short warmup, then constant  # noqa: E731
    print(f"overfit: {len(ids)} seeded random train photos, {n_boxes} pseudo-label boxes, {n_epochs} epochs of {len(loader)} steps, no augmentation")

    def check(tag):
        m = agreement(sub, predict(model, ds, device), floor, tr["eval_operating_threshold"], tr["max_dets"])
        print(f"  {tag}: agreement with these photos' pseudo-labels: AP {m['AP']:.3f} AP50 {m['AP50']:.3f} recall {m['recall50']:.3f} "
              f"precision@0.3 {m['precision']:.3f} recall@0.3 {m['recall']:.3f}")
        return m

    check("before training (random 2-class head)")
    step, first_loss, loss = 0, None, 0.0
    for epoch in range(1, n_epochs + 1):
        loss, step, seconds = train_one_epoch(model, loader, optimizer, device, schedule, step, tr["grad_clip_norm"], False)
        first_loss = first_loss if first_loss is not None else loss
        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}: mean loss {loss:.4f} ({seconds:.0f} s)", flush=True)
    final = check("after training")
    print(f"loss {first_loss:.3f} (epoch 1) -> {loss:.3f} (epoch {n_epochs})")
    draw_dir = REPO_ROOT / o["draw_dir"]
    check_output_dir(draw_dir, protected)
    draw_dir.mkdir(parents=True, exist_ok=True)
    for old in draw_dir.glob("*.jpg"):
        old.unlink()
    preds = predict(model, ds, device)
    for i in range(min(3, len(ds))):
        image, target = BottleDataset(REPO_ROOT / cfg["data"]["train_coco"], paths, build_transforms(False), ids)[i]
        pil = torch.clamp(image * 255, 0, 255).byte().permute(1, 2, 0).numpy()
        from PIL import Image
        pil = Image.fromarray(pil)
        d = ImageDraw.Draw(pil)
        w = max(2, pil.height // 300)
        for b in target["boxes"].tolist():
            d.rectangle(b, outline=(0, 200, 0), width=w + 2)   # pseudo-label: green, thick
        boxes, scores = preds[target["photo_id"]]
        for b, s in zip(boxes, scores):
            if s >= tr["eval_operating_threshold"]:
                d.rectangle(list(b), outline=(255, 0, 0), width=w)   # model prediction: red, thin
        pil.thumbnail((1100, 1100))
        pil.save(draw_dir / f"{target['photo_id']}.jpg", quality=85)
    print(f"drawn 3 of the {len(ids)} photos in {draw_dir.relative_to(REPO_ROOT)}/ (gitignored): green = pseudo-label, red = prediction")
    return 0


if __name__ == "__main__":
    sys.exit(main())
