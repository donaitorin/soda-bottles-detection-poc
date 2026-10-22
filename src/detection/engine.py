"""Training and inference helpers: learning-rate schedule, one epoch, prediction, checkpoints, val agreement."""
import math
import time
from pathlib import Path

import numpy as np
import torch

from src.evaluation.detection import ignore_below_floor, operating_point, per_photo_results, summarize


def lr_at(step: int, total_steps: int, base_lr: float, warmup_steps: int, warmup_factor: float, final_factor: float) -> float:
    """Linear warmup from base_lr * warmup_factor to base_lr, then cosine decay down to base_lr * final_factor.
    Warmup protects the freshly initialised box predictor from large early gradients."""
    if step < warmup_steps:
        return base_lr * (warmup_factor + (1.0 - warmup_factor) * step / max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * (final_factor + (1.0 - final_factor) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def train_one_epoch(model, loader, optimizer, device, schedule, step0: int, clip_norm: float | None = None,
                    amp: bool = False) -> tuple[float, int, float]:
    """One pass over the loader. `schedule(step)` returns the lr for a global step. Returns (mean loss, next step, seconds)."""
    model.train()
    losses, step, start = [], step0, time.perf_counter()
    for images, targets in loader:
        images = [i.to(device) for i in images]
        targets = [{"boxes": t["boxes"].to(device), "labels": t["labels"].to(device)} for t in targets]
        for group in optimizer.param_groups:
            group["lr"] = schedule(step)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            loss = sum(model(images, targets).values())
        if not torch.isfinite(loss):
            raise RuntimeError(f"loss is {loss.item()} at step {step}: stop and look at the data and the learning rate")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if clip_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        losses.append(loss.item())
        step += 1
    sync(device)
    return float(np.mean(losses)), step, time.perf_counter() - start


@torch.inference_mode()
def predict(model, dataset, device) -> dict:
    """{photo id: (boxes xyxy original pixels, scores)} for every item of a dataset built WITHOUT augmentation."""
    model.eval()
    out = {}
    for i in range(len(dataset)):
        image, target = dataset[i]
        result = model([image.to(device)])[0]
        out[target["photo_id"]] = (result["boxes"].cpu().numpy().astype(float), result["scores"].cpu().numpy().astype(float))
    return out


def agreement(gt: dict, preds: dict, floor: float, operating_threshold: float, max_dets: int) -> dict:
    """AP, AP50, recall (all scores), precision and recall at the operating score, against the COCO dict `gt`.
    AGREEMENT with the labels in `gt` (Grounding DINO pseudo-labels for the project's train/val/test)."""
    sizes = {im["photo_id"]: (im["width"], im["height"]) for im in gt["images"]}
    preds, ignored = ignore_below_floor({p: preds[p] for p in sizes}, sizes, floor)
    res_all, _ = per_photo_results(gt, preds, max_dets)
    at_op = {p: (b[s >= operating_threshold], s[s >= operating_threshold]) for p, (b, s) in preds.items()}
    res_op, _ = per_photo_results(gt, at_op, max_dets)
    m = summarize(list(res_all.values()), max_dets)
    op = operating_point(list(res_op.values()))
    return {**m, "precision": op["precision"], "recall": op["recall"], "f1": op["f1"], "ignored_below_floor": ignored}


def save_checkpoint(path: Path, model, optimizer, epoch: int, step: int, extra: dict | None = None) -> None:
    """Write atomically (temp file, then rename) so an interrupted run never leaves a half-written checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "step": step,
                **(extra or {})}, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path, model, optimizer=None, device="cpu") -> dict:
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    return state
