"""Compare the two pre-labelers on about 10 TRAIN photos, side by side, for the owner to choose.

For each selected photo one image with three panels is written (only photo ids as labels):
    COCO Faster R-CNN | Grounding DINO | merged (Faster R-CNN boxes win on overlaps)
plus summary.csv with box counts, how many boxes the two models share (IoU >= 0.5), drop counts per
filter reason, and Grounding DINO box counts at several thresholds and prompts.
Only train photos are used. No val or test photo is read, and no setting is tuned on them.

Usage: python scripts/prelabel_compare.py [--config configs/prelabel.yaml]
"""
import argparse
import csv
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.common.device import get_device  # noqa: E402
from src.data.images import load_image  # noqa: E402
from src.data.paths import check_output_dir  # noqa: E402
from src.prelabel import frcnn, grounding_dino  # noqa: E402
from src.prelabel.filters import box_iou, filter_boxes, wedge_mask  # noqa: E402
from src.prelabel.merge import merge_prefer_first  # noqa: E402

BLUE, ORANGE = (30, 100, 255), (255, 140, 0)
DROP_REASONS = ["oversized", "wedge_dropped", "wedge_clipped", "tiny", "duplicate_nms", "duplicate_contained"]


def read_ids(path: Path) -> list[str]:
    return [l.strip() for l in path.read_text().split("\n") if l.strip()]


def matched_count(a: np.ndarray, b: np.ndarray, iou_threshold: float) -> int:
    """One-to-one matches between two box sets (greedy, highest IoU first)."""
    iou = box_iou(a, b)
    used_a, used_b, n = set(), set(), 0
    for flat in np.argsort(-iou, axis=None):
        i, j = divmod(int(flat), iou.shape[1])
        if iou[i, j] < iou_threshold:
            break
        if i not in used_a and j not in used_b:
            used_a.add(i), used_b.add(j)
            n += 1
    return n


def panel(image: Image.Image, boxes, colors, title: str, height: int, font) -> Image.Image:
    scale = height / image.height
    img = image.resize((round(image.width * scale), height), Image.LANCZOS)
    draw = ImageDraw.Draw(img)
    for (x1, y1, x2, y2), color in zip(boxes, colors):
        draw.rectangle([x1 * scale, y1 * scale, x2 * scale, y2 * scale], outline=color, width=3)
    out = Image.new("RGB", (img.width, height + 40), "white")
    ImageDraw.Draw(out).text((8, 8), title, fill="black", font=font)
    out.paste(img, (0, 40))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/prelabel.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load((REPO_ROOT / args.config).read_text())
    s1 = yaml.safe_load((REPO_ROOT / "configs/stage1.yaml").read_text())
    cmp_cfg, fcfg = cfg["compare"], cfg["filters"]

    out_dir = REPO_ROOT / cmp_cfg["output_dir"]
    check_output_dir(out_dir, [REPO_ROOT / d for d in s1["protected_dirs"]])  # never write in the photo folders
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()

    train = read_ids(REPO_ROOT / s1["splits"]["dir"] / "train.txt")
    names = {r["photo_id"]: r["file_name"] for r in csv.DictReader((REPO_ROOT / s1["id_maps_dir"] / "photo_ids.csv").open())}
    meta = {r["photo_id"]: r for r in csv.DictReader((REPO_ROOT / s1["metadata_csv"]).open())}
    assert set(cmp_cfg["must_include"]) <= set(train), "the hard cases must be train photos"
    device = get_device() if cfg["device"] == "auto" else torch.device(cfg["device"])
    rng = random.Random(cfg["seed"])
    print(f"device: {device} | train photos available: {len(train)}")

    fr_model, target_id = frcnn.load_model(cfg["frcnn"], device, REPO_ROOT)
    dino_model, dino_proc = grounding_dino.load_model(cfg["dino"], device, REPO_ROOT)

    def load(photo_id):
        image = load_image(REPO_ROOT / s1["raw_dir"] / names[photo_id])
        return image, wedge_mask(np.asarray(image), **{k: fcfg["wedge"][k] for k in ("value", "tolerance", "min_component_fraction")})

    def run_frcnn(image, mask):
        boxes, scores = frcnn.detect(fr_model, target_id, image, device)
        return filter_boxes(boxes, scores, image.size, mask, fcfg)

    # ---- choose the photos -------------------------------------------------------------------------------
    density = {}
    for pid in train:  # COCO Faster R-CNN box count per train photo, used for the density strata
        image, mask = load(pid)
        density[pid] = len(run_frcnn(image, mask)[0])
    selected = {pid: "hard case from the labeling guide" for pid in cmp_cfg["must_include"]}
    by_pixels = sorted(train, key=lambda p: -int(meta[p]["width"]) * int(meta[p]["height"]))
    for pid in [p for p in by_pixels if p not in selected][:cmp_cfg["n_highest_resolution"]]:
        selected[pid] = "highest resolution"
    store_of = lambda p: p.split("_")[0]  # noqa: E731
    for store in sorted({store_of(p) for p in train}):
        if not any(store_of(p) == store for p in selected):
            selected[rng.choice([p for p in train if store_of(p) == store and p not in selected])] = f"store {store}"
    counts = sorted(density.values())
    cuts = (counts[len(counts) // 3], counts[2 * len(counts) // 3])
    tercile = lambda n: 0 if n < cuts[0] else 1 if n < cuts[1] else 2  # noqa: E731
    for t, label in enumerate(("low", "medium", "high")):
        if not any(tercile(density[p]) == t for p in selected):
            selected[rng.choice([p for p in train if tercile(density[p]) == t and p not in selected])] = f"{label} density"
    print(f"selected {len(selected)} train photos (density terciles cut at {cuts[0]} and {cuts[1]} boxes):")
    for pid, why in selected.items():
        print(f"  {pid}: {why}, {density[pid]} COCO boxes")

    # ---- run both models ---------------------------------------------------------------------------------
    font = ImageFont.load_default(size=20)
    rows = []
    dcfg, min_t = cfg["dino"], min(cfg["dino"]["box_thresholds_to_report"] + [cfg["dino"]["box_threshold"]])
    for k, pid in enumerate(sorted(selected, key=lambda p: -density[p]), 1):
        image, mask = load(pid)
        fb, fs, fdrops = run_frcnn(image, mask)
        raw = {}
        for prompt in [dcfg["prompt"]] + dcfg["extra_prompts"]:   # one low-threshold run per prompt, counted at each threshold
            raw[prompt] = grounding_dino.detect(dino_model, dino_proc, image, prompt, device, min_t, dcfg["text_threshold"])
        rb, rs = raw[dcfg["prompt"]]
        keep = rs >= dcfg["box_threshold"]
        db, ds, ddrops = filter_boxes(rb[keep], rs[keep], image.size, mask, fcfg)
        mb, ms, src = merge_prefer_first(fb, fs, db, ds, cfg["merge"]["iou"])

        heights = fb[:, 3] - fb[:, 1]
        row = {"photo_id": pid, "why": selected[pid], "width": image.width, "height": image.height,
               "frcnn_boxes": len(fb), "dino_boxes": len(db), "shared_iou_0.5": matched_count(fb, db, cmp_cfg["n_matching_iou"]),
               "merged_boxes": len(mb), "median_box_height_px": round(statistics.median(heights)) if len(fb) else 0}
        row["only_frcnn"], row["only_dino"] = len(fb) - row["shared_iou_0.5"], len(db) - row["shared_iou_0.5"]
        row.update({f"frcnn_drop_{r}": fdrops[r] for r in DROP_REASONS})
        row.update({f"dino_drop_{r}": ddrops[r] for r in DROP_REASONS})
        for prompt, (_, scores) in raw.items():
            for t in dcfg["box_thresholds_to_report"]:
                row[f"dino[{prompt}]>={t}"] = int((scores >= t).sum())
        rows.append(row)

        height_px = cmp_cfg["panel_height_px"]
        panels = [
            panel(image, fb, [BLUE] * len(fb), f"{pid}  |  COCO Faster R-CNN: {len(fb)} boxes", height_px, font),
            panel(image, db, [ORANGE] * len(db), f"Grounding DINO '{dcfg['prompt']}' >= {dcfg['box_threshold']}: {len(db)} boxes", height_px, font),
            panel(image, mb, [BLUE if s == 0 else ORANGE for s in src],
                  f"merged: {len(mb)} boxes (blue = Faster R-CNN, orange = only Grounding DINO)", height_px, font),
        ]
        sheet = Image.new("RGB", (sum(p.width for p in panels) + 16, panels[0].height), "white")
        x = 0
        for p in panels:
            sheet.paste(p, (x, 0))
            x += p.width + 8
        sheet.save(out_dir / f"{k:02d}_{pid}.jpg", quality=85)
        print(f"{k:02d} {pid}: frcnn {len(fb)} | dino {len(db)} | shared {row['shared_iou_0.5']} | merged {len(mb)}")

    with (out_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    total = Counter()
    for r in rows:
        for key in ("frcnn_boxes", "dino_boxes", "shared_iou_0.5", "merged_boxes", "only_frcnn", "only_dino"):
            total[key] += r[key]
    print("totals:", dict(total))
    print(f"images and summary.csv in {out_dir.relative_to(REPO_ROOT)}/ (gitignored)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
