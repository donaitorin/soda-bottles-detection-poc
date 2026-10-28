"""Stage 7 task 4: seconds per photo of variant A (detector + embeddings + classifier) and variant B (A + OCR on the crops tall enough).

Timing only: n photos of the val split are processed, no label or answer is looked at or reported. The first calls on MPS include
one-time setup, so everything is warmed up before timing (as in scripts/demo_identify.py).

Usage: python scripts/ocr_timing.py [--min-height 150]
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
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.common.device import get_device  # noqa: E402
from src.data.images import load_image  # noqa: E402
from src.detection.model import build_model  # noqa: E402
from src.embeddings.classifier import LinearClassifier  # noqa: E402
from src.embeddings.crops import crop_box_with_margin  # noqa: E402
from src.embeddings.embed import embed, load_dinov2  # noqa: E402
from src.ocr.evidence import keywords_read  # noqa: E402
from src.ocr.reader import OcrReader, OcrUnavailable  # noqa: E402
from src.prelabel import frcnn  # noqa: E402
from src.prelabel.filters import below_height_floor  # noqa: E402


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage7.yaml")
    parser.add_argument("--min-height", type=int, default=150, help="variant B reads only crops at least this tall (px)")
    args = parser.parse_args()
    c7 = yaml.safe_load((REPO_ROOT / args.config).read_text())
    cfg = yaml.safe_load((REPO_ROOT / c7["config_stage6"]).read_text())
    pcfg = yaml.safe_load((REPO_ROOT / "configs/prelabel.yaml").read_text())
    scfg = yaml.safe_load((REPO_ROOT / "configs/stage5.yaml").read_text())
    s1 = yaml.safe_load((REPO_ROOT / "configs/stage1.yaml").read_text())
    dc, ec = cfg["demo"], cfg["embeddings"]
    names = {r["photo_id"]: r["file_name"] for r in csv.DictReader((REPO_ROOT / s1["id_maps_dir"] / "photo_ids.csv").open())}
    ids = sorted(l.strip() for l in (REPO_ROOT / s1["splits"]["dir"] / c7["timing"]["split"]).with_suffix(".txt").read_text().split("\n") if l.strip())
    n = c7["timing"]["n_photos"]
    ids = [ids[int(i * len(ids) / n)] for i in range(n)]                         # evenly spread over the sorted list
    device = get_device()
    det = build_model({**scfg["model"], "box_score_thresh": dc["detector"]["operating_score"]}, REPO_ROOT)
    det.load_state_dict(torch.load(REPO_ROOT / dc["detector"]["checkpoint"], map_location="cpu", weights_only=True)["model"])
    det.eval().to(device)
    dino = load_dinov2(ec, device, REPO_ROOT)
    clf = LinearClassifier.load(REPO_ROOT / cfg["classifier"]["model_dir"] / f"{cfg['classifier']['name']}.npz")
    try:
        ocr = OcrReader(c7["ocr"], REPO_ROOT, device)
    except OcrUnavailable as error:
        raise SystemExit(str(error))
    warm = Image.new("RGB", (320, 320), (128, 128, 128))
    frcnn.detect(det, 1, warm, device)
    embed(dino, [warm, warm], device, {**ec, "batch_size": 2})
    ocr.read(warm)
    sync(device)
    rows = []
    for pid in ids:
        image = load_image(REPO_ROOT / s1["raw_dir"] / names[pid])
        t0 = time.perf_counter()
        boxes, _ = frcnn.detect(det, 1, image, device)
        sync(device)
        boxes = boxes[~below_height_floor(boxes, image.height, pcfg["filters"]["min_height_fraction"])] if len(boxes) else boxes
        t_det = time.perf_counter() - t0
        t1 = time.perf_counter()
        crops = [image.crop(crop_box_with_margin(b, image.width, image.height, 0.10)) for b in boxes]
        vecs = embed(dino, crops, device, ec) if crops else np.zeros((0, 768), np.float32)
        sync(device)
        if len(vecs):
            clf.predict_proba(vecs)
        t_emb = time.perf_counter() - t1
        times = {}
        for h in (args.min_height, 300, 400):
            sel = [c for c in crops if c.height >= h]
            t2 = time.perf_counter()
            for c in sel:
                keywords_read(" ".join(t for t, _ in ocr.read(c)), c7["ocr"]["fuzzy_min_ratio"])
            sync(device)
            times[h] = (len(sel), time.perf_counter() - t2)
        rows.append({"boxes": len(crops), "A": t_det + t_emb, "det": t_det, "emb": t_emb, "ocr": times})
    med = statistics.median
    print(f"{n} val photos on {device} (timing only). Seconds per photo, median [min, max]:")
    print(f"  boxes per photo: median {med(r['boxes'] for r in rows)} [{min(r['boxes'] for r in rows)}, {max(r['boxes'] for r in rows)}]")
    a = [r["A"] for r in rows]
    print(f"  variant A (detector + embeddings + classifier): {med(a):.2f} [{min(a):.2f}, {max(a):.2f}]  (detector {med(r['det'] for r in rows):.2f}, embeddings+classifier {med(r['emb'] for r in rows):.2f})")
    for h in (args.min_height, 300, 400):
        o = [r["ocr"][h][1] for r in rows]
        k = [r["ocr"][h][0] for r in rows]
        b = [r["A"] + r["ocr"][h][1] for r in rows]
        print(f"  variant B, reading crops >= {h} px: OCR adds {med(o):.2f} [{min(o):.2f}, {max(o):.2f}] for {med(k)} crops per photo -> total {med(b):.2f} [{min(b):.2f}, {max(b):.2f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
