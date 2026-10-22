"""Run a detector on the val photos and save the raw predictions (photo ids only) for evaluate_detections.py.

  --model frcnn : torchvision Faster R-CNN with COCO weights, class `bottle`, score threshold 0.05 (for mAP).
                  Two variants are saved: `raw` (the detector output) and `filtered` (the same Stage 2 filters that
                  produced the pseudo-labels: oversized, grey wedge, tiny, NMS, containment).
  --model dino  : frozen Grounding DINO + the same filters: a pipeline sanity check against the pseudo-labels
                  that came from the same model.
Only val photos are read (test is not touched before Stage 8). Outputs go to the gitignored data/predictions/stage4/.

Usage: python scripts/predict_val.py --model frcnn|dino
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.common.device import get_device  # noqa: E402
from src.data.images import load_image  # noqa: E402
from src.data.paths import check_output_dir  # noqa: E402
from src.detection.model import build_model  # noqa: E402
from src.prelabel import frcnn, grounding_dino  # noqa: E402
from src.prelabel.filters import filter_boxes, wedge_mask  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage4.yaml")
    parser.add_argument("--model", choices=["frcnn", "dino"], required=True)
    parser.add_argument("--checkpoint", help="frcnn only: a fine-tuned checkpoint of Stage 5 (best.pt or epoch_XX.pt) instead of the COCO weights")
    parser.add_argument("--name", help="output file name stem (default: <model>_val); needed with --checkpoint")
    args = parser.parse_args()
    cfg = yaml.safe_load((REPO_ROOT / args.config).read_text())
    pcfg = yaml.safe_load((REPO_ROOT / cfg["prelabel_config"]).read_text())
    s1 = yaml.safe_load((REPO_ROOT / "configs/stage1.yaml").read_text())
    torch.manual_seed(cfg["seed"])
    out_dir = REPO_ROOT / cfg["predictions_dir"]
    check_output_dir(out_dir, [REPO_ROOT / d for d in s1["protected_dirs"]])
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = [l.strip() for l in (REPO_ROOT / s1["splits"]["dir"] / f"{cfg['split']}.txt").read_text().split("\n") if l.strip()]
    names = {r["photo_id"]: r["file_name"] for r in csv.DictReader((REPO_ROOT / s1["id_maps_dir"] / "photo_ids.csv").open())}
    device = get_device() if pcfg["device"] == "auto" else torch.device(pcfg["device"])
    fcfg, wedge = pcfg["filters"], pcfg["filters"]["wedge"]
    wedge_args = {k: wedge[k] for k in ("value", "tolerance", "min_component_fraction")}

    if args.model == "frcnn" and args.checkpoint:
        scfg = yaml.safe_load((REPO_ROOT / "configs/stage5.yaml").read_text())
        model = build_model({**scfg["model"], "box_score_thresh": cfg["frcnn_score_threshold_for_map"]}, REPO_ROOT)
        state = torch.load(REPO_ROOT / args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        model.eval().to(device)
        target_id = 1  # the fine-tuned model has 2 classes: 0 = background, 1 = bottle
        print(f"checkpoint {args.checkpoint} (epoch {state.get('epoch', '?')})")
        detect = lambda image: frcnn.detect(model, target_id, image, device)  # noqa: E731
        variants = ["raw", "filtered"]
    elif args.model == "frcnn":
        model, target_id = frcnn.load_model({**pcfg["frcnn"], "box_score_thresh": cfg["frcnn_score_threshold_for_map"]}, device, REPO_ROOT)
        detect = lambda image: frcnn.detect(model, target_id, image, device)  # noqa: E731
        variants = ["raw", "filtered"]
    else:
        model, processor = grounding_dino.load_model(pcfg["dino"], device, REPO_ROOT)
        d = pcfg["dino"]
        detect = lambda image: grounding_dino.detect(model, processor, image, d["prompt"], device, d["box_threshold"], d["text_threshold"])  # noqa: E731
        variants = ["frozen"]

    result = {v: {} for v in variants}
    start = time.perf_counter()
    for k, pid in enumerate(ids, 1):
        image = load_image(REPO_ROOT / s1["raw_dir"] / names[pid])
        boxes, scores = detect(image)
        if "raw" in result:
            result["raw"][pid] = {"boxes": boxes.round(2).tolist(), "scores": scores.round(4).tolist()}
        filtered = filter_boxes(boxes, scores, image.size, wedge_mask(np.asarray(image), **wedge_args), fcfg)
        key = "filtered" if args.model == "frcnn" else "frozen"
        result[key][pid] = {"boxes": filtered[0].round(2).tolist(), "scores": filtered[1].round(4).tolist()}
    path = out_dir / f"{args.name or args.model + '_val'}.json"
    path.write_text(json.dumps({"model": args.model, "split": cfg["split"], "variants": result}))
    print(f"{args.model}: {len(ids)} {cfg['split']} photos on {device} in {time.perf_counter() - start:.0f} s -> {path.relative_to(REPO_ROOT)} (gitignored)")
    for v in variants:
        print(f"  {v}: {sum(len(p['boxes']) for p in result[v].values())} boxes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
