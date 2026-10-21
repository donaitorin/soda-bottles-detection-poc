"""Check the converted Label Studio boxes: numerically for every task, visually on 10 photos.

1. Converts every prediction in tasks.json back to pixels and compares with predictions.json (max difference).
2. Draws the converted boxes on 10 photos (the largest, a portrait, a landscape, and seeded random ones) into
   the gitignored data/review/ls_check/, and writes the 3 photos for the owner's alignment check in Label Studio.
Usage: python scripts/check_ls_conversion.py
"""
import argparse
import json
import random
import sys
from pathlib import Path

import yaml
from PIL import ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import csv  # noqa: E402

from src.data.images import load_image  # noqa: E402
from src.data.paths import check_output_dir  # noqa: E402
from src.prelabel.labelstudio import ls_result_to_xyxy  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/prelabel.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load((REPO_ROOT / args.config).read_text())
    s1 = yaml.safe_load((REPO_ROOT / "configs/stage1.yaml").read_text())
    ls = cfg["label_studio"]
    tasks = json.loads((REPO_ROOT / ls["tasks_json"]).read_text())
    preds = json.loads((REPO_ROOT / cfg["run"]["output_dir"] / "predictions.json").read_text())["photos"]
    names = {r["photo_id"]: r["file_name"] for r in csv.DictReader((REPO_ROOT / s1["id_maps_dir"] / "photo_ids.csv").open())}

    worst, n, n_clamped, max_clamp = 0.0, 0, 0, 0.0
    for task in tasks:
        pid = task["data"]["photo_id"]
        results = task["predictions"][0]["result"]
        assert len(results) == len(preds[pid]["boxes"]), pid
        w, h = preds[pid]["width"], preds[pid]["height"]
        for result, box in zip(results, preds[pid]["boxes"]):
            # The conversion clamps boxes to the image, so compare against the clamped box. The model can
            # return boxes poking a few pixels outside the photo; those are counted separately.
            clamped = [min(max(box[0], 0), w), min(max(box[1], 0), h), min(max(box[2], 0), w), min(max(box[3], 0), h)]
            shift = max(abs(a - b) for a, b in zip(clamped, box))
            n_clamped += shift > 0
            max_clamp = max(max_clamp, shift)
            worst = max(worst, max(abs(a - b) for a, b in zip(ls_result_to_xyxy(result), clamped)))
            v = result["value"]
            assert 0 <= v["x"] <= 100 and 0 <= v["y"] <= 100 and v["width"] >= 0 and v["height"] >= 0
            assert v["x"] + v["width"] <= 100 + 1e-6 and v["y"] + v["height"] <= 100 + 1e-6
            n += 1
    print(f"{len(tasks)} tasks, {n} boxes: largest difference after converting back to pixels = {worst:.4f} px "
          f"(rounding of the stored pixel boxes); {n_clamped} boxes were clamped to the image edge, by at most {max_clamp:.2f} px")
    assert worst < 0.01, "round trip differs by more than rounding"

    sizes = {t["data"]["photo_id"]: (preds[t["data"]["photo_id"]]["width"], preds[t["data"]["photo_id"]]["height"]) for t in tasks}
    largest = max(sizes, key=lambda p: sizes[p][0] * sizes[p][1])
    portrait = [p for p in sorted(sizes) if sizes[p] == (1184, 1440)]
    landscape = [p for p in sorted(sizes) if sizes[p][0] > sizes[p][1]]
    chosen = {"largest": largest, "portrait": portrait[len(portrait) // 2], "landscape": landscape[0]}
    rng = random.Random(cfg["seed"])
    extra = rng.sample([p for p in sorted(sizes) if p not in chosen.values()], cfg["label_studio"]["n_check_photos"] - 3)
    out_dir = REPO_ROOT / ls["check_dir"]
    check_output_dir(out_dir, [REPO_ROOT / d for d in s1["protected_dirs"]])
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()
    by_id = {t["data"]["photo_id"]: t for t in tasks}
    for pid in list(chosen.values()) + extra:
        image = load_image(REPO_ROOT / s1["raw_dir"] / names[pid])
        draw = ImageDraw.Draw(image)
        for result in by_id[pid]["predictions"][0]["result"]:
            draw.rectangle(ls_result_to_xyxy(result), outline=(255, 0, 0), width=max(2, image.height // 300))
        image.thumbnail((1100, 1100))
        image.save(out_dir / f"{pid}.jpg", quality=85)
    (REPO_ROOT / ls["tasks_json"]).parent.joinpath("alignment_check_ids.txt").write_text(
        "".join(f"{kind}: {pid} ({sizes[pid][0]}x{sizes[pid][1]}, {len(by_id[pid]['predictions'][0]['result'])} boxes)\n"
                for kind, pid in chosen.items()))
    for kind, pid in chosen.items():
        print(f"alignment check in Label Studio - {kind}: {pid} ({sizes[pid][0]}x{sizes[pid][1]})")
    print(f"drawn photos in {out_dir.relative_to(REPO_ROOT)}/ (gitignored)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
