"""Try the CURRENT system on one photo (a small developer tool, not an evaluation).

Pipeline: frcnn_v2 last epoch at the operating score 0.3 (raw output plus the 1.5% height floor), the same 10% margin crops as the
shelf crops of Stage 6b, DINOv2 CLS embeddings, search in the pgvector database `soda_catalog` (catalog (b): the 30 originals + owner-labeled train shelf entries + unknown negatives), then the
Stage 6b unknown rules with the DEMO thresholds of configs/stage6.yaml (tuned once on all 180 labeled val crops: NOT an evaluation
result; with them the system answers nothing automatically, so expect yellow and grey boxes; --t-sim/--t-margin/--t-vote explore other values).

Output: the photo with numbered boxes coloured by decision (green automatic, blue candidates = brand + variants or a short list, yellow review, grey unknown) and a side panel, plus a JSON
with every box. Test-split photos are REFUSED (the input file's hash is compared with the raw photos of the test split). Raw photo file
names (they hold store and uploader names) are never printed or written: a photo that is in the dataset is named by its anonymous id.

Usage: python scripts/demo_identify.py --image PATH [--out PATH]   (inputs can live in data/demo/inputs/, outputs go to data/demo/outputs/)
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
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.common.device import get_device  # noqa: E402
from src.data.images import load_image  # noqa: E402
from src.data.paths import check_output_dir  # noqa: E402
from src.detection.model import build_model  # noqa: E402
from src.embeddings import catalog_db  # noqa: E402
from src.embeddings.answers import CANDIDATES, candidate_sets, candidates_info, to_candidates, two_level  # noqa: E402
from src.embeddings.crops import crop_box_with_margin  # noqa: E402
from src.embeddings.demo import build_json, check_not_test, content_index, draw_result, file_sha256  # noqa: E402
from src.embeddings.embed import embed, load_dinov2  # noqa: E402
from src.embeddings.identify import AUTO, REVIEW, UNKNOWN, decide, features_from_sims  # noqa: E402
from src.prelabel import frcnn  # noqa: E402
from src.prelabel.filters import below_height_floor  # noqa: E402


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()


def reading_order(boxes: np.ndarray, height: int) -> np.ndarray:
    """Top to bottom in bands of 6% of the photo height, left to right inside a band: numbers follow the shelves."""
    band = np.floor(((boxes[:, 1] + boxes[:, 3]) / 2) / (0.06 * height))
    return np.lexsort((boxes[:, 0], band))


def run(args) -> int:
    cfg = yaml.safe_load((REPO_ROOT / args.config).read_text())
    pcfg = yaml.safe_load((REPO_ROOT / cfg["prelabel_config"]).read_text()) if "prelabel_config" in cfg else yaml.safe_load((REPO_ROOT / "configs/prelabel.yaml").read_text())
    scfg = yaml.safe_load((REPO_ROOT / "configs/stage5.yaml").read_text())
    s1 = yaml.safe_load((REPO_ROOT / "configs/stage1.yaml").read_text())
    dc, ec, idc, products = cfg["demo"], cfg["embeddings"], cfg["identify"], cfg["products"]
    protected = [REPO_ROOT / d for d in s1["protected_dirs"]]
    path = Path(args.image)
    if not path.is_file():
        raise SystemExit("Input file not found.")                     # no path in the message: it could be a raw photo
    names = {r["photo_id"]: r["file_name"] for r in csv.DictReader((REPO_ROOT / s1["id_maps_dir"] / "photo_ids.csv").open())}
    test_ids = {l.strip() for l in (REPO_ROOT / s1["splits"]["dir"] / "test.txt").read_text().split("\n") if l.strip()}
    index = content_index(REPO_ROOT / s1["raw_dir"], names, REPO_ROOT / dc["hash_cache"])
    pid = check_not_test(file_sha256(path), index, test_ids)           # exits with the anonymous id if it is a test photo
    image_name = pid or path.name
    stem = Path(image_name).stem
    out_img = Path(args.out) if args.out else REPO_ROOT / dc["output_dir"] / f"{stem}_identified.jpg"
    check_output_dir(out_img.parent, protected)
    out_img.parent.mkdir(parents=True, exist_ok=True)
    out_json = out_img.with_suffix(".json")
    th = {"t_sim": args.t_sim if args.t_sim is not None else dc["thresholds"]["t_sim"],
          "t_margin": args.t_margin if args.t_margin is not None else dc["thresholds"]["t_margin"],
          "t_vote": args.t_vote if args.t_vote is not None else dc["thresholds"]["t_vote"],
          "t_neg": args.t_neg if args.t_neg is not None else dc["thresholds"].get("t_neg")}
    print(f"photo: {image_name} ({'dataset photo' if pid else 'not in the dataset'}) | thresholds {th} (DEMO thresholds, not an evaluation result)")

    device = get_device()
    t0 = time.perf_counter()
    det = build_model({**scfg["model"], "box_score_thresh": dc["detector"]["operating_score"]}, REPO_ROOT)
    state = torch.load(REPO_ROOT / dc["detector"]["checkpoint"], map_location="cpu", weights_only=True)
    det.load_state_dict(state["model"])
    det.eval().to(device)
    dino = load_dinov2(ec, device, REPO_ROOT)
    warm = Image.new("RGB", (320, 320), (128, 128, 128))              # first calls on MPS include one-time setup: do it before timing
    frcnn.detect(det, 1, warm, device)
    embed(dino, [warm, warm], device, {**ec, "batch_size": 2})
    sync(device)
    t_load = time.perf_counter() - t0
    image = load_image(path)

    t1 = time.perf_counter()
    boxes, scores = frcnn.detect(det, 1, image, device)
    sync(device)
    keep = ~below_height_floor(boxes, image.height, pcfg["filters"]["min_height_fraction"]) if len(boxes) else np.zeros(0, bool)
    boxes, scores = boxes[keep], scores[keep]
    if len(boxes):
        order = reading_order(boxes, image.height)
        boxes, scores = boxes[order], scores[order]
    t_det = time.perf_counter() - t1

    t2 = time.perf_counter()
    crops = [image.crop(crop_box_with_margin(b, image.width, image.height, 0.10)) for b in boxes]
    vecs = embed(dino, crops, device, ec) if crops else np.zeros((0, 768), np.float32)
    sync(device)
    t_emb = time.perf_counter() - t2

    t3 = time.perf_counter()
    decisions, top3s, cands = [], [], []
    if len(vecs):
        db = cfg["database"]
        try:
            conn = catalog_db.connect(db, REPO_ROOT, db["name"])
        except Exception as error:  # noqa: BLE001
            raise SystemExit(f"Cannot reach the catalog database {db['name']} ({type(error).__name__}). Is the Postgres container running (docker compose up -d postgres) and is it built (python scripts/build_catalog.py)?")
        try:
            hits = [catalog_db.search(conn, v, 1000) for v in vecs]
        finally:
            conn.close()
        entry_ids = sorted(h["id"] for h in hits[0])
        pos = {e: i for i, e in enumerate(entry_ids)}
        meta = {h["id"]: h for h in hits[0]}
        entry_product = np.array([meta[e]["product_id"] for e in entry_ids])
        entry_photo = np.array([meta[e]["source_photo"] for e in entry_ids])
        sims = np.zeros((len(vecs), len(entry_ids)), np.float32)
        for qi, hs in enumerate(hits):
            for h in hs:
                sims[qi, pos[h["id"]]] = h["similarity"]
        feat = features_from_sims(sims, entry_product, entry_photo, products, idc["top_k_photos"], balance=cfg["shelf_round"]["catalog"]["balance_vote"])
        outcome = decide(feat, th["t_sim"], th["t_margin"], th["t_vote"], th["t_neg"])
        # candidates (blue) replace review where there is something useful to say: brand + variants, or a short list
        ca = dc["candidates"]
        sets = candidate_sets(feat["prod_sim"], ca["delta"])
        applies, variants = two_level(feat["prod_sim"], sets, products, cfg["brands"], ca["t_bm"] if ca.get("brand_answers", True) else float("inf"))
        final, kind = to_candidates(outcome, sets, applies, ca["max_list"])
        decisions = list(final)
        for qi in range(len(vecs)):
            top3s.append([(products[j], float(feat["prod_sim"][qi, j])) for j in feat["top3"][qi]])
            cands.append(candidates_info(feat["prod_sim"][qi], variants[qi] if kind[qi] == "brand" else sets[qi], kind[qi] == "brand", products, cfg["brands"])
                         if final[qi] == CANDIDATES else None)
    t_search = time.perf_counter() - t3

    title = image_name
    canvas = draw_result(image, boxes, decisions, top3s, dc["colours"], dc["panel"], title, cands, dc["names"])
    canvas.save(out_img, quality=90)
    counts = {k: decisions.count(k) for k in (AUTO, CANDIDATES, REVIEW, UNKNOWN)}
    result = build_json(image_name, image.size, boxes, scores, decisions, top3s,
                        {"pipeline": {"detector": f"frcnn_v2 {Path(dc['detector']['checkpoint']).name}", "operating_score": dc["detector"]["operating_score"],
                                      "height_floor": pcfg["filters"]["min_height_fraction"], "crop_margin": 0.10, "embedding": ec["model_id"] + " CLS",
                                      "catalog": f"{cfg['database']['name']} ({'originals + shelf entries + unknown negatives' if cfg['catalog'].get('use_shelf') else 'originals'})", "thresholds": th, "candidates": dc["candidates"],
                                      "note": "DEMO thresholds, tuned once on the labeled val crops: not an evaluation result"},
                         "decisions": counts, "seconds": {"load_models_and_warm_up": round(t_load, 2), "detection": round(t_det, 3), "embeddings": round(t_emb, 3), "search_and_rules": round(t_search, 3)}}, cands)
    out_json.write_text(json.dumps(result, indent=1))
    print(f"{len(boxes)} boxes: automatic {counts[AUTO]}, candidates {counts[CANDIDATES]}, review {counts[REVIEW]}, unknown {counts[UNKNOWN]}")
    print(f"timing on {device}: load models + warm-up {t_load:.1f} s (once) | detection {t_det:.2f} s | embeddings {t_emb:.2f} s ({len(crops)} crops) | database search + rules {t_search:.2f} s")
    print(f"image: {out_img.relative_to(REPO_ROOT) if out_img.is_relative_to(REPO_ROOT) else out_img.name}\njson:  {out_json.relative_to(REPO_ROOT) if out_json.is_relative_to(REPO_ROOT) else out_json.name}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", required=True, help="the photo (any path; test-split photos are refused)")
    parser.add_argument("--out", help="output image path (default: data/demo/outputs/<id or file name>_identified.jpg); the JSON is written next to it")
    parser.add_argument("--config", default="configs/stage6.yaml")
    parser.add_argument("--t-sim", type=float, help="override the demo top-1 similarity threshold")
    parser.add_argument("--t-margin", type=float, help="override the demo margin threshold")
    parser.add_argument("--t-vote", type=float, help="override the demo vote threshold")
    parser.add_argument("--t-neg", type=float, help="override the demo negative-gap threshold (unknown if a negative example is closer than the top-1 product by more than this)")
    args = parser.parse_args()
    try:
        return run(args)
    except SystemExit:
        raise
    except Exception as error:  # noqa: BLE001  # never print a traceback: its messages can hold the input path (a raw file name)
        print(f"Failed: {type(error).__name__} (details withheld so that no raw file name can appear; rerun with a copy of the photo under another name to debug)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
