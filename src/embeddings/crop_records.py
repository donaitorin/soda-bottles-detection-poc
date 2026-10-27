"""Where every crop of the experiments comes from (photo + box), and how to cut it: 10% margin, 0% margin, or the SAM 2 mask on a
neutral background. Catalog originals, owner-labeled train shelf entries and labeled val crops all go through the SAME function, so
the catalog and the query crops are always cropped the same way. Raw file names are only used to open files, never printed.
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.images import load_image
from src.embeddings.crops import crop_box_with_margin
from src.embeddings.sam_masks import masks_for_boxes, neutral_background

VARIANTS = ("margin10", "margin0", "sam")


def _rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def build_records(cfg: dict, s1: dict, root: Path, val_ids, shelf_ids, orig_keys) -> dict:
    """{'val': [...], 'shelf': [...], 'orig': [...]} of {key, path, box (xyxy pixels)}, in the order of the given ids."""
    names = {r["photo_id"]: r["file_name"] for r in _rows(root / s1["id_maps_dir"] / "photo_ids.csv")}
    raw = root / s1["raw_dir"]
    mdir = root / cfg["crops"]["mapping_dir"]

    def from_mapping(name, ids):
        m = {r["crop_id"]: r for r in _rows(mdir / name / "mapping.csv")}
        return [{"key": c, "path": raw / names[m[c]["photo_id"]], "box": tuple(float(m[c][k]) for k in ("x1", "y1", "x2", "y2"))} for c in ids]
    crops = {r["file"]: r for r in _rows(root / cfg["catalog"]["crop"]["crops_dir"] / "crops.csv")}
    orig = [{"key": k, "path": root / cfg["catalog"]["photos_dir"] / k, "box": tuple(float(v) for v in crops[k]["box"].split(";"))} for k in orig_keys]
    return {"val": from_mapping(cfg["crops"]["sample"]["name"], val_ids), "shelf": from_mapping(cfg["shelf_round"]["name"], shelf_ids), "orig": orig}


def render(records: list[dict], variant: str, margin: float, sam=None, device=None) -> list[Image.Image]:
    """One image per record, in order. margin10 / margin0: the box grown by `margin` (0.10) or not at all. sam: the SAM 2 mask of the box
    on a white background, inside the 10% margin region (the same region as margin10, so only the background differs)."""
    assert variant in VARIANTS, variant
    out = [None] * len(records)
    by_path = defaultdict(list)
    for i, r in enumerate(records):
        by_path[r["path"]].append(i)
    for path, idxs in by_path.items():
        image = load_image(path)
        boxes = np.array([records[i]["box"] for i in idxs], float)
        masks = masks_for_boxes(*sam, image, boxes, device) if variant == "sam" else None
        for j, i in enumerate(idxs):
            region = crop_box_with_margin(boxes[j], image.width, image.height, 0.0 if variant == "margin0" else margin)
            crop = image.crop(region)
            out[i] = neutral_background(crop, masks[j][region[1]:region[3], region[0]:region[2]]) if variant == "sam" else crop
    return out
