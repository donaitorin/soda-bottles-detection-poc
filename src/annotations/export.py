"""Build COCO files from the pre-labels and (optionally) Label Studio annotations of the owner.

Label sources, per photo:
  prelabel_unverified  the Grounding DINO boxes as they are (nobody looked at them)
  owner_corrected      the boxes of the owner's submitted Label Studio annotation; it wins over the pre-label
Every metric computed against `prelabel_unverified` photos measures agreement with Grounding DINO, not accuracy.

Formats: pixel xyxy inside the project (like torchvision), COCO `bbox` = [x, y, width, height] in pixels,
Label Studio = percent. The Label Studio side is converted with src.prelabel.labelstudio.
"""
from collections import Counter

import numpy as np

from src.prelabel.filters import below_height_floor
from src.prelabel.labelstudio import ls_result_to_xyxy

PRELABEL, OWNER = "prelabel_unverified", "owner_corrected"
DISCLAIMER = ("Boxes with label_source=prelabel_unverified are Grounding DINO pseudo-labels that nobody verified: "
              "a metric computed against them measures agreement with Grounding DINO, not accuracy.")


def xyxy_to_coco_bbox(box) -> list[float]:
    """COCO bbox [x, y, width, height] in pixels from a pixel xyxy box."""
    x1, y1, x2, y2 = (float(v) for v in box)
    return [x1, y1, x2 - x1, y2 - y1]


def clamp_xyxy(box, width: float, height: float) -> list[float]:
    x1, y1, x2, y2 = (float(v) for v in box)
    return [min(max(x1, 0.0), width), min(max(y1, 0.0), height), min(max(x2, 0.0), width), min(max(y2, 0.0), height)]


def pick_annotation(task: dict) -> dict | None:
    """The annotation to use for a Label Studio task: the most recently updated one that was not skipped."""
    ok = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
    if not ok:
        return None
    return max(ok, key=lambda a: (a.get("updated_at") or "", a.get("id") or 0))


def read_owner_annotations(ls_export: list[dict], from_name: str, choices_name: str, known_sizes: dict) -> tuple[dict, Counter]:
    """Owner annotations by photo id from a Label Studio JSON export (the full JSON, not the COCO export,
    which drops the per-photo choices and lead_time).

    Returns ({photo id: {boxes (xyxy px), products, lead_time, updated_at}}, notes). A submitted annotation
    replaces the pre-label as a whole, including when it holds zero boxes (the owner says: no bottles).
    """
    owner, notes = {}, Counter()
    for task in ls_export:
        pid = (task.get("data") or {}).get("photo_id")
        if pid not in known_sizes:
            notes["task_with_unknown_photo_id"] += 1
            continue
        ann = pick_annotation(task)
        if ann is None:
            notes["task_without_submitted_annotation"] += 1
            continue
        boxes, products = [], None
        for result in ann.get("result", []):
            if result.get("from_name") == from_name and result.get("type") == "rectanglelabels":
                if (result["original_width"], result["original_height"]) != tuple(known_sizes[pid]):
                    raise ValueError(f"{pid}: annotation image size differs from the photo size")
                boxes.append(ls_result_to_xyxy(result))
            elif result.get("from_name") == choices_name and result.get("type") == "choices":
                products = list(result["value"]["choices"])
        owner[pid] = {"boxes": boxes, "products": products, "lead_time": ann.get("lead_time"),
                      "updated_at": ann.get("updated_at")}
        if len(task.get("annotations", [])) > 1:
            notes["tasks_with_several_annotations_latest_used"] += 1
    return owner, notes


def build_coco(photo_ids: list[str], predictions: dict, owner: dict, file_names: dict, store_of: dict,
               split: str, min_height_fraction: float, model_version: str) -> tuple[dict, Counter]:
    """COCO dict for one split plus counters (photos/boxes by source, boxes dropped by reason)."""
    stats: Counter = Counter()
    images, annotations = [], []
    for image_id, pid in enumerate(sorted(photo_ids), 1):
        pred = predictions[pid]
        width, height = pred["width"], pred["height"]
        if pid in owner:
            source, boxes, scores = OWNER, owner[pid]["boxes"], [None] * len(owner[pid]["boxes"])
        else:
            source, boxes, scores = PRELABEL, pred["boxes"], pred["scores"]
        stats[f"photos_{source}"] += 1
        boxes = np.array([clamp_xyxy(b, width, height) for b in boxes], dtype=float).reshape(-1, 4)
        degenerate = (boxes[:, 2] - boxes[:, 0] <= 0) | (boxes[:, 3] - boxes[:, 1] <= 0)  # clamping can flatten a box
        short = ~degenerate & below_height_floor(boxes, height, min_height_fraction)
        stats[f"dropped_degenerate_{source}"] += int(degenerate.sum())
        stats[f"dropped_below_height_floor_{source}"] += int(short.sum())
        keep = ~degenerate & ~short
        stats[f"boxes_{source}"] += int(keep.sum())
        images.append({"id": image_id, "file_name": file_names[pid], "width": width, "height": height,
                       "photo_id": pid, "store_code": store_of[pid], "split": split, "label_source": source,
                       "products": owner[pid]["products"] if pid in owner else None,
                       "lead_time_s": owner[pid]["lead_time"] if pid in owner else None})
        for box, score, kept in zip(boxes, scores, keep):
            if not kept:
                continue
            ann = {"id": len(annotations) + 1, "image_id": image_id, "category_id": 1,
                   "bbox": [round(v, 2) for v in xyxy_to_coco_bbox(box)], "iscrowd": 0, "label_source": source}
            ann["area"] = round(ann["bbox"][2] * ann["bbox"][3], 2)
            if score is not None:
                ann["score"] = score
            annotations.append(ann)
    coco = {"info": {"description": f"soda bottle POC, split {split}", "prelabel_model_version": model_version,
                     "min_height_fraction": min_height_fraction, "disclaimer": DISCLAIMER},
            "licenses": [], "categories": [{"id": 1, "name": "bottle"}], "images": images, "annotations": annotations}
    return coco, stats
