"""Sanity checks on the COCO files written by export.py. Errors fail the gate; warnings are reported."""
import re
from collections import Counter

import numpy as np

from src.annotations.export import OWNER, PRELABEL
from src.prelabel.filters import below_height_floor, box_iou


def check_coco(coco: dict, expected_ids: list[str], min_height_fraction: float, duplicate_iou: float,
               boxes_per_photo: tuple[int, int], id_regex: str) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    images = {im["id"]: im for im in coco["images"]}
    if len(images) != len(coco["images"]):
        errors.append("duplicate image ids")
    ids = [im["photo_id"] for im in coco["images"]]
    if sorted(ids) != sorted(expected_ids):
        errors.append(f"photos differ from the split list: {len(set(ids) ^ set(expected_ids))} ids not in both")
    for im in coco["images"]:
        if not re.match(id_regex, im["file_name"]) or not im["file_name"].startswith(im["photo_id"]):
            errors.append(f"{im['photo_id']}: file_name is not a photo id")
        if im["label_source"] not in (PRELABEL, OWNER):
            errors.append(f"{im['photo_id']}: unknown label_source {im['label_source']}")
    if [c["name"] for c in coco["categories"]] != ["bottle"]:
        errors.append("categories must be exactly ['bottle']")
    ann_ids = [a["id"] for a in coco["annotations"]]
    if len(set(ann_ids)) != len(ann_ids):
        errors.append("duplicate annotation ids")

    per_image: dict[int, list] = {i: [] for i in images}
    for a in coco["annotations"]:
        im = images.get(a["image_id"])
        if im is None:
            errors.append(f"annotation {a['id']} points to a missing image")
            continue
        x, y, w, h = a["bbox"]
        eps = 0.01  # boxes are rounded to 2 decimals
        if x < -eps or y < -eps or x + w > im["width"] + eps or y + h > im["height"] + eps:
            errors.append(f"{im['photo_id']}: box outside the image")
        if w <= 0 or h <= 0 or a["area"] <= 0:
            errors.append(f"{im['photo_id']}: box with area zero")
        if a["category_id"] != 1:
            errors.append(f"{im['photo_id']}: category must be 1")
        if h < min_height_fraction * im["height"] - eps:
            errors.append(f"{im['photo_id']}: box under the {min_height_fraction:.1%} height floor")
        per_image[a["image_id"]].append([x, y, x + w, y + h])

    low, high = boxes_per_photo
    for image_id, boxes in per_image.items():
        pid = images[image_id]["photo_id"]
        if boxes:
            b = np.array(boxes)
            iou = box_iou(b, b)
            np.fill_diagonal(iou, 0)
            n_dup = int((iou > duplicate_iou).sum() // 2)
            if n_dup:
                errors.append(f"{pid}: {n_dup} duplicate box pair(s) at IoU > {duplicate_iou}")
        if not low <= len(boxes) <= high:
            warnings.append(f"{pid}: {len(boxes)} boxes (outside {low}..{high})")
    return errors, warnings


def counts_per_photo(coco: dict) -> list[int]:
    c = Counter(a["image_id"] for a in coco["annotations"])
    return sorted(c.get(im["id"], 0) for im in coco["images"])
