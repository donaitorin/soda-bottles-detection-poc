"""Conversions between pixel boxes and Label Studio, and task building.

Coordinates:
  - pixel boxes (torchvision, Grounding DINO after post-processing): xyxy = (x1, y1, x2, y2) in pixels of
    the original image, origin at the top-left corner.
  - Label Studio `rectanglelabels`: x, y (top-left corner), width, height, all in PERCENT (0 to 100) of the
    image width (x, width) and image height (y, height), plus original_width / original_height in pixels.
A mix-up between the two (pixels vs percent, or width/height swapped) shows up as shifted boxes on
non-square images, so every conversion lives here and is tested.
"""
import hashlib
import json

PERCENT = 100.0


def xyxy_to_ls_result(box, width: int, height: int, score: float, from_name: str, to_name: str,
                      label: str, result_id: str | None = None) -> dict:
    """One Label Studio `rectanglelabels` result from a pixel xyxy box."""
    x1, y1, x2, y2 = (float(v) for v in box)
    # Clamp to the image: a box can poke out by a pixel or two after the model's own rescaling.
    x1, x2 = max(0.0, min(x1, width)), max(0.0, min(x2, width))
    y1, y2 = max(0.0, min(y1, height)), max(0.0, min(y2, height))
    result = {
        "from_name": from_name, "to_name": to_name, "type": "rectanglelabels",
        "original_width": int(width), "original_height": int(height), "image_rotation": 0,
        "value": {
            "x": PERCENT * x1 / width, "y": PERCENT * y1 / height,
            "width": PERCENT * (x2 - x1) / width, "height": PERCENT * (y2 - y1) / height,
            "rotation": 0, "rectanglelabels": [label],
        },
        "score": float(score),
    }
    if result_id is not None:
        result["id"] = result_id
    return result


def ls_result_to_xyxy(result: dict) -> tuple[float, float, float, float]:
    """Inverse of xyxy_to_ls_result: pixel xyxy from a Label Studio result (uses its original size)."""
    value, width, height = result["value"], result["original_width"], result["original_height"]
    x1 = value["x"] / PERCENT * width
    y1 = value["y"] / PERCENT * height
    return x1, y1, x1 + value["width"] / PERCENT * width, y1 + value["height"] / PERCENT * height


def config_hash(cfg: dict) -> str:
    """Short hash of the frozen pre-labeling settings; stored as the prediction's model_version."""
    frozen = {"prelabeler": cfg["prelabeler"],
              "dino": {k: cfg["dino"][k] for k in ("model_id", "revision", "prompt", "box_threshold", "text_threshold")},
              "filters": cfg["filters"]}
    return hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()[:8]


def model_version(cfg: dict) -> str:
    return f"{cfg['prelabeler']}@{cfg['dino']['revision'][:8]}+cfg{config_hash(cfg)}"


def build_task(photo_id: str, split: str, store_code: str, image_url: str, boxes, scores,
               width: int, height: int, cfg: dict) -> dict:
    """A Label Studio task with its prediction. Only ids appear: no file names."""
    ls = cfg["label_studio"]
    results = [xyxy_to_ls_result(b, width, height, s, ls["from_name"], ls["to_name"], ls["label"], f"{photo_id}_{k}")
               for k, (b, s) in enumerate(zip(boxes, scores))]
    task = {"data": {"image": image_url, "photo_id": photo_id, "split": split, "store_code": store_code}}
    task["predictions"] = [{
        "model_version": model_version(cfg),
        "score": float(sum(scores) / len(scores)) if len(scores) else 0.0,
        "result": results,
    }]
    return task
