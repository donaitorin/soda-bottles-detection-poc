"""Detection evaluation on COCO files (reused by every later stage).

IMPORTANT: when the ground truth is made of Grounding DINO pseudo-labels, every number computed here is
AGREEMENT with Grounding DINO, not accuracy. The scripts print that wording; keep it in every table.

How it works
  - Matching and per-photo bookkeeping come from pycocotools' COCOeval (the reference implementation of COCO mAP).
  - Predictions shorter than the minimum height (1.5% of the image height) are IGNORED: removed before matching.
    The ground truth has no boxes that small, so such a prediction could never match; removing it keeps it from
    counting as a false positive. The number removed is reported.
  - `summarize` re-implements COCOeval's accumulation step on the per-photo results, so that a bootstrap can
    resample WHOLE STORES (a photo list with repeats). A test checks that it reproduces COCOeval's own AP numbers.
"""
import contextlib
import io

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.prelabel.filters import below_height_floor

REC_THRS = np.linspace(0.0, 1.0, 101)  # COCO's 101 recall points


def ignore_below_floor(preds: dict, sizes: dict, min_height_fraction: float) -> tuple[dict, int]:
    """preds: {photo id: (boxes xyxy (n, 4), scores (n,))}; sizes: {photo id: (width, height)}.
    Returns the preds without the boxes under the floor and how many were removed."""
    out, removed = {}, 0
    for pid, (boxes, scores) in preds.items():
        boxes, scores = np.asarray(boxes, float).reshape(-1, 4), np.asarray(scores, float)
        short = below_height_floor(boxes, sizes[pid][1], min_height_fraction) if len(boxes) else np.zeros(0, bool)
        removed += int(short.sum())
        out[pid] = (boxes[~short], scores[~short])
    return out, removed


def per_photo_results(gt: dict, preds: dict, max_dets: int = 300) -> dict:
    """COCOeval per-photo matching. gt: COCO dict (category 1 = bottle); preds as in ignore_below_floor.
    Returns ({photo id: evalImg dict or None}, (reference AP, reference AP50) from COCOeval itself). Dense shelves hold up to ~125 boxes, so maxDets is 300, not 100."""
    coco_gt = COCO()
    coco_gt.dataset = {"images": gt["images"], "annotations": gt["annotations"], "categories": gt["categories"]}
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt.createIndex()
    image_id = {im["photo_id"]: im["id"] for im in gt["images"]}
    results = []
    for pid, (boxes, scores) in preds.items():
        for (x1, y1, x2, y2), s in zip(boxes, scores):
            results.append({"image_id": image_id[pid], "category_id": 1, "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(s)})
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(results) if results else COCO()
        if not results:
            coco_dt.dataset = {"images": gt["images"], "annotations": [], "categories": gt["categories"]}
            coco_dt.createIndex()
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.params.maxDets = [1, 10, max_dets]
        ev.params.areaRng = [[0, 1e10]]
        ev.params.areaRngLbl = ["all"]
        ev.evaluate()
        ev.accumulate()
    by_image = dict(zip(ev.params.imgIds, ev.evalImgs))
    # COCOeval's own AP and AP50 (precision array: IoU thresholds x recall points x classes x areas x maxDets),
    # kept to check `summarize` against the reference implementation.
    precision = ev.eval["precision"][:, :, 0, 0, -1]
    reference = (float(precision[precision > -1].mean()), float(precision[0][precision[0] > -1].mean()))
    return {im["photo_id"]: by_image[im["id"]] for im in gt["images"]}, reference


def summarize(eval_imgs: list, max_dets: int = 300) -> dict:
    """AP (IoU .5:.95), AP50, recall at IoU 0.5 (all detections kept) from a list of per-photo results (repeats allowed)."""
    eval_imgs = [e for e in eval_imgs if e is not None]
    if not eval_imgs:
        return {"AP": float("nan"), "AP50": float("nan"), "recall50": float("nan")}
    scores = np.concatenate([e["dtScores"][:max_dets] for e in eval_imgs])
    order = np.argsort(-scores, kind="mergesort")
    dtm = np.concatenate([e["dtMatches"][:, :max_dets] for e in eval_imgs], axis=1)[:, order]
    dtig = np.concatenate([e["dtIgnore"][:, :max_dets] for e in eval_imgs], axis=1)[:, order]
    n_gt = int(sum(int((np.asarray(e["gtIgnore"]) == 0).sum()) for e in eval_imgs))
    if n_gt == 0:
        return {"AP": float("nan"), "AP50": float("nan"), "recall50": float("nan")}
    ap_per_iou, recall50 = [], 0.0
    for t in range(dtm.shape[0]):
        tp = np.cumsum(np.logical_and(dtm[t] > 0, ~dtig[t].astype(bool)))
        fp = np.cumsum(np.logical_and(dtm[t] == 0, ~dtig[t].astype(bool)))
        rc = tp / n_gt
        pr = (tp / (tp + fp + np.spacing(1))).tolist()
        for i in range(len(pr) - 1, 0, -1):  # precision envelope: monotonically decreasing
            if pr[i] > pr[i - 1]:
                pr[i - 1] = pr[i]
        q = np.zeros(len(REC_THRS))
        idx = np.searchsorted(rc, REC_THRS, side="left")
        for ri, pi in enumerate(idx):
            if pi < len(pr):
                q[ri] = pr[pi]
        ap_per_iou.append(q.mean())
        if t == 0:
            recall50 = float(rc[-1]) if len(rc) else 0.0
    return {"AP": float(np.mean(ap_per_iou)), "AP50": float(ap_per_iou[0]), "recall50": recall50}


def operating_point(eval_imgs: list) -> dict:
    """Precision, recall and F1 at IoU 0.5 for the predictions given (already thresholded by score)."""
    tp = fp = n_gt = 0
    for e in eval_imgs:
        if e is None:
            continue
        ig = e["dtIgnore"][0].astype(bool)
        tp += int(np.sum((e["dtMatches"][0] > 0) & ~ig))
        fp += int(np.sum((e["dtMatches"][0] == 0) & ~ig))
        n_gt += int((np.asarray(e["gtIgnore"]) == 0).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / n_gt if n_gt else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": n_gt - tp}


def gt_counts(gt: dict) -> dict:
    """Ground-truth boxes per photo id."""
    counts = {im["photo_id"]: 0 for im in gt["images"]}
    pid_of = {im["id"]: im["photo_id"] for im in gt["images"]}
    for a in gt["annotations"]:
        counts[pid_of[a["image_id"]]] += 1
    return counts


def bootstrap_by_store(photo_store: dict, metric_fn, n: int, seed: int, level: float = 0.95) -> dict:
    """Percentile interval of metric_fn(list of photo ids) over resamples of WHOLE STORES (with replacement).
    With only a handful of stores the interval is wide and only a rough guide."""
    rng = np.random.default_rng(seed)
    stores = sorted(set(photo_store.values()))
    photos_of = {s: [p for p, st in photo_store.items() if st == s] for s in stores}
    samples = []
    for _ in range(n):
        picked = rng.choice(len(stores), size=len(stores), replace=True)
        samples.append(metric_fn([p for i in picked for p in photos_of[stores[i]]]))
    out = {}
    for key in samples[0]:
        vals = np.array([s[key] for s in samples], float)
        vals = vals[~np.isnan(vals)]
        lo, hi = np.percentile(vals, [(1 - level) / 2 * 100, (1 + level) / 2 * 100]) if len(vals) else (float("nan"),) * 2
        out[key] = (float(lo), float(hi))
    return out
