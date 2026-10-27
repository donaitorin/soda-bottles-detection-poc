"""Crops from boxes, and the stratified sample the owner labels by hand (the Stage 6 yardstick).

Boxes are xyxy pixels. `crop_box_with_margin` enlarges a box by a share of its size and clamps it to the photo.
Crop ids are `<photo id>_<k>` with k the index of the box among the photo's annotations in the COCO file.
"""
import random
from collections import defaultdict

import numpy as np


def crop_id(photo_id: str, k: int) -> str:
    return f"{photo_id}_{k:03d}"


def crop_box_with_margin(box, width: int, height: int, margin: float) -> tuple[int, int, int, int]:
    """Integer crop rectangle (x1, y1, x2, y2): the box grown by `margin` of its width/height on each side, clamped."""
    x1, y1, x2, y2 = (float(v) for v in box)
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    cx1, cy1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
    cx2, cy2 = min(width, int(round(x2 + mx))), min(height, int(round(y2 + my)))
    return cx1, cy1, max(cx2, cx1 + 1), max(cy2, cy1 + 1)


def allocate_quota(available: dict, total: int, minimum: int) -> dict:
    """Split `total` crops over stores: proportional to the number of boxes, at least `minimum` per store, never more
    than the store can give. `available` is {store: (n boxes, max crops the store can give)}. Deterministic."""
    cap = {s: min(n, c) for s, (n, c) in available.items()}
    quota = {s: min(minimum, cap[s]) for s in available}
    while sum(quota.values()) < total:
        room = [s for s in available if quota[s] < cap[s]]
        if not room:
            break
        left = total - sum(quota.values())
        weight = sum(available[s][0] for s in room)
        grew = False
        for s in sorted(room, key=lambda s: -available[s][0]):
            add = min(int(left * available[s][0] / weight), cap[s] - quota[s])
            if add > 0:
                quota[s] += add
                grew = True
        if not grew:  # leftovers of the integer division: one at a time to the largest stores with room
            for s in sorted(room, key=lambda s: -available[s][0]):
                if sum(quota.values()) < total and quota[s] < cap[s]:
                    quota[s] += 1
    return quota


def sample_crops(candidates: list[tuple[str, int, str]], quota: dict, max_per_photo: int, seed: int) -> list[tuple[str, int, str]]:
    """candidates: (photo id, box index, store). For each store, take its quota at random but spread over its photos
    (round robin over shuffled photos, at most `max_per_photo` from one photo). Seeded and reproducible."""
    rng = random.Random(seed)
    by_store = defaultdict(lambda: defaultdict(list))
    for pid, k, store in sorted(candidates):
        by_store[store][pid].append((pid, k, store))
    chosen = []
    for store in sorted(by_store):
        photos = sorted(by_store[store])
        rng.shuffle(photos)
        for pid in photos:
            rng.shuffle(by_store[store][pid])
        taken = defaultdict(int)
        picked = []
        while len(picked) < quota.get(store, 0):
            progress = False
            for pid in photos:
                if len(picked) >= quota.get(store, 0):
                    break
                if taken[pid] < max_per_photo and taken[pid] < len(by_store[store][pid]):
                    picked.append(by_store[store][pid][taken[pid]])
                    taken[pid] += 1
                    progress = True
            if not progress:
                break
        chosen += picked
    return chosen


def pick_targeted(candidates: list[tuple[str, int, str]], prod_sim, products: list[str], target: list[str], per_product: int, top_rank: int,
                  max_per_photo: int, max_per_store: int, taken: set) -> list[tuple[str, int, str, str]]:
    """Crops for the rare products, picked by retrieval (not by label): for each target product in turn, the candidates whose top-`top_rank`
    products (by similarity to the catalog originals) include it, most similar first. `taken` holds (photo, box) pairs that are already
    chosen and is updated. A photo gives at most `max_per_photo` crops in total and a store at most `max_per_store`. Deterministic.
    candidates: (photo id, box index, store); prod_sim: (N, len(products)). Returns (photo, box, store, product it was picked for)."""
    prod_sim = np.asarray(prod_sim)
    rank = np.argsort(-prod_sim, axis=1)[:, :top_rank]
    per_photo, per_store, out = defaultdict(int), defaultdict(int), []
    for pid, k, store in candidates:                       # the crops already chosen count towards the photo and store caps
        if (pid, k) in taken:
            per_photo[pid] += 1
    for name in target:
        j = products.index(name)
        eligible = [i for i in np.argsort(-prod_sim[:, j], kind="stable") if j in rank[i]]
        got = 0
        for i in eligible:
            pid, k, store = candidates[i]
            if got >= per_product:
                break
            if (pid, k) in taken or per_photo[pid] >= max_per_photo or per_store[store] >= max_per_store:
                continue
            taken.add((pid, k))
            per_photo[pid] += 1
            per_store[store] += 1
            got += 1
            out.append((pid, k, store, name))
    return out


def resize_for_display(image, min_long: int, max_long: int):
    """Upscale small crops (or shrink huge ones) for DISPLAY in Label Studio only; embeddings always use the original pixels."""
    from PIL import Image
    long_side = max(image.size)
    scale = min_long / long_side if long_side < min_long else (max_long / long_side if long_side > max_long else 1.0)
    if scale == 1.0:
        return image
    return image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.LANCZOS)
