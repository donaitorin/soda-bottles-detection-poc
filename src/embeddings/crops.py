"""Crops from boxes, and the stratified sample the owner labels by hand (the Stage 6 yardstick).

Boxes are xyxy pixels. `crop_box_with_margin` enlarges a box by a share of its size and clamps it to the photo.
Crop ids are `<photo id>_<k>` with k the index of the box among the photo's annotations in the COCO file.
"""
import random

from collections import defaultdict


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
