"""Find exact and near-duplicate photos and prepare a visual review.

Nothing is excluded here. The script writes, in the gitignored data/review/duplicates/:
  - one side-by-side image per flagged pair (labeled with photo ids, never file names)
  - decisions.csv with a SUGGESTED decision per pair; the owner edits it and confirms.
It also fills the two duplicate flags in data/metadata.csv.

Method: exact copies have the same SHA-256 of the file bytes. Near duplicates use a
perceptual hash (pHash: a hash of the low-frequency DCT content of the picture) of
16x16 = 256 bits. The distance between two photos is the Hamming distance (number of
differing bits). Photos of different shelves usually differ in many bits; the same
photo re-saved or resized differs in very few. Because the split is strictly by store, a
near duplicate inside one store only adds redundancy; a pair across two stores (the same
photo uploaded under two stores) is the real risk, so those pairs are listed first.

Usage: python scripts/find_duplicates.py --config configs/stage1.yaml
"""
import argparse
import csv
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

import imagehash
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.images import load_image  # noqa: E402
from src.data.paths import check_output_dir  # noqa: E402

DECISION_FIELDS = ["pair_id", "id_a", "id_b", "distance", "same_store", "exact_copy", "suggested_keep", "decision"]


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def side_by_side(img_a: Image.Image, img_b: Image.Image, header: str, labels: tuple[str, str],
                 cross_store: bool, height: int) -> Image.Image:
    def fit(img):
        return img.resize((max(1, round(img.width * height / img.height)), height), Image.LANCZOS)
    a, b = fit(img_a), fit(img_b)
    bar, gap = 56, 12
    canvas = Image.new("RGB", (a.width + b.width + gap, height + bar), "white")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, canvas.width, bar], fill=(170, 30, 30) if cross_store else (70, 70, 70))
    font = ImageFont.load_default(size=18)
    draw.text((8, 6), header, fill="white", font=font)
    draw.text((8, 30), f"A (left): {labels[0]}    B (right): {labels[1]}", fill="white", font=font)
    canvas.paste(a, (0, bar))
    canvas.paste(b, (a.width + gap, bar))
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage1.yaml")
    parser.add_argument("--max-hamming", type=int, help="override the threshold in the config (for exploring)")
    parser.add_argument("--review-dir", help="override the review folder (for exploring)")
    args = parser.parse_args()
    cfg = yaml.safe_load((REPO_ROOT / args.config).read_text())
    dup = cfg["duplicates"]
    if args.max_hamming is not None:
        dup["max_hamming"] = args.max_hamming
    if args.review_dir:
        dup["review_dir"] = args.review_dir
    review_dir = REPO_ROOT / dup["review_dir"]
    # The script deletes old P*.jpg files in review_dir, so a typo in --review-dir must never
    # point it at the photo folders.
    try:
        check_output_dir(review_dir, [REPO_ROOT / d for d in cfg["protected_dirs"]])
    except ValueError as err:
        raise SystemExit(f"error: {err}")
    maps_dir = REPO_ROOT / cfg["id_maps_dir"]
    meta_path = REPO_ROOT / cfg["metadata_csv"]

    meta = {r["photo_id"]: r for r in csv.DictReader(meta_path.open(newline=""))}
    names = {r["photo_id"]: r["file_name"] for r in csv.DictReader((maps_dir / "photo_ids.csv").open(newline=""))}
    # Screenshots are excluded anyway, so they are not compared.
    ids = sorted(i for i in names if meta[i]["screenshot"] != "True")
    paths = {i: REPO_ROOT / cfg["raw_dir"] / names[i] for i in ids}

    sha = {i: sha256_of(paths[i]) for i in ids}
    bits = np.stack([imagehash.phash(load_image(paths[i]), hash_size=dup["hash_size"]).hash.flatten()
                     for i in ids])
    dist = (bits[:, None, :] != bits[None, :, :]).sum(-1)  # Hamming distance matrix
    n_bits = bits.shape[1]

    off_diag = dist + np.eye(len(ids), dtype=int) * n_bits  # ignore the diagonal
    nn = off_diag.min(axis=1)
    print(f"compared {len(ids)} photos ({len(ids) * (len(ids) - 1) // 2} pairs), hash = {n_bits} bits")
    print("nearest-neighbor distance per photo (bits):", dict(sorted(Counter(
        "0-4" if d <= 4 else "5-8" if d <= 8 else "9-12" if d <= 12 else "13-16" if d <= 16
        else "17-24" if d <= 24 else "25-48" if d <= 48 else ">48" for d in nn).items())))
    iu = np.triu_indices(len(ids), 1)
    pair_d = dist[iu]
    print("pairs at or below:", {t: int((pair_d <= t).sum()) for t in (0, 4, 8, 12, 16, 24)})

    flagged = []
    for a, b in zip(*iu):
        exact = sha[ids[a]] == sha[ids[b]]
        if exact or dist[a, b] <= dup["max_hamming"]:
            flagged.append((ids[a], ids[b], int(dist[a, b]), exact))
    cross = lambda p: p[0].split("_")[0] != p[1].split("_")[0]  # noqa: E731
    flagged.sort(key=lambda p: (not cross(p), p[2], p[0], p[1]))  # cross-store first, then closest first

    review_dir = REPO_ROOT / dup["review_dir"]
    review_dir.mkdir(parents=True, exist_ok=True)
    for old in review_dir.glob("P*.jpg"):
        old.unlink()  # drop images of a previous run
    csv_path = review_dir / "decisions.csv"
    previous = {}
    if csv_path.exists():  # never lose decisions the owner already typed
        previous = {(r["id_a"], r["id_b"]): r["decision"] for r in csv.DictReader(csv_path.open(newline=""))}

    rows = []
    for k, (a, b, d, exact) in enumerate(flagged, 1):
        same_store = not cross((a, b))
        pixels = lambda i: int(meta[i]["width"]) * int(meta[i]["height"])  # noqa: E731
        keep = b if pixels(b) > pixels(a) else a  # the larger resolution
        if exact:
            suggestion = "drop_b"           # an identical file: one copy is enough
        elif same_store:
            suggestion = "keep_both"        # same store: no leak between splits, only redundancy
        else:
            suggestion = ""                 # different stores: the owner must decide
        decision = previous.get((a, b), suggestion)
        pair_id = f"P{k:03d}"
        rows.append({"pair_id": pair_id, "id_a": a, "id_b": b, "distance": d, "same_store": same_store,
                     "exact_copy": exact, "suggested_keep": keep, "decision": decision})
        header = (f"{pair_id}   distance {d}/{n_bits} bits   "
                  f"{'SAME STORE' if same_store else 'DIFFERENT STORES'}   {'EXACT COPY' if exact else ''}")
        labels = tuple(f"{i} ({meta[i]['width']}x{meta[i]['height']})" for i in (a, b))
        side_by_side(load_image(paths[a]), load_image(paths[b]), header, labels, not same_store,
                     dup["review_height_px"]).save(review_dir / f"{pair_id}_{a}_{b}.jpg", quality=88)

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DECISION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    if args.max_hamming is not None or args.review_dir:
        print("exploration run: data/metadata.csv flags were NOT updated")
        return 0
    exact_ids = {i for r in rows if r["exact_copy"] for i in (r["id_a"], r["id_b"])}
    near_ids = {i for r in rows for i in (r["id_a"], r["id_b"])}
    for i, row in meta.items():
        row["exact_duplicate"] = i in exact_ids
        row["near_duplicate_candidate"] = i in near_ids
    with meta_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(next(iter(meta.values())).keys()))
        writer.writeheader()
        writer.writerows(meta.values())

    n_cross = sum(r["same_store"] is False for r in rows)
    print(f"flagged pairs: {len(rows)} (different stores: {n_cross}, same store: {len(rows) - n_cross}, "
          f"exact copies: {sum(r['exact_copy'] for r in rows)}) at max_hamming={dup['max_hamming']}")
    print(f"review images and decisions.csv: {review_dir.relative_to(REPO_ROOT)}/ (gitignored)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
