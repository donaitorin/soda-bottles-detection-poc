"""OCR engine trial on bottle crops: does the engine read the brand or variant word, what does it read wrongly, how fast.

DECISION INPUT ONLY. Run it in a throwaway environment that has easyocr and/or pytesseract (NOT in soda-poc, and nothing from this
trial goes into requirements.txt or the pipeline). Crops: the 30 catalog crops (known text) plus about 30 labeled val crops (a
mix of products and stores, a few `unknown`), taken from the ORIGINAL photo pixels. Per crop: the text read, the keywords found
(fuzzy, accent- and case-insensitive), the expected keywords of its product, and the time. Outputs (gitignored, ids only) in
data/review/ocr_trial/: results_<engine>.json and contact sheets.

Usage (in the throwaway env): python scripts/ocr_trial.py --engine easyocr [--device cpu|mps] | --engine tesseract
"""
import argparse
import csv
import difflib
import json
import random
import re
import statistics
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[1]


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().upper()
    return re.sub(r"[^A-Z0-9 \-]", " ", text)


def keywords_found(text: str, keywords: list[str], min_ratio: float) -> list[str]:
    """Keywords present in the text with a tolerance for 1 or 2 wrong letters (OCR confuses O/0, I/L...).
    Compared with spaces and hyphens removed, over a sliding window; 4-letter words must match exactly."""
    flat = re.sub(r"[ \-]", "", normalize(text))
    found = []
    for k in keywords:
        kf = k.replace("-", "")
        hit = kf in flat
        if not hit and len(kf) >= 5:
            n = len(kf)
            hit = any(difflib.SequenceMatcher(None, flat[i:i + n], kf).ratio() >= min_ratio for i in range(max(0, len(flat) - n + 1)))
        if hit:
            found.append(k)
    return found


def words_found(text: str, keywords: list[str], min_ratio: float) -> list[str]:
    """STRICT variant: compares whole words (and pairs of adjacent words, for COCA COLA) instead of any substring, so a keyword
    inside a longer word (FANTA inside FANTASIA, from "bebida de fantasia") does not count. It misses words the OCR glued to
    their neighbours (ZEROAZUCAR)."""
    tokens = [t for t in re.split(r"[ \-]+", normalize(text)) if t]
    cands = tokens + [a + b for a, b in zip(tokens, tokens[1:])]
    found = []
    for k in keywords:
        kf = k.replace("-", "")
        thr = 1.0 if len(kf) <= 4 else min_ratio
        if any(difflib.SequenceMatcher(None, c, kf).ratio() >= thr for c in cands):
            found.append(k)
    return found


def load_rgb(path) -> Image.Image:
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def build_samples(cfg, select: str = "random") -> list[dict]:
    samples = []
    for path in sorted((REPO_ROOT / cfg["catalog_crops_dir"]).glob("*/*.*")):
        if path.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
            samples.append({"id": f"catalog/{path.parent.name}/{path.name}", "set": "catalog", "label": path.parent.name, "store": "-", "path": str(path)})
    s1 = yaml.safe_load((REPO_ROOT / "configs/stage1.yaml").read_text())
    names = {r["photo_id"]: r["file_name"] for r in csv.DictReader((REPO_ROOT / s1["id_maps_dir"] / "photo_ids.csv").open())}
    mapping = {r["crop_id"]: r for r in csv.DictReader((REPO_ROOT / cfg["val_mapping"]).open())}
    labels = list(csv.DictReader((REPO_ROOT / cfg["val_labels"]).open()))
    rng = random.Random(cfg["seed"])
    by_label = defaultdict(list)
    for r in labels:
        by_label[r["label"]].append(r)
    height = lambda r: int(mapping[r["crop_id"]]["crop_y2"]) - int(mapping[r["crop_id"]]["crop_y1"])  # noqa: E731
    for label, quota in cfg["val_quota"].items():
        pool = sorted(by_label[label], key=lambda r: r["crop_id"])
        rng.shuffle(pool)
        if select == "largest":   # the tallest crops of each label: does reading improve when the text has more pixels?
            pool = sorted(pool, key=height)
        by_store = defaultdict(list)
        for r in pool:
            by_store[r["store_code"]].append(r)
        if select == "largest":
            picked = pool[::-1][:quota]
        picked = picked if select == "largest" else []
        while select != "largest" and len(picked) < min(quota, len(pool)):     # round robin over stores
            for store in sorted(by_store):
                if by_store[store] and len(picked) < quota:
                    picked.append(by_store[store].pop())
        for r in picked:
            m = mapping[r["crop_id"]]
            samples.append({"id": f"val/{r['crop_id']}", "set": "val", "label": label, "store": r["store_code"],
                            "photo": str(REPO_ROOT / s1["raw_dir"] / names[m["photo_id"]]),
                            "region": [int(m[k]) for k in ("crop_x1", "crop_y1", "crop_x2", "crop_y2")]})
    return samples


def get_image(sample) -> Image.Image:
    if sample["set"] == "catalog":
        return load_rgb(sample["path"])
    return load_rgb(sample["photo"]).crop(tuple(sample["region"]))


def make_reader(args, cfg):
    if args.engine == "easyocr":
        import easyocr
        import torch
        gpu = args.device == "mps"
        model_dir = REPO_ROOT / cfg["model_dir"]
        model_dir.mkdir(parents=True, exist_ok=True)
        reader = easyocr.Reader(cfg["easyocr"]["languages"], gpu=gpu, model_storage_directory=str(model_dir), verbose=False)
        device = getattr(reader, "device", "?")
        print(f"easyocr {easyocr.__version__}: requested {args.device}, reader.device = {device}; mps available: {torch.backends.mps.is_available()}")

        def read(img: Image.Image) -> str:
            return " ".join(t for _, t, _ in reader.readtext(np.asarray(img), detail=1))
        return read, str(device)
    import pytesseract
    print("tesseract", pytesseract.get_tesseract_version())
    tc = cfg["tesseract"]

    def read(img: Image.Image) -> str:
        if img.height < tc["upscale_to_min_height_px"]:
            scale = tc["upscale_to_min_height_px"] / img.height
            img = img.resize((round(img.width * scale), tc["upscale_to_min_height_px"]), Image.LANCZOS)
        return " ".join(pytesseract.image_to_string(ImageOps.grayscale(img), config=tc["config"]).split())
    return read, "cpu"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ocr_trial.yaml")
    parser.add_argument("--engine", choices=["easyocr", "tesseract"], required=True)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu", help="easyocr only")
    parser.add_argument("--min-height", type=int, default=0, help="upscale crops shorter than this (px) before reading, for both engines")
    parser.add_argument("--tag", default="", help="suffix for the output names")
    parser.add_argument("--select", choices=["random", "largest"], default="random", help="which labeled val crops: a stratified random mix, or the tallest per label")
    args = parser.parse_args()
    cfg = yaml.safe_load((REPO_ROOT / args.config).read_text())
    if "soda-poc" in sys.prefix:
        raise SystemExit("run this in the throwaway environment, not in soda-poc")
    out_dir = REPO_ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = build_samples(cfg, args.select)
    read, device = make_reader(args, cfg)
    tag = (f"{args.engine}_{device}" if args.engine == "easyocr" else args.engine) + (f"_{args.tag}" if args.tag else "")
    results, times = [], []
    for k, s in enumerate(samples):
        img = get_image(s)
        if args.min_height and img.height < args.min_height:
            img = img.resize((round(img.width * args.min_height / img.height), args.min_height), Image.LANCZOS)
        t0 = time.perf_counter()
        text = read(img)
        dt = time.perf_counter() - t0
        if k >= 2:
            times.append(dt)   # the first calls include one-time setup
        found = keywords_found(text, cfg["keywords"], cfg["fuzzy_min_ratio"])
        found_word = words_found(text, cfg["keywords"], cfg["fuzzy_min_ratio"])
        expected = cfg["expect"][s["label"]]
        results.append({"id": s["id"], "set": s["set"], "label": s["label"], "store": s["store"], "size": list(img.size), "text": text,
                        "found": found, "found_word": found_word, "expected": expected, "hit": [e for e in expected if e in found],
                        "hit_word": [e for e in expected if e in found_word],
                        "wrong": [f for f in found if f not in expected and not (f == "COCA-COLA" and s["label"].startswith("coca"))],
                        "wrong_word": [f for f in found_word if f not in expected and not (f == "COCA-COLA" and s["label"].startswith("coca"))],
                        "seconds": round(dt, 3)})
    (out_dir / f"results_{tag}.json").write_text(json.dumps(results, indent=1))

    print(f"\n== {tag}: {len(results)} crops, median {statistics.median(times):.2f} s per crop (mean {statistics.mean(times):.2f} s, after 2 warm-ups)")
    for subset in ("catalog", "val"):
        rows = [r for r in results if r["set"] == subset]
        print(f"-- {subset} crops ({len(rows)}): any text read in {sum(1 for r in rows if r['text'].strip())}, any keyword read in {sum(1 for r in rows if r['found'])}")
        per = defaultdict(lambda: Counter())
        for r in rows:
            c = per[r["label"]]
            c["n"] += 1
            c["exp"] += len(r["expected"])
            c["hit"] += len(r["hit"])
            c["hitw"] += len(r["hit_word"])
            c["wrong_reads"] += len(r["wrong"])
            c["wrongw"] += len(r["wrong_word"])
            c["any_keyword"] += bool(r["found"])
        print("   (lenient = keyword anywhere in the text, fuzzy; strict = whole words only)")
        print("   product                    crops  expected-read lenient/strict  wrong-reads lenient/strict  crops-with-any-keyword")
        for label, c in per.items():
            exp = f"{c['hit']} / {c['hitw']} of {c['exp']}" if c["exp"] else "-"
            print(f"   {label:26} {c['n']:5}  {exp:>29}  {c['wrong_reads']:>12} / {c['wrongw']:<12}  {c['any_keyword']:>22}")
    # contact sheets with the text read under each crop
    cell, cols = cfg["contact_sheet"]["cell_px"], cfg["contact_sheet"]["columns"]
    font = ImageFont.load_default(size=11)
    for subset in ("catalog", "val"):
        rows = [r for r in results if r["set"] == subset]
        sample_of = {s["id"]: s for s in samples}
        n_rows = -(-len(rows) // cols)
        sheet = Image.new("RGB", (cols * cell, n_rows * (cell + 62)), "white")
        d = ImageDraw.Draw(sheet)
        for i, r in enumerate(rows):
            x, y = (i % cols) * cell, (i // cols) * (cell + 62)
            thumb = get_image(sample_of[r["id"]])
            thumb.thumbnail((cell - 8, cell - 8))
            sheet.paste(thumb, (x + (cell - thumb.width) // 2, y + 2))
            lines = [f"{r['label']} {r['id'].split('/')[-1][:18]}", f"read: {r['text'][:60] or '(nothing)'}", f"found: {', '.join(r['found']) or '-'}"]
            for j, line in enumerate(lines):
                d.text((x + 4, y + cell - 4 + j * 14), line[:38 if j else 34], fill=(0, 0, 0) if j != 2 else (0, 110, 0), font=font)
        sheet.save(out_dir / f"contact_{subset}_{tag}.jpg", quality=88)
    print(f"results and contact sheets in {out_dir.relative_to(REPO_ROOT)}/ (gitignored)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
