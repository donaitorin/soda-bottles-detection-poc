"""Helpers of the demo tool (scripts/demo_identify.py): refusing test photos, mapping decisions to colours and text, drawing the result.

The demo shows what the CURRENT system says about one photo. Its thresholds are demo thresholds, not an evaluation result.
Raw photo file names contain store and uploader names: they are never printed or written; a photo that is in the dataset is
named by its anonymous id.
"""
import csv
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.embeddings.answers import CANDIDATES
from src.embeddings.identify import AUTO, REVIEW, UNKNOWN


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def content_index(raw_dir: Path, names: dict, cache: Path) -> dict:
    """{sha256 of the file bytes: photo id} for every raw photo. Cached in a gitignored csv, rebuilt when the number of photos changes."""
    if cache.exists():
        rows = list(csv.DictReader(cache.open(newline="")))
        if len(rows) == len(names):
            return {r["sha256"]: r["photo_id"] for r in rows}
    index = {file_sha256(raw_dir / file_name): pid for pid, file_name in names.items()}
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["photo_id", "sha256"])
        w.writerows((pid, sha) for sha, pid in index.items())
    return index


def check_not_test(input_hash: str, index: dict, test_ids: set[str]) -> str | None:
    """The anonymous photo id if the file is a photo of the dataset (refuses if it is a TEST photo), else None.
    Raises SystemExit with a message that holds only the anonymous id, never a file name or the input path."""
    pid = index.get(input_hash)
    if pid is not None and pid in test_ids:
        raise SystemExit(f"Refused: this file is the test-split photo {pid}. The test split is not used before Stage 8; use a train or val photo, or a photo that is not in the dataset.")
    return pid


def decision_colour(decision: str, colours: dict) -> tuple:
    return tuple(colours[decision])


def short_name(product: str, names: dict) -> str:
    return names.get(product, product)


def ocr_line(ocr: dict | None) -> list[str]:
    """One panel line for variant B: the keywords read (a trailing ~ marks a fuzzy match) and the rules that fired. Nothing when nothing was read."""
    if not ocr or not (ocr["words"] or ocr["rules"]):
        return []
    words = " ".join(w + ("~" if k == "fuzzy" else "") for w, k in ocr["words"].items())
    return [f"   OCR: {words or '-'}" + (f" | {', '.join(ocr['rules'])}" if ocr["rules"] else "")]


def panel_entry(n: int, decision: str, top3: list[tuple[str, float]], candidates: dict | None = None, names: dict | None = None, ocr: dict | None = None) -> list[str]:
    """Text lines for one box: automatic shows the product and its similarity, candidates the brand and its variants (or a short list)
    with similarities, review the top 3 products with their similarities, unknown the top-1 product and similarity that was rejected."""
    names = names or {}
    if decision == AUTO:
        return [f"#{n}  {top3[0][0]}  {top3[0][1]:.2f}"] + ocr_line(ocr)
    if decision == CANDIDATES:
        head = f"#{n}  {names.get(candidates['brand'], candidates['brand'])}: which one?" if candidates["kind"] == "brand" else f"#{n}  one of:"
        return [head] + [f"   {short_name(p, names)}  {s:.2f}" for p, s in candidates["items"]] + ocr_line(ocr)
    if decision == REVIEW:
        return [f"#{n}  review"] + [f"   {i}. {p}  {s:.2f}" for i, (p, s) in enumerate(top3, 1)] + ocr_line(ocr)
    return [f"#{n}  unknown", f"   rejected: {top3[0][0]}  {top3[0][1]:.2f}"] + ocr_line(ocr)


def _font(px: int):
    return ImageFont.load_default(size=px)


def draw_result(image: Image.Image, boxes: np.ndarray, decisions: list[str], top3s: list[list[tuple[str, float]]], colours: dict,
                panel_cfg: dict, title: str, candidates: list | None = None, names: dict | None = None, ocr: list | None = None) -> Image.Image:
    """The photo with every box numbered and coloured by decision (green automatic, yellow review, grey unknown), and a side panel
    (legend first, then one entry per number). Large type so it reads on a phone; a long panel is split into columns."""
    px, col_w, max_h = panel_cfg["font_px"], panel_cfg["column_width_px"], panel_cfg["max_photo_height_px"]
    scale = min(1.0, max_h / image.height, 1800 / image.width)
    photo = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.LANCZOS)
    d = ImageDraw.Draw(photo)
    font, line_w = _font(max(14, px * 2 // 3)), max(2, photo.height // 450)
    for n, (box, dec) in enumerate(zip(boxes, decisions), 1):
        c = decision_colour(dec, colours)
        x1, y1, x2, y2 = (float(v) * scale for v in box)
        d.rectangle([x1, y1, x2, y2], outline=c, width=line_w)
        label = str(n)
        tw = d.textlength(label, font=font)
        d.rectangle([x1, y1, x1 + tw + 6, y1 + font.size + 4], fill=c)
        d.text((x1 + 3, y1 + 1), label, fill=(0, 0, 0), font=font)

    pfont, line_h = _font(px), int(px * 1.25)
    legend = [(AUTO, "automatic (answer given)")] + ([(CANDIDATES, "candidates (brand or short list)")] if CANDIDATES in colours else []) + [
        (REVIEW, "review (a person decides)"), (UNKNOWN, "unknown (rejected)")]
    candidates = candidates or [None] * len(decisions)
    ocr = ocr or [None] * len(decisions)
    entries = [panel_entry(n, dec, t3, c, names, o) for n, (dec, t3, c, o) in enumerate(zip(decisions, top3s, candidates, ocr), 1)]
    head_h = line_h * (2 + len(legend)) + 10
    heights = [len(e) * line_h + 8 for e in entries]
    max_cols = panel_cfg.get("max_columns", 3)
    total_h = sum(heights)
    # Column height: at least the photo height, and tall enough that the panel needs at most `max_cols` columns, so the picture stays
    # closer to a portrait shape (easier to read on a phone) instead of a very wide strip.
    per_col_h = max(photo.height, head_h + (total_h + max_cols - 1) // max_cols + max(heights, default=0) + 20)
    columns, used = [[]], head_h
    for i, h in enumerate(heights):
        if used + h > per_col_h and columns[-1]:
            columns.append([]); used = 0
        columns[-1].append(i); used += h
    canvas = Image.new("RGB", (photo.width + col_w * len(columns), max(photo.height, per_col_h)), (255, 255, 255))
    canvas.paste(photo, (0, 0))
    cd = ImageDraw.Draw(canvas)
    x0 = photo.width + 12
    cd.text((x0, 6), title, fill=(0, 0, 0), font=pfont)
    y = 6 + line_h
    cd.text((x0, y), f"{len(boxes)} boxes | demo thresholds, not an evaluation", fill=(60, 60, 60), font=_font(max(12, px * 2 // 3)))
    y += line_h
    for dec, text in legend:
        cd.rectangle([x0, y + 4, x0 + px, y + px], fill=decision_colour(dec, colours))
        cd.text((x0 + px + 10, y), text, fill=(0, 0, 0), font=pfont)
        y += line_h
    for ci, idxs in enumerate(columns):
        x = photo.width + ci * col_w + 12
        y = head_h if ci == 0 else 8
        for i in idxs:
            c = decision_colour(decisions[i], colours)
            for j, line in enumerate(entries[i]):
                cd.text((x, y), line, fill=tuple(int(v * 0.75) for v in c) if j == 0 else (50, 50, 50), font=pfont)
                y += line_h
            y += 8
    return canvas


def build_json(image_name: str, size: tuple[int, int], boxes: np.ndarray, scores: np.ndarray, decisions: list[str], top3s: list, meta: dict,
               candidates: list | None = None, ocr: list | None = None) -> dict:
    candidates = candidates or [None] * len(decisions)
    ocr = ocr or [None] * len(decisions)
    out = []
    for n, (b, s, dec, t3, c, o) in enumerate(zip(boxes, scores, decisions, top3s, candidates, ocr), 1):
        box = {"id": n, "box_xyxy": [round(float(v), 1) for v in b], "detector_score": round(float(s), 4), "decision": dec,
               "top3": [{"product": p, "similarity": round(float(v), 4)} for p, v in t3]}
        if c is not None:
            box["candidates"] = {"kind": c["kind"], "brand": c["brand"], "items": [{"product": p, "similarity": round(float(v), 4)} for p, v in c["items"]]}
        if o is not None:                                 # variant B: what the OCR read on this box and which rules fired
            box["ocr"] = {"read": o["read"], "text": o["text"], "words": o["words"], "rules": o["rules"]}
        out.append(box)
    return {"image": image_name, "image_size_wh": list(size), **meta, "boxes": out}
