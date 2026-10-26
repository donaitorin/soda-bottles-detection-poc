"""Augmentation preview: one contact sheet per product, each original catalog crop followed by its generated variants.

The owner reviews these sheets BEFORE anything goes into the database. The variants are generated with the same function and the
same per-photo seeds the real catalog build will use, so what is previewed is exactly what would be built. Nothing is stored in
a database here. Outputs (gitignored): data/review/catalog_augment/<product id>.jpg.

Usage: python scripts/preview_catalog_augment.py
"""
import argparse
import sys
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.images import load_image  # noqa: E402
from src.data.paths import check_output_dir  # noqa: E402
from src.embeddings.augment import augment, describe, seed_for  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage6.yaml")
    parser.add_argument("--cell-width", type=int, default=150)
    parser.add_argument("--cell-height", type=int, default=330)
    args = parser.parse_args()
    cfg = yaml.safe_load((REPO_ROOT / args.config).read_text())
    s1 = yaml.safe_load((REPO_ROOT / "configs/stage1.yaml").read_text())
    ac = cfg["catalog"]["augment"]
    crops_dir = REPO_ROOT / cfg["catalog"]["crop"]["crops_dir"]
    out_dir = REPO_ROOT / ac["preview_dir"]
    check_output_dir(out_dir, [REPO_ROOT / d for d in s1["protected_dirs"]])
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()
    cw, ch, cap = args.cell_width, args.cell_height, 44
    font = ImageFont.load_default(size=10)
    total = 0
    for product in cfg["products"]:
        files = sorted(p for p in (crops_dir / product).glob("*.*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        cols = 1 + ac["per_photo"]
        sheet = Image.new("RGB", (cols * cw, len(files) * (ch + cap)), "white")
        d = ImageDraw.Draw(sheet)
        for r, path in enumerate(files):
            rel = f"{product}/{path.name}"
            original = load_image(path)
            cells = [("original", [path.name], original)] + [(f"aug_{k:02d}", describe(p), im) for k, (im, p) in
                                                             enumerate(augment(original, ac, seed_for(cfg["seed"], rel)), 1)]
            for c, (label, lines, im) in enumerate(cells):
                thumb = im.copy()
                thumb.thumbnail((cw - 6, ch - 6))
                x, y = c * cw, r * (ch + cap)
                sheet.paste(thumb, (x + (cw - thumb.width) // 2, y + (ch - thumb.height) // 2))
                d.rectangle([x, y, x + cw - 1, y + ch + cap - 1], outline=(215, 215, 215))
                d.text((x + 4, y + ch + 1), label, fill=(190, 0, 0) if label == "original" else (0, 0, 0), font=font)
                for j, line in enumerate(lines[:3]):
                    d.text((x + 4, y + ch + 12 + j * 10), line[:26], fill=(70, 70, 70), font=font)
            total += ac["per_photo"]
        sheet.save(out_dir / f"{product}.jpg", quality=90)
        print(f"{product}: {len(files)} originals x {ac['per_photo']} variants -> {(out_dir / (product + '.jpg')).relative_to(REPO_ROOT)}")
    print(f"{total} augmented variants previewed (nothing stored in any database). Review the sheets, then tell me what to change.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
