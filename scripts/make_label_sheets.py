"""Contact sheets of the owner-labeled TRAIN shelf crops, one set per label, to spot mislabels (classic vs returnable, zero vs classic...).

Only the crops that become catalog entries: labeled with a product or `unknown` (not `not_a_bottle` or `cannot_tell`). Each cell is
the crop (as shown in Label Studio) with its anonymous crop id; send corrections as a list of crop ids. Output (ignored):
data/review/train_shelf_labels/<label>_<NN>.jpg. Nothing is read from or written to the labels file.

Usage: python scripts/make_label_sheets.py
"""
import argparse
import csv
import sys
from pathlib import Path

import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.paths import check_output_dir  # noqa: E402
from src.embeddings.sheets import contact_sheet, paginate  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage6.yaml")
    parser.add_argument("--per-sheet", type=int, default=20)
    args = parser.parse_args()
    cfg = yaml.safe_load((REPO_ROOT / args.config).read_text())
    s1 = yaml.safe_load((REPO_ROOT / "configs/stage1.yaml").read_text())
    sr = cfg["shelf_round"]
    out_dir = REPO_ROOT / "data/review/train_shelf_labels"
    check_output_dir(out_dir, [REPO_ROOT / d for d in s1["protected_dirs"]])
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()
    with (REPO_ROOT / cfg["labels_export"]["output_dir"] / f"{sr['name']}_labels.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    images_dir = REPO_ROOT / cfg["crops"]["images_dir"]
    for label in cfg["products"] + ["unknown"]:
        ids = sorted(r["crop_id"] for r in rows if r["label"] == label)
        pages = paginate(ids, args.per_sheet)
        for n, page in enumerate(pages, 1):
            imgs = [Image.open(images_dir / f"{c}.jpg") for c in page]
            title = f"{label}  (sheet {n} of {len(pages)}, {len(ids)} crops)  |  labels are the owner's; send corrections as crop ids"
            contact_sheet(imgs, page, title, cols=5, cell=(300, 420)).save(out_dir / f"{label}_{n:02d}.jpg", quality=88)
        print(f"{label}: {len(ids)} crops, {len(pages)} sheet(s)")
    print(f"sheets in {out_dir.relative_to(REPO_ROOT)}/ (gitignored)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
