"""Contact sheets of the val crops the adopted classifier gets WRONG, grouped by true label -> predicted label.

The model is the one saved by scripts/fit_classifier.py (trained on train shelf entries only; no val label was used to train it). A crop is
wrong when its most probable class (6 products + unknown) is not its owner label; only crops labeled with a product or `unknown` are checked
(`not_a_bottle` and `cannot_tell` have no right answer). One sheet set per TRUE label; each cell shows the crop, the predicted class with its
probability and the anonymous crop id. Output (ignored): data/review/stage6d_errors/<true label>_<NN>.jpg. Prints the true -> predicted counts.

Usage: python scripts/make_error_sheets.py
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.paths import check_output_dir  # noqa: E402
from src.embeddings.classifier import LinearClassifier  # noqa: E402
from src.embeddings.eval_data import load_val_queries  # noqa: E402
from src.embeddings.sheets import contact_sheet, paginate  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage6.yaml")
    parser.add_argument("--per-sheet", type=int, default=20)
    args = parser.parse_args()
    cfg = yaml.safe_load((REPO_ROOT / args.config).read_text())
    s1 = yaml.safe_load((REPO_ROOT / "configs/stage1.yaml").read_text())
    products, cc = cfg["products"], cfg["classifier"]
    val = load_val_queries(cfg, REPO_ROOT)
    model = LinearClassifier.load(REPO_ROOT / cc["model_dir"] / f"{cc['name']}.npz")
    proba = model.predict_proba(val["vec"])
    pred = np.array(model.classes, dtype=object)[proba.argmax(axis=1)]
    truth = val["truth"]
    out_dir = REPO_ROOT / "data/review/stage6d_errors"
    check_output_dir(out_dir, [REPO_ROOT / d for d in s1["protected_dirs"]])
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()
    labeled = np.isin(truth, products + ["unknown"])
    wrong = labeled & (pred != truth)
    print(f"{int(wrong.sum())} wrong of {int(labeled.sum())} labeled val crops ({int((wrong & np.isin(truth, products)).sum())} of {int(np.isin(truth, products).sum())} known crops, {int((wrong & (truth == 'unknown')).sum())} of {int((truth == 'unknown').sum())} unknown crops)")
    for label in products + ["unknown"]:
        idx = [i for i in np.where(wrong & (truth == label))[0]]
        if not idx:
            print(f"{label}: no errors")
            continue
        idx.sort(key=lambda i: (str(pred[i]), -proba[i].max()))
        groups = Counter(pred[i] for i in idx)
        print(f"{label}: " + ", ".join(f"-> {p} {n}" for p, n in groups.most_common()))
        for n, page in enumerate(paginate(idx, args.per_sheet), 1):
            imgs = [Image.open(REPO_ROOT / cfg["crops"]["images_dir"] / f"{val['crop_ids'][i]}.jpg") for i in page]
            caps = [f"{pred[i]} {proba[i].max():.2f}  {val['crop_ids'][i]}" for i in page]
            title = f"TRUE {label} -> predicted: " + ", ".join(f"{p} {c}" for p, c in groups.most_common()) + "  (owner label vs classifier; dev. sample)"
            contact_sheet(imgs, caps, title, cols=4, cell=(360, 460), font_px=17).save(out_dir / f"{label}_{n:02d}.jpg", quality=88)
    print(f"sheets in {out_dir.relative_to(REPO_ROOT)}/ (gitignored)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
