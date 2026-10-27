"""Inputs of the identification measurements: labeled val crops, the 30 catalog originals and the labeled TRAIN shelf crops (all as arrays).

Val crops are the yardstick (owner labels, 6 stores); the TRAIN shelf crops are the extra catalog entries of the second step. They come
from different stores, so a shelf entry is never from the store of the val crop it is compared with.
"""
import csv
from pathlib import Path

import numpy as np


def _rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_val_queries(cfg: dict, root: Path) -> dict:
    """crop_ids, truth, stores and CLS vectors of the labeled val crops (from the cached embeddings of evaluate_identification.py)."""
    labels = _rows(root / cfg["labels_export"]["output_dir"] / "val_sample_labels.csv")
    z = np.load(root / cfg["crops"]["mapping_dir"] / cfg["crops"]["sample"]["name"] / "embeddings.npz")
    pos = {c: i for i, c in enumerate(z["crop_ids"])}
    ids = np.array([r["crop_id"] for r in labels])
    return {"crop_ids": ids, "truth": np.array([r["label"] for r in labels], dtype=object), "stores": np.array([r["store_code"] for r in labels]),
            "vec": z["cls"][[pos[c] for c in ids]]}


def load_originals(cfg: dict, root: Path) -> dict:
    """The 30 catalog originals: vec, product, photo."""
    cat = np.load(root / cfg["catalog_files"]["embeddings_npz"])
    orig = ~cat["is_augmented"]
    return {"vec": cat["cls"][orig], "product": cat["product"][orig], "photo": cat["source_photo"][orig]}


def load_shelf(cfg: dict, root: Path) -> dict:
    """Owner-labeled TRAIN shelf crops: vec, label (a product id or `unknown`), photo (= crop id), store. Crops labeled
    `not_a_bottle` or `cannot_tell` are left out (they are not examples of any product or of `unknown`)."""
    sr = cfg["shelf_round"]
    labels = _rows(root / cfg["labels_export"]["output_dir"] / f"{sr['name']}_labels.csv")
    keep = [r for r in labels if r["label"] in set(cfg["products"]) | {"unknown"}]
    z = np.load(root / sr["candidates_npz"])
    pos = {c: i for i, c in enumerate(z["crop_ids"])}
    ids = np.array([r["crop_id"] for r in keep])
    return {"vec": z["cls"][[pos[c] for c in ids]], "label": np.array([r["label"] for r in keep], dtype=object), "photo": ids,
            "store": np.array([r["store_code"] for r in keep])}
