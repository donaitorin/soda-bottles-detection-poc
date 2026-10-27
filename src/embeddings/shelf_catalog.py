"""Catalog variants for the second step: the 30 studio originals plus owner-labeled shelf crops from TRAIN photos.

Four catalogs (entries are `vec`, `product` and `photo`; a shelf entry's `photo` is its crop id, so each shelf crop counts as its own
source photo in the k-NN vote):
  originals                   the 30 studio photos (what the database holds today)
  originals+shelf             + shelf crops labeled with a product
  originals+shelf+negatives   + shelf crops labeled `unknown`, as negative entries (product id `unknown`)
  shelf                       shelf crops with a product only (no originals, no negatives): what the studio photos still contribute
Class balance: shelf labels are very uneven (zero and classic dominate). A product's score is its BEST entry, which grows with
the number of entries (more tries, higher maximum), so the shelf entries of every product are capped (a seeded random subset of
`cap_per_product`; unknown negatives have their own cap) and the k-NN vote can be weighted by 1 / (entries of the product)
(`identify.features_from_sims(balance=True)`). Rare products keep all their entries.
"""
import numpy as np

MODES = ("originals", "originals+shelf", "originals+shelf+negatives", "shelf")


def capped_subset(labels: np.ndarray, ids: np.ndarray, cap: int, cap_unknown: int, seed: int) -> np.ndarray:
    """Indices to keep: all entries of a label with at most the cap, else a seeded random subset of exactly the cap (ids sorted first,
    so the result does not depend on the input order)."""
    rng = np.random.default_rng(seed)
    keep = []
    for label in sorted(set(labels)):
        idx = np.where(labels == label)[0]
        idx = idx[np.argsort(ids[idx], kind="stable")]
        limit = cap_unknown if label == "unknown" else cap
        if len(idx) > limit:
            idx = np.sort(rng.choice(idx, size=limit, replace=False))
        keep += list(idx)
    return np.array(sorted(keep), int)


def assemble(orig: dict, shelf: dict, mode: str, cap: int, cap_unknown: int, seed: int) -> dict:
    """The catalog arrays {vec, product, photo} for one mode. `shelf` has vec, label (product id or `unknown`) and photo (crop id)."""
    assert mode in MODES, mode
    parts = []
    if mode != "shelf":
        parts.append((orig["vec"], orig["product"], orig["photo"]))
    if mode != "originals":
        sel_label = shelf["label"]
        usable = np.array([l != "unknown" or mode == "originals+shelf+negatives" for l in sel_label])
        idx = np.where(usable)[0]
        kept = idx[capped_subset(sel_label[idx], shelf["photo"][idx], cap, cap_unknown, seed)]
        parts.append((shelf["vec"][kept], sel_label[kept], shelf["photo"][kept]))
    return {"vec": np.concatenate([p[0] for p in parts]), "product": np.concatenate([p[1] for p in parts]),
            "photo": np.concatenate([p[2] for p in parts])}


def shelf_db_entries(shelf: dict, brand_of: dict, cap: int, cap_unknown: int, seed: int) -> tuple[list[dict], np.ndarray]:
    """Database rows and vectors for the capped shelf entries of catalog (b): products and `unknown` negatives (the same subset
    `assemble` picks). A shelf entry's `source_photo` is its anonymous crop id; its source is the owner's own labeling of a train crop."""
    keep = capped_subset(shelf["label"], shelf["photo"], cap, cap_unknown, seed)
    entries = [{"product_id": str(shelf["label"][i]), "brand": brand_of.get(shelf["label"][i], "unknown"), "size": None, "material": None,
                "source_photo": str(shelf["photo"][i]), "source_url": "train shelf crop (owner label)", "license": "own photos (project data)",
                "packaging_checked": "n/a", "is_augmented": False, "aug_index": None, "aug_params": None, "crop_method": "dino_box_margin10"} for i in keep]
    return entries, shelf["vec"][keep]
