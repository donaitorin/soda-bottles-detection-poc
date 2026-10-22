"""Detection dataset for the generic `bottle` class, read from a COCO file written by scripts/export_coco.py.

Formats: the COCO file stores bbox = [x, y, width, height] in pixels; torchvision detection models want xyxy
pixels. The conversion is `coco_bbox_to_xyxy` (tested). Label 0 is the background, so `bottle` is 1.
The labels may be Grounding DINO pseudo-labels (see label_source): a model trained on them imitates them,
errors included, and any metric against them is agreement, not accuracy.
"""
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms.v2.functional import pil_to_tensor

from src.data.images import load_image

BOTTLE = 1


def coco_bbox_to_xyxy(bbox) -> list[float]:
    x, y, w, h = (float(v) for v in bbox)
    return [x, y, x + w, y + h]


def build_transforms(train: bool, cfg: dict | None = None) -> v2.Compose:
    """Training augmentations move the boxes together with the image (tv_tensors.BoundingBoxes).
    No large rotations: shelves are photographed upright. The affine scale zooms in or out a little; boxes cut
    by the border are clamped and tiny leftovers are removed by SanitizeBoundingBoxes."""
    steps = []
    if train and cfg:
        steps += [v2.RandomHorizontalFlip(cfg["hflip_p"]),
                  v2.ColorJitter(brightness=cfg["brightness"], contrast=cfg["contrast"], saturation=cfg["saturation"], hue=cfg["hue"]),
                  v2.RandomAffine(degrees=0, scale=tuple(cfg["scale"]), fill=114)]
    steps += [v2.ToDtype(torch.float32, scale=True), v2.SanitizeBoundingBoxes(min_size=cfg["sanitize_min_px"] if cfg else 1.0)]
    return v2.Compose(steps)


class BottleDataset(Dataset):
    """Items: (image float tensor [0, 1] CxHxW, target {boxes (n, 4) xyxy pixels, labels (n,) int64 = 1, photo_id, image_id})."""

    def __init__(self, coco_path: str | Path, photo_paths: dict, transforms=None, photo_ids: list[str] | None = None):
        coco = json.loads(Path(coco_path).read_text())
        self.images = [im for im in coco["images"] if photo_ids is None or im["photo_id"] in set(photo_ids)]
        by_image: dict[int, list] = {im["id"]: [] for im in self.images}
        for a in coco["annotations"]:
            if a["image_id"] in by_image:
                by_image[a["image_id"]].append(coco_bbox_to_xyxy(a["bbox"]))
        self.boxes = by_image
        self.photo_paths = photo_paths
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        im = self.images[index]
        image = load_image(self.photo_paths[im["photo_id"]])  # EXIF-safe, RGB
        assert image.size == (im["width"], im["height"]), f"{im['photo_id']}: photo size differs from the COCO file"
        boxes = torch.tensor(self.boxes[im["id"]], dtype=torch.float32).reshape(-1, 4)
        target = {"boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=(im["height"], im["width"])),
                  "labels": torch.full((len(boxes),), BOTTLE, dtype=torch.int64)}
        tensor = tv_tensors.Image(pil_to_tensor(image))
        if self.transforms is not None:
            tensor, target = self.transforms(tensor, target)
        target["boxes"] = target["boxes"].as_subclass(torch.Tensor)  # plain tensor for the model
        target["photo_id"], target["image_id"] = im["photo_id"], im["id"]
        return tensor, target


def collate(batch):
    return tuple(zip(*batch))
