import json

import numpy as np
import torch
from PIL import Image
from torchvision import tv_tensors

from src.detection.dataset import BottleDataset, build_transforms, coco_bbox_to_xyxy, collate
from src.detection.engine import agreement, lr_at, save_checkpoint, load_checkpoint

AUG = {"hflip_p": 1.0, "brightness": 0.0, "contrast": 0.0, "saturation": 0.0, "hue": 0.0, "scale": [1.0, 1.0], "sanitize_min_px": 1.0}


def _make(tmp_path, boxes_xywh, size=(200, 100)):
    """One synthetic photo (white image, a dark rectangle where the bottle is) and its COCO file."""
    w, h = size
    img = Image.new("RGB", size, (255, 255, 255))
    for x, y, bw, bh in boxes_xywh:
        img.paste((0, 0, 0), (int(x), int(y), int(x + bw), int(y + bh)))
    path = tmp_path / "photo.png"
    img.save(path)
    coco = {"images": [{"id": 1, "photo_id": "s01_aaaaaaaa", "file_name": "s01_aaaaaaaa.png", "width": w, "height": h}],
            "annotations": [{"id": i + 1, "image_id": 1, "category_id": 1, "bbox": list(b), "area": b[2] * b[3], "iscrowd": 0}
                            for i, b in enumerate(boxes_xywh)],
            "categories": [{"id": 1, "name": "bottle"}]}
    (tmp_path / "gt.json").write_text(json.dumps(coco))
    return tmp_path / "gt.json", {"s01_aaaaaaaa": path}, coco


def test_coco_bbox_becomes_xyxy():
    assert coco_bbox_to_xyxy([10, 20, 30, 40]) == [10, 20, 40, 60]


def test_dataset_item_has_xyxy_boxes_label_one_and_float_image(tmp_path):
    gt, paths, _ = _make(tmp_path, [(10, 20, 30, 40)])
    image, target = BottleDataset(gt, paths, build_transforms(False))[0]
    assert image.shape == (3, 100, 200) and image.dtype == torch.float32 and 0 <= float(image.min()) and float(image.max()) <= 1
    assert target["boxes"].tolist() == [[10, 20, 40, 60]] and target["labels"].tolist() == [1]
    assert not isinstance(target["boxes"], tv_tensors.BoundingBoxes)


def test_a_photo_without_boxes_gives_an_empty_target(tmp_path):
    gt, paths, _ = _make(tmp_path, [])
    _, target = BottleDataset(gt, paths, build_transforms(False))[0]
    assert target["boxes"].shape == (0, 4) and target["labels"].shape == (0,)


def test_horizontal_flip_moves_the_box_with_the_pixels(tmp_path):
    gt, paths, _ = _make(tmp_path, [(10, 20, 30, 40)])
    image, target = BottleDataset(gt, paths, build_transforms(True, AUG))[0]
    x1, y1, x2, y2 = (int(round(v)) for v in target["boxes"][0].tolist())
    assert (x1, x2) == (200 - 40, 200 - 10) and (y1, y2) == (20, 60)
    inside = image[:, y1 + 2:y2 - 2, x1 + 2:x2 - 2]
    outside = image[:, 0:10, 0:10]
    assert float(inside.max()) < 0.05 and float(outside.min()) > 0.95   # the dark pixels are where the box says


def test_collate_keeps_lists(tmp_path):
    gt, paths, _ = _make(tmp_path, [(10, 20, 30, 40)])
    ds = BottleDataset(gt, paths, build_transforms(False))
    images, targets = collate([ds[0], ds[0]])
    assert len(images) == 2 and len(targets) == 2


def test_lr_schedule_warms_up_then_decays_to_the_final_factor():
    kw = dict(total_steps=100, base_lr=0.01, warmup_steps=10, warmup_factor=0.1, final_factor=0.01)
    assert abs(lr_at(0, **kw) - 0.001) < 1e-12
    assert abs(lr_at(10, **kw) - 0.01) < 1e-12
    assert lr_at(5, **kw) < lr_at(10, **kw)
    assert lr_at(50, **kw) < lr_at(20, **kw)
    assert abs(lr_at(100, **kw) - 0.0001) < 1e-9


def test_agreement_is_perfect_for_identical_boxes(tmp_path):
    _, _, coco = _make(tmp_path, [(10, 20, 30, 40), (100, 20, 30, 40)])
    preds = {"s01_aaaaaaaa": (np.array([[10, 20, 40, 60], [100, 20, 130, 60]], float), np.array([0.9, 0.8]))}
    m = agreement(coco, preds, 0.015, 0.3, 300)
    assert m["AP"] == 1.0 and m["precision"] == 1.0 and m["recall"] == 1.0


def test_checkpoint_round_trip_restores_weights_and_epoch(tmp_path):
    model = torch.nn.Linear(3, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    save_checkpoint(tmp_path / "c.pt", model, opt, epoch=3, step=42)
    other = torch.nn.Linear(3, 2)
    state = load_checkpoint(tmp_path / "c.pt", other, torch.optim.SGD(other.parameters(), lr=0.1, momentum=0.9))
    assert state["epoch"] == 3 and state["step"] == 42
    assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), other.state_dict().values()))
    assert not list(tmp_path.glob("*.tmp"))
