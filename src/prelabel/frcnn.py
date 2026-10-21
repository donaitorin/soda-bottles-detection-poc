"""COCO Faster R-CNN as a pre-labeler: keeps only the `bottle` class."""
import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
from torchvision.transforms.v2.functional import pil_to_tensor, to_dtype


def load_model(cfg: dict, device: torch.device, repo_root) -> tuple[torch.nn.Module, int]:
    """Return (model, label id of the target class). The id is read from the weights metadata."""
    torch.hub.set_dir(str(repo_root / cfg["weights_cache"]))
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn_v2(
        weights=weights,
        box_score_thresh=cfg["box_score_thresh"],
        box_detections_per_img=cfg["box_detections_per_img"],
        rpn_pre_nms_top_n_test=cfg["rpn_pre_nms_top_n_test"],
        rpn_post_nms_top_n_test=cfg["rpn_post_nms_top_n_test"],
    ).eval().to(device)
    return model, weights.meta["categories"].index(cfg["target_class"])


@torch.inference_mode()
def detect(model, target_id: int, image: Image.Image, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Boxes as float xyxy in ORIGINAL image pixels, and their scores, for the target class only."""
    tensor = to_dtype(pil_to_tensor(image), torch.float32, scale=True).to(device)  # uint8 -> float [0, 1]
    out = model([tensor])[0]
    keep = out["labels"] == target_id
    return out["boxes"][keep].cpu().numpy().astype(np.float64), out["scores"][keep].cpu().numpy().astype(np.float64)
