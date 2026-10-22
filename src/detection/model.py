"""Faster R-CNN for the single generic class `bottle` (background + bottle = 2 classes)."""
import torch
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_model(cfg: dict, repo_root, pretrained: bool = True) -> torch.nn.Module:
    """COCO-pretrained `fasterrcnn_resnet50_fpn_v2` with a new 2-class box predictor.

    Dense shelves hold up to ~125 bottles, so the detection cap and the RPN proposal counts are raised from
    torchvision's defaults (100 / 1000) in the same proportion. `min_size` / `max_size` set the size the model
    resizes the photo to internally; boxes come back in ORIGINAL pixels either way.
    """
    torch.hub.set_dir(str(repo_root / cfg["weights_cache"]))
    model = fasterrcnn_resnet50_fpn_v2(
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None,
        weights_backbone=None,
        box_score_thresh=cfg["box_score_thresh"],
        box_detections_per_img=cfg["box_detections_per_img"],
        rpn_pre_nms_top_n_train=cfg["rpn_pre_nms_top_n_train"], rpn_post_nms_top_n_train=cfg["rpn_post_nms_top_n_train"],
        rpn_pre_nms_top_n_test=cfg["rpn_pre_nms_top_n_test"], rpn_post_nms_top_n_test=cfg["rpn_post_nms_top_n_test"],
        min_size=cfg["min_size"], max_size=cfg["max_size"],
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)  # 0 = background, 1 = bottle
    return model
