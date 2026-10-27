"""SAM 2 masks for known boxes (Stage 6 item 4, experiment 4b): the mask of the bottle inside a Grounding DINO box.

SAM 2 (Apache 2.0, `facebook/sam2.1-hiera-small`, pinned revision, through `transformers`) gets the box as a prompt and returns one
mask per box (single-mask output). Used only to put the crop on a NEUTRAL background, so that neighbours and shelf do not reach the
embedding. Masks are in photo pixels.
"""
import numpy as np
import torch
from PIL import Image


def load_sam2(cfg: dict, device: torch.device, repo_root):
    from transformers import Sam2Model, Sam2Processor  # imported here: only experiment 4b needs the weights
    kw = {"revision": cfg["revision"], "cache_dir": str(repo_root / cfg["cache_dir"])}
    return Sam2Model.from_pretrained(cfg["model_id"], **kw).eval().to(device), Sam2Processor.from_pretrained(cfg["model_id"], **kw)


@torch.inference_mode()
def masks_for_boxes(model, processor, image: Image.Image, boxes: np.ndarray, device: torch.device, chunk: int = 32) -> np.ndarray:
    """(N, H, W) bool masks, one per xyxy box (photo pixels). The photo is encoded once and the boxes are prompted in chunks."""
    inputs = processor(images=image, input_boxes=[np.asarray(boxes, float).tolist()], return_tensors="pt").to(device)
    emb = model.get_image_embeddings(inputs["pixel_values"])
    out = []
    for i in range(0, len(boxes), chunk):
        part = inputs["input_boxes"][:, i:i + chunk]
        res = model(image_embeddings=emb, input_boxes=part, multimask_output=False)
        m = processor.post_process_masks(res.pred_masks.cpu(), inputs["original_sizes"].cpu(), binarize=True)[0]   # (n, 1, H, W)
        out.append(m[:, 0].numpy().astype(bool))
    return np.concatenate(out) if out else np.zeros((0, image.height, image.width), bool)


def neutral_background(crop: Image.Image, mask_crop: np.ndarray, fill=(255, 255, 255)) -> Image.Image:
    """The crop with everything outside the mask painted `fill` (the colour the embedding pads with, like the catalog backgrounds)."""
    arr = np.asarray(crop.convert("RGB")).copy()
    arr[~mask_crop] = fill
    return Image.fromarray(arr)
