"""Grounding DINO (zero-shot, text-prompted detector) as a pre-labeler."""
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


def load_model(cfg: dict, device: torch.device, repo_root):
    """Load the pinned revision; weights are cached in the gitignored models/hf_cache."""
    kwargs = {"revision": cfg["revision"], "cache_dir": str(repo_root / cfg["cache_dir"])}
    processor = AutoProcessor.from_pretrained(cfg["model_id"], **kwargs)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(cfg["model_id"], **kwargs).eval().to(device)
    return model, processor


@torch.inference_mode()
def detect(model, processor, image: Image.Image, prompt: str, device: torch.device,
           box_threshold: float, text_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Boxes as float xyxy in ORIGINAL image pixels, and scores.

    `target_sizes` is (height, width) of the original image: the post-processing uses it to turn the
    model's normalized boxes back into pixels of the original photo. Passing (width, height) by mistake
    gives shifted boxes on every non-square photo, which is the classic bug here.
    """
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    outputs = model(**inputs)
    result = processor.post_process_grounded_object_detection(
        outputs, inputs["input_ids"], threshold=box_threshold, text_threshold=text_threshold,
        target_sizes=[(image.height, image.width)])[0]
    return result["boxes"].float().cpu().numpy().astype(np.float64), result["scores"].float().cpu().numpy().astype(np.float64)
