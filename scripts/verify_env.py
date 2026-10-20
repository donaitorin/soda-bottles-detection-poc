"""Stage 0 check: does PyTorch see the accelerator, and does a pretrained
Faster R-CNN run on it?

Usage: python scripts/verify_env.py --config configs/stage0_verify.yaml [--image PATH]
"""
import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # so `src` is importable when run as a script

from src.common.device import get_device  # noqa: E402  (sets the MPS fallback before torch loads)

import torch  # noqa: E402
import torchvision  # noqa: E402
import yaml  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402
from torchvision.models.detection import (  # noqa: E402
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.transforms.v2.functional import pil_to_tensor, to_dtype  # noqa: E402
from torchvision.utils import draw_bounding_boxes  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def find_image(path: Path) -> Path | None:
    if path.is_file():
        return path
    candidates = sorted(p for p in path.glob("*") if p.suffix.lower() in IMAGE_SUFFIXES
                        and not p.name.startswith("verify_output"))
    return candidates[0] if candidates else None


def sync(device: torch.device) -> None:
    # Accelerators run asynchronously; wait so the timer measures real work.
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def timed_inference(model, image: torch.Tensor, device: torch.device):
    model.to(device)
    x = image.to(device)
    with torch.inference_mode():
        model([x])  # warm-up: the first call on MPS/CUDA includes one-time setup cost
        sync(device)
        start = time.perf_counter()
        out = model([x])[0]
        sync(device)
    return out, time.perf_counter() - start


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage0_verify.yaml")
    parser.add_argument("--image", help="override the image in the config")
    args = parser.parse_args()
    cfg = yaml.safe_load((REPO_ROOT / args.config).read_text())
    torch.manual_seed(cfg["seed"])

    # 1. Device
    device = get_device()
    print(f"torch {torch.__version__} | torchvision {torchvision.__version__}")
    print(f"cuda available: {torch.cuda.is_available()} | "
          f"mps built: {torch.backends.mps.is_built()}, available: {torch.backends.mps.is_available()}")
    print(f"get_device() -> {device}")
    probe = (torch.ones(3, 3, device=device) @ torch.ones(3, 3, device=device)).sum().item()
    assert probe == 27.0, f"matmul on {device} gave {probe}"
    print(f"tensor op on {device}: OK")
    if device.type == "cpu":
        print("WARNING: no accelerator found. Full training / pre-labeling should go to Colab.")

    # 2. Model (COCO weights are cached inside the repo's ignored models/ folder)
    torch.hub.set_dir(str(REPO_ROOT / cfg["weights_cache"]))
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn_v2(weights=weights).eval()
    categories = weights.meta["categories"]
    target_id = categories.index(cfg["target_class"])  # index in the COCO label list (not hardcoded)
    print(f"model loaded; COCO '{cfg['target_class']}' has label id {target_id}")

    # 3. Image
    image_path = find_image(REPO_ROOT / (args.image or cfg["image"]))
    if image_path is None:
        print(f"WARNING: no image found in {cfg['image']}; using random noise as a smoke test "
              "(boxes will be meaningless).")
        pil = Image.fromarray(torch.randint(0, 255, (480, 640, 3), dtype=torch.uint8).numpy())
    else:
        # exif_transpose applies the camera's rotation flag. Without it, phone photos
        # load sideways and any box would be in the wrong coordinates.
        pil = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
        print(f"image: {image_path.relative_to(REPO_ROOT)} ({pil.width}x{pil.height})")
    # uint8 [0,255] -> float [0,1]; the model resizes and normalizes internally.
    image = to_dtype(pil_to_tensor(pil), torch.float32, scale=True)

    # 4. Inference on the chosen device
    try:
        out, seconds = timed_inference(model, image, device)
    except RuntimeError as err:
        print(f"FAILED on {device}: {err}")
        print("Report this before changing versions; Colab is the fallback.")
        return 1
    keep = out["scores"] >= cfg["score_threshold"]
    boxes, labels, scores = out["boxes"][keep].cpu(), out["labels"][keep].cpu(), out["scores"][keep].cpu()
    n_target = int((labels == target_id).sum())
    print(f"{device}: {seconds:.2f} s/image | {len(boxes)} detections >= {cfg['score_threshold']} "
          f"({n_target} '{cfg['target_class']}')")

    # 5. Annotated output. Boxes are xyxy in pixels, which is what draw_bounding_boxes expects.
    names = [f"{categories[int(l)]} {s:.2f}" for l, s in zip(labels, scores)]
    drawn = draw_bounding_boxes(to_dtype(image, torch.uint8, scale=True), boxes, names, width=3)
    out_path = REPO_ROOT / cfg["output"]
    Image.fromarray(drawn.permute(1, 2, 0).numpy()).save(out_path)
    print(f"annotated image: {out_path.relative_to(REPO_ROOT)}")

    # 6. CPU comparison
    if cfg["compare_cpu"] and device.type != "cpu":
        _, cpu_seconds = timed_inference(model, image, torch.device("cpu"))
        print(f"cpu: {cpu_seconds:.2f} s/image | speed-up of {device.type}: {cpu_seconds / seconds:.1f}x")
        if seconds >= cpu_seconds:
            print(f"WARNING: {device.type} is not faster than the CPU. Tell the user; consider Colab.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
