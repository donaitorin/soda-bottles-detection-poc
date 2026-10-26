"""Catalog augmentations: extra looks of the same product photo, each stored as its own catalog entry tagged `augmented`.

Why these and not the detector's: a product photo is a clean, frontal, evenly lit bottle; a shelf crop is tilted, seen from an
angle, darker or brighter and slightly blurred. So the augmentations imitate exactly that: a small rotation, a mild perspective,
a zoom, brightness, contrast, and a slight blur. Two things are deliberately NOT done:
  - no horizontal flip: it MIRRORS the label text (COCA-COLA, ZERO, RETORNABLE), which is the evidence that separates variants;
  - no hue or saturation change: colour is one of the main cues between products (Coca-Cola red, Light silver, Fanta orange,
    Sprite green).
Parameters are sampled with a seeded generator and returned with each image, so a preview and the real build are identical.
"""
import math
import zlib

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.v2 import functional as F

FORBIDDEN = ("horizontal_flip", "hue", "saturation")


def seed_for(base_seed: int, name: str) -> int:
    """A stable seed per catalog photo (not Python's hash(), which changes between runs)."""
    return (base_seed * 1_000_003 + zlib.crc32(name.encode())) % (2**32)


def sample_params(rng: np.random.Generator, cfg: dict, n: int) -> list[dict]:
    """`n` parameter sets. The perspective is four corner offsets as shares of the half-size (so it scales with the image)."""
    out = []
    for _ in range(n):
        out.append({
            "rotation_deg": float(rng.uniform(-cfg["rotation_deg"], cfg["rotation_deg"])),
            "scale": float(rng.uniform(*cfg["scale"])),
            "perspective": [[float(rng.uniform(-1, 1)) * cfg["perspective_distortion"] for _ in range(2)] for _ in range(4)],
            "brightness": float(1 + rng.uniform(-cfg["brightness"], cfg["brightness"])),
            "contrast": float(1 + rng.uniform(-cfg["contrast"], cfg["contrast"])),
            "blur_sigma": float(rng.uniform(*cfg["blur_sigma"])),
        })
    return out


def apply_params(image: Image.Image, p: dict, fill=None) -> Image.Image:
    """Brightness, contrast and a Gaussian blur first, then rotation and zoom about the centre and a perspective warp. Same output
    size. The geometric steps leave empty corners; they are filled with the background colour of the photo AFTER the lighting
    change (median of the four corners), so a darker variant gets a darker background and no light frame appears around it.
    `fill` forces a colour instead."""
    x = F.pil_to_tensor(image.convert("RGB"))
    h, w = x.shape[-2:]
    x = F.adjust_brightness(x, p["brightness"])
    x = F.adjust_contrast(x, p["contrast"])
    k = 2 * math.ceil(3 * p["blur_sigma"]) + 1
    x = F.gaussian_blur(x, [k, k], [p["blur_sigma"], p["blur_sigma"]])
    if fill is None:
        m = max(2, min(h, w) // 40)
        corners = torch.cat([x[:, :m, :m].reshape(3, -1), x[:, :m, -m:].reshape(3, -1), x[:, -m:, :m].reshape(3, -1), x[:, -m:, -m:].reshape(3, -1)], dim=1)
        fill = [int(v) for v in corners.float().median(dim=1).values]
    f = list(fill)
    x = F.affine(x, angle=p["rotation_deg"], translate=[0, 0], scale=p["scale"], shear=[0.0, 0.0], fill=f)
    start = [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]
    end = [[sx + dx * w / 2, sy + dy * h / 2] for (sx, sy), (dx, dy) in zip(start, p["perspective"])]
    x = F.perspective(x, start, end, fill=f)
    return F.to_pil_image(x)


def augment(image: Image.Image, cfg: dict, seed: int, n: int | None = None) -> list[tuple[Image.Image, dict]]:
    """`n` (default cfg per_photo) augmented copies of one crop with their parameters."""
    rng = np.random.default_rng(seed)
    return [(apply_params(image, p), p) for p in sample_params(rng, cfg, n or cfg["per_photo"])]


def describe(p: dict) -> list[str]:
    """Short caption lines for a contact sheet."""
    return [f"rot {p['rotation_deg']:+.1f} deg  scale {p['scale']:.2f}", f"bright {p['brightness']:.2f}  contrast {p['contrast']:.2f}",
            f"blur {p['blur_sigma']:.2f}  persp {max(abs(v) for c in p['perspective'] for v in c):.2f}"]
