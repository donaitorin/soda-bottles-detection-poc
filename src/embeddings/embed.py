"""DINOv2 embeddings of bottle crops: one L2-normalised 768-dimension vector per crop.

Cosine similarity between two such vectors is just their dot product. The same function embeds the catalog crops and the shelf
crops, so both go through exactly the same preprocessing.
"""
import numpy as np
import torch
from PIL import Image


def pad_to_square(image: Image.Image, fill=(255, 255, 255)) -> Image.Image:
    """Centre the crop on a square canvas. A bottle crop is tall and narrow; resizing it to a square would stretch it, and a
    centre crop would cut off the cap and the base, so it is padded instead."""
    side = max(image.size)
    canvas = Image.new("RGB", (side, side), tuple(fill))
    canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    return canvas


def preprocess(images: list[Image.Image], size: int, fill, mean, std) -> torch.Tensor:
    """(N, 3, size, size) float tensor: pad to a square, resize (bicubic), scale to [0, 1], normalise with the ImageNet statistics."""
    arrays = []
    for im in images:
        sq = pad_to_square(im.convert("RGB"), fill).resize((size, size), Image.BICUBIC)
        arrays.append(np.asarray(sq, dtype=np.float32) / 255.0)
    x = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2)
    return (x - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)


def content_patch_mask(sizes: list[tuple[int, int]], image_size: int, patch: int = 14) -> torch.Tensor:
    """(N, patches) bool: True for the patches whose centre lies on the CROP, False on the padding. `sizes` are the (width, height) of
    the crops before padding. The crop is centred on a square, so its extent on the resized square is known exactly."""
    n = image_size // patch
    centres = (torch.arange(n) + 0.5) * patch                                  # patch centres in resized-square pixels
    masks = []
    for w, h in sizes:
        side = max(w, h)
        half_w, half_h = image_size * w / side / 2, image_size * h / side / 2
        mid = image_size / 2
        col = (centres >= mid - half_w) & (centres <= mid + half_w)
        row = (centres >= mid - half_h) & (centres <= mid + half_h)
        masks.append((row[:, None] & col[None, :]).reshape(-1))
    return torch.stack(masks)


def load_dinov2(cfg: dict, device: torch.device, repo_root):
    from transformers import AutoModel  # imported here so that tests of the preprocessing do not need the weights
    model = AutoModel.from_pretrained(cfg["model_id"], revision=cfg["revision"], cache_dir=str(repo_root / cfg["cache_dir"]))
    return model.eval().to(device)


@torch.inference_mode()
def embed(model, images: list[Image.Image], device: torch.device, cfg: dict) -> np.ndarray:
    """(N, D) float32 array of unit-length embeddings, in the order of `images`."""
    out = []
    for i in range(0, len(images), cfg["batch_size"]):
        batch = images[i:i + cfg["batch_size"]]
        x = preprocess(batch, cfg["image_size"], cfg["pad_fill"], cfg["mean"], cfg["std"]).to(device)
        hidden = model(pixel_values=x).last_hidden_state          # (batch, 1 + patches, D); token 0 is the CLS token
        if cfg["pooling"] == "cls":
            v = hidden[:, 0]
        elif cfg["pooling"] == "mean":
            v = hidden[:, 1:].mean(dim=1)
        else:                                                      # mean_content: mean of the patches on the crop, padding patches left out
            m = content_patch_mask([im.size for im in batch], cfg["image_size"]).to(device).unsqueeze(-1).float()
            v = (hidden[:, 1:] * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        out.append(torch.nn.functional.normalize(v, dim=-1).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 0), np.float32)
