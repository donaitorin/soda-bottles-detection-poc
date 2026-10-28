"""The OCR reader of variant B. EasyOCR is imported ONLY when a reader is built, never at import time, so variant A runs without
the OCR packages (requirements-ocr.txt) and asking for OCR without them stops with one clear message."""
from pathlib import Path

import numpy as np
from PIL import Image

INSTALL_HINT = "Variant B (use_ocr) needs the OCR packages: run `pip install -r requirements-ocr.txt` inside the soda-poc environment."


class OcrUnavailable(RuntimeError):
    """The OCR packages are not installed (or cannot be imported)."""


def require_ocr() -> None:
    """Raise OcrUnavailable with the install hint unless easyocr and rapidfuzz can be imported."""
    for module in ("easyocr", "rapidfuzz"):
        try:
            __import__(module)
        except ImportError as error:
            raise OcrUnavailable(f"{INSTALL_HINT} (could not import {module})") from error


def upscale_for_reading(image: Image.Image, min_height: int) -> Image.Image:
    """A crop shorter than `min_height` is enlarged to it (same aspect, LANCZOS); a taller one is read as it is. Small text needs pixels:
    in the trial more words were found after upscaling."""
    if image.height >= min_height:
        return image
    return image.resize((max(1, round(image.width * min_height / image.height)), min_height), Image.LANCZOS)


class OcrReader:
    """EasyOCR on the project device (MPS when available). `read(image)` returns the text pieces EasyOCR found, as (text, confidence)."""

    def __init__(self, cfg: dict, root: Path, device):
        require_ocr()
        import easyocr
        model_dir = root / cfg["model_dir"]
        model_dir.mkdir(parents=True, exist_ok=True)
        self.min_height = cfg["upscale_to_min_height_px"]
        self.reader = easyocr.Reader(cfg["languages"], gpu=device.type in ("mps", "cuda"), model_storage_directory=str(model_dir), verbose=False)

    def read(self, image: Image.Image) -> list[tuple[str, float]]:
        img = upscale_for_reading(image, self.min_height)
        return [(text, float(conf)) for _, text, conf in self.reader.readtext(np.asarray(img), detail=1)]
