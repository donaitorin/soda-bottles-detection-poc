"""Image loading shared by the whole project."""
from pathlib import Path

from PIL import Image, ImageOps


def load_image(path: str | Path) -> Image.Image:
    """Open an image as RGB with its EXIF orientation applied.

    Phone photos store the sensor orientation in an EXIF flag instead of rotating the
    pixels. Without exif_transpose they load sideways and every box ends up in the wrong
    place. Today every photo in data/raw has orientation 1 (normal), so this is a safety
    net. data/raw is never modified: the fix is applied in memory, at load time.
    The RGB conversion handles PNGs with an alpha channel or a palette.
    """
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB")
