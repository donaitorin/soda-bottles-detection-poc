import io

import imagehash
import numpy as np
from PIL import Image

HASH_SIZE = 16  # same as configs/stage1.yaml


def _random_scene(seed: int) -> Image.Image:
    # Smooth synthetic "photo": a few big random blobs, like low-frequency picture content.
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (12, 9, 3), dtype=np.uint8)
    return Image.fromarray(small).resize((600, 800), Image.BICUBIC)


def _distance(a: Image.Image, b: Image.Image) -> int:
    return int((imagehash.phash(a, hash_size=HASH_SIZE).hash != imagehash.phash(b, hash_size=HASH_SIZE).hash).sum())


def test_resized_and_recompressed_copy_is_close():
    original = _random_scene(1)
    buf = io.BytesIO()
    original.resize((300, 400)).save(buf, "JPEG", quality=60)
    assert _distance(original, Image.open(buf)) <= 12


def test_different_scenes_are_far():
    assert _distance(_random_scene(1), _random_scene(2)) > 50
