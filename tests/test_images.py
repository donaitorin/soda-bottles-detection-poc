from PIL import Image

from src.data.images import load_image


def _save_with_orientation(path, orientation):
    # 40 wide x 20 tall: left half red, right half blue.
    image = Image.new("RGB", (40, 20), (255, 0, 0))
    image.paste((0, 0, 255), (20, 0, 40, 20))
    exif = Image.Exif()
    exif[274] = orientation  # EXIF tag 274 = Orientation
    image.save(path, exif=exif)


def test_orientation_6_is_rotated_to_upright(tmp_path):
    path = tmp_path / "rot.jpg"
    _save_with_orientation(path, 6)  # "rotate 90 degrees clockwise to display"
    loaded = load_image(path)
    assert loaded.size == (20, 40)  # width and height swapped
    # Rotating clockwise moves the left (red) half to the top.
    assert loaded.getpixel((10, 5))[0] > 200
    assert loaded.getpixel((10, 35))[2] > 200


def test_orientation_1_is_unchanged(tmp_path):
    path = tmp_path / "normal.jpg"
    _save_with_orientation(path, 1)
    assert load_image(path).size == (40, 20)


def test_png_with_alpha_becomes_rgb(tmp_path):
    path = tmp_path / "alpha.png"
    Image.new("RGBA", (8, 8), (1, 2, 3, 128)).save(path)
    assert load_image(path).mode == "RGB"
