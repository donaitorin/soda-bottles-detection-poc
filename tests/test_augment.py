import numpy as np
from PIL import Image

from src.embeddings.augment import FORBIDDEN, apply_params, augment, sample_params, seed_for

CFG = {"per_photo": 6, "rotation_deg": 7, "perspective_distortion": 0.12, "scale": [0.85, 1.15],
       "brightness": 0.25, "contrast": 0.25, "blur_sigma": [0.1, 1.2]}


def _bottle(color=(200, 20, 20)):
    img = Image.new("RGB", (80, 200), (255, 255, 255))
    img.paste(color, (20, 20, 60, 180))
    return img


def test_parameters_are_inside_the_configured_ranges_and_reproducible():
    a = sample_params(np.random.default_rng(1), CFG, 50)
    b = sample_params(np.random.default_rng(1), CFG, 50)
    assert a == b
    assert all(abs(p["rotation_deg"]) <= 7 and 0.85 <= p["scale"] <= 1.15 and 0.75 <= p["brightness"] <= 1.25
               and 0.75 <= p["contrast"] <= 1.25 and 0.1 <= p["blur_sigma"] <= 1.2
               and all(abs(v) <= 0.12 for c in p["perspective"] for v in c) for p in a)
    assert not any(k in p for p in a for k in FORBIDDEN)


def test_seed_is_stable_per_photo_name():
    assert seed_for(0, "coca_zero/coca_zero_01.jpg") == seed_for(0, "coca_zero/coca_zero_01.jpg") != seed_for(0, "coca_zero/coca_zero_02.jpg")


def test_output_keeps_the_size_and_the_same_seed_gives_the_same_pixels():
    img = _bottle()
    a, b = augment(img, CFG, seed=3), augment(img, CFG, seed=3)
    assert len(a) == 6 and all(im.size == img.size for im, _ in a)
    assert all(np.array_equal(np.asarray(x), np.asarray(y)) for (x, _), (y, _) in zip(a, b))
    assert any(not np.array_equal(np.asarray(x), np.asarray(img)) for x, _ in a)


def test_no_mirroring_the_asymmetric_side_stays_on_its_side():
    img = Image.new("RGB", (80, 200), (255, 255, 255))
    img.paste((0, 0, 0), (10, 40, 38, 160))       # a dark mark on the LEFT half only
    for out, _ in augment(img, CFG, seed=5, n=20):
        arr = np.asarray(out.convert("L")).astype(float)
        assert arr[:, :40].mean() < arr[:, 40:].mean()


def test_colour_stays_the_same_colour_no_hue_or_saturation_shift():
    for out, _ in augment(_bottle((220, 20, 20)), CFG, seed=7, n=20):
        arr = np.asarray(out).astype(int)
        core = arr[90:110, 35:45].reshape(-1, 3).mean(axis=0)          # the middle of the red bottle body
        assert core[0] > core[1] + 40 and core[0] > core[2] + 40       # still clearly red
        assert abs(core[1] - core[2]) < 30                             # no drift towards green or blue


def test_identity_parameters_leave_the_image_almost_unchanged():
    ident = {"rotation_deg": 0.0, "scale": 1.0, "perspective": [[0, 0]] * 4, "brightness": 1.0, "contrast": 1.0, "blur_sigma": 0.1}
    img = _bottle()
    out = apply_params(img, ident)
    assert np.abs(np.asarray(out).astype(int) - np.asarray(img).astype(int)).mean() < 2.0


def test_a_dark_variant_gets_a_dark_background_no_light_frame():
    img = _bottle()
    p = {"rotation_deg": 6.0, "scale": 0.9, "perspective": [[0.1, 0.1], [-0.1, 0.1], [0.1, -0.1], [-0.1, -0.1]],
         "brightness": 0.75, "contrast": 1.0, "blur_sigma": 0.1}
    arr = np.asarray(apply_params(img, p)).astype(int)
    corner, edge_mid = arr[2, 2], arr[100, 2]
    assert corner.max() < 215 and abs(corner - edge_mid).max() < 25      # the filled corner matches the (darkened) background
