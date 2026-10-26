import numpy as np
import torch
from PIL import Image
from transformers import Dinov2Config, Dinov2Model

from src.embeddings.embed import embed, pad_to_square, preprocess

CFG = {"image_size": 56, "pad_fill": [255, 255, 255], "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
       "pooling": "cls", "batch_size": 2}


def _tiny_model():
    torch.manual_seed(0)
    return Dinov2Model(Dinov2Config(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                                    image_size=56, patch_size=14)).eval()


def _img(w, h, color):
    return Image.new("RGB", (w, h), color)


def test_padding_keeps_the_crop_centred_on_a_white_square():
    sq = pad_to_square(_img(20, 60, (0, 0, 0)))
    assert sq.size == (60, 60) and sq.getpixel((30, 30)) == (0, 0, 0) and sq.getpixel((2, 30)) == (255, 255, 255)


def test_preprocess_shape_and_normalisation():
    x = preprocess([_img(20, 60, (255, 255, 255))], 56, [255, 255, 255], CFG["mean"], CFG["std"])
    assert x.shape == (1, 3, 56, 56)
    assert abs(float(x[0, 0, 0, 0]) - (1 - 0.485) / 0.229) < 1e-5     # white pixel, red channel


def test_embeddings_are_unit_length_deterministic_and_ordered():
    model, imgs = _tiny_model(), [_img(20, 60, (200, 10, 10)), _img(40, 40, (10, 200, 10)), _img(30, 90, (10, 10, 200))]
    a, b = embed(model, imgs, torch.device("cpu"), CFG), embed(model, imgs, torch.device("cpu"), CFG)
    assert a.shape == (3, 32) and np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5) and np.allclose(a, b)
    single = embed(model, imgs[1:2], torch.device("cpu"), CFG)
    assert np.allclose(a[1], single[0], atol=1e-5)          # the batch does not change a vector
    assert not np.allclose(a[0], a[1])


def test_mean_pooling_also_gives_unit_vectors():
    v = embed(_tiny_model(), [_img(20, 60, (5, 5, 5))], torch.device("cpu"), {**CFG, "pooling": "mean"})
    assert abs(float(np.linalg.norm(v[0])) - 1.0) < 1e-5
