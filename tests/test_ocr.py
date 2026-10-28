import builtins
import sys

import pytest

from src.ocr import reader as ocr_reader
from src.ocr.evidence import keywords_read, tokens


def test_whole_words_only_so_fantasia_is_not_fanta():
    assert keywords_read("BEBIDA DE FANTASIA") == set()
    assert keywords_read("fanta naranja") == {"FANTA"}
    assert keywords_read("SPRITE") == {"SPRITE"}


def test_accents_case_and_noise_are_ignored():
    assert keywords_read("  retornable!! 1,5 L ") == {"RETORNABLE"}
    assert keywords_read("Retornáble") == {"RETORNABLE"}          # accent removed, then exact
    assert keywords_read("x#@ ~ 12") == set()


def test_zero_exact_or_glued_to_azucar_but_not_inside_other_words():
    assert keywords_read("ZERO") == {"ZERO"}
    assert keywords_read("ZEROAZUCAR") == {"ZERO"}
    assert keywords_read("ZERO AZUCARES") == {"ZERO"}
    assert keywords_read("ZEROS") == set() and keywords_read("ZER0") == set() and keywords_read("HEROES") == set()


def test_fuzzy_only_for_long_words():
    assert keywords_read("LIGNT") == set()             # 5 letters, one wrong: ratio 0.8 < 0.85 (the declared tolerance)
    assert keywords_read("RETORNABLF") == {"RETORNABLE"}   # 10 letters, one wrong: 0.9
    assert keywords_read("SPRITF") == set()            # ratio 0.833 < 0.85


def test_coca_cola_as_two_words_or_one():
    assert keywords_read("COCA COLA") == {"COCA-COLA"} and keywords_read("Coca-Cola") == {"COCA-COLA"} and keywords_read("COCACOLA") == {"COCA-COLA"}
    assert keywords_read("COCA") == set()


def test_two_brands_are_both_reported():
    assert keywords_read("FANTA SPRITE") == {"FANTA", "SPRITE"}
    assert tokens("coca-cola light") == ["COCA", "COLA", "LIGHT"]


def test_upscale_only_when_shorter():
    from PIL import Image
    assert ocr_reader.upscale_for_reading(Image.new("RGB", (100, 300)), 900).size == (300, 900)
    assert ocr_reader.upscale_for_reading(Image.new("RGB", (100, 1000)), 900).size == (100, 1000)


@pytest.fixture
def hidden_ocr_packages(monkeypatch):
    """Make `import easyocr` / `import rapidfuzz` fail as if requirements-ocr.txt were not installed."""
    real = builtins.__import__

    def fake(name, *a, **k):
        if name.split(".")[0] in ("easyocr", "rapidfuzz"):
            raise ImportError(f"No module named {name!r}")
        return real(name, *a, **k)
    for m in [m for m in sys.modules if m.split(".")[0] in ("easyocr", "rapidfuzz")]:
        monkeypatch.delitem(sys.modules, m)
    monkeypatch.setattr(builtins, "__import__", fake)


def test_without_the_ocr_packages_use_ocr_fails_with_one_clear_message(hidden_ocr_packages):
    with pytest.raises(ocr_reader.OcrUnavailable, match=r"pip install -r requirements-ocr.txt"):
        ocr_reader.require_ocr()
    with pytest.raises(ocr_reader.OcrUnavailable, match="requirements-ocr.txt"):
        ocr_reader.OcrReader({"model_dir": "x", "upscale_to_min_height_px": 900, "languages": ["en"]}, __import__("pathlib").Path("."), type("D", (), {"type": "cpu"})())
    with pytest.raises(ocr_reader.OcrUnavailable):
        keywords_read("FANTA")


def test_variant_a_modules_run_without_the_ocr_packages(hidden_ocr_packages):
    """Everything variant A uses imports and works with the OCR packages hidden: nothing in it touches src.ocr."""
    import importlib
    import numpy as np
    for name in ("src.embeddings.identify", "src.embeddings.experiments", "src.embeddings.classifier", "src.embeddings.answers", "src.embeddings.demo"):
        importlib.import_module(name)
    from src.embeddings.experiments import decide_prob, prob_features
    feat = prob_features(np.array([[0.9, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01]]), ["a", "b", "c", "d", "e", "f"])
    assert decide_prob(feat, 0.3, 0.75)[0] == "automatic"


def test_other_brand_words_are_exact_whole_words_only():
    from src.ocr.evidence import keywords_detail, keywords_read
    assert keywords_detail("PEPSI MAX") == {"PEPSI": "exact"} and keywords_read("bilz y pap") == {"BILZ"}
    assert keywords_read("EPSI") == set() and keywords_read("PEPSIS") == set() and keywords_read("BILZA") == set()      # no fuzzy matching for them
    assert keywords_read("PEPSI", include_other=False) == set()


def test_exact_and_fuzzy_kinds():
    from src.ocr.evidence import keywords_detail
    assert keywords_detail("FANT 1,5") == {"FANTA": "fuzzy"} and keywords_detail("ANTA") == {"FANTA": "fuzzy"}
    assert keywords_detail("FANTA") == {"FANTA": "exact"} and keywords_detail("FANT FANTA") == {"FANTA": "exact"}
    assert keywords_detail("ZEROAZUCAR") == {"ZERO": "exact"}


def test_demo_use_ocr_stops_with_the_install_message_when_the_packages_cannot_be_imported():
    """Run the real demo script with easyocr made un-importable: it must exit with the clear message before touching any photo or model."""
    import subprocess
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    code = ("import sys, runpy; sys.modules['easyocr'] = None; sys.argv = ['demo_identify.py', '--image', 'does-not-matter.jpg', '--use-ocr']; "
            "runpy.run_path('scripts/demo_identify.py', run_name='__main__')")
    r = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True)
    assert r.returncode != 0 and "pip install -r requirements-ocr.txt" in (r.stderr + r.stdout)
