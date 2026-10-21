import re
import unicodedata
from pathlib import Path

import yaml

from src.data.ids import assign_store_codes, make_photo_id, parse_name, photo_hash

CFG = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/stage1.yaml").read_text())
# Synthetic name with the same shape as the real ones (no real data).
NAME = "123456789_TEST STORE - ÑUÑOA ST N° 10_jane.doe_01.02.2026_abc123def-PHOTO-000007.jpg"


def test_photo_id_format_and_determinism():
    pid = make_photo_id("s07", NAME)
    assert re.fullmatch(r"s07_[0-9a-f]{8}", pid)
    assert pid == make_photo_id("s07", NAME)


def test_id_does_not_depend_on_unicode_form():
    nfc = unicodedata.normalize("NFC", NAME)
    nfd = unicodedata.normalize("NFD", NAME)
    assert nfc != nfd  # the degree sign / accents really differ in bytes
    assert photo_hash(nfc) == photo_hash(nfd)


def test_different_names_give_different_hashes():
    assert photo_hash(NAME) != photo_hash(NAME.replace("000007", "000008"))


def test_parse_name():
    info = parse_name(NAME, CFG["name_regex"])
    assert info["store"] == "123456789"
    assert info["uploader"] == "jane.doe"
    assert info["date"] == "01.02.2026"
    assert info["store_name"].startswith("TEST STORE")


def test_parse_name_returns_none_for_screenshots():
    assert parse_name("Captura de pantalla 2000-01-01 000000.png", CFG["name_regex"]) is None


def test_store_codes_are_stable_when_a_store_is_added():
    first = assign_store_codes({}, {"300", "100", "200"})
    assert first == {"100": "s01", "200": "s02", "300": "s03"}
    # A new store with a smaller id must NOT renumber the existing ones.
    second = assign_store_codes(first, {"100", "200", "300", "050"})
    assert {k: second[k] for k in first} == first
    assert second["050"] == "s04"
