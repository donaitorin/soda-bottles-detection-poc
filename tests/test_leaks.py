from pathlib import Path

import yaml

from src.data.leaks import ALLOW_MARKER, build_needles, scan_name_shapes, scan_text

CFG = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/stage1.yaml").read_text())
# Synthetic names (no real data).
NAMES = ["123456789_TEST STORE - ÑUÑOA ST N° 10_jane.doe_01.02.2026_abc123def-PHOTO-000007.jpg",  # leak-check: allow-synthetic
         "Captura de pantalla 2000-01-01 000000.png"]  # leak-check: allow-synthetic
NEEDLES = build_needles(NAMES, CFG["name_regex"], ["123456789"])


def test_finds_each_kind_without_returning_the_text():
    text = "ok line\nstore 123456789 here\nuploader JANE.DOE wrote it\nTest Store - Ñuñoa st n° 10\n"
    hits = scan_text(text, NEEDLES)
    assert (2, "store_id") in hits and (3, "uploader") in hits and (4, "store_name") in hits
    assert all(isinstance(h[0], int) and isinstance(h[1], str) for h in hits)
    assert not any("123456789" in str(h) for h in hits)


def test_full_file_name_is_found_even_in_another_unicode_form():
    import unicodedata
    text = unicodedata.normalize("NFD", NAMES[1])
    assert (1, "file_name") in scan_text(text, NEEDLES)


def test_clean_text_has_no_hits():
    assert scan_text("photo s07_3fa2c9d1 in store s07\nnothing here", NEEDLES) == []


def test_short_strings_are_not_needles():
    needles = build_needles(["12_AB_cd_01.02.2026_xyz.jpg"], CFG["name_regex"])  # leak-check: allow-synthetic
    assert all(len(n) >= 4 for _, n in needles)


# --- mapping-free name-shape check ---------------------------------------------------------
# The "real-looking" names are assembled from pieces so this file does not match the pattern itself.
DATE = "01.02." + "2026"
REAL_LOOKING = "5001" + "00000_SOME SHOP_someone_" + DATE + "_" + "abcdef-PHOTO-" + "000123.jpg"


def test_real_looking_name_outside_the_allowlist_is_caught():
    assert scan_name_shapes(f"see {REAL_LOOKING} for details") == [(1, "raw_file_name_shape")]
    assert (1, "raw_file_name_shape") in scan_name_shapes("x_" + DATE + "_y")
    assert (1, "raw_file_name_shape") in scan_name_shapes("abc-PH-" + "000123.jpg")
    assert scan_name_shapes("Captura de pantalla " + "2026-09-04 165522.png") == [(1, "screenshot_name_shape")]


def test_marker_on_the_same_line_allows_synthetic_fixtures():
    assert scan_name_shapes(f"NAME = '{REAL_LOOKING}'  # {ALLOW_MARKER}") == []
    # the marker only protects its own line
    assert scan_name_shapes(f"# {ALLOW_MARKER}\n{REAL_LOOKING}") == [(2, "raw_file_name_shape")]


def test_generic_documentation_text_is_not_flagged():
    text = "names look like <store_id>_<store name>_<uploader>_<dd.mm.yyyy>_<hash>-PHOTO-<n>.jpg\n" \
           "screenshots (`Captura de pantalla ...`) are excluded"
    assert scan_name_shapes(text) == []
