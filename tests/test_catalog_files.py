import csv

from PIL import Image

from src.embeddings.catalog_files import check_catalog_photos

PRODUCTS = ["coca_classic", "sprite"]
ARGS = dict(products=PRODUCTS, extensions=[".jpg", ".png"], required_columns=["file", "product_id", "source_url", "license", "uruguay_packaging_checked"],
            min_short_side=300, min_per_product=2, target_per_product=3)


def _photo(root, product, n, size=(400, 600), color=None):
    folder = root / product
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{product}_{n:02d}.jpg"
    Image.new("RGB", size, color or (n * 20 % 255, n * 40 % 255, 40 * (1 + PRODUCTS.index(product)))).save(path)
    return f"{product}/{path.name}"


def _sources(root, files, **override):
    with (root / "sources.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ARGS["required_columns"])
        w.writeheader()
        for rel in files:
            row = {"file": rel, "product_id": rel.split("/")[0], "source_url": "https://example.org/x", "license": "CC BY-SA 3.0",
                   "uruguay_packaging_checked": "yes", **override}
            w.writerow(row)


def _check(root):
    return check_catalog_photos(root, root / "sources.csv", **ARGS)


def test_a_clean_folder_passes(tmp_path):
    files = [_photo(tmp_path, p, n) for p in PRODUCTS for n in (1, 2, 3)]
    _sources(tmp_path, files)
    errors, warnings, counts = _check(tmp_path)
    assert errors == [] and warnings == [] and counts["sprite"] == 3


def test_missing_sources_row_empty_url_license_and_bad_flag_are_errors(tmp_path):
    files = [_photo(tmp_path, p, n) for p in PRODUCTS for n in (1, 2)]
    _sources(tmp_path, files[:-1])
    assert any("no row in sources.csv" in e for e in _check(tmp_path)[0])
    _sources(tmp_path, files, source_url=" ", license="", uruguay_packaging_checked="maybe")
    text = " | ".join(_check(tmp_path)[0])
    assert "source_url is empty" in text and "license is empty" in text and "must be yes or no" in text


def test_wrong_names_folders_duplicates_and_too_few_photos_are_errors(tmp_path):
    files = [_photo(tmp_path, "coca_classic", 1), _photo(tmp_path, "coca_classic", 2), _photo(tmp_path, "sprite", 1)]
    (tmp_path / "sprite" / "photo.jpg").write_bytes(b"x")
    (tmp_path / "fanta").mkdir()
    Image.new("RGB", (400, 600), (20, 40, 90)).save(tmp_path / "sprite" / "sprite_02.jpg")   # same pixels as coca_classic_01
    Image.new("RGB", (400, 600), (20, 40, 90)).save(tmp_path / "coca_classic" / "coca_classic_01.jpg")
    _sources(tmp_path, files + ["sprite/sprite_02.jpg"])
    text = " | ".join(_check(tmp_path)[0])
    assert "name must be sprite_NN" in text and "not one of the target product ids" in text and "identical file" in text


def test_small_images_and_unconfirmed_packaging_are_warnings(tmp_path):
    files = [_photo(tmp_path, p, n, size=(200, 250)) for p in PRODUCTS for n in (1, 2)]
    _sources(tmp_path, files, uruguay_packaging_checked="no")
    errors, warnings, _ = _check(tmp_path)
    assert errors == [] and any("short side 200" in w for w in warnings) and any("Uruguayan packaging not confirmed" in w for w in warnings)


def test_a_missing_folder_is_an_error_without_a_path(tmp_path):
    errors, _, _ = check_catalog_photos(tmp_path / "nope", tmp_path / "nope" / "sources.csv", **ARGS)
    assert len(errors) == 1 and str(tmp_path) not in errors[0]


def test_rename_plan_is_sorted_numbered_and_lowercase():
    from src.embeddings.catalog_files import plan_renames
    plan = plan_renames("sprite", ["sprite_3.JPG", "a_photo.jpeg", "sprite_1.png"])
    assert plan == [("a_photo.jpeg", "sprite_01.jpeg"), ("sprite_1.png", "sprite_02.png"), ("sprite_3.JPG", "sprite_03.jpg")]
