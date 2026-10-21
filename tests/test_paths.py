import pytest

from src.data.paths import check_output_dir, is_inside


def test_inside_and_outside(tmp_path):
    raw = tmp_path / "raw"
    (raw / "sub").mkdir(parents=True)
    assert is_inside(raw, raw)
    assert is_inside(raw / "sub", raw)
    assert not is_inside(tmp_path / "review", raw)
    assert not is_inside(tmp_path, raw)  # the parent is not inside


def test_dotdot_cannot_hide_a_protected_folder(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (tmp_path / "review").mkdir()
    assert is_inside(tmp_path / "review" / ".." / "raw", raw)


def test_symlink_cannot_hide_a_protected_folder(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    link = tmp_path / "innocent"
    link.symlink_to(raw, target_is_directory=True)
    with pytest.raises(ValueError):
        check_output_dir(link, [raw])


def test_check_output_dir(tmp_path):
    raw, test_images = tmp_path / "raw", tmp_path / "test_images"
    raw.mkdir()
    test_images.mkdir()
    check_output_dir(tmp_path / "review" / "duplicates", [raw, test_images])  # allowed
    for bad in (raw, raw / "x", test_images, test_images / "y" / "z"):
        with pytest.raises(ValueError):
            check_output_dir(bad, [raw, test_images])
