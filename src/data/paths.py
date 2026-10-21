"""Path safety helpers: scripts that write or delete files must never touch the photos."""
from pathlib import Path


def is_inside(path: Path, parent: Path) -> bool:
    """True if `path` is `parent` or lies below it. Paths are resolved first, so `..`
    and symlinks cannot hide a location inside a protected folder."""
    path, parent = Path(path).resolve(), Path(parent).resolve()
    return path == parent or parent in path.parents


def check_output_dir(output_dir: Path, protected_dirs: list[Path]) -> None:
    """Raise ValueError if output_dir is, or is inside, any protected (photo) folder."""
    for protected in protected_dirs:
        if is_inside(output_dir, protected):
            raise ValueError(f"refusing to write inside a photo folder: output dir is within a protected directory "
                             f"({Path(protected).name})")
