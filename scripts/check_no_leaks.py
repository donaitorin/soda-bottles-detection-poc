"""Fail if any file that could be committed contains a real store id, store name or
address, uploader name or photo file name.

The strings to look for are parsed from the file names in data/raw (through the local
mapping data/id_maps/photo_ids.csv) and in data/test_images. Matched strings are never
printed: only file, line and kind of match.

Two checks run: (1) the strings from the local mapping (skipped with a warning when the mapping
does not exist) and (2) a mapping-free check for anything shaped like a raw photo file name
(`_dd.mm.yyyy_`, `-PHOTO-<digits>`, `Captura de pantalla <date>`). Synthetic test fixtures opt out
by carrying the marker `leak-check: allow-synthetic` on the same line.

Modes:
  default   every git-tracked file plus modified or untracked files that are not ignored
  --staged  only the staged version of staged files (for the pre-commit hook)
  --extra   additional files to scan, e.g. the ignored .handoff/latest.md

Usage: python scripts/check_no_leaks.py [--staged] [--extra PATH ...]
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.ids import parse_name  # noqa: E402
from src.data.leaks import build_needles, scan_name_shapes, scan_text  # noqa: E402


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage1.yaml")
    parser.add_argument("--staged", action="store_true")
    parser.add_argument("--extra", nargs="*", default=[])
    args = parser.parse_args()
    cfg = yaml.safe_load((REPO_ROOT / args.config).read_text())

    maps = REPO_ROOT / cfg["id_maps_dir"]
    needles = []
    if (maps / "photo_ids.csv").exists():
        names = [r["file_name"] for r in csv.DictReader((maps / "photo_ids.csv").open(newline=""))]
        store_ids = [r["store_id"] for r in csv.DictReader((maps / "store_codes.csv").open(newline=""))]
        test_dir = REPO_ROOT / "data" / "test_images"  # not in the mapping, but its names are personal too
        if test_dir.is_dir():
            names += [p.name for p in test_dir.iterdir() if p.is_file() and parse_name(p.name, cfg["name_regex"])]
        needles = build_needles(names, cfg["name_regex"], store_ids)
    else:
        print("WARNING: data/id_maps/photo_ids.csv not found, so real ids and names cannot be searched "
              "(run scripts/build_ids.py). Only the name-shape check runs.")

    problems = 0

    def check(label: str, text: str) -> None:
        nonlocal problems
        for _, kind in scan_text(label, needles) + scan_name_shapes(label):  # the path itself can leak too
            print(f"{label}: file path contains a real {kind}")
            problems += 1
        for line_no, kind in scan_text(text, needles):
            print(f"{label}:{line_no}: real {kind}")
            problems += 1
        for line_no, kind in scan_name_shapes(text):
            print(f"{label}:{line_no}: text shaped like a photo file name ({kind})")
            problems += 1

    if args.staged:
        paths = git("diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z").decode().split("\0")
        for path in filter(None, paths):
            data = git("show", f":{path}")  # the staged version, not the working-tree one
            if b"\0" not in data:
                check(path, data.decode("utf-8", errors="replace"))
        scanned = len(list(filter(None, paths)))
    else:
        paths = git("ls-files", "-c", "-o", "--exclude-standard", "-z").decode().split("\0")
        paths = [p for p in paths if p] + list(args.extra)
        scanned = 0
        for path in paths:
            full = REPO_ROOT / path
            if not full.is_file():
                continue
            data = full.read_bytes()
            scanned += 1
            if b"\0" not in data:
                check(path, data.decode("utf-8", errors="replace"))
    print(f"leak check: {scanned} files scanned, {len(needles)} identifying strings + name-shape patterns, {problems} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
