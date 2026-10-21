"""Detect real photo identifiers (store ids, store names, uploader names, file names)
in text that is about to be committed. Matched strings are never returned or printed:
only the line number and the kind of match."""
import re
import unicodedata

from src.data.ids import parse_name

MIN_NEEDLE_LENGTH = 4  # shorter strings match too many harmless words


def _norm(text: str) -> str:
    return unicodedata.normalize("NFC", text).casefold()


def build_needles(file_names: list[str], name_regex: str, store_ids: list[str] = ()) -> list[tuple[str, str]]:
    """(kind, normalized string) for every identifying string in the given file names.
    Names that do not match the usual pattern (e.g. screenshots) only contribute the full name."""
    found = {("store_id", s) for s in store_ids}
    for name in file_names:
        found.add(("file_name", name))
        info = parse_name(name, name_regex)
        if info:
            found |= {("store_id", info["store"]), ("store_name", info["store_name"]),
                      ("uploader", info["uploader"])}
    return sorted((kind, _norm(s)) for kind, s in found if len(s) >= MIN_NEEDLE_LENGTH)


def scan_text(text: str, needles: list[tuple[str, str]]) -> list[tuple[int, str]]:
    """Return sorted (line number, kind) pairs; never the matched text."""
    hits = set()
    for line_no, line in enumerate(text.split("\n"), 1):
        line = _norm(line)
        for kind, needle in needles:
            if needle in line:
                hits.add((line_no, kind))
    return sorted(hits)



# --- Mapping-free check -----------------------------------------------------------------
# Anything shaped like a raw photo file name is flagged even when the local mapping does not
# exist (for example in a fresh clone). Lines holding synthetic test fixtures opt out with the
# marker below; a real-looking name without the marker is still caught.
ALLOW_MARKER = "leak-check: allow-synthetic"
NAME_SHAPES = (
    ("raw_file_name_shape", re.compile(r"_\d{2}\.\d{2}\.\d{4}_")),           # _dd.mm.yyyy_ between underscores
    ("raw_file_name_shape", re.compile(r"-(?:PHOTO|PH)-\d{4,}", re.IGNORECASE)),  # -PHOTO- or -PH- followed by a long number
    ("screenshot_name_shape", re.compile(r"Captura de pantalla \d{4}-\d{2}-\d{2}", re.IGNORECASE)),
)


def scan_name_shapes(text: str) -> list[tuple[int, str]]:
    """Return sorted (line number, kind) pairs for lines shaped like a raw photo file name;
    never the matched text. Lines containing ALLOW_MARKER are skipped."""
    hits = set()
    for line_no, line in enumerate(text.split("\n"), 1):
        if ALLOW_MARKER in line:
            continue
        for kind, pattern in NAME_SHAPES:
            if pattern.search(line):
                hits.add((line_no, kind))
    return sorted(hits)
