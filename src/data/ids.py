"""Anonymous ids for photos and stores.

Photo file names contain uploader names and store names and addresses, so they
must never be committed. Everything committed uses store codes (s01, s02, ...)
and photo ids (<store code>_<first 8 hex of sha256 of the file name>).
"""
import hashlib
import re
import unicodedata


def normalize_name(file_name: str) -> str:
    # macOS can return the same name composed (NFC) or decomposed (NFD) for accents
    # and symbols such as the degree sign. Normalizing first makes the id independent
    # of the filesystem.
    return unicodedata.normalize("NFC", file_name)


def photo_hash(file_name: str, length: int = 8) -> str:
    return hashlib.sha256(normalize_name(file_name).encode("utf-8")).hexdigest()[:length]


def make_photo_id(store_code: str, file_name: str, length: int = 8) -> str:
    return f"{store_code}_{photo_hash(file_name, length)}"


def parse_name(file_name: str, name_regex: str) -> dict | None:
    """Return the named groups (store, store_name, uploader, date, tail) or None."""
    match = re.match(name_regex, normalize_name(file_name))
    return match.groupdict() if match else None


def assign_store_codes(existing: dict[str, str], store_ids: set[str],
                       prefix: str = "s", width: int = 2) -> dict[str, str]:
    """Codes are assigned once: known stores keep their code, new stores get the
    next free codes in sorted store id order. Existing codes are never renumbered."""
    codes = dict(existing)
    used = {int(c[len(prefix):]) for c in codes.values()}
    next_n = max(used, default=0) + 1
    for store_id in sorted(store_ids - set(codes)):
        if next_n >= 10 ** width:
            raise ValueError(f"more than {10 ** width - 1} stores do not fit in {width} digits")
        codes[store_id] = f"{prefix}{next_n:0{width}d}"
        next_n += 1
    return codes
