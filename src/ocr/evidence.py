"""From OCR text to evidence: which of the six keywords were read. Rules (fixed before any number, see reports/stage_07_proposal.md):
 - text is uppercased, accents removed, anything but letters, digits, spaces and hyphens becomes a space;
 - WHOLE words only: FANTASIA is not FANTA (the legal text "bebida de fantasia" must not read as Fanta);
 - words of 5 or more letters (SPRITE, LIGHT, FANTA, RETORNABLE) also match with a rapidfuzz ratio >= `min_ratio` (OCR confuses O/0, I/L);
 - the 4-letter ZERO must match exactly, or glued to AZUCAR/AZUCARES (the OCR reads the line "ZERO AZUCAR" as ZEROAZUCAR);
 - COCA-COLA is COCA and COLA as adjacent words, or COCACOLA in one word (rarely readable: never required).
"""
import re
import unicodedata

from src.ocr.reader import require_ocr

BRAND_WORDS = ("SPRITE", "FANTA")
VARIANT_WORDS = ("LIGHT", "ZERO", "RETORNABLE")
OTHER_BRAND_WORDS = ("BILZ", "PEPSI")                 # brands that must be `unknown`; exact whole words only (owner decision)
KEYWORDS = ("COCA-COLA",) + VARIANT_WORDS + BRAND_WORDS
_ZERO_GLUED = re.compile(r"^ZERO(AZUCAR(ES)?)?$")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().upper()
    return re.sub(r"[^A-Z0-9 \-]", " ", text)


def tokens(text: str) -> list[str]:
    """Normalised whole words (hyphens split words: COCA-COLA gives COCA, COLA)."""
    return [t for t in re.split(r"[ \-]+", normalize(text)) if t]


def _similar(a: str, b: str, min_ratio: float) -> bool:
    from rapidfuzz import fuzz
    return fuzz.ratio(a, b) / 100.0 >= min_ratio


def keywords_detail(text: str, min_ratio: float = 0.85, include_other: bool = True) -> dict[str, str]:
    """{keyword: "exact" or "fuzzy"} for the keywords found as whole words. "fuzzy" = accepted only through the rapidfuzz ratio (for example
    the fragments FANT and ANTA for FANTA). ZERO glued to AZUCAR(ES) and RETORNABLES count as exact. BILZ and PEPSI (`include_other`) are
    exact whole words only."""
    require_ocr()
    toks = tokens(text)
    found: dict[str, str] = {}

    def add(word: str, kind: str):
        if found.get(word) != "exact":                      # an exact reading wins over a fuzzy one
            found[word] = kind
    for t in toks:
        if _ZERO_GLUED.match(t):
            add("ZERO", "exact")
        for word in ("SPRITE", "FANTA", "LIGHT", "RETORNABLE"):
            if t == word:
                add(word, "exact")
            elif len(word) >= 5 and len(t) >= 4 and _similar(t, word, min_ratio):
                add(word, "fuzzy")
        if t == "RETORNABLES":
            add("RETORNABLE", "exact")
        if include_other and t in OTHER_BRAND_WORDS:
            add(t, "exact")
    if any(a == "COCA" and b == "COLA" for a, b in zip(toks, toks[1:])) or "COCACOLA" in toks:
        add("COCA-COLA", "exact")
    return found


def keywords_read(text: str, min_ratio: float = 0.85, include_other: bool = True) -> set[str]:
    """The keywords found as whole words in the text (see keywords_detail)."""
    return set(keywords_detail(text, min_ratio, include_other))
