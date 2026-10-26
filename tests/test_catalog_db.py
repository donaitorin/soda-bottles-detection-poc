"""Integration tests against the running Postgres container, in a THROWAWAY database that the test creates and drops.
They are skipped when the container or the .env credentials are not available. They never touch `soda_catalog`."""
import uuid
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.embeddings import catalog_db

REPO_ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((REPO_ROOT / "configs/stage6.yaml").read_text())["database"]


@pytest.fixture()
def conn():
    name = CFG["test_database_prefix"] + uuid.uuid4().hex[:10]
    try:
        catalog_db.create_database(CFG, REPO_ROOT, name)
        c = catalog_db.connect(CFG, REPO_ROOT, name)
    except Exception as error:  # noqa: BLE001
        pytest.skip(f"Postgres not available: {type(error).__name__}")
    catalog_db.init_schema(c, 4, CFG["hnsw"])
    yield c
    c.close()
    catalog_db.drop_database(CFG, REPO_ROOT, name)


def _entry(product, photo, aug=False):
    return {"product_id": product, "brand": "coca", "size": "1.5 L", "material": "PET", "source_photo": photo, "source_url": "https://example.org",
            "license": "unknown", "packaging_checked": "no", "is_augmented": aug, "aug_index": 1 if aug else None,
            "aug_params": {"rotation_deg": 1.0} if aug else None, "crop_method": "box10"}


def test_vector_literal_is_a_pgvector_text_value():
    assert catalog_db.vector_literal([1.0, 0.5]) == "[1.0000000,0.5000000]"


def test_unsafe_database_names_are_refused():
    with pytest.raises(ValueError):
        catalog_db._safe("x; DROP DATABASE soda_poc")


def test_reload_search_and_counts_in_a_throwaway_database(conn):
    vecs = np.eye(4, dtype=np.float32)
    entries = [_entry("coca_zero", "coca_zero/a.jpg"), _entry("sprite", "sprite/b.jpg"), _entry("fanta", "fanta/c.jpg"), _entry("fanta", "fanta/c.jpg", aug=True)]
    assert catalog_db.reload_entries(conn, entries + [], np.vstack([vecs[:3], vecs[2:3]])) == 4
    hits = catalog_db.search(conn, [0.0, 0.0, 1.0, 0.0], 2)
    assert hits[0]["product_id"] == "fanta" and abs(hits[0]["similarity"] - 1.0) < 1e-5
    assert catalog_db.search(conn, [0.0, 0.0, 1.0, 0.0], 3, include_augmented=False)[0]["is_augmented"] is False
    assert catalog_db.counts(conn) == [("coca_zero", 1, 0), ("fanta", 1, 1), ("sprite", 1, 0)]
    assert catalog_db.reload_entries(conn, entries[:1], vecs[:1]) == 1        # a rebuild replaces everything (idempotent)
