"""The reference catalog in Postgres + pgvector, in its OWN database (`soda_catalog`), through the pure-Python driver pg8000.

Never the Label Studio data and never `docker compose down -v`. Credentials come from the environment or the gitignored .env.
Similarity is cosine: pgvector's `<=>` is the cosine DISTANCE, so similarity = 1 - distance (the vectors are unit length).
"""
import json
import os
import re
from pathlib import Path

import pg8000.native

SCHEMA = """
CREATE TABLE IF NOT EXISTS catalog_entries (
    id            serial PRIMARY KEY,
    product_id    text NOT NULL,
    brand         text NOT NULL,
    size          text,
    material      text,
    source_photo  text NOT NULL,          -- catalog photo, relative to data/catalog/product_photos
    source_url    text NOT NULL,
    license       text NOT NULL,
    packaging_checked text NOT NULL,
    is_augmented  boolean NOT NULL,
    aug_index     integer,                -- 1..N for an augmented entry, NULL for the original crop
    aug_params    jsonb,
    crop_method   text NOT NULL,
    embedding     vector({dim}) NOT NULL
)
"""


def read_env(repo_root: Path) -> dict:
    """POSTGRES_USER and POSTGRES_PASSWORD from the environment, else from .env."""
    values = {k: os.environ[k] for k in ("POSTGRES_USER", "POSTGRES_PASSWORD") if k in os.environ}
    env_file = repo_root / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            m = re.match(r"^\s*([A-Z_]+)\s*=\s*(.*?)\s*$", line)
            if m and m.group(1) in ("POSTGRES_USER", "POSTGRES_PASSWORD"):
                values.setdefault(m.group(1), m.group(2))
    return values


def connect(cfg: dict, repo_root: Path, database: str):
    env = read_env(repo_root)
    return pg8000.native.Connection(user=env["POSTGRES_USER"], password=env["POSTGRES_PASSWORD"], host=cfg["host"], port=cfg["port"], database=database)


def _safe(name: str) -> str:
    if not re.fullmatch(r"[a-z0-9_]+", name):
        raise ValueError(f"unsafe database name {name!r}")
    return name


def create_database(cfg: dict, repo_root: Path, name: str) -> bool:
    """Create the database if it does not exist (connects to the maintenance database). Returns True if it was created."""
    conn = connect(cfg, repo_root, "postgres")
    try:
        if conn.run("SELECT 1 FROM pg_database WHERE datname = :n", n=name):
            return False
        conn.run(f"CREATE DATABASE {_safe(name)}")
        return True
    finally:
        conn.close()


def drop_database(cfg: dict, repo_root: Path, name: str) -> None:
    conn = connect(cfg, repo_root, "postgres")
    try:
        conn.run("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = :n AND pid <> pg_backend_pid()", n=name)
        conn.run(f"DROP DATABASE IF EXISTS {_safe(name)}")
    finally:
        conn.close()


def init_schema(conn, dim: int, hnsw: dict) -> None:
    conn.run("CREATE EXTENSION IF NOT EXISTS vector")
    conn.run(SCHEMA.format(dim=int(dim)))
    conn.run(f"CREATE INDEX IF NOT EXISTS catalog_entries_embedding_hnsw ON catalog_entries USING hnsw (embedding vector_cosine_ops) "
             f"WITH (m = {int(hnsw['m'])}, ef_construction = {int(hnsw['ef_construction'])})")


def vector_literal(v) -> str:
    return "[" + ",".join(f"{float(x):.7f}" for x in v) + "]"


def reload_entries(conn, entries: list[dict], embeddings) -> int:
    """Idempotent rebuild: empty the table, insert every entry. `entries` hold the metadata columns, `embeddings` the vectors."""
    conn.run("TRUNCATE catalog_entries RESTART IDENTITY")
    for e, v in zip(entries, embeddings):
        conn.run("INSERT INTO catalog_entries (product_id, brand, size, material, source_photo, source_url, license, packaging_checked, is_augmented, "
                 "aug_index, aug_params, crop_method, embedding) VALUES (:product_id, :brand, :size, :material, :source_photo, :source_url, :license, "
                 ":packaging_checked, :is_augmented, :aug_index, CAST(:aug_params AS jsonb), :crop_method, CAST(:emb AS vector))",
                 product_id=e["product_id"], brand=e["brand"], size=e.get("size") or None, material=e.get("material") or None,
                 source_photo=e["source_photo"], source_url=e["source_url"], license=e["license"], packaging_checked=e["packaging_checked"],
                 is_augmented=bool(e["is_augmented"]), aug_index=e.get("aug_index"), aug_params=json.dumps(e["aug_params"]) if e.get("aug_params") else None,
                 crop_method=e["crop_method"], emb=vector_literal(v))
    return conn.run("SELECT count(*) FROM catalog_entries")[0][0]


def search(conn, query, k: int, include_augmented: bool = True) -> list[dict]:
    """The k nearest catalog entries to a unit-length query vector, with their cosine similarity."""
    where = "" if include_augmented else "WHERE NOT is_augmented"
    rows = conn.run(f"SELECT id, product_id, source_photo, is_augmented, 1 - (embedding <=> CAST(:q AS vector)) AS sim FROM catalog_entries {where} "
                    f"ORDER BY embedding <=> CAST(:q AS vector) LIMIT :k", q=vector_literal(query), k=int(k))
    return [{"id": r[0], "product_id": r[1], "source_photo": r[2], "is_augmented": r[3], "similarity": float(r[4])} for r in rows]


def counts(conn) -> list[tuple]:
    return [tuple(r) for r in conn.run("SELECT product_id, count(*) FILTER (WHERE NOT is_augmented), count(*) FILTER (WHERE is_augmented) FROM catalog_entries GROUP BY product_id ORDER BY product_id")]
