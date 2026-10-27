"""Check that the database search gives the same nearest entry as the offline numpy search over catalog (b) (the catalog of the
database: 30 originals + capped owner-labeled train shelf entries + unknown negatives, `catalog.use_shelf`), on the labeled val crops.

Usage: python scripts/check_catalog_db.py
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.embeddings import catalog_db  # noqa: E402
from src.embeddings.eval_data import load_originals, load_shelf, load_val_queries  # noqa: E402
from src.embeddings.identify import cosine_matrix  # noqa: E402
from src.embeddings.shelf_catalog import assemble  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage6.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load((REPO_ROOT / args.config).read_text())
    sc, db = cfg["shelf_round"]["catalog"], cfg["database"]
    val = load_val_queries(cfg, REPO_ROOT)
    cat = assemble(load_originals(cfg, REPO_ROOT), load_shelf(cfg, REPO_ROOT), "originals+shelf+negatives", sc["cap_per_product"], sc["cap_unknown"], cfg["seed"])
    sims = cosine_matrix(val["vec"], cat["vec"])
    conn = catalog_db.connect(db, REPO_ROOT, db["name"])
    try:
        rows = conn.run("SELECT count(*) FROM catalog_entries")[0][0]
        bad = 0
        for i, v in enumerate(val["vec"]):
            hit = catalog_db.search(conn, v, 1)[0]
            j = int(np.argmax(sims[i]))
            bad += not (hit["product_id"] == cat["product"][j] and abs(hit["similarity"] - sims[i, j]) < 1e-3)
        per = conn.run("SELECT product_id, count(*) FROM catalog_entries GROUP BY product_id ORDER BY product_id")
    finally:
        conn.close()
    print(f"database {db['name']}: {rows} rows ({', '.join(f'{p} {n}' for p, n in per)}); offline catalog (b): {len(cat['product'])} entries ({dict(sorted(Counter(cat['product']).items()))})")
    print(f"{len(val['vec'])} val crops searched in Postgres against the offline search: {bad} disagreement(s)")
    return 1 if bad or rows != len(cat["product"]) else 0


if __name__ == "__main__":
    sys.exit(main())
