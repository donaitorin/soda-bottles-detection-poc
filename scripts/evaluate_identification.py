"""Stage 6b: identification of the labeled val crops against the catalog, with the unknown rules (variant A, embeddings only).

DEVELOPMENT NUMBERS on 180 crops from 6 stores, not accuracy: the sample also tunes the thresholds, so the thresholds are tuned
LEAVING ONE STORE OUT (tuned on the other 5 stores, applied to the held-out one) and the predictions are pooled. Only crops
labeled with a product or `unknown` count for accuracy; `not_a_bottle` and `cannot_tell` are reported separately. Intervals are
a bootstrap that resamples whole STORES (6 stores: wide, a rough guide). Per-product numbers are shown as counts; percentages
only for products with at least 10 crops (`coca_light` has 3).

Also: the database search is checked against the offline numpy search; the plain catalog (30 originals) against the augmented one
(paired bootstrap by store); CLS against mean pooling; the unknown rejection with and without store s03; the domain gap.

Usage: python scripts/evaluate_identification.py
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.common.device import get_device  # noqa: E402
from src.data.images import load_image  # noqa: E402
from src.data.paths import check_output_dir  # noqa: E402
from src.embeddings import catalog_db  # noqa: E402
from src.embeddings.embed import embed, load_dinov2  # noqa: E402
from src.embeddings.id_metrics import confusion, outcome_counts, summarize  # noqa: E402
from src.embeddings.identify import AUTO, REVIEW, UNKNOWN, cosine_matrix, features, leave_one_store_out  # noqa: E402
from src.evaluation.detection import bootstrap_by_store, difference_metric  # noqa: E402

NOTE = "DEVELOPMENT numbers on 180 crops from 6 stores (leave-one-store-out), not accuracy"
KEYS = [("top1_acc", "Top-1 (known crops)"), ("top3_acc", "Top-3 (known crops)"), ("brand_acc", "Brand accuracy (known crops)"),
        ("auto_correct_rate", "Known crops answered automatically and correctly"), ("unknown_rejection", "Unknown rejection (unknown or review)"),
        ("unknown_strict_rejection", "Unknown rejection (strictly unknown)"), ("false_known", "False known rate (unknown answered automatically)"),
        ("confident_wrong_rate", "Confident wrong rate (among automatic answers)"), ("share_automatic", "Share automatic"),
        ("share_review", "Share review"), ("share_unknown", "Share unknown"), ("utility", "Mean utility")]


def pct(v):
    return "n/a" if v != v else f"{100 * v:.1f}"


def fmt(v, ci=None):
    if ci is None or ci[0] != ci[0]:
        return pct(v)
    return f"{pct(v)} [{pct(ci[0])}, {pct(ci[1])}]"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage6.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load((REPO_ROOT / args.config).read_text())
    s1 = yaml.safe_load((REPO_ROOT / "configs/stage1.yaml").read_text())
    protected = [REPO_ROOT / d for d in s1["protected_dirs"]]
    products, ec, idc = cfg["products"], cfg["embeddings"], cfg["identify"]
    brand_of = cfg["brands"]
    print(f"ALL NUMBERS BELOW ARE {NOTE.upper()}.\n")

    # ---- query crops: the labeled val sample, cut from the original photos with the same 10% margin as the catalog crops
    labels = list(csv.DictReader((REPO_ROOT / cfg["labels_export"]["output_dir"] / "val_sample_labels.csv").open()))
    mapping = {r["crop_id"]: r for r in csv.DictReader((REPO_ROOT / cfg["crops"]["mapping_dir"] / cfg["crops"]["sample"]["name"] / "mapping.csv").open())}
    names = {r["photo_id"]: r["file_name"] for r in csv.DictReader((REPO_ROOT / s1["id_maps_dir"] / "photo_ids.csv").open())}
    crop_ids = np.array([r["crop_id"] for r in labels])
    truth = np.array([r["label"] for r in labels], dtype=object)
    stores = np.array([r["store_code"] for r in labels])
    cache = REPO_ROOT / cfg["crops"]["mapping_dir"] / cfg["crops"]["sample"]["name"] / "embeddings.npz"
    if cache.exists() and set(np.load(cache)["crop_ids"]) == set(crop_ids):
        z = np.load(cache)
        order = {c: i for i, c in enumerate(z["crop_ids"])}
        q = {"cls": z["cls"][[order[c] for c in crop_ids]], "mean": z["mean"][[order[c] for c in crop_ids]]}
    else:
        device = get_device()
        model = load_dinov2(ec, device, REPO_ROOT)
        images = []
        for cid in crop_ids:
            m = mapping[cid]
            images.append(load_image(REPO_ROOT / s1["raw_dir"] / names[m["photo_id"]]).crop(tuple(int(m[k]) for k in ("crop_x1", "crop_y1", "crop_x2", "crop_y2"))))
        q = {"cls": embed(model, images, device, {**ec, "pooling": "cls"}), "mean": embed(model, images, device, {**ec, "pooling": "mean"})}
        check_output_dir(cache.parent, protected)
        np.savez(cache, crop_ids=crop_ids, cls=q["cls"], mean=q["mean"])
    cat = np.load(REPO_ROOT / cfg["catalog_files"]["embeddings_npz"])
    entry_product, entry_photo, is_aug = cat["product"], cat["source_photo"], cat["is_augmented"]
    print(f"{len(crop_ids)} labeled val crops, catalog {len(entry_product)} entries ({int((~is_aug).sum())} originals, {int(is_aug.sum())} augmented)")
    lab = Counter(truth)
    print("labels: " + ", ".join(f"{k} {lab[k]}" for k in products + cfg["other_labels"]))

    use_aug = cfg["catalog"]["use_augmented"]
    main_mask = None if use_aug else ~is_aug          # the catalog configuration that is in the database
    main_name = "augmented catalog (210 entries)" if use_aug else "plain catalog (30 originals)"
    # ---- the database search agrees with the offline search
    db = cfg["database"]
    conn = catalog_db.connect(db, REPO_ROOT, db["name"])
    try:
        bad = 0
        in_db = np.ones(len(entry_product), bool) if main_mask is None else main_mask
        sims = cosine_matrix(q["cls"], cat["cls"][in_db])
        prod_in_db = entry_product[in_db]
        for i in range(len(crop_ids)):
            hit = catalog_db.search(conn, q["cls"][i], 1)[0]
            j = int(np.argmax(sims[i]))
            bad += not (hit["product_id"] == prod_in_db[j] and abs(hit["similarity"] - sims[i, j]) < 1e-3)
        n_rows = conn.run("SELECT count(*) FROM catalog_entries")[0][0]
    finally:
        conn.close()
    print(f"database check: {len(crop_ids)} queries searched in Postgres (soda_catalog, {n_rows} rows) vs the offline search: {bad} disagreement(s)\n")

    store_of = dict(zip(crop_ids, stores))
    index_of = {c: i for i, c in enumerate(crop_ids)}
    bs = cfg["bootstrap"]

    def run(pool: str, mask, exclude=()):
        feat = features(q[pool], cat[pool], entry_product, entry_photo, products, idc["top_k_photos"], mask)
        outcome, used = leave_one_store_out(feat, truth, stores, idc, tuple(exclude))
        return feat, outcome, used

    def metric_fn(feat, outcome):
        top1 = np.array(products, dtype=object)[feat["top1"]]
        top3 = [[products[j] for j in row] for row in feat["top3"]]
        return lambda ids: summarize([index_of[c] for c in ids], truth, outcome, top1, top3, brand_of, products, idc["utility"])

    feat, outcome, used = run("cls", main_mask)
    top1 = np.array(products, dtype=object)[feat["top1"]]
    f_main = metric_fn(feat, outcome)
    all_ids = list(crop_ids)
    point = f_main(all_ids)
    ci = bootstrap_by_store(store_of, f_main, bs["n"], cfg["seed"], bs["level"])
    print(f"### Main result: {main_name}, CLS embeddings, thresholds tuned leaving one store out ({NOTE})")
    print(f"{point['n_known']} crops with a product label, {point['n_unknown']} unknown; 95% interval by store in brackets\n| Metric (%) | Value [95% CI by store] |\n|---|---|")
    for k, name in KEYS[:-1]:
        print(f"| {name} | {fmt(point[k], ci[k])} |")
    print(f"| Mean utility (see config) | {point['utility']:.3f} [{ci['utility'][0]:.3f}, {ci['utility'][1]:.3f}] |")
    print("\nThresholds tuned for each held-out store (on the other 5 stores):\n| Held-out store | Crops | top-1 similarity | margin | vote | tuning utility |\n|---|---:|---:|---:|---:|---:|")
    for u in used:
        print(f"| {u['store']} | {int((stores == u['store']).sum())} | {u['t_sim']:.3f} | {u['t_margin']:.2f} | {u['t_vote']:.1f} | {u['utility']:.3f} |")

    print("\nPrecision against coverage (descriptive, no tuning, no margin rule: answer a crop with its top-1 product when the top-1 similarity is at least t; a truly unknown crop answered with a product counts as wrong):")
    print("| t (top-1 similarity) | Crops answered (known + unknown) | Correct | Precision | Unknown crops answered |\n|---|---:|---:|---:|---:|")
    lab_idx = np.where(np.isin(truth, products + ["unknown"]))[0]
    for tt in (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90):
        sel = [i for i in lab_idx if feat["top1_sim"][i] >= tt]
        ok = sum(1 for i in sel if truth[i] == top1[i])
        print(f"| {tt:.2f} | {len(sel)} of {len(lab_idx)} | {ok} | {pct(ok / len(sel)) if sel else 'n/a'}% | {sum(1 for i in sel if truth[i] == 'unknown')} |")
    mk = np.where(np.isin(truth, products))[0]
    print(f"margin between the best and the second best product on known crops: median {np.median(feat['margin'][mk]):.3f}, 90th percentile {np.percentile(feat['margin'][mk], 90):.3f} (a margin rule of 0.05 leaves almost nothing automatic)")

    print("\nPer product (COUNTS; a percentage only when there are at least 10 crops; `coca_light` has 3):\n| Product | Crops | Top-1 correct | Answered automatic and correct | Review | Rejected as unknown | Automatic and wrong |\n|---|---:|---:|---:|---:|---:|---:|")
    per_product = {}
    for p in products:
        sel = np.where(truth == p)[0]
        n = len(sel)
        top1_ok = int((top1[sel] == p).sum())
        auto_ok = int(((outcome[sel] == AUTO) & (top1[sel] == p)).sum())
        rev, rej = int((outcome[sel] == REVIEW).sum()), int((outcome[sel] == UNKNOWN).sum())
        wrong = int(((outcome[sel] == AUTO) & (top1[sel] != p)).sum())
        per_product[p] = {"n": n, "top1": top1_ok, "auto_correct": auto_ok, "review": rev, "unknown": rej, "auto_wrong": wrong}
        show = lambda k: f"{k} of {n}" + (f" ({100 * k / n:.0f}%)" if n >= 10 else "")  # noqa: E731
        flag = " (**low: fewer than 5 crops, no conclusion**)" if n < 5 else (" (low)" if n < 10 else "")
        print(f"| {p}{flag} | {n} | {show(top1_ok)} | {show(auto_ok)} | {rev} | {rej} | {wrong} |")

    print("\nRetrieval top-1 confusion on crops labeled with a product (rows: truth, columns: nearest product; no rejection rules):")
    rows, cols, mtx = [], [], None
    known = np.where(np.isin(truth, products))[0]
    cm = np.zeros((len(products), len(products)), int)
    for i in known:
        cm[products.index(truth[i]), products.index(top1[i])] += 1
    print("| truth \\ nearest | " + " | ".join(products) + " |\n|---|" + "---:|" * len(products))
    for r, p in enumerate(products):
        print(f"| {p} | " + " | ".join(str(v) for v in cm[r]) + " |")
    rows, cols, mtx = confusion(range(len(truth)), truth, outcome, top1, products)
    print("\nSystem answers (rows: truth; columns: automatic product, review, unknown):")
    print("| truth \\ answer | " + " | ".join(cols) + " |\n|---|" + "---:|" * len(cols))
    for r, name in enumerate(rows):
        print(f"| {name} | " + " | ".join(str(v) for v in mtx[r]) + " |")

    print("\nWhat the system answers for crops that are not identification targets (counts):")
    other = {}
    for label in ("not_a_bottle", "cannot_tell"):
        c = outcome_counts(range(len(truth)), truth, outcome, label)
        other[label] = c
        print(f"- {label} ({c['n']} crops): automatic {c[AUTO]} (a product answered for a non-bottle), review {c[REVIEW]}, unknown {c[UNKNOWN]}")
    nab = np.where(truth == "not_a_bottle")[0]
    print("  not_a_bottle automatic answers by product: " + (", ".join(f"{k} {v}" for k, v in Counter(top1[i] for i in nab if outcome[i] == AUTO).items()) or "none"))

    print("\n### Store s03 sensitivity (s03 holds 30 of the 67 unknown crops)")
    no_s03 = [c for c in crop_ids if store_of[c] != "s03"]
    feat_b, outcome_b, used_b = run("cls", main_mask, exclude=("s03",))
    rows3 = [("all 6 stores (main)", point), ("evaluated without s03 (same thresholds)", f_main(no_s03)),
             ("evaluated without s03, thresholds never tuned on s03", metric_fn(feat_b, outcome_b)(no_s03))]
    print("| Setting | Unknown crops | Unknown rejection (unknown or review) | Strictly unknown | False known | Top-1 (known) |\n|---|---:|---:|---:|---:|---:|")
    for name, m in rows3:
        print(f"| {name} | {m['n_unknown']} | {pct(m['unknown_rejection'])} | {pct(m['unknown_strict_rejection'])} | {pct(m['false_known'])} | {pct(m['top1_acc'])} |")

    print(f"\n### Plain catalog (30 originals) against the augmented catalog (210 entries), paired bootstrap by store ({NOTE})")
    feat_p, outcome_p, used_p = run("cls", ~is_aug)
    f_plain = metric_fn(feat_p, outcome_p)
    feat_a, outcome_a, used_a = run("cls", None)
    f_aug = metric_fn(feat_a, outcome_a)
    pp, pa = f_plain(all_ids), f_aug(all_ids)
    diff = difference_metric(f_aug, f_plain)
    dpoint, dci = diff(all_ids), bootstrap_by_store(store_of, diff, bs["n"], cfg["seed"], bs["level"])
    print("| Metric (%) | Plain | Augmented | Augmented minus plain [95% CI by store] | Excludes zero |\n|---|---:|---:|---|---|")
    for k, name in KEYS:
        lo, hi = dci[k]
        scale, f = (1, lambda v: f"{v:.3f}") if k == "utility" else (100, pct)
        print(f"| {name} | {f(pp[k])} | {f(pa[k])} | {scale * dpoint[k]:+.1f} [{scale * lo:+.1f}, {scale * hi:+.1f}] | {'yes' if lo > 0 or hi < 0 else 'no'} |")

    print(f"\n### CLS against mean pooling of the patch tokens ({main_name}, same protocol; point estimates)")
    feat_m, outcome_m, _ = run("mean", main_mask)
    pm = metric_fn(feat_m, outcome_m)(all_ids)
    print("| Metric (%) | CLS | Mean pooling |\n|---|---:|---:|")
    for k in ("top1_acc", "top3_acc", "brand_acc", "unknown_rejection", "false_known", "confident_wrong_rate"):
        print(f"| {dict(KEYS)[k]} | {pct(point[k])} | {pct(pm[k])} |")

    print("\n### Domain gap: similarity of shelf crops to catalog product photos (originals only), against catalog-to-catalog similarity")
    orig = ~is_aug
    fo = features(q["cls"], cat["cls"], entry_product, entry_photo, products, idc["top_k_photos"], orig)
    k_idx = np.where(np.isin(truth, products))[0]
    correct = np.array([fo["prod_sim"][i, products.index(truth[i])] for i in k_idx])
    wrong = np.array([np.delete(fo["prod_sim"][i], products.index(truth[i])).max() for i in k_idx])
    oc, op, oph = cat["cls"][orig], entry_product[orig], entry_photo[orig]
    cc, cw = [], []
    for i in range(len(oc)):
        s = cosine_matrix(oc, oc)
        same = (op == op[i]) & (np.arange(len(oc)) != i)
        cc.append(s[i, same].max())
        cw.append(s[i, op != op[i]].max())
    q3 = lambda a: f"median {np.median(a):.3f} (quartiles {np.percentile(a, 25):.3f} to {np.percentile(a, 75):.3f})"  # noqa: E731
    print(f"- shelf crop to its CORRECT product ({len(correct)} crops): {q3(correct)}")
    print(f"- shelf crop to the best WRONG product: {q3(wrong)}")
    print(f"- catalog photo to its product's OTHER photos (leave one photo out, {len(cc)} photos): {q3(np.array(cc))}")
    print(f"- catalog photo to the best OTHER product: {q3(np.array(cw))}")
    print(f"- shelf crops where the correct product is the nearest: {int((correct > wrong).sum())} of {len(correct)}")
    unk_sim = fo["top1_sim"][truth == "unknown"]
    print(f"- unknown shelf crops: top-1 similarity {q3(unk_sim)}")

    out = REPO_ROOT / "data/results/stage6b/results.json"
    check_output_dir(out.parent, protected)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"note": NOTE, "main": point, "main_ci": ci, "per_product": per_product, "thresholds": used, "not_a_bottle_cannot_tell": other,
                               "plain": pp, "augmented": pa, "pooling_mean": pm, "db_disagreements": bad}, indent=1, default=float))
    print(f"\nresults written to {out.relative_to(REPO_ROOT)} (gitignored)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
