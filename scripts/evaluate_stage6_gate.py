"""Stage 6 gate, with the ADOPTED scorer (logistic regression on the CLS embedding): the formal development numbers.

DEVELOPMENT numbers on the 180 labeled val crops from 6 stores, not accuracy: the sample also tunes the unknown / automatic probability
thresholds, so they are tuned LEAVING ONE STORE OUT (on the other 5 stores, applied to the held-out one) and the predictions are pooled;
intervals are bootstraps that resample whole STORES (6 stores: wide). The model itself is trained on train shelf entries only (no val
label was used to train it). Only crops labeled with a product or `unknown` count for accuracy; `not_a_bottle` and `cannot_tell` are
reported separately. Gate criteria as written in reports/stage_06d.md section 7 (declared before computing): brand accuracy >= 90%, top-3 >= 80%,
false known rate < 5% (point estimates), unknown rejection measured with and without store s03, non-targets reported, confusions named.

Usage: python scripts/evaluate_stage6_gate.py
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.paths import check_output_dir  # noqa: E402
from src.embeddings.classifier import LinearClassifier  # noqa: E402
from src.embeddings.eval_data import load_val_queries  # noqa: E402
from src.embeddings.experiments import decide_prob, full_metrics, loso, prob_features, tune_prob  # noqa: E402
from src.embeddings.id_metrics import confusion, outcome_counts  # noqa: E402
from src.embeddings.identify import AUTO, REVIEW, UNKNOWN, subset  # noqa: E402
from src.evaluation.detection import bootstrap_by_store  # noqa: E402

NOTE = "DEVELOPMENT numbers on 180 val crops from 6 stores (leave-one-store-out), agreement with the owner's labels, not accuracy"
KEYS = [("top1_acc", "Top-1 (known crops)"), ("top3_acc", "Top-3 (known crops)"), ("brand_acc", "Brand accuracy (known crops)"),
        ("coca_top1", "Coca variant top-1 (60 Coca crops)"), ("auto_correct_rate", "Known crops answered automatically and correctly"),
        ("unknown_rejection", "Unknown rejection (unknown or review)"), ("unknown_strict_rejection", "Unknown rejection (strictly unknown)"),
        ("false_known", "False known rate (unknown answered automatically)"), ("confident_wrong_rate", "Confident wrong rate (among automatic)"),
        ("share_automatic", "Share automatic"), ("share_review", "Share review"), ("share_unknown", "Share unknown")]


def pct(v):
    return "n/a" if v != v else f"{100 * v:.1f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage6.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load((REPO_ROOT / args.config).read_text())
    s1 = yaml.safe_load((REPO_ROOT / "configs/stage1.yaml").read_text())
    products, idc, brand_of, bs, cc = cfg["products"], cfg["identify"], cfg["brands"], cfg["bootstrap"], cfg["classifier"]
    lc = cfg["experiments"]["logreg"]
    print(f"ALL NUMBERS BELOW ARE {NOTE.upper()}.\n")
    val = load_val_queries(cfg, REPO_ROOT)
    truth, stores, ids = val["truth"], val["stores"], list(val["crop_ids"])
    store_of, index_of = dict(zip(ids, stores)), {c: i for i, c in enumerate(ids)}
    card = json.loads((REPO_ROOT / cc["model_dir"] / f"{cc['name']}_card.json").read_text())
    model = LinearClassifier.load(REPO_ROOT / cc["model_dir"] / f"{cc['name']}.npz")
    proba = model.predict_proba(val["vec"])
    feat = prob_features(proba, products)
    tune_fn = lambda f, t: tune_prob(f, t, lc["t_grid"], idc["utility"])  # noqa: E731
    decide_fn = lambda f, th: decide_prob(f, th["t_unk"], th["t_auto"])  # noqa: E731
    outcome, used = loso({"p": feat}, truth, stores, products, tune_fn, decide_fn)
    top1 = np.array(products, dtype=object)[feat["top1"]]
    top3 = [[products[j] for j in row] for row in feat["top3"]]
    fn = lambda sel: full_metrics([index_of[c] for c in sel], truth, outcome, top1, top3, brand_of, products, idc["utility"])  # noqa: E731
    point, ci = fn(ids), bootstrap_by_store(store_of, fn, bs["n"], cfg["seed"], bs["level"])
    print(f"Model: logistic regression on CLS, trained on {card['training_data']['n_crops']} train shelf entries (C = {card['settings']['C']}, card dated {card['date_utc']}); "
          f"labels file hash {card['training_data']['labels_file_sha256'][:12]}")
    print(f"\n### Main result ({NOTE})\n{point['n_known']} crops with a product label, {point['n_unknown']} unknown; 95% interval by store in brackets\n| Metric (%) | Value [95% CI by store] |\n|---|---|")
    for k, name in KEYS:
        print(f"| {name} | {pct(point[k])} [{pct(ci[k][0])}, {pct(ci[k][1])}] |" if ci[k][0] == ci[k][0] else f"| {name} | {pct(point[k])} |")
    print(f"| Mean utility (see config) | {point['utility']:.3f} [{ci['utility'][0]:.3f}, {ci['utility'][1]:.3f}] |")
    print("\nThresholds tuned for each held-out store (on the other 5 stores):\n| Held-out store | Crops | unknown below (probability) | automatic from (probability) | tuning utility |\n|---|---:|---:|---:|---:|")
    for u in used:
        print(f"| {u['store']} | {int((stores == u['store']).sum())} | {u['t_unk']:.2f} | {u['t_auto']:.2f} | {u['utility']:.3f} |")

    print("\n### Gate criteria (initial targets of docs/PLAN.md; a target is met only by its point estimate)")
    crit = [("Brand accuracy >= 90%", point["brand_acc"] >= 0.90, f"{pct(point['brand_acc'])}% [{pct(ci['brand_acc'][0])}, {pct(ci['brand_acc'][1])}]"),
            ("Top-3 >= 80%", point["top3_acc"] >= 0.80, f"{pct(point['top3_acc'])}% [{pct(ci['top3_acc'][0])}, {pct(ci['top3_acc'][1])}]"),
            ("False known rate < 5%", point["false_known"] < 0.05, f"{pct(point['false_known'])}% [{pct(ci['false_known'][0])}, {pct(ci['false_known'][1])}]")]
    print("| Criterion | Met | Value [95% CI by store] |\n|---|---|---|")
    for name, ok, val_s in crit:
        print(f"| {name} | {'yes' if ok else '**no**'} | {val_s} |")

    print("\n### Per product (COUNTS; a percentage only from 10 crops; `coca_light` has 3 crops)\n| Product | Crops | Top-1 correct | Answered automatic and correct | Review | Rejected as unknown | Automatic and wrong |\n|---|---:|---:|---:|---:|---:|---:|")
    per = {}
    for p in products:
        sel = np.where(truth == p)[0]
        n = len(sel)
        t1, ac = int((top1[sel] == p).sum()), int(((outcome[sel] == AUTO) & (top1[sel] == p)).sum())
        rv, rj, aw = int((outcome[sel] == REVIEW).sum()), int((outcome[sel] == UNKNOWN).sum()), int(((outcome[sel] == AUTO) & (top1[sel] != p)).sum())
        per[p] = dict(n=n, top1=t1, auto_correct=ac, review=rv, unknown=rj, auto_wrong=aw)
        show = lambda k: f"{k} of {n}" + (f" ({100 * k / n:.0f}%)" if n >= 10 else "")  # noqa: E731
        print(f"| {p}{' (counts only)' if n < 10 else ''} | {n} | {show(t1)} | {show(ac)} | {rv} | {rj} | {aw} |")

    known = np.where(np.isin(truth, products))[0]
    cm = np.zeros((len(products), len(products)), int)
    for i in known:
        cm[products.index(truth[i]), products.index(top1[i])] += 1
    print("\n### Confusion between products: most probable PRODUCT (no rejection rules), counts; rows truth, columns predicted\n| truth \\ predicted | " + " | ".join(products) + " |\n|---|" + "---:|" * len(products))
    for r, p in enumerate(products):
        print(f"| {p} | " + " | ".join(str(v) for v in cm[r]) + " |")
    pairs = sorted(((cm[r, c], products[r], products[c]) for r in range(len(products)) for c in range(len(products)) if r != c and cm[r, c] > 0), reverse=True)
    print("Named confusions, most frequent first: " + "; ".join(f"`{a}` taken for `{b}` {n}x" for n, a, b in pairs[:8]))
    rows, cols, mtx = confusion(range(len(truth)), truth, outcome, top1, products)
    print("\nSystem answers (rows: truth; columns: automatic product, review, unknown; candidates are a way of showing review):\n| truth \\ answer | " + " | ".join(cols) + " |\n|---|" + "---:|" * len(cols))
    for r, name in enumerate(rows):
        print(f"| {name} | " + " | ".join(str(v) for v in mtx[r]) + " |")

    print("\n### What the system answers for crops that are not identification targets (counts)")
    other = {}
    for label in ("not_a_bottle", "cannot_tell"):
        c = outcome_counts(range(len(truth)), truth, outcome, label)
        other[label] = c
        print(f"- {label} ({c['n']} crops): automatic {c[AUTO]}, review {c[REVIEW]}, unknown {c[UNKNOWN]}")

    print("\n### Store s03 sensitivity (s03 holds 30 of the 67 unknown crops)")
    labeled = np.isin(truth, products + ["unknown"])
    no3 = [c for c in ids if store_of[c] != "s03"]
    out_b = np.full(len(truth), UNKNOWN, dtype=object)
    for s in sorted(set(stores)):                         # thresholds NEVER tuned on s03
        ti = np.where(labeled & (stores != s) & (stores != "s03"))[0]
        th = tune_fn(subset(feat, ti), truth[ti])
        te = np.where(stores == s)[0]
        out_b[te] = decide_fn(subset(feat, te), th)
    fn_b = lambda sel: full_metrics([index_of[c] for c in sel], truth, out_b, top1, top3, brand_of, products, idc["utility"])  # noqa: E731
    print("| Setting | Unknown crops | Unknown rejection (unknown or review) | Strictly unknown | False known | Top-1 (known) |\n|---|---:|---:|---:|---:|---:|")
    for name, m in (("all 6 stores (main)", point), ("evaluated without s03 (same thresholds)", fn(no3)), ("evaluated without s03, thresholds never tuned on s03", fn_b(no3))):
        print(f"| {name} | {m['n_unknown']} | {pct(m['unknown_rejection'])} | {pct(m['unknown_strict_rejection'])} | {pct(m['false_known'])} | {pct(m['top1_acc'])} |")

    out = REPO_ROOT / "data/results/stage6_gate/results.json"
    check_output_dir(out.parent, [REPO_ROOT / d for d in s1["protected_dirs"]])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"note": NOTE, "main": point, "main_ci": ci, "per_product": per, "thresholds": used, "non_targets": other,
                               "criteria": {n: bool(ok) for n, ok, _ in crit}}, indent=1, default=float))
    print(f"\nresults written to {out.relative_to(REPO_ROOT)} (gitignored)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
