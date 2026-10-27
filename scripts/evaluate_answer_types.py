"""Two new answer types, measured against the current rules on the same labeled val crops: a calibrated candidate list and the
two-level (brand + variants) answer.

DEVELOPMENT numbers (agreement with the owner's labels on 180 val crops from 6 stores), not accuracy; the test split is not touched.
For each held-out store the thresholds of the current rules, the list distance `delta` (conformal: the list should contain the correct
product for ~90% of known crops) and the brand margin `t_bm` are calibrated on the OTHER stores; predictions are pooled.
Answer for a crop (precedence): unknown (similarity below t_sim) > automatic > candidates (brand + variants when the two-level answer
applies, else a list of at most `answers.max_list` products) > review. Candidates only replace `review`.

Usage: python scripts/evaluate_answer_types.py [--catalog originals|originals+shelf|originals+shelf+negatives|shelf]
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
from src.embeddings.answers import CANDIDATES, calibrate_by_store, candidate_sets, list_metrics, to_candidates, two_level  # noqa: E402
from src.embeddings.classifier import LinearClassifier  # noqa: E402
from src.embeddings.experiments import decide_prob, prob_features, loso, tune_prob  # noqa: E402
from src.embeddings.eval_data import load_originals, load_shelf, load_val_queries  # noqa: E402
from src.embeddings.identify import AUTO, REVIEW, UNKNOWN, features, leave_one_store_out  # noqa: E402
from src.embeddings.shelf_catalog import MODES, assemble  # noqa: E402
from src.evaluation.detection import bootstrap_by_store  # noqa: E402

NOTE = "DEVELOPMENT numbers on 180 val crops from 6 stores (leave-one-store-out), agreement with the owner's labels, not accuracy"


def pct(v):
    return "n/a" if v != v else f"{100 * v:.1f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage6.yaml")
    parser.add_argument("--catalog", default="originals", choices=MODES)
    parser.add_argument("--scorer", default="catalog", choices=["catalog", "classifier"], help="classifier: the saved logistic regression (models/stage6); the lists are then built from its probabilities, no brand answers")
    args = parser.parse_args()
    cfg = yaml.safe_load((REPO_ROOT / args.config).read_text())
    s1 = yaml.safe_load((REPO_ROOT / "configs/stage1.yaml").read_text())
    protected = [REPO_ROOT / d for d in s1["protected_dirs"]]
    products, idc, brand_of, ac, bs, sc = cfg["products"], cfg["identify"], cfg["brands"], cfg["answers"], cfg["bootstrap"], cfg["shelf_round"]["catalog"]
    print(f"ALL NUMBERS BELOW ARE {NOTE.upper()}. " + (f"Catalog: {args.catalog}." if args.scorer == "catalog" else "Scorer: logistic regression on CLS (probabilities).") + "\n")
    val, orig = load_val_queries(cfg, REPO_ROOT), load_originals(cfg, REPO_ROOT)
    truth, stores, ids = val["truth"], val["stores"], list(val["crop_ids"])
    if args.scorer == "classifier":
        model = LinearClassifier.load(REPO_ROOT / cfg["classifier"]["model_dir"] / f"{cfg['classifier']['name']}.npz")
        feat = prob_features(model.predict_proba(val["vec"]), products)
        lc = cfg["experiments"]["logreg"]
        outcome, _ = loso({"p": feat}, truth, stores, products, lambda f, t: tune_prob(f, t, lc["t_grid"], idc["utility"]), lambda f, th: decide_prob(f, th["t_unk"], th["t_auto"]))
    else:
        shelf = load_shelf(cfg, REPO_ROOT) if args.catalog != "originals" else None
        cat = assemble(orig, shelf, args.catalog, sc["cap_per_product"], sc["cap_unknown"], cfg["seed"])
        feat = features(val["vec"], cat["vec"], cat["product"], cat["photo"], products, idc["top_k_photos"], balance=sc["balance_vote"] and args.catalog != "originals")
        outcome, _ = leave_one_store_out(feat, truth, stores, idc)
    prod_sim = feat["prod_sim"]
    cal = calibrate_by_store(prod_sim, truth, stores, products, brand_of, ac)

    n = len(truth)
    sets, applies, variants = np.zeros((n, len(products)), bool), np.zeros(n, bool), np.zeros((n, len(products)), bool)
    for s in sorted(set(stores)):
        i = np.where(stores == s)[0]
        sets[i] = candidate_sets(prod_sim[i], cal[s]["delta"])
        applies[i], variants[i] = two_level(prod_sim[i], sets[i], products, brand_of, cal[s]["t_bm"] if args.scorer == "catalog" else float("inf"))
    known = np.isin(truth, products)
    true_idx = np.array([products.index(t) if t in products else 0 for t in truth])
    print("Calibration per held-out store (on the other stores' known crops): " + "; ".join(
        f"{s}: delta {c['delta']:.3f}, t_bm {'never' if not np.isfinite(c['t_bm']) else format(c['t_bm'], '.2f')}" for s, c in cal.items()))

    # ---- lists, on the known crops, whatever the unknown rules say (the list is a ranking device)
    order = np.argsort(-prod_sim, axis=1)
    def top_n(m):
        out = np.zeros_like(sets)
        out[np.arange(n)[:, None], order[:, :m]] = True
        return out
    methods = {"top-1 only": top_n(1), "fixed top-2": top_n(2), "fixed top-3 (what `review` shows today)": top_n(3), f"calibrated list (target {int(100 * ac['coverage'])}%)": sets}
    k_ids = np.where(known)[0]
    print("\n### Candidate list on the known crops (the correct product must be inside the list)")
    print("| Method | Coverage | Mean list size | Size 1 | Size 2 | Size 3 or more | Lists of 2+ inside one brand |\n|---|---:|---:|---:|---:|---:|---:|")
    for name, m in methods.items():
        r = list_metrics(m[k_ids], true_idx[k_ids], products, brand_of)
        print(f"| {name} | {pct(r['coverage'])}% | {r['mean_size']:.2f} | {pct(r['size1'])}% | {pct(r['size2'])}% | {pct(r['size3plus'])}% | {pct(r['multi_one_brand'])}% ({r['n_multi']} lists) |")
    cl = lambda sel: list_metrics(sets[np.asarray(sel, int)], true_idx[np.asarray(sel, int)], products, brand_of)  # noqa: E731
    ci = bootstrap_by_store({i: stores[i] for i in k_ids}, lambda sel: {k: cl(sel)[k] for k in ("coverage", "mean_size", "multi_one_brand")}, bs["n"], cfg["seed"], bs["level"])
    print(f"\ncalibrated list: coverage {pct(cl(k_ids)['coverage'])}% [{pct(ci['coverage'][0])}, {pct(ci['coverage'][1])}] (95% by store), mean size {cl(k_ids)['mean_size']:.2f} [{ci['mean_size'][0]:.2f}, {ci['mean_size'][1]:.2f}]")
    print("List size by true product (known crops; counts, percentages only from 10 crops):")
    for p in products:
        sel = np.where(truth == p)[0]
        r = list_metrics(sets[sel], true_idx[sel], products, brand_of)
        print(f"- {p} ({len(sel)} crops): covered {int(round(r['coverage'] * len(sel)))}, mean list size {r['mean_size']:.2f}")

    # ---- two-level answer
    ap = np.where(applies & known)[0]
    best_brand = np.array([brand_of[products[j]] for j in feat["top1"]])
    true_brand = np.array([brand_of[t] if t in products else "-" for t in truth])
    print("\n### Two-level answer (brand + candidate variants) on the known crops")
    n_auto = int((outcome[known] == AUTO).sum())
    cur_review = known & (outcome == REVIEW)
    print(f"known crops: {int(known.sum())}; answered automatically today: {n_auto}; sent to review today: {int(cur_review.sum())}")
    print(f"the two-level answer applies to {len(ap)} known crops ({pct(len(ap) / known.sum())}%); of these, {int((outcome[ap] == REVIEW).sum())} are review today, {int((outcome[ap] == AUTO).sum())} automatic, {int((outcome[ap] == UNKNOWN).sum())} rejected as unknown today")
    if len(ap):
        print(f"brand right: {int((best_brand[ap] == true_brand[ap]).sum())} of {len(ap)} ({pct((best_brand[ap] == true_brand[ap]).mean())}%); correct variant inside the shown variants: {int(variants[ap, true_idx[ap]].sum())} of {len(ap)} ({pct(variants[ap, true_idx[ap]].mean())}%); mean number of variants shown {variants[ap].sum(axis=1).mean():.2f}")
    un = np.where(applies & (truth == "unknown"))[0]
    print(f"unknown crops that would get a brand answer: {len(un)} of {int((truth == 'unknown').sum())}")

    # ---- final answer mix, candidates in place of review
    final, kind = to_candidates(outcome, sets, applies, ac["max_list"])
    cols = [AUTO, CANDIDATES, REVIEW, UNKNOWN]
    print(f"\n### What the system answers with candidates in place of review (max list size {ac['max_list']}); counts")
    print("| Truth | Crops | automatic | candidates (brand) | candidates (list) | review | unknown | Correct product inside the candidates |\n|---|---:|---:|---:|---:|---:|---:|---:|")
    groups = [("a target product", known), ("unknown", truth == "unknown"), ("not_a_bottle", truth == "not_a_bottle"), ("cannot_tell", truth == "cannot_tell")]
    summary = {}
    for name, mask in groups:
        sel = np.where(mask)[0]
        c = Counter(final[sel])
        cb, cl_ = int(((final == CANDIDATES) & (kind == "brand"))[sel].sum()), int(((final == CANDIDATES) & (kind == "list"))[sel].sum())
        inside = ""
        if name == "a target product":
            cs = sel[final[sel] == CANDIDATES]
            ok = sum(1 for i in cs if (variants[i] if kind[i] == "brand" else sets[i])[true_idx[i]])
            inside = f"{ok} of {len(cs)}"
        print(f"| {name} | {len(sel)} | {c[AUTO]} | {cb} | {cl_} | {c[REVIEW]} | {c[UNKNOWN]} | {inside} |")
        summary[name] = {"n": len(sel), **{k: int(c[k]) for k in cols}, "brand": cb, "list": cl_}
    out = REPO_ROOT / "data/results/stage6b_answers" / (f"results_{args.catalog.replace('+', '_')}.json" if args.scorer == "catalog" else "results_classifier.json")
    check_output_dir(out.parent, protected)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"note": NOTE, "catalog": args.catalog, "scorer": args.scorer, "calibration": cal, "lists": {k: list_metrics(m[k_ids], true_idx[k_ids], products, brand_of) for k, m in methods.items()}, "mix": summary}, indent=1, default=float))
    print(f"\nresults written to {out.relative_to(REPO_ROOT)} (gitignored)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
