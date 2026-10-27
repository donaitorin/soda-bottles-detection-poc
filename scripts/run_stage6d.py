"""Stage 6d: three experiments on separating the Coca-Cola variants, against catalog (b) as the reference.

DEVELOPMENT numbers (agreement with the owner's labels on 180 val crops from 6 stores; thresholds and weights tuned LEAVING ONE STORE
OUT; paired bootstrap by store), not accuracy; the test split is not touched. The keep-rules are in reports/stage_06d.md, written before
any number was computed, and are applied mechanically here.
  4a colour    HSV histogram of the central part of the crop: alone (diagnostic), then combined with the embedding score
  4b context   10% margin (reference) vs 0% margin vs the SAM 2 mask on a white background, for catalog and query crops alike
  4c logreg    logistic regression on the 253 train shelf entries (CLS alone; CLS + colour)

Usage: python scripts/run_stage6d.py [--exp all|colour|context|logreg]
"""
import argparse
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
from src.data.paths import check_output_dir  # noqa: E402
from src.embeddings.crop_records import build_records, render  # noqa: E402
from src.embeddings.embed import embed, load_dinov2  # noqa: E402
from src.embeddings.eval_data import load_originals, load_shelf, load_val_queries  # noqa: E402
from src.embeddings.experiments import (COCA, SoftmaxRegression, balanced_accuracy, coca_confusion, colour_descriptor, combine_sims, decide_prob,  # noqa: E402
                                        full_metrics, loso, prob_features, top1_on_known, tune_prob)
from src.embeddings.id_metrics import outcome_counts  # noqa: E402
from src.embeddings.identify import AUTO, REVIEW, UNKNOWN, cosine_matrix, decide, features_from_sims, tune  # noqa: E402
from src.embeddings.sam_masks import load_sam2  # noqa: E402
from src.embeddings.shelf_catalog import assemble  # noqa: E402
from src.evaluation.detection import bootstrap_by_store, difference_metric  # noqa: E402

NOTE = "DEVELOPMENT numbers on 180 val crops from 6 stores (leave-one-store-out), agreement with the owner's labels, not accuracy"
ROWS = [("coca_top1", "Coca variant top-1 (60 Coca crops)"), ("top1_acc", "Top-1 (86 known crops)"), ("top3_acc", "Top-3 (known)"), ("brand_acc", "Brand (known)"),
        ("auto_correct_rate", "Known crops answered automatically and correctly"), ("confident_wrong_rate", "Confident wrong rate (among automatic)"), ("unknown_strict_rejection", "Strictly unknown rejection"), ("false_known", "False known rate"), ("share_automatic", "Share automatic"), ("utility", "Mean utility")]


def pct(v):
    return "n/a" if v != v else f"{100 * v:.1f}"


class Ctx:
    """Everything shared by the experiments: data, models (loaded lazily), cached embeddings."""

    def __init__(self, cfg, s1):
        self.cfg, self.s1 = cfg, s1
        self.products, self.brand_of, self.idc = cfg["products"], cfg["brands"], cfg["identify"]
        self.sc, self.bs = cfg["shelf_round"]["catalog"], cfg["bootstrap"]
        self.val, self.orig, self.shelf = load_val_queries(cfg, REPO_ROOT), load_originals(cfg, REPO_ROOT), load_shelf(cfg, REPO_ROOT)
        self.truth, self.stores = self.val["truth"], self.val["stores"]
        self.ids = list(self.val["crop_ids"])
        self.index_of = {c: i for i, c in enumerate(self.ids)}
        self.store_of = dict(zip(self.ids, self.stores))
        self.records = build_records(cfg, s1, REPO_ROOT, self.val["crop_ids"], self.shelf["photo"], self.orig["photo"])
        self.rdir = REPO_ROOT / cfg["experiments"]["results_dir"]
        check_output_dir(self.rdir, [REPO_ROOT / d for d in s1["protected_dirs"]])
        self.rdir.mkdir(parents=True, exist_ok=True)
        self.device, self._dino, self._sam, self._cache, self._images = get_device(), None, None, {}, {}

    def dino(self):
        if self._dino is None:
            self._dino = load_dinov2(self.cfg["embeddings"], self.device, REPO_ROOT)
        return self._dino

    def images(self, variant):
        """Rendered crop images of every record for a cropping variant (kept in memory for the colour descriptor)."""
        if variant not in self._images:
            sam = None
            if variant == "sam":
                if self._sam is None:
                    self._sam = load_sam2(self.cfg["experiments"]["sam2"], self.device, REPO_ROOT)
                sam = self._sam
            self._images[variant] = {k: render(self.records[k], variant, self.cfg["crops"]["margin"], sam, self.device) for k in ("val", "shelf", "orig")}
        return self._images[variant]

    def vectors(self, variant):
        """CLS embeddings {val, shelf, orig} for a cropping variant (cached on disk)."""
        if variant in self._cache:
            return self._cache[variant]
        path = self.rdir / f"emb_{variant}.npz"
        if path.exists():
            z = np.load(path, allow_pickle=False)
            out = {k: z[k] for k in ("val", "shelf", "orig")}
        else:
            ims = self.images(variant)
            ec = {**self.cfg["embeddings"], "pooling": "cls"}
            out = {k: embed(self.dino(), ims[k], self.device, ec) for k in ims}
            np.savez(path, **out)
        self._cache[variant] = out
        return out

    def colour(self):
        path = self.rdir / "colour.npz"
        if path.exists():
            z = np.load(path)
            return {k: z[k] for k in ("val", "shelf", "orig")}
        c = self.cfg["experiments"]["colour"]
        ims = self.images("margin10")
        out = {k: np.stack([colour_descriptor(im, c["centre"], c["bins"]) for im in ims[k]]) for k in ims}
        np.savez(path, **out)
        return out

    def catalog(self, vec_orig, vec_shelf):
        """Catalog (b) arrays for given vectors of the originals and the shelf entries (the subset depends on labels and ids only)."""
        o = {"vec": vec_orig, "product": self.orig["product"], "photo": self.orig["photo"]}
        s = {"vec": vec_shelf, "label": self.shelf["label"], "photo": self.shelf["photo"]}
        return assemble(o, s, "originals+shelf+negatives", self.sc["cap_per_product"], self.sc["cap_unknown"], self.cfg["seed"])

    # ---- one result = features + outcomes + metric function
    def result(self, name, feat, outcome, used):
        top1 = np.array(self.products, dtype=object)[feat["top1"]]
        top3 = [[self.products[j] for j in row] for row in feat["top3"]]
        fn = lambda sel: full_metrics([self.index_of[c] for c in sel], self.truth, outcome, top1, top3, self.brand_of, self.products, self.idc["utility"])  # noqa: E731
        return {"name": name, "feat": feat, "outcome": outcome, "top1": top1, "used": used, "fn": fn, "point": fn(self.ids)}

    def retrieval(self, name, sims_fn, cat, weights=None, q_col=None, c_col=None):
        """Retrieval result with the Stage 6b rules. `weights`: colour weights to choose from leaving one store out."""
        def feat_of(w):
            sims = cosine_matrix(sims_fn[0], sims_fn[1]) if w is None else combine_sims(cosine_matrix(sims_fn[0], sims_fn[1]), cosine_matrix(q_col, c_col), w)
            return features_from_sims(sims, cat["product"], cat["photo"], self.products, self.idc["top_k_photos"], balance=self.sc["balance_vote"])
        options = {"w=none": feat_of(None)} if weights is None else {f"w={w}": feat_of(w) for w in weights}
        choose = (lambda opts, idx: max(opts, key=lambda k: (round(top1_on_known(opts[k], self.truth, idx), 6), -float(k[2:])))) if weights else None
        tune_fn = lambda f, t: tune(f, t, self.idc)  # noqa: E731
        decide_fn = lambda f, th: decide(f, th["t_sim"], th["t_margin"], th["t_vote"], th["t_neg"])  # noqa: E731
        outcome, used = loso(options, self.truth, self.stores, self.products, tune_fn, decide_fn, choose)
        chosen = Counter(u["option"] for u in used)
        feat = options[chosen.most_common(1)[0][0]]
        if weights:   # features of the option each store used, so that top-1 / top-3 follow the same choice
            feat = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in feat.items()}
            for u in used:
                idx = np.where(self.stores == u["store"])[0]
                for k, v in options[u["option"]].items():
                    if isinstance(v, np.ndarray):
                        feat[k][idx] = v[idx]
        return self.result(name, feat, outcome, used)

    # ---- reporting
    def compare(self, ref, other, title, rule=True):
        diff = difference_metric(other["fn"], ref["fn"])
        dpoint, dci = diff(self.ids), bootstrap_by_store(self.store_of, diff, self.bs["n"], self.cfg["seed"], self.bs["level"])
        print(f"\n### {title}\n| Metric (%) | {ref['name']} | {other['name']} | difference [95% CI by store] | Excludes zero |\n|---|---:|---:|---|---|")
        for k, label in ROWS:
            lo, hi = dci[k]
            scale, f = (1, lambda v: f"{v:.3f}") if k == "utility" else (100, pct)
            print(f"| {label} | {f(ref['point'][k])} | {f(other['point'][k])} | {scale * dpoint[k]:+.1f} [{scale * lo:+.1f}, {scale * hi:+.1f}] | {'yes' if lo > 0 or hi < 0 else 'no'} |")
        print(f"Diagnostic (threshold-free, NOT in the keep-rule): AUROC of the top-1 score for telling known from unknown crops: {self.auroc(ref):.3f} -> {self.auroc(other):.3f} (0.5 = no separation; the utility above also moves with the thresholds the other stores happened to give)")
        for r in (ref, other):
            c = {lab: outcome_counts(range(len(self.truth)), self.truth, r["outcome"], lab) for lab in ("not_a_bottle", "cannot_tell")}
            print(f"Non-targets answered by {r['name']}: " + "; ".join(f"{lab} ({v['n']}): automatic {v[AUTO]}, review {v[REVIEW]}, unknown {v[UNKNOWN]}" for lab, v in c.items()))
        print(f"Automatic answers: {r_auto(ref, self)} -> {r_auto(other, self)} (labeled crops: automatic and correct / automatic in total)")
        self.counts(other, ref)
        verdict = None
        if rule:
            c, lo, hi = 100 * dpoint["coca_top1"], dci["coca_top1"][0], dci["coca_top1"][1]
            checks = {"1 Coca top-1 >= +5 points": c >= 5, "2 top-1 (known) not down": dpoint["top1_acc"] >= 0,
                      "3 utility not lower by more than 0.02": dpoint["utility"] >= -0.02, "3 false known not up by more than 2 points": dpoint["false_known"] <= 0.02,
                      "4 Coca top-1 interval not entirely below zero": hi >= 0}
            kept = all(checks.values())
            winner = kept and lo > 0
            verdict = {"kept": kept, "clear_winner": winner, "checks": checks, "coca_diff": c, "coca_ci": (100 * lo, 100 * hi)}
            print(f"Keep-rule: " + "; ".join(f"{k}: {'ok' if v else 'FAIL'}" for k, v in checks.items()) + f" -> **{'KEPT' + (' (clear winner)' if winner else ' (promising, not established)') if kept else 'NOT kept'}**")
        return {"dpoint": dpoint, "dci": dci, "verdict": verdict}

    def auroc(self, r):
        """Probability that a known crop has a higher top-1 score than an unknown crop (ties count half), on the labeled val crops."""
        k, u = r["feat"]["top1_sim"][np.isin(self.truth, self.products)], r["feat"]["top1_sim"][self.truth == "unknown"]
        return float(((k[:, None] > u[None, :]).sum() + 0.5 * (k[:, None] == u[None, :]).sum()) / (len(k) * len(u)))

    def counts(self, other, ref):
        print("Per product, COUNTS of top-1 correct (known val crops; a percentage only from 10 crops; `coca_light` has 3):")
        print("| Product | Crops | " + ref["name"] + " | " + other["name"] + " |\n|---|---:|---:|---:|")
        for p in self.products:
            sel = np.where(self.truth == p)[0]
            print(f"| {p} | {len(sel)} | {int((ref['top1'][sel] == p).sum())} | {int((other['top1'][sel] == p).sum())} |")
        for r in (ref, other):
            self.confusion(r)

    def confusion(self, r):
        cols, m = coca_confusion(self.truth, r["top1"], self.products)
        print(f"Coca confusion, {r['name']} (rows: truth; columns: nearest product, no rejection rules):")
        print("| truth \\ nearest | " + " | ".join(cols) + " |\n|---|" + "---:|" * len(cols))
        for i, t in enumerate(COCA):
            print(f"| {t} | " + " | ".join(str(v) for v in m[i]) + " |")


def r_auto(r, x):
    lab = np.isin(x.truth, x.products + ["unknown"])
    a = (r["outcome"] == AUTO) & lab
    return f"{int((a & (r['top1'] == x.truth)).sum())} / {int(a.sum())}"


def four_way(prod_sim, truth, products):
    """Accuracy of picking the right variant among the 4 Coca products only, on the Coca crops (counts)."""
    cols = [products.index(p) for p in COCA]
    sel = np.where(np.isin(truth, COCA))[0]
    pick = np.array(COCA, dtype=object)[prod_sim[sel][:, cols].argmax(axis=1)]
    return int((pick == truth[sel]).sum()), len(sel), pick


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage6.yaml")
    parser.add_argument("--exp", default="all", choices=["all", "colour", "context", "logreg"])
    args = parser.parse_args()
    cfg = yaml.safe_load((REPO_ROOT / args.config).read_text())
    s1 = yaml.safe_load((REPO_ROOT / "configs/stage1.yaml").read_text())
    x = Ctx(cfg, s1)
    ex = cfg["experiments"]
    print(f"ALL NUMBERS BELOW ARE {NOTE.upper()}.\n")
    results, out_json = {}, {"note": NOTE}

    # ---- reference: catalog (b), CLS, 10% margin
    v10 = x.vectors("margin10")
    chk = max(float(np.abs(v10["val"] - x.val["vec"]).max()), float(np.abs(v10["shelf"] - x.shelf["vec"]).max()), float(np.abs(v10["orig"] - x.orig["vec"]).max()))
    print(f"check: re-cut and re-embedded 10% margin crops differ from the cached Stage 6 embeddings by at most {chk:.1e}")
    cat10 = x.catalog(v10["orig"], v10["shelf"])
    ref = x.retrieval("reference (b)", (v10["val"], cat10["vec"]), cat10)
    print(f"Reference: catalog (b) {len(cat10['product'])} entries, CLS, 10% margin: Coca variant top-1 {pct(ref['point']['coca_top1'])}, top-1 {pct(ref['point']['top1_acc'])}, utility {ref['point']['utility']:.3f}")
    x.confusion(ref)
    results["reference"] = ref

    # ---- 4a colour
    if args.exp in ("all", "colour", "logreg"):
        col = x.colour()
        ccat = x.catalog(col["orig"], col["shelf"])
        if args.exp in ("all", "colour"):
            print("\n## 4a Colour")
            alone = x.retrieval("colour alone", (col["val"], ccat["vec"]), ccat)
            for name, r in (("embedding (reference)", ref), ("colour descriptor alone", alone)):
                ok, n, _ = four_way(r["feat"]["prod_sim"], x.truth, x.products)
                print(f"- {name}: right variant among the 4 Coca products for {ok} of {n} Coca crops ({pct(ok / n)}%); 6-way top-1 over the 86 known crops {pct(r['point']['top1_acc'])}%")
            n_zero = int((x.truth == "coca_zero").sum())
            print(f"  (the majority variant alone, always `coca_zero`, would be {n_zero} of 60)")
            x.confusion(alone)
            comb = x.retrieval("embedding + colour", (v10["val"], cat10["vec"]), cat10, weights=ex["colour"]["weights"], q_col=col["val"], c_col=ccat["vec"])
            print("Weights chosen leaving one store out: " + ", ".join(f"{u['store']}: {u['option']}" for u in comb["used"]))
            out_json["colour"] = x.compare(ref, comb, "4a step 2: embedding + colour vs the reference")
            results["colour"] = comb

    # ---- 4b context
    if args.exp in ("all", "context"):
        print("\n## 4b Context")
        for variant, label in (("margin0", "0% margin"), ("sam", "SAM 2 mask on white")):
            v = x.vectors(variant)
            cat = x.catalog(v["orig"], v["shelf"])
            r = x.retrieval(label, (v["val"], cat["vec"]), cat)
            out_json[variant] = x.compare(ref, r, f"4b: {label} (catalog and query crops cut the same way) vs the reference (10% margin)")
            results[variant] = r
        sheet_dir = x.rdir.parent.parent / "review" / "stage6d_sam"
        check_output_dir(sheet_dir, [REPO_ROOT / d for d in s1["protected_dirs"]])
        sheet_dir.mkdir(parents=True, exist_ok=True)
        from src.embeddings.sheets import contact_sheet
        pick = list(range(0, 180, 9))
        ims = x.images("sam")["val"]
        contact_sheet([ims[i] for i in pick], [x.ids[i] + " " + str(x.truth[i]) for i in pick], "SAM 2 masked val crops (every 9th), white background", 5, (300, 420), 18).save(sheet_dir / "val_sam_sample.jpg", quality=88)
        print(f"sample of SAM 2 masked crops: {sheet_dir.relative_to(REPO_ROOT)}/val_sam_sample.jpg (gitignored)")

    # ---- 4c logistic regression
    if args.exp in ("all", "logreg"):
        print("\n## 4c Logistic regression on the 253 train shelf entries")
        lc = ex["logreg"]
        col = x.colour()
        classes = x.products + ["unknown"]
        y = np.array([classes.index(l) for l in x.shelf["label"]])
        s_store = x.shelf["store"]
        vs = x.vectors("sam")
        for label, xtr, xva in (("CLS", v10["shelf"], v10["val"]), ("CLS + colour", np.hstack([v10["shelf"], col["shelf"]]), np.hstack([v10["val"], col["val"]])),
                                ("CLS on SAM crops (exploratory)", vs["shelf"], vs["val"])):
            scores = {}
            for c in lc["c_grid"]:
                accs = []
                for s in sorted(set(s_store)):
                    tr, te = s_store != s, s_store == s
                    m = SoftmaxRegression(c).fit(xtr[tr], y[tr], len(classes))
                    accs.append((m.predict_proba(xtr[te]).argmax(axis=1) == y[te]).mean())
                scores[c] = float(np.mean(accs))
            best_c = max(scores, key=lambda c: (round(scores[c], 6), -c))
            print(f"{label}: leave-one-TRAIN-store-out accuracy on the shelf entries by C: " + ", ".join(f"{c}: {100 * a:.1f}%" for c, a in scores.items()) + f" -> C = {best_c}")
            model = SoftmaxRegression(best_c).fit(xtr, y, len(classes))
            feat = prob_features(model.predict_proba(xva), x.products)
            tune_fn = lambda f, t: tune_prob(f, t, lc["t_grid"], x.idc["utility"])  # noqa: E731
            outcome, used = loso({"p": feat}, x.truth, x.stores, x.products, tune_fn, lambda f, th: decide_prob(f, th["t_unk"], th["t_auto"]))
            r = x.result(f"logreg {label}", feat, outcome, used)
            if label == "CLS":   # control: the same pipeline with the shelf labels shuffled must fall to chance, or something leaks
                rng = np.random.default_rng(cfg["seed"])
                shuf = SoftmaxRegression(best_c).fit(xtr, rng.permutation(y), len(classes))
                sf = prob_features(shuf.predict_proba(xva), x.products)
                known = np.isin(x.truth, x.products)
                chance = float((np.array(x.products, dtype=object)[sf["top1"]][known] == x.truth[known]).mean())
                print(f"Control: same model with the SHELF LABELS SHUFFLED: top-1 on the 86 known val crops {pct(chance)}% (chance for 6 products is about 17%; the majority class is {pct(max((x.truth[known] == p).mean() for p in x.products))}%)")
            if "exploratory" in label:
                out_json[f"logreg_{label}"] = x.compare(results["logreg_CLS"], r, "4c EXPLORATORY (SAM 2 crops failed its own keep-rule in 4b, so this is not a combination step): logistic regression on SAM 2 crops vs logistic regression on CLS")
            else:
                out_json[f"logreg_{label}"] = x.compare(ref, r, f"4c: logistic regression on {label} vs the reference")
            results[f"logreg_{label}"] = r
        if "logreg_CLS" in results and "logreg_CLS + colour" in results:
            out_json["logreg_colour_step"] = x.compare(results["logreg_CLS"], results["logreg_CLS + colour"], "4c: adding colour to the logistic regression (against logreg CLS)")

    path = x.rdir / "results.json"
    path.write_text(json.dumps(out_json, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o)))
    print(f"\nresults written to {path.relative_to(REPO_ROOT)} (gitignored)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
