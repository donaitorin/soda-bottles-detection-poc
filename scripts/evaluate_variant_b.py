"""Stage 7 tasks 5 and 6: variant A against variant B (OCR evidence) on the same 180 labeled val crops.

DEVELOPMENT numbers (agreement with the owner's labels, 6 stores), not accuracy; the test split is not touched. A and B are both tuned leaving one
store out (B also chooses its OCR weights that way); the paired bootstrap resamples whole stores. The rules, the grid and the keep-rule are in
configs/stage7.yaml and reports/stage_07_proposal.md, declared before any number. Val crops are read once (cached in data/ocr/stage7/, ids and text only).

Usage: python scripts/evaluate_variant_b.py [--reread]
"""
import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.common.device import get_device  # noqa: E402
from src.data.paths import check_output_dir  # noqa: E402
from src.embeddings.classifier import LinearClassifier  # noqa: E402
from src.embeddings.crop_records import build_records, render  # noqa: E402
from src.embeddings.eval_data import load_val_queries  # noqa: E402
from src.embeddings.experiments import COCA, coca_confusion, full_metrics  # noqa: E402
from src.embeddings.identify import AUTO, REVIEW, UNKNOWN  # noqa: E402
from src.evaluation.detection import bootstrap_by_store, difference_metric  # noqa: E402
from src.ocr.combine import evidence_from_readings  # noqa: E402
from src.ocr.evidence import keywords_detail  # noqa: E402
from src.ocr.reader import OcrReader, OcrUnavailable  # noqa: E402
from src.ocr.variant_b import build_option, choose_and_tune, loso_b, status, weight_grid  # noqa: E402

NOTE = "DEVELOPMENT numbers on 180 val crops from 6 stores (leave-one-store-out), agreement with the owner's labels, not accuracy"
ROWS = [("top1_acc", "Top-1 (known crops)"), ("coca_top1", "Coca variant top-1 (60 Coca crops)"), ("top3_acc", "Top-3 (known crops)"),
        ("auto_correct_rate", "Known answered automatically and correctly"), ("unknown_strict_rejection", "Strictly unknown (unknown crops)"),
        ("false_known", "False known rate"), ("confident_wrong_rate", "Confident wrong rate (among automatic)"),
        ("share_automatic", "Share automatic"), ("share_review", "Share review"), ("share_unknown", "Share unknown"), ("utility", "Mean utility")]


def pct(v):
    return "n/a" if v != v else f"{100 * v:.1f}"


def read_val(cfg6, s1, c7, ids, path):
    records = build_records(cfg6, s1, REPO_ROOT, ids, [], [])["val"]
    images = render(records, "margin10", c7["ocr"]["margin"])
    reader = OcrReader(c7["ocr"], REPO_ROOT, get_device())
    reader.read(images[0])
    out = {}
    for cid, im in zip(ids, images):
        out[str(cid)] = {"height": im.height, "pieces": [[t, round(c, 3)] for t, c in reader.read(im)]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    return out


def fixes_section(title, a, b, truth, products, evidence, heights, floor):
    print(f"\n## {title}")
    sa, sb = status(truth, a["outcome"], a["top1"], products), status(truth, b["outcome"], b["top1"], products)
    lab = np.isin(truth, products + ["unknown"])
    fix, new = lab & (sa == "wrong") & (sb == "right"), lab & (sa == "right") & (sb == "wrong")
    kind = np.array(["no OCR word" if not ev else ("fuzzy" if "fuzzy" in ev.values() else "exact") for ev in evidence], dtype=object)

    def split(mask):
        c = Counter(kind[mask])
        return f"exact {c['exact']}, fuzzy {c['fuzzy']}, crops with no OCR word {c['no OCR word']}"
    print(f"\n### Fixes and new errors, OUTCOME of the {int(lab.sum())} labeled crops (right = automatic with the right product, or unknown for an unknown crop; wrong = automatic with a wrong product, or a known product rejected; review is neutral)")
    print(f"- fixes (wrong in A, right in B): {int(fix.sum())}  ({split(fix)})")
    print(f"- new errors (right in A, wrong in B): {int(new.sum())}  ({split(new)})")
    print(f"- **net {int(fix.sum() - new.sum()):+d}**")
    moves = Counter((sa[i], sb[i]) for i in np.where(lab)[0] if sa[i] != sb[i])
    print("- all status changes (A -> B): " + ", ".join(f"{x}->{y} {n}" for (x, y), n in sorted(moves.items())))
    changed = [i for i in np.where(lab)[0] if sa[i] != sb[i]]
    print(f"- of the {len(changed)} crops whose status changed, {sum(not evidence[i] for i in changed)} have NO OCR word (the change comes from the different thresholds tuned for this setting, not from OCR) and {sum(bool(evidence[i]) for i in changed)} have a word")
    known = np.isin(truth, products)
    f1, n1 = known & (a["top1"] != truth) & (b["top1"] == truth), known & (a["top1"] == truth) & (b["top1"] != truth)
    print(f"- top-1 product on the {int(known.sum())} known crops: fixes {int(f1.sum())} ({split(f1)}), new errors {int(n1.sum())} ({split(n1)}), net {int(f1.sum() - n1.sum()):+d}")
    rules = Counter(r for i in np.where(fix | new)[0] for r in b["infos"][i]["rules"])
    print("- rules that fired on the crops that changed: " + (", ".join(f"{r} {n}" for r, n in rules.items()) or "none"))
    changed_by_which = Counter()
    for i in changed:
        changed_by_which[("fix" if fix[i] else "new error" if new[i] else f"{sa[i]}->{sb[i]}", str(truth[i]), f"A:{a['outcome'][i]}/{a['top1'][i]} -> B:{b['outcome'][i]}/{b['top1'][i]}", ",".join(f"{w_}({k_})" for w_, k_ in sorted(evidence[i].items())) or "-", ",".join(b['infos'][i]['rules']) or "-")] += 1
    print("- every crop whose status changed (truth; A -> B outcome/top-1; words read; rules):")
    for (what, t, move, words, rl), n in sorted(changed_by_which.items()):
        print(f"  - {what} x{n}: {t}; {move}; words {words}; rules {rl}")

    # ---- weak spots
    print("\n### The three weak spots of A (Stage 6) and what B does with them (counts)")
    groups = {"sprite rejected as unknown": (truth == "sprite") & (a["outcome"] == UNKNOWN), "fanta rejected as unknown": (truth == "fanta") & (a["outcome"] == UNKNOWN),
              "coca_zero taken for coca_light": (truth == "coca_zero") & (a["top1"] == "coca_light")}
    word = {"sprite rejected as unknown": "SPRITE", "fanta rejected as unknown": "FANTA", "coca_zero taken for coca_light": "ZERO"}
    print("| Group (in A) | Crops | Tall enough to read | Any keyword read | Its word read | Fixed in B | To review in B | Still wrong in B |\n|---|---:|---:|---:|---:|---:|---:|---:|")
    for g, m in groups.items():
        idx = np.where(m)[0]
        if g.startswith("coca_zero"):
            fixed, rev, left = int((b["top1"][idx] == "coca_zero").sum()), 0, int((b["top1"][idx] != "coca_zero").sum())
        else:
            fixed = int(((b["outcome"][idx] == AUTO) & (b["top1"][idx] == truth[idx])).sum())
            rev = int((b["outcome"][idx] == REVIEW).sum())
            left = len(idx) - fixed - rev
        print(f"| {g} | {len(idx)} | {int((heights[idx] >= floor).sum())} | {sum(bool(evidence[i]) for i in idx)} | {sum(word[g] in evidence[i] for i in idx)} | {fixed} | {rev} | {left} |")
    tot = {g: int(m.sum()) for g, m in groups.items()}
    print(f"(group sizes {tot}; total {sum(tot.values())})")
    return int(fix.sum() - new.sum())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage7.yaml")
    parser.add_argument("--reread", action="store_true")
    args = parser.parse_args()
    c7 = yaml.safe_load((REPO_ROOT / args.config).read_text())
    cfg6 = yaml.safe_load((REPO_ROOT / c7["config_stage6"]).read_text())
    s1 = yaml.safe_load((REPO_ROOT / "configs/stage1.yaml").read_text())
    products, idc, brand_of, bs, cc = cfg6["products"], cfg6["identify"], cfg6["brands"], cfg6["bootstrap"], cfg6["classifier"]
    t_grid, wutil = cfg6["experiments"]["logreg"]["t_grid"], idc["utility"]
    print(f"ALL NUMBERS BELOW ARE {NOTE.upper()}.\n")
    val = load_val_queries(cfg6, REPO_ROOT)
    truth, stores, ids = val["truth"], val["stores"], list(val["crop_ids"])
    store_of, index_of = dict(zip(ids, stores)), {c: i for i, c in enumerate(ids)}
    proba = LinearClassifier.load(REPO_ROOT / cc["model_dir"] / f"{cc['name']}.npz").predict_proba(val["vec"])

    cache = REPO_ROOT / c7["variant_b"]["readings_val"]
    try:
        readings = json.loads(cache.read_text()) if cache.exists() and not args.reread else read_val(cfg6, s1, c7, ids, cache)
    except OcrUnavailable as error:
        raise SystemExit(str(error))
    floor, ratio = c7["ocr"]["min_height_px"], c7["ocr"]["fuzzy_min_ratio"]
    evidence = evidence_from_readings(readings, ids, floor, ratio, keywords_detail)
    heights = np.array([readings[str(c)]["height"] for c in ids])
    words_of = {p: set(w) for p, w in c7["expect"].items()}
    ev_cfg, other = c7["evidence"], c7["other_brand_words"]

    grid = weight_grid(ev_cfg["grid"])
    options = [build_option(proba, products, evidence, w, ev_cfg, other, words_of) for w in grid]
    a = loso_b(options[:1], truth, stores, products, t_grid, wutil)                   # variant A: the all-zero option alone
    b = loso_b(options, truth, stores, products, t_grid, wutil)

    def metrics(r):
        top3 = r["top3"]
        return lambda sel: full_metrics([index_of[c] for c in sel], truth, r["outcome"], r["top1"], top3, brand_of, products, wutil)
    fa, fb = metrics(a), metrics(b)
    pa, pb = fa(ids), fb(ids)
    diff = difference_metric(fb, fa)
    dpoint, dci = diff(ids), bootstrap_by_store(store_of, diff, bs["n"], cfg6["seed"], bs["level"])
    ci_a, ci_b = bootstrap_by_store(store_of, fa, bs["n"], cfg6["seed"], bs["level"]), bootstrap_by_store(store_of, fb, bs["n"], cfg6["seed"], bs["level"])

    print(f"Reading floor {floor} px (owner decision); {int((heights >= floor).sum())} of {len(ids)} val crops are tall enough to be read; "
          f"{sum(bool(e) for e in evidence)} have a keyword.")
    kw_by_label = defaultdict(Counter)
    for lab, ev in zip(truth, evidence):
        for wd, kd in ev.items():
            kw_by_label[wd][f"{lab} ({kd})"] += 1
    print("Keywords read on the val crops (crops per truth label; counts):")
    for wd in ("COCA-COLA", "LIGHT", "ZERO", "RETORNABLE", "SPRITE", "FANTA", "BILZ", "PEPSI"):
        print(f"- {wd}: " + (", ".join(f"{l} {n}" for l, n in sorted(kw_by_label[wd].items())) if kw_by_label[wd] else "not read on any val crop"))
    variant_coca = sum(1 for lab_, ev in zip(truth, evidence) if lab_ in COCA and set(ev) & {"LIGHT", "ZERO", "RETORNABLE"})
    print(f"Coca-Cola crops where LIGHT, ZERO or RETORNABLE was read: {variant_coca}  -> w_variant {'has no effect on val and cannot be tuned: ties go to 0' if variant_coca == 0 else 'can act'}")

    print(f"\n### A against B ({NOTE})\n| Metric (%) | A | B | B minus A [95% CI by store] | Excludes zero |\n|---|---:|---:|---|---|")
    for k, label in ROWS:
        lo, hi = dci[k]
        scale, f = (1, lambda v: f"{v:.3f}") if k == "utility" else (100, pct)
        print(f"| {label} | {f(pa[k])} | {f(pb[k])} | {scale * dpoint[k]:+.1f} [{scale * lo:+.1f}, {scale * hi:+.1f}] | {'yes' if lo > 0 or hi < 0 else 'no'} |")
    print("\nWeights and thresholds used for each held-out store (B):\n| Held-out store | w_brand | w_variant | w_unknown | contradiction | t_unk | t_auto | tuning utility |\n|---|---:|---:|---:|---:|---:|---:|---:|")
    for u in b["used"]:
        print(f"| {u['store']} | {u['w_brand']} | {u['w_variant']} | {u['w_unknown']} | {u['contradiction']} | {u['t_unk']:.2f} | {u['t_auto']:.2f} | {u['utility']:.3f} |")

    net = fixes_section("Variant B as tuned (leaving one store out)", a, b, truth, products, evidence, heights, floor)
    forced_w = c7["diagnostic_forced"]
    fopt = build_option(proba, products, evidence, forced_w, ev_cfg, other, words_of)
    fb_res = loso_b([fopt], truth, stores, products, t_grid, wutil)
    pf = metrics(fb_res)(ids)
    print(f"\nDIAGNOSTIC (forced setting {forced_w}, thresholds tuned leaving one store out; NOT the keep-rule): top-1 {pct(pf['top1_acc'])} (A {pct(pa['top1_acc'])}), strictly unknown {pct(pf['unknown_strict_rejection'])} (A {pct(pa['unknown_strict_rejection'])}), false known {pct(pf['false_known'])} (A {pct(pa['false_known'])}), utility {pf['utility']:.3f} (A {pa['utility']:.3f}), share automatic {pct(pf['share_automatic'])} (A {pct(pa['share_automatic'])})")
    fixes_section("DIAGNOSTIC: B forced on", a, fb_res, truth, products, evidence, heights, floor)

    # ---- per product / confusion
    print("\n### Per product, COUNTS (top-1 correct | answered automatic and correct), A then B; `coca_light` has 3 crops")
    print("| Product | Crops | A top-1 | B top-1 | A auto+correct | B auto+correct | A rejected as unknown | B rejected as unknown |\n|---|---:|---:|---:|---:|---:|---:|---:|")
    for p in products:
        s = np.where(truth == p)[0]
        print(f"| {p} | {len(s)} | {int((a['top1'][s] == p).sum())} | {int((b['top1'][s] == p).sum())} | {int(((a['outcome'][s] == AUTO) & (a['top1'][s] == p)).sum())} | "
              f"{int(((b['outcome'][s] == AUTO) & (b['top1'][s] == p)).sum())} | {int((a['outcome'][s] == UNKNOWN).sum())} | {int((b['outcome'][s] == UNKNOWN).sum())} |")
    for name, r in (("A", a), ("B", b)):
        cols, m = coca_confusion(truth, r["top1"], products)
        print(f"\nCoca confusion, {name} (rows: truth; columns: nearest product, no rejection rules):\n| truth \\ nearest | " + " | ".join(cols) + " |\n|---|" + "---:|" * len(cols))
        for i, t in enumerate(COCA):
            print(f"| {t} | " + " | ".join(str(v) for v in m[i]) + " |")

    # ---- keep-rule
    top1_ci = dci["top1_acc"]
    checks = {"1 fixes minus new errors >= +3 (outcome)": net >= 3, "2 top-1 (known) not lower than A": dpoint["top1_acc"] >= 0,
              "3 false known not up by more than 2 points": dpoint["false_known"] <= 0.02, "3 utility not down by more than 0.02": dpoint["utility"] >= -0.02,
              "4 paired top-1 interval not entirely below zero": top1_ci[1] >= 0}
    kept = all(checks.values())
    clear = kept and top1_ci[0] > 0
    print("\n### Keep-rule (declared before any number)\n" + "\n".join(f"- {k}: {'ok' if v else '**FAIL**'}" for k, v in checks.items()))
    print(f"-> **{'KEPT' + (' (clear improvement)' if clear else ' (promising, not established)') if kept else 'NOT kept: OCR did not help on val'}**; top-1 difference {100 * dpoint['top1_acc']:+.1f} points [{100 * top1_ci[0]:+.1f}, {100 * top1_ci[1]:+.1f}]")

    # ---- demo weights (optimistic) and non-targets
    labeled_idx = np.where(np.isin(truth, products + ["unknown"]))[0]
    opt, th = choose_and_tune(options, labeled_idx, truth, t_grid, wutil)
    print(f"\nDEMO setting, tuned once on all {len(labeled_idx)} labeled val crops (optimistic, not an evaluation): {opt['w']}, thresholds {{t_unk: {th['t_unk']}, t_auto: {th['t_auto']}}}")
    out = REPO_ROOT / c7["variant_b"]["results_dir"] / "results.json"
    check_output_dir(out.parent, [REPO_ROOT / d for d in s1["protected_dirs"]])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"note": NOTE, "A": pa, "B": pb, "diff": dpoint, "diff_ci": dci, "used": b["used"], "keep": {k: bool(v) for k, v in checks.items()}, "kept": bool(kept),
                               "demo": {**opt["w"], **th}}, indent=1, default=float))
    print(f"results written to {out.relative_to(REPO_ROOT)} (gitignored)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
