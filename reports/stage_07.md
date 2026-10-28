# Stage 7: OCR as evidence (pipeline variant B), report

> **Every number is a development number**: agreement with the owner's labels on the 180 val crops from 6 stores (a few crops per group, `coca_light` has 3), both variants tuned leaving one store out, intervals by resampling whole stores. Not accuracy. The test split was not touched. The OCR readings behind the decisions on words and heights were made on the 253 TRAIN shelf entries (`reports/stage_07_ocr_reading.md`); the val crops were read once, for the comparison.

## 0. Result in short
- **Both variants run and were compared on val. OCR did not help.** Variant B, tuned leaving one store out, **chose "no OCR" (all weights 0) for every one of the 6 held-out stores**, so B is identical to A: top-1 76.7% both, Coca variant top-1 78.3%, strictly unknown 94.0%, false known 0.0%, utility 0.444. **Fixes 0, new errors 0**, net +0. The keep-rule fails on its first condition (net >= +3).
- **None of the 14 weak-spot crops of Stage 6 had a single keyword read** (5 `sprite` and 5 `fanta` rejected as unknown, 4 `coca_zero` taken for `coca_light`; 12 of them were tall enough for the 150 px floor). OCR cannot act on them.
- **Forcing the rules on makes things slightly worse, not better** (diagnostic, section 4): top-1 75.6%, false known 3.0%, utility 0.323; the FANTA boost lifts 2 `fanta` crops from review to a correct automatic answer and also lifts 2 other crops (a `coca_zero` with a promo sleeve, and an `unknown` crop) to a wrong automatic `fanta`.
- Variant A stays the pipeline. Variant B stays implemented, tested and runnable (module, `requirements-ocr.txt`, `--use-ocr` in the demo), to be measured again on the demo set, where the crops will be larger and sharper.

## 1. What was built (tasks 5 to 8)
- `src/ocr/combine.py`: the evidence rules of the proposal (SPRITE/FANTA boost with classifier context >= 0.05; LIGHT/ZERO/RETORNABLE boost only with Coca-Cola context; contradiction to review; not-a-target combinations and the owner-added exact words **BILZ and PEPSI** push to `unknown`; two brands means no boost). `src/ocr/variant_b.py`: weight grid, tuning and leave-one-store-out. `scripts/evaluate_variant_b.py`. Tests: 197 pass (177 at the start of Stage 7).
- Floor: `ocr.min_height_px: 150` in `configs/stage7.yaml` (owner decision), overridable in the demo with `--ocr-min-height`. The val result is reported at 150 px only.
- Demo: `python scripts/demo_identify.py --image PATH --use-ocr` (also `--ocr-min-height`, `--w-brand`, `--w-variant`, `--w-unknown`, `--no-contradiction`). The panel shows the words read (a `~` marks a fuzzy match) and the rules that fired, the JSON has the OCR text, words and rules per box. Without the OCR packages it stops with the install message before loading anything (tested by running the real script with `easyocr` un-importable). The demo weights are **UNTUNED** (the declared diagnostic setting), because the val tuning chose zero weights and a demo with zero weights would show nothing.
- One small declared extension of the proposal: the hard rules (contradiction, two brands) got an on/off switch in the tuning grid (`contradiction` in {0, 1}), because the proposal said "all weights 0 is exactly variant A", which needs the hard rules to be switchable too.

## 2. A against B on the 180 val crops (leave one store out; 95% interval by store)
| Metric (%) | A | B | B minus A [95% CI by store] |
|---|---:|---:|---|
| Top-1 (86 known crops) | 76.7 | 76.7 | +0.0 [+0.0, +0.0] |
| Coca variant top-1 (60 Coca crops) | 78.3 | 78.3 | +0.0 [+0.0, +0.0] |
| Top-3 | 94.2 | 94.2 | +0.0 |
| Known answered automatically and correctly | 40.7 | 40.7 | +0.0 |
| Strictly unknown (unknown crops) | 94.0 | 94.0 | +0.0 |
| False known rate | 0.0 | 0.0 | +0.0 |
| Share automatic / review / unknown | 24.2 / 22.2 / 53.6 | same | +0.0 |
| Mean utility | 0.444 | 0.444 | +0.000 |

A reproduces the Stage 6 gate numbers exactly (same classifier, same thresholds per store). The intervals have zero width because the two variants made the same decision on every crop. **Weights chosen per held-out store: all zero** (`w_brand` 0, `w_variant` 0, `w_unknown` 0, contradiction off); thresholds as in A (t_unk 0.30; t_auto 0.75, 0.80 for s15). The grid is walked from all zeros and only a strict improvement of the tuning utility replaces the best, so a tie goes to A.

**Keep-rule (declared before any number; condition 1 evaluated on the outcome, as written in `configs/stage7.yaml`):**
| Condition | Result |
|---|---|
| 1. Fixes minus new errors >= +3 (outcome of the 153 labeled crops) | **fails** (0 and 0, net +0) |
| 2. Top-1 not lower than A | ok (+0.0) |
| 3. False known not up by more than 2 points; utility not down by more than 0.02 | ok (+0.0; +0.000) |
| 4. Paired top-1 interval not entirely below zero | ok ([+0.0, +0.0]) |
| **Verdict** | **Not kept: OCR did not help on val** |

**LIGHT, ZERO, RETORNABLE:** read on **no** Coca-Cola crop of val (ZERO once on a `not_a_bottle` crop, RETORNABLE on a `fanta` crop and on an `unknown` crop, neither with Coca context), so `w_variant` has no effect on val and **cannot be tuned**; ties go to 0, as you asked. The boost stays implemented and tested for the demo set.

## 3. What the OCR read on val (counts of crops, by truth label; read at the 150 px floor: 148 of 180 crops tall enough, 17 with a keyword)
| Word | Read on |
|---|---|
| COCA-COLA, LIGHT | no crop |
| ZERO | 1 `not_a_bottle` (exact) |
| RETORNABLE | 1 `fanta` (exact), 1 `unknown` (exact) |
| SPRITE | 1 `sprite` (exact) |
| FANTA | 2 `fanta` (exact), 1 `fanta` (fuzzy), 1 `coca_classic` (exact), 1 `coca_zero` (exact), 1 `unknown` (fuzzy) |
| BILZ | 1 `unknown` (exact) |
| PEPSI | 7 `unknown` (exact) |

PEPSI and BILZ behave as the TRAIN rule predicted (read only on `unknown`), but 7 of those 8 crops were already rejected as unknown by A, so the push gave no gain.

**The three weak spots (counts; groups defined on A):**
| Group in A | Crops | Tall enough to read (>= 150 px) | Any keyword read | Its word read | Fixed in B | Still wrong in B |
|---|---:|---:|---:|---:|---:|---:|
| `sprite` rejected as unknown | 5 | 4 | 0 | 0 | 0 | 5 |
| `fanta` rejected as unknown | 5 | 4 | 0 | 0 | 0 | 5 |
| `coca_zero` taken for `coca_light` | 4 | 4 | 0 | 0 | 0 | 4 |
| Total | 14 | 12 | 0 | 0 | **0** | 14 |

## 4. Diagnostic: B forced on (NOT part of the keep-rule; setting declared before its numbers)
Because the tuning declined OCR, B was also run with the fixed setting `w_brand 2, w_variant 2, w_unknown 2, contradiction on` (thresholds still tuned leaving one store out) to see what the rules do when they act:
- Top-1 75.6% (A 76.7), strictly unknown 94.0% (same), **false known 3.0% (A 0.0)**, utility **0.323 (A 0.444)**, share automatic 28.1% (A 24.2).
- By the keep-rule's definitions: **fixes (wrong in A, right in B) 0; new errors (right in A, wrong in B) 1; net -1.** The new error is a `fanta` crop answered correctly by A that reads both FANTA and RETORNABLE: the rule "SPRITE/FANTA together with RETORNABLE is not a target (a returnable Fanta is unknown)" pushed it to `unknown`; most likely the word comes from deposit text and not from a returnable Fanta.
- **Exact words against fuzzy fragments** (your question, all 9 crops whose status changed): the FANTA boost moved crops from review to automatic `fanta`: with an **exact** word, 1 right (`fanta`) and 1 wrong (a `coca_zero` with a promo sleeve reading "AHORRO FANTA"); with a **fuzzy** fragment (FANT/ANTA), 1 right (`fanta`) and 1 wrong (an `unknown` crop). So exact and fuzzy each give +1 and -1: the fuzzy fragments are neither the problem nor the solution on val. One more exact FANTA read lifted a `coca_classic` from unknown to review (neutral gain). Three of the 9 status changes have no OCR word at all (they come from the different thresholds tuned for the forced setting).
- Lesson: **a brand word on a crop is not proof that the bottle is that product** (promo sleeves, shelf-edge labels, neighbouring products). The classifier-context condition (>= 0.05) was not strong enough to stop the two wrong lifts; the contradiction rule does not catch them either, because the boost has already made the top-1 agree with the word.

## 5. Why the Coca variant words are never read
Facts measured on TRAIN and val (crop height in original pixels):
| | TRAIN median / crops >= 400 px | val median / crops >= 400 px |
|---|---|---|
| coca_classic | 286 / 9 of 61 | 443 / 12 of 20 |
| coca_zero | 267 / 2 of 21 | 282 / 11 of 28 |
| coca_light | 278 / 1 of 17 | 300 / 1 of 3 |
| coca_classic_returnable | 266 / 5 of 38 | 291 / 3 of 9 |
| sprite | 309 / 5 of 28 | 369 / 5 of 10 |
| fanta | 309 / 0 of 10 | 559 / 10 of 16 |

- **Crop size.** Most crops are 260 to 300 px tall. On a Coca-Cola variant label the word that matters (`ZERO AZUCAR`, `LIGHT`) is a thin line of small letters: on the tallest `coca_zero` val crop (930 px, which I looked at) the line is only a few pixels high, and the OCR returned one noise fragment. Upscaling to 900 px adds pixels but not detail. The few tall crops do not fix it: 11 of the 28 val `coca_zero` crops are at least 400 px tall and none had ZERO read.
- **The script logo.** The Coca-Cola logo is a stylised script; EasyOCR's printed-text model never read COCA or COLA on any TRAIN or val crop. So there is never a "COCA-COLA" signal to give Coca context, and the context rule must rely on the classifier's Coca probability.
- **Where OCR does work:** the Sprite and Fanta labels carry the brand name in large block letters (SPRITE read on 13 of 28 TRAIN crops, 11 of 23 from 250 px). Even so, on val only 1 `sprite` crop and 3 `fanta` crops had their word read, and none of the 10 weak-spot crops of these two products.
- Also seen on val: the same OCR reads promo text (a `coca_zero` crop reads "AHORRO FANTA").

## 6. Speed (10 val photos, MPS, median seconds per photo)
Variant A 0.90 s (detector 0.27, embeddings and classifier 0.60). Variant B at the 150 px floor: +2.02 s for about 34 crops read, total 2.96 s [0.87, 5.40]. A real demo run of a photo with 44 boxes read 42 crops in 5.1 s. About 0.06 s per crop read.

## 7. What to expect on the demo set
- **Larger and sharper crops** (close shots of a few bottles) should let the OCR read what it cannot read here: Sprite and Fanta names (already readable), probably `RETORNABLE`, and `LIGHT`/`ZERO` only if the line is big enough to be legible: a few pixels do not become letters. I would not expect the Coca-Cola logo to be read at any size.
- **The risks seen on val will remain:** promo sleeves ("AHORRO FANTA" on a Coca-Cola zero), deposit text ("retornable") on a non-returnable product, and brand words on neighbouring products. The weights are untuned in the demo, so the first demo-set measurement should tune them there (leave-one-store-out or a split by shelf), report fixes and new errors again, and consider a stricter boost (for example context >= 0.2 instead of 0.05, or no boost when the OCR word and the classifier disagree strongly). These are proposals for Stage 8 or for the demo-set measurement, not changes made now.
- A demo-set measurement of B against A costs about 2 to 3 s per photo and the same reading protocol (`--use-ocr`, floor 150 px unless changed with `--ocr-min-height`).

## 8. Gate assessment against docs/PLAN.md (Stage 7)
| PLAN gate | Assessment |
|---|---|
| Both variants run | **Met.** A (`demo_identify.py`) and B (`--use-ocr`), same detector, same classifier and crops, thresholds tuned per variant leaving one store out. |
| Both variants compared on val | **Met.** Section 2, with a paired bootstrap by store, per-product counts and the Coca confusion matrices (identical for A and B; `coca_zero` taken for `coca_light` still 4). The comparison on the demo set is still open (PLAN item 8: "once, on the demo set"). |
| OCR stays as an option only if it reduces the specific confusions found in Stage 6 (otherwise the report says it did not help) | **Not met: OCR did not help on val.** It fixed none of the 14 weak-spot crops (0 fixes, 0 new errors as tuned; with the rules forced on, 0 fixes and 1 new error). It is therefore **not adopted** into the pipeline; per the owner's decision it stays implemented and runnable for a re-measurement on the demo set. |
| Measure the share of crops where OCR reads anything useful, and the time cost per photo | Done: 17 of 148 readable val crops (11%) had a keyword; 18 of 250 TRAIN crops (7%) at the 150 px floor; time in section 6. |

## 9. Problems and caveats
- 180 val crops from 6 stores; 14 weak-spot crops; `coca_light` has 3 crops. With zero fixes there is no interval to read: the conclusion is "OCR had nothing to act on in this sample", not "OCR cannot help in general".
- The tuning grid has 64 weight settings but picked zero on every fold, so there is no optimistic selection to worry about in B; the forced diagnostic is a single declared setting.
- Variant B's demo weights are untuned; any demo-set claim needs its own measurement.
- `sources.csv` is still unknown, so a FINAL report is blocked until `check_catalog_photos.py --final` passes (owner task from Stage 6).
- **Dependency list for the final report:** `requirements-ocr.txt` (easyocr 1.7.2 Apache-2.0, rapidfuzz MIT, opencv-python-headless Apache-2.0 whose wheel bundles **LGPL FFmpeg libraries**, pyclipper MIT, **python-bidi LGPL**, scikit-image/shapely/tifffile/lazy-loader BSD-3, imageio BSD-2, ninja Apache-2.0). The two LGPL items are owner-approved exceptions (unmodified, dynamically used, isolated in this file; reason: the best OCR engine in the trial, it runs on MPS; Tesseract is the licence-clean fallback). `requirements.txt` stays permissive only.

## 10. How to verify
```bash
python -m pytest -q                                   # 197 pass
python scripts/evaluate_variant_b.py                  # A against B, keep-rule, fixes and new errors, diagnostic (reads the val crops once, cached in data/ocr/stage7/)
python scripts/demo_identify.py --image PATH --use-ocr   # variant B in the demo (UNTUNED weights; --ocr-min-height, --w-brand, ...)
```
