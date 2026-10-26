# Stage 6 preparation report: 6 target products, catalog photos, OCR engine comparison

> No identification result yet. This report records the change of scope (6 products), the state of the catalog photos and the OCR engine comparison. Numbers about detector agreement elsewhere are agreement with Grounding DINO pseudo-labels, not accuracy; nothing here measures accuracy.

## 1. `frcnn_v2`: the run exists now and is evaluated (section 7)

At first the run was not on this machine (no log, no checkpoints); the owner then ran it. The console log is clean: 12 epochs in one run, no resume, no error, about 87 s per epoch. Its evaluation is in section 7.

## 2. Target list reduced to 6 products (owner decision)

`coca_classic`, `coca_classic_returnable` (the only returnable kept), `coca_zero`, `coca_light`, `sprite`, `fanta` (new, its own brand). Everything else is `unknown`: Pepsi, Sprite Zero and the returnable versions of Zero, Sprite and Fanta. The removed ids are out of scope, not "low data". Updated: `CLAUDE.md`, `docs/PLAN.md` (goal and scope, Stage 0 item 3, Stage 6, Unknown handling, Stage 7 keywords, metrics table), `docs/catalog_variants.md` (rewritten, with the new look-alike risks for the open set), `configs/stage6.yaml` (products, brands), `configs/label_studio/crop_project.xml` (6 ids plus `unknown`, `not_a_bottle`, `cannot_tell`; hotkeys 1 to 6, `u`, `n`, `c`), and the Stage 3 photo project config `bottle_project.xml` for consistency. Earlier reports keep their original (10-product) wording as history. No val crop had been labeled, and the crop sample does not depend on the label set, so the 180 crop tasks were **not** rebuilt; only the project's labeling configuration changes.

## 3. Catalog photos

The owner's 30 photos (5 per product) were renamed to `<product id>_NN.<ext>` by `scripts/normalize_catalog_photos.py` (sorted by original name, NN from 01, lowercase extension, nothing overwritten); the five empty folders of removed products (`coca_zero_returnable`, `pepsi`, `pepsi_returnable`, `sprite_returnable`, `sprite_zero`) were removed. The old to new table is in the handoff. `sources.csv` has 30 rows (`source_url` and `license` `unknown`, `uruguay_packaging_checked` `no`, the original name in `notes`). `check_catalog_photos.py`: 30 photos, **0 errors**, 30 warnings (packaging not confirmed). The photos are ignored by git (`git check-ignore` confirms the folder, `sources.csv` and the extension rules).

**Bottle crops** (`scripts/crop_catalog_photos.py`): frozen Grounding DINO (prompt `bottle.`, same thresholds as the shelf pre-labels, no Stage 2 filters because the "oversized" filter would drop a bottle that fills the photo), same 10% margin. Result: 29 photos gave exactly one box; **1 photo gave two boxes** (`fanta/fanta_01.jpg`: the bottle and a second box on its base). After looking at it, the box around both was used (an explicit `overrides` entry in `configs/stage6.yaml`, because it is clearly one bottle); the default rule for several boxes is the largest. **No photo gave zero boxes.** Crops are in the ignored `data/catalog/crops/` and a contact sheet of all 30 is in `data/review/catalog_crops/contact_sheet.jpg`. Looking at the sheet: the crops are clean single bottles on white; two of the photos (`coca_classic_01`, `coca_zero_01`) show a football-themed design, which looks like a limited edition: worth checking against the Uruguayan packaging.

## 4. Catalog augmentation settings (config only; built in 6b)

About 5 or 6 per photo on the bottle crop: small rotations (up to 7 degrees), mild perspective (0.12), scale 0.85 to 1.15, brightness and contrast (0.25), slight blur (sigma 0.1 to 1.2). **No horizontal flip** (it mirrors the label text, which is the evidence that separates variants) and **no hue or saturation change** (colour separates the products: Coca-Cola red, Light silver, Fanta orange, Sprite green). These differ from the Stage 5 detector augmentations on purpose; the reason is written in `configs/stage6.yaml`.

## 5. Two pipeline variants (plan)

A: embeddings only (catalog similarity plus unknown rules). B: embeddings plus OCR evidence. Same detector, catalog and val sample; thresholds calibrated separately; compared on the val sample as development numbers (leave-one-store-out) and once on the demo set as real numbers: top-1, per-product confusion matrices, unknown rejection rate, false known rate, automatic / review / unknown shares, paired bootstrap of the difference by store, the pairs `coca_classic` / `coca_zero` / `coca_light` and `coca_classic` / `coca_classic_returnable` reported separately, the share of crops where OCR reads something useful, and the time per photo of each variant. The Stage 9 API gets a request option (`use_ocr`) and the response names the variant. Written into `docs/PLAN.md` Stages 7 to 9 and the metrics table.

## 6. OCR engine comparison (nothing installed; facts from pip dry runs and package metadata)

Method: `pip install --dry-run` against the `soda-poc` environment (what would be added, with download sizes from the package files), package metadata for licences, no installation, no model download. Speed and quality are **not measured**, because that needs an installation.

| | EasyOCR | PaddleOCR | Tesseract |
|---|---|---|---|
| In the environment now | **No** (not installed; only PyTorch and torchvision are) | No | No (no `tesseract` binary on this Mac) |
| Licence of the engine | Apache-2.0 | Apache-2.0 (paddleocr, paddlepaddle, paddlex) | Apache-2.0 (engine and `pytesseract`) |
| What pip would add | 10 packages, about 63 MB (opencv-python-headless 46 MB, scikit-image 11.5 MB, shapely, python-bidi, pyclipper...) | `paddlepaddle` 100 MB plus about 50 packages, about 98 MB more (opencv-contrib 61 MB, pandas 10.6 MB, modelscope, aiohttp, pydantic...), roughly 200 MB | 1 package (`pytesseract`) plus the system binary from Homebrew |
| Models | downloaded at first use (detector and a Latin-script recogniser, about 100 MB in total from memory, to confirm) | downloaded at first use (tens of MB) | language data files (`eng`, `spa`), tens of MB |
| Licences that break our rule (permissive only; MPL only for unmodified) | `python-bidi` is **LGPL** | `python-bidi` is **LGPL** and `crc32c` is **LGPL-2.1+** | none |
| Apple Silicon | wheels for arm64; runs on PyTorch, which already uses MPS here; MPS use by EasyOCR is reported but **unverified** | `paddlepaddle` 3.3.1 has a macOS arm64 wheel, **CPU only** (no Metal backend); runs unverified | native arm64 binary, CPU |
| Expected fit for bottle labels | scene-text recogniser, handles curved and stylised text reasonably; Spanish via the Latin model | usually the strongest on scene text, but the heaviest and the most dependencies | made for clean document text; weakest on logos and curved label text |

**Recommendation: try EasyOCR first** (lightest addition to this environment, PyTorch-based, Apache-2.0), with Tesseract as the lightweight fallback and PaddleOCR only if EasyOCR reads too little. The catch is the **LGPL `python-bidi`** package that EasyOCR (and PaddleOCR) pull in: it is a dynamic dependency used for right-to-left scripts that we do not need, but it falls outside the licence rule, so it needs an explicit decision. A cheap, label-free test exists: the 30 catalog crops carry known text ("ZERO AZUCARES", "LIGHT", "RETORNABLE", "SPRITE", "FANTA", "COCA-COLA"), so a trial can measure how often each engine reads each keyword and the seconds per crop (CPU and MPS) before any val label is needed. Proposed, pending approval: run that trial in a throwaway environment outside `soda-poc`.

## 7. `frcnn_v2` evaluation on val (every number is agreement with Grounding DINO pseudo-labels, not accuracy; test not touched)

Evaluated exactly like `frcnn_v1`: the same scripts (`predict_val.py --checkpoint`, `evaluate_detections.py`), the same matching, the same store bootstrap (6 stores, 1,000 resamples, seed 0), predictions under the 1.5% height floor ignored, both `best.pt` (epoch 5, chosen by val AP@0.5, so optimistic) and the last epoch (12, not chosen on val). Curve: `reports/figures/stage_06_frcnn_v2_training.svg`.

| Agreement, % [95% CI by store], raw output | COCO baseline | v1 best.pt (ep. 5) | v1 last (ep. 30) | v2 best.pt (ep. 5) | v2 last (ep. 12) |
|---|---|---|---|---|---|
| AP@0.5:0.95 | 68.5 [61.4, 71.8] | 78.2 [72.4, 81.3] | 78.8 [72.2, 81.7] | 79.0 [74.0, 81.9] | 79.1 [73.9, 81.7] |
| AP@0.5 | 88.6 [79.6, 91.4] | 94.0 [90.6, 95.5] | 91.6 [85.9, 93.3] | 94.0 [90.9, 95.2] | 92.8 [89.3, 94.3] |
| Recall at IoU 0.5 (all scores) | 94.3 [88.5, 96.1] | 97.5 [96.8, 98.1] | 93.2 [88.8, 94.9] | 97.1 [96.4, 97.8] | 95.4 [94.0, 96.1] |
| Precision, score >= 0.3 | 69.6 [60.8, 73.1] | 73.2 [62.8, 77.7] | 86.5 [81.4, 91.5] | 78.6 [70.9, 82.7] | 83.4 [75.6, 88.2] |
| Recall, score >= 0.3 | 87.0 [78.4, 90.0] | 94.2 [91.3, 95.3] | 89.6 [83.1, 91.6] | 93.1 [90.1, 94.4] | 91.0 [86.0, 92.9] |
| F1, score >= 0.3 | 77.3 [69.0, 80.2] | 82.4 [74.4, 85.3] | 88.0 [82.4, 90.7] | 85.3 [79.5, 87.8] | 87.1 [80.8, 89.7] |
| TP / FP / FN at score 0.3 | 1,877 / 820 / 280 | 2,032 / 745 / 125 | n/a | 2,009 / 546 / 148 | 1,963 / 390 / 194 |

With the Stage 2 filters: v2 best.pt AP@0.5 92.9 [90.9, 94.5], precision 81.8, recall 92.5; v2 last AP@0.5 92.3 [89.5, 93.8], precision 85.7, recall 90.5.

Paired bootstrap of the difference by store (a minus b, percentage points; the interval excludes zero unless marked):

| Comparison (raw output) | AP@0.5:0.95 | AP@0.5 | Recall (all scores) | Precision >= 0.3 | Recall >= 0.3 | F1 >= 0.3 |
|---|---|---|---|---|---|---|
| v2 best.pt minus v1 best.pt | +0.7 [0.2, 1.6] | 0.0 [-0.6, 0.4] (zero inside) | -0.5 [-0.6, -0.1] | +5.5 [4.0, 8.5] | -1.1 [-1.4, -0.8] | +2.9 [1.8, 5.2] |
| v2 last minus v1 best.pt | +0.9 [0.1, 2.1] | -1.1 [-1.7, -0.9] | -2.2 [-2.9, -1.9] | +10.3 [8.8, 13.4] | -3.2 [-5.4, -2.3] | +4.7 [4.1, 6.3] |
| v2 last minus v1 last | +0.4 [-0.2, 1.9] (zero inside) | +1.2 [0.5, 3.6] | +2.1 [1.0, 5.3] | -3.1 [-6.1, -2.0] | +1.4 [0.9, 2.8] | -0.9 [-2.0, -0.5] |
| v2 best.pt minus COCO | +10.5 [9.6, 13.5] | +5.4 [3.5, 11.7] | +2.8 [1.4, 8.4] | +9.0 [5.0, 16.0] | +6.1 [4.2, 11.9] | +7.9 [5.6, 12.8] |
| v2 last minus COCO | +10.7 [9.5, 13.2] | +4.2 [2.5, 9.8] | +1.1 [-0.2, 5.7] (zero inside) | +13.8 [9.0, 21.6] | +4.0 [2.2, 7.6] | +9.7 [7.0, 14.4] |
| v2 best.pt minus v2 last | -0.2 [-0.8, 0.4] (zero inside) | +1.2 [0.5, 1.8] | +1.7 [1.4, 2.8] | -4.8 [-6.7, -3.2] | +2.1 [1.4, 4.2] | -1.8 [-2.7, -0.7] |

(With the filters: v2 best.pt minus COCO + filters AP@0.5 +4.5 [2.7, 9.9]; v2 last minus COCO + filters +3.8 [1.9, 8.5].) Both v2 checkpoints clearly beat the COCO baseline in AP@0.5 and AP@0.5:0.95 (interval above zero), as `frcnn_v1` did. Between v1 and v2 the differences are small (about 1 point in AP@0.5, none in AP@0.5:0.95 for the last epochs), and they sit on the precision-recall trade-off: the models that train longer or stop later are more selective.

Disagreements by cause (v2 last, raw output, score 0.3; automatic proxy causes as in Stage 5; `python scripts/stage5_error_analysis.py --config configs/stage5_v2.yaml`): 1,963 matched, **390 extra predictions** (can 34, fragment 123, partial overlap 76, small 6, other 151) and **194 missed pseudo-labels** (can 19, fragment 29, partial overlap 83, small 5, other 58). Against `frcnn_v1` best.pt (745 extra, 125 missed) it makes about half as many extra boxes, and misses more. The model still copies most pseudo-label errors (240 of the 288 val pseudo-labels that look like cans or fragments, against 261 for v1 best.pt).

**Recommendation: keep `frcnn_v2` last epoch (`models/stage5/frcnn_v2/epoch_12.pt`) as the detector for the next stages.** Reasons: (1) it was not chosen on val, so its numbers are not inflated by selection (the best.pt numbers are); (2) against the COCO baseline it wins clearly, and against the last epoch of `frcnn_v1` it is equal or slightly better (AP@0.5 +1.2, recall +2.1, AP@0.5:0.95 unchanged) with 12 instead of 30 epochs (26 minutes against 43); (3) its agreement recall at score 0.3 is 91.0 [86.0, 92.9], above the 90% target; (4) it sends about half as many extra boxes (cans, fragments) into the identification stage, where every extra box is a chance for a false known identification. The price is more missed pseudo-labels than `best.pt` (194 against 125 to 148); if recall matters more than false boxes later, the operating score can be lowered from 0.3 without retraining (all-score recall is 95.4%). The `best.pt` checkpoints are selection-biased and not recommended. All of this is agreement with Grounding DINO, which includes its errors; the demo set will say how it does on real bottles.

## 8. Crop labels (the owner's 180 val crops; `scripts/export_crop_labels.py` on the Label Studio JSON export)

Audit of the export: **clean**. All 180 sample crops have exactly one valid label; no missing label, no crop with more than one label, no duplicate task, no id outside the sample; the crop ids in the export, `data/crops/val_sample/mapping.csv` and `tasks.json` are the same 180. Median time per crop 3.4 s (16.6 minutes in total, inside the 12 to 25 minute estimate).

| Label | Crops | Share |
|---|---:|---:|
| coca_classic | 20 | 11% |
| coca_classic_returnable | 9 | 5% |
| coca_zero | 28 | 16% |
| coca_light | **3** | 2% |
| sprite | 10 | 6% |
| fanta | 16 | 9% |
| unknown | 67 | 37% |
| not_a_bottle | 15 | 8% |
| cannot_tell | 12 | 7% |

Per store (crops by label; stores are anonymous ids):

| Store | classic | classic returnable | zero | light | sprite | fanta | unknown | not a bottle | cannot tell | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| s01 | 1 | 0 | 7 | 1 | 1 | 1 | 12 | 0 | 0 | 23 |
| s03 | 2 | 3 | 6 | 1 | 1 | 3 | 30 | 7 | 1 | 54 |
| s06 | 2 | 0 | 4 | 0 | 1 | 3 | 3 | 4 | 2 | 19 |
| s11 | 3 | 3 | 4 | 0 | 2 | 1 | 12 | 0 | 2 | 27 |
| s15 | 9 | 3 | 3 | 1 | 2 | 5 | 10 | 3 | 5 | 41 |
| s16 | 3 | 0 | 4 | 0 | 3 | 3 | 0 | 1 | 2 | 16 |

What this limits:
- **`coca_light` has only 3 val crops (fewer than 5):** nothing can be concluded for it later (an accuracy of 1 of 3 or 3 of 3 means nothing), and it appears in only 3 of the 6 stores. `coca_classic_returnable` (9) and `sprite` (10) are low as well; every per-product number must be shown with its count, and the leave-one-store-out numbers are pooled over stores.
- Store coverage is uneven: `s16` has no `unknown` crop and `s06` only 3, so rejection rules cannot be tuned or tested store by store; `s03` contributes 30 of the 67 unknowns.
- **Pseudo-label precision on val: `not_a_bottle` is 15 of 180 = 8.3% (Wilson 95% interval 5.1% to 13.3%), so about 92% of the pseudo-label boxes are bottles (a sample estimate; it says nothing about bottles the detector missed).** By store: s03 7 of 54, s06 4 of 19, s15 3 of 41, s16 1 of 16, none in s01 and s11. This is lower than the 13% that the Stage 5 proxies flagged as "can or fragment", so those proxies overestimate the label errors.
- `cannot_tell` (12, 6.7%) are crops where even the owner could not decide; they are not used to measure identification.

## 9. OCR trial (throwaway environment `ocr-trial`; `soda-poc` untouched; nothing added to `requirements.txt`)

`scripts/ocr_trial.py` (with `configs/ocr_trial.yaml`) runs an engine on the 30 catalog crops (known text) plus 30 labeled val crops from the original photo pixels, in the throwaway environment (EasyOCR 1.7.2 with PyTorch 2.14.1; `python-bidi` is installed there as agreed, for the trial only). It counts, per product, how many of the expected words (brand and variant) are read, with a lenient match (the keyword anywhere in the text, tolerating 1 or 2 wrong letters) and a strict one (whole words only), and the wrong-keyword reads.

**EasyOCR runs on MPS** (`reader.device = mps` confirmed) and on CPU. Time per crop (median, after warm-up): **MPS 0.07 s, CPU 0.09 s on the original crops; CPU 0.27 s and MPS 0.07 s when crops are upscaled to 900 px tall**; the first call (model loading, first download of about 100 MB of weights) takes several seconds.

| Catalog crops (30, clean photos, known text) | Expected words read (lenient / strict) | Wrong-keyword reads (lenient / strict) |
|---|---|---|
| coca_classic (COCA-COLA) | 0 / 0 of 5 | 0 / 0 |
| coca_classic_returnable (COCA-COLA, RETORNABLE) | 3 / 3 of 10 | 1 / 0 |
| coca_zero (COCA-COLA, ZERO) | 2 / 0 of 10 | 0 / 0 |
| coca_light (COCA-COLA, LIGHT) | 5 / 5 of 10 | 0 / 0 |
| sprite (SPRITE) | 5 / 5 of 5 | 0 / 0 |
| fanta (FANTA) | 5 / 5 of 5 | 0 / 0 |

- It reads **Sprite, Fanta, Light and RETORNABLE** when the label is clear (all of Sprite and Fanta, all five Light crops, 3 of 5 RETORNABLE), and it reads Spanish label text ("SABOR ORIGINAL", "SIN AZÚCARES").
- It **never reads the Coca-Cola script logo** ("Gccoc", "@ccocl"), so `COCA-COLA` is read in 0 of 20 Coca-Cola catalog crops; `ZERO` is read only when the OCR glues it to the next word ("ZEROAZUCAR"), which the strict whole-word rule misses.
- **False readings:** the one wrong keyword is `FANTA` found inside the word FANTASIA of the legal text "bebida de fantasía" on a Coca-Cola returnable label; the lenient rule counts it, the strict rule does not. Whole-word matching is needed.
- **On shelf crops it reads almost nothing useful:** random mix of 30 labeled val crops (products and a few unknown, spread over stores): text of some kind in 14 of 30 crops (23 of 30 after upscaling to 900 px), but **a keyword in 0 of 30**. The crops are small (the median product crop is about 330 px tall and many show a label only a few pixels high). Using the tallest crops of each label (30 crops, 150 to 1,200 px tall, most above 500 px) gives a keyword in 6 of 30: Fanta read on 3 of 4 Fanta crops (and RETORNABLE once), nothing for Coca-Cola, Sprite or Light. The false readings there are real text that misleads: `FANTA` on two Coca-Cola crops that are **promo packs tying a Coca-Cola bottle to a Fanta bottle** ("AHORRO FANTA" on the sleeve), and `RETORNABLE` on an `unknown` crop (a returnable of another product).
- Of the 86 product-labeled val crops, 42 are at least 400 px tall and 27 at least 500 px tall; below that, the text is too small for this engine.

**Tesseract: not run yet.** The system package is not installed and I may not install system packages. The exact command for you to run: `brew install tesseract` (English data is enough: the words are Latin script). Then I run `python scripts/ocr_trial.py --engine tesseract` in the throwaway environment (where `pytesseract` is already installed). PaddleOCR was not trialed (about 200 MB, CPU only on this Mac, and the same LGPL dependency).

**Recommendation for variant B (not wired into any code yet).** Keep variant B in the plan, with **EasyOCR as the engine** (it runs on MPS at about 0.07 s per crop, Apache-2.0, nothing heavy), but expect its evidence on shelf photos like these to be thin: it reads Fanta and Sprite logos and printed words (LIGHT, RETORNABLE, ZERO) only on large, sharp crops (catalog photos, and the tallest shelf crops), and it cannot read the Coca-Cola logo, so it will not help to tell `coca_classic` from `coca_zero` on small shelf crops. Its most plausible value is on the demo set (better photos, larger crops) and as a **contradiction signal** (for example a clear "FANTA" or "SPRITE" on a crop identified as Coca-Cola) rather than as positive evidence. Rules to carry into Stage 7: whole-word matching only; require brand context (a bare RETORNABLE does not identify a product); expect false evidence from promo sleeves and legal text; read only crops above a minimum height. Tesseract, once run, is the fallback to compare; the decision on it waits for that run.

## Limits

- The OCR trial covers EasyOCR only (Tesseract waits for `brew install tesseract`); 60 crops, small samples, and the keyword counts are lower bounds (some photos show the back of the bottle).
- The val crop labels are 180 crops from 6 stores; `coca_light` has 3.
- The Uruguayan packaging of the catalog photos and their sources are still to be filled in by the owner.

## How to verify (inside `soda-poc`)

```bash
python -m pytest -q
python scripts/check_catalog_photos.py          # 30 photos, 0 errors, 30 warnings
python scripts/crop_catalog_photos.py           # re-creates the crops and the contact sheet
git check-ignore -v data/catalog/product_photos/fanta/fanta_01.jpg data/catalog/product_photos/sources.csv
git status --short                               # nothing under data/ except splits/*.txt
```
