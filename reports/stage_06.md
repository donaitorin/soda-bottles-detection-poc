# Stage 6: reference catalog, embeddings and the product scorer (summary of 6a to 6d)

> **Every number in this report is a DEVELOPMENT number: agreement with the owner's labels on 180 val crops from 6 stores (thresholds tuned leaving one store out), not accuracy.** Intervals are bootstraps that resample whole stores (6 stores: wide, a rough guide). The val and test labels of the boxes are Grounding DINO pseudo-labels; the product labels of the crops are the owner's. The test split was never used. Accuracy will only exist on the future owner-corrected demo set.

**Status: the Stage 6 gate is MET FOR TWO OF THREE numeric targets and MEASURED for the rest; brand accuracy is 89.5%, one crop short of the initial 90% hypothesis.** Waiting for the owner's confirmation before Stage 7. A **final** report still needs `python scripts/check_catalog_photos.py --final` to pass (see the caveats).

## 1. What was done, step by step

| Step | What | Result |
|---|---|---|
| **6a** | Crop tooling (10% margin boxes), a val crop sample of 180 crops stratified by 6 stores, labeled by the owner WITHOUT suggestions (6 products, `unknown`, `not_a_bottle`, `cannot_tell`) in a second Label Studio project; catalog photo folder; `frcnn_v2` detector run (12 epochs, compressed cosine schedule; last epoch chosen: AP@0.5 92.8 [89.3, 94.3], recall 91.0 at score 0.3, +4.2 AP@0.5 [2.5, 9.8] over the COCO baseline) | Labels: classic 20, returnable 9, zero 28, light 3, sprite 10, fanta 16, unknown 67, `not_a_bottle` 15 (8.3%: the val pseudo-label precision is about 92%), `cannot_tell` 12 |
| **6b** | DINOv2-base CLS embeddings (87 M parameters, about 10 ms per crop on MPS); 30 owner product photos (5 per product) cropped with the frozen Grounding DINO; own database `soda_catalog` (pgvector, cosine HNSW); nearest-entry scoring with unknown rules; augmented catalog; mean pooling; domain gap | Originals only: top-1 53.5% [44.7, 61.8], top-3 90.7%, brand 88.4% [81.7, 94.8], **0 automatic answers**, strictly unknown 26.9%. Augmented catalog: top-1 -12.8 [-23.4, -4.3] (not kept). Domain gap: a shelf crop is as close to the best wrong product (median 0.764) as to its right one (0.760) |
| **6c** | The plan's fallback for a large domain gap: 300 TRAIN crops labeled by the owner (200 random by store + 100 targeted for rare products), added to the catalog; unknown crops as negative entries; class-balance rules; candidate lists | Shelf products alone did not help (top-1 +1.2, utility down); **unknown negatives did**: strictly unknown 29.9% to 74.6% (+44.8 [+22.6, +58.1]), utility 0.020 to 0.190. Catalog (b) = originals + shelf entries + negatives (193 rows) went into the database. The studio photos add nothing once shelf examples exist |
| **6d** | Three experiments on separating the Coca-Cola variants (keep-rules written first): colour, context (0% margin, SAM 2 mask), logistic regression | **Logistic regression on the CLS embedding: clear winner** (Coca variant top-1 50.0% to 78.3%, strictly unknown 74.6% to 94.0%, false known 3.0% to 0.0%). Colour, 0% margin and SAM 2 crops not kept. scikit-learn confirmed against the torch version and adopted |

## 2. What was tried, kept, and why (numbers against the previous reference)

| Idea | Kept? | Why (development numbers) |
|---|---|---|
| Catalog from 30 studio photos, nearest-entry search | superseded | top-1 53.5%, brand 88.4%, but 0 automatic answers: shelf crops are as close to wrong products as to right ones |
| Catalog augmentations (rotation, perspective, scale, brightness, blur) | no | top-1 -12.8 [-23.4, -4.3] |
| Mean pooling of patch tokens | no | 12.8%: the white padding is 62% of the 224 px square; with padding patches left out it is 47.7%, still below CLS 53.5% |
| Shelf crops from train photos in the catalog (a) | no, alone | top-1 +1.2 [-16.2, +16.5], top-3 -7.0, 3 automatic answers, all wrong |
| + crops labeled unknown as negative entries (b) | **yes** (catalog fallback) | strictly unknown +44.8 points, rule declared beforehand |
| Colour histogram + embedding (4a) | no | Coca +6.7 [-5.6, +20.0] but false known 3.0 to 6.0% and utility down; a colour histogram alone is as good as the embedding at the 4 variants (33 vs 31 of 60) |
| 0% margin; SAM 2 mask on white (4b) | no | 0%: unknown handling much worse (74.6% to 37.3%); SAM 2: Coca +10.0 [-9.3, +29.3] but utility -0.023 (limit -0.02), a hint only. SAM 2 is fast on MPS (0.2 to 0.5 s per photo) |
| **Logistic regression on CLS, trained on the 253 shelf entries (4c)** | **yes, adopted scorer** | Coca +28.3 [+19.2, +36.4], top-1 +22.1 [+13.4, +31.2], utility 0.190 to 0.444, 35 of 37 automatic answers right; control with shuffled labels 19.8% (chance) |
| + colour descriptor on the logistic regression | no | false known 6.0% |
| scikit-learn against the torch version | scikit-learn kept | same most probable class for 180 of 180 crops, probabilities within 6e-3 at the chosen C, identical metrics and outcomes (details in `reports/stage_06d.md`, section 8) |

## 3. The adopted system (what the demo runs)
frcnn_v2 `epoch_12` at score 0.3 (+ 1.5% height floor) → 10% margin crop → DINOv2-base CLS embedding (768) → **logistic regression** (scikit-learn: L2, multinomial, balanced class weights, standardised features, `C = 0.1` chosen by leaving one TRAIN store out; classes: 6 products + `unknown`; trained on 253 owner-labeled train crops: classic 61, returnable 38, zero 21, light 17, sprite 28, fanta 10, unknown 78) → rules on the probabilities: `unknown` if the unknown class is the most probable or the top product's probability is below `t_unk`; `automatic` from `t_auto`; otherwise `review`, shown as **candidates** (blue, at most 3 products within a calibrated probability distance of the top-1) when the list is short. The model is saved in `models/stage6/` with a model card (training counts, hash of the labels file, C, seed, thresholds, date). The database catalog (b) stays as the fallback scorer (`--scorer catalog`). Demo thresholds (`t_unk 0.30`, `t_auto 0.75`, list distance 0.4526) were tuned ONCE on all 180 val crops: **optimistic, not an evaluation**; train photos are optimistic in the demo too (the classifier saw their crops; the tool prints a note).

## 4. The gate, with the adopted scorer (`python scripts/evaluate_stage6_gate.py`)
Thresholds tuned leaving one store out (they came out almost the same for every store: unknown below 0.30, automatic from 0.75, once 0.80). Criteria declared before computing (`reports/stage_06d.md`, section 7).

| Criterion (initial target of `docs/PLAN.md`) | Met | Value [95% CI by store] |
|---|---|---|
| Brand accuracy (Coca-Cola, Sprite, Fanta) >= 90% | **no, by 0.5 point (77 of 86 crops; 78 would be 90.7%)** | 89.5% [78.4, 97.2] |
| Product top-3 >= 80% | yes | 94.2% [88.9, 98.1] |
| False known rate < 5% | yes | 0.0% [0.0, 0.0] (0 of 67 unknown crops answered with a product) |
| Unknown handling measured | yes | below |
| Confusions between products named | yes | below |

**Identification (86 known crops):** top-1 76.7% [60.8, 85.3] (the most probable product), Coca variant top-1 78.3% [60.0, 89.3] (60 crops). Share of answers over the 153 crops labeled with a product or unknown: automatic 24.2% [15.8, 31.3], review 22.2%, unknown 53.6%. **35 of the 37 automatic answers are right** (confident wrong rate 5.4% [0.0, 15.4], 2 crops; the Stage 8 target is below 5%, to be measured on the demo set).

**Per product (counts; a percentage only from 10 crops; `coca_light` has 3 crops):**

| Product | Crops | Top-1 correct | Automatic and correct | Review | Rejected as unknown | Automatic and wrong |
|---|---:|---:|---:|---:|---:|---:|
| coca_classic | 20 | 17 (85%) | 13 (65%) | 3 | 3 | 1 |
| coca_classic_returnable | 9 | 8 | 6 | 2 | 1 | 0 |
| coca_zero | 28 | 20 (71%) | 7 (25%) | 15 | 5 | 1 |
| coca_light | 3 | 2 | 1 | 2 | 0 | 0 |
| sprite | 10 | 5 (50%) | 2 (20%) | 3 | 5 | 0 |
| fanta | 16 | 14 (88%) | 6 (38%) | 5 | 5 | 0 |

**Named confusions** (most probable product, no rejection rules): **`coca_zero` taken for `coca_light` 4 times** (the largest; they differ by small text on a similar label), `coca_zero` for `coca_classic_returnable` 2 and for `coca_classic` 1, `sprite` taken for `coca_zero` 2, for classic 1, returnable 1 and light 1, `fanta` taken for `coca_zero` 2. The two confusions that dominated before (`coca_zero` to `coca_classic_returnable` 8 times, `coca_classic` to returnable 6 times in catalog (b)) are now 2 and 1. Remaining weak spots: `sprite` (5 of 10 correct, 5 rejected as unknown) and `fanta` (5 rejected): few training crops (28 and 10). Contact sheets of every wrong val crop, grouped true to predicted, are in `data/review/stage6d_errors/` (ignored).

**Unknown handling:** of 67 truly unknown crops, 63 are answered `unknown`, 4 `review`, **0 automatically**: strictly unknown rejection 94.0% [92.2, 97.8], unknown-or-review 100%. **Without store `s03`** (30 of the 67 unknown crops): 37 unknown crops, strictly unknown 94.6%, false known 0.0%, top-1 75.7%; the same if the thresholds are also never tuned on s03: one store does not drive the conclusion. The unknown class is partly learned from similar shelves (see the caveats). **Not targets:** of 15 `not_a_bottle` crops 13 unknown, 2 review, 0 automatic; of 12 `cannot_tell` crops 7 unknown, 5 review, 0 automatic.

**Candidate lists** (`evaluate_answer_types.py --scorer classifier`, delta calibrated leaving one store out for 90% coverage, at most 3): coverage 91.9% [85.5, 97.1], mean list size 2.62 [2.15, 3.32] (sizes 1 / 2 / 3+: 58% / 7% / 35%), against the catalog similarity list 90.7% and 3.69; so the demo uses the probability list (rule declared beforehand). With candidates in place of review: of the 86 known crops 37 automatic, 23 candidate lists (the right product inside for 18), 7 review, 19 unknown; of the 67 unknown crops 4 get a list. Brand answers are off (owner decision).

**Progress against the earlier systems on the same 180 crops:**

| | Top-1 (known) | Brand | Strictly unknown | False known | Automatic answers (correct / total) | Mean utility |
|---|---:|---:|---:|---:|---|---:|
| 6b: 30 originals, nearest entry | 53.5% | 88.4% | 26.9% | 0.0% | 0 / 0 | 0.175 |
| 6c: catalog (b), nearest entry | 54.7% | 87.2% | 74.6% | 3.0% | 0 / 2 | 0.190 |
| 6d: logistic regression on CLS (adopted) | **76.7%** | **89.5%** | **94.0%** | **0.0%** | **35 / 37** | **0.444** |

## 5. Caveats (read before relying on any number)
1. **One small sample.** 180 val crops from 6 stores; the intervals are wide (brand accuracy [78.4, 97.2]). `coca_light` has 3 crops (counts only), `sprite` 10, `fanta` 16. The thresholds are tuned on the same sample leaving one store out, which limits but does not remove the optimism.
2. **Similar chains across train and val.** The shelf entries (train stores) and the val crops (val stores) come from similar shelves of similar chains, so the learned `unknown` class and the learned variants may do worse in a new chain. Only the future demo set (and Stage 8 on test) can tell.
3. **Small training set for the weak products:** `fanta` 10, `coca_light` 17, `coca_zero` 21 training crops; the targeted pick found only 3 `coca_light` of 34 targeted.
4. **Labels.** The shelf and val product labels are the owner's single pass (no second labeler); some crops show promo wraps with two brands, which are inherently ambiguous. `not_a_bottle` was 8.3% of the val crops: some boxes are not bottles, and the detector's misses are not measured here.
5. **The classifier is a trained model, not a catalog:** the plan's "embeddings replace training a classifier" no longer holds for the scorer. The 30 studio photos are not used by it and never won in the catalog (c).
6. **Catalog photo sources.** `sources.csv` is still `unknown` for most rows (owner task, not blocking). `python scripts/check_catalog_photos.py --final` fails until it is filled in, so **no FINAL report can be written yet** (this is a development summary). The catalog photos are internet product photos to be replaced before any commercial use.
7. **Optimistic demo.** The demo thresholds and the train photos are optimistic (they use data the numbers above are about); the honest numbers are the leave-one-store-out ones in this report.
8. Why the classifier beats nearest-entry search was not tested (hypothesis: it learns which embedding directions separate the products instead of trusting the global similarity).

## 6. How to verify the gate (inside `soda-poc`; Postgres only for the fallback scorer)
```bash
python -m pytest -q                                           # 177+ tests
python scripts/fit_classifier.py                              # refit and re-save the model (deterministic) with its model card
python scripts/evaluate_stage6_gate.py                        # section 4: criteria, per-product counts, confusions, s03, non-targets
python scripts/evaluate_answer_types.py --scorer classifier   # candidate lists
python scripts/make_error_sheets.py                           # the wrong val crops, data/review/stage6d_errors/
python scripts/run_stage6d.py                                 # the 6d experiments (keep-rules in reports/stage_06d.md)
python scripts/demo_identify.py --image data/demo/inputs/val_example_a.jpg    # the demo with the classifier
```
Earlier steps: `reports/stage_06a.md`, `stage_06b_prep.md`, `stage_06b.md`, `stage_06c.md`, `stage_06d.md`.

## 7. Decision for the owner
Confirm that Stage 6 is accepted (or ask for more: for example more labeled shelf crops of sprite, fanta, zero and light, or a fix for the 0.5-point brand miss) before Stage 7 (OCR as evidence). My view: accept; Stage 7 has a clear, smaller job (`coca_zero` vs `coca_light`, and sprite/fanta text) and the pipeline already runs without OCR as variant A.
