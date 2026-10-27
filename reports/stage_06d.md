# Stage 6d: three experiments on separating the Coca-Cola variants

> **Every number is a DEVELOPMENT number: agreement with the owner's labels on 180 val crops from 6 stores (thresholds and weights tuned leaving one store out), not accuracy.** Paired bootstrap by store (6 stores: wide intervals, a rough guide). Test untouched. Reference = catalog (b) (30 originals + 163 capped owner-labeled train shelf entries incl. 40 unknown negatives), DINOv2 CLS, 10% margin crops, balanced vote, the Stage 6b rules, thresholds tuned leaving one store out.

## 0. Keep-rules (written BEFORE any number of this stage was computed)

Why these experiments: with any catalog the top-1 is about 54% and the confusions are between the Coca-Cola variants (`coca_zero` crops go to `coca_classic_returnable`, `coca_classic` to returnable; light is never found). The experiments test three causes: colour is partly lost in the embedding (4a), neighbours and background drive the confusions (4b), the nearest-entry rule is a poor scorer for the embedding (4c).

**Metrics.** The target metric is **Coca variant top-1**: of the labeled crops whose truth is one of the 4 Coca-Cola products (60 crops: classic 20, returnable 9, zero 28, light 3), the share whose top-1 product is the right variant. Also reported: top-1 over all 86 known crops, top-3, brand accuracy, strictly unknown rejection, false known rate, automatic and correct counts, mean utility, per-product COUNTS (`coca_light` has 3 crops: counts only, never a percentage), and the classic / returnable / zero / light confusion matrix (rows truth, columns nearest product, with the rest grouped) for every variant.

**Keep-rule (the same for each experiment, applied to its final variant against the reference with the paired bootstrap by store).** An experiment is **kept** (adopted as a change to the pipeline) only if ALL hold:
1. Coca variant top-1 up by **at least +5 points** (point estimate);
2. top-1 over all known crops not down (point estimate >= 0);
3. mean utility not lower than the reference by more than 0.02, and false known rate not higher by more than 2 points (point estimates);
4. the paired 95% interval of the Coca variant top-1 difference is not entirely below zero.
It is called a **clear winner** only if, in addition, the paired 95% interval of the Coca variant top-1 difference excludes zero above. With 6 val stores this is a demanding bar; an experiment that is kept but not a clear winner is reported as "promising, not established".
Per experiment:
- **4a colour.** Step 1 (diagnostic, no keep-rule): the colour descriptor alone, scored like the embedding (nearest entry of each product in catalog (b)), on the 60 Coca crops: its 4-variant accuracy, next to the embedding's on the same crops. Step 2: the combined score `(cls + w * colour) / (1 + w)`, `w` tuned leaving one store out on top-1 of known crops (grid in `configs/stage6.yaml`), then the usual rules. The keep-rule applies to step 2.
- **4b context.** Compared separately with the reference: 0% margin crop; SAM 2 mask on a neutral (white) background. Each is applied to catalog and query crops alike (originals, shelf entries, val crops). Each is judged by the keep-rule on its own.
- **4c linear classifier.** Multinomial logistic regression on the 253 train shelf entries (6 products + unknown; class-balanced loss), regularisation chosen by leaving one TRAIN store out, evaluated on the 180 val crops. Unknown = predicted class unknown, or probability of the top product below a threshold; automatic / review thresholds tuned leaving one val store out (same utility). Two inputs: CLS alone, CLS + colour descriptor (if 4a shows the colour descriptor is informative). Judged by the keep-rule against the reference; the second input also against the first.
- **Combination.** Only variants that pass the keep-rule are combined, one step at a time, each step shown against the previous one with the same rule. Nothing is combined by default.

Implementation notes decided up front: scikit-learn is not installed, so 4c uses a small torch implementation (no new dependency; scikit-learn is BSD-3 and could be added on request); SAM 2 is `facebook/sam2.1-hiera-small` (Apache 2.0, pinned revision), used through `transformers` with the Grounding DINO box as prompt.

## 1. SAM 2 speed on MPS (`python scripts/sam2_timing.py`)
Loading takes 12 s once (download on first use, 0.8 s afterwards); then 0.2 to 0.5 s per photo for 6 to 8 boxes (27 to 60 ms per box) on MPS, the same order on CPU. Not too slow: all 463 images of the experiment (180 val crops, 253 shelf entries, 30 catalog photos) take a few minutes.

## 2. Results (`python scripts/run_stage6d.py`; the numbers below were computed after the rules above were written)

Reference: catalog (b), CLS, 10% margin: **Coca variant top-1 50.0%** (30 of 60), top-1 54.7% (86 known crops), top-3 83.7%, brand 87.2%, strictly unknown rejection 74.6%, false known 3.0%, 2 automatic answers (both wrong), mean utility 0.190. A control: re-cutting and re-embedding the crops differs from the Stage 6 vectors by at most 8e-3 (JPEG round trip of the catalog crops), and reproduces the reference numbers.

| Variant | Coca variant top-1 (diff, 95% CI by store) | Top-1 known (diff) | Strictly unknown | False known | Mean utility | Keep-rule |
|---|---|---|---:|---:|---:|---|
| reference (b) | 50.0% | 54.7% | 74.6% | 3.0% | 0.190 | - |
| 4a embedding + colour | 56.7% (+6.7 [-5.6, +20.0]) | 59.3% (+4.7) | 94.0% | 6.0% | 0.074 | NOT kept (utility, false known) |
| 4b 0% margin | 51.7% (+1.7 [-16.9, +21.1]) | 58.1% (+3.5) | 37.3% | 0.0% | 0.148 | NOT kept (Coca < +5, utility) |
| 4b SAM 2 mask, white | 60.0% (+10.0 [-9.3, +29.3]) | 60.5% (+5.8) | 71.6% | 1.5% | 0.167 | NOT kept (utility -0.023, by 0.003) |
| **4c logistic regression, CLS** | **78.3% (+28.3 [+19.2, +36.4])** | **76.7% (+22.1 [+13.4, +31.2])** | **94.0%** | **0.0%** | **0.444 (+0.25 [+0.1, +0.4])** | **KEPT, clear winner** |
| 4c logistic regression, CLS + colour | 85.0% (+35.0 [+26.9, +43.1]) | 77.9% (+23.3) | 94.0% | 6.0% | 0.367 | NOT kept (false known +3.0) |

### 4a Colour
**Step 1, the descriptor alone** (HSV histogram of the central part of the crop, 12x3x3 bins, Hellinger-normalised; nearest entry per product in catalog (b)): among the 4 Coca products it picks the right variant for **33 of 60 Coca crops (55.0%)**, the embedding for 31 of 60 (51.7%), always answering `coca_zero` would give 28 of 60. So a plain colour histogram separates the variants about as well as DINOv2: the hypothesis that colour is under-used by the embedding has some support, and the errors differ (colour confuses classic with returnable, 9 of 20 classic crops; the embedding confuses zero with returnable and light). Its 6-way top-1 over the 86 known crops is 55.8% (embedding 54.7%).
**Step 2, combined score** `(cls + w * colour) / (1 + w)`, `w` chosen leaving one store out (0.25 for five stores, 0.5 for one): Coca variant top-1 +6.7 [-5.6, +20.0], top-1 +4.7 [-3.7, +13.5], top-3 +7.0 [0.0, +15.4], strictly unknown +19.4 [+13.0, +28.6]; but false known 3.0% to 6.0%, mean utility 0.190 to 0.074, and 25 automatic answers on labeled crops (19 correct) where the reference had 2. **Not kept** by the rule (utility and false known). The threshold-free diagnostic (AUROC of the top-1 score for known against unknown crops) rises 0.758 to 0.822: the combined score separates known from unknown better, but the thresholds tuned on the other stores let too many wrong answers through on the held-out ones.

### 4b Context
Catalog originals, shelf entries and val crops were cut the same way in each variant. **0% margin:** nothing improves the variants (Coca +1.7 [-16.9, +21.1]) and the unknown handling gets much worse (strictly unknown 74.6% to 37.3%, an interval that excludes zero): the 10% margin helps, probably by keeping the bottle's outline and cap. **SAM 2 mask on white:** Coca variant top-1 +10.0 [-9.3, +29.3] (zero crops 11 to 17 of 28), top-1 +5.8, but top-3 -4.7, unknown rejection slightly lower and utility 0.190 to 0.167, which misses the rule's -0.02 limit by 0.003. **Not kept.** It is a hint (promising, wide interval), not a result. The masks look right (sample sheet `data/review/stage6d_sam/val_sam_sample.jpg`; small, dark or partial bottles get partial masks). Cropping context is therefore not the main cause of the confusions.

### 4c Logistic regression on the 253 shelf entries
Trained on the 253 owner-labeled train shelf entries (6 products + unknown, class-balanced loss, small torch implementation: scikit-learn is not installed; regularisation `C` chosen by leaving one TRAIN store out, 85.2% accuracy on the held-out train stores; `C` = 0.1), evaluated on the 180 val crops (different stores), automatic / review / unknown thresholds tuned leaving one val store out.
**CLS alone is the clear winner:** Coca variant top-1 **50.0% to 78.3%** (+28.3 [+19.2, +36.4]), top-1 54.7% to 76.7% (+22.1 [+13.4, +31.2]), top-3 83.7% to 94.2%, brand 87.2% to 89.5%, **strictly unknown 74.6% to 94.0%**, false known 3.0% to 0.0%, mean utility 0.190 to **0.444** [+0.1, +0.4], AUROC known-vs-unknown 0.758 to 0.946. It answers 24.2% of the labeled crops automatically (40.7% of the known crops) with **35 correct out of 37 automatic answers** (confident wrong rate 5.4%); the reference had 2 automatic answers, both wrong. Non-targets: 13 of 15 `not_a_bottle` crops unknown, 2 review, none automatic; 7 of 12 `cannot_tell` unknown, 5 review, none automatic.
Per product, top-1 correct (counts, reference to logistic regression): classic 13 to 17 of 20, returnable 5 to 8 of 9, zero 11 to 20 of 28, light 1 to 2 of 3 (counts only), sprite 4 to 5 of 10, fanta 13 to 14 of 16.
**Control:** the same pipeline with the shelf labels shuffled gives 19.8% top-1 on the 86 known crops (chance is about 17%): nothing leaks from the val labels into the model, which sees only train photos and train labels.
**With the colour descriptor added:** Coca variant top-1 85.0% (zero 23 of 28, light 3 of 3), but false known 6.0%, 54 correct of 62 automatic answers, utility 0.367: **not kept** (rule 3 against the reference: false known +3.0 points; against logistic regression on CLS alone the step is +6.7 [+2.9, +10.9] on the Coca variants but false known +6.0 [+2.2, +7.8] and utility -0.077). **Exploratory** (not a combination step: SAM 2 crops had failed their own rule): logistic regression on SAM 2 crops is no better than on plain crops (Coca 76.7% against 78.3%, utility 0.435, AUROC 0.909).

### Coca-Cola confusion matrices (nearest product, no rejection rules; counts; rows = truth, each cell: right variant first)
Cells are `classic / returnable / zero / light / sprite-or-fanta` answers.
| Variant | truth classic (20) | truth returnable (9) | truth zero (28) | truth light (3) |
|---|---|---|---|---|
| reference (b) | 13 / 6 / 1 / 0 / 0 | 3 / 5 / 0 / 1 / 0 | 3 / 8 / 11 / 4 / 2 | 0 / 2 / 0 / 1 / 0 |
| colour alone | 8 / 9 / 0 / 3 / 0 | 2 / 5 / 1 / 1 / 0 | 5 / 6 / 17 / 0 / 0 | 0 / 0 / 0 / 3 / 0 |
| 4a embedding + colour | 12 / 5 / 1 / 2 / 0 | 4 / 5 / 0 / 0 / 0 | 3 / 9 / 15 / 0 / 1 | 1 / 0 / 0 / 2 / 0 |
| 4b 0% margin | 12 / 5 / 3 / 0 / 0 | 4 / 5 / 0 / 0 / 0 | 5 / 6 / 12 / 4 / 1 | 0 / 1 / 0 / 2 / 0 |
| 4b SAM 2 mask | 11 / 4 / 2 / 2 / 1 | 2 / 6 / 1 / 0 / 0 | 3 / 3 / 17 / 3 / 2 | 1 / 0 / 0 / 2 / 0 |
| **4c logreg CLS** | **17 / 1 / 1 / 0 / 1** | **0 / 8 / 1 / 0 / 0** | **1 / 2 / 20 / 4 / 1** | **1 / 0 / 0 / 2 / 0** |
| 4c logreg CLS + colour | 17 / 2 / 1 / 0 / 0 | 0 / 8 / 1 / 0 / 0 | 1 / 2 / 23 / 1 / 1 | 0 / 0 / 0 / 3 / 0 |
| 4c logreg on SAM crops (exploratory) | 16 / 2 / 2 / 0 / 0 | 0 / 8 / 1 / 0 / 0 | 3 / 2 / 19 / 3 / 1 | 0 / 0 / 0 / 3 / 0 |
The zero to returnable confusion (8 times in the reference) falls to 2 with the logistic regression, and classic to returnable from 6 to 1. What remains: zero to light (4).

## 3. Combination
Only one variant passes its keep-rule (4c, CLS alone), so there is nothing to combine; the two additions tried on top of it (colour, SAM 2 crops) are shown above, each against the logistic regression, and neither passes.

## 4. Recommendation
1. **Replace the nearest-entry search by the logistic regression on the CLS embedding** (trained on the owner-labeled train shelf entries, products + unknown). It is the clear winner on every measure that matters: variants (+28 points), unknown handling (94% strictly unknown, 0% false known), and for the first time a usable automatic answer (35 of 37 right). The colour descriptor, the 0% margin and the SAM 2 crops should not be adopted now.
2. **Why it wins is not tested here.** Plausible: a linear classifier learns which directions of the embedding separate the products (label colour, text) instead of using the global similarity, and it uses the shelf entries as training data rather than as nearest neighbours. Not verified; do not read more into it.
3. **Caveats.** One sample: 180 val crops, 6 stores, `coca_light` has 3 crops (counts only), `fanta` 10 and `coca_zero` 21 train entries. The train and val shelves are similar chains in different stores: a new chain may do worse (only Stage 8 on the test split tells). The model uses the shelf entries only, not the 30 studio photos (they never won in (c)). Every number is agreement with the owner's labels on this sample, not accuracy.
4. **Next (needs approval):** (a) add the classifier to the demo (fit once on all 253 entries, saved to `models/`; the database catalog stays as a fallback and for the candidate lists); (b) re-measure the candidate list and two-level answers from the classifier's probabilities (a calibrated probability list may be shorter than the similarity list that was no shorter than top-3); (c) more labeled shelf examples of `coca_light`, `coca_zero` and `fanta` would help the weakest products; (d) with this result Stage 7 (OCR) has a smaller job: decide after the classifier is in the demo and its remaining errors are looked at.

## 5. How to verify (inside `soda-poc`)
```bash
python -m pytest -q
python scripts/sam2_timing.py
python scripts/run_stage6d.py            # all three experiments; --exp colour | context | logreg for one
```
Caches (ignored): `data/results/stage6d/{emb_margin10,emb_margin0,emb_sam,colour}.npz`, results in `data/results/stage6d/results.json`.
