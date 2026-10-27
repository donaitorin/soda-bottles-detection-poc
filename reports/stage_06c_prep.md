# Stage 6c (prep): shelf examples round, candidate answers, mean-pooling check

> **Every number here is a DEVELOPMENT number: agreement with the owner's labels on 180 val crops from 6 stores (leave one store out), not accuracy.** The test split is untouched. Nothing in this report measures the shelf examples yet: the train crops are not labeled. The tooling is built and tested; the measurements wait for the labels.

## 1. Mean pooling sanity check (`python scripts/check_pooling.py`)

The 12.8% top-1 of mean pooling is real, and the owner's guess is right: **the white padding dominates the mean.** A bottle crop is tall and narrow, so **62% of the 224x224 square is padding** (median, for catalog crops and shelf crops alike). The mean of all patch tokens is mostly the mean of white patches, so all vectors look alike: with plain mean pooling **92% of the shelf crops have the same nearest catalog entry** (`sprite_03`). One part of the guess is not the cause: catalog and shelf crops have the SAME padding share (0.62), so it is not a catalog-versus-shelf difference; it is the padding itself.

Fix tested: `pooling: mean_content` (new, `src/embeddings/embed.py`, `content_patch_mask`): the mean over the patches whose centre lies on the crop (padding patches left out). Same protocol (30 originals, thresholds tuned leaving one store out, 86 known crops):

| Pooling | Top-1 | Top-3 | Brand | Strictly unknown | False known | Most common nearest entry |
|---|---:|---:|---:|---:|---:|---|
| cls | 53.5 | 90.7 | 88.4 | 26.9 | 0.0 | 14% of crops |
| mean (all patches) | 12.8 | 55.8 | 12.8 | 10.4 | 1.5 | **92% of crops** |
| mean_content | 47.7 | 84.9 | 87.2 | 17.9 | 1.5 | 15% of crops |

The bug explains the collapse: fixed, mean pooling goes from 12.8% to 47.7%. It is still below CLS on every metric (point estimates, one sample), so **CLS stays**, as agreed ("unless the fixed mean pooling is clearly better"). The Stage 6b statement "mean pooling is far worse" was true but for a mechanical reason, not because the method is weak.

## 2. The train crop round (tooling built, labeling pending)

`python scripts/build_train_crop_round.py` (seed 0; train photos only: it asserts every photo is in `data/splits/train.txt`) built **300 crops from 168 train photos, 7 train stores**: 200 random (stratified by store, at most 8 per photo, like the val sample) and 100 targeted. Targeted = for each of `coca_light`, `coca_classic_returnable` and `sprite` (the products with 3, 9 and 10 labeled val crops), the crops whose top-3 against the 30 originals include that product, most similar first, at most 4 per photo and 35% per store: 34 + 34 + 32. The labeler cannot tell random from targeted and sees no suggestion. Retrieval only finds candidates; the owner's label decides, and the export script reports how many of each product actually came out (overall, by random/targeted, and for each targeted product how many really are that product).

Per store: s04 33, s05 54, s07 59, s12 31, s17 35, s18 33, s19 55. What the 30 originals suggest for all 300 (not labels): coca_classic 96, sprite 70, coca_zero 40, fanta 39, returnable 32, light 23. Expect fewer targeted `coca_light` than 34: the retrieval is weak (that is the domain gap), which is why the label counts are reported afterwards.

## 3. Class balance (decision, made before seeing any labeled number)

Zero and classic will dominate the shelf examples. Three places could favour a product just for having more entries, and each has a fix (`src/embeddings/shelf_catalog.py`, `identify.features_from_sims`):
- **Product score = best entry.** A maximum grows with the number of tries, so a product with 60 entries beats one with 5 by chance. Fix: **cap per product** (`cap_per_product: 25`, seeded random subset, independent of input order); rare products keep all entries. Unknown negatives have their own cap (40).
- **k-NN vote.** Counting the 5 nearest entries favours the product with more entries. Fix: **each neighbour counts 1 / (entries of its product)** (`balance_vote: true`).
- **Rules.** Margin and similarity rules use only the per-product best score, so they inherit the two fixes above; nothing else depends on entry counts.
Why these and not "best similarity only": the best similarity per product is what we already use and it is the biased quantity; the cap bounds its bias and the weighted vote removes the other one. The comparison script prints a table of how often each product is the nearest answer against its share of the labels, and a sensitivity table (cap and balanced vote, no cap and plain vote, each alone) so the choice is checked, not assumed. Residual bias remains (25 versus 5 entries still differ); the sensitivity table shows how much.

## 4. The three comparisons (`python scripts/evaluate_shelf_catalog.py`, after the labels)

Paired bootstrap by val store on the same 180 crops, thresholds re-tuned leaving one store out:
(a) originals vs originals + shelf examples; (b) (a) vs the same with the crops labeled `unknown` in train added as negative entries (`product_id` unknown; a crop closer to a negative than to its top-1 product by more than `t_neg`, tuned like the other thresholds, is unknown; negatives also take part in the vote). **(b) is kept only by a rule declared before any number exists:** strict unknown rejection up by at least 3 points, and top-1, brand and automatic-correct not down by more than 2 points (point estimates) with no paired interval entirely below zero. (c) originals + shelf vs shelf only (no originals, no negatives).
The script was checked end to end on RANDOM fake labels kept in the scratchpad (numbers near chance, as they must be; nothing fake was written into `data/`). Unit tests cover the cap, the catalog modes, empty products, the negative gap and the balanced vote (`tests/test_shelf_catalog.py`).

## 5. Candidate answers, measured now on the 30 originals (`python scripts/evaluate_answer_types.py`)

Calibration per held-out store on the other stores' known crops. Delta (the list distance) is about 0.02 for every store (0.016 to 0.023); the brand margin `t_bm` is 0.02 to 0.04.

**Variable-size candidate list** (every product within delta of the top-1; delta calibrated for 90% coverage), 86 known crops:

| Method | Coverage | Mean size | Size 1 | Size 2 | Size 3+ | Lists of 2+ inside one brand |
|---|---:|---:|---:|---:|---:|---:|
| top-1 only | 53.5% | 1.00 | 100% | 0 | 0 | n/a |
| fixed top-2 | 77.9% | 2.00 | 0 | 100% | 0 | 73.3% |
| fixed top-3 (today's review list) | 90.7% | 3.00 | 0 | 0 | 100% | 67.4% |
| calibrated list | **91.9% [85.7, 96.4]** | 3.23 [2.97, 3.43] | 9.3% | 14.0% | 76.7% | 69.2% (78 lists) |

The calibration works (actual coverage 91.9% for a 90% target), but **the list is no shorter than the fixed top-3**: similarities of the correct and the wrong products sit within about 0.02 of each other (the domain gap of Stage 6b), so covering 90% needs about three products. Only 9% of the lists have one product, 14% two. When a list has 2 or more products it stays inside one brand 69% of the time (mostly Coca-Cola variants).

**Two-level answer** (brand clear: margin >= `t_bm`; then the brand plus its variants inside the list): applies to 50 of 86 known crops (58%), 49 of them `review` today. The brand is right for 46 of 50 (92.0%) and the correct variant is among the shown ones in the same 46 (92.0%), but **the mean number of variants shown is 3.4 of the 4 Coca-Cola products**: "Coca-Cola" is a useful answer, "Coca-Cola: classic, returnable, zero or light" is barely more than the brand. 4 of 67 unknown crops would get a brand answer.

**Answer mix with `candidates` in place of `review`** (max list 3; counts): of 86 known crops 0 automatic, 49 brand answers, 18 short lists, 11 review, 8 unknown; the correct product is inside the candidates for 63 of the 67 candidate answers. **Cost: 33 of the 67 unknown crops get a short list** (they were `review` or `unknown` before): a list for a bottle that is not any target product is misleading, so the demo colours it blue like the others; the owner should weigh this. `not_a_bottle`: 14 of 15 still unknown, 1 gets a brand answer.

Reading: both answer types are honest and calibrated, but with the current catalog they mostly restate "it is a Coca-Cola, one of several variants". Whether the shelf examples shrink the lists is exactly what the second step measures (`python scripts/evaluate_answer_types.py --catalog originals+shelf`).

**In the demo tool** (new blue `candidates` type; `demo.candidates` in `configs/stage6.yaml`, delta 0.0206 and `t_bm` 0.03 calibrated once on all 86 known crops, so optimistic): on the val photo 78 boxes: 0 automatic, 43 candidates, 15 review, 20 unknown; train photo 63: 0, 16, 3, 44. The panel shows "Coca-Cola: which one?" with the variants and similarities, or "one of:" with the short list.

## 6. Labeling steps for the owner (project `soda-bottles-crops-train`)

Same tool as the val crops, a SEPARATE project, so the val project and its export stay untouched. Label Studio at http://localhost:8080 (`docker compose up -d` if it is not running).
1. **Create project:** name `soda-bottles-crops-train`.
2. **Labeling Interface > Code:** paste the content of `configs/label_studio/crop_project.xml` (unchanged), Save.
3. **Settings > Cloud Storage > Add Source Storage > Local files:** Absolute local path `/label-studio/files/labelstudio/crops`, leave "Treat every bucket object as a source file" OFF, Add Storage (do not press Sync).
4. **Import:** upload `data/crops/train_shelf/tasks.json` (300 tasks).
5. **Label** with the keys 1 to 6, `u`, `n`, `c` and Ctrl+Enter, no skipping (about 17 minutes at the val pace of 3.4 s per crop).
6. **Export:** Export > **JSON** (not COCO) and save it as exactly `data/labelstudio/exports/crop_train_shelf.json`.
Then tell me the labeling is done; I run `python scripts/export_crop_labels.py --ls-export data/labelstudio/exports/crop_train_shelf.json --name train_shelf` (audit, counts per product, random vs targeted) and continue with the measurements.

## 7. Verify (inside `soda-poc`)

```bash
python -m pytest -q                                  # 166 tests
python scripts/check_pooling.py                      # section 1
python scripts/evaluate_answer_types.py              # section 5
python scripts/demo_identify.py --image data/demo/inputs/val_example_a.jpg   # blue candidates in the panel
# after the labels:
python scripts/export_crop_labels.py --ls-export data/labelstudio/exports/crop_train_shelf.json --name train_shelf
python scripts/evaluate_shelf_catalog.py             # the three comparisons
python scripts/evaluate_answer_types.py --catalog originals+shelf
```

## Problems found
- The owner's "12.8% is below chance" was right to be suspicious: see section 1 (padding, now explained and fixed in a new pooling mode).
- The database still holds the 30 originals (decision 4); `build_catalog.py` is unchanged.
