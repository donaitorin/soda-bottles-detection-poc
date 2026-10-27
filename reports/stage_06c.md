# Stage 6c: shelf examples in the catalog, candidate answers per catalog

> **Every number is a DEVELOPMENT number: agreement with the owner's labels on 180 val crops from 6 stores (thresholds tuned leaving one store out), not accuracy.** Intervals are paired bootstraps that resample whole val stores (6 stores: wide, a rough guide). The test split is untouched. The database still holds the 30 originals; nothing was changed there.

## 1. The labels of the train round (`export_crop_labels.py --name train_shelf`)
Audit clean (300 of 300, one valid label each, ids match). Median 1.8 s per crop, 14.0 min in total. Labels: coca_classic 61, coca_classic_returnable 38, coca_zero 21, coca_light 17, sprite 28, fanta 10, unknown 78, not_a_bottle 11 (3.7%), cannot_tell 36. The 253 labeled with a product or unknown are the shelf entries (`not_a_bottle` and `cannot_tell` are not used).

| How picked | Crops | classic | returnable | zero | light | sprite | fanta | unknown | not a bottle | cannot tell |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 200 | 14 | 19 | 21 | 14 | 4 | 7 | 74 | 11 | 36 |
| targeted | 100 | 47 | 19 | 0 | 3 | 24 | 3 | 4 | 0 | 0 |

**The targeted pick worked for sprite and returnable and failed for light:** of 34 crops picked for `coca_light`, **3** are light (31 are `coca_classic`); of 34 picked for `coca_classic_returnable`, 18 are; of 32 for `sprite`, 24 are. Retrieval against the studio photos cannot find light (the domain gap again). Shelf entries per product are very uneven: classic 61 against fanta 10 and zero 21 (zero is rarer in train than in val: 21 of 253 against 28 of 86 known val crops). The cap (25 per product, 40 unknown) therefore only touches classic and unknown.

## 2. The three comparisons (`python scripts/evaluate_shelf_catalog.py`)
Catalogs: originals 30 entries; originals + shelf 153 (classic 30, returnable 30, zero 26, light 22, sprite 30, fanta 15); + negatives 193 (40 unknown); shelf only 123.

| Metric (%) | originals | (a) + shelf | (a) minus originals [95% CI] |
|---|---:|---:|---|
| Top-1 (known) | 53.5 | 54.7 | +1.2 [-16.2, +16.5] |
| Top-3 | 90.7 | 83.7 | -7.0 [-19.3, +1.6] |
| Brand | 88.4 | 87.2 | -1.2 [-8.3, +3.8] |
| Strictly unknown | 26.9 | 29.9 | +3.0 [-4.4, +20.8] |
| False known | 0.0 | 4.5 | +4.5 [0.0, +8.5] |
| Automatic and correct | 0 of 86 | 0 of 86 | 3 automatic answers appear, all 3 wrong |
| Mean utility | 0.175 | 0.020 | -0.16 [-0.3, 0.0] |

**(a) Shelf examples alone do not help.** Top-1 is flat, top-3 and utility are lower, and the first automatic answers appear and are wrong (3 of 3). Per product (counts of correct top-1, originals to originals + shelf): classic 14 to 13 of 20, returnable 2 to 5 of 9, zero 12 to 11 of 28, light 0 to 1 of 3, sprite 8 to 4 of 10, fanta 10 to 13 of 16. Where the wrong ones go: **8 of the 28 `coca_zero` crops now go to `coca_classic_returnable`** (38 returnable entries, 14 of them from one pallet-and-crates store), and sprite loses 4. The shelf entries make crops look alike across variants instead of separating them.

**(b) Negatives: kept, by the rule declared beforehand.** Strictly unknown rejection **29.9% to 74.6%, +44.8 points [+22.6, +58.1]** (excludes zero), unknown-or-review 95.5 to 97.0, false known 4.5 to 3.0, mean utility 0.020 to **0.190 [+0.1, +0.3 over (a)]**, known top-1, top-3, brand unchanged (a negative never changes which product is nearest). Against the ORIGINAL catalog (utility 0.175, strict unknown 26.9%): utility +0.015, strictly unknown +47.7 points, top-1 +1.2, top-3 -7.0. So the useful part of the shelf round is the unknown examples, not the product examples. Caveat: the 78 unknown train crops and the val unknown crops are from different stores but similar shelves (other brands in the same chains); the gain may be smaller in a new store (only a held-out store tells; the test split does that in Stage 8).

**(c) Shelf only: identical to (a) on every metric.** Not a bug: with shelf examples present the studio photo is almost never the best entry (best entry is an original for 0.8% of product-level scores; median best similarity of a val crop to the originals 0.69 against 0.82 to the shelf crops), so removing the 30 originals changes nothing. The studio photos contribute nothing once shelf examples exist, at this size.

**Class balance check.** Share of nearest-product answers on the 86 known crops against label share: classic 23.3% vs 23.3%, returnable 29.1% vs 10.5%, zero 15.1% vs 32.6%, light 8.1% vs 3.5%, sprite 4.7% vs 11.6%, fanta 19.8% vs 18.6%. The over- and under-answered products are NOT the ones with more or fewer entries (returnable 30 entries like classic; zero, with 26, is under-answered), so entry counts are not the explanation. Sensitivity (a): cap + balanced vote 54.7 top-1; no cap and plain vote 55.8; cap only 54.7; balanced vote only 55.8: the choice moves top-1 by about one crop (1 of 86), no other metric moves. The cap and the weighted vote were harmless; they were not what mattered.

## 3. Candidate answers per catalog (`evaluate_answer_types.py --catalog ...`; delta calibrated for 90% coverage leaving one store out; max list 3)

| Catalog | List coverage | Mean list size | Size 1 / 2 / 3+ | Lists of 2+ in one brand | Two-level applies (known) | Known crops with candidates | **Unknown crops with candidates** | Unknown crops: automatic |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| originals | 91.9% | 3.23 | 9 / 14 / 77% | 69% | 50 of 86 | 67 of 86 | **33 of 67** | 0 |
| (a) originals + shelf | 90.7% | 3.69 | 8 / 13 / 79% | 47% | 1 of 86 | 26 of 86 | **21 of 67** | 3 |
| (b) + negatives | 90.7% | 3.69 | 8 / 13 / 79% | 47% | 1 of 86 | 25 of 86 | **10 of 67** | 2 |
| (c) shelf only | 90.7% | 3.67 | 8 / 13 / 79% | 47% | 1 of 86 | 27 of 86 | **21 of 67** | 3 |

With shelf examples the calibrated list gets longer (delta 0.05 against 0.02: the shelf-to-shelf similarities of the Coca-Cola variants are close, and the extra entries add confusion between them), and the two-level brand answer all but disappears (no brand margin reaches 95% brand precision except in one fold). What the negatives do for the candidates: unknown crops that get a candidate list fall from 33 (originals) and 21 (a) to **10** of 67, and 50 of 67 are strictly unknown; known crops with candidates fall to 25 of 86 (and 50 go to review, 11 to unknown).

**Recommendation (for you to decide):** (1) **Use catalog (b)** for the next demo and the database: its utility (0.190) and unknown rejection (74.6%) are clearly the best, product entries from the shelf are kept because they do no harm, the 30 originals can stay as a fallback (they never win, they cost nothing). (2) **Keep blue `candidates` with max_list 3, but expect little of it under (b):** it fires on 25 of 86 known crops and 10 of 67 unknown crops; the two-level brand answer only works with the ORIGINAL catalog (50 of 86 known crops, brand right 92%, but 3.4 of 4 variants shown). If you want brand answers, the original-only catalog is the one that gives them, at the price of 33 unknown crops getting lists. (3) The real remaining problem is **separating the Coca-Cola variants** (zero, classic, returnable, light): top-1 is about 54% whichever catalog is used; Stage 7 (OCR of the small label text) targets it, and more labeled shelf examples of zero and light would help (light has 17, zero 21, light targeting found 3).

## 4. Problems found
- Targeting by retrieval cannot find `coca_light` (3 of 34).
- The shelf-product examples alone slightly hurt (top-3, utility, 3 wrong automatic answers); the gain is entirely in the unknown negatives.
- `export_crop_labels.py` printed "pseudo-label precision on val" for a train sample: reworded to "in this sample".
- Nothing changed in the database; rebuilding it with shelf entries and negatives needs a small `build_catalog.py` extension (and the demo needs `t_neg`): a proposed next task, not done.

## 5. How to verify (inside `soda-poc`)
```bash
python scripts/export_crop_labels.py --ls-export data/labelstudio/exports/crop_train_shelf.json --name train_shelf
python scripts/evaluate_shelf_catalog.py
for c in originals originals+shelf originals+shelf+negatives shelf; do python scripts/evaluate_answer_types.py --catalog $c; done
```
