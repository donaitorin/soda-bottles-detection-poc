# Stage 4 report: baseline without training, and the evaluation code

> **Every number below is agreement with Grounding DINO pseudo-labels, not accuracy.** The val labels are the unverified Grounding DINO boxes (owner decision, Stage 3). "High" means "similar to Grounding DINO", including its mistakes. No accuracy figure exists until the owner-corrected demo set does. Test was not touched.

## What was built

| Piece | File | Role |
|---|---|---|
| Evaluation module | `src/evaluation/detection.py` | matching from pycocotools' `COCOeval`; predictions under the 1.5% height floor are removed before matching (so they are neither matched nor false positives) and counted; `summarize` re-implements COCO's accumulation per photo so a bootstrap can resample whole stores; operating-point precision/recall/F1; store bootstrap |
| Predictions | `scripts/predict_val.py` | Faster R-CNN (raw and with the Stage 2 filters) and frozen Grounding DINO on the 41 val photos, saved to the ignored `data/predictions/stage4/` |
| Evaluation | `scripts/evaluate_detections.py` | tables below, per store, per box-count bin, checks against `COCOeval` |
| Error look | `scripts/stage4_error_look.py` | drawings and a breakdown of the disagreements |
| Config | `configs/stage4.yaml` | thresholds, runs, bootstrap size, drawn photos |
| Tests | `tests/test_evaluation.py` (7) | perfect/empty/missed/extra predictions, floor ignoring, own accumulation equals `COCOeval` on noisy data, stores travel together in the bootstrap |

New dependency (approved): `pycocotools==2.0.11` (BSD-2-Clause, wheel from PyPI, requires only `numpy`, which was already installed; `pip check` is clean).

How to read the numbers (for learning): **AP@0.5** is the area under the precision-recall curve when a prediction counts as correct if its box overlaps a label by IoU of at least 0.5; **AP@0.5:0.95** averages that over stricter overlaps (0.5, 0.55, ... 0.95), so it also rewards tight boxes; **recall** is the share of labels found; **precision** is the share of predictions that match a label. Precision, recall and F1 need one score cut, so they are given at the Faster R-CNN score of 0.3 already used in Stage 2; AP uses all predictions down to score 0.05. The 95% intervals come from resampling the 6 val stores with replacement (1,000 times, seed 0); with so few stores they are wide and only a rough guide.

## Sanity check of the pipeline (Grounding DINO frozen on val, against its own pseudo-labels)

Not a baseline: the model is compared with itself. Result: 2,157 boxes predicted, TP 2,157, FP 0, FN 0, **AP and recall 100.0** in every metric. This confirms the whole chain (COCO export, prediction script, floor ignoring, matching, accumulation) at the level of identity. Also, the own accumulation reproduces `COCOeval`'s AP and AP@0.5 with a difference of 0 on the real data, for all three runs.

## Faster R-CNN with COCO weights, no training (class `bottle`), val: agreement with Grounding DINO

41 photos, 2,157 pseudo-label boxes, 6 stores. The "filtered" row applies to the Faster R-CNN output the same filters (oversized, grey wedge, tiny, NMS, containment) that produced the pseudo-labels, so part of its higher precision is by construction.

| Agreement, % [95% CI by store] | Raw output | With Stage 2 filters |
|---|---|---|
| AP@0.5:0.95 | 68.5 [61.4, 71.8] | 68.6 [61.8, 71.9] |
| AP@0.5 | 88.6 [79.6, 91.4] | 88.4 [81.4, 91.5] |
| Recall at IoU 0.5 (all scores down to 0.05) | 94.3 [88.5, 96.1] | 92.4 [86.4, 94.9] |
| Precision, score >= 0.3, IoU 0.5 | 69.6 [60.8, 73.1] | 76.8 [72.1, 79.6] |
| Recall, score >= 0.3, IoU 0.5 | 87.0 [78.4, 90.0] | 86.1 [78.0, 89.6] |
| F1, score >= 0.3, IoU 0.5 | 77.3 [69.0, 80.2] | 81.2 [75.3, 84.0] |
| Predictions kept / ignored under the 1.5% floor | 5,569 / 60 | 3,820 / 0 (the filters already include the floor) |
| TP / FP / FN at score >= 0.3 | 1,877 / 820 / 280 | 1,858 / 561 / 299 |

Per store (with Stage 2 filters; agreement, %):

| Store | Photos | Pseudo-label boxes | AP@0.5 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| s01 | 4 | 204 | 88.4 | 67.0 | 87.7 |
| s03 | 13 | 860 | 90.4 | 78.2 | 88.6 |
| s06 | 7 | 117 | 64.2 | 69.3 | 59.8 |
| s11 | 6 | 298 | 94.7 | 83.0 | 93.3 |
| s15 | 9 | 592 | 87.0 | 77.0 | 85.0 |
| s16 | 2 | 86 | 85.6 | 74.2 | 76.7 |

Per photo density (photos split into three bins by number of pseudo-label boxes; with Stage 2 filters; agreement, %):

| Bin | Photos | AP@0.5 | Precision | Recall |
|---|---:|---:|---:|---:|
| 1 to 39 boxes | 13 | 66.0 | 44.7 | 68.8 |
| 40 to 64 boxes | 14 | 90.8 | 79.3 | 88.4 |
| 65 to 101 boxes | 14 | 91.3 | 86.1 | 88.7 |

What this tells us: on dense bottle shelves the COCO model agrees with Grounding DINO about 90% (AP@0.5), and the loose agreement (AP@0.5:0.95 near 69) says its boxes are less tight than Grounding DINO's. It agrees much less on sparse photos and in store `s06`, which is where the drinks are not plain bottles (see below). These numbers are the **comparison point for Stage 5**, nothing more. They do not say how good the COCO model is at finding bottles.

## Error look (Faster R-CNN, filtered, score >= 0.3, IoU 0.5; val)

Drawings of 9 photos are in the ignored `data/review/stage4_errors/` (green = agrees, red = Faster R-CNN box with no pseudo-label, blue = pseudo-label missed by Faster R-CNN). Breakdown of the disagreements, which adds up to the table above (561 extra, 299 missed):

| Disagreement | Count | Meaning |
|---|---:|---|
| Extra box, no pseudo-label with IoU of at least 0.1 | 257 | Faster R-CNN says bottle where the pseudo-labels have nothing |
| Extra box, overlapping a pseudo-label with IoU below 0.5 | 304 | same object, box placed differently (partial or occluded bottles, tight vs loose boxes) |
| Missed pseudo-label, no prediction near it | 151 | |
| Missed pseudo-label, a prediction overlaps it with IoU below 0.5 | 148 | |
| Missed pseudo-labels shorter than 3% of the image height | 19 of 299 | small bottles are **not** the main cause of misses |

What the drawings show:
- **Cans.** Photo `s03_05caed03` (a shelf of energy-drink cans) has 81 extra boxes against 15 pseudo-labels: the COCO `bottle` class also fires on cans, and Grounding DINO with the prompt "bottle." mostly does not. This is the largest single source of "extra" boxes and a difference in what the two models call a bottle, not an error of either in the abstract. Cans are out of scope in this project.
- **The can-only val photo** `s06_b8c050ce` (a pseudo-label false positive, from the Stage 3 report): the pseudo-label is a box on one can multipack, shown in blue; Faster R-CNN does not box it (score below 0.3) and adds one extra box at the left edge. Here the baseline is arguably closer to the truth than the pseudo-label, but it counts as a miss, which is the point of the warning above.
- **Dense bottle shelves** (`s15_5bcb84b5`): most boxes agree (green). The differences are narrow boxes on bottles partly hidden behind others, boxes around bottle caps, and slightly different widths for neighbouring bottles. Most of the 304 + 148 "poor overlap" cases look like this: both models see something, and disagree about its extent.
- Some pseudo-label boxes sit on fragments or on can packs (the Stage 3 findings); they appear as blue "misses" of Faster R-CNN.

## Limits

- Agreement, not accuracy (see the top). The three runs share the same 41 photos from 6 stores.
- The intervals are wide because only 6 stores are resampled. A difference between two models smaller than the interval width is not a clear difference; Stage 5 proposes a paired comparison.
- The "filtered" variant benefits from the fact that the pseudo-labels went through the same filters.
- Faster R-CNN was evaluated at score thresholds chosen in Stage 2 (0.3) and by convention (0.05 for AP); no tuning on val.

## How to verify the gate (inside `soda-poc`)

```bash
python -m pytest -q
python scripts/predict_val.py --model frcnn
python scripts/predict_val.py --model dino
python scripts/evaluate_detections.py      # expect the Grounding DINO check at 100.0 and the tables above
python scripts/stage4_error_look.py
git status --short                          # nothing under data/ except splits/*.txt
```
