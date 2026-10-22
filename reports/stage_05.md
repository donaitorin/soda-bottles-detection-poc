# Stage 5 report: Faster R-CNN fine tuning on Grounding DINO pseudo-labels

> **Every number in this report is agreement with Grounding DINO pseudo-labels, not accuracy.** The train and val labels are the unverified Grounding DINO boxes (owner decision, Stage 3). A high value means "behaves like Grounding DINO", including its mistakes; the detector was trained to imitate them. Test was not read by any Stage 5 script. Only the future owner-corrected demo set will give accuracy.

## Summary

- The full run finished without interruption or error (30 epochs, 2,566 s of training, about 85 s per epoch; the console log shows one line per epoch and no resume).
- **Gate: met, as agreement.** The fine-tuned model clearly beats the Faster R-CNN COCO baseline in the paired bootstrap by store (AP@0.5 +5.4 points, interval [3.7, 11.3], for `best.pt`; +3.0, [1.7, 6.3], for the last epoch, which was not selected on val), and agreement recall at score 0.3 is 94.2% (target 90%). The main disagreements were identified by cause (below): fragments and cans that the pseudo-labels contain and the model copies, and partial overlaps.
- **Val agreement peaks early** (epoch 5) and then declines slowly while the training loss keeps falling. It is discussed below: it is partly a change in how confident the model is, differences between epochs are small, and `best.pt` was chosen by val AP@0.5, which makes its val number optimistic. A shorter schedule is proposed.

## Training

Settings as approved (`configs/stage5.yaml`): `fasterrcnn_resnet50_fpn_v2` with COCO weights and a new 2-class predictor, 800 px short side, batch 2, SGD (momentum 0.9, weight decay 1e-4), lr 0.005 with 1-epoch warmup and cosine decay to 0.01 x lr, gradient clipping 10, 30 epochs, seed 0, augmentation: horizontal flip, colour jitter, scale 0.85 to 1.15. Checkpoints were saved every epoch; after the evaluation only `best.pt` (epoch 5) and `epoch_30.pt` were kept in `models/stage5/frcnn_v1/` and the 29 other epoch checkpoints were deleted, **freeing 9.8 GB** (506 MB remain).

Curves (numbers only, no photos): [`reports/figures/stage_05_training.svg`](figures/stage_05_training.svg), from `data/logs/stage5/frcnn_v1/train_log.csv` (`scripts/plot_training.py`). Selected epochs (val columns are agreement with Grounding DINO):

| Epoch | Train loss | Val AP@0.5 | Val AP@0.5:0.95 | Val recall at IoU 0.5 (all scores) | Val precision / recall at score 0.3 |
|---:|---:|---:|---:|---:|---|
| 1 | 1.198 | 0.898 | 0.721 | 0.950 | 0.722 / 0.900 |
| 5 (best.pt) | 0.341 | **0.940** | 0.782 | 0.975 | 0.732 / 0.942 |
| 10 | 0.264 | 0.937 | 0.793 | 0.964 | 0.816 / 0.919 |
| 20 | 0.216 | 0.915 | 0.787 | 0.939 | 0.860 / 0.902 |
| 30 (last) | 0.192 | 0.916 | 0.788 | 0.932 | 0.865 / 0.896 |

## Val agreement with Grounding DINO: fine-tuned model against the Stage 4 baseline

41 val photos, 2,157 pseudo-label boxes, 6 stores; 95% intervals by resampling whole stores (1,000 times). Predictions under the 1.5% height floor are ignored (best.pt raw: 9 ignored).

| Agreement, % [95% CI by store] | Baseline COCO, raw | Fine-tuned best.pt (epoch 5), raw | Fine-tuned last epoch (30), raw | Baseline COCO + filters | Fine-tuned best.pt + filters |
|---|---|---|---|---|---|
| AP@0.5:0.95 | 68.5 [61.4, 71.8] | 78.2 [72.4, 81.3] | 78.8 [72.2, 81.7] | 68.6 [61.8, 71.9] | 78.2 [72.7, 81.0] |
| AP@0.5 | 88.6 [79.6, 91.4] | 94.0 [90.6, 95.5] | 91.6 [85.9, 93.3] | 88.4 [81.4, 91.5] | 93.7 [90.7, 95.3] |
| Recall at IoU 0.5 (all scores) | 94.3 [88.5, 96.1] | 97.5 [96.8, 98.1] | 93.2 [88.8, 94.9] | 92.4 [86.4, 94.9] | 96.6 [94.8, 97.3] |
| Precision, score >= 0.3 | 69.6 [60.8, 73.1] | 73.2 [62.8, 77.7] | 86.5 [81.4, 91.5] | 76.8 [72.1, 79.6] | 77.2 [71.6, 82.3] |
| Recall, score >= 0.3 | 87.0 [78.4, 90.0] | 94.2 [91.3, 95.3] | 89.6 [83.1, 91.6] | 86.1 [78.0, 89.6] | 93.8 [90.4, 95.0] |
| F1, score >= 0.3 | 77.3 [69.0, 80.2] | 82.4 [74.4, 85.3] | 88.0 [82.4, 90.7] | 81.2 [75.3, 84.0] | 84.7 [79.9, 87.6] |

**The `best.pt` column is optimistic**: epoch 5 was chosen because it had the highest val AP@0.5 of 30 noisy checkpoints, so part of its advantage is selection. The last-epoch column (not selected on val) is the fairer comparison with the baseline.

Paired bootstrap of the difference by store (a minus b, percentage points; the same resampled stores are used for both models, so the interval measures the difference itself and not the store-to-store variation that both share):

| Comparison | AP@0.5:0.95 | AP@0.5 | Recall (all scores) | Precision, score >= 0.3 | Recall, score >= 0.3 | F1, score >= 0.3 |
|---|---|---|---|---|---|---|
| best.pt raw minus COCO raw | +9.8 [9.3, 11.7] | +5.4 [3.7, 11.3] | +3.3 [1.8, 8.5] | +3.6 [-1.6, 9.8] (zero inside) | +7.2 [5.2, 12.9] | +5.0 [2.4, 8.9] |
| best.pt + filters minus COCO + filters | +9.6 [9.0, 10.9] | +5.2 [3.5, 9.8] | +4.2 [2.2, 8.4] | +0.4 [-3.4, 4.9] (zero inside) | +7.6 [5.6, 12.3] | +3.5 [1.2, 6.8] |
| last epoch raw minus COCO raw | +10.3 [9.4, 11.6] | +3.0 [1.7, 6.3] | -1.0 [-1.8, 0.4] (zero inside) | +16.9 [11.4, 26.1] | +2.5 [0.9, 4.9] | +10.7 [7.6, 16.1] |
| best.pt raw minus last epoch raw | -0.5 [-1.9, 0.8] (zero inside) | +2.4 [1.7, 4.8] | +4.3 [3.1, 8.3] | -13.3 [-19.6, -10.9] | +4.6 [3.6, 8.1] | -5.6 [-8.4, -4.7] |

Reading: the fine-tuned model agrees more with Grounding DINO than the COCO model in AP@0.5 and AP@0.5:0.95 for every comparison (interval above zero), and the gap in AP@0.5:0.95 (about 10 points) says its boxes are much closer to Grounding DINO's boxes (it learned their extent). The intervals only reflect which stores were resampled (6 stores, 41 photos): they do not include the uncertainty of the pseudo-labels themselves, of the choice of epoch, or of repeating the training with another seed. Precision at score 0.3 is not a clean comparison, because the score scales of the models differ (see below).

Per store, `best.pt` raw (agreement, %): s01 AP@0.5 95.0; s03 94.5; s06 80.5 (the can-heavy store; baseline 61.9); s11 96.7; s15 93.4; s16 90.5. By number of pseudo-label boxes per photo: 1 to 39 boxes 75.3; 40 to 64 boxes 95.4; 65 to 101 boxes 96.8. The weak spot is still sparse photos and store `s06`, where precision is low (below 50 at score 0.3 in s06), because the model also boxes cans.

## The decline after epoch 5 (mild overfitting?)

What the log shows: val AP@0.5 is 0.898 at epoch 1, 0.940 at epoch 5 (the maximum), then drifts to a plateau (epochs 11 to 30: mean 0.918, standard deviation 0.004, range 0.908 to 0.924), while the training loss keeps falling (0.34 at epoch 5, 0.19 at epoch 30). Reading it carefully:

1. **The decline is small compared with the noise that matters.** From the early mean (epochs 2 to 10: 0.929) to the plateau (0.918) it is about 1.1 points. The store-bootstrap interval of a single model's AP@0.5 is about +-4 points (and 6 stores is very few). Epoch-to-epoch jitter on the plateau is about 0.4 points. The paired comparison of epoch 5 and epoch 30 does exclude zero for AP@0.5 (+2.4 [1.7, 4.8]), but **epoch 5 was picked as the best of 30 noisy values**, which inflates that gap; the interval cannot account for the selection.
2. **It depends on the metric.** AP@0.5:0.95, the box-extent metric, does not decline (epochs 2 to 10 mean 0.773, epochs 11 to 30 mean 0.786; epoch 5 against epoch 30: -0.5 points, zero inside the interval). What shrinks is the **tail of low-confidence detections**: raw detections above score 0.05 fall from 4,160 (epoch 5) to 2,609 (epoch 30), recall counting all scores falls from 97.5% to 93.2%, while precision at score 0.3 rises from 73% to 86.5% and F1 at 0.3 rises from 82% to 88%. The model becomes sharper and more confident, which AP@0.5 (and `best.pt`, chosen by it) reads as a loss and precision/F1 at a fixed score reads as a gain.
3. **So the evidence fits two explanations that this data cannot separate**: classic overfitting to the train pseudo-labels (the model memorises their errors; the overfit-10 check showed it can), or only a shift in score calibration as the loss falls. Training loss falling while the val AP@0.5 stays flat or slightly lower is the usual overfitting pattern, but the size of the effect is small and mixed across metrics.

**Proposal for future runs: a shorter schedule of 10 to 12 epochs**, with warmup 1 epoch and the cosine decay compressed to the new length (not the old schedule stopped early, which would end at a high learning rate). That is about 25 minutes on this Mac. Use the **last epoch** (or a choice by AP@0.5:0.95, which is stable across epochs) rather than the maximum of a noisy AP@0.5 on val, so that the val number is not inflated by selection. If more val stores become available, a comparison with the 30-epoch run on them would settle whether it is overfitting.

## Error analysis on val (fine-tuned best.pt, raw output, score >= 0.3, IoU >= 0.5)

(`python scripts/stage5_error_analysis.py`; drawings in the ignored `data/review/stage5_errors/`: green = agrees, red = extra prediction, blue = missed pseudo-label, the letter is the cause.)

Matched 2,032 boxes; **745 extra predictions** and **125 missed pseudo-labels**. The causes are automatic proxies, checked by eye on the drawings, not ground truth:

| Cause (proxy) | Extra predictions | Missed pseudo-labels |
|---|---:|---:|
| Can: the box lies on a can found by Grounding DINO with the prompt "soda can." (a lower bound: it finds only part of the cans) | 43 | 7 |
| Fragment: a small box (under 60% of the area) at least 80% inside a larger box | 296 | 20 |
| Partial overlap: overlaps a box on the other side with IoU 0.1 to 0.5 (occluded or overlapping bottles, or a different box extent) | 103 | 61 |
| Small bottle: shorter than 3% of the image height | 13 | 4 |
| Other: nothing near it on the other side | 290 | 33 |

What stands out:
- **The model imitates pseudo-label errors.** 288 of the 2,157 val pseudo-labels (13%) look like label errors by these proxies (166 fragments inside a larger box, which the labeling guide's 40% rule would not label, and 122 boxes on cans, a lower bound). The model **matched 261 of them** (146 fragments, 115 cans) and refused only 27. This is the Stage 5 expectation from the plan note, now measured. The fragment proxy can also hit small genuine bottles behind a larger one, so treat the 13% as an upper-bound estimate for fragments and a lower bound for cans.
- **Cans:** the largest single photo is a shelf of energy-drink cans: 121 of the 745 extras (16%) are in one photo, where the pseudo-labels have only 15 boxes. Most of its boxes fall under "other" because the can proxy finds only a part of the cans. Together with the sparse-photo and `s06` results, cans are the main reason precision is low in some photos.
- **Fragments:** the fine-tuned model also adds fragment boxes that are not in the pseudo-labels (296 extras), for example small boxes on caps of back-row bottles. The Stage 2 filters remove many of them (precision at 0.3 rises from 73.2 to 77.2 with the filters at the same recall).
- **Partial overlap (occlusion):** about 100 extras and 60 misses are one object seen with a different extent, in the drawings mostly bottles partly covered by their neighbours, and tight against loose boxes. Automatic analysis cannot separate occlusion from box extent; this is a joint cause.
- **Small bottles are not the problem:** 13 extras and 4 misses (under 3% of the image height).
- **Pseudo-label errors that the model refused:** 27 (20 fragments, 7 cans), i.e. cases where the model disagrees with an error.

## Limits

- All numbers are agreement with Grounding DINO, on 41 photos from 6 stores. The bootstrap covers store sampling only.
- `best.pt` is selected on val AP@0.5; its val number is optimistic. No test data was used.
- The cause proxies are heuristics; the can proxy is a lower bound.
- One training run (seed 0): no estimate of run-to-run variation.
- **Real Label Studio export check: pending**, by decision of the coordinator; it will first be exercised when the owner corrects the demo set. Until then the owner-annotation path of `scripts/export_coco.py` is covered only by tests on synthetic exports.

## How to verify (inside `soda-poc`)

```bash
python -m pytest -q
python scripts/predict_val.py --model frcnn --checkpoint models/stage5/frcnn_v1/best.pt --name frcnn_ft_best_val
python scripts/predict_val.py --model frcnn --checkpoint models/stage5/frcnn_v1/epoch_30.pt --name frcnn_ft_last_val
python scripts/evaluate_detections.py      # tables and the paired differences above
python scripts/stage5_error_analysis.py
python scripts/plot_training.py
git status --short                          # nothing under data/ or models/
```
