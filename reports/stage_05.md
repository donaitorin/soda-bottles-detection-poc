# Stage 5 report (in progress): Faster R-CNN fine tuning on pseudo-labels

> **Status: stopped before the full training run, by instruction.** Done: dataset, model, augmentations, learning-rate schedule, checkpoints with resume, MPS timing check, overfit-10 sanity check. Not done: the full run, the comparison with the Stage 4 baseline, the error analysis.
>
> **The training labels are Grounding DINO pseudo-labels, so the detector will imitate Grounding DINO, errors included, and every val number is agreement with Grounding DINO, not accuracy.** Test is not read by any Stage 5 script.

## What was built

| Piece | File | Role |
|---|---|---|
| Dataset | `src/detection/dataset.py` | reads the COCO file; COCO `[x, y, w, h]` to torchvision xyxy pixels (tested); label 1 = `bottle`; photos with no boxes allowed; augmentations that move boxes with the image |
| Model | `src/detection/model.py` | `fasterrcnn_resnet50_fpn_v2` with COCO weights, new 2-class box predictor, dense-shelf settings (300 detections, RPN counts raised) |
| Engine | `src/detection/engine.py` | warmup + cosine learning rate (tested), one training epoch with a finite-loss check, prediction, agreement metrics (same evaluation code as Stage 4), atomic checkpoints |
| Scripts | `scripts/train_timing.py`, `scripts/train_detector.py` | timing and memory check; overfit sanity check; full run with a checkpoint every epoch, `--resume`, CSV log |
| Config | `configs/stage5.yaml` | all settings |
| Tests | `tests/test_detection.py` (8), `tests/test_check_no_leaks_script.py` (2) | box conversion, flip moves the box with the pixels, empty photos, schedule, agreement, checkpoint round trip; the leak check script always prints a summary and fails on a missing `--extra` file |

Augmentations (all `torchvision.transforms.v2`, boxes follow the image): horizontal flip, colour jitter, a mild zoom (scale 0.85 to 1.15, no rotation); boxes shrunk below 2 px are dropped. The new predictor starts random; the backbone keeps its COCO weights, with frozen batch-norm statistics and the last three ResNet stages trainable (torchvision's default when pretrained weights are used).

## MPS timing and memory (Apple M4 Pro, 24 GB, torch 2.14.1; `scripts/train_timing.py`)

Seconds per training iteration (median of 5 after 2 warm-ups; real train photos with augmentation):

| Short side (px) | Batch | AMP (fp16) | s / iteration | Photos / s | Epoch of 174 photos | 30 epochs | MPS driver memory |
|---|---:|---|---:|---:|---:|---:|---:|
| 800 | 1 | no | 0.38 | 2.64 | 1.1 min | 0.5 h | 4.4 GB |
| 800 | 2 | no | 0.73 | 2.74 | 1.1 min | 0.5 h | 7.4 GB |
| 800 | 4 | no | 1.82 | 2.19 | 1.3 min | 0.7 h | 14.9 GB |
| 800 | 2 | yes | 0.74 | 2.70 | 1.1 min | 0.5 h | 9.7 GB |
| 1024 | 2 | no | 1.02 | 1.96 | 1.5 min | 0.7 h | 13.7 GB |

Photo decoding and augmentation (4 worker processes) keep up: the training loop never waited for data. Readings: batch 2 is as fast per photo as batch 1 and uses 7.4 GB of the 24 GB; batch 4 is slower per photo and uses 15 GB; fp16 gives no speed-up and more memory, so AMP stays off; a larger resize (1024) costs about 40% more time. Grounding DINO also works at 800 px, which is why 800 is kept.

**Measured end to end:** one real epoch over the 174 train photos plus the val agreement evaluation took 94 s of training and about 2 min of wall time in total (the script also loads the model: about 35 s once). **A full run of 30 epochs is estimated at about 1 hour 5 minutes** (about 2.1 min per epoch with the per-epoch val evaluation) on the Mac. **No Colab is needed.** A checkpoint is about 330 MB (model plus optimizer), so 30 epochs need about 10 GB in the ignored `models/stage5/`.

That one-epoch run was a timing run (one-epoch schedule, deleted afterwards); its val agreement (AP@0.5 0.90, recall 0.95) only says the pipeline learns quickly. It is not a result and nothing was tuned on it.

## Overfit-10 sanity check (`python scripts/train_detector.py --overfit`)

10 seeded random train photos (513 pseudo-label boxes), no augmentation, batch 2, constant learning rate 0.005 after a short warmup, 300 iterations (about 5 minutes):

| | Before training | After training |
|---|---|---|
| Mean training loss | 2.99 (epoch 1) | **0.06** (epoch 60) |
| AP@0.5 against these photos' pseudo-labels | 0.039 | **0.990** |
| AP@0.5:0.95 | 0.014 | 0.948 |
| Recall at IoU 0.5 | 0.47 | 0.998 |
| Precision and recall at score 0.3 | 0.08 / 0.47 | 0.998 / 0.998 |

The model reproduces the labels of the 10 photos almost exactly, so the dataset, the box conversion, the loss and the training loop work. The drawings (`data/review/stage5_overfit/`, ignored; green = pseudo-label, red = prediction) show it also reproduces the pseudo-label **fragment boxes** inside bottles: the model copies what it is shown, errors included, which is the expected consequence of training on unverified pseudo-labels.

## Proposed hyperparameters (in `configs/stage5.yaml`, to be confirmed before the full run)

| Setting | Value | Reason |
|---|---|---|
| Batch size | 2 | same speed per photo as 1, 7.4 GB of 24 GB; more stable gradients than 1 |
| Optimizer | SGD, momentum 0.9, weight decay 1e-4 | the plan's recipe |
| Learning rate | 0.005 at batch 2 | inside the plan's 0.005 to 0.01; the common reference (0.02 for 16 photos) scales to 0.0025, so 0.005 is on the bold side and is protected by warmup and clipping |
| Warmup | 1 epoch (87 steps), linear from 0.1 x lr | the new predictor is random; big early steps would damage the COCO features |
| Schedule | cosine decay to 0.01 x lr at the last step | smooth end, no manual drops |
| Gradient clipping | norm 10 | guards against a rare exploding step; the loop also stops if the loss is not finite |
| Epochs | 30 | inside 20 to 40; about 2,600 steps; about 1 h 5 min |
| Size / AMP | short side 800 px, fp16 off | see the timing table |
| Seed | 0; the shuffle and augmentation are reseeded each epoch, so a resumed run continues the same sequence | |
| Checkpoints | every epoch (`models/stage5/frcnn_v1/epoch_XX.pt`), plus `best.pt` by val AP@0.5 | resume with `--resume` |
| Log | `data/logs/stage5/frcnn_v1/train_log.csv`: loss, learning rate, val AP, AP@0.5, recall, precision, recall@0.3, F1 per epoch | plotted in the final report |

## Pending (not part of this stop)

- **Real Label Studio export check: pending**, by decision of the coordinator: the owner-annotation path of `scripts/export_coco.py` will first be tested with a real export when the owner corrects the demo set. Until then it is covered only by tests on synthetic exports.
- Full run, evaluation with the Stage 4 script, the paired bootstrap of the difference by store, the loss and agreement plot, and the error analysis.

## How to verify (inside `soda-poc`)

```bash
python -m pytest -q
python scripts/train_timing.py
python scripts/train_detector.py --overfit
git status --short                          # nothing under data/ or models/
```
