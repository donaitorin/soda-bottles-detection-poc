# Stage 2 report: pre-labeling with Grounding DINO

**Status: waiting for the owner's alignment check in Label Studio and the minimum-height decision.** Nothing in `data/raw/` was modified. No val or test result was used for any decision.

## Decisions

- **Pre-labeler: Grounding DINO alone** (`IDEA-Research/grounding-dino-base`, revision `12bdfa3120f3e7ec7b434d90674b3396eccf88eb`, Apache 2.0, Swin-B backbone, `model.safetensors` 933,400,872 bytes). Chosen by the owner after the side-by-side comparison on 10 train photos. Prompt `bottle.`, box threshold 0.2, text threshold 0.25, all frozen in `configs/prelabel.yaml`. The hash of the frozen settings is stored as the `model_version` of every prediction: `grounding_dino@12bdfa31+cfg40dd2243`.
- **Faster R-CNN (COCO)** stays in the repo as the Stage 4 baseline and a possible fallback if correcting dense shelves turns out slow. The merge code is kept, unused. SAM 2 moves to Stage 6.
- **Containment filter** stays at 0.9 (see the numbers below).

## Comparison on 10 train photos (the basis of the choice)

The photos: the 5 hard cases from the labeling guide, the 2 highest-resolution train photos, one photo from each remaining train store, all density levels. Boxes after filters:

| | Faster R-CNN | Grounding DINO | Shared (IoU 0.5) | Merged |
|---|---:|---:|---:|---:|
| Total over 10 photos | 574 | 494 | 389 | 679 |

Faster R-CNN found more individual bottles in dense rows but produced overlapping boxes inside shrink-wrapped packs and false positives on can packs and bags; Grounding DINO boxes were tighter and fewer. Both box cans and can packs, which the owner deletes in Label Studio.

Speed on one dense train photo (1184x1440, 2 warm-ups, median of 5): Grounding DINO 0.43 s on MPS against 1.58 s on CPU (the same 76 boxes on both devices, IoU >= 0.9); Faster R-CNN 0.15 s against 0.95 s. The full run over 253 photos took 146 s on MPS, so Colab was not needed.

**Threshold finding:** the usual box threshold 0.3 returned no boxes on a dense shelf (maximum score 0.297 on both devices); 0.2 was used. Over the 10 photos, Grounding DINO box counts before filters for `bottle.` were 1556 / 847 / 531 / 186 at thresholds 0.1 / 0.15 / 0.2 / 0.3. `soda bottle.` and `plastic bottle.` gave similar totals at 0.2 (574 and 543) and were not used.

## Filters

Applied in this order, every drop counted: oversized boxes (area above 20% or width above 60% of the image), grey-wedge boxes (pixels equal to RGB 128,128,128 within +-2, connected to the border and not tiny; boxes more than 50% inside are dropped, the others clipped), tiny boxes (height under 1.5% of the image height; the comparison below was run with the earlier 20 px floor), duplicates (NMS at IoU 0.7, then boxes at least 90% inside a larger, higher-score box).

| | Raw boxes | Oversized | Wedge dropped | Wedge clipped | Tiny | NMS | Contained |
|---|---:|---:|---:|---:|---:|---:|---:|
| 10 comparison photos | 531 | 9 | 0 | 38 | 2 | 4 | 22 |
| All 253 photos (final rule: 1.5% of image height) | 16,135 | 312 | 0 | 1,313 | 42 | 177 | 476 |

- **Containment:** it removed 22 of 531 boxes (**4.1%**) on the 10 comparison photos, under the 5% limit agreed with the coordinator, and 476 of 16,135 (**3.0%**) in the full run. For Faster R-CNN on the 10 photos it removed 85 boxes; looking at two photos, most were fragments inside a bottle, with a few possibly partly hidden back-row bottles.
- **Wedge:** clipping works (no box mostly inside a wedge, 1,313 clipped at the edge).
- 26 boxes extended up to 2.6 px outside the photo; the conversion clamps them to the image.

## Pre-labels (all 253 non-excluded photos, frozen settings)

15,128 boxes kept: train 10,566, val 2,157, test 2,405 (final rule; the first run with the 20 px floor kept 15,136). `model_version` is `grounding_dino@12bdfa31+cfg2381fb54` (median 61 per photo, maximum 124, no photo without boxes). Outputs are in the ignored `data/prelabels/` (`predictions.json`, `run_summary.csv`, `box_heights.csv`), keyed by photo id.

## Final minimum-height rule: DECIDED, 1.5% of the image height

Computed from the **train photos only** (10,600 boxes before the tiny-box floor; there are no train photos below 1200 px or above 3200 px long side):

| Long side | Photos | Boxes | Height px: min / 1st pct / 5th pct / median | Height as % of image height: min / 1st / 5th / median |
|---|---:|---:|---|---|
| 1200-1600 | 122 | 7,210 | 11.5 / 33.0 / 55.8 / 205.7 | 0.8 / 2.3 / 3.9 / 14.3 |
| 1600-3200 | 52 | 3,390 | 31.1 / 48.7 / 129.2 / 254.3 | 1.6 / 2.6 / 5.2 / 11.6 |

Boxes each candidate would drop (train): absolute < 20 px: 26 (0.25%); < 30 px: 55 (0.52%); < 40 px: 134 (1.26%); relative < 1.0% of image height: 2 (0.02%); < 1.5%: 34 (0.32%); < 2.0%: 58 (0.55%); < 3.0%: 219 (2.07%).

**Decision (owner): relative rule, keep boxes at least 1.5% of the image height** (about 22 px on a 1440 px tall photo, 45 px on a 3000 px one, 61 px on a 4096 px one). Reason: what a human sees on screen is the size relative to the photo, and 20 px is 1.4% of a typical photo but only 0.5% of the largest; the number of boxes affected is tiny either way (0.25% against 0.32% of train boxes), so this is about consistency of the labeling guide, not about the data. Done: config key `filters.min_height_fraction: 0.015` (replaces `min_height_px`), pre-labels rerun, `tasks.json` regenerated, labeling guide case 8 updated. The same function (`below_height_floor`) is used at the Stage 3 export (dropped boxes are counted per split) and in the Stage 4 and 5 evaluation (predictions under the floor are ignored, neither matched nor false positives). PLAN.md Stages 3, 4 and 5 say so.

## Label Studio (verified on a throwaway instance, not the owner's)

A throwaway Label Studio 1.23 container (own volume, port 8081, removed afterwards) was used to check the pieces:
- The committed config `configs/label_studio/bottle_project.xml` is accepted as a project label config.
- `data/labelstudio/tasks.json` imports: 253 tasks, 253 predictions.
- Photo serving: `/data/local-files/?d=labelstudio/images/<photo id>.jpg` returns **404 until a "Local files" source storage is connected to the project**, and **200 `image/jpeg` with the exact bytes of the original (same size) after connecting it**, so the relative symlinks resolve inside the container mount. Do not press "Sync Storage": it would create one extra task per image.
- On import the project took our `model_version` and had "use predictions to prelabel tasks" enabled by default.
- Not verified here: how the page looks in the browser (I checked through the API); that is what the owner's 3-photo alignment check is for.
- The converted boxes were checked numerically (all 15,128 boxes round-trip to the stored pixels with a difference of 0.0000 px) and visually on 10 photos in the ignored `data/review/ls_check/`. The tests in `tests/test_labelstudio.py` cover a known box on a portrait photo, a landscape photo (width/height not mixed), round trips on four resolutions, clamping at the edges, and task structure with ids only.

## Notes and limitations

- The pre-label quality measurement happens at the end of Stage 3, against the corrected labels, and is biased toward the pre-labeler (annotators tend to accept the boxes they are shown); it is a lower bound on the real error. Stage 4's zero-shot Grounding DINO baseline on val has the same bias, since the val labels start from its own boxes.
- The leak check's file-name-shape pattern only knows the current naming pattern of the photos.
- New dependencies for Grounding DINO (`transformers` 5.20.2 and what it pulls in, all pinned): every license is MIT, BSD, Apache-2.0 or similarly permissive, except `tqdm` (MPL-2.0 AND MIT), accepted for unmodified use. `certifi` and `requests` are not installed.

## How to verify the gate (inside `soda-poc`, and in Label Studio)

```bash
python -m pytest -q                       # all tests
python scripts/check_ls_conversion.py     # round trip of all boxes, and the 3 photos for the alignment check
python scripts/check_no_leaks.py          # 0 problems
git status --short                        # nothing under data/ except splits/*.txt
```
Then follow the Label Studio steps in the handoff: create the project from `configs/label_studio/bottle_project.xml`, connect Local files storage, import `data/labelstudio/tasks.json`, and check the alignment of the 3 photos in `data/labelstudio/alignment_check_ids.txt`.

## Owner results (after the first import)

The owner created the project from the committed config, connected Local files storage (Tasks mode, never synced), imported the first `tasks.json` and checked the alignment of the three photos (portrait, landscape, largest): the boxes line up on all three, photos are served correctly and the pre-labels show by default. The scenes are hard (for example bottles in crates with little visible), which is fine for the POC; the labeling guide's 40% rule covers them. After the minimum-height decision the pre-labels were rerun (`model_version` `grounding_dino@12bdfa31+cfg2381fb54`; the tiny filter drops 42 boxes: train 39, val 1, test 2, counted before duplicate removal, so slightly more than the 34 train boxes of the table above). The owner deletes all tasks and imports the new `tasks.json` before correcting anything.
