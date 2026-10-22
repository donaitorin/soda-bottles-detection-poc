# Stage 3 report: labels, export and sanity checks (reduced scope)

> **Read this first.** By owner decision, train, val and test use the Grounding DINO pre-labels **as they are**: nobody corrected them. Any number computed against these labels measures **agreement with Grounding DINO, not accuracy**. This report shows no accuracy figure; the only numbers are counts and the results of structural checks.

## Decisions (owner, via the coordinator)

- The goal of the POC is to define, understand and test the end-to-end process, not to measure accuracy on hard field photos. The owner reviewed several photos and found the Grounding DINO boxes good enough for that.
- Train, val and test: pseudo-labels from `data/prelabels/predictions.json` (`label_source: prelabel_unverified`). An owner annotation in a Label Studio export wins (`owner_corrected`). Nothing is submitted in Label Studio.
- The real yardstick will be a future **demo set** (well-taken photos, pre-labeled with the same pipeline, corrected by the owner). Only demo numbers are accuracy.
- The manual review step stays in the documented process ("required in a real project, skipped in this POC by owner decision", with the effort: about 15,000 boxes, about 60 per photo). The Label Studio project, config and export path keep working for the demo set and the Stage 6 crop labels.
- The split rules (by store, test used once in Stage 8) still apply.

## What was built

| Piece | File | What it does |
|---|---|---|
| Config | `configs/stage3.yaml` | paths, tag names, sanity thresholds; the height rule is read from `configs/prelabel.yaml` (one source) |
| Export logic | `src/annotations/export.py` | pixel xyxy to COCO `bbox` (x, y, width, height), clamping, the 1.5% floor, merge of owner annotations over pseudo-labels, reading the Label Studio JSON export |
| Sanity logic | `src/annotations/sanity.py` | the checks below, returning errors and warnings |
| Scripts | `scripts/export_coco.py`, `scripts/check_coco.py` | export to `data/annotations/coco/` and the gate check with 20 drawn photos in `data/review/coco_check/` (both ignored) |
| Tests | `tests/test_annotations.py` (12 tests, synthetic ids) | COCO bbox format, clamping, floor applied and counted, owner wins (including an annotation with zero boxes), skipped annotations ignored and latest used, size mismatch rejected, each sanity check catches its problem |

A third box format appears in this stage: COCO stores `bbox` as `[x, y, width, height]` in **pixels** (top-left corner plus size). The project now has three: torchvision xyxy pixels, Label Studio percent, COCO xywh pixels. Each conversion is a tested function.

**Why the Label Studio JSON export and not the built-in COCO export:** the COCO export drops the per-photo `Choices` (which products appear in a photo) and the `lead_time` of each annotation. The JSON export keeps both. The script reads the JSON, takes the latest submitted (not skipped) annotation of each task, and replaces the pre-label of that photo as a whole. A submitted annotation with no boxes means "no bottles in this photo". The script also copies the export, untouched, to `data/annotations/ls_backups/` with a UTC timestamp: that is the backup routine (Label Studio data lives only in `volumes/label-studio`; never run `docker compose down -v`).

## Result of the export (current photos, no owner annotations)

| Split | Photos | Boxes | Photos `prelabel_unverified` | Photos `owner_corrected` | Dropped by the 1.5% floor at export |
|---|---:|---:|---:|---:|---:|
| train | 174 | 10,566 | 174 | 0 | 0 |
| val | 41 | 2,157 | 41 | 0 | 0 |
| test | 38 | 2,405 | 38 | 0 | 0 |

The floor drops nothing at export because the pre-labels already apply it (42 boxes were dropped in Stage 2, before this point); it will drop boxes the owner draws below the floor in a future Label Studio export. The files carry the `model_version` (`grounding_dino@12bdfa31+cfg2381fb54`) and a disclaimer in `info`, and each image carries its `label_source`.

## Sanity checks (`python scripts/check_coco.py`)

All three splits: **0 errors, 0 warnings**. Checks: the photos in each file are exactly those of `data/splits/<split>.txt`; no photo in two splits; file names are photo ids; boxes inside the image; area above zero; no duplicate boxes (IoU above 0.9); every box at least 1.5% of the image height; boxes per photo between 1 and 150. Boxes per photo (min / median / max): train 1 / 62 / 124, val 1 / 55 / 101, test 21 / 61 / 120. Two photos have a single box (`s07_cb6e3dce` in train, `s06_b8c050ce` in val). I looked at both: they are **can-only photos** (6-can multipacks). In one, the single box is on a can pack; in the other, on a tiny spot near the top of the shelf. So both single boxes are pseudo-label false positives, and these two photos have no real bottle label at all; the guide says cans are never labeled. The leak check on the four files in `data/annotations/coco/` reports 0 problems (photo ids only).

## The 20 drawn photos (`data/review/coco_check/`, seeded random, red = unverified)

The boxes are placed correctly (the check also asserts that each COCO image size equals the photo size). What the drawings show are the known **pseudo-label errors**, which Stage 5 will imitate: in dense stacks, small fragment boxes inside or between bottles (clear in a 3-row display with bottles piled in front); boxes on bottles whose body is mostly hidden inside crates; and boxes on partly visible bottles at the photo edge. These are exactly the cases the labeling guide (40% rule) would settle in a manual review. Look at the folder to judge them yourself.

## Limits

- Nothing was tested with a **real** Label Studio JSON export, because no annotation has been submitted. The owner-annotation path is covered by tests that build exports in the documented format (task with `annotations`, `result` entries of type `rectanglelabels` and `choices`, `lead_time`, `updated_at`, `was_cancelled`). A real export is the first check when the demo set is labeled; the steps are in the handoff.
- The pseudo-labels include the Grounding DINO errors listed above. They are in the train, val and test files and will count as "correct" in every agreement metric.
- Per-product photo counts and correction times come from owner annotations, so they are empty now.

## How to verify the gate (inside `soda-poc`)

```bash
python -m pytest -q
python scripts/export_coco.py
python scripts/check_coco.py            # expect: 0 errors in train, val and test; 20 drawings
python scripts/check_no_leaks.py --extra data/annotations/coco/train.json data/annotations/coco/val.json data/annotations/coco/test.json data/annotations/coco/summary.json
git status --short                       # nothing under data/ except splits/*.txt
```
