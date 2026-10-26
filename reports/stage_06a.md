# Stage 6a report: crop tooling, val crop sample, catalog photo folder (no embeddings yet)

> Nothing in this report measures identification. 6a only prepares what the owner has to provide: the labels of a val crop sample and the catalog product photos. The val crops are cut from Grounding DINO pseudo-label boxes; the labels the owner adds are the first human ground truth in the project, and they are the yardstick for Stage 6b.

## Decisions applied (owner, via the coordinator)

- **Catalog = product photos, not labeled shelf crops** (about 5 clean photos per product, Open Food Facts first, CC BY-SA; other public product images accepted for this internal POC only; Uruguayan packaging of the variant). No train crops are labeled. The photos are never used for evaluation and the final report lists them as a dependency to replace before any commercial use.
- **Val crop sample** of 150 to 200 crops, stratified by store, labeled by the owner **without suggestions**; labels: the 10 ids, `unknown`, `not_a_bottle`, `cannot_tell`. The `not_a_bottle` share is reported as the pseudo-label precision on val; a targeted batch of unknowns is added if there are too few.
- **Domain gap** (studio photos against shelf crops) is measured in 6b; shelf crops are added to the catalog only as a second step, if needed.
- **Catalog augmentations** (small rotations, mild perspective, scale, brightness and contrast, slight blur) as separate entries tagged `augmented`; kept only if they improve identification on the val sample. No AI-generated images by default.
- **`pg8000` approved and pinned** (dependency table below). **Detector for the pipeline:** a second run `frcnn_v2` (12 epochs, compressed cosine), last epoch, run by the owner; evaluated afterwards with the paired bootstrap.
- `docs/PLAN.md` Stage 6 (items 1 to 10), the Unknown-handling calibration (option (a) decided), Stage 5 (item 6b) and the stack table were updated.

## What was built

| Piece | File | Role |
|---|---|---|
| Crop logic | `src/embeddings/crops.py` | crop id `<photo id>_<k>`; box + 10% margin clamped to the photo; allocation of the sample over stores (proportional, minimum per store, caps); seeded sampling spread over photos |
| Crop labels | `src/embeddings/crop_labels.py` | reads the owner's labels from a Label Studio JSON export (latest submitted annotation; exactly one valid label; unknown crop ids and skipped tasks counted, not used) |
| Catalog files | `src/embeddings/catalog_files.py` | checks the photo folder and `sources.csv` |
| Sample builder | `scripts/build_crop_sample.py` | crops, context images, `mapping.csv`, `tasks.json`; refuses to overwrite an existing sample; `--name`, `--exclude`, `--only-photos` for a second (targeted) batch |
| Label export | `scripts/export_crop_labels.py` | clean CSV, timestamped backup of the export, counts, time per crop, `not_a_bottle` share, warning if fewer than 40 unknowns |
| Catalog check | `scripts/check_catalog_photos.py [--init]` | validates names, sources, duplicates, sizes, counts; `--init` creates the empty folders and a header-only `sources.csv` |
| Label Studio config | `configs/label_studio/crop_project.xml` | crop on the left, the photo with the yellow box on the right, one single-choice question, hotkeys |
| Configs | `configs/stage6.yaml`, `configs/stage5_v2.yaml`, `configs/stage4.yaml` (runs and paired comparisons for v2) | |
| Template | `docs/catalog_sources_template.csv` | columns of `sources.csv` |
| Tests | `tests/test_crops.py` (7), `tests/test_crop_labels.py` (3), `tests/test_catalog_files.py` (5) | 106 tests pass in total |

## The val crop sample (`python scripts/build_crop_sample.py`)

180 crops from 41 photos (the whole val split), stratified by store: s01 23, s03 54, s06 19, s11 27, s15 41, s16 16 (proportional to the pseudo-label boxes of each store, at least 15, at most 8 per photo, seeded). Crops are cut with a 10% margin; crop height median 280 px (only 6 crops are under 100 px tall; small ones are upscaled for display only). The sample includes pseudo-label boxes at their natural rate, so fragments and cans appear as `not_a_bottle`, and low-score boxes appear too (57 of the 180 have a Grounding DINO score under 0.3). Tasks are ordered by photo and left to right, and carry no prediction or suggestion. Files (all ignored): `data/labelstudio/crops/<crop id>.jpg`, `data/labelstudio/crops/context/<crop id>.jpg`, `data/crops/val_sample/mapping.csv` and `tasks.json`. Ids only; the leak check passes on them.

## Verification on a throwaway Label Studio (own container and volume, port 8082, removed afterwards; the owner's instance was not touched)

- The committed `crop_project.xml` is accepted as a project configuration.
- `tasks.json` imports: 180 tasks, 0 predictions.
- With a Local files storage on `/label-studio/files/labelstudio/crops`, both the crop and the context image (the nested `context/` folder) are served (HTTP 200, exact bytes).
- Two annotations submitted through the API in the documented format were exported with Export, JSON; `scripts/export_crop_labels.py` read them correctly (labels, lead time, counts, backup copy). The export of Label Studio 1.23.2 contains the fields the readers expect (`annotations[].result[].value.choices`, `lead_time`, `updated_at`, `was_cancelled`), and **by default it only includes tasks that have an annotation**. The test annotations and the container were deleted afterwards.
- **Not verified:** the keyboard hotkeys (no browser here) and the layout in a browser. The hotkeys use the standard `hotkey` attribute of `Choice`; if one does not work, clicking the option works the same, and Label Studio's keyboard-shortcuts button lists the active keys.
- The Stage 3 export path (`scripts/export_coco.py --ls-export`) is still **pending** on a real export from the owner's project (decision of the coordinator); the throwaway test only shows that the real export format matches what both readers expect.

## Catalog photo folder

`python scripts/check_catalog_photos.py --init` created `data/catalog/product_photos/<product id>/` (ten empty folders, ignored by git) and a header-only `sources.csv`. The check requires: folder names equal to the 10 ids; file names `<product id>_NN.<ext>` (`.jpg`, `.jpeg`, `.png`, `.webp`); every photo listed in `sources.csv` with a source URL, a license (or `unknown`) and `uruguay_packaging_checked` yes or no; readable and non-duplicate images; at least 3 photos per product (target 5); a warning below 300 px on the short side or when the Uruguayan packaging is not confirmed. Today it reports 10 errors (no photos yet), as expected. The exact instructions for the owner are in the handoff.

## New dependency: `pg8000` (approved), pinned with what it pulled in

All from PyPI via the resolver; `pip check` is clean. `pg8000` is the only package we asked for.

| Package | Version | Required by | License | Homepage |
|---|---|---|---|---|
| pg8000 | 1.31.5 | top-level pin | BSD-3-Clause | codeberg.org/tlocke/pg8000 |
| scramp | 1.4.18 | pg8000 | MIT-0 | codeberg.org/tlocke/scramp |
| asn1crypto | 1.5.1 | scramp | MIT | github.com/wbond/asn1crypto |
| python-dateutil | 3.0.0 | pg8000 | BSD-3-Clause and Apache-2.0 (dual) | github.com/dateutil/dateutil |
| six | 1.17.0 | python-dateutil | MIT | github.com/benjaminp/six |
| tzdata | 2026.5 | python-dateutil | Apache-2.0 | github.com/python/tzdata |

It is not used yet; the database work is 6b.

## Second detector run (`frcnn_v2`)

`configs/stage5_v2.yaml` is the Stage 5 config with 12 epochs (warmup 1 epoch, cosine decay compressed to 12 epochs) and run name `frcnn_v2`; nothing else changes. It takes about 26 minutes (12 epochs of about 2.1 minutes with the per-epoch val evaluation) and about 4 GB of checkpoints. The owner runs it with the command in the handoff. Afterwards: the last epoch (12) is evaluated with the paired bootstrap against the `frcnn_v1` last epoch and the COCO baseline (`configs/stage4.yaml` already lists those runs and pairs), and `frcnn_v2` is used for the pipeline unless it is clearly worse (the interval of the AP@0.5 or AP@0.5:0.95 difference against the `frcnn_v1` last epoch entirely below zero).

## Limits

- The sample is 180 crops from 6 stores: results on it will have wide intervals, and 6b numbers are development numbers.
- The crops come from pseudo-label boxes, so bottles the detector missed are not in the sample; the sample measures identification and the pseudo-label precision, not recall.
- Hotkeys untested in a browser.

## How to verify (inside `soda-poc`)

```bash
python -m pytest -q
python scripts/build_crop_sample.py            # refuses to overwrite the existing sample (by design)
python scripts/check_catalog_photos.py         # 10 errors until the owner adds the photos
python scripts/check_no_leaks.py --extra data/crops/val_sample/mapping.csv data/crops/val_sample/tasks.json
git status --short                              # nothing under data/ except splits/*.txt
```
