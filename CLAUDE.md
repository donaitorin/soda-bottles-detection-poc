# soda-bottles-detection-poc

Learning POC: detect soda bottles in store shelf photos and identify which ones are one of the 6 target products, and send every other bottle to `unknown`. The full plan, with its 10 stages and their validation gates, is in `docs/PLAN.md`. Read it before any task.

## How we work

- We work on **one plan stage at a time**. Do not start a new stage until the previous one has passed its gate and I have confirmed it.
- Before writing code for a stage, propose the tasks and wait for my approval.
- When a stage is done, write a summary in `reports/stage_XX.md` (what was done, results, how I can verify the gate, problems found) and wait for my confirmation.
- I am learning computer vision. When you implement something non-obvious (coordinate conversions, detector settings, thresholds), add a short comment in the code and explain it in the summary.
- Every time we stop and wait for approval, write a handoff file at `.handoff/latest.md` (overwritten each time, gitignored, never committed) so the user can pass it to the project coordinator, who has no access to the repo. Contents in order: status; suggested commits (in order, for each one the files to stage and the commit message; every file belongs to exactly one commit, so I never stage part of a file); verification commands with trimmed output, including `git status --short` (and confirm that no image, annotation file or file under `data/` other than `.gitkeep` and splits shows up); diffs against HEAD (the uncommitted working tree) of changed docs/configs/CLAUDE.md/.gitignore/environment.yml/requirements.txt/docker-compose.yml (code files: diff if under ~200 lines, else a summary and path); numbered decisions or questions. Handoffs never carry real identifiers (store ids, store names or addresses, uploader names, photo file names): diffs against HEAD also show deleted lines, so redact them (replace each match by `<redacted-...>`) and run `python scripts/check_no_leaks.py --extra .handoff/latest.md` before telling the owner the handoff is ready; it must report 0 problems. This check is the LAST step before telling the owner the handoff is ready, and its output (the summary line) is appended at the end of the handoff. The script always prints a summary line and exits non-zero on any problem, on a missing `--extra` file or on an internal error; when running it through `conda run`, do not read only the last line of the output (`conda run` adds a trailing blank line). Keep the chat reply short and point to the file.
- **Never run `git add`, `git commit` or `git push`.** I review every diff in VS Code and commit manually. Read-only git commands (status, diff, log, ls-files, check-ignore) are fine. Leave all changes in the working tree, unstaged.

## Technical constraints

- **Do not use YOLO or Ultralytics.** The detector is torchvision's Faster R-CNN (`fasterrcnn_resnet50_fpn_v2`).
- Pre-labeling (Stage 2): Grounding DINO (`IDEA-Research/grounding-dino-base` via `transformers`, pinned revision, weights cached in `models/hf_cache/`, prompt `bottle.`, settings frozen in `configs/prelabel.yaml`). The torchvision Faster R-CNN with COCO weights stays in the repo as the Stage 4 baseline and a fallback pre-labeler. SAM 2 is used only in Stage 6 (masks for crops). Manual validation in Label Studio.
- Permissive licenses (MIT, BSD, Apache 2.0). MPL-2.0 is accepted only for unmodified third-party dependencies. Ask before adding anything else. **Documented exception (owner decision, applies when Stage 7 starts):** EasyOCR for pipeline variant B with its LGPL dependency `python-bidi`, used unmodified and imported normally (reason: the best OCR engine in the trial, runs on MPS; Tesseract is the licence-clean fallback). EasyOCR and its extra dependencies go in a separate pinned `requirements-ocr.txt`, installed only for variant B; `requirements.txt` (the core pipeline) stays permissive only. The pipeline must run fully without the OCR requirements (variant A) and fail with a clear message if `use_ocr` is requested without them. The exception and its reason go into the final report's dependency list.
- Python 3.11 in a conda environment named `soda-poc` (conda-forge only), defined in `environment.yml`. Pinned dependencies in `requirements.txt`, installed through the pip section of `environment.yml` (PyTorch has no official conda packages, so torch and torchvision come from pip). No `.venv`, no pyenv.
- **Never run anything in conda's `base` env.** Every script and test runs inside `soda-poc` (`conda activate soda-poc`, or `conda run -n soda-poc ...`).
- Fixed random seeds for anything with randomness.
- Configuration lives in `configs/*.yaml`, not hardcoded in scripts. Scripts run from the CLI.
- Code, comments, file names, docs and commit messages in English.

## Environment: Mac

- There is no CUDA on this Mac. Use a `get_device()` function (in `src/common/device.py`) that picks `cuda`, then `mps`, then `cpu`, and use it across the whole project.
- Set `PYTORCH_ENABLE_MPS_FALLBACK=1` for operations MPS does not support.
- If something does not run on MPS or is too slow (for example, full training or pre-labeling every photo), tell me and propose running it on Colab instead of looking for long workarounds.

## Data

- **Never commit photos or labeling files.** Everything under `data/` is gitignored except `.gitkeep` files and `data/splits/*.txt` (lists of anonymous photo ids only, never file names). That includes the ~300 shelf photos in `data/raw/`, pre-labels, Label Studio exports, COCO annotations, crops, masks and catalog images. `models/` (checkpoints and downloaded weights) is also ignored.
- **Never run `docker compose down -v`** (or delete `volumes/`): all Label Studio data (annotations, product choices, lead times) lives only in the Docker volume `volumes/label-studio`. Backup routine: a JSON export (Export, JSON) after each labeling session; `scripts/export_coco.py --ls-export FILE` copies it with a UTC timestamp to `data/annotations/ls_backups/`.
- At the end of every task, run `git status --short` and make sure no image or annotation file shows up as untracked or modified. If the `.gitignore` is missing something, fix it and tell me.
- `data/raw/` is never modified. Everything derived goes into other folders.
- The detector has a single generic `bottle` class for soda bottles, regardless of brand. Brand and variant are resolved later with embeddings, never in the detector labels.
- **Cans are out of scope**: there is no `can` class and cans are not labeled.
- **Target products (6 ids, reduced from 10 by owner decision for the tests):** `coca_classic`, `coca_classic_returnable` (the only returnable kept), `coca_zero`, `coca_light`, `sprite`, `fanta` (new; its own brand). Details in `docs/catalog_variants.md`. The removed ids (`coca_zero_returnable`, `sprite_returnable`, `sprite_zero`, `pepsi`, `pepsi_returnable`) are out of scope, not "low data". Size and material (PET or glass) are secondary catalog metadata, not hard success criteria.
- **Correcting labels on the VAL crop sample (owner rule):** only clearly wrong labels are corrected, and the report shows the numbers before and after the correction. Reviewing only the crops the model gets wrong biases the yardstick towards the model.
- **Any bottle that is not one of these 6 products ends up as `unknown`** (other brands, water, juice, Pepsi, Sprite Zero, and the returnable versions of Zero, Sprite and Fanta). The detector still labels all of them as `bottle`; the decision is made after the embeddings (see "Unknown handling" in `docs/PLAN.md`).
- **Labels are Grounding DINO pseudo-labels (owner decision).** Train, val and test use the pre-labels as they are (`label_source: prelabel_unverified`); an owner annotation in a Label Studio JSON export wins (`owner_corrected`). Every metric on val or test measures **agreement with Grounding DINO, not accuracy**, and every report must say so each time such a number is shown. Only the future demo set (owner-corrected) gets accuracy numbers.
- The test split is not used to tune anything before stage 8.
- Photos are split by store or shelf, never by individual photo.
- **File names are never committed.** Photo file names contain uploader names and store names and addresses. From Stage 1 on, each store gets a code (`s01`, `s02`, ...; assigned once, in sorted store id order; stores added later get the next free code; existing codes are never renumbered) and each photo an anonymous id `<store code>_<first 8 hex of sha256 of the file name>`, e.g. `s07_3fa2c9d1`. Real store ids are not committed either. Two mappings live in `data/` (gitignored): store code to store id, and photo id to file name. Only store codes and photo ids go into `data/splits/*.txt` and anything else committed (reports, configs, docs, test fixtures). Do not rewrite git history.
- Coordinates: torchvision uses xyxy boxes in pixels; Label Studio uses x, y, width, height as percentages (0 to 100). Every conversion goes through functions with tests.

## Commands

One-time setup (all inside the `soda-poc` env, never in `base`):

```bash
conda env create -f environment.yml                       # first setup (only the file applies `nodefaults`)
conda env update -n soda-poc -f environment.yml --prune   # after changing environment.yml / requirements.txt
conda activate soda-poc
nbstripout --install                                  # one-time per clone: strips notebook outputs on commit
```

Stage 1 (inside `soda-poc`):

```bash
python scripts/build_ids.py                      # builds store codes and photo ids in data/id_maps/ (gitignored); safe to rerun
python scripts/lookup_photo.py s07_3fa2c9d1      # photo id -> path under data/raw/ (needs the local mapping)
python scripts/build_metadata.py                 # data/metadata.csv (gitignored): size, store code, date, flags
python scripts/find_duplicates.py                # exact + near duplicates; review images and decisions.csv in data/review/duplicates/ (gitignored)
python scripts/make_exclusions.py                # data/splits/excluded.txt (screenshots; add --apply-duplicate-decisions after the owner confirms decisions.csv)
python scripts/make_splits.py                    # data/splits/{train,val,test}.txt, strictly by store, fixed seed
python scripts/check_splits.py                   # Stage 1 gate: no store in two splits, ids only, shares within tolerance
python scripts/check_no_leaks.py [--staged] [--extra .handoff/latest.md]   # real store ids, names, uploader names, file names (from the local mapping) and anything shaped like a raw photo file name; synthetic fixtures carry the marker `leak-check: allow-synthetic`
```

Stage 2 (inside `soda-poc`; photo ids only, outputs in ignored folders):

```bash
python scripts/prelabel_timing.py --model dino    # MPS vs CPU timing on one train photo (also --model frcnn)
python scripts/prelabel_compare.py                # side-by-side pre-labeler comparison on ~10 train photos -> data/review/prelabel_compare/
python scripts/prelabel_run.py                    # frozen Grounding DINO on all 253 non-excluded photos -> data/prelabels/ (gitignored)
python scripts/build_ls_tasks.py                 # relative symlinks named by photo id + data/labelstudio/tasks.json (gitignored)
python scripts/check_ls_conversion.py            # round trip of every converted box; draws 10 photos; lists the 3 photos for the alignment check
```

Stage 3 (inside `soda-poc`; photo ids only, outputs in the ignored `data/annotations/`):

```bash
python scripts/export_coco.py [--ls-export PATH]   # COCO per split from the pre-labels (+ owner annotations from a Label Studio JSON export) -> data/annotations/coco/
python scripts/check_coco.py                       # Stage 3 gate: sanity checks and 20 drawn photos in data/review/coco_check/
```

Stage 4 (inside `soda-poc`; val only, test is not touched; every number is agreement with Grounding DINO, not accuracy):

```bash
python scripts/predict_val.py --model frcnn      # Faster R-CNN COCO on val (raw and filtered) -> data/predictions/stage4/ (gitignored)
python scripts/predict_val.py --model dino       # frozen Grounding DINO on val: pipeline sanity check, not a baseline
python scripts/evaluate_detections.py            # AP, recall, precision with bootstrap CI by store, per store and per density
python scripts/stage4_error_look.py              # draws disagreements with the pseudo-labels in data/review/stage4_errors/
```

Stage 5 (inside `soda-poc`; trains on the Grounding DINO pseudo-labels, so every val number is agreement, not accuracy):

```bash
python scripts/train_timing.py                   # seconds per iteration, memory and epoch time on the real device (before any long run)
python scripts/train_detector.py --overfit       # sanity check: overfit 10 train photos
caffeinate -i conda run --no-capture-output -n soda-poc python scripts/train_detector.py            # full run (checkpoint every epoch in models/stage5/, log in data/logs/stage5/)
caffeinate -i conda run --no-capture-output -n soda-poc python scripts/train_detector.py --resume   # continue after an interruption
python scripts/predict_val.py --model frcnn --checkpoint models/stage5/frcnn_v1/best.pt --name frcnn_ft_best_val   # fine-tuned model on val (also --checkpoint .../epoch_30.pt --name frcnn_ft_last_val)
python scripts/evaluate_detections.py            # now also the paired bootstrap of the difference by store (configs/stage4.yaml, `paired`)
python scripts/stage5_error_analysis.py          # disagreements with the pseudo-labels by cause, drawings in data/review/stage5_errors/
python scripts/plot_training.py                  # loss and val agreement curves -> reports/figures/stage_05_training.svg (numbers only, committed)
```

Stage 6a (inside `soda-poc`; crops and labels are photo-id based and gitignored; the catalog photos are product photos from the internet, never shelf photos):

```bash
python scripts/build_crop_sample.py                          # ~180 val crops for the owner to label (no suggestions) -> data/labelstudio/crops/, data/crops/val_sample/{mapping.csv,tasks.json}
python scripts/export_crop_labels.py --ls-export FILE        # crop labels from a Label Studio JSON export -> data/annotations/crop_labels/ (+ timestamped backup copy)
python scripts/normalize_catalog_photos.py [--apply]       # one-time: rename catalog photos to <product id>_NN.<ext>, remove empty folders of removed products, fill sources.csv (dry run by default)
python scripts/crop_catalog_photos.py                        # crop each catalog photo to its bottle with the frozen Grounding DINO -> data/catalog/crops/, contact sheet in data/review/catalog_crops/
python scripts/check_catalog_photos.py [--init]              # catalog photo folder (data/catalog/product_photos/<product id>/<product id>_NN.<ext> + sources.csv); --init creates the empty folders
caffeinate -i conda run --no-capture-output -n soda-poc python scripts/train_detector.py --config configs/stage5_v2.yaml   # second detector run, 12 epochs
```

Stage 6b (inside `soda-poc`):

```bash
python scripts/embed_timing.py                   # DINOv2 speed and memory on MPS (before anything is built on it)
python scripts/preview_catalog_augment.py        # one contact sheet per product in data/review/catalog_augment/: originals next to their augmented variants (owner reviews before the database)
python scripts/check_catalog_photos.py --final   # the gate for a FINAL report: fails while sources.csv still says unknown or the packaging is unconfirmed
python scripts/build_catalog.py                  # catalog embeddings + own database soda_catalog (originals only, see configs/stage6.yaml catalog.use_augmented)
python scripts/evaluate_identification.py        # development numbers on the labeled val crops, leave one store out
python scripts/diagnose_domain_gap.py            # diagnostic (in memory only): domain gap, degraded catalog
```

Demo tool (inside `soda-poc`; try the current system on one photo; a developer tool, not an evaluation):

```bash
python scripts/demo_identify.py --image data/demo/inputs/photo.jpg [--out PATH] [--scorer classifier|catalog] [--t-unk X --t-auto Y | --t-sim X --t-margin Y --t-vote Z --t-neg W]
# frcnn_v2 epoch_12 at score 0.3 -> 10% margin crops -> DINOv2 CLS -> the ADOPTED classifier (logistic regression, models/stage6; needs `python scripts/fit_classifier.py`) -> automatic / candidates / review / unknown. `--scorer catalog` is the fallback: search in soda_catalog (catalog (b), 193 rows; needs the Postgres container) with the Stage 6b rules.
# Photos of the TRAIN split are optimistic in the demo (the classifier was trained on their crops, and they are in catalog (b)); the tool prints a NOTE. Use val photos or photos that are not in the dataset. Output (ignored): data/demo/outputs/<id or file name>_identified.jpg and .json
# green = automatic, blue = candidates (a short list of products), yellow = review, grey = unknown. The thresholds in configs/stage6.yaml `demo.classifier` (and `demo.thresholds` for the catalog fallback) are DEMO thresholds, not an evaluation result (re-tune with scripts/fit_classifier.py and scripts/tune_demo_thresholds.py).
# Test-split photos are REFUSED (content hash compared with the raw test photos); raw file names are never printed: a dataset photo is named by its anonymous id. Needs the Postgres container and python scripts/build_catalog.py.
```

Stage 6c, second step (inside `soda-poc`; development numbers on the 180 labeled val crops, leave one store out; the TRAIN crops come from train photos only):

```bash
python scripts/check_pooling.py                  # mean pooling sanity check: padding share, cls vs mean vs mean_content
python scripts/build_train_crop_round.py         # 300 TRAIN crops (200 random by store + 100 targeted for rare products) -> data/crops/train_shelf/{mapping.csv,tasks.json,candidate_embeddings.npz}; import tasks.json into the Label Studio project soda-bottles-crops-train, export JSON to data/labelstudio/exports/crop_train_shelf.json
python scripts/export_crop_labels.py --ls-export data/labelstudio/exports/crop_train_shelf.json --name train_shelf   # -> data/annotations/crop_labels/train_shelf_labels.csv (val_sample_labels.csv untouched)
python scripts/make_label_sheets.py              # contact sheets per label of the labeled TRAIN shelf entries -> data/review/train_shelf_labels/ (ignored), to spot mislabels; corrections as crop ids
python scripts/build_catalog.py                  # rebuild the database after any label correction (catalog (b), catalog.use_shelf)
python scripts/check_catalog_db.py               # database search vs offline search on the val crops, catalog (b)
python scripts/evaluate_shelf_catalog.py         # (a) originals vs +shelf, (b) + unknown negatives, (c) shelf only; paired bootstrap by store
python scripts/evaluate_answer_types.py [--catalog originals|originals+shelf|originals+shelf+negatives|shelf]   # candidate list (conformal) and two-level answer vs the current rules
```

Stage 6d, variant experiments (inside `soda-poc`; development numbers on the 180 labeled val crops, catalog (b) as the reference; SAM 2 `facebook/sam2.1-hiera-small`, Apache-2.0, weights in models/hf_cache):

```bash
python scripts/sam2_timing.py                    # SAM 2 speed on MPS (0.2 to 0.5 s per photo)
python scripts/run_stage6d.py [--exp all|colour|context|logreg]   # 4a colour descriptor, 4b 0% margin and SAM 2 masks, 4c logistic regression (scikit-learn); keep-rules in reports/stage_06d.md
```

Stage 6 close-out: the ADOPTED scorer (logistic regression on the CLS embedding, scikit-learn BSD-3; inside `soda-poc`):

```bash
python scripts/fit_classifier.py                 # fit ONCE on all labeled TRAIN shelf entries -> models/stage6/logreg_cls.npz + logreg_cls_card.json (training counts, labels-file hash, C, seed, thresholds, date); prints the demo thresholds to copy into configs/stage6.yaml demo.classifier (optimistic)
python scripts/make_error_sheets.py              # contact sheets of the val crops the classifier gets wrong, grouped true -> predicted -> data/review/stage6d_errors/ (ignored)
python scripts/evaluate_answer_types.py --scorer classifier   # candidate lists from the classifier's probabilities (same 90% coverage target, leave one store out)
python scripts/evaluate_stage6_gate.py           # the Stage 6 gate with the adopted scorer: brand, top-3, unknown handling, named confusions (development numbers)
```

Stage 7, variant B (inside `soda-poc`; Stage 7 done, OCR not adopted but kept runnable; OCR packages only for variant B, weights in models/ocr; TRAIN shelf entries only for any choice, val is not read; variant B must stay runnable whatever the val result, to be measured again on the demo set):

```bash
pip install -r requirements-ocr.txt              # easyocr and friends (python-bidi and the FFmpeg libs of opencv-python-headless are LGPL: owner-approved exceptions); then `pip check`
python scripts/ocr_reading_rate.py [--reread]    # what the OCR reads on the labeled TRAIN shelf entries by crop height, words per product, extra-word proposals (rules in configs/stage7.yaml)
python scripts/ocr_timing.py                     # seconds per photo of variant A and B on MPS (10 val photos, timing only)
python scripts/evaluate_variant_b.py [--reread]   # A against B on the 180 val crops (leave one store out, paired bootstrap by store), keep-rule, fixes / new errors, weak spots, forced-on diagnostic; rules in configs/stage7.yaml; reads the val crops once (cache in data/ocr/stage7/)
python scripts/demo_identify.py --image PATH --use-ocr [--ocr-min-height N] [--w-brand X --w-variant X --w-unknown X --no-contradiction]   # variant B in the demo (classifier scorer only; UNTUNED weights in configs/stage7.yaml demo_b; panel shows words read and rules; stops with an install message without requirements-ocr.txt)
```

OCR trial (decision input only; the ONE exception to "everything runs in soda-poc": it runs in the throwaway conda env `ocr-trial`, never in soda-poc; EasyOCR, python-bidi (LGPL) and pytesseract are accepted for the trial only and are never added to requirements.txt or the pipeline):

```bash
conda run --no-capture-output -n ocr-trial python scripts/ocr_trial.py --engine easyocr --device mps    # or --device cpu, --min-height 900, --select largest
conda run --no-capture-output -n ocr-trial python scripts/ocr_trial.py --engine tesseract               # needs the system package (`brew install tesseract`, done). The trial is finished and the env `ocr-trial` was removed: re-create it to rerun
```

Pre-commit leak check (the owner runs this once per clone; the assistant never installs it):

```bash
bash scripts/hooks/install.sh                    # symlinks scripts/hooks/pre-commit into .git/hooks; it runs check_no_leaks.py --staged in the soda-poc env
```

More commands are added as they are created.
