# POC Plan: Soda Bottle Detection on Store Shelves

## Goal and scope

The POC validates whether, from store shelf photos, a system can detect soda bottles, identify which ones are one of the 6 target products (Coca-Cola Classic, Classic returnable, Zero and Light; Sprite; Fanta) and send every other bottle to `unknown`, with a low rate of confident wrong identifications. It follows the same architecture as the bathroom asset POC: a generic detector, identification through embeddings, text evidence, and a confidence engine.

**Central question:** can the system count and identify the 6 target products in a shelf photo, sending every other bottle to `unknown`, accurately enough to reduce manual counting?

**In scope**

- Generic detection of soda bottles in the photo.
- Identification of the 6 target products (Coca-Cola Classic, Classic returnable, Zero and Light; Sprite; Fanta), with every other bottle (Pepsi, Sprite Zero, other returnables, other brands, water, juice) sent to `unknown`, plus size when visible. The list was reduced from 10 ids to 6 for the tests by owner decision.
- OCR of the packaging as additional evidence.
- Confidence with three outcomes: automatic, human review, unknown.
- Full pipeline from photo to JSON inventory, with metrics.

**Out of scope:** cans, REX data, YOLO models, other product categories, planogram compliance, mobile app, and production infrastructure.

**Recommended design decision:** the detector learns a single generic class, `bottle` (cans are out of scope), and brand and variant are resolved with embeddings. That way the detector does not need retraining every time a brand or SKU is added, just like in the original document.

## Architecture and stack

The system has two paths that share data: an offline path to build the dataset and train, and an online path that processes each new photo. The whole stack is Python with permissive licenses, except SAM 3, which is left as an option to evaluate.

```
OFFLINE: dataset, detector and catalog          ONLINE: each new photo

1. Shelf photos (split by store)                 New photo (never seen by the model)
        |                                                |
2. Pre-labeling (Grounding DINO)                          v
        |                                         Bottle detection (trained Faster R-CNN)  <-- weights from step 5
3. Manual validation (Label Studio, COCO export)         |
        |                     |                          v
5. Detector fine tuning       6. Reference catalog  Crop of each bottle (box + margin or SAM mask)
   (Faster R-CNN)                (train crops +          |
                                  product photos)        v
                                      |             Identification (DINOv2 + top-K in pgvector)  <-- index from step 6
                              6. Catalog embeddings      |
                                 (DINOv2 -> pgvector)    v
                                                    Text evidence (PaddleOCR on the packaging)
                                                         |
                                                         v
                                                    Confidence (automatic / review / unknown)
                                                         |
                                                         v
                                                    JSON inventory (count per product + annotated photo)
```

The offline path produces two things the online path uses: the detector weights and the catalog index in pgvector. Numbers are the plan stages.

| Component | Tool | Role in the POC | License |
| --- | --- | --- | --- |
| Baseline / fallback pre-labeler | torchvision Faster R-CNN with COCO weights (`bottle` class) | Stage 4 baseline; fallback pre-labeler | BSD-3 |
| Pre-labeling (chosen) | Grounding DINO ([grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base)) via `transformers` | Proposes boxes from a text prompt such as "bottle." | Apache 2.0 |
| Masks for crops (stage 6) | SAM 2 | Box prompts give masks for clean crops; not used in stage 2 | Apache 2.0 |
| Pre-labeling alternative | [SAM 3](https://github.com/facebookresearch/sam3) | Detects and segments from text in a single model | Meta's own SAM License; review before commercial use |
| Manual validation | Label Studio Community | Correcting pre-labels and exporting to COCO | Apache 2.0 |
| Detector | Faster R-CNN ResNet50 FPN v2 (`torchvision`) with COCO weights | Detects `bottle` on the shelf | BSD-3 |
| Embeddings | DINOv2 | One vector per crop to compare against the catalog | Apache 2.0 |
| Vector search | PostgreSQL + pgvector | Top-K by cosine similarity | PostgreSQL License |
| Postgres driver | `pg8000` (pure Python; `psycopg` is LGPL, so it is not used) | Catalog and search queries | BSD-3-Clause |
| OCR | PaddleOCR | Reads "Zero", "sin azúcar", "1,5 L" on the packaging | Apache 2.0 |
| Metrics | `pycocotools` | mAP, recall, precision | BSD-2-Clause |
| Error analysis | FiftyOne (optional) | Inspect false positives and negatives on the photos | Apache 2.0 |
| API | FastAPI | Photo to inventory endpoint | MIT |

Licenses are indicative: before any commercial use, someone should confirm them, including the licenses of the pretrained weights.

## Assumptions and requirements

The plan assumes an own set of shelf photos, a GPU or Apple Silicon accelerator for pre-labeling and training, and a reproducible Python environment. If any of these is missing, stage 0 catches it before time is invested.

- **Photos:** about 300 supermarket shelf photos with beverages are already available. Each photo contains many bottles, so that should yield thousands of instances, which is enough for the POC.
- **Version control:** photos and labeling files never go into git. That covers `data/raw/`, the pre-labels, Label Studio exports, the COCO annotations used to train the detector, crops, catalog images, masks, checkpoints and downloaded models. Only code, configs, docs and small reports are committed.
- **Diversity:** different stores, lighting, angles and distances. Ten photos of the same shelf add less than ten different shelves.
- **Accelerator:** Grounding DINO, SAM and training benefit from a GPU. On a Mac, PyTorch uses MPS, which is enough for development and small runs; full training or pre-labeling every photo may be faster on Colab or a cloud VM with a 12 to 16 GB GPU.
- **Environment:** Python 3.11 in a conda environment (`soda-poc`, conda-forge only, defined in `environment.yml`; torch and torchvision installed with pip because PyTorch has no official conda packages), compatible PyTorch and torchvision versions, Docker for Postgres with pgvector and for Label Studio. Nothing runs in conda's `base` env.
- **Label provenance (owner decision, Stage 3):** the goal of this POC is to define, understand and test the end-to-end process, not to measure real accuracy on hard field photos. Train, val and test use the Grounding DINO pre-labels as they are (pseudo-labels, `label_source: prelabel_unverified`); nobody corrects them. **Every metric computed on val or test measures agreement with Grounding DINO, not accuracy, and every report must say so each time such a number is shown.** The split rules (by store, test used once in stage 8) still apply, so the process stays correct when real labels exist.
- **Demo set (decided by the owner, plan only for now; it is the real yardstick):** the current photos are hard field photos and stay as train, val and test. Later the owner adds a fourth set, **demo**, of well-taken photos without extreme cases, to demonstrate the system. It is pre-labeled with the same pipeline and then corrected by the owner in Label Studio (every box and the product of each crop). It is never used for training, tuning or calibration, and is evaluated once at the end. **Only demo-set numbers are reported as accuracy.** Ideally it comes from stores not in the current photos; if it shares stores with train, the report says its result is optimistic. Demo photos used for the stage 6 reference catalog are not used to evaluate. The same anonymous id, mapping and leak-check rules apply to the new photos.
- **Catalog (decided by the owner):** closed list of the 6 target products, in `docs/catalog_variants.md`, decided in stage 0. The reference images come from about 5 clean product photos per product that the owner provides (Open Food Facts first, CC BY-SA; otherwise other public product images, accepted for this internal POC only), not from labeled shelf crops. They must match the Uruguayan packaging of the variant and are a dependency to replace before any commercial use.

**Suggested repo structure**

```
soda-bottles-detection-poc/
├── data/                 # gitignored, except .gitkeep files and splits/
│   ├── raw/              # the ~300 original photos, never modified (gitignored)
│   ├── splits/           # train.txt, val.txt, test.txt (anonymous photo ids only, committed)
│   ├── prelabels/        # pre-labeler output (gitignored)
│   ├── annotations/      # validated COCO from Label Studio (gitignored)
│   └── catalog/          # reference images per product (gitignored)
├── models/               # checkpoints and downloaded weights (gitignored)
├── src/
│   ├── prelabel/         # stage 2
│   ├── detection/        # dataset, training, evaluation (stages 4 and 5)
│   ├── embeddings/       # catalog and search (stage 6)
│   ├── ocr/              # stage 7
│   ├── confidence/       # stage 8
│   └── api/              # stage 9
├── notebooks/            # exploration and error analysis
├── reports/              # metrics per stage
├── environment.yml       # conda env: python 3.11 + pip install -r requirements.txt
├── requirements.txt      # pinned pip dependencies
└── docker-compose.yml    # postgres+pgvector, label-studio
```

## Stages

There are ten stages in order, and each one ends with a validation gate. Do not move on to the next one until the gate is met: a dataset error that slips through stage 3 contaminates everything that comes after.

### Stage 0: definition and environment

**Goal:** settle what is detected, what is identified, and with which tools.

1. Define the detector class: a single generic `bottle` class for soda bottles, labeled the same way regardless of brand, flavor or size. Brand and variant are not labeled here: the embeddings resolve them in stage 6. Cans are out of scope: there is no `can` class and they are not labeled.
2. Write a one-page labeling guide. It settles the ambiguous cases once:
    - Are bottles in back rows that are only half visible labeled?
    - The box fits the visible part, never the estimated part behind another bottle.
    - Do water or juice bottles count as `bottle`? Recommended: yes, because telling them apart from soda is also the embeddings' job. If the answer is no, the guide says so explicitly, since a class that depends on what is inside the bottle is hard to label consistently.
3. Define the closed list of catalog products. It started with 10 ids and was **reduced to 6 for the tests by owner decision**; it is listed in `docs/catalog_variants.md`: `coca_classic`, `coca_classic_returnable` (the only returnable kept), `coca_zero`, `coca_light`, `sprite` and `fanta` (new; it used to be `unknown`). Glass returnable Coca-Cola Classic bottles in crates count as `coca_classic_returnable`. Size and material (PET or glass) are secondary metadata stored in the catalog, not hard success criteria. Any other bottle (Pepsi, Sprite Zero, the returnable versions of Zero, Sprite and Fanta, other brands, water, juice) is `unknown`; there is no `other brand` class. The removed ids are out of scope, not "low data". This list is used from stage 6 on, not for the detector.
4. Create the `.gitignore` before any photo enters the repo. It ignores everything under `data/` (photos, pre-labels, Label Studio exports, COCO annotations, crops, masks, catalog images) except `.gitkeep` files and `data/splits/*.txt`, plus `models/`, Label Studio and Postgres volumes, and notebook outputs.
5. Bring up `docker-compose` with Postgres + pgvector and Label Studio. Their data volumes live in ignored folders.
6. Create the conda environment `soda-poc` (conda-forge only, Python 3.11) from `environment.yml`.
7. Verify that PyTorch sees the accelerator (`torch.cuda.is_available()` on NVIDIA, `torch.backends.mps.is_available()` on Mac).

**Gate:** the guide resolves at least 5 real ambiguous examples taken from your photos, the verification script runs on a real photo, and the device decision for local work (MPS or CPU) is recorded in `reports/stage_00.md`, and after copying one test image into `data/raw/`, `git status` does not show it.

### Stage 1: data inventory and dataset split

**Goal:** have the data organized and a train / val / test split that does not cheat.

1. Copy the ~300 photos into `data/raw/` and never modify them. Check with `git status` that none of them shows up before making any commit.
2. Fix EXIF orientation (`ImageOps.exif_transpose` from Pillow). Rotated phone photos are the most common cause of shifted boxes.
3. Remove duplicates and near duplicates with perceptual hashing (`imagehash`).
4. Give each store a code (`s01`, `s02`, ...; assigned once, in sorted store id order; stores added later get the next free code; existing codes are never renumbered) and each photo an anonymous id `<store code>_<first 8 hex of sha256 of the file name>`, for example `s07_3fa2c9d1`. Original file names contain uploader names and store names and addresses, and store ids are not committed either, so **file names and real store ids are never committed**: two mappings live in `data/` (gitignored), store code to store id and photo id to file name. Only store codes and photo ids go into `data/splits/*.txt` and anything else that is committed (reports, configs, docs, test fixtures).
5. Record resolution, store and date of each photo in a CSV inside `data/` (gitignored, keyed by id).
6. Exclude the 4 screenshots (`Captura de pantalla ...`) through an exclusion list; they are never deleted or moved in `data/raw/`. Flag photos whose long side is under 800 px so the owner can decide on them.
7. Split 70 / 15 / 15 **strictly by store**, not by photo and not by (store, visit date): the same shelf on two dates looks almost identical and would leak between splits. The 48-photo store goes to train; the other stores are assigned to reach 70 / 15 / 15 by photo count within about ±5 points, with a fixed seed. Among the valid assignments prefer, in this order: (1) the largest value of the smaller of the val and test store counts (test is the final measure, so it must not get few stores), (2) the most stores in val and test, (3) the most visits there, (4) the smallest deviation. The seed and the store assignment (store codes) are recorded in the report.
8. Freeze test: it is not looked at or used to tune anything until stage 8.

**Gate:** no store appears in two splits, `data/splits/*.txt` contain only anonymous ids, and there is a report with the number of photos, resolutions and stores per split.

### Stage 2: pre-labeling with Grounding DINO

**Goal:** generate candidate boxes for all photos and load the predictions into Label Studio. There is no hand-labeled ground truth before pre-labeling: the owner goes straight to correcting pre-labels in stage 3.

1. **Candidates compared:** the torchvision Faster R-CNN with COCO weights (`bottle` class, raised detection and RPN limits, score threshold 0.3) and Grounding DINO, on 10 train photos (the hard cases from the labeling guide, the two highest-resolution photos, one photo per remaining train store, all density levels), side by side in an ignored folder. **The owner chose Grounding DINO alone.** Faster R-CNN stays in the repo as the baseline for stage 4 and as a fallback if correcting dense shelves turns out slow; the merge code is kept but unused.
2. **Grounding DINO:** `IDEA-Research/grounding-dino-base` (Apache 2.0) via `transformers`, pinned revision, weights cached in `models/hf_cache/`. Prompt `bottle.` (lowercase, ending with a dot, per its [model card](https://huggingface.co/IDEA-Research/grounding-dino-base)), box threshold 0.2 and text threshold 0.25. The box threshold is 0.2 and not the usual 0.3 because 0.3 returned no boxes on a dense shelf. All settings are frozen in `configs/prelabel.yaml`; the hash of that config is stored as the `model_version` of every prediction. It runs on MPS (about 0.4 s per photo), so no Colab is needed.
3. **Filtering** (every drop counted by reason): boxes covering a whole shelf, duplicates (NMS plus containment at 0.9), boxes shorter than the minimum height (item 4) and boxes inside the grey side wedges (pixels equal to RGB 128,128,128 connected to the border; boxes mostly inside are dropped, the others are clipped).
4. **Minimum box height (decided by the owner):** a box must be at least **1.5% of the image height** (`min_height_fraction: 0.015` in `configs/prelabel.yaml`, one function shared by the pre-labels, the stage 3 export and the stage 4/5 evaluation). The rule is relative, so photos of any resolution are treated alike, and it is applied automatically so nobody measures by eye. It was chosen from the Grounding DINO height distribution on train photos (see the stage 2 report). If bottles are small in large photos, tiling (SAHI) is an option that needs approval first.
5. **Label Studio:** the `RectangleLabels` `bottle` config plus the per-photo product `Choices` field (stage 3, item 1), in `configs/label_studio/bottle_project.xml`. Photos are served through relative symlinks named by photo id (no file names in Label Studio, tasks or committed files). Pre-labels are converted to Label Studio `predictions` with tested functions: Label Studio uses coordinates as **percentages** of image width and height (0 to 100), not pixels. Tasks for all non-excluded photos (train, val and test) are imported through the UI.
6. **Pre-label all non-excluded photos** (253) with the frozen settings. No decision uses val or test results.

**Gate:** the converted boxes round-trip in the tests and line up on 3 photos (one portrait, one landscape, one of the largest) in the Label Studio UI; tasks with predictions for all photos import correctly; the final minimum-height rule is decided; no file name appears in any task, config, report or committed file. Pre-label quality (precision and recall at IoU 0.5 against the corrected labels) is measured **on the demo set, once the owner's corrections exist** (train, val and test are not corrected in this POC), and is **biased toward the pre-labeler**: annotators tend to accept the boxes they are shown, so it is a lower bound on the real error, not a clean estimate.

### Stage 3: labels, review and export

**Goal:** a COCO dataset with the provenance of every label, built by a script that also works for the demo set later.

**Manual review: required in a real project, skipped in this POC by owner decision.** In a real project the owner would correct every val and test photo (every box verified, because they are the yardstick) and ideally train too, plus a second review of 10% of the photos and the per-photo product field. The effort is known from the pre-labels: about 15,000 boxes in 253 photos, about 60 boxes per photo (median 61, up to 124); the minutes per photo were never measured here (the planned timing pilot was not run). The Label Studio project (`configs/label_studio/bottle_project.xml`), the import file and the export path stay working: they are used for the demo set and for the Stage 6 crop labels.

**In this POC:**

1. **Pseudo-labels:** train, val and test take the Grounding DINO boxes from `data/prelabels/predictions.json` as they are. Each photo is marked `label_source: prelabel_unverified`. If an owner annotation exists for a photo in a Label Studio export, it wins (`label_source: owner_corrected`).
2. **Export script with tests** (`scripts/export_coco.py`, `src/annotations/`): COCO files per split in the ignored `data/annotations/coco/`, merging owner annotations (if any) with the pseudo-labels. It reads the **Label Studio JSON export** (Export, JSON), not the built-in COCO export, which drops the per-photo product choices and `lead_time`; the same script works for the demo set. The export drops every box shorter than 1.5% of the image height (the stage 2 rule) and counts how many, per split. Files hold photo ids only (checked with the leak check).
3. **Sanity checks and drawings** (`scripts/check_coco.py`): boxes inside the image, area above zero, no duplicates (IoU above 0.9), boxes per photo in a plausible range, the 1.5% floor, photos equal to the split lists, file names are photo ids; 20 random photos drawn and looked at.
4. **Backups:** Label Studio data lives only in the Docker volume `volumes/label-studio`. Never run `docker compose down -v`. The backup routine is a JSON export after each labeling session; the export script copies it, with a UTC timestamp, to `data/annotations/ls_backups/`.
5. When owner annotations exist (demo set, Stage 6 crops): report photos per product from the per-photo product field, and the correction time from `lead_time`; measure the pre-label quality (precision and recall at IoU 0.5 of the original predictions against the corrected labels), which is biased toward the pre-labeler.

**Gate:** the sanity script reports no errors, the 20 drawn photos look right, no file under `data/annotations/` holds anything but photo ids, and the report states that train, val and test are unverified pseudo-labels.

### Stage 4: baselines without training

**Goal:** know the floor before training.

1. Evaluate on val the torchvision Faster R-CNN with COCO weights **without training**, using its `bottle` class, which already exists in COCO (the code from stage 2). Val labels are Grounding DINO pseudo-labels, so the numbers are **agreement with Grounding DINO, not accuracy**, and every table says so.
2. The Grounding DINO zero-shot baseline is **trivial under these labels** (the model is compared with itself), so it is skipped, or reported only as a sanity check of the evaluation pipeline (it should reach about 100% agreement, apart from the height floor and the filters).
3. Measure mAP@0.5, mAP@0.5:0.95 and recall with `pycocotools` (the COCO reference implementation, pinned; `src/evaluation/`, `scripts/evaluate_detections.py`), with `maxDets` 300 because dense shelves hold up to about 125 boxes. Report two Faster R-CNN variants: the raw output and the output with the stage 2 filters, at a low score threshold (0.05) for mAP and at the stage 2 operating point (0.3) for precision, recall and F1. **Predictions shorter than 1.5% of the image height are ignored** (neither matched nor counted as false positives), because the ground truth has no boxes that small; the number ignored is reported.

4. Report metrics with a bootstrap confidence interval resampled by store (whole stores are resampled, not photos), because val and test hold only a few stores each; the interval tells us how much a difference between two runs means.

**Gate:** the Faster R-CNN baseline (agreement with Grounding DINO) and the pipeline sanity check are recorded in `reports/`, using the same evaluation script that will be used later; when the demo set exists, the baselines are also measured on it, where the numbers are accuracy.

### Stage 5: Faster R-CNN fine tuning

**Goal:** an own detector that beats the Faster R-CNN COCO baseline on the shelf photos.

**Note (owner decision):** the detector is trained on Grounding DINO pseudo-labels, so it **imitates Grounding DINO, including its known errors** (for example fragment boxes inside bottles in dense stacks, or boxes on can packs). Its val and test numbers are agreement with Grounding DINO, not accuracy; accuracy is measured only on the demo set.

1. Write the PyTorch `Dataset`. Each item returns the image and a `target` with `boxes` in xyxy format (float, in pixels) and `labels` (int64). Label 0 is reserved for background, so `bottle` is 1.
2. Load `fasterrcnn_resnet50_fpn_v2` with COCO weights and replace the `FastRCNNPredictor` with one with `num_classes = 2` (background + bottle).
3. Tune for dense shelves:
    - `box_detections_per_img` from 100 (default) to 300 or more, or bottles get lost on full shelves.
    - Raise `rpn_pre_nms_top_n` and `rpn_post_nms_top_n` in the same proportion.
    - Raise `min_size` and `max_size` of the internal transform if bottles are small, or reuse the tiling from stage 2.
3b. Predictions shorter than 1.5% of the image height are ignored in every evaluation (neither matched nor false positives), with the same function as in stage 4; training targets use the same floor because the labels already respect it.
4. Augmentations with `torchvision.transforms.v2` and `tv_tensors.BoundingBoxes`, which transform boxes together with the image: horizontal flip, brightness and color jitter, scale. No large rotations: shelves are photographed upright.
5. **Sanity check before training for real:** deliberately overfit 10 images (seeded random train photos, no augmentation). The loss must drop almost to zero and predictions must match the labels. If it does not, there is a bug in the dataset, not in the model. **Timing first:** before any long run, measure seconds per iteration and memory on the real device for a few batch sizes and sizes, and estimate the run time; if a full run would take more than a few hours, propose Colab (only photo-id-named copies, nothing from Colab committed).
6. Full training: SGD with lr 0.005 to 0.01, momentum 0.9, weight decay 1e-4, between 20 and 40 epochs, mixed precision (AMP) only if the timing check shows a gain. Batch size, linear warmup and cosine decay are set from the timing and memory check and recorded in `configs/stage5.yaml`. **Long runs on the Mac** run under `caffeinate -i`, save a checkpoint every epoch in `models/` (ignored) and can resume from the last one (`--resume`). The loss per epoch and the val agreement of each checkpoint go to a CSV in an ignored folder and are plotted in the report. Keep the best checkpoint by val AP@0.5 (agreement with the pseudo-labels).
6b. **Second run `frcnn_v2` (owner decision after the first run):** 12 epochs with warmup 1 epoch and the cosine decay compressed to that length (`configs/stage5_v2.yaml`), using its **last epoch, not chosen on val**. It is evaluated with the paired bootstrap against the `frcnn_v1` last epoch and the COCO baseline, and it is the detector of the pipeline unless it is clearly worse (the interval of the AP@0.5 or AP@0.5:0.95 difference against `frcnn_v1` last epoch entirely below zero).
7. Error analysis: draw false positives and false negatives on val and group them by cause (occlusion, reflection, small bottle, pseudo-label error).

8. Report metrics with a bootstrap confidence interval resampled by store (whole stores are resampled, not photos), because val and test hold only a few stores each; the interval tells us how much a difference between two runs means. **Compare with the stage 4 baseline with a paired bootstrap of the difference** (the same resampled stores for both models): "clearly beats the baseline" means the interval of the difference excludes zero.

**Gate:** val agreement (mAP@0.5) clearly beats the Faster R-CNN COCO baseline (paired bootstrap interval of the difference above zero), val agreement recall reaches the agreed target (for example 90%), and the main error causes are identified. Every number is labeled as agreement with Grounding DINO.

### Stage 6: reference catalog and embeddings

**Goal:** identify the product (one of the 6 ids, or `unknown`) of each detected bottle by comparing against a catalog.

1. **Catalog from product photos (owner decision):** embeddings replace training a classifier, so the catalog needs only a few reference images per product. It is built **only** from `data/catalog/product_photos/<product id>/<product id>_NN.<ext>` (ignored by git; about 5 clean photos per product that the owner provides) with `sources.csv` next to it (file, product id, source URL, license if known, whether the Uruguayan packaging was checked). The source of every catalog entry is recorded in the catalog table. Photos must match the Uruguayan packaging of the variant. They are never used for evaluation, and the final report lists them as a dependency to replace before any commercial use. There is no labeling of train crops for the catalog. While `sources.csv` still has `unknown` sources or licenses (or the packaging is unconfirmed), development goes on, but **a final report cannot be written**: `python scripts/check_catalog_photos.py --final` must pass first.
1b. **Catalog crops:** each product photo holds exactly one bottle on a plain background, so before embedding it is cropped to the bottle with the frozen Grounding DINO (prompt `bottle.`, the same 10% margin as the shelf crops, no Stage 2 filters). More than one box is almost surely fragments of the same bottle: the largest box is used and the photo listed; no box: the whole photo is used and the photo listed. Crops go to an ignored folder, with a contact sheet of all crops in an ignored review folder for the owner to check.
2. **Catalog variety:** the product photos are mostly frontal, so each one gets several automatic augmentations (small rotations, mild perspective, scale, brightness and contrast, slight blur; about 5 or 6 per photo, applied to the bottle crop), each stored as its own catalog entry tagged `augmented` with its source photo. **A contact sheet per product (originals next to their variants) is reviewed by the owner before anything goes into the database** (`scripts/preview_catalog_augment.py`; the preview uses the same seeds as the real build). **No horizontal flips (they mirror the label text) and no hue or saturation changes (colour separates the products: Coca-Cola red, Light silver, Fanta orange, Sprite green)**; these differ from the detector training augmentations on purpose. In 6b, measure on the val sample whether the augmented catalog identifies better than the plain one, and keep it only if it does. AI-generated product images are not used by default (they tend to alter label text, cap colour and bottle shape, which are exactly the cues that separate the variants); if the owner wants to try them later, it is a separate, tagged experiment measured on the val sample.
3. **Domain gap:** studio photos with a plain background look different from shelf crops. Measure it in 6b (similarity of val crops to their correct product, product photos only). If identification is poor because of it, propose a small number of shelf crops in the catalog as a second step; this is not done by default. Any shelf crops proposed for the catalog must come from **train photos only**, never from val or test (val crops are what is measured on), and would need a new labeling round: propose it and wait for approval.
4. Try two cropping methods for the query crops: the box with a margin of about 10%, and the SAM 2 mask (box prompts, Apache 2.0; check its speed on MPS first) over a neutral background. Keep whichever gives better results.
5. Compute the embedding with DINOv2 (base): a 768-dimension vector, normalized.
6. Store in pgvector (through `pg8000`) in **its own database** (for example `soda_catalog`) inside the running Postgres container, never in the Label Studio database; tests use a throwaway database that they create and drop; never `docker compose down -v`. Columns for product, brand, size, material (PET or glass), source image, source URL and license, whether the entry is augmented, and an HNSW index on cosine distance.
7. **Val crop sample, labeled by hand without suggestions (the yardstick):** about 180 val crops cut from the pseudo-label boxes, stratified by store (proportional, at least 15 per store, at most 8 per photo), in a second Label Studio project (single choice per crop): the 6 product ids, `unknown`, `not_a_bottle`, `cannot_tell`. The `not_a_bottle` share is a direct measure of the pseudo-label precision on val. If too few `unknown` crops come out (under about 40), a small targeted batch from photos with other brands is added. This sample also calibrates the unknown rules (see "Unknown handling": option (a)); the demo set stays whole for the final evaluation. Crops of the demo set are labeled with the same tool later.
8. Report the number of catalog photos per product (the 6 ids; with their augmentations). Products with too few photos are flagged as **low data** and reported separately, not dropped.
9. Measure on the val sample top-1 and top-3 accuracy, brand accuracy (Coca-Cola, Sprite or Fanta) and a confusion matrix across the 6 products plus `unknown`, plus the unknown metrics below. These are **development numbers** (the sample also tunes the rules, so thresholds are tuned leaving one store out); the demo set gives the accuracy. Also report what the system answers for the `not_a_bottle` crops (ideally `unknown` or `review`, since the detector sends boxes like these), exclude `not_a_bottle` and `cannot_tell` from the accuracy, and report the unknown rejection rate with and without store `s03` (it holds 30 of the 67 unknown crops) so that one store does not drive the conclusion. DINOv2 speed on MPS is measured first; anything too slow is reported, not worked around.
10. Report metrics with a bootstrap confidence interval resampled by store (whole stores are resampled, not photos), because val holds only a few stores; the interval tells us how much a difference between two runs means.

**Gate:** brand accuracy is high, unknown handling is measured, and confusions between products are identified by name (for example, `coca_classic` versus `coca_classic_returnable`). If product photos and shelf photos end up far apart in the embedding space, first add a few shelf crops to the catalog (item 3); only if that is not enough, open a metric-learning sub-stage (fine tune DINOv2 with a contrastive or triplet loss) using pairs of shelf crops.

#### Unknown handling (open-set recognition)

Any bottle that is not one of the 6 target products must end up as `unknown`. The detector still labels every bottle as `bottle` (including water, juice and other brands); the decision happens after the embeddings. Stage 6 builds the pieces and stage 8 calibrates and evaluates them.

1. **Ground truth:** the owner labels the val crop sample in stage 6 (and the demo set crops later): every crop gets one of the 6 ids, `unknown`, `not_a_bottle` or `cannot_tell`. The sample must include enough `unknown` crops (other brands, water, juice); without labeled unknowns the thresholds cannot be calibrated.
2. **Negative gallery:** besides the 6 products, the catalog may hold reference images of common non-target bottles (for example water, juices, other sodas), all tagged `unknown`. If the nearest neighbors are mostly from this gallery, the crop is `unknown`. Measure whether this helps compared with thresholds alone.
3. **Rejection rules,** applied per crop after top-K retrieval:
    - top-1 similarity below a threshold → `unknown`;
    - margin between the best product and the second best *different* product below a threshold → `review`, or `unknown` if the similarity is also low;
    - top-K neighbors disagree on the product (k-NN vote below a minimum share) → `review` or `unknown`;
    - OCR contradicts the top-1 (for example reads "FANTA" when top-1 is a Coca-Cola product) → `review`; this rule exists only in pipeline variant B (with OCR).
4. **Calibration:** all thresholds are tuned on val only, possibly per product if one product behaves differently. The calibration ground truth is the **val crop sample** (about 180 crops labeled by the owner, including unknowns; decided: option (a)), which keeps the demo set whole for the final evaluation. If the sample turns out too small to calibrate, a calibration part of the demo set is the fallback (option (b)), to be agreed first. Test is used once, in stage 8.
5. **Metrics** (also in the metrics table): accuracy on known products; unknown rejection rate (share of truly unknown crops classified as `unknown` or `review`); false known rate (share of truly unknown crops assigned to one of the 6 products with `automatic` status). The false known rate is the most important number and should stay below 5%.

### Stage 7: OCR as evidence (pipeline variant B)

**Goal:** build the second pipeline variant, embeddings plus OCR evidence, so that the two variants can be compared. The final pipeline exists in **two variants** (owner request): **A, embeddings only** (catalog similarity plus the unknown rules, stage 6) and **B, embeddings plus OCR** (text read on the crop). Same detector, same catalog, same val calibration sample for both; the thresholds of each variant are calibrated separately.

1. **OCR engine:** PaddleOCR was the plan. Before installing anything, compare PaddleOCR, EasyOCR (Apache 2.0, PyTorch) and Tesseract on licence, install weight and speed on this Mac (Apple Silicon: MPS or CPU); recommend one and ask before installing it.
2. Run the OCR on the same crop used for the embedding.
3. Normalize the text: uppercase, no accents, no double spaces.
4. Keywords per product: "COCA-COLA", "ZERO", "ZERO AZUCARES", "LIGHT", "RETORNABLE" (the returnable text on the label may be the clearest way to separate `coca_classic_returnable` from `coca_classic`), "SPRITE", "FANTA", and sizes such as "500 ML", "1,5 L" or "2,25 L".
5. Compare with fuzzy matching (`rapidfuzz`), because OCR confuses O with 0 and similar characters.
6. Use the evidence as agreement or contradiction signals for stage 8 (contradiction sends the crop to `review`).
7. Measure the percentage of crops in which OCR reads anything useful, and the time cost per photo of each variant.
8. **Compare A and B** on the val sample as development numbers (thresholds calibrated separately for each variant, leave-one-store-out) and, once, on the demo set as the real numbers: top-1 accuracy, per-product confusion matrices, unknown rejection rate, false known rate, and the automatic / review / unknown shares. Use a paired bootstrap of the difference by store. Report separately the pairs where OCR is expected to matter most: `coca_classic` vs `coca_zero` vs `coca_light`, and `coca_classic` vs `coca_classic_returnable`.

**Gate:** both variants run and are compared on val. OCR stays as an option only if it reduces the specific confusions found in stage 6 (otherwise the report says it did not help).

### Stage 8: confidence and decision

**Goal:** make the system know when it does not know.

1. Gather the signals per detection: detector score, top-1 similarity, margin between the top-1 product and the best different product, and (variant B only) whether OCR agrees or contradicts.
2. Combine them into a simple score: a weighted sum or a logistic regression trained on val.
3. **Calibrate each variant (A and B) separately** on the val crop sample labeled by the owner (about 180 crops, including unknowns; see "Unknown handling") for the three outcomes: automatic, human review and unknown, using the rejection rules from "Unknown handling" in stage 6. Cosine similarity is not a probability: 0.94 does not mean 94% certainty. Val must contain labeled `unknown` crops, otherwise this step cannot be done.
4. Evaluate **only once** on the demo set, for each variant, with the thresholds already fixed (it is never used to tune or calibrate anything; demo photos used for the reference catalog are excluded). This is the only accuracy number. Test is also used once, but it has no product labels, so only the detection agreement and the automatic / review / unknown percentages are reported for it, as behavior of the pipeline, not accuracy.

5. Report metrics with a bootstrap confidence interval resampled by store (whole stores are resampled, not photos), because val and test hold only a few stores each; the interval tells us how much a difference between two runs means.

**Gate:** on the demo set, the rate of confident wrong identifications stays below 5% (the target from the original document), the false known rate (truly unknown crops assigned to a target product with `automatic` status) stays below 5%, and the percentages falling into automatic, review and unknown are reported.

### Stage 9: full pipeline and demo

**Goal:** process a new photo end to end and measure the operational value.

1. `POST /analyze` endpoint in FastAPI that receives a photo and returns the inventory as JSON. **A request option selects the pipeline variant** (for example `use_ocr: true` or `false`: B embeddings plus OCR, or A embeddings only), and the response says which variant produced it (`"variant": "embeddings"` or `"embeddings_ocr"`).
2. Also return the photo annotated with boxes, product and status.
3. Resolve review cases in Label Studio or in a minimal UI.
4. Measure time per photo, for each variant, and compare it with a manual count of the same shelf.
5. The final report gives **accuracy only for the demo set** (well-taken photos, corrected by the owner). Numbers on val and test are agreement with Grounding DINO pseudo-labels and are labeled that way every time; if the demo set shares stores with train, its number is marked optimistic.

```json
{
  "image": "shelf_042.jpg",
  "variant": "embeddings_ocr",
  "detections": [
    {"box": [412, 188, 470, 402], "product": "coca_zero", "size": "1.5 L",
     "confidence": 0.95, "status": "automatic"},
    {"box": [478, 190, 534, 400], "product": "coca_classic", "size": "1.5 L",
     "confidence": 0.81, "status": "review"},
    {"box": [540, 192, 596, 398], "product": "unknown", "size": null,
     "confidence": 0.41, "status": "unknown"}
  ],
  "summary": {"coca_zero": 1, "coca_classic": 1, "unknown": 1}
}
```

**Gate:** a never-seen photo is processed end to end, and the final report gathers the metrics from every stage.

## Metrics per stage

Targets are initial hypotheses, as in the original document: the POC exists to find out which values are achievable.

| Stage | Metric | Measured on | Initial target |
| --- | --- | --- | --- |
| 3 | Boxes dropped by the height floor; sanity errors | exported COCO, per split | Record; 0 errors |
| 3 | Pre-label precision and recall (IoU 0.5) against owner-corrected labels, biased toward the pre-labeler; correction minutes per photo | demo set (when it exists) | Record |
| 4 | Faster R-CNN COCO baseline mAP@0.5, mAP@0.5:0.95, recall: **agreement with Grounding DINO** (also accuracy on demo) | val (agreement), demo (accuracy) | Record as floor |
| 5 | Trained detector mAP@0.5 and recall: **agreement with Grounding DINO** (also accuracy on demo) | val (agreement), demo (accuracy) | Beat the baseline; recall of 90% or more |
| 6 | Brand accuracy (Coca-Cola, Sprite, Fanta), product top-1 and top-3 | demo crops (manual labels) | Brand 90% or more; top-3 80% or more |
| 6 | Accuracy on known products | demo crops | Record |
| 6 / 8 | Unknown rejection rate (truly unknown crops sent to `unknown` or `review`) | val sample (calibration), demo (once) | Record |
| 6 / 8 | False known rate (truly unknown crops assigned to a product as `automatic`) | val sample (calibration), demo (once) | Below 5% |
| 7 | Variant B (embeddings + OCR) minus variant A (embeddings only): top-1, per-product confusions, unknown rejection, false known rate, paired bootstrap by store; share of crops where OCR reads something useful; time per photo of each variant | val sample (development), demo (real) | Any improvement on the known confusions; record |
| 8 | Rate of confident wrong identifications | demo, only once | Below 5% |
| 8 | Automatic / review / unknown percentage | demo (accuracy) and test (behavior only) | Record |
| 9 | Time per photo versus manual count | new photos | Record |

## Risks and common mistakes

Most problems in a POC like this come from the data, not the model.

| Risk | How it shows | Mitigation |
| --- | --- | --- |
| Data leakage between splits | Excellent test metrics that do not hold on new photos | Split by store or shelf (stage 1) |
| EXIF orientation | Boxes shifted or rotated relative to the bottle | `exif_transpose` before anything else (stage 1) |
| Badly converted coordinates | Boxes correct in Python but shifted in Label Studio | Percentages in Label Studio, xyxy pixels in torchvision; test the conversion with one photo |
| Trusting the pre-labels | The detector learns the pre-labeler's mistakes | **Accepted for this POC by owner decision** (train, val and test are unverified pseudo-labels). In a real project: review every box. Here: every report says "agreement with Grounding DINO"; accuracy only on the owner-corrected demo set |
| Agreement read as accuracy | A high mAP on val that only means "same as Grounding DINO" | Each table says what it measures; the Grounding DINO baseline is trivial and is not reported as a result |
| Losing the Label Studio data | Labels and product choices exist only in the Docker volume | Never `docker compose down -v`; a timestamped JSON export after each session (stage 3) |
| Detection cap | Bottles missing on full shelves even though the model sees them | Raise `box_detections_per_img` (stage 5) |
| Small bottles in large photos | Low recall in far rows | Tiling or larger `min_size` (stages 2 and 5) |
| Reflections and occlusion | False negatives on covered or shiny bottles | Brightness augmentations; the guide defines what gets labeled |
| Nearly identical products | Classic and Zero, or returnable and non-returnable, confused in the matrix | OCR (stage 7) and embedding fine tuning (stage 6b) |
| Packaging changes | A special edition does not match the catalog | `unknown` status and fast catalog onboarding |
| Looking at test during development | Optimistic thresholds | Test frozen until stage 8 |
| Photos or labels committed by accident | Images or annotation files appear in a commit | `.gitignore` created in stage 0, before any photo is copied; check `git status` before each commit |

## Learning resources

Ordered by the stage where they are used. The torchvision tutorial is the most direct starting point: it does almost exactly what stage 5 does.

| Stage | Resource | What for |
| --- | --- | --- |
| 2 | [Grounding DINO in Transformers](https://huggingface.co/docs/transformers/model_doc/grounding-dino) | Model usage, prompts and post-processing |
| 6 | [SAM 2](https://github.com/facebookresearch/sam2) | Segmenting with a box as prompt |
| 2 | [SAHI](https://github.com/obss/sahi) | Tiled inference for small objects |
| 3 | [Label Studio: import pre-annotations](https://labelstud.io/guide/predictions) | `predictions` JSON format |
| 5 | [torchvision detection fine tuning tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html) | Dataset, predictor replacement and training loop |
| 5 | [Transforms v2 for detection](https://pytorch.org/vision/stable/auto_examples/transforms/plot_transforms_e2e.html) | Augmentations that move boxes with the image |
| 5 | [SKU-110K paper](https://arxiv.org/pdf/1904.00853) | Why dense shelves are hard for detectors |
| 6 | [DINOv2](https://github.com/facebookresearch/dinov2) | Embedding extraction |
| 6 | [pgvector](https://github.com/pgvector/pgvector) | HNSW indexes and distance operators |
| 7 | [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) | Text detection and recognition |
