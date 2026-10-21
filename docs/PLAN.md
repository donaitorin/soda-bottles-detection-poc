# POC Plan: Soda Bottle Detection on Store Shelves

## Goal and scope

The POC validates whether, from store shelf photos, a system can detect soda bottles, identify which ones are one of the 10 target products (Coca-Cola, Sprite and Pepsi lines) and send every other bottle to `unknown`, with a low rate of confident wrong identifications. It follows the same architecture as the bathroom asset POC: a generic detector, identification through embeddings, text evidence, and a confidence engine.

**Central question:** can the system count and identify the 10 target products in a shelf photo, sending every other bottle to `unknown`, accurately enough to reduce manual counting?

**In scope**

- Generic detection of soda bottles in the photo.
- Identification of the 10 target products (Coca-Cola Classic, Zero and Light; Sprite and Sprite Zero; Pepsi; with returnable versions where they exist), with every other bottle sent to `unknown`, plus size when visible.
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
2. Pre-labeling (Grounding DINO + SAM 2)                 v
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
| Pre-labeling | Grounding DINO ([grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base)) via `transformers` | Proposes boxes from a text prompt such as "bottle." | Apache 2.0 |
| Box refinement | SAM 2 | Fits each box to the actual bottle outline; masks for clean crops | Apache 2.0 |
| Pre-labeling alternative | [SAM 3](https://github.com/facebookresearch/sam3) | Detects and segments from text in a single model | Meta's own SAM License; review before commercial use |
| Manual validation | Label Studio Community | Correcting pre-labels and exporting to COCO | Apache 2.0 |
| Detector | Faster R-CNN ResNet50 FPN v2 (`torchvision`) with COCO weights | Detects `bottle` on the shelf | BSD-3 |
| Embeddings | DINOv2 | One vector per crop to compare against the catalog | Apache 2.0 |
| Vector search | PostgreSQL + pgvector | Top-K by cosine similarity | PostgreSQL License |
| OCR | PaddleOCR | Reads "Zero", "sin azúcar", "1,5 L" on the packaging | Apache 2.0 |
| Metrics | `pycocotools` or `torchmetrics` | mAP, recall, precision | BSD / Apache 2.0 |
| Error analysis | FiftyOne (optional) | Inspect false positives and negatives on the photos | Apache 2.0 |
| API | FastAPI | Photo to inventory endpoint | MIT |

Licenses are indicative: before any commercial use, someone should confirm them, including the licenses of the pretrained weights.

## Assumptions and requirements

The plan assumes an own set of shelf photos, a GPU or Apple Silicon accelerator for pre-labeling and training, and a reproducible Python environment. If any of these is missing, stage 0 catches it before time is invested.

- **Photos:** about 300 supermarket shelf photos with beverages are already available. Each photo contains many bottles, so that should yield thousands of instances, which is enough for the POC.
- **Version control:** photos and labeling files never go into git. That covers `data/raw/`, the pre-labels, Label Studio exports, the COCO annotations used to train the detector, crops, catalog images, masks, checkpoints and downloaded models. Only code, configs, docs and small reports are committed.
- **Diversity:** different stores, lighting, angles and distances. Ten photos of the same shelf add less than ten different shelves.
- **Accelerator:** Grounding DINO, SAM and training need a GPU. On a Mac, PyTorch uses MPS, which is enough for development and small runs; full training or pre-labeling every photo may be faster on Colab or a cloud VM with a 12 to 16 GB GPU.
- **Environment:** Python 3.11 in a conda environment (`soda-poc`, conda-forge only, defined in `environment.yml`; torch and torchvision installed with pip because PyTorch has no official conda packages), compatible PyTorch and torchvision versions, Docker for Postgres with pgvector and for Label Studio. Nothing runs in conda's `base` env.
- **Catalog:** closed list of the 10 target products, in `docs/catalog_variants.md`, decided in stage 0.

**Suggested repo structure**

```
soda-bottles-detection-poc/
├── data/                 # gitignored, except .gitkeep files and splits/
│   ├── raw/              # the ~300 original photos, never modified (gitignored)
│   ├── splits/           # train.txt, val.txt, test.txt (anonymous photo ids only, committed)
│   ├── prelabels/        # Grounding DINO + SAM output (gitignored)
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
3. Define the closed list of catalog products. It has 10 ids, listed in `docs/catalog_variants.md`: `coca_classic`, `coca_classic_returnable`, `coca_zero`, `coca_zero_returnable`, `coca_light`, `sprite`, `sprite_returnable`, `sprite_zero`, `pepsi` and `pepsi_returnable`. Returnable bottles can be PET or glass (glass ones in crates count as returnable products, Classic or Zero). Size and material (PET or glass) are secondary metadata stored in the catalog, not hard success criteria. Any other bottle (other brands, water, juice) is `unknown`; there is no `other brand` class. This list is used from stage 6 on, not for the detector.
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

### Stage 2: pre-labeling with Grounding DINO + SAM 2

**Goal:** generate candidate boxes for all photos and measure how good they are.

1. Run Grounding DINO with the prompt `bottle.`. According to its [model card](https://huggingface.co/IDEA-Research/grounding-dino-base), prompts must be lowercase and end with a dot. Also try `soda bottle.` and `plastic bottle.`.
2. Start with a box threshold of 0.3 and a text threshold of 0.25, and tune by looking at results.
3. If bottles come out small in large photos, apply tiling: split the photo into overlapping tiles, detect in each one and merge with NMS. The SAHI library does this.
4. Filter typical errors: boxes covering a whole shelf, duplicates (NMS) and tiny boxes.
5. Pass each box to SAM 2 as a prompt, get the mask and recompute the box fitted to the outline. Keep the masks: they are useful in stage 6.
6. Revisit the provisional ~20 px minimum box height from the labeling guide using the box size distribution (resolutions range from 374 to 4096 px, so measure it on the original image).
7. Convert to Label Studio JSON with the `predictions` key and type `rectanglelabels`. Careful: Label Studio uses coordinates as **percentages** of image width and height (0 to 100), not pixels.

**Gate:** hand-label 20 to 30 train photos from scratch. Measure the precision and recall of the pre-labeling against them at IoU 0.5, with and without SAM. If recall is low (for example under 70%), tune prompts, thresholds or tiling before moving on. If SAM does not improve box fit, drop it to keep things simple.

### Stage 3: manual validation in Label Studio

**Goal:** turn the pre-labels into a reliable dataset in COCO format.

1. Create the project with a `RectangleLabels` configuration using the classes from stage 0 (only `bottle`), plus a per-photo multi-select `Choices` field: "Which target products appear in this photo?", with the 10 product ids and "other bottles present". It takes seconds per photo and is not a per-box label.
2. Import the tasks with their predictions and correct them: delete false positives, add missing bottles, adjust boxes.
3. Label val and test with the same process, but verify every single box: they are the yardstick for everything.
4. Do a second review of 10% of the photos, ideally by another person or by yourself on a different day.
5. Export to COCO and run a sanity script: boxes inside the image, area greater than zero, no duplicates (IoU greater than 0.9), reasonable count per photo.
6. Draw the boxes on 20 random photos and look at them.
7. Record how long it took to correct each photo: it is useful data for the final report.
8. Using the per-photo product field, report the number of photos per product (the 10 ids and "other bottles present") for each split. Products that appear in very few train photos are flagged as **low data** and reported separately, not dropped.
9. Check that val and test contain every target product and some photos with other bottles. If not, rebalance the split before stage 4.

**Gate:** the sanity script reports no errors, the second review finds few differences, and the 20 drawn photos look right.

### Stage 4: baselines without training

**Goal:** know the floor before training.

1. Evaluate on val the torchvision Faster R-CNN with COCO weights **without training**, using its `bottle` class, which already exists in COCO.
2. Evaluate Grounding DINO zero-shot on val with the best prompts from stage 2.
3. Measure mAP@0.5, mAP@0.5:0.95 and recall with `pycocotools` or `torchmetrics`.

4. Report metrics with a bootstrap confidence interval resampled by store (whole stores are resampled, not photos), because val and test hold only a few stores each; the interval tells us how much a difference between two runs means.

**Gate:** both baselines are recorded in `reports/`, using the same evaluation script that will be used later.

### Stage 5: Faster R-CNN fine tuning

**Goal:** an own detector that beats both baselines on the shelf photos.

1. Write the PyTorch `Dataset`. Each item returns the image and a `target` with `boxes` in xyxy format (float, in pixels) and `labels` (int64). Label 0 is reserved for background, so `bottle` is 1.
2. Load `fasterrcnn_resnet50_fpn_v2` with COCO weights and replace the `FastRCNNPredictor` with one with `num_classes = 2` (background + bottle).
3. Tune for dense shelves:
    - `box_detections_per_img` from 100 (default) to 300 or more, or bottles get lost on full shelves.
    - Raise `rpn_pre_nms_top_n` and `rpn_post_nms_top_n` in the same proportion.
    - Raise `min_size` and `max_size` of the internal transform if bottles are small, or reuse the tiling from stage 2.
4. Augmentations with `torchvision.transforms.v2` and `tv_tensors.BoundingBoxes`, which transform boxes together with the image: horizontal flip, brightness and color jitter, scale. No large rotations: shelves are photographed upright.
5. **Sanity check before training for real:** deliberately overfit 10 images. The loss must drop almost to zero and predictions must match the labels. If it does not, there is a bug in the dataset, not in the model.
6. Full training: SGD with lr 0.005 to 0.01, momentum 0.9, weight decay 1e-4, between 20 and 40 epochs, mixed precision (AMP). Keep the best checkpoint by val mAP.
7. Error analysis: draw false positives and false negatives on val and group them by cause (occlusion, reflection, small bottle, label error).

8. Report metrics with a bootstrap confidence interval resampled by store (whole stores are resampled, not photos), because val and test hold only a few stores each; the interval tells us how much a difference between two runs means.

**Gate:** val mAP@0.5 clearly beats both baselines, val bottle recall reaches the agreed target (for example 90%), and the main error causes are identified.

### Stage 6: reference catalog and embeddings

**Goal:** identify the product (one of the 10 ids, or `unknown`) of each detected bottle by comparing against a catalog.

1. Build the catalog: 5 to 10 images per product, mixing product photos with real shelf crops. Crops come **only from train**, never from val or test.
2. Try two cropping methods: the box with a margin of about 10%, and the SAM mask over a neutral background. Keep whichever gives better results.
3. Compute the embedding with DINOv2 (base): a 768-dimension vector, normalized.
4. Store in pgvector, with columns for product, brand, size, material (PET or glass) and source image, and an HNSW index on cosine distance.
5. Label the product of the val and test crops (in Label Studio, with `Choices` on each crop): one of the 10 ids or `unknown`. This is the ground truth for this stage (see "Unknown handling" below).
6. Report the number of train crops per product (the 10 ids and `unknown`). Products with too few train crops are flagged as **low data** and reported separately, not dropped.
7. Measure top-1 and top-3 accuracy, brand accuracy (Coca-Cola, Sprite or Pepsi) and a confusion matrix across the 10 products plus `unknown`, plus the unknown metrics below.

8. Report metrics with a bootstrap confidence interval resampled by store (whole stores are resampled, not photos), because val and test hold only a few stores each; the interval tells us how much a difference between two runs means.

**Gate:** brand accuracy is high, unknown handling is measured, and confusions between products are identified by name (for example, `coca_classic` versus `coca_classic_returnable`). If product photos and shelf photos end up far apart in the embedding space, open sub-stage 6b: fine tune DINOv2 with metric learning (contrastive or triplet loss) using pairs of train crops.

#### Unknown handling (open-set recognition)

Any bottle that is not one of the 10 target products must end up as `unknown`. The detector still labels every bottle as `bottle` (including water, juice and other brands); the decision happens after the embeddings. Stage 6 builds the pieces and stage 8 calibrates and evaluates them.

1. **Ground truth:** when labeling crops in stage 6, every crop gets one of the 10 ids or `unknown`. Val and test must include enough `unknown` crops (other brands, water, juice). Without labeled unknowns the thresholds cannot be calibrated.
2. **Negative gallery:** besides the 10 products, the catalog may hold reference images of common non-target bottles (for example water, juices, other sodas), all tagged `unknown`. If the nearest neighbors are mostly from this gallery, the crop is `unknown`. Measure whether this helps compared with thresholds alone.
3. **Rejection rules,** applied per crop after top-K retrieval:
    - top-1 similarity below a threshold → `unknown`;
    - margin between the best product and the second best *different* product below a threshold → `review`, or `unknown` if the similarity is also low;
    - top-K neighbors disagree on the product (k-NN vote below a minimum share) → `review` or `unknown`;
    - OCR contradicts the top-1 (for example reads "PEPSI" when top-1 is a Coca-Cola product) → `review`.
4. **Calibration:** all thresholds are tuned on val only, possibly per product if one product behaves differently. Test is used once, in stage 8.
5. **Metrics** (also in the metrics table): accuracy on known products; unknown rejection rate (share of truly unknown crops classified as `unknown` or `review`); false known rate (share of truly unknown crops assigned to one of the 8 products with `automatic` status). The false known rate is the most important number and should stay below 5%.

### Stage 7: OCR as evidence

**Goal:** use the packaging text to resolve the confusions found in stage 6.

1. Run PaddleOCR on each crop.
2. Normalize the text: uppercase, no accents, no double spaces.
3. Define keywords per product: "ZERO", "SIN AZUCAR", "LIGHT", "SPRITE", "PEPSI", "ZERO AZUCARES", "RETORNABLE" (the returnable text on the label may be the clearest way to separate returnable from non-returnable bottles), and sizes such as "500 ML", "1,5 L" or "2,25 L".
4. Compare with fuzzy matching (`rapidfuzz`), because OCR confuses O with 0 and similar characters.
5. Measure the percentage of crops in which OCR reads something useful.

**Gate:** compare on val visual-only identification against visual + OCR. OCR stays only if it reduces the specific confusions from stage 6.

### Stage 8: confidence and decision

**Goal:** make the system know when it does not know.

1. Gather the signals per detection: detector score, top-1 similarity, margin between the top-1 product and the best different product, and whether OCR agrees or contradicts.
2. Combine them into a simple score: a weighted sum or a logistic regression trained on val.
3. Calibrate on val the thresholds for the three outcomes: automatic, human review and unknown, using the rejection rules from "Unknown handling" in stage 6. Cosine similarity is not a probability: 0.94 does not mean 94% certainty. Val must contain labeled `unknown` crops, otherwise this step cannot be done.
4. Evaluate **only once** on test, with the thresholds already fixed.

5. Report metrics with a bootstrap confidence interval resampled by store (whole stores are resampled, not photos), because val and test hold only a few stores each; the interval tells us how much a difference between two runs means.

**Gate:** on test, the rate of confident wrong identifications stays below 5% (the target from the original document), the false known rate (truly unknown crops assigned to a target product with `automatic` status) stays below 5%, and the percentages falling into automatic, review and unknown are reported.

### Stage 9: full pipeline and demo

**Goal:** process a new photo end to end and measure the operational value.

1. `POST /analyze` endpoint in FastAPI that receives a photo and returns the inventory as JSON.
2. Also return the photo annotated with boxes, product and status.
3. Resolve review cases in Label Studio or in a minimal UI.
4. Measure time per photo and compare it with a manual count of the same shelf.

```json
{
  "image": "shelf_042.jpg",
  "detections": [
    {"box": [412, 188, 470, 402], "product": "coca_zero", "size": "500 ml",
     "confidence": 0.95, "status": "automatic"},
    {"box": [478, 190, 534, 400], "product": "coca_classic", "size": "500 ml",
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
| 2 | Pre-labeling precision and recall (IoU 0.5) | 20 to 30 hand-labeled train photos | Recall of 70% or more |
| 3 | Correction minutes per photo | Label Studio | Record, no target |
| 4 | Baseline mAP@0.5, mAP@0.5:0.95, recall | val | Record as floor |
| 5 | Trained detector mAP@0.5 and recall | val | Beat the baselines; recall of 90% or more |
| 6 | Brand accuracy (Coca-Cola, Sprite, Pepsi), product top-1 and top-3 | val crops | Brand 90% or more; top-3 80% or more |
| 6 | Accuracy on known products | val crops | Record |
| 6 / 8 | Unknown rejection rate (truly unknown crops sent to `unknown` or `review`) | val (calibration), test (once) | Record |
| 6 / 8 | False known rate (truly unknown crops assigned to a product as `automatic`) | val (calibration), test (once) | Below 5% |
| 7 | Top-1 improvement with OCR versus visual only | val crops | Any improvement on the known confusions |
| 8 | Rate of confident wrong identifications | test, only once | Below 5% |
| 8 | Automatic / review / unknown percentage | test | Record |
| 9 | Time per photo versus manual count | new photos | Record |

## Risks and common mistakes

Most problems in a POC like this come from the data, not the model.

| Risk | How it shows | Mitigation |
| --- | --- | --- |
| Data leakage between splits | Excellent test metrics that do not hold on new photos | Split by store or shelf (stage 1) |
| EXIF orientation | Boxes shifted or rotated relative to the bottle | `exif_transpose` before anything else (stage 1) |
| Badly converted coordinates | Boxes correct in Python but shifted in Label Studio | Percentages in Label Studio, xyxy pixels in torchvision; test the conversion with one photo |
| Trusting the pre-labels | The detector learns Grounding DINO's mistakes | Review every box; stage 3 is never skipped |
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
| 2 | [SAM 2](https://github.com/facebookresearch/sam2) | Segmenting with a box as prompt |
| 2 | [SAHI](https://github.com/obss/sahi) | Tiled inference for small objects |
| 3 | [Label Studio: import pre-annotations](https://labelstud.io/guide/predictions) | `predictions` JSON format |
| 5 | [torchvision detection fine tuning tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html) | Dataset, predictor replacement and training loop |
| 5 | [Transforms v2 for detection](https://pytorch.org/vision/stable/auto_examples/transforms/plot_transforms_e2e.html) | Augmentations that move boxes with the image |
| 5 | [SKU-110K paper](https://arxiv.org/pdf/1904.00853) | Why dense shelves are hard for detectors |
| 6 | [DINOv2](https://github.com/facebookresearch/dinov2) | Embedding extraction |
| 6 | [pgvector](https://github.com/pgvector/pgvector) | HNSW indexes and distance operators |
| 7 | [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) | Text detection and recognition |
