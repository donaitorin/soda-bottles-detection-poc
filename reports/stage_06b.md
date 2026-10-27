# Stage 6b report: catalog, retrieval, unknown rules and development numbers

> **Every identification number in this report is a DEVELOPMENT number on 180 labeled val crops from 6 stores (thresholds tuned leaving one store out), not accuracy.** The demo set will be the yardstick. Detector numbers elsewhere are agreement with Grounding DINO pseudo-labels. Test was not touched.
>
> **Status: tasks 1 to 9 done; the Stage 6 gate is NOT met** (section "Gate and what to do next"). The headline: with the 30 catalog photos the nearest product is right for 53.5% of the known crops (top-3 90.7%, brand 88.4%), the correct and the best wrong product are almost equally similar (a large domain gap), so the rules, tuned to avoid confident errors, send almost every crop to `review` and answer nothing automatically.

## Decisions applied (owner)

- **Detector for the next stages: `frcnn_v2` last epoch (`epoch_12.pt`).** The other 11 epoch checkpoints of `frcnn_v2` were deleted; `best.pt` and `epoch_12.pt` stay in `models/stage5/frcnn_v2/`. **Freed 3.7 GB** (3,712,236 KB; 506 MB remain).
- **OCR:** Tesseract trial finished and the final engine recommendation is EasyOCR (details in `reports/stage_06b_prep.md`, section 9). The throwaway environment `ocr-trial` was removed (`conda env remove -n ocr-trial`; it is no longer in `conda env list`). `soda-poc` never had EasyOCR, python-bidi or pytesseract.
- **Plan additions** (`docs/PLAN.md` Stage 6): the catalog goes in its own database (for example `soda_catalog`) in the running Postgres container, never in the Label Studio database, tests use a throwaway database they create and drop, never `docker compose down -v`; shelf crops proposed for the catalog (experiment 8b) must come from train photos only and need a new labeling round (propose and wait); the system's answer for the 15 `not_a_bottle` crops is reported (ideally `unknown` or `review`); the unknown rejection rate is reported with and without store `s03` (30 of the 67 unknown crops); DINOv2 speed on MPS is measured first.
- **Augmentation zoom capped at 1.05** (owner, after the preview: caps and bases are cues); all other augmentation settings approved as they were; the contact sheets were regenerated with the new limit and the same seeds. **OCR for Stage 7:** EasyOCR accepted with `python-bidi` (LGPL) as a documented exception, in a separate pinned `requirements-ocr.txt` installed only for variant B, used unmodified; the pipeline must run fully without it (variant A) and fail clearly if `use_ocr` is requested without it (nothing to implement yet; recorded in `CLAUDE.md` and `docs/PLAN.md`).
- **`sources.csv`:** the owner is filling in `source_url` and `license` for the 30 photos. Stage 6b is not blocked on it, but **a final report cannot be written while any row says unknown**: `python scripts/check_catalog_photos.py --final` fails (today: 90 errors, a `source_url` and a `license` unknown and the packaging unconfirmed on each of the 30 rows) and passes only when every row has a real source, license and `uruguay_packaging_checked` yes. The normal check (without `--final`) still passes with warnings.

## Task 1: embeddings (`src/embeddings/embed.py`, `configs/stage6.yaml` section `embeddings`)

DINOv2 base (`facebook/dinov2-base`, Apache-2.0, 87 M parameters, 346 MB of weights, revision `f9e44c814b77203eaa57a6bdbbd535f21ede1415` pinned), through the `transformers` already installed (no new package). One 768-dimension vector per crop, L2-normalised (cosine similarity is then a dot product), CLS-token pooling (mean-pooled patch tokens are available and are compared on the catalog before one is kept). **Preprocessing:** the crop is padded to a square with white and resized to 224 px (bicubic), then normalised with the ImageNet statistics. It is not centre-cropped, because a bottle crop is tall and narrow and a centre crop would cut off the cap and the base; white padding matches the catalog backgrounds (the padding colour is a setting to revisit if the domain gap turns out to matter). The same function embeds catalog crops and shelf crops. Tests (`tests/test_embed.py`, 4, with a tiny randomly initialised DINOv2 so that no download is needed): padding, preprocessing shape and normalisation, unit length, determinism, batch independence, mean pooling.

**Speed on MPS** (`python scripts/embed_timing.py`, 70 real crops: the 30 catalog crops and 40 shelf crops; Apple M4 Pro, torch 2.14.1):

| Batch size | Seconds per batch | Per crop | MPS driver memory |
|---:|---:|---:|---:|
| 1 | 0.020 | 20.2 ms | 363 MB |
| 8 | 0.108 | 13.6 ms | 1,451 MB |
| 16 | 0.203 | 12.7 ms | 1,451 MB |
| 32 | 0.394 | 12.3 ms | 1,451 MB |

Model load 10.5 s once; all 70 crops in 0.8 s. **It is fast; nothing is too slow and nothing needed a workaround.** The whole catalog (30 originals plus about 180 augmented) and the 180 val crops embed in a few seconds.

## Task 2: catalog augmentation (`src/embeddings/augment.py`)

About 6 variants per catalog crop (`configs/stage6.yaml`, `catalog.augment`), each with its parameters recorded, from a seeded generator with a stable seed per photo (CRC32 of the catalog file name), so the preview and the real build are identical. Per variant: brightness (0.75 to 1.25), contrast (0.75 to 1.25), Gaussian blur (sigma 0.1 to 1.2), then rotation (up to 7 degrees) and zoom (0.85 to 1.05) about the centre, then a mild perspective (corner offsets up to 12% of the half-size). The empty corners are filled with the photo's own background colour after the lighting change (median of the four corners), so a darker variant has a darker background and no light frame. **Deliberately not used:** horizontal flip (it mirrors the label text, the evidence that separates variants) and hue or saturation changes (colour is a main cue: Coca-Cola red, Light silver, Fanta orange, Sprite green); a test checks that an asymmetric mark stays on its side and that a red bottle stays red. Tests: `tests/test_augment.py` (7): ranges and reproducibility, stable seeds, size preserved and same seed gives the same pixels, no mirroring, no colour drift, identity parameters, dark variant gets a dark background.

The zoom upper limit was lowered from 1.15 to 1.05 after the owner's review and the sheets regenerated (same seeds). The first version of the preview showed a visible light frame around darker variants (the fill was applied before the lighting change); that was fixed by the order described above and the sheets were regenerated.

## Task 3: augmentation preview (`python scripts/preview_catalog_augment.py`)

Six contact sheets in the ignored `data/review/catalog_augment/<product id>.jpg`: one row per original crop (5 rows), the original first, then its 6 variants, each captioned with its parameters. 180 variants previewed, nothing stored in any database. What I see when looking at them (for the owner to judge):
- Label text stays readable and mirrored text never appears; colours stay faithful (the silver of Light, the orange of Fanta, the green of Sprite, the red of Coca-Cola).
- **Zoom-in above about 1.1 sometimes clips the cap or the base** (for example some Fanta and Coca-Cola Zero variants). If you consider the cap and base important cues, a lower limit such as 1.05 is a one-line change.
- **Strong brightness with low contrast washes out the silver label of Light**, and blur near 1.0 softens small text such as "light" or "ZERO AZUCAR"; these look like plausible shelf conditions, but they are the strongest changes.
- Dark variants have a grey background; faint thin frame lines can remain at the edges from the interpolation.

## Task 4: catalog build (`python scripts/build_catalog.py`)

30 bottle crops (5 per product) and their 6 augmented variants each (same seeds as the preview, zoom up to 1.05) = **210 entries**, embedded with DINOv2 (CLS, 768 dimensions) and written to the ignored `data/catalog/embeddings.npz`. Loaded into Postgres + pgvector in the **own database `soda_catalog`** of the running container (created by the script; the Label Studio data and the `soda_poc` database are untouched; `pg8000`; cosine HNSW index; table `catalog_entries` with product, brand, size, material, source photo, source URL, license, packaging check, `is_augmented`, `aug_index`, `aug_params`, crop method, embedding). Every entry carries its source from `sources.csv` (all still `unknown` today; the final-report gate `check_catalog_photos.py --final` fails until the owner fills them in). The tests (`tests/test_catalog_db.py`) use a throwaway database that they create and drop (the list of databases afterwards shows only `postgres` and `soda_poc`).

**Which entries are in the database:** only the **30 originals**. The experiment below shows that the augmented catalog identified worse, and the rule was to keep it only if it identified better; `catalog.use_augmented: false` records the decision, and the augmented entries stay in `embeddings.npz` for experiments. A search in Postgres gave the same nearest product and similarity as the offline search for all 180 query crops (0 disagreements).

## Tasks 5 and 6: retrieval, unknown rules, query embeddings

`src/embeddings/identify.py`: cosine similarity of a crop to every catalog entry; one score per source photo (best over its entries, so augmented copies never count twice) and per product (best photo); features: top-1 product and similarity, margin to the best different product, k-NN vote (share of the 5 nearest photos from the top-1 product). Rules: top-1 similarity below a threshold is `unknown`; a low vote or a small margin is `review`; otherwise `automatic`. Thresholds are tuned by maximising a utility (correct automatic +1, confident wrong -5, a known crop rejected -1, an unknown rejected +1, review -0.1, review for an unknown +0.5; `configs/stage6.yaml`). The 180 labeled val crops were embedded from the original photo pixels with the same 10% margin as the catalog crops (`data/crops/val_sample/embeddings.npz`).

## Task 7: development numbers, leave one store out (plain catalog, CLS embeddings)

For each of the 6 stores the thresholds are tuned on the other 5 and applied to it; predictions are pooled; intervals resample whole stores (6 stores: wide). Accuracy-style numbers use the 86 crops labeled with a product and the 67 labeled `unknown`; `not_a_bottle` (15) and `cannot_tell` (12) are reported separately.

| Metric (%) | Value [95% CI by store] |
|---|---|
| Top-1, nearest product (86 known crops) | 53.5 [44.7, 61.8] |
| Top-3 | 90.7 [83.6, 95.5] |
| Brand accuracy (Coca-Cola, Sprite, Fanta) | 88.4 [81.7, 94.8] |
| Known crops answered automatically and correctly | 0.0 |
| Share of answers: automatic / review / unknown | 0.0 / 83.0 [77.4, 87.2] / 17.0 [12.8, 22.6] |
| Unknown crops sent to `unknown` or `review` | 100.0 (67 of 67) |
| Unknown crops sent strictly to `unknown` | 26.9 [19.0, 45.2] (18 of 67) |
| False known rate (unknown answered automatically) | 0.0 (0 of 67) |

**How to read this.** The rules were tuned to avoid confident wrong answers, and since only about half of the nearest products are right, the best policy under that utility is to answer nothing automatically: all 86 known crops end up in `review` or `unknown` (78 review, 8 unknown). The "100% unknown rejection" and the 0% false known rate are therefore **trivial** (everything goes to review); the informative number is that only 27% of the unknown crops are strictly rejected as `unknown`. A descriptive precision-against-coverage table (no tuning) shows no operating point: answering the 38 crops with a top-1 similarity of at least 0.80 is right for 50% of them, and 6 of those are unknown crops; at 0.90, 3 crops are answered. The margin between the best and the second best product has a median of 0.007 on known crops (90th percentile 0.020), so a margin rule has almost nothing to work with.

Per product (counts; a percentage only with at least 10 crops; **`coca_light` has 3 crops: no conclusion, counts only**; `coca_classic_returnable` has 9: low):

| Product | Crops | Nearest product correct | Answered automatically and correctly | Review | Rejected as unknown |
|---|---:|---:|---:|---:|---:|
| coca_classic | 20 | 14 of 20 (70%) | 0 | 19 | 1 |
| coca_classic_returnable (low) | 9 | 2 of 9 | 0 | 9 | 0 |
| coca_zero | 28 | 12 of 28 (43%) | 0 | 25 | 3 |
| coca_light (3 crops) | 3 | 0 of 3 | 0 | 3 | 0 |
| sprite | 10 | 8 of 10 (80%) | 0 | 7 | 3 |
| fanta | 16 | 10 of 16 (62%) | 0 | 15 | 1 |

Confusions by name (nearest product; rows are the truth): **`coca_zero` crops go to `coca_classic` 10 times of 28 (and 3 to `coca_classic_returnable`, 2 to `coca_light`)**; `coca_classic_returnable` crops go to `coca_classic` 7 of 9 (the returnable and the regular bottle look alike at shelf size); `coca_light` (3 crops) went to `coca_classic` 2 and `coca_zero` 1; `fanta` crops to `coca_zero` 3 and `coca_classic` 2; the full matrix is in the script output.

**What the system answers for the crops that are not targets:** the 15 `not_a_bottle` crops (cans, cartons, fragments, double-bottle boxes) are answered **`unknown` 14 times and `review` once, never with a product**: such boxes are far from every catalog photo, which is what we wanted. The 12 `cannot_tell` crops: `unknown` 8, `review` 4, never a product.

**Store s03 sensitivity (30 of the 67 unknown crops):** evaluating without s03 (37 unknown crops) the unknown rejection (unknown or review) stays 100%, strictly unknown rises from 26.9% to 35.1%, and top-1 on the known crops goes from 53.5% to 57.1%; if the thresholds are also never tuned on s03 the strict rejection is 27.0% and the top-1 57.1%. One store does not drive the conclusion.

## Task 8: experiments

**(a) Plain catalog (30 originals) against the augmented catalog (210 entries), paired bootstrap by store.** Augmented minus plain: top-1 **-12.8 points [-23.4, -4.3]**, brand accuracy **-10.5 [-16.7, -5.1]**, top-3 -1.2 [-7.6, +5.0], mean utility -0.07 (interval excludes zero), false known rate +3.0 [0.0, +5.7] (zero inside). The augmented catalog identifies clearly **worse**, so by the agreed rule it is **not kept** (the database holds the originals only). A likely reason: the darker, blurred, rotated copies sit closer to many shelf crops in general, pulling them towards whichever product has a copy that resembles the crop's look, regardless of the product (for example `coca_zero` crops going to `coca_classic`).

**CLS against mean pooling of the patch tokens:** top-1 53.5% (CLS) against 12.8% (mean), top-3 90.7% against 55.8%: mean pooling is far worse here (its vectors are all very similar), so CLS stays. **Addendum (6c prep):** the collapse of plain mean pooling is a padding effect (62% of the square is white padding); with the padding patches left out (`mean_content`) it reaches 47.7%, still below CLS: see `reports/stage_06c_prep.md`.

**(b) Domain gap (product photos only).** Cosine similarity: a shelf crop to its correct product has a median of **0.760** (quartiles 0.685 to 0.822), to the best wrong product **0.764** (0.689 to 0.821): **no separation**; the correct product is the nearest for 46 of 86 known crops. Between catalog photos the similarity to the same product's other photos is 0.954 (0.949 to 0.974) and to the best other product 0.948 (0.906 to 0.962): the clean photos are tightly clustered, and shelf crops sit far from all of them. Unknown shelf crops have a top-1 similarity of 0.634 (0.377 to 0.739), lower than known crops, which is why similarity is useful for rejection and not for choosing the variant. **The gap is large.** A diagnostic (`scripts/diagnose_domain_gap.py`, in memory only) shows it is not only resolution: the plain top-1 is similar for small and large crops (3 of 7 under 150 px tall, 15 of 29 for 150 to 300 px, 28 of 50 above 300 px), and adding the catalog photos degraded to shelf resolution did not help (top-1 -5.8 [-18.2, +2.0], brand -4.7 [-8.4, -1.1]). What differs is the pose, occlusion by neighbours, lighting and background, which clean frontal photos do not show.

**(c) SAM 2 mask crops:** not tried; the gap is not mainly the background around the bottle, and a mask would not add shelf-like views to the catalog.

## Gate and what to do next

The plan's gate (brand accuracy high, unknown handling measured, confusions named) is **not met**: brand accuracy is 88.4% [81.7, 94.8] (initial target 90%, development number, 86 crops), the nearest product is right 53.5% of the time, and the unknown rules cannot find an operating point that answers anything automatically with an acceptable error. The confusions are named above (Zero against Classic, returnable against regular Classic, Light too rare to say). The unknown handling is measured, and what works is rejecting boxes that are not bottles (14 of 15 as `unknown`).

**Proposal for the second step the plan foresaw (needs the owner's approval, new labeling round):** add a small number of **shelf crops from train photos only** (never val or test) to the catalog, tagged as shelf crops with their store, and measure again on the same 180 val crops with the same protocol. Suggested size: about 300 train crops in the same Label Studio crop tool (same labels, hotkeys 1 to 6, `u`, `n`, `c`), which at the owner's measured 3.4 seconds per crop is about 17 minutes of labeling; sampled stratified by store, with about 100 of them chosen among crops whose top-3 retrieval contains the rare products (`coca_light`, `coca_classic_returnable`, `sprite`) so that those products get shelf examples (a labeling aid for the catalog, not the yardstick; the val sample stays untouched and unsuggested). Expected result: shelf examples of each product close to the queries, which should lift the top-1 well above 53.5%; whether it reaches a usable operating point is to be measured, not assumed. Other options, not recommended first: metric-learning fine tuning of the embedding (needs the same labeled shelf crops), colour or text cues for Classic against Zero against Light (OCR cannot read small crops), higher-resolution photos (the demo set).

## How to verify (inside `soda-poc`)

```bash
python -m pytest -q
python scripts/embed_timing.py
python scripts/preview_catalog_augment.py
python scripts/check_catalog_photos.py          # 0 errors, 30 warnings (packaging unconfirmed)
python scripts/check_catalog_photos.py --final  # fails until sources.csv is filled in
python scripts/build_catalog.py                 # catalog embeddings + the soda_catalog database (originals only)
python scripts/evaluate_identification.py       # development numbers, leave one store out
python scripts/diagnose_domain_gap.py           # diagnostic, in memory only
git status --short                               # nothing under data/ or models/
```
