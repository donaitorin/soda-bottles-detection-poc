# Stage 6b report (in progress): embeddings, augmentation preview; stopped for the owner's review

> Status: tasks 1 to 3 of the approved Stage 6b plan are done (embedding module with its speed check, augmentation module, augmentation preview). The catalog has **not** been built and nothing has been stored in any database: the owner reviews the contact sheets first. Nothing here measures identification yet; later 6b numbers are development numbers on 180 crops from 6 stores, not accuracy.

## Decisions applied (owner)

- **Detector for the next stages: `frcnn_v2` last epoch (`epoch_12.pt`).** The other 11 epoch checkpoints of `frcnn_v2` were deleted; `best.pt` and `epoch_12.pt` stay in `models/stage5/frcnn_v2/`. **Freed 3.7 GB** (3,712,236 KB; 506 MB remain).
- **OCR:** Tesseract trial finished and the final engine recommendation is EasyOCR (details in `reports/stage_06b_prep.md`, section 9). The throwaway environment `ocr-trial` was removed (`conda env remove -n ocr-trial`; it is no longer in `conda env list`). `soda-poc` never had EasyOCR, python-bidi or pytesseract.
- **Plan additions** (`docs/PLAN.md` Stage 6): the catalog goes in its own database (for example `soda_catalog`) in the running Postgres container, never in the Label Studio database, tests use a throwaway database they create and drop, never `docker compose down -v`; shelf crops proposed for the catalog (experiment 8b) must come from train photos only and need a new labeling round (propose and wait); the system's answer for the 15 `not_a_bottle` crops is reported (ideally `unknown` or `review`); the unknown rejection rate is reported with and without store `s03` (30 of the 67 unknown crops); DINOv2 speed on MPS is measured first.
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

About 6 variants per catalog crop (`configs/stage6.yaml`, `catalog.augment`), each with its parameters recorded, from a seeded generator with a stable seed per photo (CRC32 of the catalog file name), so the preview and the real build are identical. Per variant: brightness (0.75 to 1.25), contrast (0.75 to 1.25), Gaussian blur (sigma 0.1 to 1.2), then rotation (up to 7 degrees) and zoom (0.85 to 1.15) about the centre, then a mild perspective (corner offsets up to 12% of the half-size). The empty corners are filled with the photo's own background colour after the lighting change (median of the four corners), so a darker variant has a darker background and no light frame. **Deliberately not used:** horizontal flip (it mirrors the label text, the evidence that separates variants) and hue or saturation changes (colour is a main cue: Coca-Cola red, Light silver, Fanta orange, Sprite green); a test checks that an asymmetric mark stays on its side and that a red bottle stays red. Tests: `tests/test_augment.py` (7): ranges and reproducibility, stable seeds, size preserved and same seed gives the same pixels, no mirroring, no colour drift, identity parameters, dark variant gets a dark background.

The first version of the preview showed a visible light frame around darker variants (the fill was applied before the lighting change); that was fixed by the order described above and the sheets were regenerated.

## Task 3: augmentation preview (`python scripts/preview_catalog_augment.py`)

Six contact sheets in the ignored `data/review/catalog_augment/<product id>.jpg`: one row per original crop (5 rows), the original first, then its 6 variants, each captioned with its parameters. 180 variants previewed, nothing stored in any database. What I see when looking at them (for the owner to judge):
- Label text stays readable and mirrored text never appears; colours stay faithful (the silver of Light, the orange of Fanta, the green of Sprite, the red of Coca-Cola).
- **Zoom-in above about 1.1 sometimes clips the cap or the base** (for example some Fanta and Coca-Cola Zero variants). If you consider the cap and base important cues, a lower limit such as 1.05 is a one-line change.
- **Strong brightness with low contrast washes out the silver label of Light**, and blur near 1.0 softens small text such as "light" or "ZERO AZUCAR"; these look like plausible shelf conditions, but they are the strongest changes.
- Dark variants have a grey background; faint thin frame lines can remain at the edges from the interpolation.

## Pending (after the owner's review of the sheets)

Task 4 catalog build (embeddings, database `soda_catalog`, entries tagged with their source), 5 retrieval and unknown rules, 6 query embeddings of the 180 labeled val crops, 7 leave-one-store-out development numbers (counts only for `coca_light`, `not_a_bottle` answers, with and without `s03`), 8 experiments (plain against augmented catalog, domain gap), 9 report.

## How to verify (inside `soda-poc`)

```bash
python -m pytest -q
python scripts/embed_timing.py
python scripts/preview_catalog_augment.py
python scripts/check_catalog_photos.py          # 0 errors, 30 warnings (packaging unconfirmed)
python scripts/check_catalog_photos.py --final  # fails until sources.csv is filled in
git status --short                               # nothing under data/ or models/
```
