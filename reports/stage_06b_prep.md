# Stage 6 preparation report: 6 target products, catalog photos, OCR engine comparison

> No identification result yet. This report records the change of scope (6 products), the state of the catalog photos and the OCR engine comparison. Numbers about detector agreement elsewhere are agreement with Grounding DINO pseudo-labels, not accuracy; nothing here measures accuracy.

## 1. `frcnn_v2` could not be evaluated: there is no run on this machine

The owner ran the second detector run, but nothing from it exists in this repository: `data/logs/stage5/console_frcnn_v2.log`, `data/logs/stage5/frcnn_v2/` and `models/stage5/frcnn_v2/` do not exist, no `train_detector` process is running, and a search of the disk for `console_frcnn_v2*` and a `frcnn_v2` folder found nothing (only the `frcnn_v1` outputs exist). So I could not read a log or evaluate the last epoch. The evaluation is ready to run as soon as `models/stage5/frcnn_v2/epoch_12.pt` exists (the commands are in the handoff); `configs/stage4.yaml` already lists the v2 runs and the paired comparisons against the `frcnn_v1` last epoch and the COCO baseline. Until then the detector for the pipeline is still undecided between `best.pt` of `frcnn_v1` and v2.

## 2. Target list reduced to 6 products (owner decision)

`coca_classic`, `coca_classic_returnable` (the only returnable kept), `coca_zero`, `coca_light`, `sprite`, `fanta` (new, its own brand). Everything else is `unknown`: Pepsi, Sprite Zero and the returnable versions of Zero, Sprite and Fanta. The removed ids are out of scope, not "low data". Updated: `CLAUDE.md`, `docs/PLAN.md` (goal and scope, Stage 0 item 3, Stage 6, Unknown handling, Stage 7 keywords, metrics table), `docs/catalog_variants.md` (rewritten, with the new look-alike risks for the open set), `configs/stage6.yaml` (products, brands), `configs/label_studio/crop_project.xml` (6 ids plus `unknown`, `not_a_bottle`, `cannot_tell`; hotkeys 1 to 6, `u`, `n`, `c`), and the Stage 3 photo project config `bottle_project.xml` for consistency. Earlier reports keep their original (10-product) wording as history. No val crop had been labeled, and the crop sample does not depend on the label set, so the 180 crop tasks were **not** rebuilt; only the project's labeling configuration changes.

## 3. Catalog photos

The owner's 30 photos (5 per product) were renamed to `<product id>_NN.<ext>` by `scripts/normalize_catalog_photos.py` (sorted by original name, NN from 01, lowercase extension, nothing overwritten); the five empty folders of removed products (`coca_zero_returnable`, `pepsi`, `pepsi_returnable`, `sprite_returnable`, `sprite_zero`) were removed. The old to new table is in the handoff. `sources.csv` has 30 rows (`source_url` and `license` `unknown`, `uruguay_packaging_checked` `no`, the original name in `notes`). `check_catalog_photos.py`: 30 photos, **0 errors**, 30 warnings (packaging not confirmed). The photos are ignored by git (`git check-ignore` confirms the folder, `sources.csv` and the extension rules).

**Bottle crops** (`scripts/crop_catalog_photos.py`): frozen Grounding DINO (prompt `bottle.`, same thresholds as the shelf pre-labels, no Stage 2 filters because the "oversized" filter would drop a bottle that fills the photo), same 10% margin. Result: 29 photos gave exactly one box; **1 photo gave two boxes** (`fanta/fanta_01.jpg`: the bottle and a second box on its base). After looking at it, the box around both was used (an explicit `overrides` entry in `configs/stage6.yaml`, because it is clearly one bottle); the default rule for several boxes is the largest. **No photo gave zero boxes.** Crops are in the ignored `data/catalog/crops/` and a contact sheet of all 30 is in `data/review/catalog_crops/contact_sheet.jpg`. Looking at the sheet: the crops are clean single bottles on white; two of the photos (`coca_classic_01`, `coca_zero_01`) show a football-themed design, which looks like a limited edition: worth checking against the Uruguayan packaging.

## 4. Catalog augmentation settings (config only; built in 6b)

About 5 or 6 per photo on the bottle crop: small rotations (up to 7 degrees), mild perspective (0.12), scale 0.85 to 1.15, brightness and contrast (0.25), slight blur (sigma 0.1 to 1.2). **No horizontal flip** (it mirrors the label text, which is the evidence that separates variants) and **no hue or saturation change** (colour separates the products: Coca-Cola red, Light silver, Fanta orange, Sprite green). These differ from the Stage 5 detector augmentations on purpose; the reason is written in `configs/stage6.yaml`.

## 5. Two pipeline variants (plan)

A: embeddings only (catalog similarity plus unknown rules). B: embeddings plus OCR evidence. Same detector, catalog and val sample; thresholds calibrated separately; compared on the val sample as development numbers (leave-one-store-out) and once on the demo set as real numbers: top-1, per-product confusion matrices, unknown rejection rate, false known rate, automatic / review / unknown shares, paired bootstrap of the difference by store, the pairs `coca_classic` / `coca_zero` / `coca_light` and `coca_classic` / `coca_classic_returnable` reported separately, the share of crops where OCR reads something useful, and the time per photo of each variant. The Stage 9 API gets a request option (`use_ocr`) and the response names the variant. Written into `docs/PLAN.md` Stages 7 to 9 and the metrics table.

## 6. OCR engine comparison (nothing installed; facts from pip dry runs and package metadata)

Method: `pip install --dry-run` against the `soda-poc` environment (what would be added, with download sizes from the package files), package metadata for licences, no installation, no model download. Speed and quality are **not measured**, because that needs an installation.

| | EasyOCR | PaddleOCR | Tesseract |
|---|---|---|---|
| In the environment now | **No** (not installed; only PyTorch and torchvision are) | No | No (no `tesseract` binary on this Mac) |
| Licence of the engine | Apache-2.0 | Apache-2.0 (paddleocr, paddlepaddle, paddlex) | Apache-2.0 (engine and `pytesseract`) |
| What pip would add | 10 packages, about 63 MB (opencv-python-headless 46 MB, scikit-image 11.5 MB, shapely, python-bidi, pyclipper...) | `paddlepaddle` 100 MB plus about 50 packages, about 98 MB more (opencv-contrib 61 MB, pandas 10.6 MB, modelscope, aiohttp, pydantic...), roughly 200 MB | 1 package (`pytesseract`) plus the system binary from Homebrew |
| Models | downloaded at first use (detector and a Latin-script recogniser, about 100 MB in total from memory, to confirm) | downloaded at first use (tens of MB) | language data files (`eng`, `spa`), tens of MB |
| Licences that break our rule (permissive only; MPL only for unmodified) | `python-bidi` is **LGPL** | `python-bidi` is **LGPL** and `crc32c` is **LGPL-2.1+** | none |
| Apple Silicon | wheels for arm64; runs on PyTorch, which already uses MPS here; MPS use by EasyOCR is reported but **unverified** | `paddlepaddle` 3.3.1 has a macOS arm64 wheel, **CPU only** (no Metal backend); runs unverified | native arm64 binary, CPU |
| Expected fit for bottle labels | scene-text recogniser, handles curved and stylised text reasonably; Spanish via the Latin model | usually the strongest on scene text, but the heaviest and the most dependencies | made for clean document text; weakest on logos and curved label text |

**Recommendation: try EasyOCR first** (lightest addition to this environment, PyTorch-based, Apache-2.0), with Tesseract as the lightweight fallback and PaddleOCR only if EasyOCR reads too little. The catch is the **LGPL `python-bidi`** package that EasyOCR (and PaddleOCR) pull in: it is a dynamic dependency used for right-to-left scripts that we do not need, but it falls outside the licence rule, so it needs an explicit decision. A cheap, label-free test exists: the 30 catalog crops carry known text ("ZERO AZUCARES", "LIGHT", "RETORNABLE", "SPRITE", "FANTA", "COCA-COLA"), so a trial can measure how often each engine reads each keyword and the seconds per crop (CPU and MPS) before any val label is needed. Proposed, pending approval: run that trial in a throwaway environment outside `soda-poc`.

## Limits

- The `frcnn_v2` evaluation is pending (section 1).
- The OCR comparison has no speed or accuracy numbers.
- The Uruguayan packaging of the catalog photos and their sources are still to be filled in by the owner.

## How to verify (inside `soda-poc`)

```bash
python -m pytest -q
python scripts/check_catalog_photos.py          # 30 photos, 0 errors, 30 warnings
python scripts/crop_catalog_photos.py           # re-creates the crops and the contact sheet
git check-ignore -v data/catalog/product_photos/fanta/fanta_01.jpg data/catalog/product_photos/sources.csv
git status --short                               # nothing under data/ except splits/*.txt
```
