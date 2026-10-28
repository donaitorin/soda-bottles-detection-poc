# Stage 7, tasks 1 to 4: OCR dependencies, reader, evidence words, reading rate and speed

> Tasks 5 to 8 (evidence rules, A against B, demo flag, final report) are **not started**; they wait for the owner's decision below. All readings are on the **253 owner-labeled TRAIN shelf entries** (the val crops were not read; the only val use is photo timing, with no label or answer looked at). Counts, not percentages, for small groups. Nothing here is an accuracy number.

## 1. Install (task 1)
- `requirements-ocr.txt`: easyocr 1.7.2, rapidfuzz 3.14.6, opencv-python-headless 5.0.0.93, pyclipper 1.4.0, python-bidi 0.6.12, scikit-image 0.26.0, shapely 2.2.0, imageio 2.38.1, lazy-loader 0.6, ninja 1.13.2, tifffile 2026.3.3 (licences as in the proposal; **python-bidi** and the **FFmpeg libraries inside the opencv wheel** are LGPL: owner-approved exceptions, unmodified, isolated in this file).
- **Dry run first** (`pip install --dry-run -r requirements.txt -r requirements-ocr.txt`): only these 11 packages would be added; **no package of `requirements.txt` changes** (torch 2.14.1 and torchvision 0.29.1 untouched). After the install a before/after `pip freeze` comparison shows **no changed or removed line**, and `pip check` reports "No broken requirements found".
- Weights: `models/ocr/` (ignored; `craft_mlt_25k.pth`, `latin_g2.pth`, sha256 in `model_card.json`), copied from the trial download.

## 2. Reader and evidence (tasks 2 and 3)
- `src/ocr/reader.py` imports EasyOCR only when a reader is built; `OcrUnavailable` carries the one message "Variant B (use_ocr) needs the OCR packages: run `pip install -r requirements-ocr.txt` ...". `src/ocr/evidence.py`: whole words only, accents/case removed, fuzzy (rapidfuzz ratio >= 0.85) for the 5+ letter keywords, ZERO exact or glued to AZUCAR(ES), COCA-COLA as two adjacent words or one.
- **Tests (186 pass: 177 before + 9 new)**: FANTASIA is not FANTA; glued ZEROAZUCAR; fuzzy limits; and **with `easyocr`/`rapidfuzz` made un-importable**, `use_ocr` (the reader and `keywords_read`) raises that message **and** the variant A modules (identify, classifier, answers, demo, experiments) import and decide an example normally. Nothing in variant A imports `src.ocr`.

## 3. Reading rate by crop height, TRAIN (task 4)
Crops are the same 10% margin crops as for the classifier, upscaled to at least 900 px tall. "consistent" = share of the known-labeled crops with a keyword read that show no decisive word (SPRITE, FANTA, LIGHT, ZERO, RETORNABLE) their label does not allow.

| min height | crops | any text | any keyword | known + keyword | wrong word | unknown + keyword |
|---|---|---|---|---|---|---|
| 0 | 253 | 194 | 18 | 17 | 0 | 1 / 78 |
| 200 | 235 | 186 | 17 | 16 | 0 | 1 / 71 |
| 250 | 183 | 153 | 16 | 15 | 0 | 1 / 53 |
| 300 | 95 | 80 | 10 | 9 | 0 | 1 / 28 |
| 400 | 28 | 21 | 3 | 3 | 0 | 0 / 6 |
| 500 | 17 | 13 | 2 | 2 | 0 | 0 / 2 |

**Minimum height by the rule declared in `configs/stage7.yaml` before any number: 150 px** (the lowest value where >= 5 crops have a keyword and >= 90% are consistent). Be careful with this number: it is mechanical, because **no wrong word was read at any height** (0 wrong), so nothing ever pushes the minimum up. Cost matters more (section 5): reading every crop of 150 px or more costs about 2 s per photo, reading 300 px or more about 0.3 s.

**The decisive word read, by minimum height (read / crops of the product at least that tall):**

| product | word | all | >= 250 | >= 300 | >= 400 | >= 500 |
|---|---|---|---|---|---|---|
| coca_light | LIGHT | 0/17 | 0/12 | 0/5 | 0/1 | 0/0 |
| coca_zero | ZERO | 0/21 | 0/13 | 0/5 | 0/2 | 0/1 |
| coca_classic_returnable | RETORNABLE | 0/38 | 0/26 | 0/11 | 0/5 | 0/3 |
| sprite | SPRITE | 13/28 | 11/23 | 8/15 | 3/5 | 2/3 |
| fanta | FANTA | 4/10 | 4/9 | 1/5 | 0/0 | 0/0 |

Reading it:
- **The Coca-Cola variant words are never read** (0 of 76 crops of light, zero and returnable, at any height, including the few tall ones). So OCR **cannot fix "zero taken for light"**, the weak spot with the most errors. The Coca-Cola script logo is not read either (COCA, COLA: not read on any crop).
- **SPRITE is read often** (13/28; 11/23 from 250 px). **FANTA is read on 4/10** crops, but note how: only **1 is the exact word FANTA**; the other 3 are `FANT` (2) and `ANTA` (1), accepted by the fuzzy rule (4-letter fragment against the 5-letter keyword, ratio 0.889). If the fuzzy rule is limited to tokens of 5 or more letters, FANTA falls to 1/10 on TRAIN.
- Wrongly read words: **0**. The only keyword on an `unknown` crop is one `SPRITE` (in the unknown pool: possibly a Sprite Zero, which the rules would correctly send to unknown anyway).

## 4. What the OCR reads on TRAIN, per product (owner question about "sin azucar")
Most frequent normalised words (crops >= 150 px) are almost all noise (`A`, `C`, `GA`, `6`) for the Coca-Cola products. **Watch words** (AZUCAR, SIN, SUGAR, FREE, CERO, ZERO, LIGHT, CALORIAS, DIET, RETORNABLE(S), COCA, COLA): **not read on any crop**. In particular **no `coca_light` crop shows any "sin azucar" style text** (16 of 17 have some text, but the pieces of 4 or more letters are 7 meaningless fragments such as `CUGCLA`, `LIVGOLA`; 0 contain SIN/AZUCAR/SUGAR), so on this data that text would not confuse zero and light. `coca_zero`: only 6 of 21 crops have any text at all, and the one piece of 4+ letters is `{GGCLLO`.

**Extra words by the declared rule** (>= 3 crops of one label, 0 of any other, 4+ letters, above the minimum height): only **`BILZ` (6 crops, all `unknown`) and `PEPSI` (4 crops, all `unknown`)**. They are brands that must be `unknown`, not evidence for any of the 6 products, so they could only be used as a "push to unknown" word (whole word; noisy variants such as `EPSI`, `FERSI`, `KEM` are not words). **Not added**: it is the owner's call (question 3).

## 5. Seconds per photo on MPS (10 val photos, timing only, models warmed up)
Photos have a median of 58.5 boxes [12, 122].

| variant | seconds per photo, median [min, max] |
|---|---|
| A: detector + embeddings + classifier | 0.90 [0.39, 1.49] (detector 0.27, embeddings + classifier 0.60) |
| B, reading crops >= 150 px (34.5 crops per photo) | OCR adds 2.02 [0.37, 4.14] -> total 2.96 [0.87, 5.40] |
| B, reading crops >= 300 px (6 per photo) | OCR adds 0.29 [0.00, 2.92] -> total 1.28 [0.50, 4.02] |
| B, reading crops >= 400 px (0 per photo at the median) | OCR adds 0.00 [0.00, 2.14] -> total 0.96 [0.50, 3.23] |

About 0.06 s per crop read (253 crops: mean 0.057 s). On this data most shelf crops are shorter than 300 px, so the height minimum is also the main speed lever.

## 6. Recommendation on tasks 5 to 8
The picture is clearer than the proposal expected:
- **Zero taken for light cannot be addressed by OCR on this data**: ZERO and LIGHT are never read, at any height (0/38 on TRAIN).
- **Sprite and fanta rejected as unknown (5 + 5 val crops, median heights 251 and 286 px)** are the only place OCR can act: on TRAIN, SPRITE is read on about half the crops from 250 px (11/23) and FANTA on 4/9 (1 exact word). So a realistic ceiling is **about 3 to 5 fixes out of 14 target crops**; the keep-rule asks for at least +3 net, so passing is possible but not likely, and the paired interval will include zero.
- It is **cheap** (about 0.3 s per photo with a 300 px minimum) and the owner wants B to stay runnable for the demo set anyway.

My recommendation: **run tasks 5 to 8, kept small**: the approved rules unchanged, but expect the SPRITE/FANTA rescue, the contradiction-to-review and the not-a-target rules to be the only ones that can fire on val; the LIGHT/ZERO/RETORNABLE boost stays implemented and tested (it cannot be tuned on data where the words are never read, so its weight is fixed to the value the tuning grid picks by ties, i.e. 0 on val, and it will be measured on the demo set). The report would then state plainly that OCR leaves the zero/light confusion untouched. If you prefer to stop here, the OCR module stays in the repo as an option and the Stage 7 report would say "OCR did not help" on the evidence above.

## 7. Questions
See the handoff: (1) minimum height 150 by the declared rule or 300 for speed; (2) fuzzy rule for 4-letter fragments of FANTA; (3) BILZ and PEPSI as a push-to-unknown word; (4) run tasks 5 to 8 as described, or stop.

## 8. How to verify
`pip check`; `python -m pytest -q` (186 pass); `python scripts/ocr_reading_rate.py` (uses the cached readings; `--reread` reads again, 253 crops in about 15 s); `python scripts/ocr_timing.py`.
