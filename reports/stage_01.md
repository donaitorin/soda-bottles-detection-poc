# Stage 1 report: data inventory and dataset split

**Status: ready for the owner's review.** Nothing in `data/raw/` was modified, moved or deleted. Test stays frozen until Stage 8.

## What was done

| Task | Result |
|------|--------|
| Anonymous ids | 19 store codes (`s01`..`s19`) plus `sx` for the 4 screenshots; 257 photo ids `<store code>_<first 8 hex of sha256 of the file name>`. The two mappings live in the ignored `data/id_maps/`. Names are NFC-normalized before hashing so the id does not depend on how macOS stores accents. |
| Image loading | `src/data/images.py: load_image()` applies the EXIF orientation in memory and converts to RGB. Every photo already has orientation 1, so this is a safety net (tested with a synthetic orientation-6 image). |
| Metadata | `data/metadata.csv` (ignored): size, store code, visit date, flags. |
| Duplicates | None found (see below). |
| Exclusion list | `data/splits/excluded.txt`: the 4 screenshots (`sx_...`), nothing else. |
| Split | `data/splits/{train,val,test}.txt`: photo ids only, strictly by store. |
| Checks | `scripts/check_splits.py` (gate) and `scripts/check_no_leaks.py` (privacy), plus a pre-commit hook script the owner installs. |

## Inventory

257 files in `data/raw/`: 253 `.jpg` and 4 `.png` (screenshots, excluded). No HEIC. Long side: min 374 px, median 1440 px, max 4096 px; only the 4 screenshots are under 800 px. EXIF orientation is normal on every photo and none has a capture date or GPS tags. 19 stores, 34 store-and-date visits; 253 usable photos.

## Split (seed 42)

| Split | Photos | Share | Stores | Visits | Long side px (min / median / max) |
|-------|-------:|------:|-------:|-------:|-----------------------------------|
| train | 174 | 68.8% | 7 | 18 | 1301 / 1440 / 3000 |
| val | 41 | 16.2% | 6 | 9 | 1356 / 1440 / 4096 |
| test | 38 | 15.0% | 6 | 7 | 1356 / 1440 / 4080 |

Targets were 70 / 15 / 15; the largest deviation is 1.23 points (tolerance 5).

**Store assignment (store codes only):**
- train: `s04` `s05` `s07` `s12` `s17` `s18` `s19` (`s05`, the 48-photo store, was forced into train)
- val: `s01` `s03` `s06` `s11` `s15` `s16`
- test: `s02` `s08` `s09` `s10` `s13` `s14`

**How it was chosen:** whole stores are assigned, never single photos, so the same shelf (also on two different days) cannot be in train and test. The search tried 10,000 random store orders with seed 42 and found 6,830 distinct assignments within the tolerance. Among them it kept the best by this order: (1) the largest value of the smaller of the val and test store counts (test is the final measure, so it must not get few stores), (2) the most stores in val and test, (3) the most visits there, (4) the smallest deviation. The winner has 6 stores in val and 6 in test (an earlier ranking by total stores gave 7 and 5). 7 and 7 is impossible within the tolerance: train would fall below 65%. The result is deterministic: rerunning gives identical files.

## Duplicates and retakes

The detector used a perceptual hash (pHash, 256 bits) and flags pairs at 12 bits or fewer, plus exact file copies (SHA-256). It found **no exact copies and no near duplicates**: no pair is closer than 25 bits. On synthetic variants the hash catches resizes and recompression (2 bits or fewer) and a +20% brightness change (12 bits or fewer), but not crops (a 5% crop already gives 84+ bits). Retakes of the same shelf are not reliably detected by pHash; as a lower bound, 8 pairs up to 64 bits exist, all inside a single store (the closest, 36 bits, is the same shelf photographed twice from nearly the same spot). They cannot leak between splits because the split is by store; the better measure of diversity is the number of stores and visits per split, given in the table above.

## Privacy

Committed files contain only store codes and photo ids. The leak check searches every tracked, modified or untracked non-ignored file (and, on request, the handoff file) for the real store ids, store names and addresses, uploader names and file names parsed from `data/raw/` and `data/test_images/` (312 strings), without printing any match. A second, mapping-free check always runs, also in a fresh clone and in `--staged` mode: anything shaped like a raw photo file name (`_dd.mm.yyyy_`, `-PHOTO-` plus digits, `Captura de pantalla` plus a date) is flagged; synthetic test fixtures opt out with the marker `leak-check: allow-synthetic` on the same line. Result: 0 problems. The pre-commit hook (`scripts/hooks/`) was installed by the owner, so every commit runs the check.

## Problems and limitations

1. **Small val and test.** 41 and 38 photos, from 6 stores each. Metrics will be noisy; Stages 4, 5, 6 and 8 report a bootstrap confidence interval resampled by store (in the plan). With 6 stores in test, that interval will still be wide.
2. **Product coverage is unknown yet.** The per-photo product field is only added in Stage 3; after Stage 3 the plan checks that val and test contain every target product and some other bottles, and the split is rebalanced before Stage 4 if not.
3. **One store is 19% of the data** (48 of 253 photos) and sits in train.
4. **The Stage 0 test photo** is from a store that is not in `data/raw/`, so it has no id and is in no split. It is only used for the environment check.
5. Hash threshold of 12 bits: there is nothing to review because no pair is flagged.

## How to verify the gate (inside `soda-poc`)

```bash
python scripts/check_splits.py       # per-split table and "OK": no store or photo in two splits, ids only
python scripts/check_no_leaks.py     # leak check, expects 0 problems
git status --short                   # only the files listed in the handoff; nothing under data/ except splits/*.txt
python -m pytest -q                  # all tests
```
