# Stage 0 report: definition and environment

**Status: closed pending the owner's final review.** All decisions below were confirmed by the owner. Stage 1 has not started.

## What was done

| Task | Result |
|------|--------|
| `.gitignore` and folder skeleton | Done. Everything under `data/` and `models/` is ignored except `.gitkeep` and `data/splits/*.txt`; image, mask, array and checkpoint extensions are ignored everywhere (also uppercase); `volumes/`, `.env` and `.handoff/` are ignored. |
| Conda env `soda-poc` | Done (Python 3.11, conda-forge only, `environment.yml` + pinned `requirements.txt`). Pins upgraded to torch 2.14.1 / torchvision 0.29.1 after the MPS test below. |
| `get_device()` | Done in `src/common/device.py` (cuda, then mps, then cpu), 4 tests. It also sets `PYTORCH_ENABLE_MPS_FALLBACK=1` before torch loads. |
| Verification script | Done: `scripts/verify_env.py` + `configs/stage0_verify.yaml`. Runs on a real photo. |
| `docker-compose.yml` | Done: `pgvector/pgvector:0.8.7-pg16` and `heartexlabs/label-studio:1.23.2`, data in ignored `volumes/`, photos mounted read-only into Label Studio. |
| Photo inventory | Done: `scripts/inventory_photos.py` (read-only). |
| Labeling guide | **FINAL**, with 8 real examples. The ~20 px minimum height is provisional (Stage 2). |
| Catalog variants | **FINAL list of 10 product ids** (returnable Pepsi and Sprite added). Sizes and frequencies are rough. |

## Gate checklist

| Gate item | Status | How to verify |
|-----------|--------|---------------|
| Guide resolves at least 5 real ambiguous examples from your photos | **Done.** 8 examples (A to H), all decided by the owner. | Read `docs/labeling_guide.md`, "Real examples" and "Decisions I need". |
| Verification script runs on a real photo | **Done.** 63 detections at 0.5 on the test photo (62 `bottle`), annotated image in `data/test_images/verify_output.jpg`. | `conda activate soda-poc && python scripts/verify_env.py` |
| Device decision for local work recorded | **Done:** MPS locally (see below). | `python scripts/verify_env.py` |
| `git status` does not show a photo copied into `data/raw/` | **Done.** 257 files in `data/raw/`, none tracked. | `git status --short` (empty) and `git ls-files data` (only `.gitkeep` files) |

## Device decision (MPS or CPU)

**Decision: use MPS locally, with torch 2.14.1 / torchvision 0.29.1** (approved by the owner).

Faster R-CNN timing on the real test photo (881x1344), median of 5 runs after 2 warm-ups:

| Setup | MPS | CPU | Detections at 0.5 |
|-------|-----|-----|-------------------|
| torch 2.8.0 / torchvision 0.23.0 (old pins) | **42.6 s** | 0.94 s | 63 |
| torch 2.14.1 / torchvision 0.29.1, throwaway env | **0.16 s** | 0.94 s | 63 |
| torch 2.14.1 / torchvision 0.29.1, `soda-poc` (final) | **0.15 s** | 0.95 s | 63 (62 `bottle`) |

- With the old pins MPS was about 45 times *slower* than CPU: the profiler showed 98% of the time in one operation (`aten::nonzero`). With the new pair MPS is about 6 times *faster* and the result is identical.
- After the upgrade: `pytest` 4 passed, `pip check` clean, and `soda-poc` gives the same 63 detections as the throwaway env.
- Licenses: torchvision is BSD. torch's package metadata lists a composite of permissive licenses (Apache-2.0, BSD-2-Clause, BSD-3-Clause, Apache-2.0 with LLVM exception); its main license is BSD-3. All satisfy the project's permissive-license rule.
- The config option `device: auto | cpu | mps` lets any script force the CPU.
- Grounding DINO and SAM 2 get their own speed check in Stage 2.

## Photo inventory (257 files in `data/raw/`)

- Extensions: 253 `.jpg`, 4 `.png` (the PNGs are screenshots named `Captura de pantalla ...`, 308x376 px, not shelf photos). No HEIC.
- Long side: min 374 px, median 1440 px, max 4096 px. Most common size: 1184x1440 (136 photos). 251 portrait, 6 landscape.
- EXIF: orientation is normal for all 257, and none has a capture date or GPS tags. The metadata seems to have been stripped.
- Grouping for the Stage 1 split: file names carry a store id and a visit date, so grouping is possible. 253 photos match: 19 stores, 34 store-and-date visits, 5 stores with more than one visit date. Photos per store: min 2, median 10, max 48 (one store is 19% of the data). The 4 screenshots have no store.

## Problems found

1. The torch 2.8.0 / MPS slowness above (solved by the upgrade).
2. The pretrained COCO detector boxes juice cartons as `bottle` (example D in the guide).
3. One store holds 48 of 257 photos (19%). Decision: it goes to train; the split is strictly by store (see Stage 1 in `docs/PLAN.md`).
4. Many photos have grey wedges at the sides (perspective correction).
5. Returnable and non-returnable Coca-Cola bottles look very similar, and so do Classic and Zero at 1.5 L (see `docs/catalog_variants.md`).

## Decisions taken in this stage

- 10 target products (returnable Sprite and Pepsi added); material and size are catalog metadata; glass returnables in crates count as returnable products.
- Low-data products are flagged after Stage 3, not dropped.
- Stage 7 keywords include "ZERO AZUCARES".
- File names are never committed; Stage 1 uses anonymous ids `<store_id>_<short hash>`.
- Stage 1 split: strictly by store, the 48-photo store in train, screenshots excluded, photos under 800 px flagged.
- Working rule: the assistant does not run `git add`, `git commit` or `git push`; the owner commits.

Still to do by the owner: run `nbstripout --install` when the first notebook is added, and create the Label Studio admin account.
