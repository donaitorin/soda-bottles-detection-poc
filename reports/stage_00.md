# Stage 0 report: definition and environment

**Status: waiting for the owner's decisions** (labeling guide cases, catalog details, torch version). Stage 1 has not started.

## What was done

| Task | Result |
|------|--------|
| `.gitignore` and folder skeleton | Done. Everything under `data/` and `models/` is ignored except `.gitkeep` and `data/splits/*.txt`; image, mask, array and checkpoint extensions are ignored everywhere (also uppercase); `volumes/`, `.env` and `.handoff/` are ignored. |
| Conda env `soda-poc` | Done (Python 3.11.17, conda-forge only, `environment.yml` + pinned `requirements.txt`). |
| `get_device()` | Done in `src/common/device.py` (cuda, then mps, then cpu), 4 tests. It also sets `PYTORCH_ENABLE_MPS_FALLBACK=1` before torch loads. |
| Verification script | Done: `scripts/verify_env.py` + `configs/stage0_verify.yaml`. Runs on a real photo. |
| `docker-compose.yml` | Done: `pgvector/pgvector:0.8.7-pg16` and `heartexlabs/label-studio:1.23.2`, data in ignored `volumes/`, photos mounted read-only into Label Studio. |
| Photo inventory | Done: `scripts/inventory_photos.py` (read-only). |
| Labeling guide | Drafted with 8 real examples; decisions pending. |
| Catalog variants | Drafted from a visual look at about 30 photos; decisions pending. |

## Gate checklist

| Gate item | Status | How to verify |
|-----------|--------|---------------|
| Guide resolves at least 5 real ambiguous examples from your photos | **Drafted, needs your decisions.** 8 examples (A to H) with a default for each. | Read `docs/labeling_guide.md`, "Real examples" and "Decisions I need". |
| Verification script runs on a real photo | **Done.** 63 detections at 0.5 on the test photo (62 `bottle`), annotated image in `data/test_images/verify_output.jpg`. | `conda activate soda-poc && python scripts/verify_env.py` |
| Device decision for local work recorded | **Pending your approval** (see below). | |
| `git status` does not show a photo copied into `data/raw/` | **Done.** 257 files in `data/raw/`, none tracked. | `git status --short` (empty) and `git ls-files data` (only `.gitkeep` files) |

## Device decision (MPS or CPU)

Faster R-CNN timing on the real test photo (881x1344), median of 5 runs after 2 warm-ups:

| Setup | MPS | CPU |
|-------|-----|-----|
| torch 2.8.0 / torchvision 0.23.0 (current pins) | **42.6 s** | 0.94 s |
| torch 2.14.1 / torchvision 0.29.1 (newest, throwaway env) | **0.16 s** | 0.94 s |

- With the current pins MPS is about 45 times *slower* than CPU. The profiler showed 98% of the time in one operation (`aten::nonzero`, about 5.7 s per call); the backbone alone was fast (0.056 s).
- With the newest pair MPS is about 6 times *faster* than CPU and returns the same 63 detections. It was tested in a throwaway env (already removed). `soda-poc` and `requirements.txt` were **not** changed.
- **Recommendation:** upgrade the pins to torch 2.14.1 / torchvision 0.29.1 and use MPS locally. If you prefer to keep 2.8.0, force the CPU (`device: cpu` in the config) and use Colab for Stages 2 and 5.
- Grounding DINO and SAM 2 get their own speed check in Stage 2. A new torch version also needs `pytest` and the verify script to be re-run.

## Photo inventory (257 files in `data/raw/`)

- Extensions: 253 `.jpg`, 4 `.png` (the PNGs are screenshots named `Captura de pantalla ...`, 308x376 px, not shelf photos). No HEIC.
- Long side: min 374 px, median 1440 px, max 4096 px. Most common size: 1184x1440 (136 photos). 251 portrait, 6 landscape.
- EXIF: orientation is normal for all 257, and none has a capture date or GPS tags. The metadata seems to have been stripped.
- Grouping for the Stage 1 split: file names carry a store id and a visit date, so grouping is possible. 253 photos match: 19 stores, 34 store-and-date visits, 5 stores with more than one visit date. Photos per store: min 2, median 10, max 48 (one store is 19% of the data). The 4 screenshots have no store.

## Problems found

1. The torch 2.8.0 / MPS slowness above.
2. The pretrained COCO detector boxes juice cartons as `bottle` (example D in the guide).
3. One store holds 48 of 257 photos, which will make a 70/15/15 split by store uneven. Stage 1 must decide how to handle it.
4. Many photos have grey wedges at the sides (perspective correction).
5. Returnable and non-returnable Coca-Cola bottles look very similar, and so do Classic and Zero at 1.5 L (see `docs/catalog_variants.md`).

## Open decisions for you

See the end of `docs/labeling_guide.md` and `docs/catalog_variants.md`, plus the torch pin above. Run `nbstripout --install` yourself when you add the first notebook.
