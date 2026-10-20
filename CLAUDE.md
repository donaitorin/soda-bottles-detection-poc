# soda-bottles-detection-poc

Learning POC: detect soda bottles in store shelf photos and identify which ones are Coca-Cola and which variant. The full plan, with its 10 stages and their validation gates, is in `docs/PLAN.md`. Read it before any task.

## How we work

- We work on **one plan stage at a time**. Do not start a new stage until the previous one has passed its gate and I have confirmed it.
- Before writing code for a stage, propose the tasks and wait for my approval.
- When a stage is done, write a summary in `reports/stage_XX.md` (what was done, results, how I can verify the gate, problems found) and wait for my confirmation.
- I am learning computer vision. When you implement something non-obvious (coordinate conversions, detector settings, thresholds), add a short comment in the code and explain it in the summary.
- Small commits, one per task, with clear messages. Do not push: I do that myself.

## Technical constraints

- **Do not use YOLO or Ultralytics.** The detector is torchvision's Faster R-CNN (`fasterrcnn_resnet50_fpn_v2`).
- Pre-labeling: Grounding DINO (`IDEA-Research/grounding-dino-base` via `transformers`) and SAM 2. Manual validation in Label Studio.
- Only dependencies with permissive licenses (MIT, BSD, Apache 2.0). Ask me before adding anything else.
- Python 3.11, virtual environment in `.venv`, pinned dependencies in `requirements.txt`.
- Fixed random seeds for anything with randomness.
- Configuration lives in `configs/*.yaml`, not hardcoded in scripts. Scripts run from the CLI.
- Code, comments, file names, docs and commit messages in English.

## Environment: Mac

- There is no CUDA on this Mac. Use a `get_device()` function (in `src/common/device.py`) that picks `cuda`, then `mps`, then `cpu`, and use it across the whole project.
- Set `PYTORCH_ENABLE_MPS_FALLBACK=1` for operations MPS does not support.
- If something does not run on MPS or is too slow (for example, full training or pre-labeling every photo), tell me and propose running it on Colab instead of looking for long workarounds.

## Data

- **Never commit photos or labeling files.** Everything under `data/` is gitignored except `.gitkeep` files and `data/splits/*.txt` (lists of file names only). That includes the ~300 shelf photos in `data/raw/`, pre-labels, Label Studio exports, COCO annotations, crops, masks and catalog images. `models/` (checkpoints and downloaded weights) is also ignored.
- Before every commit, check `git status` and make sure no image or annotation file is staged. If the `.gitignore` is missing something, fix it before committing.
- `data/raw/` is never modified. Everything derived goes into other folders.
- The detector has a single generic `bottle` class for soda bottles, regardless of brand. Brand and variant are resolved later with embeddings, never in the detector labels.
- The test split is not used to tune anything before stage 8.
- Photos are split by store or shelf, never by individual photo.
- Coordinates: torchvision uses xyxy boxes in pixels; Label Studio uses x, y, width, height as percentages (0 to 100). Every conversion goes through functions with tests.

## Commands

(Filled in as they are added.)
