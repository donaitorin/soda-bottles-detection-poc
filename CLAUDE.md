# soda-bottles-detection-poc

Learning POC: detect soda bottles in store shelf photos and identify which ones are one of the 8 target products, and send every other bottle to `unknown`. The full plan, with its 10 stages and their validation gates, is in `docs/PLAN.md`. Read it before any task.

## How we work

- We work on **one plan stage at a time**. Do not start a new stage until the previous one has passed its gate and I have confirmed it.
- Before writing code for a stage, propose the tasks and wait for my approval.
- When a stage is done, write a summary in `reports/stage_XX.md` (what was done, results, how I can verify the gate, problems found) and wait for my confirmation.
- I am learning computer vision. When you implement something non-obvious (coordinate conversions, detector settings, thresholds), add a short comment in the code and explain it in the summary.
- Every time we stop and wait for approval, write a handoff file at `.handoff/latest.md` (overwritten each time, gitignored, never committed) so the user can pass it to the project coordinator, who has no access to the repo. Contents in order: status; commits since the last handoff; verification commands with trimmed output; diffs of changed docs/configs/CLAUDE.md/.gitignore/environment.yml/requirements.txt/docker-compose.yml (code files: diff if under ~200 lines, else a summary and path); numbered decisions or questions. Keep the chat reply short and point to the file.
- Small commits, one per task, with clear messages. Do not push: I do that myself.

## Technical constraints

- **Do not use YOLO or Ultralytics.** The detector is torchvision's Faster R-CNN (`fasterrcnn_resnet50_fpn_v2`).
- Pre-labeling: Grounding DINO (`IDEA-Research/grounding-dino-base` via `transformers`) and SAM 2. Manual validation in Label Studio.
- Only dependencies with permissive licenses (MIT, BSD, Apache 2.0). Ask me before adding anything else.
- Python 3.11 in a conda environment named `soda-poc` (conda-forge only), defined in `environment.yml`. Pinned dependencies in `requirements.txt`, installed through the pip section of `environment.yml` (PyTorch has no official conda packages, so torch and torchvision come from pip). No `.venv`, no pyenv.
- **Never run anything in conda's `base` env.** Every script and test runs inside `soda-poc` (`conda activate soda-poc`, or `conda run -n soda-poc ...`).
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
- **Cans are out of scope**: there is no `can` class and cans are not labeled.
- **Target products (8 ids):** `coca_classic`, `coca_classic_returnable`, `coca_zero`, `coca_zero_returnable`, `coca_light`, `sprite`, `sprite_zero`, `pepsi`. Details in `docs/catalog_variants.md`. Size is secondary metadata, not a hard success criterion.
- **Any bottle that is not one of these 8 products ends up as `unknown`** (other brands, water, juice). The detector still labels all of them as `bottle`; the decision is made after the embeddings (see "Unknown handling" in `docs/PLAN.md`).
- The test split is not used to tune anything before stage 8.
- Photos are split by store or shelf, never by individual photo.
- Coordinates: torchvision uses xyxy boxes in pixels; Label Studio uses x, y, width, height as percentages (0 to 100). Every conversion goes through functions with tests.

## Commands

One-time setup (all inside the `soda-poc` env, never in `base`):

```bash
conda env create -f environment.yml                       # first setup (only the file applies `nodefaults`)
conda env update -n soda-poc -f environment.yml --prune   # after changing environment.yml / requirements.txt
conda activate soda-poc
nbstripout --install                                  # one-time per clone: strips notebook outputs on commit
```

More commands are added as they are created.
