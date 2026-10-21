# Labeling guide: the `bottle` class

**Status: FINAL** (decisions confirmed by the project owner). The ~20 px minimum height is provisional and is revisited in Stage 2.

## The rule

There is **one class: `bottle`**. Label every soda-style bottle the same way, whatever the brand, flavor, size or contents. Brand and product are *never* part of the detector label; embeddings resolve them in Stage 6 (anything that is not one of the 8 target products becomes `unknown` there).

- The box is tight around the **visible** part of the bottle (cap to base), never the estimated part hidden behind another object.
- One box per bottle.
- Cans are out of scope: they are not labeled and there is no `can` class.
- Image coordinates always refer to the photo after fixing EXIF orientation (Stage 1).

## Ambiguous cases

| # | Case | Rule | Status |
|---|------|-----------------------|--------|
| 1 | Back-row bottle only half visible | Label if about 40% or more of the bottle is visible and it is clearly a bottle. Box = visible part only. (Accepted.) | decided |
| 2 | Bottle behind another bottle (only cap or neck visible) | Do not label. | decided |
| 2b | Bottle inside a crate (glass returnables are target products) | Label only clearly visible bottles, following case 1. | decided |
| 3 | Water, juice, iced tea, energy drink bottles | **Label as `bottle`** (agreed). They end up `unknown` after the embeddings. | agreed |
| 4 | Cans, and multipacks of cans | Never labeled. | agreed |
| 5 | Bottle lying down or tilted | Label, tight box around the tilted shape. | decided |
| 6 | Bottles inside shrink-wrap or bundles | Label each bottle whose outline is individually distinguishable, not one box per pack. | decided |
| 7 | Bottle behind glass, with glare | Label if the outline is recognizable; skip if not. | decided |
| 8 | Very small, far-away bottles | Skip if the box would be under about 20 px tall, measured on the **original** image. **Provisional:** resolutions range from 374 to 4096 px, so revisit it in Stage 2 with the box size distribution. | provisional |
| 9 | Bottle cut by the photo edge | Label, box clipped to the image. | decided |
| 10 | Bottle in a shopping cart or held by a person | Label like any other. | decided |
| 11 | Large (2-3 L) and small (below 500 ml) bottles | Same class, same rules. | decided |
| 12 | Posters, cardboard cut-outs, printed bottles | Not labeled. | decided |
| 13 | Glass bottles | Label as `bottle`. | decided |
| 14 | Price tag or promo sleeve covering part of the bottle | Label; the box includes the covered part only if its outline is visible. | decided |
| 15 | Juice or milk **cartons** (Tetra Pak) | Not labeled: a carton is not a bottle. | decided |
| 16 | The 4 screenshot PNGs (`Captura de pantalla ...`) in `data/raw/` | Not shelf photos: excluded from the dataset through an exclusion list in Stage 1. They are never deleted or moved in `data/raw/`. | decided |

## Real examples from the photos (gate: at least 5)

Photos are identified by anonymous photo id (`<store code>_<hash>`), because the full file names contain uploader names and store addresses and must never be committed. Find a file with `python scripts/lookup_photo.py <photo id>`.

| # | Photo | What is ambiguous | Case | My default |
|---|-------|-------------------|------|------------|
| A | `s14_569b6500` and `s14_46eef7a8` | Large 2-3 L bottles on a pallet, the lower layers wrapped in plastic in groups. Bottles overlap and the wrap blurs the outline. | 6 | One box per bottle whose outline is clear; skip those showing only a sliver. |
| B | `s05_242677e1`; `s07_694d03ff` | Returnable bottles standing in red and blue crates. The crate walls hide the lower body, and the bottles inside have only cap and shoulders visible. | 1, 2b | Label only the clearly visible ones (about 40% of the bottle visible); otherwise skip. |
| C | `s16_1ee0e960`; `s02_faf3e2e2` | "Pack ahorro" promo sleeves tie a Coca-Cola bottle to a Fanta bottle. The sleeve covers most of the label, so the product is hard to see. | 14 | One box per bottle. The class stays `bottle`; Stage 6 will likely send these to `review` or `unknown`. |
| D | the Stage 0 test photo (in `data/test_images/`, not part of `data/raw/`) | Juice **cartons** (Vivo) at the top shelf. The pretrained COCO detector boxes them as `bottle` with 0.5 to 0.85 confidence. | 15 | Do not label cartons. This is the kind of false positive fine tuning must remove. |
| E | `s10_2962ca81` | Wide shot of a drinks aisle and a fridge. Dozens of bottles in the background are only a few tens of pixels tall. | 8 | Skip bottles below the provisional ~20 px threshold; it is revisited in Stage 2. |
| F | `s07_cfed3ea4` | A 3-row Coca-Cola display where the top rows are cut by the shelf above and the photo edge, with 3+ rows of overlapping bottles. | 1, 2, 9 | Label the visible bottles only, box clipped to the image. |
| G | `s03_fadb075c`, `s19_b14dcca4` | Shelves of 6-can multipacks next to a few PET bottles. | 4 | Label the PET bottles, never the cans. |
| H | `s12_16174372` | Small glass bottles of premium tonics and flavored water, next to cans. | 13 | Label as `bottle`. |

## Data quirks seen while browsing

- Many photos have **grey wedges on the left and right edges** (the app seems to straighten the perspective). There are no bottles in the grey areas; boxes must not extend into them.
- No photo has EXIF data (orientation is always normal and there are no dates or GPS), so there is nothing to fix with `exif_transpose` for this set. Keep the call anyway.

## Decisions (all confirmed by the owner)

1. 40% visibility rule: accepted. About 20 px minimum height: provisional, measured on the original image, revisited in Stage 2.
2. Shrink-wrap: label each bottle whose outline is individually distinguishable.
3. Cartons: not labeled.
4. Screenshots: excluded from the dataset through an exclusion list in Stage 1; never deleted or moved in `data/raw/`.
5. Crates: label only clearly visible bottles, following case 1. This matters because glass returnables are target products.
