# Labeling guide: the `bottle` class

**Status: DRAFT.** The defaults below are my recommendations. Every row marked *decide* waits for the project owner. The guide becomes final once the owner has answered them.

## The rule

There is **one class: `bottle`**. Label every soda-style bottle the same way, whatever the brand, flavor, size or contents. Brand and product are *never* part of the detector label; embeddings resolve them in Stage 6 (anything that is not one of the 8 target products becomes `unknown` there).

- The box is tight around the **visible** part of the bottle (cap to base), never the estimated part hidden behind another object.
- One box per bottle.
- Cans are out of scope: they are not labeled and there is no `can` class.
- Image coordinates always refer to the photo after fixing EXIF orientation (Stage 1).

## Ambiguous cases

| # | Case | Default (recommended) | Decide |
|---|------|-----------------------|--------|
| 1 | Back-row bottle only half visible | Label if about 40% or more of the bottle is visible and it is clearly a bottle. Box = visible part only. | yes |
| 2 | Bottle behind another bottle (only cap or neck visible) | Do not label. | yes |
| 3 | Water, juice, iced tea, energy drink bottles | **Label as `bottle`** (agreed). They end up `unknown` after the embeddings. | agreed |
| 4 | Cans, and multipacks of cans | Never labeled. | agreed |
| 5 | Bottle lying down or tilted | Label, tight box around the tilted shape. | yes |
| 6 | Bottles inside shrink-wrap or bundles | One box per bottle that is individually distinguishable, not one box per pack. | yes |
| 7 | Bottle behind glass, with glare | Label if the outline is recognizable; skip if not. | yes |
| 8 | Very small, far-away bottles | Skip if the box would be under about 20 px tall (to be re-checked on the data). | yes |
| 9 | Bottle cut by the photo edge | Label, box clipped to the image. | yes |
| 10 | Bottle in a shopping cart or held by a person | Label like any other. | yes |
| 11 | Large (2-3 L) and small (below 500 ml) bottles | Same class, same rules. | yes |
| 12 | Posters, cardboard cut-outs, printed bottles | Not labeled. | yes |
| 13 | Glass bottles | Label as `bottle`. | yes |
| 14 | Price tag or promo sleeve covering part of the bottle | Label; the box includes the covered part only if its outline is visible. | yes |
| 15 | Juice or milk **cartons** (Tetra Pak) | Not labeled: a carton is not a bottle. | yes |
| 16 | The 4 screenshot PNGs (`Captura de pantalla ...`) in `data/raw/` | Not shelf photos: exclude them from the dataset (Stage 1). | yes |

## Real examples from the photos (gate: at least 5)

Photos are identified by store id and the trailing photo number, because the full file names contain uploader names. Find one with `ls data/raw | grep <store> | grep <number>`.

| # | Photo | What is ambiguous | Case | My default |
|---|-------|-------------------|------|------------|
| A | store 500457984, `PHOTO-000153` and `PHOTO-000158` | Large 2-3 L bottles on a pallet, the lower layers wrapped in plastic in groups. Bottles overlap and the wrap blurs the outline. | 6 | One box per bottle whose outline is clear; skip those showing only a sliver. |
| B | store 500294593, `PHOTO-000001`; store 500314265, `PHOTO-000160` | Returnable bottles standing in red and blue crates. The crate walls hide the lower body, and the bottles inside have only cap and shoulders visible. | 1, 2 | Label if the cap-to-shoulder part is clear and about 40% of the body shows; otherwise skip. |
| C | store 500458029, `PHOTO-000193`; store 500280016, `PH-000327` | "Pack ahorro" promo sleeves tie a Coca-Cola bottle to a Fanta bottle. The sleeve covers most of the label, so the product is hard to see. | 14 | One box per bottle. The class stays `bottle`; Stage 6 will likely send these to `review` or `unknown`. |
| D | store 501101631, `PH-000010` (the test photo) | Juice **cartons** (Vivo) at the top shelf. The pretrained COCO detector boxes them as `bottle` with 0.5 to 0.85 confidence. | 15 | Do not label cartons. This is the kind of false positive fine tuning must remove. |
| E | store 500353325, `94ad1752.jpg` | Wide shot of a drinks aisle and a fridge. Dozens of bottles in the background are only a few tens of pixels tall. | 8 | Skip bottles below the size threshold. Decide the threshold after seeing box sizes. |
| F | store 500314265, `PHOTO-000182` | A 3-row Coca-Cola display where the top rows are cut by the shelf above and the photo edge, with 3+ rows of overlapping bottles. | 1, 2, 9 | Label the visible bottles only, box clipped to the image. |
| G | store 500280430 `PHOTO-000072`, store 501129389 `PHOTO-000079` | Shelves of 6-can multipacks next to a few PET bottles. | 4 | Label the PET bottles, never the cans. |
| H | store 500367002, `PHOTO-000245` | Small glass bottles of premium tonics and flavored water, next to cans. | 13 | Label as `bottle`. |

## Data quirks seen while browsing

- Many photos have **grey wedges on the left and right edges** (the app seems to straighten the perspective). There are no bottles in the grey areas; boxes must not extend into them.
- No photo has EXIF data (orientation is always normal and there are no dates or GPS), so there is nothing to fix with `exif_transpose` for this set. Keep the call anyway.

## Decisions I need from the owner

1. Cases 1, 2 and 8: are the 40% visibility rule and the ~20 px minimum height acceptable?
2. Case 6 and example A: label bottles inside shrink-wrap, or skip every wrapped bottle?
3. Case 15: confirm that cartons are not labeled.
4. Case 16: confirm that the 4 screenshots are excluded.
5. Example B: crates hide bottles. Label only clearly visible ones (my default), or none inside crates?
