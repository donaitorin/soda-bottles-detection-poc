# Catalog: the 8 target products

**Status: DRAFT.** The product list comes from the project owner. The size and frequency notes below come from a visual look at 30 photos (one per store plus extras from the 11 biggest stores, thumbnails and a few close-ups), not a count. Crop counts are impossible before labeling (Stage 3), so products that look rare are flagged instead.

## Target products

| id | Product |
|----|---------|
| `coca_classic` | Coca-Cola Classic, bottle |
| `coca_classic_returnable` | Coca-Cola Classic, returnable bottle |
| `coca_zero` | Coca-Cola Zero, bottle |
| `coca_zero_returnable` | Coca-Cola Zero, returnable bottle |
| `coca_light` | Coca-Cola Light, bottle |
| `sprite` | Sprite, bottle |
| `sprite_zero` | Sprite Zero, bottle |
| `pepsi` | Pepsi, bottle |

- The identification target is the **product** (the 8 ids). Size is secondary metadata stored in the catalog and evaluated only where it is visible; it is not a hard success criterion.
- **Every other bottle is `unknown`**: other brands (Fanta, Kem, Bilz y Pap, 7UP, Canada Dry, Limon Soda...), water, juice. See "Unknown handling" in `docs/PLAN.md`.

## Sizes and frequency seen in the sample

Sizes are what I could read on labels or price tags; most bottles are too small in the photos to read.

| id | Sizes seen | Frequency in sample | Notes |
|----|-----------|---------------------|-------|
| `coca_classic` | 1.5 L clearly legible; 2 to 3 L bottles also appear (shrink-wrapped and loose) | frequent | Present in most stores. |
| `coca_zero` | 1.5 L legible; larger bottles with a black and red label | frequent | |
| `coca_light` | large bottles with a silver/white label | occasional | Roughly 3 of the 30 photos. |
| `sprite` | 2 to 2.5 L (clear bottle, green label) | frequent | Green-capped bottles in returnable sections too. |
| `pepsi` | 2 to 2.5 L (blue label) | frequent | A "2.5 L RETORNABLE" Pepsi also exists (see risks). |
| `coca_classic_returnable` | about 1.5 L, plus glass in crates | occasional | |
| `coca_zero_returnable` | not seen | **rare, possibly absent** | Check before building the catalog. |
| `sprite_zero` | not clearly seen | **rare, possibly absent** | Check before building the catalog. |

I did not see any small (500 ml or below) Coca-Cola, Sprite or Pepsi bottles standing alone, except possibly in some background shelves.

## Returnable vs non-returnable: what separates them

From close-ups of 1.5 L Coca-Cola bottles (non-returnable in store 500367002, `PHOTO-000255`; returnable-looking in store 500359287, `PH-000032`) and the returnable shelf in store 500314265, `PHOTO-000160`:

| Cue | Non-returnable | Returnable (as far as I can tell) |
|-----|----------------|-----------------------------------|
| Cap | red | **green** |
| Base | dark, "petaloid" feet | not clearly visible |
| Label | full red label with "Sabor original" or "Zero azucar" and "Reciclame" | red label with small extra text; wording not legible at this resolution |
| Shelf price | around $2,290 to $2,690 | cheaper (around $1,790 for the one I saw) |
| Store signage | none | green strips "ENVASES RETORNABLES, VOLVER A USAR" over the shelf |
| Material | PET | PET, and glass bottles in red crates |

I am **not certain** about the label and base differences, only about the cap color and the signage. The "RETORNABLE" text on the label (the OCR keyword in Stage 7) is probably the most reliable cue, but I could not read it on the bottles I looked at. **Known risk for Stages 6 and 7:** the two versions share the same red label, so embeddings may confuse them. Stage 7 OCR and cap color are the likely fixes.

## Other risks

1. **Classic vs Zero (1.5 L)** use almost the same red label; the difference is a small text line ("Sabor original" vs "Zero azucar"), as seen in store 500367002. This is the hardest pair for embeddings, and OCR ("ZERO") should help.
2. **Returnable versions of other brands exist** and are not in the 8 ids: Pepsi 2.5 L returnable, and probably Sprite returnable. Decision needed: do they map to `pepsi` and `sprite`, or to `unknown`?
3. Several Coca-Cola products have two label designs (the 1.5 L red label and the 2 L+ black/red band label), which means more reference images per product.
4. Glass returnable bottles in crates (stores 500294593 and 500314265) look nothing like the PET bottles. They may need their own catalog images.
5. Promo sleeves (Pack Ahorro) hide the label: expect `review` or `unknown`.

## Decisions needed from the owner

1. Are `coca_zero_returnable` and `sprite_zero` really present in your stores? If not, should I keep them in the list?
2. Pepsi and Sprite returnable bottles: map to `pepsi` / `sprite`, or `unknown`?
3. Do glass returnable bottles count as the same returnable products?
4. Confirm that the returnable cues above (green cap, "RETORNABLE" text, price) match what you know.
