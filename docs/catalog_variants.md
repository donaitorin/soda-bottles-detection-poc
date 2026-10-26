# Catalog: the 6 target products

**Status: FINAL for the product list; sizes and frequencies are still rough.** The product list comes from the project owner. History: the list had 10 ids (Pepsi, Sprite Zero and several returnable versions included) and was **reduced to 6 for the tests** by owner decision; the removed ids are out of scope, not "low data". The size and frequency notes come from a visual look at 30 photos (thumbnails and a few close-ups), not a count.

## Target products

| id | Product |
|----|---------|
| `coca_classic` | Coca-Cola Classic, regular PET bottle |
| `coca_classic_returnable` | Coca-Cola Classic, returnable bottle (the only returnable product kept) |
| `coca_zero` | Coca-Cola Zero, regular PET bottle |
| `coca_light` | Coca-Cola Light |
| `sprite` | Sprite, regular PET bottle |
| `fanta` | Fanta, regular PET bottle (new: it used to be `unknown`) |

- The identification target is the **product** (the 6 ids). Brands: Coca-Cola (4 products), Sprite, Fanta (Fanta is its own brand).
- **Everything else is `unknown`**: other brands (Pepsi, 7UP, Kem, Bilz y Pap, Canada Dry, Limon Soda...), water, juice, and also **Sprite Zero and the returnable versions of Zero, Sprite and Fanta**. There is no `other brand` class. See "Unknown handling" in `docs/PLAN.md`.
- **Glass returnable Coca-Cola Classic bottles in crates count as `coca_classic_returnable`.** A glass or PET returnable Zero is `unknown` now.
- **Size** (about 1.5 L, 2 to 3 L...) and **material** (PET or glass) are secondary metadata stored in the catalog and evaluated only where visible; they are not hard success criteria.
- Catalog reference images: about 5 clean product photos per product (30 in total), provided by the owner (see `docs/PLAN.md`, Stage 6).

## Sizes and frequency seen in the sample (rough)

| id | Sizes seen | Frequency in sample |
|----|-----------|---------------------|
| `coca_classic` | 1.5 L legible; 2 to 3 L bottles also appear (shrink-wrapped and loose) | frequent |
| `coca_zero` | 1.5 L legible; larger bottles with a black and red label | frequent |
| `coca_light` | large bottles with a silver/white label | occasional (roughly 3 of 30 photos) |
| `sprite` | 2 to 2.5 L (clear bottle, green label) | frequent |
| `coca_classic_returnable` | about 1.5 L PET, plus glass in crates | occasional |
| `fanta` | not assessed in the 30-photo sample (it was `unknown` then) | to count after the first labeling |

## Returnable vs non-returnable (Coca-Cola Classic): what separates them

Cues confirmed by the owner, per bottle:

- a **different cap color** (green in these photos; non-returnable Coca-Cola has a red cap);
- **text on the label** (the "RETORNABLE" text; the keyword used by OCR in Stage 7);
- a **different bottle shape or plastic** (non-returnable PET has a dark "petaloid" base; the returnable one looks different).

Price and shelf signage ("ENVASES RETORNABLES, VOLVER A USAR") are **context only**: they cannot be used per bottle.

**Zero vs Classic cue** (owner): Zero says "zero azúcares" and its text is black; Classic text is white.

## Known risks for Stages 6 and 7

1. **Returnable vs non-returnable Coca-Cola Classic** share the same red label, so embeddings may confuse them. Cap color and OCR ("RETORNABLE") are the likely fixes.
2. **Classic vs Zero vs Light** at 1.5 L use almost the same red or silver label; the difference is the small text line (white vs black, "ZERO AZUCARES", "LIGHT"). OCR should help.
3. Several products have two label designs (for example the 1.5 L red label and the 2 L+ black/red band), which means more reference images per product.
4. Glass bottles in crates: only partly visible, and they may need their own catalog images.
5. Promo sleeves (Pack Ahorro) hide the label: expect `review` or `unknown`.
6. **Open-set look-alikes become `unknown`:** other lemon-lime sodas (7UP, Limon Soda) against `sprite`, other orange sodas against `fanta`, and the Zero or returnable versions that were removed from the list (for example a returnable Zero against `coca_zero` or `coca_classic_returnable`) must be rejected as `unknown`. The false known rate is the number to watch.
7. Product photos from the internet (studio, plain background) differ from shelf crops: the domain gap is measured in Stage 6b.
