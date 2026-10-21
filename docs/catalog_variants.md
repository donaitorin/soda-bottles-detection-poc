# Catalog: the 10 target products

**Status: FINAL for the product list; sizes and frequencies are still rough.** The product list and the decisions below come from the project owner. The size and frequency notes come from a visual look at 30 photos (thumbnails and a few close-ups), not a count. Crop counts are impossible before labeling, so after Stage 3 the counts per product are reported and products with too few train crops are flagged as **low data** (reported separately, never dropped).

## Target products

| id | Product |
|----|---------|
| `coca_classic` | Coca-Cola Classic, bottle |
| `coca_classic_returnable` | Coca-Cola Classic, returnable bottle |
| `coca_zero` | Coca-Cola Zero, bottle |
| `coca_zero_returnable` | Coca-Cola Zero, returnable bottle |
| `coca_light` | Coca-Cola Light, bottle |
| `sprite` | Sprite, bottle |
| `sprite_returnable` | Sprite, returnable bottle |
| `sprite_zero` | Sprite Zero, bottle |
| `pepsi` | Pepsi, bottle |
| `pepsi_returnable` | Pepsi, returnable bottle |

- The identification target is the **product** (the 10 ids).
- **Size** (about 1.5 L, 2 to 3 L...) and **material** (PET or glass) are secondary metadata stored in the catalog and evaluated only where visible; they are not hard success criteria.
- **Glass returnable Coca-Cola bottles in crates count as returnable products**, and they can be Classic or Zero (`coca_classic_returnable` or `coca_zero_returnable`). Glass bottles get their own reference images because they look nothing like the PET ones.
- **Every other bottle is `unknown`**: other brands (Fanta, Kem, Bilz y Pap, 7UP, Canada Dry, Limon Soda...), water, juice. See "Unknown handling" in `docs/PLAN.md`.
- `coca_zero_returnable` and `sprite_zero` stay in the list. The owner says they are common, and a 30-photo thumbnail sample is not enough to drop them.

## Sizes and frequency seen in the sample (rough)

Sizes are what I could read on labels or price tags; most bottles are too small in the photos to read.

| id | Sizes seen | Frequency in sample |
|----|-----------|---------------------|
| `coca_classic` | 1.5 L legible; 2 to 3 L bottles also appear (shrink-wrapped and loose) | frequent |
| `coca_zero` | 1.5 L legible; larger bottles with a black and red label | frequent |
| `coca_light` | large bottles with a silver/white label | occasional (roughly 3 of 30 photos) |
| `sprite` | 2 to 2.5 L (clear bottle, green label) | frequent |
| `pepsi` | 2 to 2.5 L (blue label) | frequent |
| `coca_classic_returnable` | about 1.5 L PET, plus glass in crates | occasional |
| `pepsi_returnable` | 2.5 L (label says "2.5 L RETORNABLE") | seen in one photo |
| `sprite_returnable` | likely 2 to 2.5 L, green cap | seen in one photo, to confirm |
| `coca_zero_returnable` | not seen in the sample | unknown: owner says common |
| `sprite_zero` | not clearly seen in the sample | unknown: owner says common |

## Returnable vs non-returnable: what separates them

Cues confirmed by the owner, per bottle:

- a **different cap color** (green in these photos; non-returnable Coca-Cola has a red cap);
- **text on the label** (the "RETORNABLE" text; the keyword used by OCR in Stage 7);
- a **different bottle shape or plastic** (non-returnable PET has a dark "petaloid" base; the returnable one looks different).

Price and shelf signage ("ENVASES RETORNABLES, VOLVER A USAR") are **context only**: they cannot be used per bottle.

**Zero vs Classic cue** (owner): Zero says "zero azúcares" and its text is black; Classic text is white. This also holds for the returnable versions.

## Known risks for Stages 6 and 7

1. **Returnable vs non-returnable** share the same red label, so embeddings may confuse them. Cap color and OCR ("RETORNABLE") are the likely fixes.
2. **Classic vs Zero** at 1.5 L use almost the same red label; the difference is the small text line (white vs black, "ZERO AZUCARES"). OCR should help.
3. Several products have two label designs (for example the 1.5 L red label and the 2 L+ black/red band), which means more reference images per product.
4. Glass bottles in crates: only partly visible, and they may need their own catalog images.
5. Promo sleeves (Pack Ahorro) hide the label: expect `review` or `unknown`.
6. With 10 ids, some products (for example `coca_zero_returnable`, `sprite_zero`, `sprite_returnable`, `pepsi_returnable`) may have few crops. They are flagged as low data after Stage 3, not dropped.
