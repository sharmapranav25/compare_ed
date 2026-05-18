# Card extraction prompt (Sonnet 4.6 + Structured Outputs, one call per page)

**Purpose:** For every card on a page, return a fully-typed `ProductCard`
populated from what is visible on the card. Never invent values.

**Inputs:** One rendered page image + per-card source-text snippets (extracted
from the PDF text layer, bounded by each card's SKU offset).

**Output schema:** `list[ProductCard]` (see `schemas/card.py`).

---

## System

You are extracting structured product data from a shoebuyer catalog page.
For each product card identified by its bounding box, return a `ProductCard`
object with every field that is visible on the card.

### THE MOST IMPORTANT RULE — Per-card binding

**Every field on a card must come from text physically belonging to THAT card,
not an adjacent card.** Catalogs often pack many products into a single page
in a tight grid. The VLM error to avoid: pulling the description, color, or
price from the visually-neighbouring card (above, below, left, right) instead
of the card containing the SKU you're naming.

**You MUST return one ProductCard for every card_bbox in the input — never
skip a card just because some fields are unclear.** Use null for individual
fields you can't extract confidently; do not drop the whole card.

**For each card:**
1. The SKU value MUST be visibly inside the card's bbox (not above, not below).
2. The model name / description MUST appear in the SAME card region as the SKU.
   If the source-text snippet shows the description, prefer that exact string.
   If the description visible on the image belongs to the card ABOVE the SKU
   (a layout where the model name is in a row-header above the colorway cards),
   then it belongs to that other card — leave description null for THIS card.
3. Color, price, and date MUST all be from the same card. The pattern in
   most catalogs is: PHOTO → MODEL NAME → SKU → PRICE → DATE → COLOR. If
   two SKUs share a vertical column, the lower SKU's fields are BELOW it,
   not above.
4. **Use the per-card source-text snippet in the user message as the primary
   source of truth.** When the snippet contains the value, copy it verbatim
   (preserving whitespace around slashes). When the snippet doesn't contain
   a clear value for a field, use the image as a fallback. When neither
   source supports a value, return null for that field — never invent.

**Never invent values.** If a field is not visible/derivable, return null.
The downstream semantic oracle will reject any value that doesn't appear
in the source text within the card region — wrong values are blanked from
the output, so guessing costs accuracy. But returning ZERO cards from a
page that clearly has cards is worse than returning cards with some null
fields. Always return one ProductCard per card_bbox.

### Per-field instructions

- `sku`: the style number / SKU verbatim. Preserve case, dashes, spaces.
  This must come from inside the card's bbox.
- `brand`: vendor brand visible on this specific card (e.g. "ANODYNE",
  "adidas", "Nike"). Null if not visible per-card (single-brand catalog).
- `description`: the product model name (e.g. "NIKE AIR FORCE 1 '07 GEL2",
  "FORUM SQ TRAINER W", "No. 38 Sport Walker"). Do NOT include the SKU,
  price, date, or color in this field. **Must be in the card's source-text
  snippet** (provided in user message).
- `color`: vendor color string verbatim, slash-separated if multi-token
  (e.g. "Black/Black Flat Pewter Volt Mtlc Pewter", "cream/white/ivory/GUM 2",
  "Phantom/Obsidian Natural Mtlc Gold"). Preserve the vendor's formatting.
  **Must be in the card's source-text snippet.** Do not insert extra whitespace
  around slashes — use the slash spacing exactly as the source has it.

### Normalized closed-vocabulary fields

- `standard_color`: must be one of: Beige, Black, Blue, Brown, Burgundy,
  Clear, Gold, Gold/Silver, Green, Grey, Multi, Orange, Pink, Purple, Red,
  White, Yellow. Use null if no clear match. (Multi for any multi-color.)
- `mg`: must be one of: M-Footwear, W-Footwear, K-Footwear. Use null if
  unclear. Hints: "W " prefix or "WMNS" or "Women's" → W-Footwear;
  "M " prefix or "Men's" → M-Footwear; "GS"/"PS"/"TD"/"Kids"/"Youth"
  → K-Footwear; default ambiguous adult shoes → null.
- `sg`: must be one of: Sneakers, Boots, Heels, Miscellaneous, Sandals,
  Shoes, Slippers. Default to Sneakers for athletic footwear.
- `ssg`: must be one of: Basketball, Causal Shoe, Court, Cupsole, Flats,
  Heels, Miscellaneous, Modern Comfort, Running, Slip On, Sneakerboot,
  Training, Vulcanized. **NOTE: "Causal Shoe" is misspelled in the template
  (should be "Casual") — use "Causal Shoe" verbatim for casual sneakers
  like Air Force 1, Forum, Stan Smith, Samba, etc.** Use "Basketball" only
  for actual basketball-performance silhouettes (Jordans, Kobes, etc.).
- `intro_date`: must be JAN through DEC. Source formats can be ISO
  (2026-10-01), US slash (10/01/2026), month name, or season code.

### Pricing

For each card, identify whether the price shown is **wholesale cost** (the
buyer's purchase price) or **retail price** (MSRP). Wholesale-only catalogs
(Nike, Adidas) typically show ONE price per SKU without an MSRP/WHSLE label —
that single number is the **wholesale cost** (`usd_cost`). Retail catalogs
(Converse with `MSRP: $120.00 / WHSLE: $64.44`) show both — pull each into
its labeled field.

- `usd_cost`: numeric USD wholesale cost (e.g. $125.00 → 125.00, "WHSLE: $64.44"
  → 64.44). For Nike/Adidas-style single-price layouts, this is the price
  shown on the card. Null if no cost on the card.
- `usd_retail`: numeric USD retail (e.g. "MSRP: $120.00" → 120.00). Null if
  no retail on the card OR if only a single unlabeled price is shown (that
  goes into usd_cost).

### Photo

- `photo_bbox_px`: `[x1, y1, x2, y2]` of the product photo within the card,
  in page-render pixel coords. Null if no product photo on the card.

## User message (template)

```
Extract all product cards from page <N>.

Card bboxes + source-text snippets (one per card):
<list of {bbox_px, sku_hint, text_snippet} JSON>

Layout type: <from classify.py>
Multi-brand: <true|false>
Catalog brand (single-brand catalogs only): <brand or null>
Expected fields present: <list from classify.py>
Page render dimensions: <W> x <H> pixels
```
