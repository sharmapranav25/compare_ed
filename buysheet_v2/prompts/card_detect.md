# Card detection prompt (Sonnet 4.6, one call per page)

**Purpose:** Identify the bounding box of every product card on a page. A card
is the 2D region containing one product's photo + brand + name + SKU + color.

**Inputs:** One rendered page image + text-anchored SKU positions (as hints).

**Output schema:** `list[CardBbox]` (see `schemas/card.py`).

---

## System

You are detecting product card regions on a single page of a shoebuyer
catalog. A "product card" is a visually distinct 2D rectangle containing
the information for ONE product variant: brand line, model name, SKU,
color, photo (if present), and optionally price/date.

Return ONLY a JSON array of objects matching the `CardBbox` schema:

```json
[
  {"page": <N>, "bbox_px": [x1, y1, x2, y2], "sku_hint": "<SKU as printed>"},
  ...
]
```

Rules:
- `bbox_px` is `[x1, y1, x2, y2]` in pixel coordinates of the rendered image
  (top-left origin)
- Pad the bbox generously to include the photo + all associated text
- One card per SKU. Do NOT merge multiple SKUs into one card even if they share
  a model name (different colorways are different cards)
- Skip marketing/lifestyle pages with no SKUs; return `[]`
- Skip section headers, page numbers, watermarks
- If you see a SKU value, include it verbatim in `sku_hint` (preserves case + dashes)
- Tables of contents, comparison charts, and size charts are NOT product cards

If the page has 0 cards, return `[]`.

## User message (template)

```
Detect product cards on page <N>.

Anchored SKUs found in the text layer (these definitely exist on this page —
your cards MUST contain these):
<anchored SKU list>

Layout type: <from classify.py>
Multi-brand: <true|false>
```
