"""Per-page card extraction via Gemini (alternate extractor for benchmarking).

Mirrors the interface of extract.py:extract_cards_on_page so the same call site
can use either Sonnet or Gemini. Used by tools/extract_compare.py to run
head-to-head extraction comparisons on the same page images / card bboxes.

Uses Gemini structured outputs via response_schema. The schema is a slimmer
version of ProductCard (no card_bbox_px / photo_bbox_px — Gemini works from
the page image directly without pre-computed bboxes, since the comparison
target is extracted FIELDS not extracted geometry).

Model: Gemini 2.5 Pro by default (~5x cheaper than Opus, ~50% cheaper than
Sonnet for inference, similar vision quality). Override via env or arg.

Pricing reference (May 2026, google.com/ai/pricing):
  gemini-2.5-pro:   $1.25 / $5.00  per MTok in/out
  gemini-2.5-flash: $0.30 / $2.50  per MTok in/out (cheaper, slightly less accurate)
"""
from __future__ import annotations

import os
from typing import Optional

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from buysheet_v2.ingest import IngestedPage
from buysheet_v2.schemas.card import (
    MG, SG, SSG, CardBbox, Month, ProductCard, StandardColor,
)

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# Pricing per MTok (in / out). Gemini 2.5 Pro is the default; flash is ~4x cheaper.
GEMINI_PRICING = {
    "gemini-2.5-pro":   (1.25, 5.00),
    "gemini-2.5-flash": (0.30, 2.50),
}


# Slimmer schema for Gemini extraction. Drops geometry fields (card_bbox_px,
# photo_bbox_px) since Gemini works from the full page image without pre-
# computed bboxes — the comparison target is field values, not card geometry.
class GeminiExtractedCard(BaseModel):
    sku: str = Field(..., description="Style number / SKU as printed (preserve case + dashes)")
    brand: Optional[str] = None
    description: str = Field(..., description="Product model name")
    color: str = Field(..., description="Vendor color string verbatim")
    mg: Optional[MG] = None
    sg: Optional[SG] = None
    ssg: Optional[SSG] = None
    standard_color: Optional[StandardColor] = None
    intro_date: Optional[Month] = None
    usd_cost: Optional[float] = Field(None, ge=0)
    usd_retail: Optional[float] = Field(None, ge=0)


class GeminiExtractionResponse(BaseModel):
    cards: list[GeminiExtractedCard] = Field(default_factory=list)


# Same general guidance Sonnet gets — we want to test the MODEL, not prompt
# engineering. The prompt is intentionally vendor-agnostic.
GEMINI_SYSTEM_PROMPT = """You are extracting product cards from a shoe-catalog page for a buyer's worksheet.

For each distinct product visible on this page, return a GeminiExtractedCard with:

  - sku: the printed style number, exactly as shown (preserve case + dashes)
  - brand: vendor brand if shown on the card or its section header (else null)
  - description: the product model/silhouette name (e.g. "AIR FORCE 1 '07", "BONDI 7")
  - color: the printed colorway verbatim (no normalization)
  - mg: "M-Footwear" / "W-Footwear" / "K-Footwear" — based on visible gender signal
        (WMNS, MEN, KIDS, GS, PS, JR, etc). Null if no signal.
  - sg: one of {Sneakers, Boots, Heels, Miscellaneous, Sandals, Shoes, Slippers}
  - ssg: one of {Basketball, Causal Shoe, Court, Cupsole, Flats, Heels,
                 Miscellaneous, Modern Comfort, Running, Slip On, Sneakerboot,
                 Training, Vulcanized}
  - standard_color: one of {Beige, Black, Blue, Brown, Burgundy, Clear, Gold,
                            Gold/Silver, Green, Grey, Multi, Orange, Pink,
                            Purple, Red, White, Yellow}
  - intro_date: 3-letter month code (JAN, FEB, ..., DEC) if a launch date is shown
  - usd_cost: USD wholesale price as a number (no $ sign) if shown
  - usd_retail: USD retail price as a number if shown

Rules:
  1. Extract each card independently — do not copy values across cards
  2. Return null for any field you cannot read from THIS card alone
  3. Real SKUs are alphanumeric codes (e.g. JA1013-010, 1110518-BBLC, X826W).
     If the only "SKU" visible is a slugified description, treat it as null.
  4. Color must be VERBATIM from the page text — do not interpret
  5. If a model name spans multiple colorway rows, apply it to all sibling SKUs
"""


def extract_cards_on_page_gemini(
    page: IngestedPage,
    card_bboxes: list[CardBbox],  # unused; kept for signature compatibility
    *,
    client: Optional[genai.Client] = None,
    layout_type: str = "unknown",
    is_multi_brand: bool = False,
    expected_fields: Optional[list[str]] = None,
    catalog_brand: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> tuple[list[ProductCard], dict]:
    """Extract all product cards from a page using Gemini.

    Returns (cards, usage_metadata) — same shape as extract.extract_cards_on_page.
    card_bboxes is accepted but ignored: Gemini works from the page image
    without pre-computed bboxes, which is part of the cross-model comparison.
    """
    _ = card_bboxes, expected_fields  # signature-compatibility only

    if client is None:
        if not os.environ.get("GEMINI_API_KEY"):
            raise RuntimeError(
                "GEMINI_API_KEY env var required. Get a key at "
                "https://aistudio.google.com/app/apikey and add to .env"
            )
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    user_text = (
        f"Extract all product cards from page {page.page_no} of this catalog.\n\n"
        f"Layout type: {layout_type}\n"
        f"Multi-brand: {is_multi_brand}\n"
        f"Catalog brand (single-brand catalogs only): {catalog_brand or 'null'}\n"
        f"Page render dimensions: {page.width_px} x {page.height_px} pixels"
    )

    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=page.png_bytes, mime_type="image/png"),
            types.Part.from_text(text=user_text),
        ],
        config=types.GenerateContentConfig(
            system_instruction=GEMINI_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=GeminiExtractionResponse,
            max_output_tokens=16384,
        ),
    )

    parsed: GeminiExtractionResponse = response.parsed  # type: ignore[assignment]

    # Map GeminiExtractedCard -> ProductCard so downstream verify/write code
    # works unchanged. We don't have card_bbox_px from Gemini, so synthesize
    # a placeholder (the comparison tool ignores bbox anyway).
    cards: list[ProductCard] = []
    for gc in parsed.cards if parsed else []:
        try:
            cards.append(ProductCard(
                page=page.page_no,
                card_bbox_px=[0, 0, page.width_px, page.height_px],  # whole-page placeholder
                photo_bbox_px=None,
                sku=gc.sku,
                brand=gc.brand,
                description=gc.description,
                color=gc.color,
                mg=gc.mg,
                sg=gc.sg,
                ssg=gc.ssg,
                standard_color=gc.standard_color,
                intro_date=gc.intro_date,
                usd_cost=gc.usd_cost,
                usd_retail=gc.usd_retail,
            ))
        except Exception:
            # Skip cards Gemini returned that don't satisfy ProductCard's
            # required-field constraints (e.g. blank sku/description/color)
            continue

    in_tok = response.usage_metadata.prompt_token_count or 0
    out_tok = response.usage_metadata.candidates_token_count or 0
    in_rate, out_rate = GEMINI_PRICING.get(model, (1.25, 5.00))
    cost_usd = (in_tok / 1e6) * in_rate + (out_tok / 1e6) * out_rate

    return cards, {
        "page": page.page_no,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "cache_read_tokens": 0,  # Gemini doesn't expose prompt-cache hits the same way
        "cost_usd": cost_usd,
        "extracted_card_count": len(cards),
        "model": model,
    }
