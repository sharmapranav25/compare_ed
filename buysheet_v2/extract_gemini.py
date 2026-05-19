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

import json
import os
import re
import time
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from pydantic import BaseModel, Field

from buysheet_v2.extract import _card_text_snippet
from buysheet_v2.ingest import IngestedPage
from buysheet_v2.schemas.card import (
    MG, SG, SSG, CardBbox, Month, ProductCard, StandardColor,
)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "card_extract.md"


def _load_system_prompt() -> str:
    text = _PROMPT_PATH.read_text()
    if "## User message" in text:
        text = text.split("## User message")[0]
    return text

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# Free-tier rate limit on gemini-2.5-flash is 20 requests/minute. Pace
# ourselves below that so we don't burn batched runs on 429 retries.
# Override via env if you have a paid project with higher limits.
_MIN_INTERVAL_SEC = float(os.environ.get("GEMINI_MIN_INTERVAL_SEC", "4.0"))
_LAST_CALL_AT: list[float] = [0.0]

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


# NOTE: as of the apples-to-apples refactor, Gemini gets the SAME system prompt
# Sonnet does (loaded from prompts/card_extract.md) plus the SAME per-card hints
# (bbox + sku_hint + source_text_snippet). This makes the model the only
# variable across the 3-way comparison. The previous slim vendor-agnostic
# prompt was a deliberate "raw vision" baseline; the refactored version answers
# "how well does Gemini do when given everything Sonnet gets?".


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
    Uses the Sonnet system prompt + per-card bbox/sku/source-text hints so the
    only variable across the 3-way comparison is the model itself.
    """
    if not card_bboxes:
        return [], {"page": page.page_no, "input_tokens": 0, "output_tokens": 0,
                    "cache_read_tokens": 0, "cost_usd": 0.0,
                    "extracted_card_count": 0, "model": model}

    if client is None:
        if not os.environ.get("GEMINI_API_KEY"):
            raise RuntimeError(
                "GEMINI_API_KEY env var required. Get a key at "
                "https://aistudio.google.com/app/apikey and add to .env"
            )
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    system_prompt = _load_system_prompt()
    sku_hints = [c.sku_hint for c in card_bboxes if c.sku_hint]
    card_payload = []
    for c in card_bboxes:
        snippet = _card_text_snippet(page.text, c.sku_hint, sku_hints)
        card_payload.append({
            "bbox_px": c.bbox_px,
            "sku_hint": c.sku_hint,
            "source_text_snippet": snippet,
        })
    bbox_json = json.dumps(card_payload, indent=2)
    user_text = (
        f"Extract all product cards from page {page.page_no}.\n\n"
        f"Card bboxes + source-text snippets (one per card):\n{bbox_json}\n\n"
        f"Layout type: {layout_type}\n"
        f"Multi-brand: {is_multi_brand}\n"
        f"Catalog brand (single-brand catalogs only): {catalog_brand or 'null'}\n"
        f"Expected fields present: {expected_fields or 'all'}\n"
        f"Page render dimensions: {page.width_px} x {page.height_px} pixels"
    )

    # Proactive throttle to stay under free-tier 20 req/min limit before
    # the Gemini server has to send 429s. Module-level last-call timestamp
    # means the throttle is shared across all calls in this process.
    elapsed = time.time() - _LAST_CALL_AT[0]
    if elapsed < _MIN_INTERVAL_SEC:
        time.sleep(_MIN_INTERVAL_SEC - elapsed)

    # Gemini Flash free tier is 20 requests/minute; retry-with-backoff on 429s
    # and parse Google's suggested retryDelay when present. Up to 3 attempts.
    response = None
    for attempt in range(3):
        try:
            _LAST_CALL_AT[0] = time.time()
            response = client.models.generate_content(
                model=model,
                contents=[
                    types.Part.from_bytes(data=page.png_bytes, mime_type="image/png"),
                    types.Part.from_text(text=user_text),
                ],
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    response_schema=GeminiExtractionResponse,
                    max_output_tokens=16384,
                ),
            )
            break
        except genai_errors.ClientError as e:
            status = getattr(e, "status_code", None) or getattr(e, "code", None)
            if status != 429 or attempt == 2:
                raise
            # Parse Google's suggested retry delay from the error message
            wait = 60.0  # default
            msg = str(e)
            m = re.search(r"retry in ([\d.]+)s", msg)
            if m:
                wait = float(m.group(1)) + 2  # small safety buffer
            wait = min(wait, 90.0)  # cap so we don't sleep forever
            time.sleep(wait)
    if response is None:
        raise RuntimeError("Gemini call failed after retries (no response captured)")

    parsed: GeminiExtractionResponse = response.parsed  # type: ignore[assignment]

    # Map GeminiExtractedCard -> ProductCard so downstream verify/write code
    # works unchanged. Match each output card back to its input CardBbox by SKU
    # so card_bbox_px is the real detected bbox (needed for downstream photo
    # embedding); fall back to whole-page only if no SKU match exists.
    bbox_by_sku = {cb.sku_hint: cb.bbox_px for cb in card_bboxes if cb.sku_hint}
    page_bbox = [0, 0, page.width_px, page.height_px]
    cards: list[ProductCard] = []
    for gc in parsed.cards if parsed else []:
        try:
            cards.append(ProductCard(
                page=page.page_no,
                card_bbox_px=bbox_by_sku.get(gc.sku, page_bbox),
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
