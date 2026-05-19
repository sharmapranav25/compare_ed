"""Per-page card extraction via Sonnet 4.6 + Anthropic Structured Outputs.

For each card detected on a page (by cards.py), extract the full ProductCard
schema. One API call per page returning all cards' fields together. Pydantic
enforces field constraints at the API layer — no JSON parsing errors, no
schema drift.

Each card is sent with a per-card source-text snippet (bounded by SKU offsets
in the page text layer). This gives the VLM a textual grounding signal.

Per-card retry fallback: Claude 4.x has some inherent output variance (no
temperature control). When per-page extraction returns fewer cards than were
detected, we retry the missing cards individually with a cropped image of
just that card's bbox. This is more expensive (N extra calls) but
eliminates cross-card mis-attribution by construction.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Optional

import anthropic
from PIL import Image
from pydantic import BaseModel, Field

from buysheet_v2.ingest import IngestedPage, find_sku_in_text
from buysheet_v2.lifted.pdf_render import b64_image_block
from buysheet_v2.schemas.card import CardBbox, ProductCard

MODEL = "claude-sonnet-4-6"
# Per-token pricing for the supported Anthropic models (in / out per MTok).
# Used by extract_cards_on_page to compute cost when caller passes a non-default model.
_MODEL_PRICING = {
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-opus-4-7":   (15.0, 75.0),
    "claude-haiku-4-5":  (1.0, 5.0),
}
MAX_TOKENS = 16384  # 23 cards × ~150 tok/card + overhead fits comfortably under 16k

_PROMPT_PATH = Path(__file__).parent / "prompts" / "card_extract.md"


class CardExtractionResponse(BaseModel):
    """Per-page extraction wrapper (Pydantic root-list workaround)."""

    cards: list[ProductCard] = Field(default_factory=list)


def _load_system_prompt() -> str:
    text = _PROMPT_PATH.read_text()
    if "## User message" in text:
        text = text.split("## User message")[0]
    return text


def _card_text_snippet(
    page_text: str, sku: Optional[str], all_sku_hints: list[str], tail_chars: int = 300
) -> Optional[str]:
    """Extract the source-text region for this card (from after previous SKU to
    after this SKU + tail). Mirrors verify.py's card_text_region.

    None if the SKU isn't found in the page text (image-only PDFs).
    """
    if not sku or not page_text:
        return None
    offset = find_sku_in_text(page_text, sku)
    if offset is None:
        return None
    start, end = offset
    prev_ends = []
    for other in all_sku_hints:
        if not other or other == sku:
            continue
        o = find_sku_in_text(page_text, other)
        if o and o[1] <= start:
            prev_ends.append(o[1])
    region_start = max(prev_ends) if prev_ends else max(0, start - 250)
    return page_text[region_start : end + tail_chars].strip()


def extract_cards_on_page(
    page: IngestedPage,
    card_bboxes: list[CardBbox],
    *,
    client: Optional[anthropic.Anthropic] = None,
    layout_type: str = "unknown",
    is_multi_brand: bool = False,
    expected_fields: Optional[list[str]] = None,
    catalog_brand: Optional[str] = None,
    model: str = MODEL,
) -> tuple[list[ProductCard], dict]:
    """Extract full ProductCard schema for every detected card on the page.

    Returns (cards, usage_metadata). Empty list if card_bboxes is empty.
    """
    if not card_bboxes:
        return [], {"page": page.page_no, "input_tokens": 0, "output_tokens": 0,
                    "cache_read_tokens": 0, "extracted_card_count": 0}

    if client is None:
        client = anthropic.Anthropic()

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

    response = client.messages.parse(
        model=model,
        max_tokens=MAX_TOKENS,
        system=[{
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{
            "role": "user",
            "content": [
                b64_image_block(page.png_bytes),
                {"type": "text", "text": user_text},
            ],
        }],
        output_format=CardExtractionResponse,
    )

    parsed = response.parsed_output
    # Defensive: pin page number
    for c in parsed.cards:
        if c.page != page.page_no:
            c.page = page.page_no

    usage = {
        "page": page.page_no,
        "input_tokens": getattr(response.usage, "input_tokens", 0),
        "output_tokens": getattr(response.usage, "output_tokens", 0),
        "cache_read_tokens": getattr(response.usage, "cache_read_input_tokens", 0) or 0,
        "card_bbox_count": len(card_bboxes),
        "extracted_card_count": len(parsed.cards),
        "stop_reason": getattr(response, "stop_reason", None),
    }
    return parsed.cards, usage


# ---------------------------------------------------------------------------- per-card retry

def _crop_card_png(page_png: bytes, bbox_px: list[int]) -> bytes:
    """Crop a card bbox out of a page render. Returns PNG bytes."""
    img = Image.open(io.BytesIO(page_png)).convert("RGB")
    x1, y1, x2, y2 = bbox_px
    x1 = max(0, min(img.size[0], x1))
    y1 = max(0, min(img.size[1], y1))
    x2 = max(0, min(img.size[0], x2))
    y2 = max(0, min(img.size[1], y2))
    if x2 - x1 < 10 or y2 - y1 < 10:
        return page_png  # degenerate, fall back to full page
    cropped = img.crop((x1, y1, x2, y2))
    buf = io.BytesIO()
    cropped.save(buf, format="PNG")
    return buf.getvalue()


def extract_single_card(
    page: IngestedPage,
    card_bbox: CardBbox,
    *,
    client: Optional[anthropic.Anthropic] = None,
    layout_type: str = "unknown",
    is_multi_brand: bool = False,
    expected_fields: Optional[list[str]] = None,
    catalog_brand: Optional[str] = None,
    model: str = MODEL,
) -> tuple[Optional[ProductCard], dict]:
    """Extract a single card by sending JUST the cropped card image.

    Used as a retry fallback when the per-page batched extraction returns
    fewer cards than detected. Eliminates cross-card mis-attribution by
    construction — the VLM literally only sees one card.
    """
    if client is None:
        client = anthropic.Anthropic()

    crop_png = _crop_card_png(page.png_bytes, card_bbox.bbox_px)
    snippet = _card_text_snippet(
        page.text, card_bbox.sku_hint, [card_bbox.sku_hint] if card_bbox.sku_hint else []
    )
    system_prompt = _load_system_prompt()
    user_text = (
        f"Extract the single product card shown in this image (cropped from "
        f"page {page.page_no} of the catalog).\n\n"
        f"SKU hint from text-search: {card_bbox.sku_hint or 'none'}\n"
        f"Source-text snippet for this SKU's card region:\n{snippet or '(none — image-only page)'}\n\n"
        f"Layout type: {layout_type}\n"
        f"Multi-brand: {is_multi_brand}\n"
        f"Catalog brand (single-brand catalogs only): {catalog_brand or 'null'}\n"
        f"Expected fields present: {expected_fields or 'all'}\n\n"
        f"Return a list containing exactly ONE ProductCard for this card. "
        f"Set card_bbox_px to {card_bbox.bbox_px} (the original page coords). "
        f"For photo_bbox_px, infer from the cropped image but normalize back "
        f"to the original page coord system."
    )

    response = client.messages.parse(
        model=model,
        max_tokens=4096,
        system=[{
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{
            "role": "user",
            "content": [
                b64_image_block(crop_png),
                {"type": "text", "text": user_text},
            ],
        }],
        output_format=CardExtractionResponse,
    )

    parsed = response.parsed_output
    card = parsed.cards[0] if parsed.cards else None
    if card is not None:
        # Pin to original page + bbox; VLM may have reset these to crop-relative
        card.page = page.page_no
        card.card_bbox_px = card_bbox.bbox_px
        # CRITICAL: the VLM only sees the cropped image during a retry call, so
        # the photo_bbox it returns is in CROP-RELATIVE coords. Translate back to
        # page coords by adding card_bbox's top-left offset. Detect crop-relative
        # vs page-coord by checking if all values fit within the crop dimensions.
        if card.photo_bbox_px:
            cb_x1, cb_y1, cb_x2, cb_y2 = card_bbox.bbox_px
            crop_w = cb_x2 - cb_x1
            crop_h = cb_y2 - cb_y1
            px1, py1, px2, py2 = card.photo_bbox_px
            looks_crop_relative = (
                px2 <= crop_w + 4 and py2 <= crop_h + 4
                and px1 < cb_x1  # photo_bbox.x1 < card_bbox.x1 means it's NOT in page coords
            )
            if looks_crop_relative:
                card.photo_bbox_px = [
                    cb_x1 + px1, cb_y1 + py1,
                    cb_x1 + px2, cb_y1 + py2,
                ]

    usage = {
        "page": page.page_no,
        "sku_hint": card_bbox.sku_hint,
        "input_tokens": getattr(response.usage, "input_tokens", 0),
        "output_tokens": getattr(response.usage, "output_tokens", 0),
        "cache_read_tokens": getattr(response.usage, "cache_read_input_tokens", 0) or 0,
        "stop_reason": getattr(response, "stop_reason", None),
    }
    return card, usage


def extract_with_retry(
    page: IngestedPage,
    card_bboxes: list[CardBbox],
    *,
    client: Optional[anthropic.Anthropic] = None,
    layout_type: str = "unknown",
    is_multi_brand: bool = False,
    expected_fields: Optional[list[str]] = None,
    catalog_brand: Optional[str] = None,
    model: str = MODEL,
) -> tuple[list[ProductCard], dict]:
    """Per-page extraction with per-card retry fallback.

    Strategy:
      1. Try batched per-page extraction first (cheap, one call per page).
      2. If the response has fewer ProductCards than card_bboxes, identify
         which bboxes were missed (by sku_hint match) and retry each one
         individually with a cropped image. Each retry is ~$0.005 extra.

    Returns (cards, usage) where usage aggregates all calls made.
    """
    cards, page_usage = extract_cards_on_page(
        page, card_bboxes, client=client,
        layout_type=layout_type, is_multi_brand=is_multi_brand,
        expected_fields=expected_fields, catalog_brand=catalog_brand,
        model=model,
    )

    # Identify missing cards by sku_hint
    got_skus = {c.sku for c in cards}
    missing = []
    for cb in card_bboxes:
        if cb.sku_hint and cb.sku_hint not in got_skus:
            missing.append(cb)
        elif not cb.sku_hint:
            # No hint to match by — assume covered by bbox position match (best effort)
            pass

    total_usage = {
        "page": page.page_no,
        "input_tokens": page_usage["input_tokens"],
        "output_tokens": page_usage["output_tokens"],
        "cache_read_tokens": page_usage["cache_read_tokens"],
        "card_bbox_count": len(card_bboxes),
        "extracted_card_count": len(cards),
        "stop_reason": page_usage.get("stop_reason"),
        "retry_count": 0,
        "retry_recovered": 0,
    }

    if missing and client is None:
        client = anthropic.Anthropic()

    for cb in missing:
        try:
            card, retry_usage = extract_single_card(
                page, cb, client=client,
                layout_type=layout_type, is_multi_brand=is_multi_brand,
                expected_fields=expected_fields, catalog_brand=catalog_brand,
                model=model,
            )
            total_usage["retry_count"] += 1
            total_usage["input_tokens"] += retry_usage["input_tokens"]
            total_usage["output_tokens"] += retry_usage["output_tokens"]
            total_usage["cache_read_tokens"] += retry_usage["cache_read_tokens"]
            if card is not None:
                cards.append(card)
                total_usage["retry_recovered"] += 1
        except Exception:
            continue

    total_usage["extracted_card_count"] = len(cards)
    return cards, total_usage
