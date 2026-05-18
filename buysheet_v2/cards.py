"""Per-page card detection via Sonnet 4.6 vision.

Sends one page render to Claude with the prompt from prompts/card_detect.md
and the SKU anchors found in the text layer as hints. Returns a list of
CardBbox objects, schema-enforced via Anthropic Structured Outputs.

This is the primary primitive that generalizes across vendor layouts.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import anthropic
from pydantic import BaseModel, Field

from buysheet_v2.ingest import IngestedPage, anchor_skus_on_page
from buysheet_v2.lifted.pdf_render import b64_image_block
from buysheet_v2.schemas.card import CardBbox

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 4096

_PROMPT_PATH = Path(__file__).parent / "prompts" / "card_detect.md"


class CardDetectionResponse(BaseModel):
    """Wrapper for the VLM's per-page response (Pydantic needs a top-level object)."""

    cards: list[CardBbox] = Field(default_factory=list)


def _load_system_prompt() -> str:
    """Load the card_detect.md prompt body, stripping the user-message template."""
    text = _PROMPT_PATH.read_text()
    # Split on the "## User message" marker — keep only the system half
    if "## User message" in text:
        text = text.split("## User message")[0]
    return text


def detect_cards_on_page(
    page: IngestedPage,
    *,
    client: Optional[anthropic.Anthropic] = None,
    layout_type: str = "unknown",
    is_multi_brand: bool = False,
) -> tuple[list[CardBbox], dict]:
    """Detect product card bboxes on a single page.

    Returns (cards, usage_metadata). usage_metadata has token counts + cost
    estimate for the call.
    """
    if client is None:
        client = anthropic.Anthropic()

    anchored = anchor_skus_on_page(page.text)
    system_prompt = _load_system_prompt()
    user_text = (
        f"Detect product cards on page {page.page_no}.\n\n"
        f"Anchored SKUs found in the text layer (these definitely exist on this page — "
        f"your cards MUST contain these):\n"
        f"{json.dumps(anchored)}\n\n"
        f"Layout type: {layout_type}\n"
        f"Multi-brand: {is_multi_brand}\n"
        f"Page render dimensions: {page.width_px} x {page.height_px} pixels"
    )

    response = client.messages.parse(
        model=MODEL,
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
        output_format=CardDetectionResponse,
    )

    parsed = response.parsed_output
    # Ensure page numbers are set even if VLM omitted them
    for c in parsed.cards:
        if c.page != page.page_no:
            c.page = page.page_no

    usage = {
        "page": page.page_no,
        "input_tokens": getattr(response.usage, "input_tokens", 0),
        "output_tokens": getattr(response.usage, "output_tokens", 0),
        "cache_read_tokens": getattr(response.usage, "cache_read_input_tokens", 0) or 0,
        "anchored_sku_count": len(anchored),
        "detected_card_count": len(parsed.cards),
    }
    return parsed.cards, usage


def overlay_cards_on_page(
    page: IngestedPage, cards: list[CardBbox], *, sku_hints: bool = True
) -> bytes:
    """Render the page with card bboxes overlaid. Returns PNG bytes.

    For visual debugging: shows the VLM-detected card regions in red, with
    sku_hint labels in green. Used by `buysheet_v2 debug-cards`.
    """
    import io
    from PIL import Image, ImageDraw, ImageFont

    img = Image.open(io.BytesIO(page.png_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 24)
    except OSError:
        font = ImageFont.load_default()

    for i, card in enumerate(cards):
        x1, y1, x2, y2 = card.bbox_px
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        label = f"#{i}"
        if sku_hints and card.sku_hint:
            label += f"  {card.sku_hint}"
        draw.rectangle([x1, max(0, y1 - 30), x1 + 350, y1], fill="white")
        draw.text((x1 + 6, max(0, y1 - 28)), label, fill="red", font=font)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
