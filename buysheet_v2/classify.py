"""Doc-level layout classification — one Opus 4.7 call on the first N pages.

Reads the prompt from prompts/layout_classify.md and returns a
LayoutClassification (layout type + multi-brand flag + expected fields).
This metadata propagates to cards.py and extract.py prompts as hints.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import anthropic

from buysheet_v2.ingest import IngestedPDF
from buysheet_v2.lifted.pdf_render import b64_image_block
from buysheet_v2.schemas.doc_layout import LayoutClassification

MODEL = "claude-opus-4-7"
MAX_TOKENS = 1024
SAMPLE_PAGES = 3

_PROMPT_PATH = Path(__file__).parent / "prompts" / "layout_classify.md"


def _load_system_prompt() -> str:
    text = _PROMPT_PATH.read_text()
    if "## User message" in text:
        text = text.split("## User message")[0]
    return text


def classify_layout(
    doc: IngestedPDF, *, client: Optional[anthropic.Anthropic] = None
) -> tuple[LayoutClassification, dict]:
    """Classify the catalog layout from the first SAMPLE_PAGES rendered pages.

    Uses Opus 4.7 (small token volume, document-level decision benefits from
    the larger model). Result drives downstream per-page prompts.

    Returns (LayoutClassification, usage_metadata).
    """
    if client is None:
        client = anthropic.Anthropic()

    sample = doc.pages[:SAMPLE_PAGES]
    if not sample:
        # Degenerate empty PDF — return safe defaults
        return (
            LayoutClassification(
                layout_type="mixed", is_multi_brand=False,
                expected_fields_present=[], notes="empty document",
            ),
            {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0},
        )

    image_blocks = [b64_image_block(p.png_bytes) for p in sample]
    user_text = (
        f"Here are pages {', '.join(str(p.page_no) for p in sample)} of a "
        f"shoebuyer catalog. Classify the layout."
    )

    response = client.messages.parse(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=_load_system_prompt(),
        messages=[{
            "role": "user",
            "content": image_blocks + [{"type": "text", "text": user_text}],
        }],
        output_format=LayoutClassification,
    )

    usage = {
        "input_tokens": getattr(response.usage, "input_tokens", 0),
        "output_tokens": getattr(response.usage, "output_tokens", 0),
        "cache_read_tokens": getattr(response.usage, "cache_read_input_tokens", 0) or 0,
        "sample_page_count": len(sample),
    }
    return response.parsed_output, usage
