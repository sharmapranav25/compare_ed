"""Per-cell confidence model for extracted ProductCards.

Confidence levels (assigned by verify.py in Phase 3 — Phase 2 stub assigns
flat 0.5 to every field):

  1.0  value passed semantic oracle (string appears in source page text
       within the card region)
  0.7  value derived from vocab lookup (color_synonyms / description_map /
       silhouette_ssg_map) AND input string was confirmed in source
  0.5  value from VLM only — no source confirmation possible
  0.0  value contradicts source — flagged for review, value HIDDEN from xlsx
       with raw preserved in cell comment

Cells with overall confidence <0.9 are amber-highlighted in the workbook;
cells at 0.0 are blank in the workbook with the raw value in a cell comment.
"""
from __future__ import annotations

from typing import Optional

from buysheet_v2.schemas.card import ProductCard
from buysheet_v2.schemas.extraction_result import CardConfidence

FIELDS_TO_SCORE = (
    "brand", "sku", "description", "color", "standard_color",
    "mg", "sg", "ssg", "intro_date", "usd_cost", "usd_retail",
)


def stub_confidence(card: ProductCard, *, default: float = 0.5) -> CardConfidence:
    """Return a flat-confidence CardConfidence for one extracted card.

    Phase 2 placeholder — verify.py in Phase 3 replaces this with the real
    semantic-oracle scoring.
    """
    per_field = {}
    per_field_source = {}
    for f in FIELDS_TO_SCORE:
        v = getattr(card, f, None)
        if v is None:
            continue  # null fields don't contribute to confidence
        per_field[f] = default
        per_field_source[f] = "vlm"
    overall = (sum(per_field.values()) / len(per_field)) if per_field else 0.0
    return CardConfidence(
        sku=card.sku, page=card.page,
        per_field=per_field, per_field_source=per_field_source,
        overall=overall, flags=[],
    )
