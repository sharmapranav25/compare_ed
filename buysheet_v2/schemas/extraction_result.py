"""Per-page and per-catalog extraction result wrappers."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from buysheet_v2.schemas.card import ProductCard
from buysheet_v2.schemas.doc_layout import LayoutClassification


class PageExtraction(BaseModel):
    """All cards extracted from one page, plus metadata."""

    page: int = Field(..., ge=1)
    cards: list[ProductCard] = Field(default_factory=list)
    page_text: Optional[str] = Field(None, description="Source text layer for semantic oracle")
    error: Optional[str] = Field(None, description="Set if extraction failed for this page")


class CardConfidence(BaseModel):
    """Per-cell confidence + provenance for one ProductCard."""

    sku: str
    page: int
    per_field: dict[str, float] = Field(
        default_factory=dict,
        description="Field name -> confidence [0.0, 1.0]",
    )
    per_field_source: dict[str, str] = Field(
        default_factory=dict,
        description="Field name -> source attribution ('vlm', 'vocab_lookup', 'vlm+oracle_confirmed')",
    )
    overall: float = Field(0.0, ge=0.0, le=1.0)
    flags: list[str] = Field(default_factory=list)


class CatalogExtraction(BaseModel):
    """Full pipeline output for one PDF."""

    pdf_path: str
    vendor_key: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    layout: Optional[LayoutClassification] = None
    pages: list[PageExtraction] = Field(default_factory=list)
    confidence: list[CardConfidence] = Field(default_factory=list)
    cost_usd: float = 0.0
    tokens_input: int = 0
    tokens_output: int = 0
    tokens_cache_read: int = 0

    @property
    def all_cards(self) -> list[ProductCard]:
        return [c for p in self.pages for c in p.cards]
