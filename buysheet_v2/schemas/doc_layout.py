"""Doc-level layout classification — one Opus call on the first 3 pages."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

LayoutType = Literal[
    "grid",            # Adidas-style matrix: multiple SKUs per row, table-like
    "vertical_card",   # Nike/Hanger-style: one SKU per card, stacked vertically
    "spec_panel",      # Converse-style: dense per-product spec blocks
    "lookbook",        # Hoka-style: lifestyle/marketing layout with sparse SKUs
    "mixed",           # Multi-section catalog with varied layouts
]


class LayoutClassification(BaseModel):
    """What kind of catalog is this? Drives downstream prompt selection."""

    layout_type: LayoutType = Field(
        ..., description="Dominant page layout pattern across the catalog"
    )
    is_multi_brand: bool = Field(
        ..., description="True if multiple distinct brands appear per page (e.g. Hanger Clinic)"
    )
    expected_fields_present: list[
        Literal["sku", "brand", "description", "color", "mg", "sg", "ssg",
                "intro_date", "usd_cost", "usd_retail"]
    ] = Field(
        ...,
        description="Which template fields appear to be derivable from this catalog's content",
    )
    notes: Optional[str] = Field(
        None,
        description="Free-text observations: section structure, brand list, quirks",
    )
