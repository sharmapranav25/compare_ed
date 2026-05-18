"""ProductCard — the unit of truth for VLM-first extraction.

One card on one page = one xlsx row + one embedded photo. Fields match the 8
template columns we extract today (A-H + P + V + W), plus brand for multi-brand
catalog routing. Closed vocabularies are derived from BUYSHEET_template.xlsx
Product Data sheet (cached in vocab/buysheet_vocab.json).
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

# Closed vocabularies — keep these in sync with vocab/buysheet_vocab.json
MG = Literal["M-Footwear", "W-Footwear", "K-Footwear"]
SG = Literal[
    "Sneakers", "Boots", "Heels", "Miscellaneous", "Sandals", "Shoes", "Slippers"
]
SSG = Literal[
    "Basketball", "Causal Shoe", "Court", "Cupsole", "Flats", "Heels",
    "Miscellaneous", "Modern Comfort", "Running", "Slip On", "Sneakerboot",
    "Training", "Vulcanized",
]
StandardColor = Literal[
    "Beige", "Black", "Blue", "Brown", "Burgundy", "Clear", "Gold", "Gold/Silver",
    "Green", "Grey", "Multi", "Orange", "Pink", "Purple", "Red", "White", "Yellow",
]
Month = Literal[
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
    "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
]


class CardBbox(BaseModel):
    """Bounding box of a product card on a page, in pixel coords of the rendered image."""

    page: int = Field(..., ge=1, description="1-indexed page number")
    bbox_px: list[int] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="[x1, y1, x2, y2] in pixel coords of the page render",
    )
    sku_hint: Optional[str] = Field(
        None,
        description="If the SKU value is visible in the card, return it here verbatim",
    )


class ProductCard(BaseModel):
    """One product card extracted from one page.

    All optional fields are None when the source PDF does not contain the data
    for that field — the VLM must NOT invent values. The semantic oracle in
    verify.py asserts every non-null value appears in the source text within
    the card region.
    """

    page: int = Field(..., ge=1, description="1-indexed source page number")
    card_bbox_px: list[int] = Field(
        ..., min_length=4, max_length=4,
        description="[x1, y1, x2, y2] of the card in page-render pixel coords",
    )
    photo_bbox_px: Optional[list[int]] = Field(
        None, min_length=4, max_length=4,
        description="[x1, y1, x2, y2] of the product photo within the card (null if no photo)",
    )

    # Workbook-row fields (column letter in comment)
    sku: str = Field(..., description="Style number / SKU as printed (col B)")
    brand: Optional[str] = Field(
        None,
        description="Vendor brand on the card (e.g. ANODYNE, APEX) — used for per-tab split",
    )
    description: str = Field(..., description="Product model name (col F)")
    color: str = Field(
        ...,
        description="Vendor color string verbatim; slash-separated multi-tokens (col G)",
    )

    # Normalized vocabulary fields — must be in their closed Literal
    mg: Optional[MG] = Field(None, description="Merchandise group (col C)")
    sg: Optional[SG] = Field(None, description="Sub-group (col D)")
    ssg: Optional[SSG] = Field(None, description="Sub-sub-group (col E)")
    standard_color: Optional[StandardColor] = Field(
        None, description="Canonical color from vocab (col H)"
    )
    intro_date: Optional[Month] = Field(
        None, description="Intro month code JAN-DEC (col P)"
    )

    # Pricing
    usd_cost: Optional[float] = Field(None, ge=0, description="USD cost (col V)")
    usd_retail: Optional[float] = Field(None, ge=0, description="USD retail (col W)")
