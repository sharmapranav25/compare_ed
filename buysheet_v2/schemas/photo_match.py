"""Pydantic schema for the photo-match VLM call.

The Sonnet matcher in photo_match.py is given an annotated page (with red
numbered boxes drawn by yolo_detect.annotate_page) plus the list of SKUs
extracted from the page text, and returns an Assignment per SKU mapping
to one of the numbered boxes (or null when no match is visible).
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class PhotoMatchAssignment(BaseModel):
    """One SKU → numbered-box assignment."""

    sku: str = Field(..., description="SKU string, echoed verbatim from input")
    box: Optional[int] = Field(
        None,
        description="1-indexed box number from the annotated page, or null "
                    "if the SKU's photo is not visible / not detected.",
    )


class PhotoMatchResponse(BaseModel):
    """Top-level response shape for the matcher VLM call."""

    assignments: list[PhotoMatchAssignment] = Field(default_factory=list)


class DensityResponse(BaseModel):
    """Response shape for the yolo-density picker VLM call."""

    density: str = Field(..., description="One of: low | medium | high")
    reasoning: str = Field("", description="Short explanation (≤12 words)")
