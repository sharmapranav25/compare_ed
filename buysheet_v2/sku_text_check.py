"""Text-layer SKU pre-filter for the photo-bbox matcher.

Mirrors the SKU-presence half of the parent repo's
deterministic_check/verify.py:42-108, stripped down to just the SKU
substring check — no per-field drop/clear/keep policy (that lives in
buysheet_v2/verify.py for field-value verification at the per-card-region
level).

Purpose: drop phantom SKUs from the matcher's input list BEFORE
photo_match.match_skus_to_bboxes sees them. Without this pre-filter the
matcher's first-SKU-wins de-dup (photo_match.py:243-256) lets a
hallucinated SKU claim a real product's numbered box, displacing the
real SKU's assignment.

The card list itself is NOT mutated by this helper — phantoms still
flow through extract → consistency → verify and get caught by their
existing layers. Only the matcher's *input* is filtered.

Falls back gracefully when the text layer is missing or empty: image-
only PDFs return `text_layer_present=False`, the caller skips the
filter, and the matcher sees the full SKU list (parent repo's
deterministic_check/verify.py:68-71 behavior).
"""
from __future__ import annotations

import re
from typing import Iterable


def canon(s: str | None) -> str:
    """Strip all whitespace, uppercase. "" for None / empty input.

    Same canonicalization as parent's deterministic_check/verify.py:42-44.
    Aggressive enough that "A24021 C", "a24021c", and "A24021  C"
    all canonicalize to "A24021C" — substring match either side recovers.
    """
    return re.sub(r"\s+", "", s or "").upper()


def present_skus_in_text(
    page_text: str | None, candidate_skus: Iterable[str],
) -> tuple[set[str], bool]:
    """Return (verbatim_skus_found_in_text, text_layer_present).

    A SKU is "found" iff canon(sku) appears as a substring of canon(page_text).
    The set contains the ORIGINAL SKU strings (verbatim from
    candidate_skus) so callers can use it as a filter against the
    original card list with no SKU rewriting.

    text_layer_present is False when page_text is empty/None after
    canonicalization. Caller should skip the filter in that case so
    image-only PDFs pass through unchanged.
    """
    text_canon = canon(page_text)
    if not text_canon:
        return set(), False
    present: set[str] = set()
    for sku in candidate_skus:
        if not sku:
            continue
        if canon(sku) in text_canon:
            present.add(sku)
    return present, True
