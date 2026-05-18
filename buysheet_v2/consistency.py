"""Cross-card consistency: SKU dedup, brand voting, multi-brand workbook split.

Runs after extract/verify, before write. Operates on the full extracted card
list to enforce catalog-wide invariants:

  - SKU uniqueness: warn if two cards have the same SKU
  - Brand voting per SKU prefix family (Adidas KK*, Nike IX/JA*) so a single
    miscategorized card doesn't drift its sibling SKUs
  - Multi-brand partition: when ≥2 distinct brands appear, split into per-brand
    bucket lists for write.py to render as separate worksheet tabs

This is the "deterministic backstop" layer in the architecture — pure Python,
no API calls.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Optional

from buysheet_v2.schemas.card import ProductCard

# Tokens that signal a kids product when present in the description (word-boundary
# matched to avoid spurious substrings like "JR" inside "AJREN"). Adult-default
# Kith convention treats anything without one of these as M-Footwear unless the
# VLM explicitly saw women's signal (WMNS / WOMEN's etc — handled by _derive_mg).
_KIDS_DESC_PATTERN = re.compile(
    r"\b(GS|PS|TD|YOUTH|KIDS?|JR|JUNIOR|BABY|INFANT|TODDLER|CHILD|CHILDREN|"
    r"BOYS?|GIRLS?|YTH)\b",
    flags=re.IGNORECASE,
)
# SKU-side kids markers — vendor-specific suffixes (e.g. Adidas "-K") and
# common patterns. Conservative: word-boundary matched, only triggers when the
# token appears as a discrete segment of the SKU.
_KIDS_SKU_PATTERN = re.compile(
    r"(?:^|[-_/ ])(K|YTH|JR|GS|PS|TD|JNR)(?:$|[-_/ ])",
    flags=re.IGNORECASE,
)


def normalize_kids_default(card: ProductCard) -> bool:
    """If model picked K-Footwear without a kids signal, override to M-Footwear.

    Returns True if the card was changed (caller may want to log it).

    The VLM occasionally defaults to K-Footwear on dense pages when there's no
    explicit gender cue (e.g. Adidas KI2263 PREDATOR SALA EDIT — a Predator
    Sala futsal shoe with no WMNS/MEN/KIDS token in the catalog text). Kith
    convention defaults adult athletic footwear to M-Footwear; K-Footwear
    should require explicit evidence — either a kids token in the description
    or a kids-coded SKU segment.

    Conservative: only flips K-Footwear, never touches M/W. Safe to run before
    verify_catalog because we only ever upgrade specificity (the resulting
    `_derive_mg` call still returns None on a description with no gender token,
    so the cell still scores `vlm_vocab_constrained_no_signal` at amber 0.7 —
    the user sees a sensible default and a review prompt).
    """
    if card.mg != "K-Footwear":
        return False
    if card.description and _KIDS_DESC_PATTERN.search(card.description):
        return False  # genuine kids signal in description
    if _KIDS_SKU_PATTERN.search(card.sku):
        return False  # SKU is explicitly kids-coded
    card.mg = "M-Footwear"
    return True


def normalize_extraction(cards: list[ProductCard]) -> dict[str, int]:
    """Apply all card-level normalizations in place. Returns counts per fix."""
    counts: dict[str, int] = defaultdict(int)
    for c in cards:
        if normalize_kids_default(c):
            counts["k_footwear_to_m"] += 1
    return dict(counts)


def detect_duplicates(cards: list[ProductCard]) -> list[tuple[str, list[int]]]:
    """Find SKUs that appear on multiple cards. Returns [(sku, [card_indices])]."""
    sku_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(cards):
        sku_to_indices[c.sku].append(i)
    return [(sku, idx) for sku, idx in sku_to_indices.items() if len(idx) > 1]


def vote_brand_by_prefix(
    cards: list[ProductCard], prefix_len: int = 2, min_family_size: int = 3
) -> dict[str, str]:
    """Compute consensus brand per SKU-prefix family.

    Group cards by their SKU's first `prefix_len` characters. For each family
    with ≥min_family_size members, return the brand that ≥70% of the family
    agrees on. Used to repair single-card brand mis-attributions.

    Returns: {sku_prefix: consensus_brand}
    """
    by_prefix: dict[str, list[str]] = defaultdict(list)
    for c in cards:
        if c.brand and len(c.sku) >= prefix_len:
            by_prefix[c.sku[:prefix_len].upper()].append(c.brand)

    consensus: dict[str, str] = {}
    for prefix, brands in by_prefix.items():
        if len(brands) < min_family_size:
            continue
        counter = Counter(b.strip() for b in brands)
        top_brand, top_count = counter.most_common(1)[0]
        if top_count / len(brands) >= 0.7:
            consensus[prefix] = top_brand
    return consensus


def repair_brands(cards: list[ProductCard]) -> int:
    """Overwrite a card's brand with the consensus brand from its SKU-prefix family.

    Returns the number of cards whose brand was corrected.
    """
    consensus = vote_brand_by_prefix(cards)
    if not consensus:
        return 0
    repaired = 0
    for c in cards:
        prefix = c.sku[:2].upper() if len(c.sku) >= 2 else None
        if prefix in consensus and c.brand != consensus[prefix]:
            c.brand = consensus[prefix]
            repaired += 1
    return repaired


def _canonical_brand_key(brand: str) -> str:
    """Normalize a brand for grouping: lowercase + strip + collapse spaces.

    'Adidas', 'adidas', ' ADIDAS ' all map to 'adidas'. This avoids spurious
    multi-tab splits when the VLM returns mixed casing across a single-brand
    catalog. Display names are recovered via _canonical_display_name.
    """
    return " ".join(brand.lower().strip().split())


def _canonical_display_name(canonical_key: str, brand_seen: list[str]) -> str:
    """Pick the most-frequent original capitalization for display in the tab name."""
    matching = [b for b in brand_seen if _canonical_brand_key(b) == canonical_key]
    if not matching:
        return canonical_key
    counter = Counter(matching)
    return counter.most_common(1)[0][0]


def partition_by_brand(
    cards: list[ProductCard], catalog_brand: Optional[str] = None
) -> dict[str, list[ProductCard]]:
    """Split cards by canonicalized brand, returning {display_name: [cards]}.

    Brands are grouped case-insensitively + whitespace-normalized. The display
    name picked for each bucket is the most-frequent original casing seen on
    that brand's cards.

    Cards with brand=None inherit `catalog_brand` if provided, else fall into
    a synthetic "_unbranded" bucket.
    """
    buckets_by_key: dict[str, list[ProductCard]] = defaultdict(list)
    brands_by_key: dict[str, list[str]] = defaultdict(list)
    for c in cards:
        if c.brand and c.brand.strip():
            raw = c.brand.strip()
            key = _canonical_brand_key(raw)
            brands_by_key[key].append(raw)
        elif catalog_brand:
            raw = catalog_brand
            key = _canonical_brand_key(raw)
            brands_by_key[key].append(raw)
        else:
            key = "_unbranded"
        buckets_by_key[key].append(c)
    out: dict[str, list[ProductCard]] = {}
    for key, cards_list in buckets_by_key.items():
        if key == "_unbranded":
            out["_unbranded"] = cards_list
        else:
            display = _canonical_display_name(key, brands_by_key[key])
            out[display] = cards_list
    return out


def is_multi_brand(cards: list[ProductCard], threshold: int = 2) -> bool:
    """True if cards span ≥threshold distinct brand values (canonicalized)."""
    keys = {
        _canonical_brand_key(c.brand)
        for c in cards
        if c.brand and c.brand.strip()
    }
    return len(keys) >= threshold


def normalize_consistency(
    cards: list[ProductCard], *, catalog_brand: Optional[str] = None
) -> dict:
    """One-shot consistency pass: repair brands + return partition + warnings.

    Returns:
        {
          "partitions": {brand_name: [ProductCard]},
          "duplicates": [(sku, [indices])],
          "repaired_brands": int,
          "is_multi_brand": bool,
        }
    """
    repaired = repair_brands(cards)
    duplicates = detect_duplicates(cards)
    multi = is_multi_brand(cards)
    partitions = partition_by_brand(cards, catalog_brand=catalog_brand)
    return {
        "partitions": partitions,
        "duplicates": duplicates,
        "repaired_brands": repaired,
        "is_multi_brand": multi,
    }
