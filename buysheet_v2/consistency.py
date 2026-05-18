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


# --- Deterministic source-text backfill --------------------------------------
#
# The VLM occasionally skips fields for some siblings in a dense repeating
# section (Hoka 1168973-BBLC's $170 and 01/01 are right there in the source
# but the model only extracted them for the next sibling, -CWBT). Since the
# data is unambiguously in source, we can backfill the missing values via the
# same regex patterns the semantic oracle uses for verification. This is
# strictly an upgrade-when-VLM-was-None operation; existing extracted values
# are never overwritten.

_NUMERIC_MONTH_TO_CODE = {
    "01": "JAN", "1": "JAN", "02": "FEB", "2": "FEB",
    "03": "MAR", "3": "MAR", "04": "APR", "4": "APR",
    "05": "MAY", "5": "MAY", "06": "JUN", "6": "JUN",
    "07": "JUL", "7": "JUL", "08": "AUG", "8": "AUG",
    "09": "SEP", "9": "SEP", "10": "OCT", "11": "NOV", "12": "DEC",
}
_PRICE_RE = re.compile(r"\$\s*(\d{1,4}(?:\.\d{1,2})?)")
_DATE_MMDDYYYY = re.compile(r"\b(\d{1,2})/(\d{1,2})/\d{2,4}\b")
_DATE_MMYYYY = re.compile(r"\b(\d{1,2})/(20\d{2})\b")
# Bare MM/DD that is NOT inside a size-range list. Negative lookahead blocks
# `MM/DD-...`, `MM/DD/...`, and `MM/DD, MM/DD` (the size patterns Hoka uses
# like `03/05-14/16, 13/14, 14/15`). Real intro dates like `01/01 (Core)` and
# `01/15 NEW` have non-digit/non-separator characters immediately after, so
# the lookahead lets them through.
_DATE_MMDD = re.compile(r"\b(\d{1,2})/(\d{1,2})\b(?!\s*[-/,]?\s*\d)")
_DATE_FULL_NAMES = {
    "JANUARY": "JAN", "FEBRUARY": "FEB", "MARCH": "MAR", "APRIL": "APR",
    "MAY": "MAY", "JUNE": "JUN", "JULY": "JUL", "AUGUST": "AUG",
    "SEPTEMBER": "SEP", "OCTOBER": "OCT", "NOVEMBER": "NOV", "DECEMBER": "DEC",
}


def _backfill_usd_cost(region: str) -> Optional[float]:
    """Return the first `$N` price token in the region, parsed as float."""
    m = _PRICE_RE.search(region)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _backfill_intro_date(region: str) -> Optional[str]:
    """Return a month-resolvable date token from the region as a MONTH code.

    For bare MM/DD patterns (Hoka-style lookbook), the date appears AFTER size
    ranges in the row (`... 05/06-12/13, 13/14, 14/15 M/W $170 ... 01/01 (Core)`).
    Picking the first MM/DD match would grab the size range's leading pair
    (05/06 → MAY) instead of the intro date (01/01 → JAN), so we take the LAST
    MM/DD match instead. MMDDYYYY catches Nike-style fully-qualified dates
    before we get to the bare-MMDD fallback, so this never overrides them.
    """
    for m in _DATE_MMDDYYYY.finditer(region):
        code = _NUMERIC_MONTH_TO_CODE.get(m.group(1))
        if code:
            return code
    for m in _DATE_MMYYYY.finditer(region):
        code = _NUMERIC_MONTH_TO_CODE.get(m.group(1))
        if code:
            return code
    # Bare MM/DD: take the LAST valid date token (dates follow sizes in
    # vendor catalog rows). Filter to plausible date components first.
    bare_matches: list[str] = []
    for m in _DATE_MMDD.finditer(region):
        mm, dd = m.group(1), m.group(2)
        if 1 <= int(mm) <= 12 and 1 <= int(dd) <= 31:
            bare_matches.append(mm)
    if bare_matches:
        return _NUMERIC_MONTH_TO_CODE.get(bare_matches[-1])
    upper = region.upper()
    for full, abbr in _DATE_FULL_NAMES.items():
        if full in upper:
            return abbr
    return None


def deterministic_fill(
    cards: list[ProductCard], page_text_by: dict[int, str],
    *, catalog_brand: Optional[str] = None,
) -> dict[str, int]:
    """Backfill structured fields the VLM left None, using source-text regex.

    For each card with a None field, locates the card's text region (same
    boundary logic the oracle uses) and applies a deterministic regex. Only
    fills when the regex match is unambiguous; never overwrites a non-None
    VLM value. Counts per-field fills returned for caller logging.

    Backfills four classes of field:
      - usd_cost / intro_date: regex over the SKU's source-text region
        (Hoka 1168973-BBLC and Nike P-6000 IX7928-002 class — VLM skipped
        the field on dense pages but the data is plainly in source)
      - brand: when VLM left None and the catalog has a single dominant
        brand, fill from the catalog-level signal (Nike PPTX where 80%
        of cards had brand=None)
      - standard_color: when VLM left None but card.color is set, run the
        same vocab lookup the oracle uses for verification (Adidas where
        82 cards had a color but no derived canonical)
    """
    # Import locally to avoid pulling verify at module-import time (the
    # oracle has its own slow imports we don't want eager).
    from buysheet_v2.verify import _lookup_standard_color, card_text_region

    skus_by_page: dict[int, list[str]] = defaultdict(list)
    for c in cards:
        skus_by_page[c.page].append(c.sku)

    counts: dict[str, int] = defaultdict(int)
    for card in cards:
        page_text = page_text_by.get(card.page) or ""
        region = None
        if page_text:
            region = card_text_region(page_text, card.sku, skus_by_page[card.page])

        # Source-text regex fills
        if region is not None:
            if card.usd_cost is None:
                v = _backfill_usd_cost(region)
                if v is not None:
                    card.usd_cost = v
                    counts["usd_cost"] += 1
            if card.intro_date is None:
                v = _backfill_intro_date(region)
                if v is not None:
                    card.intro_date = v
                    counts["intro_date"] += 1

        # Brand backfill — fires regardless of region availability since the
        # catalog-level signal doesn't depend on per-card text.
        if card.brand is None and catalog_brand is not None:
            card.brand = catalog_brand
            counts["brand"] += 1

        # standard_color backfill via vocab lookup. Cheap and high-precision:
        # the same lookup the oracle uses, just applied as a fill rather than
        # only as a check.
        if card.standard_color is None and card.color:
            mapped = _lookup_standard_color(card.color)
            if mapped is not None and mapped != "IGNORE":
                card.standard_color = mapped
                counts["standard_color"] += 1

    return dict(counts)


def detect_catalog_brand(cards: list[ProductCard]) -> Optional[str]:
    """If 70%+ of cards share the same brand, return it (case-preserved).

    Used by deterministic_fill so the brand backfill applies the canonical
    display form for the catalog (Nike, not nike) when VLM-extracted cards
    use mixed casing. Returns None for multi-brand or all-blank catalogs.
    """
    brands = [c.brand.strip() for c in cards if c.brand and c.brand.strip()]
    if not brands:
        return None
    counter = Counter(b.lower() for b in brands)
    dominant_lower, count = counter.most_common(1)[0]
    if count < 0.7 * len(brands):
        return None
    for b in brands:
        if b.lower() == dominant_lower:
            return b  # first canonical-casing occurrence
    return None


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
