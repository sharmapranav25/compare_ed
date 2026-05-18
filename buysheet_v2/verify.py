"""Semantic oracle for extracted ProductCards.

The single correctness gate: every non-null (sku, field, value) on a card MUST
appear in the source page text within the card region. If not, the value is
either wrong (extraction mis-attributed across cards — the off-by-one class
of bug) or hallucinated.

The card region is bounded by [previous_sku_end, this_sku_end + tail_chars]
in the source text. This is the same per-card window the eval harness uses
for ground-truth comparison.

Confidence levels:
  1.0  value verified — string literally appears in the source card text
  0.7  value derived from vocab lookup AND the input string appears in source
       (e.g. standard_color="Black" because raw color "Black /Hyper Pink"
       appears in source and the vocab map confirmed the canonical form)
  0.5  value present but couldn't be verified (image-only page with empty
       text layer, or non-text-derivable like usd_cost when format differs)
  0.0  value contradicts source — appears in a DIFFERENT card's region; the
       value is HIDDEN from xlsx with raw preserved in cell comment

This is the missing correctness layer the v1 pipeline never had.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Optional

from buysheet_v2.schemas.card import ProductCard
from buysheet_v2.schemas.extraction_result import CardConfidence, CatalogExtraction

VOCAB_DIR = Path(__file__).parent / "vocab"


def _load_color_synonyms() -> dict[str, str]:
    raw = json.loads((VOCAB_DIR / "color_synonyms.json").read_text())
    return {k.lower(): v for k, v in raw.items()}


def _load_description_map() -> dict[str, dict]:
    return json.loads((VOCAB_DIR / "description_map.json").read_text())


_COLOR_SYNONYMS: Optional[dict[str, str]] = None
_DESCRIPTION_MAP: Optional[dict[str, dict]] = None


def color_synonyms() -> dict[str, str]:
    global _COLOR_SYNONYMS
    if _COLOR_SYNONYMS is None:
        _COLOR_SYNONYMS = _load_color_synonyms()
    return _COLOR_SYNONYMS


def description_map() -> dict[str, dict]:
    global _DESCRIPTION_MAP
    if _DESCRIPTION_MAP is None:
        _DESCRIPTION_MAP = _load_description_map()
    return _DESCRIPTION_MAP


# --- text helpers ------------------------------------------------------------

def _normalize_for_match(s: str) -> str:
    """Lowercase + strip ALL whitespace + drop slash-spacing variations.

    Aggressive normalization is intentional. Vendor PDFs (especially PPT-export
    catalogs like Converse) often lose inter-word whitespace during text
    extraction — "CHUCK 70" in the VLM render becomes "CHUCK70" in the text
    layer. By stripping all whitespace on both sides we accept either form as
    a match. Product strings are long enough that false-positive substring
    matches are vanishingly rare.

    Also folds typographic ligatures and accented characters: PyMuPDF returns
    PDF glyphs verbatim, so "Diﬀused" (U+FB00) and "Paciﬁc" (U+FB01) come
    through with single-codepoint ligatures, while the VLM normalizes them
    back to ASCII. NFKD decomposition expands those into their constituent
    ASCII characters; stripping combining marks then collapses accents too.
    """
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = s.lower()
    s = re.sub(r"\s*/\s*", "/", s)
    s = re.sub(r"\s+", "", s)
    return s.strip()


def find_sku_offset(text: str, sku: str) -> Optional[tuple[int, int]]:
    """Whitespace-tolerant SKU search in source text."""
    pat = re.compile(r"\s*".join(re.escape(c) for c in sku))
    m = pat.search(text)
    return (m.start(), m.end()) if m else None


def _sku_prefix(sku: str) -> str:
    """SKU prefix used to group colorway siblings.

    For SKUs in `BASE-COLOR` form (Hoka `1110518-BBLC`, Nike `JA1013-100`)
    returns `BASE`. For SKUs with no dash, returns the whole SKU — those
    cards have no siblings to share text with, so the function caller
    naturally degrades to the per-card region.
    """
    return sku.rsplit("-", 1)[0] if "-" in sku else sku


def sibling_section_region(
    page_text: str, sku: str, all_skus_on_page: list[str],
    head_chars: int = 200, tail_chars: int = 200,
) -> Optional[str]:
    """Return text spanning this SKU's prefix family on the page.

    Lookbook catalogs (Hoka, some Nike sections) print one description
    above a block of colorway siblings — `BONDI 7` then 7 SKUs that all
    share `1110518-*`. Each individual SKU's per-card region won't
    contain `BONDI 7` because the header sits before the first sibling.
    This helper widens the search to "everything from the earliest sibling
    minus a head buffer, through the last sibling plus a tail buffer," so
    a shared header description verifies.

    Returns None when there are no siblings (single-SKU silhouette), so
    callers can fall back to the per-card region instead.
    """
    prefix = _sku_prefix(sku)
    siblings = [
        s for s in all_skus_on_page
        if s != sku and _sku_prefix(s) == prefix
    ]
    if not siblings:
        return None
    positions: list[tuple[int, int]] = []
    for s in [sku, *siblings]:
        occ = find_sku_offset(page_text, s)
        if occ is not None:
            positions.append(occ)
    if not positions:
        return None
    start = max(0, min(p[0] for p in positions) - head_chars)
    end = max(p[1] for p in positions) + tail_chars
    return page_text[start:end]


def card_text_region(
    page_text: str, sku: str, all_skus_on_page: list[str], tail_chars: int = 200
) -> Optional[str]:
    """Return the text region belonging to this SKU's card.

    Bounded by [previous_sku_end_in_text, this_sku_end + tail_chars].
    "Previous SKU" is the SKU in source order that ends just before this one.
    """
    occ = find_sku_offset(page_text, sku)
    if occ is None:
        return None
    start, end = occ
    prev_ends = []
    for other in all_skus_on_page:
        if other == sku:
            continue
        o = find_sku_offset(page_text, other)
        if o and o[1] <= start:
            prev_ends.append(o[1])
    region_start = max(prev_ends) if prev_ends else max(0, start - 250)
    return page_text[region_start : end + tail_chars]


# --- per-field verifiers -----------------------------------------------------

_DOUBLE_LETTER_RE = re.compile(r"([a-z])\1+")


def _collapse_doubles(s: str) -> str:
    """Collapse runs of repeated ASCII letters down to a single letter.

    PyMuPDF's text extraction occasionally inserts phantom-duplicated letters
    on certain PDF font encodings (Adidas FW26 catalog: `RUNWHT` rendered as
    `RUUNWHT`, `wonder` as `wondder`, `quiet` as `quieet`). The model reads
    the visually-rendered glyphs (clean) so its extracted value mismatches
    the duplicated source-text version, even though the buyer's eye sees them
    as the same string. Collapsing doubles on the source-text side recovers
    the match without weakening the per-card region or accepting genuinely
    wrong values: the needle is left unchanged, so any color that's actually
    different will still fail.
    """
    return _DOUBLE_LETTER_RE.sub(r"\1", s)


def _value_in_region(value, region: str) -> bool:
    if value is None or region is None:
        return False
    needle = _normalize_for_match(str(value))
    haystack = _normalize_for_match(region)
    if needle in haystack:
        return True
    # Fallback: try the doubled-letter-collapsed haystack to absorb PDF text
    # extraction artifacts. Needle stays as-is so this only helps when the
    # source text has phantom-duplicated letters, not when the value itself
    # differs from source.
    return needle in _collapse_doubles(haystack)


def _numeric_in_region(value, region: str) -> bool:
    if value is None or region is None:
        return False
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    # Match both "125" and "125.00" formats
    for fmt in (f"{v:.2f}", f"{v:.0f}", f"{int(v)}" if v.is_integer() else None):
        if fmt and fmt in region:
            return True
    return False


def _derive_mg(description: Optional[str], region: Optional[str] = None) -> Optional[str]:
    """Derive MG from the SKU's own description tokens. Returns None if no signal.

    IMPORTANT: only the per-card description is used as a signal, NOT the page
    region. Earlier versions also scanned the region, but that produces false
    positives on pages where men's and women's SKUs share a region — e.g. Nike
    page 1 has "WMNS AIR FORCE 1" sibling SKUs in the same text block as men's
    Air Force 1 SKUs, so region-scanning falsely tags every men's SKU as W.
    The `region` parameter is kept for backward compatibility but ignored.
    """
    _ = region  # intentionally unused; see docstring
    if not description:
        return None
    upper = description.upper()
    # Strong prefix tokens — match within the description only
    if re.search(r"\b(WMNS|WOMEN'S|WOMENS|WOMEN)\b", upper):
        return "W-Footwear"
    if re.match(r"^W\s+[A-Z]", upper):  # description STARTS with "W "
        return "W-Footwear"
    if re.search(r"\b(MENS|MEN'S|MALE)\b", upper):
        return "M-Footwear"
    if re.match(r"^M\s+[A-Z]", upper):  # description STARTS with "M "
        return "M-Footwear"
    if re.search(r"\b(KIDS|YOUTH|CHILD|TODDLER|INFANT|\bGS\b|\bPS\b|\bTD\b)\b", upper):
        return "K-Footwear"
    if "UNISEX" in upper:
        return "K-Footwear"
    return None


_NUMERIC_DATE_TO_MONTH = {
    "01": "JAN", "02": "FEB", "03": "MAR", "04": "APR",
    "05": "MAY", "06": "JUN", "07": "JUL", "08": "AUG",
    "09": "SEP", "10": "OCT", "11": "NOV", "12": "DEC",
    "1": "JAN", "2": "FEB", "3": "MAR", "4": "APR",
    "5": "MAY", "6": "JUN", "7": "JUL", "8": "AUG",
    "9": "SEP",
}


def _intro_date_numeric_match(month_code: str, region: str) -> bool:
    """True if a numeric date in the region resolves to the given month code.

    Covers the date formats that real vendor catalogs actually use:
      MM/DD/YYYY  (Nike — "10/01/2026")
      YYYY-MM-DD  (ISO; less common but cheap to support)
      MM/YYYY     (some matrix catalogs)
      MM/DD       (Hoka lookbook — "01/01 (Core)", no year)
      full month  ("January", "Jan 2027")
    """
    code = month_code.upper()
    # MM/DD/YYYY pattern (most specific — try first)
    for m in re.finditer(r"\b(\d{1,2})/(\d{1,2})/\d{2,4}\b", region):
        if _NUMERIC_DATE_TO_MONTH.get(m.group(1)) == code:
            return True
    # YYYY-MM-DD pattern
    for m in re.finditer(r"\b\d{4}-(\d{1,2})-\d{1,2}\b", region):
        if _NUMERIC_DATE_TO_MONTH.get(m.group(1)) == code:
            return True
    # MM/YYYY pattern
    for m in re.finditer(r"\b(\d{1,2})/(20\d{2})\b", region):
        if _NUMERIC_DATE_TO_MONTH.get(m.group(1)) == code:
            return True
    # Bare MM/DD pattern (no year) — Hoka and other lookbook catalogs use
    # this for seasonal intro dates. We require both components to fall in
    # valid date ranges to limit false positives against random number pairs
    # (sizes use "7-13" with a dash, not a slash, so collision risk is low).
    for m in re.finditer(r"\b(\d{1,2})/(\d{1,2})\b", region):
        mm, dd = m.group(1), m.group(2)
        if (1 <= int(mm) <= 12 and 1 <= int(dd) <= 31
                and _NUMERIC_DATE_TO_MONTH.get(mm) == code):
            return True
    # Bare month names
    full_names = {"JANUARY": "JAN", "FEBRUARY": "FEB", "MARCH": "MAR",
                  "APRIL": "APR", "MAY": "MAY", "JUNE": "JUN", "JULY": "JUL",
                  "AUGUST": "AUG", "SEPTEMBER": "SEP", "OCTOBER": "OCT",
                  "NOVEMBER": "NOV", "DECEMBER": "DEC"}
    for full, abbr in full_names.items():
        if abbr == code and full in region.upper():
            return True
    return False


def _silhouette_lookup(description: Optional[str]) -> Optional[dict]:
    """Match description against silhouette_ssg_map.json (longest-substring wins)."""
    if not description:
        return None
    smap = _silhouette_map_cache()
    if not smap:
        return None
    desc_upper = description.upper()
    best = None
    best_len = 0
    for silhouette, mapping in smap.items():
        if silhouette.upper() in desc_upper and len(silhouette) > best_len:
            best = mapping
            best_len = len(silhouette)
    return best


_SILHOUETTE_MAP: Optional[dict] = None


def _silhouette_map_cache() -> dict:
    global _SILHOUETTE_MAP
    if _SILHOUETTE_MAP is None:
        path = VOCAB_DIR / "silhouette_ssg_map.json"
        try:
            raw = json.loads(path.read_text())
            _SILHOUETTE_MAP = raw.get("map", {}) if isinstance(raw, dict) and "map" in raw else raw
        except (FileNotFoundError, json.JSONDecodeError):
            _SILHOUETTE_MAP = {}
    return _SILHOUETTE_MAP


def _lookup_standard_color(raw_color: str) -> Optional[str]:
    """Walk the color synonym map: full string -> first slash token -> first word."""
    syn = color_synonyms()
    s = raw_color.lower().strip()
    if s in syn:
        return syn[s]
    first_slash = s.split("/")[0].strip()
    if first_slash and first_slash in syn:
        return syn[first_slash]
    first_word = first_slash.split()[0] if first_slash else ""
    if first_word and first_word in syn:
        return syn[first_word]
    return None


def verify_card(
    card: ProductCard,
    page_text: str,
    all_skus_on_page: list[str],
    *,
    catalog_brand: Optional[str] = None,
) -> CardConfidence:
    """Verify every non-null field on a card against its source card region.

    catalog_brand: if set, this brand is implicit at the catalog level (single-brand
    catalogs like Nike, Adidas). The brand field is NOT checked against per-card
    text for these — only multi-brand catalogs require per-card brand presence.
    """
    region = card_text_region(page_text, card.sku, all_skus_on_page)
    sku_in_text = region is not None and find_sku_offset(page_text, card.sku) is not None

    per_field = {}
    per_field_source = {}
    flags = []

    # SKU itself: high confidence if found in text, lower if not
    if sku_in_text:
        per_field["sku"] = 1.0
        per_field_source["sku"] = "vlm+text_anchored"
    else:
        per_field["sku"] = 0.5
        per_field_source["sku"] = "vlm_only"
        flags.append(f"sku '{card.sku}' not found in page {card.page} text")

    # Description + color: verified by source-text substring match
    for f, val in [("description", card.description), ("color", card.color)]:
        if val is None:
            continue
        # Extraction-error check: VLM sometimes copies the description into the
        # color field on cards where the color text is hard to spot. Flag this
        # explicitly so the buyer sees amber + provenance, since the value IS
        # in the source but in the WRONG field.
        if (f == "color" and card.description and
                str(val).strip().lower() == card.description.strip().lower()):
            per_field[f] = 0.0
            per_field_source[f] = "vlm_extraction_error_color_eq_description"
            flags.append(
                f"color value equals description for {card.sku} — "
                f"likely VLM extraction error (no color text visible on card)"
            )
            continue
        if region is None:
            per_field[f] = 0.5
            per_field_source[f] = "vlm_only_no_region"
            continue
        if _value_in_region(val, region):
            per_field[f] = 1.0
            per_field_source[f] = "vlm+oracle_confirmed"
        elif f == "description":
            # Lookbook catalogs (Hoka, some Nike sections) print one
            # description above a block of colorway siblings — the description
            # is shared across e.g. all `1110518-*` SKUs. Each sibling's
            # per-card region won't contain it, so widen to the sibling
            # section. Color must stay per-card (each SKU has its own).
            section = sibling_section_region(page_text, card.sku, all_skus_on_page)
            if section is not None and _value_in_region(val, section):
                per_field[f] = 0.7
                per_field_source[f] = "sibling_section_shared"
            else:
                per_field[f] = 0.0
                per_field_source[f] = "vlm_contradicts_source"
                flags.append(
                    f"description value not in card region or sibling section "
                    f"for {card.sku}: value={val!r}"
                )
        else:
            per_field[f] = 0.0
            per_field_source[f] = "vlm_contradicts_source"
            flags.append(
                f"{f} value not in card region for {card.sku}: "
                f"value={val!r}  (cross-card mis-attribution?)"
            )

    # Brand: only check per-card text if this is a multi-brand catalog. For
    # single-brand catalogs the brand is implicit and not repeated per card.
    if card.brand is not None:
        if catalog_brand is not None and catalog_brand.lower() == card.brand.lower():
            per_field["brand"] = 0.7
            per_field_source["brand"] = "catalog_level_implicit"
        elif region is None:
            per_field["brand"] = 0.5
            per_field_source["brand"] = "vlm_only_no_region"
        elif _value_in_region(card.brand, region):
            per_field["brand"] = 1.0
            per_field_source["brand"] = "vlm+oracle_confirmed"
        elif _value_in_region(card.brand, page_text):
            # Multi-brand catalogs (e.g. SPS, Hanger Clinic) often print the
            # brand once as a page header rather than per card. The VLM reads
            # the header and applies it to every SKU on the page — semantically
            # correct, but the per-card region won't contain the brand string.
            # Accept page-level presence at amber confidence.
            per_field["brand"] = 0.7
            per_field_source["brand"] = "page_level_implicit"
        else:
            per_field["brand"] = 0.0
            per_field_source["brand"] = "vlm_contradicts_source"
            flags.append(f"brand value not in card region for {card.sku}: value={card.brand!r}")

    # Numeric fields: compare numerically to region content
    for f, v in [("usd_cost", card.usd_cost), ("usd_retail", card.usd_retail)]:
        if v is None:
            continue
        if region is None:
            per_field[f] = 0.5
            per_field_source[f] = "vlm_only_no_region"
            continue
        if _numeric_in_region(v, region):
            per_field[f] = 1.0
            per_field_source[f] = "vlm+oracle_confirmed"
        else:
            per_field[f] = 0.0
            per_field_source[f] = "vlm_contradicts_source"
            flags.append(f"{f} value {v} not in card region for {card.sku}")

    # standard_color: prefer vocab lookup confirmation; fall back to "color was in
    # source so standard_color is a reasonable VLM derivation"
    if card.standard_color is not None:
        if card.color is None:
            per_field["standard_color"] = 0.5
            per_field_source["standard_color"] = "vlm_only_no_color"
        else:
            mapped = _lookup_standard_color(card.color)
            color_ok_in_region = (region is not None) and _value_in_region(card.color, region)
            if mapped is not None and mapped != "IGNORE" and mapped == card.standard_color:
                per_field["standard_color"] = 1.0 if color_ok_in_region else 0.7
                per_field_source["standard_color"] = "vocab_lookup_match"
            elif color_ok_in_region:
                per_field["standard_color"] = 0.5
                per_field_source["standard_color"] = "vlm_color_ok_std_unverified"
            else:
                per_field["standard_color"] = 0.0
                per_field_source["standard_color"] = "vlm_unverified"

    # MG: derive from description prefix / gender_vocab tokens; confirm VLM.
    # Crucial: when source has NO explicit gender signal, trust the VLM (it can
    # see the photo and read context the text doesn't carry). Only mark as
    # contradiction when the source clearly says one gender and VLM says another.
    if card.mg is not None:
        derived = _derive_mg(card.description, region)
        if derived is None:
            per_field["mg"] = 0.7
            per_field_source["mg"] = "vlm_vocab_constrained_no_signal"
        elif derived == card.mg:
            per_field["mg"] = 1.0
            per_field_source["mg"] = "vocab_derived_match"
        else:
            per_field["mg"] = 0.0
            per_field_source["mg"] = "vlm_contradicts_vocab"
            flags.append(f"mg {card.mg} contradicts vocab-derived {derived} for {card.sku}")

    # SG / SSG: try description_map cache first, then silhouette_ssg_map.
    # Trust hierarchy:
    #   1.0  cache and VLM agree
    #   0.7  no cache exists OR cache + VLM disagree but both picked valid vocab
    #        (closed-vocabulary Literal validated both — disagreement is a
    #         legitimate categorical ambiguity, not "wrong"; e.g.
    #         SUPERSTAR -> "Court" or "Causal Shoe" both defensible)
    #   0.5  unreachable in this path (VLM-only-no-cache also passes at 0.7)
    if card.sg is not None or card.ssg is not None:
        dm = description_map().get(card.description.strip()) if card.description else None
        silhouette = _silhouette_lookup(card.description) if card.description else None
        for f in ("sg", "ssg"):
            v = getattr(card, f, None)
            if v is None:
                continue
            cached = None
            if dm and dm.get(f):
                cached = dm[f]
            elif silhouette and silhouette.get(f):
                cached = silhouette[f]
            if cached is None:
                per_field[f] = 0.7
                per_field_source[f] = "vlm_vocab_constrained"
            elif cached == v:
                per_field[f] = 1.0
                per_field_source[f] = "cache_match"
            else:
                per_field[f] = 0.7
                per_field_source[f] = "cache_vlm_both_valid"

    # intro_date: must appear in source as month code OR as numeric date that maps to that month
    if card.intro_date is not None:
        if region is None:
            per_field["intro_date"] = 0.5
            per_field_source["intro_date"] = "vlm_only_no_region"
        elif card.intro_date.upper() in region.upper():
            per_field["intro_date"] = 1.0
            per_field_source["intro_date"] = "month_code_in_source"
        elif _intro_date_numeric_match(card.intro_date, region):
            per_field["intro_date"] = 1.0
            per_field_source["intro_date"] = "numeric_date_in_source"
        else:
            per_field["intro_date"] = 0.0
            per_field_source["intro_date"] = "vlm_contradicts_source"
            flags.append(f"intro_date {card.intro_date} not in card region for {card.sku}")

    # Overall: average of per-field scores
    overall = (sum(per_field.values()) / len(per_field)) if per_field else 0.0

    return CardConfidence(
        sku=card.sku, page=card.page,
        per_field=per_field, per_field_source=per_field_source,
        overall=overall, flags=flags,
    )


def _detect_catalog_brand(extraction: CatalogExtraction) -> Optional[str]:
    """If 70%+ of cards share the same brand, return it as the catalog-level brand."""
    from collections import Counter
    brands = [c.brand for c in extraction.all_cards if c.brand]
    if not brands:
        return None
    counter = Counter(b.lower() for b in brands)
    top, n = counter.most_common(1)[0]
    if n / len(brands) >= 0.7:
        # Return the canonical capitalization (first occurrence)
        for b in brands:
            if b.lower() == top:
                return b
    return None


def verify_catalog(extraction: CatalogExtraction) -> CatalogExtraction:
    """Run the oracle on every extracted card. Returns a new extraction with
    the confidence list rebuilt from real oracle scores instead of stub_confidence.
    """
    catalog_brand = _detect_catalog_brand(extraction)

    # Build per-page SKU index
    by_page: dict[int, list[str]] = {}
    for p in extraction.pages:
        by_page[p.page] = [c.sku for c in p.cards]

    new_confidence: list[CardConfidence] = []
    for p in extraction.pages:
        for card in p.cards:
            conf = verify_card(
                card, p.page_text or "", by_page.get(p.page, []),
                catalog_brand=catalog_brand,
            )
            new_confidence.append(conf)

    extraction.confidence = new_confidence
    return extraction


def oracle_summary(extraction: CatalogExtraction) -> dict:
    """Aggregate confidence stats over the catalog."""
    n_cards = len(extraction.confidence)
    if n_cards == 0:
        return {"cards": 0}
    # Thresholds:
    #   PASSING = >= 0.7  (literal match OR vocab-confirmed OR catalog-level implicit)
    #   CONTRADICTED = <= 0.0  (VLM value contradicts source — must be reviewed)
    #   UNCERTAIN = 0.0 < x < 0.7  (VLM-only, no oracle confirmation possible)
    PASS_THRESHOLD = 0.7
    per_field_correct: dict[str, int] = {}
    per_field_total: dict[str, int] = {}
    per_field_contradicted: dict[str, int] = {}
    cards_perfect = 0
    cards_partial = 0
    cards_contradicted = 0
    for c in extraction.confidence:
        all_pass = True
        any_zero = False
        for f, score in c.per_field.items():
            per_field_total[f] = per_field_total.get(f, 0) + 1
            if score >= PASS_THRESHOLD:
                per_field_correct[f] = per_field_correct.get(f, 0) + 1
            else:
                all_pass = False
            if score <= 0.0:
                per_field_contradicted[f] = per_field_contradicted.get(f, 0) + 1
                any_zero = True
        if all_pass:
            cards_perfect += 1
        elif any_zero:
            cards_contradicted += 1
        else:
            cards_partial += 1
    return {
        "cards": n_cards,
        "cards_perfect": cards_perfect,
        "cards_partial": cards_partial,
        "cards_contradicted": cards_contradicted,
        "per_field": {
            f: {
                "correct": per_field_correct.get(f, 0),
                "contradicted": per_field_contradicted.get(f, 0),
                "total": per_field_total[f],
                "rate": per_field_correct.get(f, 0) / per_field_total[f],
            }
            for f in per_field_total
        },
    }
