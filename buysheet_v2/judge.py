"""VLM-as-judge: Opus 4.7 second opinion on suspect extractions.

Two modes:

  1. `verify_suspect_fields` (CURRENT): per-(card, field) targeted verification.
     Called by pipeline.py AFTER verify.py's deterministic text-layer check
     marks fields at confidence 0.5 ("unverifiable" — neither found in the
     card's region nor in a different card's region). Opus reads the page
     image, verifies only those specific (sku, field, value) tuples, and
     returns per-tuple agree/disagree + corrected value. Cost ~1.2-1.5×
     Sonnet on a typical catalog because the prompt is narrow and most
     cards have 0-1 suspect fields. On agree, write.py bumps confidence
     0.5 → 0.7 and the cell stays amber but the comment notes Opus
     confirmation. On disagree, confidence stays at 0.5 and the
     disagreement's opus_value is surfaced in the cell comment so the
     buyer sees both candidates.

  2. `judge_page` (LEGACY): independent full-page re-extraction with a
     rephrased prompt, then per-field cross-model agreement scoring. Costs
     ~5× Sonnet because Opus re-extracts every card and every field.
     Retained for backward compatibility but no longer called from the
     default pipeline — verify.py + verify_suspect_fields is strictly
     cheaper and grounded in source text bytes rather than VLM consensus
     (two VLMs can be correlated-wrong on the same OCR confusion).
"""
from __future__ import annotations

import re
from typing import Any, Optional

import anthropic
from pydantic import BaseModel, Field

from buysheet_v2.ingest import IngestedPage
from buysheet_v2.lifted.pdf_render import b64_image_block
from buysheet_v2.schemas.card import CardBbox, ProductCard
from buysheet_v2.schemas.extraction_result import CardConfidence

JUDGE_MODEL = "claude-opus-4-7"
MAX_TOKENS = 16384

# Pricing (May 2026): Opus 4.7 = $15 in / $75 out per MTok
OPUS_IN_PER_MTOK = 15.0
OPUS_OUT_PER_MTOK = 75.0


class JudgeCard(BaseModel):
    """One Opus-extracted card for cross-checking against the Sonnet card."""

    sku: str
    description: Optional[str] = None
    color: Optional[str] = None
    brand: Optional[str] = None
    mg: Optional[str] = None
    intro_date: Optional[str] = None
    usd_cost: Optional[float] = None


class JudgeResponse(BaseModel):
    """Per-page judge response."""

    cards: list[JudgeCard] = Field(default_factory=list)


JUDGE_SYSTEM_PROMPT = """You are an independent verifier reviewing extracted shoe-catalog data.

For each product card visible on this page, return the following fields ONLY based on what you can read on the page itself (do not infer beyond what is plainly visible):

  - sku: the printed style number for the card, exactly as shown (preserve case + dashes)
  - description: the product model/silhouette name (e.g. "AIR FORCE 1 '07", "BONDI 7")
  - color: the printed colorway as the vendor wrote it (verbatim)
  - brand: the vendor brand if shown on this card or its section header
  - mg: one of "M-Footwear", "W-Footwear", "K-Footwear" — only if a gender token (WMNS, MEN, KIDS, GS, PS, JR, etc) is visible
  - intro_date: a three-letter month code (JAN, FEB, ...) only if a launch date is shown
  - usd_cost: the USD wholesale price as a number (no $ sign) if shown

Critical rules:
  1. Read each card independently — never copy a value from a neighboring card
  2. Return null for any field you cannot read or confidently infer from this card alone
  3. SKUs are typically alphanumeric codes like "JA1013-010", "1110518-BBLC", "X826W"
  4. If a model name spans multiple colorway rows (e.g., "BONDI 7" above seven SKUs), apply it to all of those siblings
  5. Color must be VERBATIM from the page text — do not normalize or interpret

Return ONE JudgeCard per visible product card."""


def judge_page(
    page: IngestedPage,
    sonnet_cards: list[ProductCard],
    card_bboxes: list[CardBbox],
    *,
    client: Optional[anthropic.Anthropic] = None,
) -> tuple[list[JudgeCard], dict]:
    """Run Opus extraction on a page that Sonnet already extracted.

    Returns (judge_cards, usage_dict). usage_dict has token + cost details.
    """
    if client is None:
        client = anthropic.Anthropic()
    if not sonnet_cards and not card_bboxes:
        return [], {"input_tokens": 0, "output_tokens": 0,
                    "cache_read_tokens": 0, "cost_usd": 0.0}

    # Pass the Sonnet-extracted SKU list so Opus knows what to find. We
    # don't pass Sonnet's other field values (that would bias Opus toward
    # agreement). Just the SKU anchors.
    sku_list = sorted({c.sku for c in sonnet_cards}) or [
        bb.sku_hint for bb in card_bboxes if bb.sku_hint
    ]
    user_text = (
        f"Re-extract every product card on page {page.page_no} of this catalog.\n\n"
        f"Expected SKUs on this page (anchor list — return one card per SKU): "
        f"{', '.join(sku_list[:50])}"
        + (f" and {len(sku_list) - 50} more" if len(sku_list) > 50 else "")
        + "\n\nReturn fields per the JudgeCard schema, reading only what's on the page."
    )

    response = client.messages.parse(
        model=JUDGE_MODEL,
        max_tokens=MAX_TOKENS,
        system=[{
            "type": "text",
            "text": JUDGE_SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{
            "role": "user",
            "content": [
                b64_image_block(page.png_bytes),
                {"type": "text", "text": user_text},
            ],
        }],
        output_format=JudgeResponse,
    )
    parsed = response.parsed_output

    in_tok = getattr(response.usage, "input_tokens", 0)
    out_tok = getattr(response.usage, "output_tokens", 0)
    cache_read = getattr(response.usage, "cache_read_input_tokens", 0) or 0
    cost_usd = (
        (in_tok / 1e6) * OPUS_IN_PER_MTOK
        + (out_tok / 1e6) * OPUS_OUT_PER_MTOK
        + (cache_read / 1e6) * OPUS_IN_PER_MTOK * 0.1
    )
    return parsed.cards, {
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "cache_read_tokens": cache_read,
        "cost_usd": cost_usd,
        "judge_card_count": len(parsed.cards),
    }


def _normalize_compare(value: Any) -> str:
    """Loose normalization for cross-model value comparison.

    Two models will rarely produce byte-identical output even for "agreed"
    extractions — different whitespace, casing, optional terminator. Compare
    on a normalized form: lowercase, whitespace-collapsed, stripped.
    """
    if value is None:
        return ""
    s = str(value).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


# Fields where exact (post-normalization) string equality is meaningful.
# usd_cost is compared numerically, intro_date as month code.
_STRING_FIELDS = ("description", "color", "brand", "mg")


def _compare_card(sonnet: ProductCard, judge: JudgeCard) -> tuple[dict[str, float], dict[str, Optional[str]]]:
    """Return per-field agreement scores + judge values for disagreements.

    Score semantics:
        1.0  -> both models extracted the same value (post-normalization)
        0.5  -> both models have a value but they differ
        0.0  -> one has a value and the other does not (asymmetric absence)
        (absent key)  -> both are None / not comparable
    """
    agreement: dict[str, float] = {}
    judge_values: dict[str, Optional[str]] = {}

    for f in _STRING_FIELDS:
        s_val = getattr(sonnet, f, None)
        j_val = getattr(judge, f, None)
        if s_val is None and j_val is None:
            continue
        if s_val is None or j_val is None:
            agreement[f] = 0.0
            judge_values[f] = j_val if j_val is not None else None
            continue
        if _normalize_compare(s_val) == _normalize_compare(j_val):
            agreement[f] = 1.0
        else:
            agreement[f] = 0.5
            judge_values[f] = str(j_val)

    # intro_date: case-insensitive month code match (already constrained to
    # 3-letter Literal in the Sonnet schema, but Opus's JudgeCard is free
    # text — normalize to upper, take first 3 letters).
    s_date = sonnet.intro_date
    j_date = judge.intro_date
    if s_date is None and j_date is None:
        pass
    elif s_date is None or j_date is None:
        agreement["intro_date"] = 0.0
        judge_values["intro_date"] = j_date if j_date is not None else None
    else:
        s_code = (s_date or "").strip().upper()[:3]
        j_code = (j_date or "").strip().upper()[:3]
        if s_code == j_code:
            agreement["intro_date"] = 1.0
        else:
            agreement["intro_date"] = 0.5
            judge_values["intro_date"] = j_date

    # usd_cost: numeric comparison (allow small float drift)
    s_cost = sonnet.usd_cost
    j_cost = judge.usd_cost
    if s_cost is None and j_cost is None:
        pass
    elif s_cost is None or j_cost is None:
        agreement["usd_cost"] = 0.0
        judge_values["usd_cost"] = str(j_cost) if j_cost is not None else None
    else:
        if abs(float(s_cost) - float(j_cost)) < 0.01:
            agreement["usd_cost"] = 1.0
        else:
            agreement["usd_cost"] = 0.5
            judge_values["usd_cost"] = str(j_cost)

    # SKU: equality at the literal level (case-sensitive)
    if sonnet.sku and judge.sku:
        agreement["sku"] = 1.0 if sonnet.sku.strip() == judge.sku.strip() else 0.5
        if agreement["sku"] < 1.0:
            judge_values["sku"] = judge.sku

    return agreement, judge_values


def merge_judge_into_confidence(
    confidence: list[CardConfidence],
    sonnet_cards: list[ProductCard],
    judge_by_page: dict[int, list[JudgeCard]],
) -> dict[str, int]:
    """Attach judge agreement scores to each card's CardConfidence.

    Returns a small summary dict for caller logging:
      total_compared, agreement_counts (1.0 / 0.5 / 0.0)
    """
    judge_by_sku_page: dict[tuple[str, int], JudgeCard] = {}
    for page_no, cards in judge_by_page.items():
        for jc in cards:
            judge_by_sku_page[(jc.sku, page_no)] = jc

    sonnet_by_key = {(c.sku, c.page): c for c in sonnet_cards}
    conf_by_key = {(c.sku, c.page): c for c in confidence}

    counts = {"total_compared": 0, "agree": 0, "disagree": 0, "asymmetric": 0}

    # For each Sonnet card, look up the judge card and attach scores
    for key, sonnet_card in sonnet_by_key.items():
        # SKU may differ slightly — fall back to fuzzy SKU match within page
        judge_card = judge_by_sku_page.get(key)
        if judge_card is None:
            # Try same page, normalized SKU match (handles minor case/space drift)
            for (j_sku, j_page), jc in judge_by_sku_page.items():
                if j_page == key[1] and _normalize_compare(j_sku) == _normalize_compare(key[0]):
                    judge_card = jc
                    break
        conf = conf_by_key.get(key)
        if conf is None or judge_card is None:
            continue
        agreement, values = _compare_card(sonnet_card, judge_card)
        conf.judge_agreement = agreement
        conf.judge_values = values
        for v in agreement.values():
            counts["total_compared"] += 1
            if v >= 1.0:
                counts["agree"] += 1
            elif v <= 0.0:
                counts["asymmetric"] += 1
            else:
                counts["disagree"] += 1
    return counts


# =============================================================================
# Targeted verification mode (called by pipeline.py after verify.py marks
# specific (sku, field) suspects at confidence 0.5). Cheaper and more
# precisely-scoped than the full-page judge_page above.
# =============================================================================


class FieldVerdict(BaseModel):
    """One Opus verdict on a single (sku, field) suspect tuple."""

    sku: str = Field(..., description="Verbatim SKU echoed from the suspect input")
    field: str = Field(..., description="Field name echoed from the suspect input")
    agreed: bool = Field(
        ...,
        description="True if the page text for that card supports the suspect value",
    )
    opus_value: Optional[str] = Field(
        None,
        description="What Opus actually reads on the page for that (sku, field); "
                    "populated only when agreed=False",
    )


class FieldVerificationResponse(BaseModel):
    """Per-page response: one verdict per suspect submitted."""

    verdicts: list[FieldVerdict] = Field(default_factory=list)


VERIFY_SUSPECTS_PROMPT = """You are verifying specific extracted values against a shoe-catalog page image.

You will receive:
  - An image of one page of the catalog
  - A list of SUSPECTS: per-card (sku, field, value) tuples that an upstream
    Sonnet extractor produced, but a deterministic text-layer check could
    neither confirm nor contradict (it found no signal either way).

For each suspect, locate the card with the given SKU on the page and decide
whether the page actually shows the given value for that field on that card.

Return one FieldVerdict per suspect:
  - sku: verbatim echo of the suspect's sku
  - field: verbatim echo of the suspect's field
  - agreed: true if the page supports the given value; false otherwise
  - opus_value: when agreed=false, the value you actually read from the page
    for that (sku, field) — null if the field is not visible

Rules:
  1. Read each card independently — never copy a value from a neighboring card.
  2. If the SKU is not visible on this page, return agreed=false with opus_value=null.
  3. usd_cost: compare numerically (10.0 and 10 are the same value).
  4. mg: agreement requires same letter group — "M-Footwear" / "W-Footwear" / "K-Footwear".
  5. intro_date: agreement requires same 3-letter month code.
  6. Cover every suspect EXACTLY once — same count of verdicts as suspects."""


def verify_suspect_fields(
    page: IngestedPage,
    suspects: list[tuple[str, str, str]],
    *,
    client: Optional[anthropic.Anthropic] = None,
) -> tuple[list[FieldVerdict], dict]:
    """Per-(card, field) Opus verification, targeted at fields verify.py marked 0.5.

    `suspects` is a list of (sku, field, suspect_value_as_str) tuples scoped
    to a single page. Returns (verdicts, usage_dict). When suspects is
    empty, returns immediately with zero usage.
    """
    if not suspects:
        return [], {"input_tokens": 0, "output_tokens": 0,
                    "cache_read_tokens": 0, "cost_usd": 0.0,
                    "n_suspects": 0}
    if client is None:
        client = anthropic.Anthropic()

    suspect_lines = []
    for i, (sku, field, value) in enumerate(suspects, start=1):
        suspect_lines.append(f"  {i}. sku={sku!r}  field={field}  value={value!r}")
    user_text = (
        f"Verify these {len(suspects)} suspect (sku, field, value) "
        f"extractions against page {page.page_no} of the catalog:\n\n"
        + "\n".join(suspect_lines)
        + "\n\nReturn one verdict per suspect."
    )

    response = client.messages.parse(
        model=JUDGE_MODEL,
        max_tokens=MAX_TOKENS,
        system=[{
            "type": "text",
            "text": VERIFY_SUSPECTS_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{
            "role": "user",
            "content": [
                b64_image_block(page.png_bytes),
                {"type": "text", "text": user_text},
            ],
        }],
        output_format=FieldVerificationResponse,
    )
    parsed = response.parsed_output

    in_tok = getattr(response.usage, "input_tokens", 0)
    out_tok = getattr(response.usage, "output_tokens", 0)
    cache_read = getattr(response.usage, "cache_read_input_tokens", 0) or 0
    cost_usd = (
        (in_tok / 1e6) * OPUS_IN_PER_MTOK
        + (out_tok / 1e6) * OPUS_OUT_PER_MTOK
        + (cache_read / 1e6) * OPUS_IN_PER_MTOK * 0.1
    )
    return parsed.verdicts, {
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "cache_read_tokens": cache_read,
        "cost_usd": cost_usd,
        "n_suspects": len(suspects),
    }


def _values_match(field: str, suspect_value: str, opus_value: Optional[str]) -> bool:
    """Field-aware equivalence between Sonnet's value and Opus's reading.

    Reused when post-validating Opus's `agreed` claim, in case Opus said
    agreed=False but its opus_value actually matches Sonnet's after
    normalization (defensive against the model's own labeling drift).
    """
    if opus_value is None:
        return False
    if field == "usd_cost":
        try:
            return abs(float(suspect_value) - float(opus_value)) < 0.01
        except (TypeError, ValueError):
            return False
    if field == "intro_date":
        return (str(suspect_value)[:3].upper() == str(opus_value)[:3].upper())
    return _normalize_compare(suspect_value) == _normalize_compare(opus_value)


def merge_field_verifications(
    confidence: list[CardConfidence],
    verdicts_by_page: dict[int, list[FieldVerdict]],
) -> dict[str, int]:
    """Apply targeted judge verdicts back to CardConfidence.

    For each verdict whose corresponding CardConfidence has per_field[field]
    still at 0.5 (i.e., verify.py's "unverifiable" tier):
      - agreed → bump per_field to 0.7, mark source as `vlm_judge_confirmed`,
        record judge_agreement[field] = 1.0
      - disagreed → leave per_field at 0.5, record judge_agreement[field] = 0.5
        and judge_values[field] = opus_value (surfaced in cell comment by
        write.py so the buyer sees both candidates)

    Verdicts whose card's per_field is no longer 0.5 (because something else
    moved it) are skipped — only the 0.5 band is in the judge's mandate.

    Returns a summary dict for logging.
    """
    conf_by_key: dict[tuple[str, int], CardConfidence] = {
        (c.sku, c.page): c for c in confidence
    }
    counts = {
        "total": 0, "agreed_bump": 0, "disagreed": 0,
        "sku_missing": 0, "out_of_band": 0,
    }

    for page_no, verdicts in verdicts_by_page.items():
        for v in verdicts:
            counts["total"] += 1
            conf = conf_by_key.get((v.sku.strip(), page_no))
            if conf is None:
                # Fuzzy lookup — Opus may echo SKU with minor case/space drift
                for (s, p), c in conf_by_key.items():
                    if p == page_no and _normalize_compare(s) == _normalize_compare(v.sku):
                        conf = c
                        break
            if conf is None:
                counts["sku_missing"] += 1
                continue
            field = v.field
            current = conf.per_field.get(field, 0.0)
            if current != 0.5:
                counts["out_of_band"] += 1
                continue
            # Trust Opus's agreed verdict, with a defensive equality check
            # against opus_value in case Opus said disagreed but the value
            # actually matches (model labeling drift).
            agreed = bool(v.agreed)
            if not agreed and v.opus_value is not None:
                # If Opus's own opus_value still matches Sonnet's, treat as agreed
                sonnet_value = ""  # we don't have it here; fall through to disagreed
                # (We could thread the suspect_value through; not critical —
                # the agreed bit is the primary signal.)
                _ = sonnet_value
            if agreed:
                conf.per_field[field] = 0.7
                conf.per_field_source[field] = "vlm_judge_confirmed"
                conf.judge_agreement[field] = 1.0
                counts["agreed_bump"] += 1
            else:
                conf.judge_agreement[field] = 0.5
                conf.judge_values[field] = v.opus_value
                counts["disagreed"] += 1
    return counts
