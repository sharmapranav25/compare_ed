"""VLM-as-judge: independent re-extraction with Opus 4.7 for cross-model verification.

For each page already extracted by Sonnet 4.6, run a parallel extraction with
Opus 4.7 using a deliberately rephrased prompt (so model-specific failure
modes are caught rather than reinforced). Compare per-card per-field values
and emit an agreement score. The result feeds into write.py's tier
classification so cells where both models agree get bumped to highest
confidence and disagreements are flagged with both candidates in the comment.

Why a different prompt: if Sonnet and Opus were given the identical prompt,
they would share most prompt-related biases. Rephrasing the task in slightly
different terms (and reordering the field list) tends to expose systematic
extraction errors that one model commits and the other doesn't.

Cost: Opus is ~5x more expensive per token than Sonnet, so judging a
30-page catalog adds roughly +$4-5 to the Sonnet pass's $1.50. Worth it for
validation runs; can be disabled via the `enabled` flag in production.
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
