"""SKU → photo-bbox matcher and YOLO-density picker.

Two Sonnet vision calls per page (with the YOLO pre-pass output):

  1. pick_yolo_density(page) → imgsz ∈ {1280, 1920, 2560}
     Run BEFORE yolo_detect.detect_shoes_in_image(...) so the chosen
     imgsz matches the page's shoe-image density. Sparse pages waste
     cycles at 2560; dense thumbnail grids miss shoes at 1280.

  2. match_skus_to_bboxes(client, annotated_path, cards, bboxes)
     Run AFTER yolo_detect produced numbered-box annotations AND
     extract.py produced the per-page ProductCard list. The matcher VLM
     reads the annotated page + the SKU list, returns {sku: box_number},
     and we translate that to {sku: bbox_in_page_coords}.

Both prompts are lifted verbatim from Pranav's compare_ed/
extract_products.py — they're well-tuned for this exact task and there's
no reason to re-engineer them. We use client.messages.parse with our
PhotoMatchResponse / DensityResponse Pydantic schemas (Anthropic
structured outputs) for consistency with the rest of buysheet_v2's
extraction calls; Pranav's version used messages.create + manual JSON
parsing, which we don't need here.
"""
from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Optional

import anthropic

from buysheet_v2.ingest import IngestedPage
from buysheet_v2.schemas.card import ProductCard
from buysheet_v2.schemas.photo_match import (
    DensityResponse, PhotoMatchAssignment, PhotoMatchResponse,
)

log = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-6"
MAX_TOKENS_DENSITY = 256
MAX_TOKENS_MATCH = 2048


# Lifted verbatim from Pranav's compare_ed/extract_products.py:117-143.
DENSITY_SYSTEM_PROMPT = """You classify how dense a catalog page is with shoe images.

You see ONE page from a wholesale shoe catalog. Pick ONE density label:

  - "low"    : 1-6 shoes on the page. Includes hero/lifestyle pages with
               a couple of feature shots, or sparse pages with a small
               number of clearly-separated product photos.
  - "medium" : roughly 7-15 shoes in a moderate grid. Each shoe is a
               clear product photo but the page is starting to feel
               populated.
  - "high"   : 16+ shoes packed into a dense grid. Each individual
               shoe image is small (about 1/20 page width or less).

Examples:
  - Hero shot of a single shoe plus 2-3 color variants on the right
    → "low"
  - A 3x3 or 4x3 grid of product photos                  → "medium"
  - A 6-column grid of tiny thumbnails across the page  → "high"

When borderline (between low/medium, or medium/high), pick the HIGHER
density — over-detecting is recoverable, missing shoes is not.

Return ONLY JSON, no prose, no code fences:
{
  "density": "low" | "medium" | "high",
  "reasoning": "<short, ≤12 words>"
}"""

DENSITY_TO_IMGSZ = {
    "low":    1280,
    "medium": 1920,
    "high":   2560,
}

# Lifted verbatim from Pranav's compare_ed/extract_products.py:155-200.
MATCH_SYSTEM_PROMPT = """You match SKUs to numbered shoe images on a catalog page.

You will see an image of a single catalog page with RED numbered boxes
(1, 2, 3, ...) drawn around every detected shoe region. The numbers were
placed by a separate detector — they are NOT in the original page.

Some numbered boxes are per-SKU product photos (the catalog assigns a
SKU/style code to that shot). Others are hero / marketing / lifestyle
shots — large feature images, models wearing the shoe, context photos
— that don't carry a SKU.

You will also be given the list of SKUs that were extracted from the
page's TEXT. For each SKU, identify the numbered box that contains its
product image.

How to decide:
  - The SKU's product photo is usually the cleanest, most isolated shot.
    It tends to sit NEAR its SKU code, color name, and price in the page
    layout — small captions directly under or next to the image.
  - Hero / marketing / lifestyle shots tend to be larger, often on the
    LEFT or TOP of the page, without a SKU code printed near them.
  - When a SKU's image is not visible on the page (or wasn't detected
    by the box drawer), return null.
  - Never assign the same box to two different SKUs. If two SKUs seem
    to share a photo, pick the better match and null the other.

Return ONLY JSON, no prose, no code fences:
{
  "assignments": [
    {"sku": "<SKU string verbatim>", "box": <integer or null>},
    ...
  ]
}

Echo every SKU verbatim — do not change casing, whitespace, or
punctuation. Cover every SKU in the input exactly once."""


def _b64_png(png_bytes: bytes) -> dict:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": base64.standard_b64encode(png_bytes).decode(),
        },
    }


def _b64_png_from_path(path: Path) -> dict:
    return _b64_png(path.read_bytes())


# ------------------------------------------------------------------ density picker

def pick_yolo_density(
    page: IngestedPage, *, client: Optional[anthropic.Anthropic] = None,
) -> tuple[int, Optional[str]]:
    """Sonnet vision classifies the page density → returns (imgsz, reasoning).

    Falls back to (1920, "fallback: pick failed") on any error. Bias is
    UP per Pranav's prompt — over-detecting is recoverable, missing
    shoes is not.
    """
    if client is None:
        client = anthropic.Anthropic()
    try:
        response = client.messages.parse(
            model=MODEL,
            max_tokens=MAX_TOKENS_DENSITY,
            system=[{"type": "text", "text": DENSITY_SYSTEM_PROMPT,
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{
                "role": "user",
                "content": [
                    _b64_png(page.png_bytes),
                    {"type": "text",
                     "text": "Classify the density of this catalog page. JSON only."},
                ],
            }],
            output_format=DensityResponse,
        )
        parsed = response.parsed_output
    except Exception as exc:  # noqa: BLE001
        log.warning("density picker failed on page %d: %s: %s — fallback medium=1920",
                    page.page_no, type(exc).__name__, exc)
        return 1920, "fallback: picker error"
    density = (parsed.density or "").strip().lower()
    imgsz = DENSITY_TO_IMGSZ.get(density, 1920)
    reasoning = parsed.reasoning or ""
    label = f"{density}: {reasoning}" if reasoning else density
    return imgsz, label


# ------------------------------------------------------------------ matcher

def match_skus_to_bboxes(
    annotated_path: Path,
    cards: list[ProductCard],
    bboxes: list[tuple[int, int, int, int]],
    *,
    client: Optional[anthropic.Anthropic] = None,
) -> dict[str, tuple[int, int, int, int]]:
    """Given an annotated page (with numbered red boxes from yolo_detect)
    + the SKU list extracted by extract.py + the corresponding YOLO bboxes,
    ask Sonnet which numbered box belongs to each SKU. Returns
    {sku: bbox_in_page_coords}. SKUs the matcher returned null for, or
    couldn't decide on, are omitted (caller falls back to phototune /
    card_bbox for those).

    The bbox list is 0-indexed; the matcher's output `box` is 1-indexed
    (per Pranav's prompt convention). We translate box → bboxes[box-1].
    """
    if not cards or not bboxes:
        return {}
    if not annotated_path.exists():
        log.warning("matcher: annotated page missing at %s — skipping", annotated_path)
        return {}
    if client is None:
        client = anthropic.Anthropic()

    n_boxes = len(bboxes)
    body_lines = [
        f"Annotated page image (boxes 1..{n_boxes} drawn in red).",
        "Extracted SKUs from the page text:",
    ]
    for i, c in enumerate(cards, start=1):
        sku = (c.sku or "").strip()
        desc = (c.description or "").strip()
        color = (c.color or "").strip()
        body_lines.append(f"  {i}. sku={sku!r}  desc={desc!r}  color={color!r}")
    body_lines.append("")
    body_lines.append(
        f"For each SKU return the matching box number (1..{n_boxes}) or null. "
        f"Echo each sku verbatim."
    )
    body = "\n".join(body_lines)

    try:
        response = client.messages.parse(
            model=MODEL,
            max_tokens=MAX_TOKENS_MATCH,
            system=[{"type": "text", "text": MATCH_SYSTEM_PROMPT,
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{
                "role": "user",
                "content": [
                    _b64_png_from_path(annotated_path),
                    {"type": "text", "text": body},
                ],
            }],
            output_format=PhotoMatchResponse,
        )
        parsed: PhotoMatchResponse = response.parsed_output
    except Exception as exc:  # noqa: BLE001
        log.warning("matcher: VLM call failed: %s: %s — no SKU→bbox mapping returned",
                    type(exc).__name__, exc)
        return {}

    # Pranav's no-dup-box semantic: first SKU wins on a tied box assignment;
    # later SKUs claiming the same box fall through to no-image fallback.
    used_boxes: set[int] = set()
    out: dict[str, tuple[int, int, int, int]] = {}
    for a in parsed.assignments:
        if not isinstance(a, PhotoMatchAssignment):
            continue
        if a.sku is None or a.box is None:
            continue
        sku = a.sku.strip()
        box = a.box
        if box < 1 or box > n_boxes:
            continue
        if box in used_boxes:
            continue
        out[sku] = tuple(bboxes[box - 1])
        used_boxes.add(box)
    return out
