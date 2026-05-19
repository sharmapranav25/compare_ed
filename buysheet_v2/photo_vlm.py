"""Per-SKU VLM photo bbox extraction with caching.

For complex multi-section pages where deterministic heuristic geometry fails
(Adidas FW26 page 2 multi-section grid, etc.), we fall back to a per-card VLM
call: crop a generous window around the SKU's PyMuPDF pixel anchor, send it
to Claude vision, ask for the product photo bbox in crop-relative coords,
translate back to page coords.

Architecture properties:
  - Always anchored on the PyMuPDF SKU position (deterministic ground truth)
  - VLM only sees a small window around ONE SKU — no cross-card mis-attribution
  - Crop coords explicitly translated to page coords (no leakage)
  - Sidecar-cached so re-running write.py costs $0

Cost: ~$0.003 per SKU. Adidas (346 SKUs) ~$1; Nike (110 SKUs) ~$0.30.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import time
from pathlib import Path
from typing import Optional

import anthropic
import pypdfium2 as pdfium
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 256  # we only need a 4-int array

PROMPT_SYS = """You are extracting the bounding box of a single product photo from a small image crop taken from a shoebuyer catalog page.

The crop contains:
  - The SKU label text "<SKU>" — this label is HIGHLIGHTED with a bright magenta circle/box for you to spot easily
  - The product photo that belongs to that SKU (usually above or near the SKU label)
  - Possibly: partial photos/text of adjacent products that you must IGNORE

Return a JSON object matching this schema:
  {"bbox_px": [x1, y1, x2, y2]}  // pixel coords relative to THIS crop (origin top-left)

Rules:
  - The bbox MUST tightly bound ONLY the LARGE product photo for the highlighted SKU "<SKU>"
  - The photo is the LARGE shoe/product image closest to the highlighted SKU label, usually directly above it
  - IGNORE small icons, badges (e.g. "CR2", "C03", color-pack indicators), swatches, accessory thumbnails, and adjacent products' photos
  - The bbox should typically be roughly the size of a typical product photo (e.g. similar height to the label text * 4-8x)
  - Coordinates are in pixel coords of the crop image (not the original page)
  - If no clear LARGE product photo for this SKU is visible, return {"bbox_px": null}
  - Do NOT include the SKU label itself or the magenta highlight in the bbox
"""


class PhotoBboxResponse(BaseModel):
    bbox_px: Optional[list[int]] = Field(
        None, min_length=4, max_length=4,
        description="[x1, y1, x2, y2] in pixel coords of the input crop, or null if no photo",
    )


def _make_window(
    sku_rect: tuple[int, int, int, int],
    col_w: int,
    page_w: int,
    page_h: int,
) -> tuple[int, int, int, int]:
    """Crop window around the SKU pixel position. Generous enough to contain
    the photo but small enough to limit which adjacent cards bleed in.

    Heuristic: window is ~1.4x col_w wide, ~3x col_w tall, mostly above the SKU.
    """
    sku_cx = (sku_rect[0] + sku_rect[2]) // 2
    sku_cy = (sku_rect[1] + sku_rect[3]) // 2
    win_w = max(80, int(col_w * 1.4))
    win_h = max(120, int(col_w * 3))
    win_x1 = max(0, sku_cx - win_w // 2)
    win_y1 = max(0, sku_cy - int(win_h * 0.8))  # 80% of window is above SKU
    win_x2 = min(page_w, win_x1 + win_w)
    win_y2 = min(page_h, sku_cy + int(win_h * 0.2))
    return (win_x1, win_y1, win_x2, win_y2)


def _render_window(
    pdf: pdfium.PdfDocument, page_no_1: int,
    window: tuple[int, int, int, int], page_w: int, page_h: int,
    sku_rect_page: Optional[tuple[int, int, int, int]] = None,
) -> bytes:
    """Render the page at the same scale as the VLM coord system, then crop.

    If sku_rect_page is provided, draws a magenta highlight around the SKU
    label (translated to crop-relative coords) so the VLM can unambiguously
    identify which SKU's photo to extract.
    """
    page = pdf[page_no_1 - 1]
    long_pt = max(page.get_width(), page.get_height())
    scale = max(page_w, page_h) / long_pt
    bitmap = page.render(scale=scale)
    pil = bitmap.to_pil().convert("RGB")
    pil = pil.crop(window)
    if sku_rect_page is not None:
        from PIL import ImageDraw
        wx1, wy1, _, _ = window
        sx1, sy1, sx2, sy2 = sku_rect_page
        # Translate SKU rect from page coords to crop coords
        rx1 = sx1 - wx1
        ry1 = sy1 - wy1
        rx2 = sx2 - wx1
        ry2 = sy2 - wy1
        # Draw a thick magenta rectangle just below the SKU label so the
        # marker doesn't obscure the SKU text itself
        draw = ImageDraw.Draw(pil)
        draw.rectangle(
            [max(0, rx1 - 4), max(0, ry1 - 4),
             min(pil.size[0], rx2 + 4), min(pil.size[1], ry2 + 4)],
            outline=(255, 0, 255), width=3,
        )
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def extract_photo_bbox_for_sku(
    client: anthropic.Anthropic,
    pdf: pdfium.PdfDocument,
    page_no: int,
    sku: str,
    sku_rect: tuple[int, int, int, int],
    col_w: int,
    page_w: int,
    page_h: int,
) -> tuple[Optional[tuple[int, int, int, int]], dict]:
    """Per-SKU VLM photo bbox extraction. Returns (page_coord_bbox, usage)."""
    window = _make_window(sku_rect, col_w, page_w, page_h)
    crop_png = _render_window(pdf, page_no, window, page_w, page_h, sku_rect_page=sku_rect)

    # Retry-with-backoff for transient Sonnet Structured Outputs errors.
    # The "Grammar compilation timed out" 400 hits ~1-5% of calls under load
    # and almost always clears on retry. Without this, a single hiccup mid-run
    # bubbles up through resolve_catalog_photo_bboxes and write.py's outer
    # try/except, causing the WHOLE catalog to fall back to phototune-only —
    # which is why Hoka photos looked misaligned (phototune bboxes are looser
    # than per-card VLM-resolved ones).
    response = None
    last_error: Optional[Exception] = None
    for attempt in range(3):
        try:
            response = client.messages.parse(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=[{
                    "type": "text",
                    "text": PROMPT_SYS.replace("<SKU>", sku),
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64.standard_b64encode(crop_png).decode(),
                            },
                        },
                        {"type": "text", "text": f"Return the bbox for SKU \"{sku}\" in this crop."},
                    ],
                }],
                output_format=PhotoBboxResponse,
            )
            break
        except Exception as e:
            last_error = e
            msg = str(e).lower()
            transient = (
                "grammar compilation" in msg
                or "internal_server_error" in msg
                or "overloaded" in msg
                or " 503" in msg or " 429" in msg
            )
            if attempt < 2 and transient:
                wait = 1.5 * (2 ** attempt)  # 1.5s, 3s
                log.warning("photo_vlm transient error on '%s' (attempt %d): %s — retrying in %.1fs",
                            sku, attempt + 1, type(e).__name__, wait)
                time.sleep(wait)
                continue
            raise
    if response is None:
        raise RuntimeError(f"photo_vlm exhausted retries for {sku}: {last_error}")
    parsed = response.parsed_output
    usage = {
        "input_tokens": getattr(response.usage, "input_tokens", 0),
        "output_tokens": getattr(response.usage, "output_tokens", 0),
        "cache_read_tokens": getattr(response.usage, "cache_read_input_tokens", 0) or 0,
    }
    if parsed.bbox_px is None:
        return None, usage
    # Translate crop coords → page coords
    cx1, cy1, cx2, cy2 = parsed.bbox_px
    wx1, wy1, _, _ = window
    page_bbox = (wx1 + cx1, wy1 + cy1, wx1 + cx2, wy1 + cy2)
    # Clamp + sanity-check
    page_bbox = (
        max(0, page_bbox[0]),
        max(0, page_bbox[1]),
        min(page_w, page_bbox[2]),
        min(page_h, page_bbox[3]),
    )
    if page_bbox[2] - page_bbox[0] < 15 or page_bbox[3] - page_bbox[1] < 15:
        return None, usage
    return page_bbox, usage


# ---------------------------------------------------------------------------- cached batch

def resolve_catalog_photo_bboxes(
    pdf_path: Path,
    extraction,
    page_dims: dict,
    *,
    sidecar_path: Optional[Path] = None,
    client: Optional[anthropic.Anthropic] = None,
    force: bool = False,
) -> tuple[dict[tuple[int, str], tuple[int, int, int, int]], dict]:
    """Per-card VLM photo bbox extraction across the full catalog, with caching.

    Loads + saves results to sidecar_path (defaults to <pdf_stem>.v2.photo_bboxes.json).
    Set force=True to ignore cache and re-call the VLM for every SKU.

    Returns ({(page, sku): page_bbox}, usage_summary).
    """
    from buysheet_v2.phototune import (
        estimate_column_width, find_sku_pixel_rect, pad_bbox,
    )

    if sidecar_path is None:
        sidecar_path = pdf_path.with_suffix("").with_name(f"{pdf_path.stem}.v2.photo_bboxes.json")

    # Load cache
    cache: dict[str, tuple[int, int, int, int]] = {}
    if sidecar_path.exists() and not force:
        try:
            raw = json.loads(sidecar_path.read_text())
            cache = {k: tuple(v) for k, v in raw.items()}
        except (json.JSONDecodeError, OSError):
            cache = {}

    if client is None:
        client = anthropic.Anthropic()

    pdf = pdfium.PdfDocument(str(pdf_path))
    usage_totals = {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0, "calls": 0}
    resolved: dict[tuple[int, str], tuple[int, int, int, int]] = {}
    try:
        for pe in extraction.pages:
            dims = page_dims.get(pe.page)
            if not dims or not pe.cards:
                continue
            page_w, page_h = dims

            # PyMuPDF SKU positions
            sku_rects: dict[str, tuple[int, int, int, int]] = {}
            for c in pe.cards:
                rect = find_sku_pixel_rect(pdf_path, pe.page, c.sku, page_w, page_h)
                if rect:
                    sku_rects[c.sku] = rect

            # Per-page col_w (used for the VLM crop window size)
            col_w = estimate_column_width(list(sku_rects.items())) or 100

            for c in pe.cards:
                key = f"{pe.page}::{c.sku}"
                rect = sku_rects.get(c.sku)
                if not rect:
                    # No PyMuPDF anchor (image-only page) — fall back to VLM's
                    # original photo_bbox or card_bbox in write.py
                    continue
                if key in cache and not force:
                    resolved[(pe.page, c.sku)] = cache[key]
                    continue
                try:
                    bbox, u = extract_photo_bbox_for_sku(
                        client, pdf, pe.page, c.sku, rect, col_w, page_w, page_h,
                    )
                except Exception as e:
                    # Per-SKU failure — log + skip this SKU only. write.py's
                    # caller will fall back to phototune for this one SKU,
                    # but the REST of the catalog keeps its VLM bboxes.
                    log.warning("photo_vlm gave up on %s/%s after retries: %s",
                                pe.page, c.sku, type(e).__name__)
                    usage_totals["calls"] += 1
                    continue
                usage_totals["input_tokens"] += u["input_tokens"]
                usage_totals["output_tokens"] += u["output_tokens"]
                usage_totals["cache_read_tokens"] += u["cache_read_tokens"]
                usage_totals["calls"] += 1
                if bbox:
                    # Pad 12% to avoid silhouette cut-off
                    padded = pad_bbox(bbox, page_w, page_h, pct=0.12)
                    resolved[(pe.page, c.sku)] = padded
                    cache[key] = padded
    finally:
        pdf.close()
        # Persist cache
        try:
            sidecar_path.write_text(json.dumps(
                {k: list(v) for k, v in cache.items()}, indent=2,
            ))
        except OSError:
            pass

    return resolved, usage_totals
